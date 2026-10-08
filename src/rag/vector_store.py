"""
Week 3: semantic/embedding-based retrieval, to cover the Week 2
language-coverage gap (docs/week2_keyword_scoring_comparison_addendum.md):
keyword scoring missed ~80% of relevant non-English articles because it
only matches literal English words.

Model choice: intfloat/multilingual-e5-small.
  - Multilingual by design (not an English model with translation bolted
    on), which targets the Week 2 failure mode directly.
  - Small enough (~470MB) to run without a GPU or long download on a laptop.
  - E5 models require a task prefix on every input: "query: " for search
    queries, "passage: " for documents being indexed. Skipping this prefix
    quietly degrades retrieval quality, so it is enforced by the helper
    functions below rather than left to the caller to remember.

There are two separate code paths in this file:

  1. score_documents_against_query() / embed_texts(): the live path used by
     src/pipeline.py on every run. It uses onnxruntime + tokenizers + numpy
     (manual mean-pooling + L2 normalize - the standard E5 recipe from the
     model's HF card) and imports neither sentence-transformers nor plain
     transformers. Both of those pull in scikit-learn -> scipy (transformers
     imports its generation utilities on `from transformers import
     AutoTokenizer`, before any model loads), and a machine-level security
     policy can block scipy's compiled extensions at import time. onnxruntime
     and tokenizers have no scipy/scikit-learn in their dependency trees,
     which is why this path uses them.

  2. build_index() / semantic_search() / main(): the offline batch
     comparison-vs-manual-labels path (not yet run - see relevance_gate.py's
     notes). It uses sentence-transformers + faiss for convenience (index
     persistence, top-k search), since it is an optional analysis script, not
     something every live run depends on. Its imports are local to the
     functions that need them so importing this module (which src/pipeline.py
     does, for path 1) never fails just because path 2's heavier dependencies
     aren't usable on a given machine.

Usage:
    python -m src.rag.vector_store
    (reuses data/sample_articles.json from Week 2 - no new GDELT calls;
    requires sentence-transformers + faiss-cpu, unlike the live path)
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_ARTICLES_PATH = REPO_ROOT / "data" / "sample_articles.json"
MANUAL_LABELS_PATH = REPO_ROOT / "data" / "week1_manual_labels.json"
FAISS_INDEX_DIR = REPO_ROOT / "data" / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
COMPARISON_OUTPUT_PATH = REPO_ROOT / "docs" / "week3_semantic_vs_keyword_comparison.md"

MODEL_NAME = "intfloat/multilingual-e5-small"

DEFAULT_QUERIES = [
    "coffee drought Brazil",
    "Brazil coffee frost",
    "coffee supply disruption",
]


def document_text_for_embedding(doc: dict) -> str:
    """E5 convention: documents get the 'passage: ' prefix. Title + text,
    falling back to title alone when text fetch failed (Week 2 documents
    have a non-trivial failure rate for this - see text_fetch_error)."""
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    body = f"{title}. {text}" if text else title
    return f"passage: {body}".strip()


def query_text_for_embedding(query: str) -> str:
    """E5 convention: queries get the 'query: ' prefix."""
    return f"query: {query}"


# --------------------------------------------------------------------------
# PATH 1 (live, pipeline.py): ONNX Runtime + tokenizers + numpy only.
#
# transformers cannot be used here: `from transformers import AutoTokenizer`,
# before any model loads, imports its generation utilities (auto_factory ->
# generation -> candidate_generator), which import sklearn.metrics.roc_curve,
# which pulls in scipy - and a machine-level security policy can block
# scipy's compiled extensions at import time. So this path drops transformers
# entirely: onnxruntime (Microsoft's inference runtime, no scipy/scikit-learn
# dependency) runs a pre-converted ONNX export of the same model, and the
# standalone tokenizers package (the Rust tokenizer library transformers uses
# internally, importable on its own with no sklearn/scipy) does the
# tokenizing.
#
# ONNX_MODEL_REPO is a community ONNX export of the same
# intfloat/multilingual-e5-small weights (same tokenizer, same embeddings),
# widely used for framework-free inference. If this repo ever moves or is
# unavailable, swap ONNX_MODEL_REPO for another ONNX export of the same base
# model; the embedding math below doesn't depend on which repo hosts the
# file.
# --------------------------------------------------------------------------

ONNX_MODEL_REPO = "Xenova/multilingual-e5-small"

# Tried in order. Xenova's transformers.js conversions are usually published
# under onnx/model.onnx (fp32) with optional quantized siblings, so this
# tries the common naming conventions rather than assuming one. If all of
# these fail, the error message below says what to check manually.
ONNX_MODEL_FILE_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/model_quantized.onnx",
    "model.onnx",
]

_ONNX_SESSION_CACHE = {}
_ONNX_TOKENIZER_CACHE = {}


def _get_onnx_tokenizer():
    if "t" not in _ONNX_TOKENIZER_CACHE:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        tokenizer_path = hf_hub_download(repo_id=ONNX_MODEL_REPO, filename="tokenizer.json")
        _ONNX_TOKENIZER_CACHE["t"] = Tokenizer.from_file(tokenizer_path)
    return _ONNX_TOKENIZER_CACHE["t"]


def _get_onnx_session():
    if "s" not in _ONNX_SESSION_CACHE:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        print(f"Loading ONNX embedding model: {ONNX_MODEL_REPO} (first run downloads it)...")
        model_path = None
        last_exc = None
        for filename in ONNX_MODEL_FILE_CANDIDATES:
            try:
                model_path = hf_hub_download(repo_id=ONNX_MODEL_REPO, filename=filename)
                print(f"  found: {filename}")
                break
            except Exception as exc:  # noqa: BLE001 - broad on purpose, trying multiple candidates
                last_exc = exc
                continue
        if model_path is None:
            raise RuntimeError(
                f"None of {ONNX_MODEL_FILE_CANDIDATES} were found in {ONNX_MODEL_REPO} "
                f"(last error: {last_exc}). Check https://huggingface.co/{ONNX_MODEL_REPO}/tree/main "
                f"for the actual .onnx file path and update ONNX_MODEL_FILE_CANDIDATES in "
                f"this file (src/rag/vector_store.py) accordingly."
            )
        _ONNX_SESSION_CACHE["s"] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _ONNX_SESSION_CACHE["s"]


def _mean_pool_numpy(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Standard E5 pooling (per the model's own HF card): mean of token
    embeddings, masked so padding tokens don't contribute. This - not the
    [CLS] token - is the pooling E5 models were trained to be scored with;
    using the wrong pooling silently produces low-quality embeddings, the
    same class of "quiet but real" mistake as skipping the query:/passage:
    prefixes. Pure numpy - no torch involved in this path.
    """
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


# Texts are embedded this many at a time. The model's attention step needs
# batch x 12 heads x tokens x tokens x 4 bytes, and every text is padded to
# the longest in its batch, up to 512 tokens. All of a date's documents in one
# batch was fine at a hundred or so (about 1.3 GB); at 424 it asked for 5.3 GB
# and failed, and the run was decided without semantic scores. Sixteen at a
# time needs about 200 MB however many documents there are.
EMBED_BATCH_SIZE = 16


def _embed_batch(texts: list, tokenizer, session) -> np.ndarray:
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    onnx_inputs = {}
    input_names = {inp.name for inp in session.get_inputs()}
    if "input_ids" in input_names:
        onnx_inputs["input_ids"] = input_ids
    if "attention_mask" in input_names:
        onnx_inputs["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, onnx_inputs)
    last_hidden_state = outputs[0]  # (batch, seq_len, hidden_dim) - first output for this export

    pooled = _mean_pool_numpy(last_hidden_state, attention_mask)
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
    return pooled / norms


def embed_texts(texts: list, batch_size: int = None) -> np.ndarray:
    """
    Embed a list of already-prefixed texts (see document_text_for_embedding /
    query_text_for_embedding) into L2-normalized vectors, using only
    onnxruntime + tokenizers + numpy. Returns an (N, dim) numpy array, in the
    order the texts were given.

    The texts go through the model EMBED_BATCH_SIZE at a time, shortest
    first, so that headlines are not padded out to the length of full
    articles. Padding is masked out of the pooling, so the batching does not
    change a text's vector beyond floating-point noise.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    batch_size = batch_size or EMBED_BATCH_SIZE
    tokenizer = _get_onnx_tokenizer()
    session = _get_onnx_session()

    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        for i, vector in zip(chunk, _embed_batch([texts[i] for i in chunk], tokenizer, session)):
            vectors[i] = vector
    return np.stack(vectors)


def score_documents_against_query(documents: list, query: str) -> list:
    """
    Compute a semantic similarity score (cosine, roughly 0..1) for each
    document against a SINGLE reference query - this is the live-pipeline
    entry point (src/pipeline.py). For one anomaly's handful of documents,
    embedding directly and taking a dot product (both sides already
    L2-normalized, so dot product == cosine similarity) is simpler than
    building/discarding a FAISS index every run, and needs none of path 2's
    heavier dependencies.

    Returns a list of floats aligned 1:1 with `documents`. This covers the
    Week 2 language-coverage gap (see this module's docstring); the live path
    runs on onnxruntime + tokenizers to avoid the scipy/scikit-learn
    dependency described there.
    """
    if not documents:
        return []

    doc_texts = [document_text_for_embedding(d) for d in documents]
    doc_embeddings = embed_texts(doc_texts)

    q_embedding = embed_texts([query_text_for_embedding(query)])

    scores = doc_embeddings @ q_embedding[0]
    return [float(s) for s in scores]


# --------------------------------------------------------------------------
# PATH 2 (offline Week 3 batch comparison): sentence-transformers + faiss.
# Imports are local to these functions so a blocked/missing dependency here
# cannot break path 1 above, which src/pipeline.py depends on for every
# live run.
# --------------------------------------------------------------------------

def load_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {MODEL_NAME} (first run downloads ~470MB)...")
    return SentenceTransformer(MODEL_NAME)


def build_index(documents: list, model):
    """
    Embed all documents and build a FAISS index using inner-product search
    over L2-normalized vectors, which is equivalent to cosine similarity.
    """
    import faiss

    texts = [document_text_for_embedding(d) for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def semantic_search(query: str, index, model, documents: list, k: int = 10):
    """Returns top-k (document, similarity_score) pairs for a query."""
    import faiss

    q_emb = model.encode([query_text_for_embedding(query)], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((documents[idx], float(score)))
    return results


def compare_to_manual_labels_semantic(documents: list, doc_scores_by_id: dict) -> str:
    """
    Same structure as Week 2's keyword-score comparison
    (src/rag/retriever.py::compare_to_manual_labels), but for the semantic
    score, specifically to check whether multilingual embeddings close the
    Week 2 language gap.
    """
    if not MANUAL_LABELS_PATH.exists():
        return "(No manual labels file found - skipping comparison.)"

    with open(MANUAL_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    labels_by_url = {entry["url"]: entry["manual_label"] for entry in labels}

    doc_by_url = {d["url"]: d for d in documents}
    matched_urls = [u for u in labels_by_url if u in doc_by_url]

    if not matched_urls:
        return "(No overlap between indexed documents and Week 1 manual labels.)"

    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    lines = [
        "# Week 3 — Semantic (FAISS/multilingual-e5) Score vs Week 1 Manual Labels",
        "",
        f"{len(matched_urls)} of {len(documents)} documents overlap with Week 1's "
        "manual labels.",
        "",
        "Directly re-testing the Week 2 finding: does a multilingual embedding "
        "model score non-English relevant articles meaningfully higher than the "
        "keyword scorer did (which scored 80% of non-English articles as exactly "
        "0.000)?",
        "",
        "| title (script) | manual label | semantic score | keyword score (Wk2) |",
        "|---|---|---|---|",
    ]

    ascii_scores, nonascii_scores = [], []
    for url in matched_urls:
        doc = doc_by_url[url]
        manual = labels_by_url[url]
        sem_score = doc_scores_by_id.get(doc["document_id"], 0.0)
        kw_score = doc.get("retrieval_score", 0.0)
        script = "ascii" if is_ascii(doc["title"]) else "non-ascii"
        (ascii_scores if script == "ascii" else nonascii_scores).append(sem_score)
        lines.append(
            f"| {doc['title'][:50]}... ({script}) | {manual} | {sem_score:.3f} | {kw_score:.3f} |"
        )

    lines += [
        "",
        f"Average semantic score, ASCII/English-titled: "
        f"{sum(ascii_scores)/len(ascii_scores):.3f}" if ascii_scores else "N/A",
        f"Average semantic score, non-ASCII-titled: "
        f"{sum(nonascii_scores)/len(nonascii_scores):.3f}" if nonascii_scores else "N/A",
        "",
        "## Interpretation",
        "",
        "If the non-ASCII average is now comparable to (not dramatically below) "
        "the ASCII average, that's direct evidence the multilingual embedding "
        "model fixes the Week 2 language-coverage gap. If it's still far lower, "
        "the gap isn't purely a 'wrong model' problem and needs more "
        "investigation (e.g. E5 prefix handling, or genuinely weaker embedding "
        "quality for some of these languages/scripts) before Week 4's threshold "
        "work assumes semantic scoring solves this.",
    ]
    return "\n".join(lines)


def main():
    import faiss

    if not SAMPLE_ARTICLES_PATH.exists():
        raise SystemExit(
            f"{SAMPLE_ARTICLES_PATH} not found - run 'python -m src.rag.retriever' "
            "(Week 2) first to generate the document set this indexes."
        )

    with open(SAMPLE_ARTICLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    documents = data["documents"]
    print(f"Loaded {len(documents)} documents from {SAMPLE_ARTICLES_PATH}")

    model = load_model()
    index, embeddings = build_index(documents, model)
    print(f"Built FAISS index: {index.ntotal} vectors, dim={embeddings.shape[1]}")

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "document_ids_in_order": [d["document_id"] for d in documents],
            },
            f,
            indent=2,
        )
    print(f"Saved index to {FAISS_INDEX_PATH}, metadata to {FAISS_METADATA_PATH}")

    doc_scores_by_id = {}
    for query in DEFAULT_QUERIES:
        results = semantic_search(query, index, model, documents, k=len(documents))
        for doc, score in results:
            if doc["query"] == query:
                doc_scores_by_id[doc["document_id"]] = score

    report = compare_to_manual_labels_semantic(documents, doc_scores_by_id)
    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nComparison report written to {COMPARISON_OUTPUT_PATH}")
    print("=" * 60)
    print(report[:2000])


if __name__ == "__main__":
    main()
