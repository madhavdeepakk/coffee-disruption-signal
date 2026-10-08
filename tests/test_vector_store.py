"""The relevance model's batching. The model itself is replaced by a small
stand-in with the same interface, so the test runs without downloading it;
what is checked is how texts are grouped, padded and put back in order."""
import numpy as np
import pytest

from src.rag import vector_store as vs


class _Encoding:
    def __init__(self, ids, mask):
        self.ids, self.attention_mask = ids, mask


class _Tokenizer:
    """One token per word, padded to the longest text in the call, as the real
    tokenizer does once padding is enabled."""
    def enable_padding(self): pass
    def enable_truncation(self, max_length): self.max_length = max_length

    def encode_batch(self, texts):
        rows = [[(sum(map(ord, w)) % 97) + 1 for w in t.split()][:self.max_length] for t in texts]
        width = max(len(r) for r in rows)
        return [_Encoding(r + [0] * (width - len(r)), [1] * len(r) + [0] * (width - len(r)))
                for r in rows]


class _Input:
    def __init__(self, name): self.name = name


class _Session:
    """Returns a fixed vector per token id, so a text's pooled vector depends
    only on its own tokens. Records the shape of every batch it is given."""
    def __init__(self):
        self.batches = []
        self.table = np.random.default_rng(0).normal(size=(98, 8)).astype(np.float32)

    def get_inputs(self): return [_Input("input_ids"), _Input("attention_mask")]

    def run(self, _, inputs):
        self.batches.append(inputs["input_ids"].shape)
        return [self.table[inputs["input_ids"]]]


@pytest.fixture
def model(monkeypatch):
    session = _Session()
    monkeypatch.setattr(vs, "_get_onnx_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(vs, "_get_onnx_session", lambda: session)
    return session


def _texts(n):
    # lengths from a headline to a long article, in no particular order
    return [" ".join(f"w{(i * 7 + j) % 50}" for j in range(3 + (i * 37) % 400)) for i in range(n)]


def test_a_large_set_is_embedded_in_small_batches(model):
    """424 documents in one batch asked for 5.3 GB and failed on a real run."""
    vs.embed_texts(_texts(424))
    assert max(rows for rows, _ in model.batches) <= vs.EMBED_BATCH_SIZE
    assert sum(rows for rows, _ in model.batches) == 424
    heads, bytes_per_float = 12, 4
    worst = max(rows * heads * width * width * bytes_per_float for rows, width in model.batches)
    assert worst < 300 * 1024 ** 2                       # the model's largest step stays under 300 MB


def test_batching_changes_no_vector_and_keeps_the_order(model):
    texts = _texts(50)
    batched = vs.embed_texts(texts)
    one_by_one = np.stack([vs.embed_texts([t])[0] for t in texts])
    assert batched.shape == one_by_one.shape
    assert np.allclose(batched, one_by_one, atol=1e-5)   # padding is masked out of the pooling
    all_at_once = vs.embed_texts(texts, batch_size=len(texts))
    assert np.allclose(batched, all_at_once, atol=1e-5)


def test_short_texts_are_not_padded_to_the_longest_article(model):
    vs.embed_texts(["coffee jumps"] * 16 + [" ".join(["word"] * 400)])
    assert sorted(width for _, width in model.batches) == [2, 400]


def test_scores_line_up_with_their_documents(model):
    docs = [{"title": f"story {i}", "text": " ".join(["coffee"] * (i * 20))} for i in range(40)]
    scores = vs.score_documents_against_query(docs, "coffee frost")
    assert len(scores) == 40 and all(isinstance(x, float) for x in scores)
    alone = [vs.score_documents_against_query([d], "coffee frost")[0] for d in docs]
    assert np.allclose(scores, alone, atol=1e-5)
    assert vs.score_documents_against_query([], "coffee frost") == []
