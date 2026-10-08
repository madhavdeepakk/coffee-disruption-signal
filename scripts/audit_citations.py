"""
Citation audit: does each cited document actually support the claim attached
to it?

The pipeline's own check (src/rag/faithfulness.py) establishes that a cited
document id is real. Its "support" score is an embedding similarity that
cannot fail in practice - every citation in the stored runs passed - so it
does not answer the question above. This script does, the only way that
holds up: a person reads each claim next to the text the model was shown and
labels it.

Three steps
-----------
1. Export a labelling sheet (offline, no network):

       python -m scripts.audit_citations --export

   Writes data/labeling/citation_audit_sheet.csv with one row per citation
   in the stored explained runs: the claim, the document's title and URL, and
   the text the model was given for it. Fill in the human_label column with
   exactly one of:

       supported    the text states what the claim says
       partial      the text supports part of the claim, or a weaker version
       unsupported  the text does not say this

   Re-exporting keeps labels already entered.

2. (Optional) add a model judge, to see whether an automated check could
   stand in for the human one:

       python -m scripts.audit_citations --judge

   For each citation a model is asked for a verdict AND a verbatim quote
   from the document that supports it. The quote is then looked up in the
   document text by this script. A "supported" verdict whose quote is not
   actually in the document is recorded as unverified - the judge is not
   taken at its word.

3. Score:

       python -m scripts.audit_citations --score

   Writes results/citation_audit.md and results/citation_audit.json: the
   measured support rate with its interval, and how well the pipeline's own
   flag and the model judge agree with the human labels. The pipeline
   evaluation report picks the JSON up automatically.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as m   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SHEET_PATH = REPO_ROOT / "data" / "labeling" / "citation_audit_sheet.csv"
TEXT_CACHE_DIR = REPO_ROOT / "data" / "text_cache"
REPORT_MD = RESULTS_DIR / "citation_audit.md"
REPORT_JSON = RESULTS_DIR / "citation_audit.json"

# What the model was shown per document. Runs made before evidence packing
# sent the first 1500 characters; later runs record the length they used.
DEFAULT_EXCERPT_CHARS = 1500

VALID_LABELS = ("supported", "partial", "unsupported")

COLUMNS = [
    "anomaly_date", "price_move", "document_id", "claim", "document_title",
    "publication_date", "url", "text_shown_to_model", "text_available",
    "auto_flag", "auto_support_score", "auto_support_rank",
    "judge_verdict", "judge_quote", "judge_quote_found", "judge_model",
    "human_label", "notes",
]

JUDGE_PROMPT = """You are checking one citation. Decide whether the DOCUMENT supports the CLAIM.

CLAIM:
{claim}

DOCUMENT (this is all the text available; the title is part of it):
{document}

Rules:
- "supported": the document states what the claim says.
- "partial": the document supports part of the claim, or a weaker version of it.
- "unsupported": the document does not say this. A document on the same topic that does not make the claim is unsupported.
- For "supported" or "partial" you must give a quote copied EXACTLY from the document, in the document's own language, that backs the claim. If you cannot find one, the verdict is "unsupported".
- Judge only what the document says. Do not use outside knowledge.

Respond with ONLY a JSON object:
{{"verdict": "supported" | "partial" | "unsupported", "quote": "..."}}
"""


# ---------------------------------------------------------------------------
# Building the sheet
# ---------------------------------------------------------------------------

def _cached_text(url: str) -> str:
    if not url:
        return ""
    path = TEXT_CACHE_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:24]}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("text", "") or ""
    except (ValueError, OSError):
        return ""


def _norm_id(raw) -> str:
    raw = str(raw or "").strip()
    return raw[4:] if raw.lower().startswith("doc_") else raw


def _row_key(row: dict) -> tuple:
    return (row["anomaly_date"], row["document_id"], row["claim"])


def build_rows(results: list, index: dict) -> list:
    rows = []
    for run in results:
        er = run.get("explanation_result") or {}
        if er.get("decision") != "EXPLAINED":
            continue
        date = m._run_date(run)
        anomaly = run.get("anomaly") or {}
        pct = anomaly.get("pct_move")
        move = f"{pct:+.1f}% ({anomaly.get('direction', '')})" if pct is not None else ""
        sources = {_norm_id(s.get("document_id")): s for s in m.run_sources(run, index)}
        report = {_norm_id(c.get("document_id")): c
                  for c in ((run.get("faithfulness_report") or {}).get("per_citation") or [])}
        excerpt_chars = ((er.get("evidence_packing") or {}).get("excerpt_chars")
                         or DEFAULT_EXCERPT_CHARS)

        for cite in er.get("citations") or []:
            doc_id = _norm_id(cite.get("document_id"))
            source = sources.get(doc_id) or index.get(doc_id) or {}
            title = source.get("title", "") or ""
            text = _cached_text(source.get("url", ""))
            shown = (text or title)[:excerpt_chars]
            auto = report.get(doc_id, {})
            rows.append({
                "anomaly_date": date,
                "price_move": move,
                "document_id": doc_id,
                "claim": (cite.get("supports") or "").strip(),
                "document_title": title,
                "publication_date": source.get("publication_date", "") or "",
                "url": source.get("url", "") or "",
                "text_shown_to_model": shown.replace("\r", " ").replace("\n", " "),
                "text_available": ("body" if text else ("title_only" if title else "none")),
                "auto_flag": auto.get("flag", ""),
                "auto_support_score": auto.get("support_score", ""),
                "auto_support_rank": (f"{auto['support_rank']}/{auto['support_rank_of']}"
                                      if auto.get("support_rank") else ""),
                "judge_verdict": "", "judge_quote": "", "judge_quote_found": "",
                "judge_model": "", "human_label": "", "notes": "",
            })
    return rows


def read_sheet(path: Path = SHEET_PATH) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_sheet(rows: list, path: Path = SHEET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM: opens cleanly in Excel
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_existing(new_rows: list, old_rows: list) -> list:
    """Carry labels, notes and judge results over from an existing sheet, so
    re-exporting after new pipeline runs never discards work already done."""
    old = {_row_key(r): r for r in old_rows
           if r.get("anomaly_date") and r.get("document_id") is not None and r.get("claim") is not None}
    keep = ("human_label", "notes", "judge_verdict", "judge_quote",
            "judge_quote_found", "judge_model")
    for row in new_rows:
        prior = old.get(_row_key(row))
        if prior:
            for field in keep:
                if prior.get(field):
                    row[field] = prior[field]
    return new_rows


def export() -> list:
    rows = merge_existing(build_rows(m.load_results(), m._gdelt_source_index()), read_sheet())
    write_sheet(rows)
    labelled = sum(1 for r in rows if (r.get("human_label") or "").strip())
    no_text = sum(1 for r in rows if r["text_available"] != "body")
    print(f"Wrote {len(rows)} citation(s) to {SHEET_PATH}")
    print(f"  already labelled: {labelled}")
    print(f"  shown with the title only, because the article body is not in this machine's "
          f"text cache: {no_text}")
    if no_text:
        print("    (Either that fetch failed at run time and the model also saw only the "
              "title, or the cache here is incomplete. Export on the machine that ran the "
              "pipeline to be sure the sheet shows what the model saw.)")
    print("Fill in human_label with: " + " / ".join(VALID_LABELS))
    return rows


# ---------------------------------------------------------------------------
# Model judge with quote verification
# ---------------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" \"'“”‘’.")


def quote_in_document(quote: str, document: str) -> bool:
    """Is the quote really in the document? Compared after collapsing
    whitespace and case; a quote under 15 characters is too short to count
    as evidence of anything."""
    q = _squash(quote)
    return len(q) >= 15 and q in _squash(document)


def judge_row(row: dict, providers: list = None) -> dict:
    from src.rag import explainer

    document = f"{row['document_title']}\n{row['text_shown_to_model']}".strip()
    prompt = JUDGE_PROMPT.format(claim=row["claim"], document=document)
    try:
        info = explainer._call_model_ex(prompt, temperature=0.0, max_output_tokens=1200,
                                        providers=providers)
        parsed = json.loads(info["response"].text)
        verdict = str(parsed.get("verdict", "")).strip().lower()
        quote = str(parsed.get("quote", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        return {"judge_verdict": "error", "judge_quote": f"{type(exc).__name__}: {exc}"[:150],
                "judge_quote_found": "", "judge_model": ""}

    if verdict not in VALID_LABELS:
        verdict = "error"
    found = quote_in_document(quote, document)
    if verdict in ("supported", "partial") and not found:
        verdict = "unverified"   # claimed support, but the quote is not in the document
    return {"judge_verdict": verdict, "judge_quote": quote[:400],
            "judge_quote_found": "yes" if found else "no", "judge_model": info["model"]}


def judge(providers: list = None, redo: bool = False) -> list:
    rows = read_sheet() or export()
    todo = [r for r in rows if redo or not r.get("judge_verdict") or r["judge_verdict"] == "error"]
    print(f"Judging {len(todo)} citation(s)...")
    for i, row in enumerate(todo, 1):
        row.update(judge_row(row, providers))
        print(f"  [{i}/{len(todo)}] {row['anomaly_date']} {row['document_id'][:8]}: "
              f"{row['judge_verdict']}")
        write_sheet(rows)   # save as we go; a rate limit should not lose progress
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cohen_kappa(pairs: list) -> float:
    """Cohen's kappa for a list of (a, b) category pairs. None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    categories = {a for a, _ in pairs} | {b for _, b in pairs}
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum((sum(1 for a, _ in pairs if a == c) / n)
                   * (sum(1 for _, b in pairs if b == c) / n) for c in categories)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 3)


def score_rows(rows: list) -> dict:
    # A spreadsheet can drop trailing empty cells when it saves, which leaves
    # the column missing (None) for that row rather than blank.
    for r in rows:
        r["human_label"] = (r.get("human_label") or "").strip().lower()
    labelled = [r for r in rows if r["human_label"] in VALID_LABELS]
    n = len(labelled)
    bad = [r["human_label"] for r in rows
           if r["human_label"] and r["human_label"] not in VALID_LABELS]

    def count(label):
        return sum(1 for r in labelled if r["human_label"] == label)

    out = {
        "n_citations": len(rows),
        "n_labelled": n,
        "unrecognised_labels": sorted(set(bad)),
        "supported": m.rate(count("supported"), n),
        "partial": m.rate(count("partial"), n),
        "unsupported": m.rate(count("unsupported"), n),
        "supported_or_partial": m.rate(count("supported") + count("partial"), n),
        "by_text_available": {},
    }
    for kind in ("body", "title_only", "none"):
        subset = [r for r in labelled if r.get("text_available") == kind]
        if subset:
            out["by_text_available"][kind] = m.rate(
                sum(1 for r in subset if r["human_label"] == "supported"), len(subset))

    # The pipeline's own flag: it said "ok" - was the citation in fact supported?
    auto_ok = [r for r in labelled if r.get("auto_flag") == "ok"]
    out["auto_flag_ok_and_supported"] = m.rate(
        sum(1 for r in auto_ok if r["human_label"] == "supported"), len(auto_ok))
    caught = [r for r in labelled if r["human_label"] == "unsupported"]
    out["unsupported_caught_by_auto_flag"] = m.rate(
        sum(1 for r in caught if r.get("auto_flag") not in ("ok", "")), len(caught))

    judged = [r for r in labelled if r.get("judge_verdict") in VALID_LABELS + ("unverified",)]
    if judged:
        def binary(label):
            return "supported" if label == "supported" else "not_supported"
        pairs = [(binary(r["human_label"]), binary(r["judge_verdict"])) for r in judged]
        out["judge"] = {
            "n": len(judged),
            "agreement": m.rate(sum(1 for a, b in pairs if a == b), len(pairs)),
            "cohen_kappa": cohen_kappa(pairs),
            "unverified_quotes": sum(1 for r in judged if r["judge_verdict"] == "unverified"),
            "unsupported_caught": m.rate(
                sum(1 for r in judged if r["human_label"] == "unsupported"
                    and r["judge_verdict"] != "supported"),
                sum(1 for r in judged if r["human_label"] == "unsupported")),
        }
    return out


def report(scores: dict, rows: list) -> str:
    L = ["# Citation Audit", ""]
    if not scores["n_labelled"]:
        L += [f"{scores['n_citations']} citation(s) exported; none labelled yet. Fill in the "
              f"human_label column of `data/labeling/citation_audit_sheet.csv` and run "
              f"`python -m scripts.audit_citations --score`.", ""]
        return "\n".join(L)

    L += [f"{scores['n_labelled']} of {scores['n_citations']} citations in the stored "
          f"explained runs were read against the text the model was shown and labelled "
          f"by hand.", "",
          "| Human label | Citations |", "|---|---|",
          f"| Supported | {m.fmt_rate(scores['supported'])} |",
          f"| Partly supported | {m.fmt_rate(scores['partial'])} |",
          f"| Unsupported | {m.fmt_rate(scores['unsupported'])} |", ""]
    if scores["unrecognised_labels"]:
        L += [f"Ignored rows with unrecognised labels: {scores['unrecognised_labels']}.", ""]

    if scores["by_text_available"]:
        L += ["Supported rate by what the model had for the document:", ""]
        names = {"body": "article body", "title_only": "title only", "none": "nothing recoverable"}
        L += [f"- {names[k]}: {m.fmt_rate(v)}" for k, v in scores["by_text_available"].items()]
        L.append("")

    L += ["## Do the automated checks agree with the human labels?", "",
          f"- Pipeline flag said ok and the citation was in fact supported: "
          f"{m.fmt_rate(scores['auto_flag_ok_and_supported'])}",
          f"- Unsupported citations the pipeline flag caught: "
          f"{m.fmt_rate(scores['unsupported_caught_by_auto_flag'])}"]
    judge = scores.get("judge")
    if judge:
        L += [f"- Model judge agrees with the human label (supported vs not): "
              f"{m.fmt_rate(judge['agreement'])}, Cohen's kappa {judge['cohen_kappa']}",
              f"- Unsupported citations the model judge caught: "
              f"{m.fmt_rate(judge['unsupported_caught'])}",
              f"- Judge verdicts downgraded because the quote was not in the document: "
              f"{judge['unverified_quotes']}"]
    else:
        L.append("- No model judge has been run (`--judge`).")
    L.append("")

    unsupported = [r for r in rows if r.get("human_label") == "unsupported"]
    if unsupported:
        L += ["## Unsupported citations", "", "| Date | Claim | Document |", "|---|---|---|"]
        for r in unsupported:
            L.append(f"| {r['anomaly_date']} | {r['claim'][:160].replace('|', '/')} | "
                     f"{r['document_title'][:90].replace('|', '/')} |")
        L.append("")

    L += ["## Limits", "",
          "- One annotator. A second pass by someone else on the same sheet would give an "
          "inter-annotator agreement figure, which this lacks.",
          "- A citation is judged against the text the model was shown. Where the article "
          "body was never fetched, that is the title alone.",
          ""]
    return "\n".join(L)


def score() -> dict:
    rows = read_sheet()
    if not rows:
        raise SystemExit(f"{SHEET_PATH} not found - run with --export first.")
    scores = score_rows(rows)
    REPORT_JSON.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    text = report(scores, rows)
    REPORT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {REPORT_MD} and {REPORT_JSON}")
    return scores


def main():
    for stream in (sys.stdout, sys.stderr):   # model text can hold any character
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(description="Human-labelled citation audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true", help="Write/refresh the labelling sheet")
    group.add_argument("--judge", action="store_true", help="Add model-judge verdicts to the sheet")
    group.add_argument("--score", action="store_true", help="Score the labelled sheet")
    parser.add_argument("--providers", default=None,
                        help="Model providers for --judge, e.g. gemini. Prefer a different "
                             "model family from the one that wrote the explanations.")
    parser.add_argument("--redo", action="store_true", help="With --judge: re-judge every row")
    args = parser.parse_args()

    if args.export:
        export()
    elif args.judge:
        providers = [p.strip() for p in args.providers.split(",")] if args.providers else None
        judge(providers, redo=args.redo)
    else:
        score()


if __name__ == "__main__":
    main()
