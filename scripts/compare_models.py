"""
Have several models read the same evidence, and compare what they conclude.

Why
---
The free model tiers run out part-way through a run, and the pipeline keeps
going by handing the next date to whichever model still has quota. That
keeps a run alive and ruins it as a measurement: the earlier 100-date run
was answered by two models, and the two did not behave alike, so a change in
the numbers could be the method or could be the model.

This script does the opposite. Every model reads every date, on exactly the
text the pipeline's own model was shown, with no fallback: a model that is
out of quota stops and is resumed later, it is never replaced. Then the same
decision rule (src/rag/blind_evidence.decide) is applied to each model's
reading, so the only thing that differs between columns is who did the
reading.

That gives three results the single run cannot:

  1. How much the decision depends on the model. If three unrelated models
     read the same articles and the rule reaches the same decision for most
     dates, the result belongs to the method. If they disagree, it does not,
     and the report says on which dates.
  2. A consensus reading: a document's direction counts only if most of the
     models that read it report it, each with a quote found in the document.
     One model misreading an article no longer decides a date.
  3. A system that does not depend on any one free tier.

Nothing is retrieved again. The news was fetched once by the pipeline; this
reads data/eval_cache/shown/ (the text each run's model was shown) and the
readings already inside results/pipeline_output_*.json.

Usage
-----
    python -m scripts.compare_models --list
    python -m scripts.compare_models --read groq:qwen/qwen3.8-27b
    python -m scripts.compare_models --read gemini:gemini-3.6-flash --max 18
    python -m scripts.compare_models --report

--read is safe to stop and repeat: finished dates are skipped, and it stops by
itself when the model's quota runs out.
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import fmt_rate, load_labels, rate  # noqa: E402
from src.rag import blind_evidence as be  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SHOWN_DIR = REPO_ROOT / "data" / "eval_cache" / "shown"
READINGS_DIR = REPO_ROOT / "data" / "eval_cache" / "model_readings"
COMMODITY = "coffee"

EXPLAINED, REFUSED = "EXPLAINED", "REFUSED"


def slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)


def parse_model(spec: str) -> tuple:
    """'groq:qwen/qwen3.8-27b' -> ('groq', 'qwen/qwen3.8-27b')."""
    provider, _, model = spec.partition(":")
    if not provider or not model:
        raise SystemExit(f"Write the model as provider:model, for example "
                         f"groq:openai/gpt-oss-120b. Got {spec!r}. --list shows the choices.")
    return provider.strip().lower(), model.strip()


# ---------------------------------------------------------------------------
# What there is to compare
# ---------------------------------------------------------------------------

def blind_runs() -> dict:
    """date -> pipeline output, for every stored run decided by the
    blind-evidence rule and not ended in a fault."""
    runs = {}
    for path in sorted(RESULTS_DIR.glob(f"pipeline_output_{COMMODITY}_*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if run.get("decision_mode") != "blind" or (run.get("outcome") or {}).get("is_fault"):
            continue
        runs[(run.get("anomaly") or {}).get("date")] = run
    return runs


def shown_documents(date: str, run: dict = None):
    """The documents the pipeline's model was shown for this date, exactly as
    shown: from the copy the run saved, or failing that rebuilt from the run's
    own record and the article-text cache. None if neither gives the exact
    text - a model is never given something merely similar."""
    path = SHOWN_DIR / f"{COMMODITY}_{date}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return rebuild_shown(run) if run else None


def rebuild_shown(run: dict):
    """Reconstruct what a run's model was shown from the run's saved source
    list and the article-text cache. Each source records how many characters
    of text the document had; if the cache does not return text of exactly
    that length, the document cannot be trusted to be the same and the whole
    date is given up on."""
    from src.rag.text_fetch import cached_article_text
    result = run.get("explanation_result") or {}
    excerpt = (result.get("evidence_packing") or {}).get("excerpt_chars") or 1500
    sources = {s.get("document_id"): s for s in run.get("sources") or []}
    documents = []
    for doc_id in result.get("evidence_document_ids") or []:
        source = sources.get(doc_id)
        if source is None or source.get("text_chars") is None:
            return None
        text = (cached_article_text(source.get("url", "")) or "") if source["text_chars"] else ""
        if len(text) != source["text_chars"]:
            return None
        documents.append({"document_id": doc_id, "title": source.get("title", ""),
                          "publication_date": source.get("publication_date", ""),
                          "url": source.get("url", ""),
                          "passed_gate": bool(source.get("passed_gate", True)),
                          "text": be.as_shown({"text": text})["text"][:excerpt]})
    return {"excerpt_chars": excerpt, "documents": documents} if documents else None


def was_read(run: dict) -> bool:
    """Did this run reach the reading step? A date refused at the relevance
    gate, or with no recent documents, never called a model: its decision is
    the same whoever would have read it."""
    return bool((run.get("explanation_result") or {}).get("evidence_document_ids"))


def stored_reading(model: str, date: str):
    path = READINGS_DIR / slug(model) / f"{date}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def readings_for(model: str, date: str, run: dict):
    """This model's reading of this date: from the pipeline run if this model
    was the one that answered it, else from --read's store. None if this
    model has not read the date."""
    result = run.get("explanation_result") or {}
    if result.get("model_used") == model and "blind_evidence" in result:
        return result["blind_evidence"].get("readings") or []
    stored = stored_reading(model, date)
    return None if stored is None else stored.get("readings") or []


def models_seen(runs: dict) -> list:
    """Every model with at least one reading, the pipeline's own first."""
    counts = {}
    for run in runs.values():
        name = (run.get("explanation_result") or {}).get("model_used")
        if name and was_read(run):
            counts[name] = counts.get(name, 0) + 1
    seen = sorted(counts, key=lambda name: -counts[name])
    if READINGS_DIR.exists():
        for directory in sorted(READINGS_DIR.iterdir()):
            for path in directory.glob("*.json"):
                try:
                    name = json.loads(path.read_text(encoding="utf-8")).get("model")
                except (ValueError, OSError):
                    continue
                if name and name not in seen:
                    seen.append(name)
                break
    return seen


# ---------------------------------------------------------------------------
# --list and --read
# ---------------------------------------------------------------------------

def list_models() -> None:
    from src.rag import explainer
    print("Configured in the pipeline (provider:model):")
    for provider, models in explainer.provider_models().items():
        for model in models:
            print(f"    {provider}:{model}")
    print("\nAvailable on your keys right now:")
    if os.environ.get("GROQ_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                            base_url="https://api.groq.com/openai/v1")
            for m in sorted(client.models.list().data, key=lambda m: m.id):
                print(f"    groq:{m.id}")
        except Exception as exc:  # noqa: BLE001 - a listing must not crash on a provider error
            print(f"    groq: could not list ({type(exc).__name__})")
    if os.environ.get("GEMINI_API_KEY"):
        try:
            for name in sorted(explainer.list_available_models()):
                print(f"    gemini:{name}")
        except Exception as exc:  # noqa: BLE001
            print(f"    gemini: could not list ({type(exc).__name__})")
    print("\nPick models from different makers: two versions of one model agreeing "
          "says less than two unrelated ones agreeing.")


def read_one(provider: str, model: str, date: str, pack: dict) -> dict:
    """One model's reading of one date's shown documents."""
    from src.rag import explainer
    documents, excerpt = pack["documents"], pack.get("excerpt_chars") or 1500
    prompt = be.build_prompt(
        COMMODITY, date, explainer.build_evidence_block(documents, excerpt_chars=excerpt))
    info = explainer.call_one_model(provider, model, prompt,
                                    temperature=explainer.EXPLANATION_TEMPERATURE,
                                    max_output_tokens=be.EXTRACTION_MAX_OUTPUT_TOKENS)
    parsed = json.loads(info["response"].text)      # ValueError = not valid JSON
    if not (isinstance(parsed, list) or isinstance(parsed.get("documents"), list)):
        raise ValueError("the answer has no list of documents")
    return {"model": info["model"], "provider": provider, "date": date,
            "latency_ms": info["latency_ms"],
            "readings": be.normalize_reading(parsed, documents, excerpt, date)}


def read_with(spec: str, max_dates: int = None, sleep_seconds: float = 35.0) -> None:
    from src.rag import explainer
    provider, model = parse_model(spec)
    explainer.check_providers([provider])
    runs = blind_runs()
    todo, no_text = [], 0
    for date, run in sorted(runs.items()):
        if not was_read(run) or readings_for(model, date, run) is not None:
            continue
        if shown_documents(date, run) is None:
            no_text += 1
            continue
        todo.append(date)
    done = sum(1 for d, r in runs.items() if was_read(r) and readings_for(model, d, r) is not None)
    print(f"{model}: {done} date(s) already read, {len(todo)} to read"
          + (f", {no_text} cannot be read (the exact text shown to the pipeline's model "
             f"is not available for them; re-run those dates)" if no_text else ""))
    if max_dates:
        todo = todo[:max_dates]

    bad = 0
    for i, date in enumerate(todo, 1):
        try:
            reading = read_one(provider, model, date, shown_documents(date, runs[date]))
        except Exception as exc:  # noqa: BLE001 - provider errors vary by type
            if explainer._is_quota_error(exc):
                print(f"\n{model} is out of quota after {i - 1} date(s) this time. Nothing is "
                      f"lost: run the same command again later and it carries on.")
                return
            bad += 1
            print(f"  [{i}/{len(todo)}] {date}  failed: {type(exc).__name__}: {str(exc)[:120]}")
            if bad >= 3:
                print("\nStopping: three dates failed. If the message says the model does not "
                      "exist, check its name with --list.")
                return
        else:
            bad = 0
            path = READINGS_DIR / slug(model) / f"{date}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reading), encoding="utf-8")
            with_reason = sum(1 for r in reading["readings"] if r["cause"] and r["quote_found"])
            print(f"  [{i}/{len(todo)}] {date}  {len(reading['readings'])} document(s) read, "
                  f"{with_reason} with a quoted reason", flush=True)
        if i < len(todo):
            time.sleep(sleep_seconds)
    print("\nNext:\n    python -m scripts.compare_models --report")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def decision(readings: list, direction: str, actual_pct: float = None) -> str:
    decided = be.decide(readings, direction, variants=be.DEFAULT_VARIANTS,
                        actual_pct=actual_pct)
    return EXPLAINED if decided["decision"] == "EXPLAIN" else REFUSED


def consensus_readings(per_model: list) -> list:
    """One reading per document from several models' readings of the same
    documents. A document's direction counts only if more than half of the
    models report that direction with a reason whose quote was found; it
    counts as a "report" only if most of those say the document itself
    reports the price move. The reason and quote shown are the first
    agreeing model's."""
    by_doc = {}
    for readings in per_model:
        for r in readings:
            by_doc.setdefault(r["document_id"], []).append(r)
    out = []
    for doc_id, readings in by_doc.items():
        good = [r for r in readings if r.get("direction") in be.DIRECTIONS
                and r.get("cause") and r.get("quote_found")]
        votes = {d: [r for r in good if r["direction"] == d] for d in be.DIRECTIONS}
        winner = next((d for d in be.DIRECTIONS if len(votes[d]) * 2 > len(per_model)), None)
        base = readings[0]
        if winner is None:
            out.append({**base, "direction": None, "kind": None, "cause": None,
                        "quote": None, "quote_found": False})
            continue
        agreeing = votes[winner]
        reports = sum(1 for r in agreeing if r.get("kind") == "report")
        out.append({**agreeing[0], "direction": winner,
                    "kind": "report" if reports * 2 > len(agreeing) else "pressure"})
    return out


def kappa(pairs: list) -> float:
    """Cohen's kappa for two raters' explain/refuse decisions: agreement
    beyond what their own explain rates would give by chance. None when it
    is undefined (no pairs, or both always gave the same single answer)."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a == EXPLAINED) / n
    pb = sum(1 for _, b in pairs if b == EXPLAINED) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 2)


def compare(runs: dict, labels: dict, models: list = None) -> dict:
    models = models or models_seen(runs)
    read_dates = sorted(d for d, r in runs.items() if was_read(r))
    fixed = {d: REFUSED for d, r in runs.items() if not was_read(r)}    # no model was asked
    direction = {d: (r.get("anomaly") or {}).get("direction") for d, r in runs.items()}
    size = {d: be.flagged_day_move(r.get("anomaly") or {}) for d, r in runs.items()}

    readings = {m: {d: rd for d in read_dates
                    for rd in [readings_for(m, d, runs[d])] if rd is not None}
                for m in models}
    decisions = {m: {d: decision(rd, direction[d], size[d]) for d, rd in readings[m].items()}
                 for m in models}

    # Consensus, on the dates every compared model has read.
    common = [d for d in read_dates if all(d in readings[m] for m in models)]
    if len(models) >= 2 and common:
        readings["consensus"] = {d: consensus_readings([readings[m][d] for m in models])
                                 for d in common}
        decisions["consensus"] = {d: decision(rd, direction[d], size[d])
                                  for d, rd in readings["consensus"].items()}

    def want(date):
        return (labels.get(date) or {}).get("expected_outcome", "").strip().upper()

    rows = {}
    for name, per_date in decisions.items():
        everything = {**fixed, **per_date}          # with the dates no model was asked about
        explain = [d for d in everything if want(d) == "EXPLAIN"]
        refuse = [d for d in everything if want(d) == "REFUSE"]
        explained = [d for d in per_date if per_date[d] == EXPLAINED]
        opposite = {"up": "down", "down": "up"}
        reasons = [r for d in per_date for r in readings[name][d] if r.get("cause")]
        by_direction = {}
        for way in ("up", "down"):
            dates = [d for d in everything if direction[d] == way]
            by_direction[way] = rate(sum(1 for d in dates if everything[d] == EXPLAINED),
                                     len(dates))
        rows[name] = {
            "dates_read": len(per_date),
            "explained": rate(len(explained), len(per_date)),
            "explained_by_direction": by_direction,
            "explain_labelled": rate(sum(1 for d in explain if everything[d] == EXPLAINED),
                                     len(explain)),
            "refuse_labelled": rate(sum(1 for d in refuse if everything[d] == REFUSED),
                                    len(refuse)),
            "reasons_with_quote_found": rate(sum(1 for r in reasons if r.get("quote_found")),
                                             len(reasons)),
            "explained_either_direction": rate(
                sum(1 for d in explained
                    if decision(readings[name][d], opposite.get(direction[d]), size[d]) == EXPLAINED),
                len(explained)),
        }

    pairs = []
    for a, b in itertools.combinations(models, 2):
        both = [d for d in read_dates if d in decisions[a] and d in decisions[b]]
        pair = [(decisions[a][d], decisions[b][d]) for d in both]
        docs = same = 0
        for d in both:
            other = {r["document_id"]: r.get("direction") for r in readings[b][d]}
            for r in readings[a][d]:
                if r["document_id"] in other:
                    docs += 1
                    same += r.get("direction") == other[r["document_id"]]
        pairs.append({"a": a, "b": b, "dates": len(both),
                      "same_decision": rate(sum(1 for x, y in pair if x == y), len(pair)),
                      "kappa": kappa(pair),
                      "same_direction_per_document": rate(same, docs)})

    disagreements = [{"date": d, "direction": direction[d], "label": want(d) or "-",
                      **{m: decisions[m][d] for m in models}}
                     for d in common if len({decisions[m][d] for m in models}) > 1]
    return {"models": models, "n_runs": len(runs), "n_read": len(read_dates),
            "n_decided_without_a_model": len(fixed), "n_common": len(common),
            "rows": rows, "pairs": pairs, "disagreements": disagreements}


def generate_report(c: dict) -> str:
    L = ["# Model Comparison", "",
         "Several models read the same documents without being told the price move, and the "
         "same rule decides from each model's reading. The text each model sees is the text "
         "the pipeline's model was shown, and no model stands in for another.", "",
         f"{c['n_runs']} runs decided by the blind-evidence rule. {c['n_read']} reached the "
         f"reading step; the other {c['n_decided_without_a_model']} were refused before any "
         f"model was asked (nothing passed the relevance gate, or nothing was recent), so "
         f"they are the same for every model. They are left out of the agreement figures, "
         f"where they would only inflate them, and counted in the label figures.", ""]

    L += ["| Reader | Dates read | Explained | Up days explained | Down days explained | "
          "Explained when it should | Refused when it should | Reasons with quote found | "
          "Also explained if the move were opposite |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in c["rows"].items():
        label = "**consensus**" if name == "consensus" else name
        L.append(f"| {label} | {r['dates_read']} | {fmt_rate(r['explained'])} | "
                 f"{fmt_rate(r['explained_by_direction']['up'])} | "
                 f"{fmt_rate(r['explained_by_direction']['down'])} | "
                 f"{fmt_rate(r['explain_labelled'])} | {fmt_rate(r['refuse_labelled'])} | "
                 f"{fmt_rate(r['reasons_with_quote_found'])} | "
                 f"{fmt_rate(r['explained_either_direction'])} |")
    L.append("")
    if "consensus" in c["rows"]:
        L.append(f"Consensus: on the {c['n_common']} date(s) every model has read, a "
                 f"document's direction counts only if more than half of the models report "
                 f"it, each with a quote found in the document.")
        L.append("")
    elif len(c["models"]) < 2:
        L.append("Only one model has read these dates, so there is nothing to compare yet: "
                 "`python -m scripts.compare_models --read provider:model`.")
        L.append("")

    if c["pairs"]:
        L += ["## Agreement between models", "",
              "| Models | Dates both read | Same decision | Cohen's kappa | "
              "Same direction, document by document |", "|---|---|---|---|---|"]
        for p in c["pairs"]:
            k = "n/a" if p["kappa"] is None else f"{p['kappa']:.2f}"
            L.append(f"| {p['a']} / {p['b']} | {p['dates']} | {fmt_rate(p['same_decision'])} | "
                     f"{k} | {fmt_rate(p['same_direction_per_document'])} |")
        L += ["", "Kappa is agreement beyond what the two models' own explain rates would "
              "produce by chance: 1 is perfect, 0 is chance. A same-decision rate on its own "
              "flatters two models that both refuse most dates.", ""]

    if c["disagreements"]:
        L += ["## Dates the models decide differently", "",
              "| Date | Move | Label | " + " | ".join(c["models"]) + " |",
              "|---|---|---|" + "---|" * len(c["models"])]
        for row in c["disagreements"]:
            L.append(f"| {row['date']} | {row['direction']} | {row['label']} | "
                     + " | ".join(row[m] for m in c["models"]) + " |")
        L += ["", "These are the dates worth reading by hand: the documents are the same, so "
              "one of the models misread something.", ""]

    L += ["## Limits", "",
          "- Agreement between models is not correctness. Models trained on similar data can "
          "share a misreading, and all of them read the same retrieved documents, so what "
          "retrieval missed is missed by every one of them.",
          "- A model that has read only some of the dates is compared on those dates alone; "
          "the consensus row covers only dates every model has read.",
          "- The label columns rest on as few should-refuse dates as the label file has.", ""]
    return "\n".join(L)


def write_report() -> dict:
    runs = blind_runs()
    if not runs:
        raise SystemExit("No runs decided by the blind-evidence rule in results/. Run the "
                         "evaluation set first: python -m scripts.run_eval_set --run")
    comparison = compare(runs, load_labels() or {})
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2),
                                                       encoding="utf-8")
    report = generate_report(comparison)
    (RESULTS_DIR / "model_comparison.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {RESULTS_DIR / 'model_comparison.md'}")
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Have several models read the same evidence and compare the decisions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show the models you can use")
    group.add_argument("--read", metavar="PROVIDER:MODEL",
                       help="Have this model read every date it has not read yet")
    group.add_argument("--report", action="store_true",
                       help="Write results/model_comparison.md from what has been read")
    parser.add_argument("--max", type=int, default=None,
                        help="With --read: at most this many dates this time")
    parser.add_argument("--sleep", type=float, default=35.0,
                        help="With --read: seconds between calls (default 35, to stay under "
                             "per-minute limits)")
    args = parser.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.list:
        list_models()
    elif args.read:
        read_with(args.read, args.max, args.sleep)
    else:
        write_report()


if __name__ == "__main__":
    main()
