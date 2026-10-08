"""
Evaluation metrics over stored pipeline runs.

Reads results/evaluation_results.json (built by src.evaluation.batch_runner)
and reports what the runs show, with a confidence interval on every rate.
No model calls and no network: every number is derived from the saved
outputs, the label file and the on-disk caches.

What is reported, and what each number does and does not mean
-------------------------------------------------------------
1. OUTCOMES. Every run is placed in exactly one tier, so the table sums to
   the number of runs. Outputs saved before the outcome block existed are
   classified here with the same classifier (src/rag/outcome.py).

2. DECISION ACCURACY against the hand-labelled set
   (data/labeling/labels_by_written_rule.csv): did the system explain
   the dates labelled EXPLAIN and refuse the dates labelled REFUSE? This is
   the only section with ground truth. It checks the decision, not whether
   the explanation names the documented cause.

3. CITATIONS. "Resolve rate" = the cited document id is one the model was
   shown. That rules out invented sources; it does NOT show the document
   supports the claim. Support is measured against human labels by
   scripts/audit_citations.py, and those results are included when present.

4. EVIDENCE. Documents retrieved and accepted, how many accepted documents
   are copies of the same story, how many were published after the anomaly
   date, and how recent the evidence is.

5. EVIDENCE DIRECTION. Share of explained runs whose accepted documents
   lean the same way as the price move, by the word-list heuristic in
   src/rag/direction.py. This is a property of the retrieved evidence, not a
   check that the explanation is right.

6. BY YEAR, and the faults.

Intervals are 95% Wilson score intervals. With 10-30 runs they are wide, and
that is the point of printing them.

Output: results/evaluation_report.json + results/evaluation_report.md

Usage:
    python -m src.evaluation.metrics
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rag import outcome as outcome_mod          # noqa: E402
from src.rag.evidence import normalize_title         # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
# The answer key. The 18 labelled dates were researched again under the
# written rule (data/labeling/LABELLING_RULE.md), each from dated market
# reports; that file is the key. The labels made before the rule existed are
# kept for comparison: --labels data/labeling/explanation_backtest_labels.csv
LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"
ORIGINAL_LABELS_PATH = REPO_ROOT / "data" / "labeling" / "explanation_backtest_labels.csv"
CITATION_AUDIT_PATH = RESULTS_DIR / "citation_audit.json"

EXPLAINED = "EXPLAINED"
REFUSED = "REFUSED"
FAULT = "FAULT"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion, as percentages.

    Used instead of the normal approximation because the samples here are
    small and the rates are often near 0 or 1, where p +/- 1.96*se gives
    nonsense (an interval of 100% +/- 0% for 33 out of 33).
    Returns (None, None) when n == 0.
    """
    if n <= 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def rate(successes: int, n: int) -> dict:
    """A rate with its count and interval, in the one shape used everywhere."""
    lo, hi = wilson_interval(successes, n)
    return {
        "k": successes, "n": n,
        "pct": round(100 * successes / n, 1) if n else None,
        "ci95": [lo, hi],
    }


def fmt_rate(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a (0 cases)"
    return f"{r['pct']}% ({r['k']}/{r['n']}; 95% CI {r['ci95'][0]}-{r['ci95'][1]}%)"


def _median(values: list):
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# ---------------------------------------------------------------------------
# Per-run helpers
# ---------------------------------------------------------------------------

def effective_tier(run: dict) -> str:
    """The run's outcome tier. Uses the stored outcome when there is one;
    otherwise classifies the stored signals with the current classifier, so
    older outputs are counted instead of silently left out of the table."""
    stored = (run.get("outcome") or {}).get("tier")
    if stored:
        return stored
    return outcome_mod.classify(
        retrieval_meta=run.get("retrieval_meta"),
        gate_result=run.get("gate_result"),
        explanation_result=run.get("explanation_result"),
        direction_summary=run.get("direction_summary"),
        faithfulness_report=run.get("faithfulness_report"),
    ).tier


def is_full_run(run: dict) -> bool:
    """Did this record go through the whole pipeline? batch_runner's default
    mode also writes gate-only records (retrieval and gate, no model call) and
    error stubs. Those carry no explain/refuse decision and must not be
    counted as one."""
    if (run.get("outcome") or {}).get("tier"):
        return True
    return bool((run.get("explanation_result") or {}).get("decision"))


def verdict(run: dict) -> str:
    """EXPLAINED / REFUSED / FAULT for a run."""
    tier = effective_tier(run)
    if tier in outcome_mod.FAULT_TIERS:
        return FAULT
    return EXPLAINED if tier in outcome_mod.EXPLAINED_TIERS else REFUSED


def is_blind(run: dict) -> bool:
    """Was this run decided by the blind-evidence rule (pipeline version 4
    onwards) rather than by the model with the direction guard after it?"""
    return run.get("decision_mode") == "blind"


def model_verdict(run: dict) -> str:
    """What the model itself decided, before the direction guard. A run whose
    explanation the guard withheld is stored as a refusal with the model's
    text kept under withheld_explanation; the model's own answer there was an
    explanation."""
    if "withheld_explanation" in (run.get("explanation_result") or {}):
        return EXPLAINED
    return verdict(run)


def guard_would_fire(run: dict, rule: str = None) -> bool:
    """Would the direction guard (src/rag/outcome.apply_direction_guard), under
    `rule` (default: the current one), withhold this run's explanation?"""
    if is_blind(run):
        return False          # that decision path has no direction guard
    return (model_verdict(run) == EXPLAINED
            and outcome_mod.guard_fires(run.get("direction_summary") or {}, rule))


def verdict_with_guard(run: dict, rule: str = None) -> str:
    """The decision this run would end with under guard `rule`, computed from
    the stored model answer and evidence tally - so runs made before the
    guard existed, or under an earlier rule, are all scored the same way."""
    v = model_verdict(run)
    return REFUSED if (v == EXPLAINED and guard_would_fire(run, rule)) else v


def _run_date(run: dict) -> str:
    return (run.get("anomaly") or {}).get("date") or run.get("date") or ""


def _domains(sources: list) -> set:
    out = set()
    for s in sources:
        host = urlparse(s.get("url", "") or "").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            out.add(host)
    return out


def _gdelt_source_index() -> dict:
    """document_id -> {url, title, publication_date} from the GDELT cache, so
    runs saved before the `sources` field existed can still be analysed."""
    index = {}
    cache_dir = REPO_ROOT / "data" / "gdelt_cache"
    if not cache_dir.exists():
        return index
    from src.rag.retriever import make_document_id
    from src.rag.gdelt_client import gdelt_seendate_to_iso_date
    for path in cache_dir.glob("*.json"):
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for art in (cached.get("response") or {}).get("articles", []) or []:
            url = art.get("url", "")
            if url:
                index[make_document_id(url)] = {
                    "url": url, "title": art.get("title", "") or "",
                    "publication_date": gdelt_seendate_to_iso_date(art.get("seendate", "")),
                }
    return index


def run_sources(run: dict, index: dict) -> list:
    """Accepted documents for a run: the saved `sources`, or a reconstruction
    from the GDELT cache by accepted document id."""
    sources = run.get("sources") or []
    if sources:
        return sources
    ids = (run.get("gate_result") or {}).get("accepted_document_ids") or []
    return [index[i] for i in ids if i in index]


def load_labels(path: Path = None) -> dict:
    path = LABELS_PATH if path is None else path
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["date"]: row for row in csv.DictReader(f)}


def not_real_moves(labels: dict, commodity: str = "coffee") -> set:
    """Dates whose flagged move is not a real market move of that size: the
    checked contract-switch days, and anything the answer key marks EXCLUDE.
    A run on such a date says nothing about the system either way, so these
    runs are left out of every figure and counted."""
    from src.config.commodities import load_contract_switches
    excluded = set(load_contract_switches(commodity))
    excluded |= {date for date, label in (labels or {}).items()
                 if (label.get("expected_outcome") or "").strip().upper() == "EXCLUDE"}
    return excluded


# ---------------------------------------------------------------------------
# Metric sections
# ---------------------------------------------------------------------------

def outcome_section(results: list) -> dict:
    tiers = Counter(effective_tier(r) for r in results)
    n = len(results)
    verdicts = Counter(verdict(r) for r in results)
    non_fault = n - verdicts[FAULT]
    backfilled = sum(1 for r in results if not (r.get("outcome") or {}).get("tier"))
    return {
        "n_runs": n,
        "tiers": dict(tiers.most_common()),
        "tiers_backfilled": backfilled,
        "explained": rate(verdicts[EXPLAINED], n),
        "refused": rate(verdicts[REFUSED], n),
        "faults": rate(verdicts[FAULT], n),
        "explained_excluding_faults": rate(verdicts[EXPLAINED], non_fault),
        "guard_would_withhold": sum(1 for r in results if guard_would_fire(r)),
    }


def decision_accuracy_section(results: list, labels: dict) -> dict:
    """Decision accuracy on labelled dates. Faults are excluded (a retry
    fixes them) but counted, and so are labelled dates with no run."""
    by_date = {_run_date(r): r for r in results}
    rows = []
    for date, label in sorted(labels.items()):
        expected = (label.get("expected_outcome") or "").strip().upper()
        if expected not in ("EXPLAIN", "REFUSE"):
            continue
        run = by_date.get(date)
        if run is None:
            rows.append({"date": date, "expected": expected, "status": "no_run",
                         "label_confidence": label.get("label_confidence", "")})
            continue
        v, vg = verdict(run), verdict_with_guard(run)
        vm, vt = model_verdict(run), verdict_with_guard(run, "ties")
        want = EXPLAINED if expected == "EXPLAIN" else REFUSED
        rows.append({
            "date": date, "expected": expected,
            "label_confidence": label.get("label_confidence", ""),
            "status": "fault" if v == FAULT else "graded",
            "verdict": v, "verdict_with_guard": vg,
            "verdict_model_alone": vm, "verdict_with_guard_ties": vt,
            "correct": None if v == FAULT else (v == want),
            "correct_with_guard": None if v == FAULT else (vg == want),
            "correct_model_alone": None if v == FAULT else (vm == want),
            "correct_with_guard_ties": None if v == FAULT else (vt == want),
            "model_used": (run.get("explanation_result") or {}).get("model_used"),
            "direction_summary": run.get("direction_summary") or {},
        })

    graded = [r for r in rows if r["status"] == "graded"]

    def block(key):
        exp = [r for r in graded if r["expected"] == "EXPLAIN"]
        ref = [r for r in graded if r["expected"] == "REFUSE"]
        return {
            "overall": rate(sum(1 for r in graded if r[key]), len(graded)),
            "explain_labelled": rate(sum(1 for r in exp if r[key]), len(exp)),
            "refuse_labelled": rate(sum(1 for r in ref if r[key]), len(ref)),
        }

    n_should_explain = sum(1 for r in graded if r["expected"] == "EXPLAIN")
    return {
        "n_labelled": len(rows),
        "n_graded": len(graded),
        # What a system that explained every date would score on the graded
        # dates. The decision score means something only where it beats this.
        "always_explain": rate(n_should_explain, len(graded)),
        "n_should_refuse": len(graded) - n_should_explain,
        "n_fault": sum(1 for r in rows if r["status"] == "fault"),
        "n_no_run": sum(1 for r in rows if r["status"] == "no_run"),
        "as_run": block("correct"),
        "model_alone": block("correct_model_alone"),
        "with_direction_guard": block("correct_with_guard"),
        "with_direction_guard_ties": block("correct_with_guard_ties"),
        "guard_rule": outcome_mod.GUARD_RULE,
        "guard_changed": [
            {"date": r["date"], "expected": r["expected"],
             "with": r["direction_summary"].get("consistent", 0),
             "against": r["direction_summary"].get("inconsistent", 0),
             "tie_only": r["verdict_with_guard"] == EXPLAINED,
             "correct": r["verdict_with_guard_ties"] == (
                 EXPLAINED if r["expected"] == "EXPLAIN" else REFUSED)}
            for r in graded
            if r["verdict_model_alone"] == EXPLAINED and r["verdict_with_guard_ties"] == REFUSED],
        "misses": [r for r in graded if not r["correct"]],
        "rows": rows,
    }


def citation_section(explained: list) -> dict:
    total = resolved = weak = low_rank = ranked = 0
    scores = []
    runs_with_report = 0
    for r in explained:
        fr = r.get("faithfulness_report")
        if not fr:
            continue
        runs_with_report += 1
        n = fr.get("n_citations", 0) or 0
        total += n
        resolved += n - (fr.get("n_missing_document", 0) or 0)
        weak += fr.get("n_weak_support", 0) or 0
        low_rank += fr.get("n_low_rank", 0) or 0
        for c in fr.get("per_citation", []) or []:
            if c.get("support_score") is not None:
                scores.append(c["support_score"])
            if c.get("support_rank") is not None:
                ranked += 1

    out = {
        "n_explained": len(explained),
        "n_explained_with_report": runs_with_report,
        "total_citations": total,
        "resolve": rate(resolved, total),
        "weak_support": rate(weak, total),
        "low_rank": rate(low_rank, ranked) if ranked else None,
        "support_score_min": min(scores) if scores else None,
        "support_score_median": _median(scores),
        # kept under the original names for the dashboard
        "citations_resolve_rate": round(100 * resolved / total, 1) if total else None,
        "weak_support_rate": round(100 * weak / total, 1) if total else None,
        "avg_faithfulness_score": round(sum(scores) / len(scores), 3) if scores else None,
    }
    if CITATION_AUDIT_PATH.exists():
        try:
            out["human_audit"] = json.loads(CITATION_AUDIT_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return out


def evidence_section(results: list, explained: list, index: dict) -> dict:
    n = len(results)
    retrieved = [r.get("documents_retrieved", 0) or 0 for r in results]
    gate_pass = sum(1 for r in results if (r.get("gate_result") or {}).get("decision") == "EXPLAIN")
    zero = sum(1 for x in retrieved if x == 0)

    accepted_total = distinct_total = after_total = dated_total = 0
    runs_with_sources = 0
    for r in results:
        sources = run_sources(r, index)
        if not sources:
            continue
        runs_with_sources += 1
        date = _run_date(r)
        accepted_total += len(sources)
        distinct_total += len({normalize_title(s.get("title", "")) or s.get("url", "")
                               for s in sources})
        for s in sources:
            pub = s.get("publication_date") or ""
            if pub:
                dated_total += 1
                if date and pub > date:
                    after_total += 1

    per_explained = []
    for r in explained:
        sources = run_sources(r, index)
        gate = r.get("gate_result") or {}
        stories = {normalize_title(s.get("title", "")) or s.get("url", "") for s in sources}
        per_explained.append({
            "date": _run_date(r),
            "n_accepted": gate.get("accepted_document_count", len(sources)) or 0,
            "n_domains": len(_domains(sources)),
            "n_stories": len(stories),
            "has_sources": bool(sources),
        })
    with_src = [d for d in per_explained if d["has_sources"]]

    def avg(key, rows):
        return round(sum(d[key] for d in rows) / len(rows), 1) if rows else 0

    return {
        "gate_pass": rate(gate_pass, n),
        "avg_documents_retrieved": round(sum(retrieved) / n, 1) if n else 0,
        "zero_retrieval": rate(zero, n),
        "runs_with_sources": runs_with_sources,
        "accepted_documents": accepted_total,
        "distinct_stories": distinct_total,
        "duplicate_share": rate(accepted_total - distinct_total, accepted_total),
        "published_after_anomaly": rate(after_total, dated_total),
        "explained_with_sources": len(with_src),
        "avg_accepted_documents": avg("n_accepted", per_explained),
        "avg_unique_domains": avg("n_domains", with_src),
        "avg_distinct_stories": avg("n_stories", with_src),
        "single_story": rate(sum(1 for d in with_src if d["n_stories"] <= 1), len(with_src)),
        # original name, for the dashboard
        "single_source_rate": (round(100 * sum(1 for d in with_src if d["n_domains"] <= 1)
                                     / len(with_src), 1) if with_src else 0),
    }


def direction_section(explained: list) -> dict:
    rows = []
    for r in explained:
        ds = r.get("direction_summary") or {}
        c, i = ds.get("consistent", 0) or 0, ds.get("inconsistent", 0) or 0
        if c + i == 0:
            continue
        rows.append({"date": _run_date(r), "consistent": c, "inconsistent": i,
                     "neutral": ds.get("neutral", 0) or 0, "aligned": c > i})
    aligned = sum(1 for d in rows if d["aligned"])
    return {
        "n_evaluated": len(rows),
        "n_explained": len(explained),
        "aligned": rate(aligned, len(rows)),
        "rows": rows,
        # original names, for the dashboard
        "direction_consistent_rate": round(100 * aligned / len(rows), 1) if rows else 0,
        "direction_conflict_rate": round(100 * (len(rows) - aligned) / len(rows), 1) if rows else 0,
    }


BLIND_SETTINGS = (("freshest", 2), ("freshest", 1), ("freshest", 3), ("pooled", 2))

# The three checks version 11 added (blind_evidence.DEFAULT_VARIANTS), and
# the rule with each one taken away, scored from the same stored readings.
FIT_THE_MOVE = ("this_market", "event_reasons", "sized_support")
WITHOUT_VARIANTS = (
    ("sized_support", "Without the size check (a report need not describe a move this large)"),
    ("event_reasons", "Counting trading reasons too (\"technical adjustments\")"),
    ("this_market", "Counting reports about another market too (robusta, local prices)"),
)

# Stricter versions of what counts as evidence (see blind_evidence.VARIANTS),
# each scored from the same stored readings. A row is left out for runs that
# were already decided that way.
BLIND_VARIANTS = (
    (FIT_THE_MOVE, "A report must fit the move: this market, an event as the reason, a move "
                   "of this size (the rule from version 11, added after the invented-move "
                   "test; readings from before it record a size only as a number)"),
    (("gate_only",), "Only documents that cleared the relevance filter itself"),
    (("current_moves",), "Ignore reports of a price move from more than a day before the article"),
    (("size_consistent",), "Ignore reports of a move under a quarter the size of the actual one"),
    (("current_moves", "size_consistent"), "Both of the last two"),
    (("reports_only",), "Only articles that report a price move, not background events "
                        "(added after seeing these dates)"),
)


def _actual_pct(run: dict):
    """The size of the day's move, for the size check. Left out for a trend
    anomaly: that is flagged on a multi-day move, and an article reporting one
    day of it is not expected to match the day's own size."""
    from src.rag import blind_evidence
    return blind_evidence.flagged_day_move(run.get("anomaly") or {})


def run_variants(run: dict) -> tuple:
    """The evidence variants the run itself was decided under (none before
    pipeline version 9, which is when the pipeline began applying any)."""
    blind = (run.get("explanation_result") or {}).get("blind_evidence") or {}
    return tuple(blind.get("variants") or ())


def blind_verdict(run: dict, rule: str, fresh_days: int, variants=(), without=()) -> str:
    """The decision a blind-mode run would end with under another setting of
    the rule, recomputed from its stored readings. `variants` are applied on
    top of the ones the run was decided under, `without` takes some of those
    away. A run that never reached the rule (gate refusal, fault) keeps the
    decision it has."""
    from src.rag import blind_evidence
    blind = (run.get("explanation_result") or {}).get("blind_evidence")
    v = verdict(run)
    if not blind or v == FAULT:
        return v
    direction = (run.get("anomaly") or {}).get("direction")
    applied = tuple(dict.fromkeys(x for x in run_variants(run) + tuple(variants)
                                  if x not in without))
    decided = blind_evidence.decide(blind.get("readings") or [], direction,
                                    fresh_days=fresh_days, rule=rule,
                                    variants=applied, actual_pct=_actual_pct(run))
    return EXPLAINED if decided["decision"] == "EXPLAIN" else REFUSED


def search_yield(runs: list) -> list:
    """For each search: how many of the documents it found were candidates for
    reading, how many of those gave a usable reason, and how many ended up
    among the documents an explanation rests on. A search that never reaches
    the last column costs a request per date and buys nothing."""
    from src.rag import blind_evidence
    rows = {}
    for r in runs:
        found_by = {s.get("document_id"): s.get("found_by") for s in r.get("sources") or []}
        if not any(found_by.values()):
            continue                      # run made before this was recorded
        blind = (r.get("explanation_result") or {}).get("blind_evidence") or {}
        usable = {x["document_id"] for x in blind_evidence.usable(
            blind.get("readings") or [], variants=run_variants(r))}
        deciding = set(blind.get("supporting") or []) if verdict(r) == EXPLAINED else set()
        for doc_id, search in found_by.items():
            row = rows.setdefault(search or "not recorded",
                                  {"candidates": 0, "usable": 0, "deciding": 0, "dates": set()})
            row["candidates"] += 1
            row["usable"] += doc_id in usable
            if doc_id in deciding:
                row["deciding"] += 1
                row["dates"].add(_run_date(r))
    return [{"search": search, "candidates": row["candidates"], "usable": row["usable"],
             "deciding": row["deciding"], "dates_decided": len(row["dates"])}
            for search, row in sorted(rows.items(),
                                      key=lambda kv: (-kv[1]["deciding"], -kv[1]["usable"], kv[0]))]


STOP_STAGES = (
    ("nothing_recent_found", "Nothing recent was found"),
    ("recent_found_none_relevant", "Recent articles found, none relevant enough to read"),
    ("read_no_quoted_reason", "Recent articles read, none gave a reason with a quote"),
    ("reason_points_other_way", "A reason was found, for the opposite move"),
)


def stop_stage(run: dict) -> str:
    """Where a blind-mode run that ended in a refusal stopped: searching,
    filtering, reading or the rule. This is what says which part to work on -
    a date refused because nothing recent was retrieved is a search problem,
    and no change to the reading or the rule will fix it."""
    from src.rag import blind_evidence
    if verdict(run) == EXPLAINED:
        return "explained"
    fresh = blind_evidence.FRESH_TRADING_DAYS

    def recent(doc):
        age = doc.get("trading_days_before_anomaly")
        return age is not None and 0 <= age <= fresh

    if run.get("documents_retrieved_recent") == 0:
        return "nothing_recent_found"
    if not any(recent(s) for s in run.get("sources") or []):
        return "recent_found_none_relevant"
    readings = ((run.get("explanation_result") or {}).get("blind_evidence") or {}) \
        .get("readings") or []
    if not blind_evidence.usable(readings, fresh, variants=run_variants(run)):
        return "read_no_quoted_reason"
    return "reason_points_other_way"


def blind_section(results: list, labels: dict) -> dict:
    """For runs decided by the blind-evidence rule: how the model's readings
    held up, how the decision splits by direction of the move, and whether
    the rule's two settings are doing the work (the same stored readings
    re-scored at other settings)."""
    runs = [r for r in results if is_blind(r) and verdict(r) != FAULT]
    if not runs:
        return {"n_runs": 0}
    reasons = verified = read = shown = 0
    for r in runs:
        for reading in ((r.get("explanation_result") or {}).get("blind_evidence") or {}) \
                .get("readings") or []:
            shown += 1
            read += bool(reading.get("read"))
            if reading.get("cause"):
                reasons += 1
                verified += bool(reading.get("quote_found"))

    by_direction = {}
    for r in runs:
        direction = (r.get("anomaly") or {}).get("direction") or "unknown"
        b = by_direction.setdefault(direction, {"runs": 0, "explained": 0})
        b["runs"] += 1
        b["explained"] += verdict(r) == EXPLAINED
    for b in by_direction.values():
        b["explained_rate"] = rate(b["explained"], b["runs"])

    settings = []
    labelled = [(r, (labels.get(_run_date(r)) or {}).get("expected_outcome", "").strip().upper())
                for r in runs]
    labelled = [(r, want) for r, want in labelled if want in ("EXPLAIN", "REFUSE")]
    exp = [r for r, want in labelled if want == "EXPLAIN"]
    ref = [r for r, want in labelled if want == "REFUSE"]

    def scored(verdicts: dict) -> dict:
        return {
            "explained_all": rate(sum(1 for v in verdicts.values() if v == EXPLAINED), len(runs)),
            "explain_labelled": rate(sum(1 for r in exp if verdicts[id(r)] == EXPLAINED), len(exp)),
            "refuse_labelled": rate(sum(1 for r in ref if verdicts[id(r)] == REFUSED), len(ref)),
        }

    for rule, days in BLIND_SETTINGS:
        settings.append({"rule": rule, "fresh_trading_days": days,
                         **scored({id(r): blind_verdict(r, rule, days) for r in runs})})
    rule, days = BLIND_SETTINGS[0]
    variants = [{"variants": list(names), "text": text,
                 **scored({id(r): blind_verdict(r, rule, days, names) for r in runs})}
                for names, text in BLIND_VARIANTS
                if not all(set(names) <= set(run_variants(r)) for r in runs)]
    if any("current_moves" in run_variants(r) for r in runs):
        # What the rule was before pipeline version 9, on the same readings.
        variants.append({
            "variants": ["without current_moves"],
            "text": "Counting reports of older and forecast moves too (the rule before version 9)",
            **scored({id(r): blind_verdict(r, rule, days, without=("current_moves",))
                      for r in runs})})
    for name, text in WITHOUT_VARIANTS:
        if any(name in run_variants(r) for r in runs):
            # What each part of the version 11 rule costs and buys, on the
            # same readings.
            variants.append({
                "variants": [f"without {name}"], "text": text,
                **scored({id(r): blind_verdict(r, rule, days, without=(name,))
                          for r in runs})})
    if any(set(FIT_THE_MOVE) & set(run_variants(r)) for r in runs):
        variants.append({
            "variants": ["without " + " ".join(FIT_THE_MOVE)],
            "text": "Without all three (the rule before version 11)",
            **scored({id(r): blind_verdict(r, rule, days, without=FIT_THE_MOVE)
                      for r in runs})})
    near_gate = [(r, x) for r in runs
                 for x in ((r.get("explanation_result") or {}).get("blind_evidence") or {})
                 .get("readings") or [] if not x.get("passed_gate", True)]
    # The flip test, from the stored readings: a reading is made without the
    # direction, so what the rule would have said had prices gone the other
    # way needs no new model call.
    from src.rag import blind_evidence
    explained = [r for r in runs if verdict(r) == EXPLAINED
                 and (r.get("explanation_result") or {}).get("blind_evidence")]
    flipped = 0
    for r in explained:
        direction = (r.get("anomaly") or {}).get("direction")
        opposite = {"up": "down", "down": "up"}.get(direction)
        readings = r["explanation_result"]["blind_evidence"].get("readings") or []
        flipped += blind_evidence.decide(readings, opposite, variants=run_variants(r),
                                         actual_pct=_actual_pct(r))["decision"] == "EXPLAIN"

    refused = [r for r in runs if verdict(r) == REFUSED]
    stages = Counter(stop_stage(r) for r in refused)
    should_explain = [
        {"date": _run_date(r), "stage": stop_stage(r),
         "retrieved": r.get("documents_retrieved"),
         "retrieved_recent": r.get("documents_retrieved_recent"),
         "accepted": len(r.get("sources") or [])}
        for r, want in labelled if want == "EXPLAIN" and verdict(r) == REFUSED]

    return {
        "n_runs": len(runs),
        "refusal_stages": {key: rate(stages.get(key, 0), len(refused)) for key, _ in STOP_STAGES},
        "refused_but_should_explain": should_explain,
        "explained_either_direction": rate(flipped, len(explained)),
        "documents_read": rate(read, shown),
        "reasons_with_quote_found": rate(verified, reasons),
        "by_direction": dict(sorted(by_direction.items())),
        "settings": settings,
        "variants": variants,
        "near_gate_read": len(near_gate),
        "near_gate_usable": sum(len(blind_evidence.usable([x], variants=run_variants(r)))
                                for r, x in near_gate),
        "search_yield": search_yield(runs),
    }


def guard_effect_section(results: list) -> dict:
    """What the direction guard does across every run, split by the direction
    of the move. On the labelled dates the guard can only be scored where
    there is a label; this shows where it acts on all of them."""
    out = {}
    for r in results:
        if verdict(r) == FAULT:
            continue
        direction = (r.get("anomaly") or {}).get("direction") or "unknown"
        b = out.setdefault(direction, {"runs": 0, "model_explained": 0, "guard_withholds": 0})
        b["runs"] += 1
        if model_verdict(r) == EXPLAINED:
            b["model_explained"] += 1
            if guard_would_fire(r):
                b["guard_withholds"] += 1
    for b in out.values():
        b["withheld_rate"] = rate(b["guard_withholds"], b["model_explained"])
        b["explained_after_guard"] = rate(b["model_explained"] - b["guard_withholds"], b["runs"])
    total_explained = sum(b["model_explained"] for b in out.values())
    total_withheld = sum(b["guard_withholds"] for b in out.values())
    return {"by_direction": dict(sorted(out.items())),
            "withheld_overall": rate(total_withheld, total_explained)}


def by_year_section(results: list) -> dict:
    out = {}
    for r in results:
        year = _run_date(r)[:4] or "?"
        bucket = out.setdefault(year, {EXPLAINED: 0, REFUSED: 0, FAULT: 0})
        bucket[verdict(r)] += 1
    return dict(sorted(out.items()))


def _flagged_dates():
    """Dates the committed detections file flags, or None if it is not there."""
    path = RESULTS_DIR / "anomaly_detections.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {row["date"] for row in csv.DictReader(f)
                if str(row.get("anomaly_flag", "")).strip().lower() == "true"}


def retrieval_health_section(results: list) -> dict:
    """Runs in which a news-API query failed after retries. The pipeline
    carries on with what the other queries returned, so these runs searched
    with less than the full set, and a refusal on one of them can be the API's
    doing rather than the news being thin."""
    degraded, none_answered = [], []
    for r in results:
        meta = r.get("retrieval_meta") or {}
        failed = meta.get("network_failures") or 0
        if failed <= 0:
            continue
        degraded.append(r)
        if failed >= (meta.get("network_attempts") or 0) and not (meta.get("cache_hits") or 0):
            none_answered.append(r)
    n = len(results)
    return {
        "runs_with_failed_queries": rate(len(degraded), n),
        "runs_with_no_query_answered": len(none_answered),
        "refused_among_those": rate(sum(1 for r in degraded if verdict(r) == REFUSED),
                                    len(degraded)),
        "refused_among_the_rest": rate(
            sum(1 for r in results if r not in degraded and verdict(r) == REFUSED),
            n - len(degraded)),
        "dates": sorted(d for d in (_run_date(r) for r in degraded) if d),
    }


def by_model_section(results: list) -> dict:
    """Explained / refused per answering model. The free tiers run out, the
    pipeline falls back to the next provider, and the models do not behave
    alike - so a pooled explained rate over a mixed set partly measures which
    model happened to answer."""
    out = {}
    for r in results:
        er = r.get("explanation_result") or {}
        name = er.get("model_used")
        v = model_verdict(r)
        if not name or v == FAULT:
            continue
        bucket = out.setdefault(name, {"explained": 0, "refused": 0})
        bucket["explained" if v == EXPLAINED else "refused"] += 1
    return {name: {**b, "explained_rate": rate(b["explained"], b["explained"] + b["refused"])}
            for name, b in sorted(out.items())}


def provenance_section(results: list) -> dict:
    versions = Counter(str(r.get("pipeline_version", "1")) for r in results)
    models = Counter((r.get("explanation_result") or {}).get("model_used") or "not recorded"
                     for r in results)
    flagged = _flagged_dates()
    not_flagged = ([] if flagged is None
                   else sorted(d for d in (_run_date(r) for r in results) if d and d not in flagged))
    return {"pipeline_versions": dict(versions), "models": dict(models.most_common()),
            "dates_not_flagged": not_flagged}


def compute_metrics(results: list, labels: dict = None) -> dict:
    """Compute all evaluation metrics from batch results."""
    n_records = len(results)
    results = [r for r in results if is_full_run(r)]
    total = len(results)
    if total == 0:
        return {"error": "no results to evaluate"}
    if labels is None:
        labels = load_labels()
    unreal = not_real_moves(labels)
    left_out = sorted(_run_date(r) for r in results if _run_date(r) in unreal)
    results = [r for r in results if _run_date(r) not in unreal]
    total = len(results)
    if total == 0:
        return {"error": "no results to evaluate"}

    explained = [r for r in results if verdict(r) == EXPLAINED]
    # Evidence statistics use only what each run saved about its own sources.
    # Older runs did not save them; those could be reconstructed from the
    # local GDELT cache, but that cache is not in the repository, and a
    # report that changes depending on what is on the machine is not one
    # anybody else can reproduce. So they are left out and the report says
    # how many runs the figures cover.
    index = {}

    outcomes = outcome_section(results)
    evidence = evidence_section(results, explained, index)
    citations = citation_section(explained)
    direction = direction_section(explained)

    provenance = provenance_section(results)
    provenance["records_without_a_decision"] = n_records - total - len(left_out)
    provenance["runs_left_out_not_real_moves"] = left_out
    provenance["retrieval_health"] = retrieval_health_section(results)
    provenance["by_model"] = by_model_section(results)

    return {
        "dates_evaluated": total,
        "dates_explained": len(explained),
        "provenance": provenance,
        "outcomes": outcomes,
        "decision_accuracy": decision_accuracy_section(results, labels) if labels else None,
        "by_year": by_year_section(results),
        "guard_effect": guard_effect_section([r for r in results if not is_blind(r)]),
        "blind": blind_section(results, labels or {}),
        # --- sections the dashboard reads, under their original keys ---
        "retrieval": {
            "gate_explain_rate": evidence["gate_pass"]["pct"],
            "gate_explain_count": evidence["gate_pass"]["k"],
            "gate_explain_ci95": evidence["gate_pass"]["ci95"],
            "llm_explain_rate": outcomes["explained"]["pct"],
            "llm_explain_count": outcomes["explained"]["k"],
            "llm_explain_ci95": outcomes["explained"]["ci95"],
            "avg_documents_retrieved": evidence["avg_documents_retrieved"],
            "zero_retrieval_rate": evidence["zero_retrieval"]["pct"],
            "zero_retrieval_count": evidence["zero_retrieval"]["k"],
            "total_dates": total,
        },
        "citation_accuracy": citations,
        "direction_alignment": direction,
        "source_diversity": evidence,
        "outcome_distribution": outcomes["tiers"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(metrics: dict) -> str:
    """Markdown report. Every sentence is generated from the numbers; nothing
    here asserts a conclusion the data in `metrics` does not contain."""
    out = metrics["outcomes"]
    ev = metrics["source_diversity"]
    cit = metrics["citation_accuracy"]
    dir_ = metrics["direction_alignment"]
    prov = metrics["provenance"]
    acc = metrics.get("decision_accuracy")
    n = metrics["dates_evaluated"]

    L = ["# Pipeline Evaluation Report", ""]
    L.append(f"**{n}** stored pipeline runs. Rates are shown with 95% Wilson intervals; "
             f"at this sample size the intervals are wide and the point estimates should "
             f"not be quoted without them.")
    L.append("")

    versions = prov["pipeline_versions"]
    if len(versions) > 1:
        L.append(f"> **Mixed pipeline versions**: {versions}. Runs from different versions "
                 f"were retrieved, gated or prompted differently and are not directly "
                 f"comparable. Re-run the older dates before quoting a combined figure.")
        L.append("")
    elif list(versions) == ["1"]:
        L.append("> All runs predate pipeline version 2 (local date filtering, "
                 "duplicate collapsing, token-budgeted prompts, prompt v2, direction guard). "
                 "This report describes the system as it was when those runs were made.")
        L.append("")
    models = ", ".join(f"{m} ({c})" for m, c in prov["models"].items())
    L.append(f"Models that answered: {models}.")
    L.append("")
    by_model = prov.get("by_model") or {}
    if len(by_model) > 1:
        L.append("More than one model answered, and they are not interchangeable. The "
                 "model's own decision (before the direction guard), by model: "
                 + "; ".join(f"{name} explained {fmt_rate(b['explained_rate'])}"
                             for name, b in by_model.items())
                 + ". The models did not see the same dates, so this is not a like-for-like "
                 "comparison, but a pooled rate over a mixed set partly measures which model "
                 "happened to answer. Set EXPLAINER_PROVIDERS to one provider for a run that "
                 "is meant to be compared with another.")
        L.append("")
    if prov.get("records_without_a_decision"):
        L.append(f"{prov['records_without_a_decision']} record(s) in the results file went "
                 f"through retrieval and the gate only (no model call) and are left out of "
                 f"everything below.")
        L.append("")
    if prov.get("runs_left_out_not_real_moves"):
        gone = prov["runs_left_out_not_real_moves"]
        L.append(f"{len(gone)} stored run(s) are on dates whose flagged move was not a real "
                 f"market move of that size - the price series switched futures contract "
                 f"that day (data/labeling/contract_switch_dates.csv) - and are left out of "
                 f"everything below: {', '.join(gone)}.")
        L.append("")
    if prov.get("dates_not_flagged"):
        L.append(f"{len(prov['dates_not_flagged'])} run(s) are on dates the current detections "
                 f"file does not flag as anomalies ({', '.join(prov['dates_not_flagged'])}). They "
                 f"were run on request and are included, but they are not detected events.")
        L.append("")

    health = prov.get("retrieval_health") or {}
    failed_runs = health.get("runs_with_failed_queries") or {}
    if failed_runs.get("k"):
        L.append(f"In {fmt_rate(failed_runs)} of runs at least one news-API query failed after "
                 f"retries"
                 + (f" (in {health['runs_with_no_query_answered']}, none was answered)"
                    if health.get("runs_with_no_query_answered") else "")
                 + f". Those runs searched with fewer queries than intended: "
                 f"{fmt_rate(health['refused_among_those'])} of them ended in a refusal, against "
                 f"{fmt_rate(health['refused_among_the_rest'])} of the rest. A refusal on such a "
                 f"run may reflect the API rather than the news; "
                 f"`python -m scripts.run_eval_set --run --retry-incomplete` re-asks only the "
                 f"failed queries.")
        L.append("")

    # 1 ---------------------------------------------------------------
    L += ["## 1. Outcomes", "", "Every run falls in exactly one tier.", "",
          "| Outcome tier | Runs |", "|---|---|"]
    for tier, count in out["tiers"].items():
        L.append(f"| {tier} | {count} |")
    L.append(f"| **Total** | **{sum(out['tiers'].values())}** |")
    L.append("")
    if out["tiers_backfilled"]:
        L.append(f"{out['tiers_backfilled']} run(s) were saved before the outcome tier "
                 f"existed and are classified here from their stored signals.")
        L.append("")
    L += [f"- Explained: {fmt_rate(out['explained'])}",
          f"- Refused: {fmt_rate(out['refused'])}",
          f"- Faults (model or retrieval unavailable): {fmt_rate(out['faults'])}",
          f"- Explained, faults excluded: {fmt_rate(out['explained_excluding_faults'])}",
          ""]
    L.append("An explanation rate is not a quality measure on its own: a system that "
             "explained every date would score 100% here. Section 2 is the one that "
             "checks decisions against ground truth.")
    L.append("")

    # 2 ---------------------------------------------------------------
    L += ["## 2. Decision accuracy on the labelled set", ""]
    if not acc or not acc["n_labelled"]:
        L += ["No label file found, so decisions cannot be checked against ground truth.", ""]
    else:
        L.append(f"{acc['n_labelled']} dates carry a label (EXPLAIN: a documented "
                 f"cause exists; REFUSE: no clean cause, the system should decline). "
                 f"Graded: {acc['n_graded']}. Ended in a fault: {acc['n_fault']}. "
                 f"No run yet: {acc['n_no_run']}.")
        L.append("")
        if acc.get("always_explain") and acc["n_graded"]:
            L.append(f"**Explaining every date would score {fmt_rate(acc['always_explain'])} "
                     f"here.** The \"correct decision\" row below is worth something only "
                     f"where it beats that.")
            if not acc.get("n_should_refuse"):
                L.append("")
                L.append("**No graded date in this answer key is one where refusing is "
                         "right.** So this section measures one thing only: how often the "
                         "system fails to explain a move that has a documented cause. "
                         "Whether it refuses when it should is not measured here at all; "
                         "that is what the invented-move test is for "
                         "(results/robustness_report.md).")
            L.append("")
        L += ["| | As run | Model alone | Guard: more against than with | "
              "Guard: a tie also withholds |", "|---|---|---|---|---|"]
        for key, name in (("overall", "Correct decision"),
                          ("explain_labelled", "Explained when it should"),
                          ("refuse_labelled", "Refused when it should")):
            L.append(f"| {name} | {fmt_rate(acc['as_run'][key])} | "
                     f"{fmt_rate(acc['model_alone'][key])} | "
                     f"{fmt_rate(acc['with_direction_guard'][key])} | "
                     f"{fmt_rate(acc['with_direction_guard_ties'][key])} |")
        L.append("")
        n_blind = (metrics.get("blind") or {}).get("n_runs", 0)
        if n_blind:
            L.append(f"{n_blind} of these runs were decided by the blind-evidence rule, which "
                     f"has no direction guard: for them all four columns show the decision as "
                     f"run. The guard columns only differ for runs made by the earlier path.")
            L.append("")
        L.append("\"As run\" is the decision each stored run ended with, under whatever "
                 "version produced it. The other three columns are computed from the same "
                 "stored runs: the model's own answer, and that answer after each guard "
                 "rule, so they are comparable across versions. The pipeline uses the "
                 "\"more against than with\" rule.")
        L.append("")
        changed = acc.get("guard_changed") or []
        if changed:
            L += ["Where the guard changes the model's decision on a labelled date:", "",
                  "| Date | Label | With / against | Fires under | Guard right? |",
                  "|---|---|---|---|---|"]
            for c in changed:
                L.append(f"| {c['date']} | {c['expected']} | {c['with']} / {c['against']} | "
                         f"{'tie rule only' if c['tie_only'] else 'both rules'} | "
                         f"{'yes' if c['correct'] else 'no'} |")
            L.append("")
        if acc["misses"]:
            L += ["Decisions that disagree with the label:", "",
                  "| Date | Label | System | Evidence direction (with / against) | Model |",
                  "|---|---|---|---|---|"]
            for m in acc["misses"]:
                ds = m["direction_summary"]
                L.append(f"| {m['date']} | {m['expected']} | {m['verdict']} | "
                         f"{ds.get('consistent', 0)} / {ds.get('inconsistent', 0)} | "
                         f"{m.get('model_used') or 'not recorded'} |")
            L.append("")
        wrongly_explained = [m for m in acc["misses"] if m["expected"] == "REFUSE"]
        if wrongly_explained:
            L.append(f"**{len(wrongly_explained)} date(s) labelled REFUSE were explained.** "
                     f"For a system whose claim is that it declines rather than invent a "
                     f"cause, this is the number that matters most, and the sample behind "
                     f"it ({acc['as_run']['refuse_labelled']['n']} graded REFUSE labels) is "
                     f"too small to bound the true rate.")
            L.append("")
        guard_note = ("The guard and both of its rules were chosen after looking at runs on "
                      "these labelled dates, so the guard columns are in-sample and are not "
                      "evidence the guard generalises - that needs inputs it was not designed "
                      "on (see results/robustness_report.md).")
        if acc.get("n_should_refuse"):
            guard_note += (" Every REFUSE-labelled date here is a day that moved against the "
                           "news of the days before it, which is exactly what the guard "
                           "detects; it has not been tested on a should-refuse day of any "
                           "other kind.")
        L.append(guard_note)
        L.append("")
        L.append("Caveats: each label rests on one round of research that nobody has "
                 "checked independently, and rows whose notes start with CHECK are "
                 "borderline; the labels "
                 "were made after the system had been run on these dates, from market "
                 "reports and without sight of its answers; a correct EXPLAIN decision "
                 "means the system chose to explain, not that its text names the documented "
                 "cause.")
        L.append("")

    # 3 ---------------------------------------------------------------
    L += ["## 3. Citations", ""]
    if cit["total_citations"]:
        L += [f"- Citations checked: {cit['total_citations']} across "
              f"{cit['n_explained_with_report']} explained run(s) "
              f"({cit['n_explained'] - cit['n_explained_with_report']} explained run(s) "
              f"have no citation report).",
              f"- Cited document was in the evidence shown to the model: {fmt_rate(cit['resolve'])}"]
        if cit["support_score_min"] is not None:
            L.append(f"- Claim-to-document similarity: minimum {cit['support_score_min']}, "
                     f"median {cit['support_score_median']}; flagged weak: {cit['weak_support']['k']}.")
        if cit.get("low_rank"):
            L.append(f"- Cited document not among the closest to its claim: {fmt_rate(cit['low_rank'])}")
        L.append("")
        L.append("What this establishes: the model did not invent source ids. What it does "
                 "not: that each cited document supports the claim attached to it. The "
                 "similarity score cannot fail in practice (any two coffee-market texts "
                 "score 0.75-0.95 with this embedding model, against a 0.60 threshold), so "
                 "a 0% weak-support rate carries no information.")
        L.append("")
        audit = cit.get("human_audit")
        if audit and audit.get("n_labelled"):
            L.append(f"Human audit (scripts/audit_citations.py): {audit['n_labelled']} "
                     f"citations labelled; supported {fmt_rate(audit['supported'])}.")
        else:
            L.append("No human citation audit has been recorded yet. Run "
                     "`python -m scripts.audit_citations --export`, label the sheet, then "
                     "`--score` to add a measured support rate here.")
        L.append("")
    else:
        L += ["No citation reports in these runs.", ""]

    # 4 ---------------------------------------------------------------
    L += ["## 4. Evidence", "",
          f"- Relevance gate passed: {fmt_rate(ev['gate_pass'])}",
          f"- Documents retrieved per run: {ev['avg_documents_retrieved']} on average; "
          f"runs with none: {ev['zero_retrieval']['k']}",
          f"- Accepted documents, over the {ev['runs_with_sources']} run(s) that saved their "
          f"sources: {ev['accepted_documents']}, of which {ev['distinct_stories']} are distinct "
          f"stories. Copies of another accepted document: {fmt_rate(ev['duplicate_share'])}",
          f"- Accepted documents dated after the anomaly date: {fmt_rate(ev['published_after_anomaly'])}",
          f"- Per explained run: {ev['avg_accepted_documents']} accepted documents, "
          f"{ev['avg_distinct_stories']} distinct stories, {ev['avg_unique_domains']} domains "
          f"(over {ev['explained_with_sources']} explained run(s) that saved their sources)",
          f"- Explained from a single distinct story: {fmt_rate(ev['single_story'])}",
          ""]
    if ev["gate_pass"]["pct"] is not None and ev["gate_pass"]["pct"] >= 90:
        L.append(f"The gate passes {ev['gate_pass']['pct']}% of dates, so in these runs it "
                 f"does almost no filtering at the date level; the refusals come from the "
                 f"model's own judgement, not from the gate.")
        L.append("")
    if ev["published_after_anomaly"]["k"]:
        L.append("Documents dated after the anomaly were used as evidence in these runs "
                 "even though the design excludes them. Pipeline version 2 filters them "
                 "locally; these runs predate that.")
        L.append("")
    if ev["duplicate_share"]["k"]:
        L.append("Domain counts overstate independence when the same wire story appears on "
                 "several sites; distinct stories is the better measure of how many "
                 "sources an explanation rests on.")
        L.append("")

    # 5 ---------------------------------------------------------------
    L += ["## 5. Evidence direction (heuristic)", ""]
    if dir_["n_evaluated"]:
        L.append(f"Of {dir_['n_explained']} explained runs, {dir_['n_evaluated']} have a "
                 f"direction tally. Accepted evidence leaned the same way as the move in "
                 f"{fmt_rate(dir_['aligned'])}.")
        L.append("")
        against = [d for d in dir_["rows"] if not d["aligned"]]
        if against:
            L.append("Explained despite evidence leaning against the move: "
                     + ", ".join(f"{d['date']} ({d['consistent']} with / "
                                 f"{d['inconsistent']} against)" for d in against) + ".")
            L.append("")
        L.append("This measures the retrieved documents with a word list "
                 "(src/rag/direction.py). It is not a check that the explanation is correct.")
        L.append("")
    else:
        L += ["No explained run has a direction tally.", ""]

    blind = metrics.get("blind") or {}
    if blind.get("n_runs"):
        L += ["### Runs decided by the blind-evidence rule", ""]
        L.append(f"{blind['n_runs']} run(s). The model read the recent documents without "
                 f"being told the move: it returned a reading for "
                 f"{fmt_rate(blind['documents_read'])} of the documents it was shown, and of "
                 f"the reasons it reported, the quote was found in the document for "
                 f"{fmt_rate(blind['reasons_with_quote_found'])}. A reason whose quote is not "
                 f"found is discarded before the rule runs.")
        L.append("")
        L += ["| Move | Runs | Explained |", "|---|---|---|"]
        for direction, b in blind["by_direction"].items():
            L.append(f"| {direction} | {b['runs']} | {fmt_rate(b['explained_rate'])} |")
        L.append("")
        either = blind.get("explained_either_direction") or {}
        if either.get("n"):
            L.append(f"Direction test on every explained date: had prices moved the opposite "
                     f"way, the same readings would still have produced an explanation for "
                     f"{fmt_rate(either)}. That can only happen when the deciding evidence is "
                     f"split evenly, and such explanations are marked tentative. Under the "
                     f"earlier design, where the model was told the move, this was the test "
                     f"it failed.")
            L.append("")
        stages = blind.get("refusal_stages") or {}
        if any((v or {}).get("n") for v in stages.values()):
            L += ["Where the refused dates stopped:", "",
                  "| Stopped at | Refused dates |", "|---|---|"]
            for key, text in STOP_STAGES:
                L.append(f"| {text} | {fmt_rate(stages[key])} |")
            L.append("")
            L.append("The first two rows are search and filtering: no change to the reading "
                     "or the rule can explain a date for which nothing recent and relevant "
                     "was retrieved.")
            L.append("")
        missed = blind.get("refused_but_should_explain") or []
        if missed:
            names = dict(STOP_STAGES)
            L += ["Dates labelled as explainable that were refused:", "",
                  "| Date | Stopped at | Retrieved | Of those recent | Relevant enough to read |",
                  "|---|---|---|---|---|"]
            for row in missed:
                recent = "not recorded" if row["retrieved_recent"] is None else row["retrieved_recent"]
                L.append(f"| {row['date']} | {names.get(row['stage'], row['stage'])} | "
                         f"{row['retrieved']} | {recent} | {row['accepted']} |")
            L.append("")
        L.append("The same stored readings under other settings of the rule (no model is "
                 "called again). The first row is the pipeline's setting. If the rows "
                 "differ a lot, the setting is doing the work and the result should not be "
                 "trusted; \"pooled\" counts every recent reading equally, the shape of "
                 "rule that made the earlier guard refuse most down days.")
        L.append("")
        L += ["| Rule | Recent means | Explained, all runs | Explained when it should | "
              "Refused when it should |", "|---|---|---|---|---|"]
        for row in blind["settings"]:
            days = row["fresh_trading_days"]
            L.append(f"| {row['rule']} | same day or up to {days} trading day"
                     f"{'s' if days != 1 else ''} before | {fmt_rate(row['explained_all'])} | "
                     f"{fmt_rate(row['explain_labelled'])} | {fmt_rate(row['refuse_labelled'])} |")
        L.append("")

        if blind.get("variants"):
            L.append("Other versions of what counts as evidence, scored the same way from "
                     "the same readings. A version is only worth adopting if it raises the "
                     "last column without lowering the one before it. With this few labelled "
                     "dates a difference of one date is not evidence.")
            L.append("")
            L += ["| What counts as evidence | Explained, all runs | Explained when it should | "
                  "Refused when it should |", "|---|---|---|---|"]
            first = blind["settings"][0]
            L.append(f"| As the pipeline runs | {fmt_rate(first['explained_all'])} | "
                     f"{fmt_rate(first['explain_labelled'])} | {fmt_rate(first['refuse_labelled'])} |")
            for row in blind["variants"]:
                L.append(f"| {row['text']} | {fmt_rate(row['explained_all'])} | "
                         f"{fmt_rate(row['explain_labelled'])} | {fmt_rate(row['refuse_labelled'])} |")
            L.append("")
        if blind.get("near_gate_read"):
            L.append(f"Recent documents read although they scored just under the relevance "
                     f"filter's bar: {blind['near_gate_read']}, of which "
                     f"{blind['near_gate_usable']} gave a reason with a quote that was found. "
                     f"The \"only documents that cleared the filter\" row above is the result "
                     f"without them.")
            L.append("")
        if blind.get("search_yield"):
            L.append("Which search found the documents. \"Read candidates\" are documents "
                     "relevant enough to be read; \"gave a reason\" are those whose reading "
                     "counted as evidence; the last two columns are documents an explanation "
                     "rests on and the number of dates they decided. A document found by two "
                     "searches is credited to the one that returned it first.")
            L.append("")
            L += ["| Search | Read candidates | Gave a reason | In an explanation | Dates |",
                  "|---|---|---|---|---|"]
            for row in blind["search_yield"]:
                L.append(f"| {row['search']} | {row['candidates']} | {row['usable']} | "
                         f"{row['deciding']} | {row['dates_decided']} |")
            L.append("")

    guard = metrics.get("guard_effect") or {}
    if (guard.get("withheld_overall") or {}).get("n"):
        L.append(f"Across every run, the direction guard withholds "
                 f"{fmt_rate(guard['withheld_overall'])} of the explanations the model wrote. "
                 f"By direction of the move:")
        L.append("")
        L += ["| Move | Runs | Model explained | Guard withholds | Explained after the guard |",
              "|---|---|---|---|---|"]
        for direction, b in guard["by_direction"].items():
            L.append(f"| {direction} | {b['runs']} | {b['model_explained']} | "
                     f"{fmt_rate(b['withheld_rate'])} | {fmt_rate(b['explained_after_guard'])} |")
        L.append("")
        L.append("If the guard withholds far more on one direction than the other, it is not "
                 "only catching unsupported explanations: it is also reporting which way the "
                 "retrieved news leans. The GDELT queries are about supply shocks (frost, "
                 "drought, tariffs, shipping), which are stories about prices rising, so on a "
                 "day prices fell the tally starts against the move. A should-refuse label set "
                 "made only of days in the direction the guard already refuses cannot show "
                 "that the guard works.")
        L.append("")

    # 6 ---------------------------------------------------------------
    L += ["## 6. By year", "", "| Year | Explained | Refused | Fault |", "|---|---|---|---|"]
    for year, b in metrics["by_year"].items():
        L.append(f"| {year} | {b[EXPLAINED]} | {b[REFUSED]} | {b[FAULT]} |")
    L.append("")
    L.append("Dates the answering model could have seen in training are a weaker test of "
             "retrieval than dates after its knowledge cutoff. That split needs the "
             "cutoff of each model used and is reported by src/evaluation/robustness.py.")
    L.append("")
    return "\n".join(L)


def load_results() -> list:
    """Batch results if present, else every stored coffee pipeline output."""
    eval_path = RESULTS_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            return json.load(f).get("results", [])
    results = []
    for path in sorted(RESULTS_DIR.glob("pipeline_output_coffee_*.json")):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(
            "No pipeline outputs found. Run the pipeline first or "
            "use: python -m src.evaluation.batch_runner")
    return results


def load_outputs(directory) -> list:
    """Every coffee pipeline output in a directory (for example an archived
    set under results/archive/pipeline_v3)."""
    results = []
    for path in sorted(Path(directory).glob("pipeline_output_coffee_*.json")):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No pipeline_output_coffee_*.json files in {directory}")
    return results


def run_evaluation(verbose: bool = True, outputs_dir=None, name: str = None,
                   labels_path=None) -> dict:
    """Compute the evaluation metrics and write the report.

    By default this reads the batch results and writes evaluation_report.md.
    With outputs_dir it reads the pipeline outputs in that directory instead,
    and with name it writes evaluation_report_<name>.md - so an archived set
    of runs gets its own report next to the current one."""
    labels = None
    if labels_path:
        if not Path(labels_path).exists():
            raise SystemExit(f"{labels_path} not found")
        labels = load_labels(Path(labels_path))
    metrics = compute_metrics(load_outputs(outputs_dir) if outputs_dir else load_results(),
                              labels=labels)
    suffix = f"_{name}" if name else ""

    metrics_path = RESULTS_DIR / f"evaluation_report{suffix}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = generate_report(metrics)
    report_path = RESULTS_DIR / f"evaluation_report{suffix}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    if verbose:
        print(report)
        print(f"\nSaved metrics to {metrics_path}")
        print(f"Saved report to {report_path}")
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation report from stored pipeline runs")
    parser.add_argument("--outputs", default=None, metavar="DIR",
                        help="Read the pipeline outputs in DIR instead of the batch results, "
                             "e.g. results/archive/pipeline_v3")
    parser.add_argument("--name", default=None,
                        help="Write evaluation_report_<name>.md/.json instead of the default "
                             "report (required with --outputs, so the current report is not "
                             "overwritten by an archived set)")
    parser.add_argument("--labels", default=None, metavar="CSV",
                        help="Score against this answer key instead of "
                             "data/labeling/labels_by_written_rule.csv, e.g. the labels made "
                             "before the written rule, "
                             "data/labeling/explanation_backtest_labels.csv (needs --name)")
    args = parser.parse_args()
    if (args.outputs or args.labels) and not args.name:
        parser.error("--outputs and --labels need --name, so the current report is not "
                     "overwritten")
    run_evaluation(outputs_dir=args.outputs, name=args.name, labels_path=args.labels)


if __name__ == "__main__":
    main()
