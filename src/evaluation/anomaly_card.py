"""
One flagged day, everything the project knows about it, on one card.

The project now answers a flagged day from four directions, and until now
each lived in its own report:

    the base detector     how large the move was against recent volatility
    the factor layer      how much of it the dollar, equities and the broad
                          complex account for
    the cross-commodity   how many other watched markets flagged the same
    check                 day
    the numeraire test    whether the move survives being priced in euro,
                          yen, yuan and rupees
    the RAG pipeline      whether any news of the day gives a reason, with
                          a verified quote

A reader who has to open five files to judge one day will not do it, and
the four quantitative layers all bear on the same question the explanation
pipeline silently assumes away: *is this a commodity story at all?*

Every field carries one of three tags, so the distinction between what was
read and what was inferred is machine-checkable rather than a writing
convention that erodes:

    OBSERVED     read directly from stored data. No model, no estimate.
    TESTED       the output of a stated method on a stated sample.
    HYPOTHESIS   a plausible mechanism that has not been tested here.

Nothing on a card is a HYPOTHESIS today. The tag exists so that the first
person to add an inferred field has to mark it, rather than having it blend
in with the measured ones.

The card deliberately does NOT produce an overall verdict. Four layers
disagreeing is information, and collapsing them into one word would throw
it away: a day that is macro-driven by the factor model, survives the
numeraire test, and has a quoted supply story is a genuinely interesting
day, and no single label describes it.

Usage:
    python -m src.evaluation.anomaly_card --commodity coffee --date 2025-04-07
    python -m src.evaluation.anomaly_card --commodity coffee --labelled
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.commodities import COMMODITIES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

OBSERVED, TESTED, HYPOTHESIS = "observed", "tested", "hypothesis"
TAGS = (OBSERVED, TESTED, HYPOTHESIS)


def tagged(value, tag: str, source: str, note: str = "") -> dict:
    """A value that cannot be quoted without saying where it came from."""
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {TAGS}, got {tag!r}")
    out = {"value": value, "tag": tag, "source": source}
    if note:
        out["note"] = note
    return out


def _read_csv(path: Path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def detector_fields(commodity: str, day: pd.Timestamp) -> dict:
    cfg = COMMODITIES[commodity]
    det = _read_csv(RESULTS_DIR / cfg.anomalies_file, parse_dates=["date"])
    if det is None:
        return {}
    det = det.set_index("date")
    if day not in det.index:
        return {}
    r = det.loc[day]
    src = cfg.anomalies_file
    out = {
        "price": tagged(float(r["price"]), OBSERVED, src),
        "z_score": tagged(None if pd.isna(r.get("z_score")) else float(r["z_score"]),
                          TESTED, src, "day-over-day return against a 75-day rolling spread"),
        "flagged": tagged(bool(r.get("anomaly_flag")), TESTED, src),
        "anomaly_type": tagged(r.get("anomaly_type") if pd.notna(r.get("anomaly_type")) else None,
                               TESTED, src),
    }
    if "contract_switch" in det.columns and bool(r.get("contract_switch")):
        out["contract_switch"] = tagged(
            True, OBSERVED, "data/labeling/contract_switch_dates.csv",
            "the series switched futures contract: this move is an artefact, not a market move")
    return out


def factor_fields(commodity: str, day: pd.Timestamp) -> dict:
    path = RESULTS_DIR / f"{commodity}_factor_adjusted.csv"
    df = _read_csv(path, parse_dates=["date"])
    if df is None:
        return {}
    df = df.set_index("date")
    if day not in df.index:
        return {}
    r = df.loc[day]
    src = path.name
    g = lambda k: None if pd.isna(r.get(k)) else float(r[k])  # noqa: E731
    return {
        "factor_class": tagged(r.get("factor_class") or None, TESTED, src,
                               "betas fitted on the 75 days before this one, excluding it"),
        "residual_z": tagged(g("residual_z"), TESTED, src),
        "factor_explained_return": tagged(g("factor_explained"), TESTED, src),
        "unexplained_return": tagged(
            g("residual"), TESTED, src,
            "what the dollar, equities and the broad complex do not account for; "
            "not attributed to any cause"),
        "window_r2": tagged(g("window_r2"), TESTED, src),
    }


def cross_commodity_fields(commodity: str, day: pd.Timestamp) -> dict:
    path = RESULTS_DIR / "cross_commodity.json"
    if not path.exists():
        return {}
    from src.evaluation.cross_commodity import classify, flag_table
    table = flag_table()
    if commodity not in table.columns or day not in table.index:
        return {}
    rows = classify(table, commodity, 2).set_index("date")
    others = [c for c in table.columns
              if c != commodity and table.loc[day, c] == True]  # noqa: E712
    return {
        "other_commodities_flagged": tagged(others, OBSERVED, "cross_commodity.json"),
        "cross_commodity_verdict": tagged(
            rows["verdict"].get(day), TESTED, "cross_commodity.json",
            "macro when at least 2 of the other watched markets flagged the same day"),
    }


def numeraire_fields(commodity: str, day: pd.Timestamp) -> dict:
    path = RESULTS_DIR / f"{commodity}_numeraire.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = {r["date"]: r for r in json.load(f)}
    r = rows.get(str(day.date()))
    if not r:
        return {}
    src = path.name
    return {
        "move_in_local_currencies_pct": tagged(r["per_currency_pct"], OBSERVED, src,
                                               "exact translation, no estimation"),
        "survival_after_translation": tagged(r["survival"], OBSERVED, src),
        "numeraire_reading": tagged(r["reading"], TESTED, src),
        "dollar_beta": tagged(r["regression"]["dollar_beta"], TESTED, src),
        "unexplained_by_dollar_pct": tagged(r["regression"]["unexplained_pct"], TESTED, src),
    }


def pipeline_fields(commodity: str, day: pd.Timestamp) -> dict:
    path = RESULTS_DIR / f"pipeline_output_{commodity}_{day.date()}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    er = out.get("explanation_result") or {}
    src = path.name
    fields = {
        "pipeline_version": tagged(out.get("pipeline_version"), OBSERVED, src),
        "outcome_tier": tagged((out.get("outcome") or {}).get("tier"), TESTED, src),
        "decision_reason": tagged(er.get("reason"), TESTED, src),
        "explanation": tagged(er.get("explanation"), OBSERVED, src,
                              "assembled from quotes verified against their articles"),
        "n_citations": tagged(len(er.get("citations") or []), OBSERVED, src),
    }
    blind = er.get("blind_evidence") or {}
    if blind:
        fields["readings_set_aside"] = tagged(
            [s.get("why") for s in (blind.get("set_aside") or [])], TESTED, src,
            "reports that did not fit the move: wrong market, a trading reason, "
            "or a move of the wrong size")
    return fields


def build_card(commodity: str, day) -> dict:
    day = pd.Timestamp(day)
    card = {"commodity": commodity, "date": str(day.date()), "fields": {}}
    for part in (detector_fields, factor_fields, cross_commodity_fields,
                 numeraire_fields, pipeline_fields):
        card["fields"].update(part(commodity, day))
    card["layers_present"] = sorted({f["source"] for f in card["fields"].values()})
    card["missing_layers"] = [
        name for name, present in (
            ("factor model", any(k.startswith("factor_") or k == "residual_z"
                                 for k in card["fields"])),
            ("cross-commodity", "cross_commodity_verdict" in card["fields"]),
            ("numeraire", "numeraire_reading" in card["fields"]),
            ("explanation pipeline", "outcome_tier" in card["fields"]),
        ) if not present]
    return card


def render(card: dict) -> str:
    L = [f"## {card['commodity']} — {card['date']}", ""]
    L.append("| Field | Value | Evidence | Source |")
    L.append("|---|---|---|---|")
    for name, f in card["fields"].items():
        v = f["value"]
        if isinstance(v, float):
            v = f"{v:.4g}"
        elif isinstance(v, dict):
            v = ", ".join(f"{k} {x:+.2f}%" for k, x in v.items() if x is not None)
        elif isinstance(v, list):
            v = ", ".join(str(x) for x in v) or "none"
        elif v is None:
            v = "-"
        v = str(v).replace("\n", " ").replace("|", "\\|")
        if len(v) > 160:
            v = v[:157] + "…"
        L.append(f"| {name.replace('_', ' ')} | {v} | {f['tag']} | `{f['source']}` |")
    if card["missing_layers"]:
        L.append("")
        L.append(f"Not run for this day: {', '.join(card['missing_layers'])}. "
                 "A missing layer is a gap in what is known, not a negative result.")
    L.append("")
    L.append("No overall verdict is given. The layers can disagree, and that is "
             "information: a day the factor model calls macro-driven, that survives "
             "translation into other currencies, and that carries a quoted supply story "
             "is a real puzzle, and one word would hide it.")
    return "\n".join(L)


def run(commodity: str, dates, verbose: bool = True) -> list:
    cards = [build_card(commodity, d) for d in dates]
    if verbose:
        for c in cards:
            print(render(c))
            print()
    return cards


def main():
    p = argparse.ArgumentParser(description="Everything known about one flagged day")
    p.add_argument("--commodity", required=True, choices=sorted(COMMODITIES))
    p.add_argument("--date", action="append", default=[])
    p.add_argument("--labelled", action="store_true",
                   help="every date in the answer key")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    dates = list(args.date)
    if args.labelled:
        lab = pd.read_csv(REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv")
        dates += list(pd.to_datetime(lab["date"]).dt.date.astype(str))
    if not dates:
        raise SystemExit("Give --date or --labelled.")

    cards = run(args.commodity, dates)
    if args.save:
        out = RESULTS_DIR / f"{args.commodity}_anomaly_cards.md"
        out.write_text("# Anomaly cards\n\n" + "\n\n".join(render(c) for c in cards),
                       encoding="utf-8")
        with open(RESULTS_DIR / f"{args.commodity}_anomaly_cards.json", "w",
                  encoding="utf-8") as f:
            json.dump(cards, f, indent=2)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
