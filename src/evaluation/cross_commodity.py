"""
Was this a disruption in one commodity, or a day the whole complex moved?

The base detector (src/modeling/anomaly_detector.py) answers "did this
price move unusually". It cannot answer "was the move about this
commodity", because it never looks at anything else. A day when the dollar
drops, or a risk-off day when everything sells together, produces an
extreme z-score in coffee that has nothing to do with coffee - and the
explanation pipeline will then go looking for a coffee story, because a
coffee story is what it was asked for.

This module is the cheapest available check on that, and it needs no data
the project does not already have: it reads the five commodities' stored
detections and counts how many of them flag on the same date.

    A day several unrelated commodities flag together is a macro day
    until something shows otherwise.

Coffee, crude oil, wheat, gold and silver are driven by different physical
things - a Brazilian frost has nothing to do with Russian wheat or COMEX
gold. What they share is the dollar they are priced in, the interest rate
that discounts them, and the risk appetite of the people holding them. So
agreement across them is evidence about the shared factors, not about any
commodity's own supply.

What this is NOT: a substitute for the factor-adjusted residual in
src/modeling/factor_model.py, which measures the same thing directly
against the dollar and the market rather than by proxy. This one is
coarser. It is here because it runs on files already in the repository,
with no new series to fetch and no model to call, and because a count of
commodities is something a reader can check by eye.

Two cautions, both of which the numbers below are reported with:

  - The commodities do not cover the same years. Wheat's series starts in
    2009 and coffee's in 2018, so a day in 2010 has fewer commodities that
    could possibly agree. Every rate here is computed over the dates where
    the commodity in question and at least two others both have data, and
    the denominator is printed.
  - Gold and silver are close substitutes and move together for reasons
    that are not macro. They are counted as separate commodities, which
    makes the macro share an overestimate. The report prints the
    precious-metals-collapsed figure beside it.

Usage:
    python -m src.evaluation.cross_commodity
    python -m src.evaluation.cross_commodity --min-others 2
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.commodities import COMMODITIES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_JSON = RESULTS_DIR / "cross_commodity.json"
OUTPUT_MD = RESULTS_DIR / "cross_commodity_report.md"

# How many OTHER commodities must flag the same day before the day is called
# macro. Two is the smallest number that is not a coincidence between a pair
# of substitutes (gold and silver); the report also prints the curve across
# 1, 2, 3 and 4 so the choice is visible rather than buried.
DEFAULT_MIN_OTHERS = 2

# Gold and silver are substitutes: they move together for reasons that are
# not macro, so a "both metals flagged" day is weaker evidence of a macro
# day than "coffee and wheat flagged". The report gives both views.
SUBSTITUTE_GROUPS = {"gold": "precious", "silver": "precious"}


def load_detections(commodity_key: str) -> pd.DataFrame:
    """The stored detections for one commodity, dates indexed, or an empty
    frame when that commodity has not been run."""
    cfg = COMMODITIES[commodity_key]
    path = RESULTS_DIR / cfg.anomalies_file
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    keep = [c for c in ("date", "anomaly_flag", "anomaly_type", "z_score",
                        "contract_switch") if c in df.columns]
    return df[keep]


def flag_table(commodities=None) -> pd.DataFrame:
    """One row per date, one boolean column per commodity that has data for
    it. A commodity with no row for a date gets NA, not False: "this market
    was closed or the series had not started" is not the same as "this
    market was calm", and treating it as False would understate agreement
    in the early years."""
    frames = {}
    for key in (commodities or list(COMMODITIES)):
        df = load_detections(key)
        if df.empty:
            continue
        s = df.set_index("date")["anomaly_flag"].astype("boolean")
        # A day the price series switched futures contract shows a move no
        # contract made, so it is not evidence of anything: drop it rather
        # than let it agree or disagree with the other commodities.
        if "contract_switch" in df.columns:
            switch = df.set_index("date")["contract_switch"].fillna(False).astype(bool)
            s = s.mask(switch)
        frames[key] = s
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def classify(table: pd.DataFrame, commodity: str, min_others: int) -> pd.DataFrame:
    """For each date this commodity flagged, how many others flagged too.

    Only dates where at least `min_others` OTHER commodities have data are
    scored, so a 2010 wheat flag is not called commodity-specific merely
    because coffee's series had not started."""
    if commodity not in table.columns:
        return pd.DataFrame()
    others = [c for c in table.columns if c != commodity]
    mine = table[commodity]
    others_with_data = table[others].notna().sum(axis=1)
    others_flagged = (table[others] == True).sum(axis=1)  # noqa: E712

    rows = pd.DataFrame({
        "date": table.index,
        "others_with_data": others_with_data.values,
        "others_flagged": others_flagged.values,
    })
    rows["flagged"] = mine.fillna(False).astype(bool).values
    rows["scorable"] = rows["others_with_data"] >= min_others
    rows["verdict"] = None
    scored = rows["flagged"] & rows["scorable"]
    rows.loc[scored & (rows["others_flagged"] >= min_others), "verdict"] = "macro"
    rows.loc[scored & (rows["others_flagged"] < min_others), "verdict"] = "commodity-specific"
    return rows


def _collapse_substitutes(table: pd.DataFrame) -> pd.DataFrame:
    """Gold and silver counted once between them: flagged if either is, with
    data if either has. Removes the easiest source of spurious agreement."""
    out, groups = {}, {}
    for col in table.columns:
        groups.setdefault(SUBSTITUTE_GROUPS.get(col, col), []).append(col)
    for name, cols in groups.items():
        if len(cols) == 1:
            out[name] = table[cols[0]]
        else:
            block = table[cols]
            any_flag = (block == True).any(axis=1)  # noqa: E712
            out[name] = any_flag.mask(block.notna().sum(axis=1) == 0).astype("boolean")
    return pd.DataFrame(out).sort_index()


def summarise(table: pd.DataFrame, min_others: int = DEFAULT_MIN_OTHERS) -> dict:
    collapsed = _collapse_substitutes(table)
    out = {"min_others": min_others,
           "commodities": list(table.columns),
           "per_commodity": {},
           "curve": {},
           "overlap": {}}

    for key in table.columns:
        rows = classify(table, key, min_others)
        scored = rows[rows["flagged"] & rows["scorable"]]
        counts = scored["verdict"].value_counts().to_dict()
        n = len(scored)
        group = SUBSTITUTE_GROUPS.get(key, key)
        collapsed_rows = classify(collapsed, group, min_others) if group in collapsed else None
        collapsed_scored = (collapsed_rows[collapsed_rows["flagged"] & collapsed_rows["scorable"]]
                            if collapsed_rows is not None and not collapsed_rows.empty else None)
        out["per_commodity"][key] = {
            "flagged_total": int(rows["flagged"].sum()),
            "scorable": n,
            "macro": int(counts.get("macro", 0)),
            "commodity_specific": int(counts.get("commodity-specific", 0)),
            "macro_share": (counts.get("macro", 0) / n) if n else None,
            "first_date": str(rows.loc[rows["flagged"], "date"].min().date())
                          if rows["flagged"].any() else None,
            "last_date": str(rows.loc[rows["flagged"], "date"].max().date())
                         if rows["flagged"].any() else None,
            "macro_share_metals_collapsed": (
                (collapsed_scored["verdict"] == "macro").mean()
                if collapsed_scored is not None and len(collapsed_scored) else None),
        }
        # How the verdict moves with the threshold, so the choice of 2 is
        # visible rather than assumed.
        out["curve"][key] = {
            str(k): float((classify(table, key, k)
                           .pipe(lambda r: r[r["flagged"] & r["scorable"]])["verdict"]
                           == "macro").mean())
            for k in (1, 2, 3, 4)
            if len(classify(table, key, k).pipe(lambda r: r[r["flagged"] & r["scorable"]]))
        }

    # Which pairs agree most. A pair that always agrees is either two views
    # of one thing (gold and silver) or a shared driver.
    for a in table.columns:
        for b in table.columns:
            if a >= b:
                continue
            both = table[[a, b]].dropna()
            if both.empty:
                continue
            fa, fb = both[a].astype(bool), both[b].astype(bool)
            either = int((fa | fb).sum())
            out["overlap"][f"{a}|{b}"] = {
                "days_both_have_data": int(len(both)),
                "both_flagged": int((fa & fb).sum()),
                "jaccard": (int((fa & fb).sum()) / either) if either else None,
            }
    return out


LABELS_PATH = REPO_ROOT / "data" / "labeling" / "labels_by_written_rule.csv"

# Words that mark a cause as being about the whole complex rather than this
# market. Matched against the answer key's own `known_cause` text, which was
# written from market reports before any of this analysis existed - so the
# key is being read, not reinterpreted.
MACRO_CAUSE_WORDS = (
    "stronger dollar", "weaker dollar", "the dollar", "dollar index",
    "across most commodities", "across commodities", "broad sell-off",
    "broad selloff", "risk-off", "trade-war", "trade war", "tariff announcements",
)


def score_labelled_dates(table: pd.DataFrame, min_others: int = DEFAULT_MIN_OTHERS,
                         commodity: str = "coffee") -> dict:
    """Which dates in the answer key are days the whole complex moved.

    This exists because of what the key's own text says on those dates. The
    key marks a date EXPLAIN when a documented cause exists. It does not ask
    whether the cause is specific to this market, and on two of the three
    macro days below the cause it records is a macro one in its own words
    ("a stronger dollar weighed on commodities"; "a sell-off across most
    commodities"). A rule built to demand a report about THIS market will
    refuse those dates and be scored wrong for it.

    Reported, not acted on: nothing here relabels the key. Changing a label
    after seeing which dates a rule failed is exactly the move this project
    refuses to make. It is evidence that the key needs a third category, to
    be decided by the people who own it."""
    if not LABELS_PATH.exists():
        return {}
    labels = pd.read_csv(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"])
    rows = classify(table, commodity, min_others).set_index("date")
    out = []
    for _, r in labels.iterrows():
        v = rows["verdict"].get(r["date"])
        # `is True` fails here: these are pandas nullable booleans, not the
        # Python singleton, so identity never matches and the column came
        # back empty.
        others = ([c for c in table.columns
                   if c != commodity and table.loc[r["date"], c] == True]  # noqa: E712
                  if r["date"] in table.index else [])
        cause = str(r.get("known_cause") or "")
        out.append({
            "date": str(r["date"].date()),
            "label": r.get("expected_outcome"),
            "verdict": v,
            "others_flagged": others,
            "cause_names_a_macro_driver": any(w in cause.lower() for w in MACRO_CAUSE_WORDS),
            "known_cause": cause,
        })
    macro = [d for d in out if d["verdict"] == "macro"]
    return {
        "n_labelled": len(out),
        "n_scored": sum(1 for d in out if d["verdict"]),
        "n_macro": len(macro),
        "n_macro_whose_key_text_names_a_macro_driver":
            sum(1 for d in macro if d["cause_names_a_macro_driver"]),
        "dates": out,
    }


def report(summary: dict) -> str:
    L = ["# Cross-Commodity Check: which flagged days are macro days", ""]
    L.append("The base detector flags a day when one price moved unusually. It does not "
             "look at any other market, so it cannot tell a supply disruption from a day "
             "the dollar moved and everything repriced. This counts how many of the five "
             "watched commodities flag on the same date.")
    L.append("")
    L.append(f"A flagged day is called **macro** when at least {summary['min_others']} of "
             "the other commodities flagged it too, and **commodity-specific** otherwise. "
             "Only days on which that many other commodities actually have price data are "
             "scored, so an early wheat flag is not called specific merely because coffee's "
             "series had not started. Days the series switched futures contract are dropped.")
    L.append("")
    L.append("| Commodity | Flagged | Scorable | Macro | Commodity-specific | Macro share | "
             "Metals counted once |")
    L.append("|---|---|---|---|---|---|---|")
    for key, r in summary["per_commodity"].items():
        share = "-" if r["macro_share"] is None else f"{100 * r['macro_share']:.1f}%"
        coll = ("-" if r["macro_share_metals_collapsed"] is None
                else f"{100 * r['macro_share_metals_collapsed']:.1f}%")
        L.append(f"| {key} | {r['flagged_total']} | {r['scorable']} | {r['macro']} | "
                 f"{r['commodity_specific']} | {share} | {coll} |")
    L.append("")
    L.append("The last column counts gold and silver as one market. They are substitutes "
             "and move together for reasons that are not macro, so the plain macro share "
             "overstates agreement for them.")
    L.append("")
    L.append("## How the verdict moves with the threshold")
    L.append("")
    L.append("Share of flagged days called macro, as the number of other commodities "
             "required rises. A share that falls away quickly means the agreement is thin.")
    L.append("")
    L.append("| Commodity | 1 other | 2 others | 3 others | 4 others |")
    L.append("|---|---|---|---|---|")
    for key, curve in summary["curve"].items():
        cells = " | ".join(f"{100 * curve[k]:.0f}%" if k in curve else "-"
                           for k in ("1", "2", "3", "4"))
        L.append(f"| {key} | {cells} |")
    L.append("")
    L.append("## Which pairs agree")
    L.append("")
    L.append("Jaccard: of the days either flagged, the share on which both did. Two markets "
             "with a shared driver score high; two with unrelated physical causes should not.")
    L.append("")
    L.append("| Pair | Days both have data | Both flagged | Jaccard |")
    L.append("|---|---|---|---|")
    for pair, r in sorted(summary["overlap"].items(),
                          key=lambda kv: -(kv[1]["jaccard"] or 0)):
        j = "-" if r["jaccard"] is None else f"{r['jaccard']:.3f}"
        L.append(f"| {pair.replace('|', ' / ')} | {r['days_both_have_data']} | "
                 f"{r['both_flagged']} | {j} |")
    L.append("")
    lab = summary.get("labelled") or {}
    if lab.get("dates"):
        L.append("## The answer key's own dates")
        L.append("")
        L.append(f"{lab['n_macro']} of the {lab['n_scored']} scored dates in "
                 "`labels_by_written_rule.csv` are days the complex moved together.")
        L.append("")
        L.append("| Date | Label | Verdict | Also flagged | Cause recorded in the key |")
        L.append("|---|---|---|---|---|")
        for d in lab["dates"]:
            if d["verdict"] != "macro":
                continue
            cause = d["known_cause"]
            cause = (cause[:95] + "…") if len(cause) > 95 else cause
            L.append(f"| {d['date']} | {d['label']} | {d['verdict']} | "
                     f"{', '.join(d['others_flagged']) or '-'} | {cause} |")
        L.append("")
        n = lab["n_macro_whose_key_text_names_a_macro_driver"]
        L.append(f"On {n} of those {lab['n_macro']}, the cause the key itself records is a "
                 "macro one, in its own words. The key marks a date EXPLAIN when a "
                 "documented cause exists; it never asks whether the cause is specific to "
                 "this market. A rule built to require a report about this market will "
                 "refuse such a date and be scored wrong for doing so.")
        L.append("")
        L.append("**Nothing here relabels anything.** Changing a label after seeing which "
                 "dates a rule failed is the move this project exists to refuse. This is "
                 "evidence that the key needs a third category - a move that is real, "
                 "documented, and not about this commodity - for the people who own the "
                 "key to decide on.")
        L.append("")
    L.append("## What this does and does not establish")
    L.append("")
    L.append("- **Observed.** The counts above are read from the stored detections. "
             "No model, no network, no estimate.")
    L.append("- **Not tested.** That a macro day is *caused* by the dollar or by risk "
             "appetite is not shown here. Agreement across commodities is consistent with "
             "a shared driver; it does not name one. "
             "`src/modeling/factor_model.py` measures that directly.")
    L.append("- A commodity-specific verdict is the weaker of the two. It says only that "
             "the other four did not flag, which a shared driver too small to flag them "
             "would also produce.")
    return "\n".join(L)


def run(min_others: int = DEFAULT_MIN_OTHERS, verbose: bool = True) -> dict:
    table = flag_table()
    if table.empty:
        raise SystemExit("No detections found. Run: python -m src.modeling.anomaly_detector")
    summary = summarise(table, min_others)
    summary["labelled"] = score_labelled_dates(table, min_others)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    text = report(summary)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(text)
    if verbose:
        print(text)
        print(f"\nSaved {OUTPUT_MD}\nSaved {OUTPUT_JSON}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--min-others", type=int, default=DEFAULT_MIN_OTHERS,
                   help="how many other commodities must flag before a day is called macro")
    args = p.parse_args()
    run(min_others=args.min_others)


if __name__ == "__main__":
    main()
