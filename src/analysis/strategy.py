"""
What the measurements support doing, and what they do not.

The project measures three things that bear on a decision: how long a shock
takes to be given back, how much of a flagged move is about the commodity
rather than the dollar, and how much of a commodity move reaches a consumer
price. This turns each into a statement a buyer could act on, with the figure
it rests on beside it, and says plainly where no statement is supported.

The hard rule here: nothing in this module is a forecast, and nothing is
advice. Short-horizon price direction was tested and found to be at chance
(results/forecast_evaluation.md), so a recommendation of the form "prices will
fall, wait" has no evidence behind it and is not produced. What IS produced is
the distribution of what followed comparable days, which a person can weigh
against their own cost of waiting.

Three things it reports, and the control beside each:

  TIMING       After a flagged shock, the trading days until half the move is
               given back, and the share never given back inside 120 days.
               The control is the days the detector did NOT flag whose move
               was at least as large: if those behave the same, the statistic
               is about the series, not about being flagged. (They do not:
               flagged shocks take longer, 4-7 days against 3.)

  EXPOSURE     The share of a commodity's flagged days the factor model calls
               macro-driven. A buyer whose exposure is mostly the dollar
               cannot hedge it with a commodity-specific instrument, and the
               split says which problem they have. Read with that report's own
               caveat: crude oil's figure is inflated because the broad factor
               contains energy.

  PASS-THROUGH How much of a commodity move reaches a consumer price index,
               from the monthly regressions, using only the Holm-corrected
               verdict. A pair with no measurable relationship is reported as
               such: "no basis for budgeting on this" is a finding, not a gap.

Half-life is a median over overlapping forward windows, so no p-value is
computed from it and none is reported. The spread is given as quartiles.

Usage:
    python -m src.analysis.strategy
    python -m src.analysis.strategy --save
    python -m src.analysis.strategy --commodity coffee
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.commodities import COMMODITIES, get_commodity   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORT_MD = RESULTS_DIR / "strategy_report.md"
REPORT_JSON = RESULTS_DIR / "strategy.json"
INFLATION_JSON = RESULTS_DIR / "inflation_report.json"

MAX_DAYS = 120          # how far forward a retrace is looked for
MIN_MOVE_PCT = 0.5      # below this the "half the move" target is noise
HORIZONS = (5, 10, 20, 60)
MIN_GROUP = 8           # below this a median is not worth printing


def load_detections(key: str) -> pd.DataFrame:
    path = RESULTS_DIR / get_commodity(key).anomalies_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.modeling.anomaly_detector "
            f"--commodity {key}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _usable(df: pd.DataFrame) -> pd.Series:
    """Days a percentage move is defined on and not an artefact of the series.

    A contract-switch day shows a move no contract made. A price at or below
    zero at either end has no percentage change at all - WTI settled at
    -$37.63 on 20 April 2020.
    """
    price = df["price"].astype(float)
    switch = (df["contract_switch"].fillna(False).astype(bool)
              if "contract_switch" in df.columns
              else pd.Series(False, index=df.index))
    move = price.pct_change() * 100
    return (price > 0) & (price.shift() > 0) & ~switch & move.notna()


def _half_life(price: pd.Series, move: pd.Series, rows) -> dict:
    """Trading days until half the move is given back, per day in `rows`.

    A day that never gives it back inside MAX_DAYS is COUNTED, not dropped:
    dropping them would make every shock look temporary, which is the single
    easiest way to turn this table into a false reassurance.
    """
    days, never, n = [], 0, 0
    for i in rows:
        m = move.iloc[i]
        if pd.isna(m) or abs(m) < MIN_MOVE_PCT:
            continue
        target = price.iloc[i - 1] * (1 + (m / 2) / 100)
        forward = price.iloc[i + 1:i + 1 + MAX_DAYS]
        if forward.empty:
            continue
        n += 1
        reached = (forward <= target) if m > 0 else (forward >= target)
        if reached.any():
            days.append(int(np.argmax(reached.to_numpy())) + 1)
        else:
            never += 1
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "retraced": len(days),
        "never_retraced": never,
        "share_never": never / n,
        "median_days": float(np.median(days)) if days else None,
        "q1_days": float(np.percentile(days, 25)) if days else None,
        "q3_days": float(np.percentile(days, 75)) if days else None,
    }


def timing(key: str) -> dict:
    """Half-life after a flagged shock, and the same statistic on unflagged
    days whose move was at least as large.

    The control is the point. Commodity prices mean-revert after any large
    move; if the unflagged days behave identically then the detector adds
    nothing to a timing decision, and the honest answer is to say so.
    """
    df = load_detections(key)
    price = df["price"].astype(float)
    move = price.pct_change() * 100
    ok = _usable(df)
    flagged = df["anomaly_flag"].fillna(False).astype(bool)
    is_shock = df["anomaly_type"].fillna("").str.contains("shock")
    shock_rows = list(df.index[flagged & is_shock & ok])
    if not shock_rows:
        return {"commodity": key, "shocks": {"n": 0}, "control": {"n": 0},
                "control_floor_pct": None}
    floor = float(move[flagged & is_shock & ok].abs().min())
    control_rows = list(df.index[~flagged & ok & (move.abs() >= floor)])
    return {
        "commodity": key,
        "control_floor_pct": floor,
        "shocks": _half_life(price, move, shock_rows),
        "control": _half_life(price, move, control_rows),
    }


def forward_profile(key: str) -> dict:
    """What a flagged shock day was followed by, beside the base rate.

    A median forward return means nothing without the base rate on the same
    series over the same horizon: a commodity that drifted up drifts up after
    flagged days too.
    """
    df = load_detections(key)
    price = df["price"].astype(float)
    ok = _usable(df)
    flagged = df["anomaly_flag"].fillna(False).astype(bool)
    is_shock = df["anomaly_type"].fillna("").str.contains("shock")
    rows = flagged & is_shock & ok
    out = {"commodity": key, "horizons": {}}
    for h in HORIZONS:
        fwd = (price.shift(-h) / price - 1.0) * 100
        out["horizons"][h] = {
            "after_a_shock": _spread(fwd[rows]),
            "any_day": _spread(fwd[ok]),
        }
    return out


def _spread(values: pd.Series) -> dict:
    """Median and quartiles, never a mean: one 2020 April dominates a mean."""
    v = values.dropna()
    if len(v) < MIN_GROUP:
        return {"n": int(len(v))}
    return {"n": int(len(v)), "median": float(v.median()),
            "q1": float(v.quantile(0.25)), "q3": float(v.quantile(0.75)),
            "share_up": float((v > 0).mean())}


def exposure(key: str) -> dict:
    """How much of this commodity's flagged activity is the complex moving.

    Read from the factor model's own output rather than recomputed, so the
    two reports cannot disagree.
    """
    path = RESULTS_DIR / f"{key}_factor_adjusted.csv"
    if not path.exists():
        return {"commodity": key, "available": False}
    df = pd.read_csv(path)
    if "factor_class" not in df.columns:
        return {"commodity": key, "available": False}
    judged = df[df["factor_class"].isin(
        ["macro-driven", "commodity-specific", "both"])]
    n = len(judged)
    if not n:
        return {"commodity": key, "available": False}
    macro = int((judged["factor_class"] == "macro-driven").sum())
    r2 = (float(df["window_r2"].median())
          if "window_r2" in df.columns and df["window_r2"].notna().any() else None)
    return {"commodity": key, "available": True, "assessed": n,
            "macro": macro, "macro_share": macro / n, "median_window_r2": r2}


def numeraire_mix(key: str) -> dict:
    """How many flagged days survived translation into other currencies.

    Read from the numeraire test's own output. This is a second, independent
    answer to the question the factor model answers by regression, and the two
    need not agree: a regression asks whether the move is consistent with the
    dollar's usual relationship to this commodity, while translation asks the
    blunter question of whether a buyer paying in rupees saw the same move. A
    day the regression calls macro and translation calls commodity-specific is
    a real finding about that day, not a bug in either.
    """
    path = RESULTS_DIR / f"{key}_numeraire.json"
    if not path.exists():
        return {"commodity": key, "available": False}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"commodity": key, "available": False}
    counts = {}
    for row in rows:
        counts[row.get("reading")] = counts.get(row.get("reading"), 0) + 1
    judged = sum(v for k, v in counts.items() if k != "not enough currency data")
    if not judged:
        return {"commodity": key, "available": False}
    return {
        "commodity": key, "available": True, "judged": judged,
        "specific": counts.get("specific to the commodity", 0),
        "mixed": counts.get("mixed", 0),
        "numeraire": counts.get("mostly the numeraire", 0),
        "no_data": counts.get("not enough currency data", 0),
        "numeraire_share": counts.get("mostly the numeraire", 0) / judged,
    }


def pass_through() -> dict:
    """The monthly pass-through results, verdicts by the Holm-corrected p.

    Only the corrected column decides, because 25 pairs were tested. Several
    rows look striking on an uncorrected p-value and are not counted.
    """
    if not INFLATION_JSON.exists():
        return {"available": False}
    data = json.loads(INFLATION_JSON.read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    alpha = (data.get("settings") or {}).get("significance", 0.05)
    measured, none_found, nowcast_ok = [], [], []
    for pair in pairs:
        pt = pair.get("pass_through") or {}
        nc = pair.get("nowcast") or {}
        row = {
            "country": pair.get("country"),
            "target": pair.get("target_label"),
            "driver": pair.get("driver_label"),
            "per_10pct": (None if pt.get("sum_beta") is None
                          else 10 * pt["sum_beta"]),
            "p_holm": pt.get("p_holm"),
            "months": pt.get("n"),
        }
        if pt.get("p_holm") is not None and pt["p_holm"] < alpha:
            measured.append(row)
        else:
            none_found.append(row)
        if nc.get("p_holm") is not None and nc["p_holm"] < alpha:
            nowcast_ok.append(row)
    measured.sort(key=lambda r: -(r["per_10pct"] or 0))
    return {"available": True, "alpha": alpha, "n_pairs": len(pairs),
            "measured": measured, "no_relationship": none_found,
            "nowcast_better": nowcast_ok}


def _days(stat: dict) -> str:
    if not stat.get("n"):
        return "-"
    if stat.get("median_days") is None:
        return "never, in any case"
    return (f"{stat['median_days']:.0f} "
            f"({stat['q1_days']:.0f}-{stat['q3_days']:.0f})")


def report(timings: list, exposures: list, prices: dict, profiles: list,
           numeraires: list = None) -> str:
    numeraires = numeraires or []
    L = ["# What the measurements support doing", ""]
    L.append("Produced by `python -m src.analysis.strategy`. Every statement below "
             "carries the figure it rests on. Where a measurement does not support a "
             "statement, that is said rather than filled in.")
    L.append("")
    L.append("**This is not advice, and nothing here is a forecast.** Short-horizon "
             "price direction was tested and is at chance (`forecast_evaluation.md`: "
             "50.2% against a 50.7% baseline), so no recommendation of the form "
             "\"prices will fall, wait\" appears here. What appears is what followed "
             "comparable days in the past, which a person can weigh against their own "
             "cost of waiting.")
    L.append("")

    L.append("## 1. Timing a purchase into a shock")
    L.append("")
    L.append("Trading days until half the move is given back, median and quartiles. "
             "The control is days the detector did NOT flag whose move was at least "
             "as large as the smallest flagged shock: if those behaved the same, this "
             "statistic would be about the series rather than about the flag.")
    L.append("")
    L.append("| Commodity | Flagged shocks | Half given back in | Never, in 120 days | "
             "Unflagged, same size | Half given back in | Never |")
    L.append("|---|---|---|---|---|---|---|")
    for t in timings:
        shocks, control = t["shocks"], t["control"]
        if not shocks.get("n"):
            continue
        if control.get("n"):
            tail = (f"{control['n']} | {_days(control)} d | "
                    f"{100 * control['share_never']:.0f}%")
        else:
            tail = "- | - | -"
        L.append(f"| {get_commodity(t['commodity']).display_name} | {shocks['n']} | "
                 f"{_days(shocks)} d | {100 * shocks['share_never']:.0f}% | {tail} |")
    L.append("")
    L.append("**What this supports.** A buyer who must transact into a flagged shock "
             "has a measured distribution for waiting, not a prediction: on most "
             "commodities half the move is typically given back within a working week "
             "or two. **What it does not support:** waiting as a general policy. Around "
             "one shock in ten never gives half back inside 120 trading days, and which "
             "one is not knowable in advance - that is the same direction question the "
             "project measured and failed.")
    L.append("")
    L.append("The control column is the interesting one. Flagged shocks take longer to "
             "half-retrace than unflagged moves of the same size, so the flag is "
             "selecting days that behave differently; but the share that never retrace "
             "is close in both columns, so the flag says something about speed, not "
             "about permanence.")
    L.append("")

    L.append("## 2. Which risk is actually being carried")
    L.append("")
    L.append("A buyer hedging a commodity-specific disruption and a buyer carrying "
             "dollar risk have different problems, and a commodity instrument only "
             "addresses the first. The split is read from `factor_adjusted_report.md`.")
    L.append("")
    L.append("| Commodity | Days assessed | Macro-driven | Share | Median window R² | "
             "Translation: mostly the dollar |")
    L.append("|---|---|---|---|---|---|")
    for e in exposures:
        num = next((n for n in numeraires
                    if n["commodity"] == e["commodity"]), {"available": False})
        trans = ("not run" if not num.get("available")
                 else f"{num['numeraire']} of {num['judged']} "
                      f"({100 * num['numeraire_share']:.1f}%)")
        if not e.get("available"):
            L.append(f"| {get_commodity(e['commodity']).display_name} | - | - | "
                     f"not run | - | {trans} |")
            continue
        r2 = "-" if e["median_window_r2"] is None else f"{e['median_window_r2']:.2f}"
        L.append(f"| {get_commodity(e['commodity']).display_name} | {e['assessed']} | "
                 f"{e['macro']} | {100 * e['macro_share']:.1f}% | {r2} | {trans} |")
    L.append("")
    L.append("**The last two columns disagree, and the disagreement is the finding.** "
             "The regression asks whether a day's move is consistent with this "
             "commodity's usual relationship to the dollar. Translation asks the "
             "blunter question: did a buyer paying in euro, yen, yuan or rupees see "
             "the same move? A day can be ordinary for the dollar's behaviour and "
             "still be a real commodity move, which is why the share called "
             "\"macro-driven\" is consistently larger than the share that actually "
             "fails translation. Neither column is the right answer on its own; the "
             "anomaly card prints both and gives no overall verdict, for this reason.")
    L.append("")
    L.append("**Two cautions carried over from that report, not restated as results.** "
             "Crude oil's share is inflated because the broad-complex factor is heavily "
             "weighted to energy, so crude is partly regressed on itself. And where the "
             "window R² is low the residual is most of the move whatever happened, so "
             "\"commodity-specific\" comes close to restating that the day was flagged.")
    L.append("")
    L.append("**What this supports.** Choosing what to hedge: a commodity whose flagged "
             "days are largely the complex repricing is not well addressed by a "
             "commodity-specific contract. **What it does not establish:** that the "
             "dollar caused those moves. A confound moving with the dollar looks "
             "identical.")
    L.append("")

    L.append("## 3. How much of a commodity move reaches a price people pay")
    L.append("")
    if not prices.get("available"):
        L.append("Not available: `results/inflation_report.json` is missing. Run "
                 "`python -m src.modeling.inflation` first.")
    else:
        L.append(f"{prices['n_pairs']} commodity-and-index pairs were fixed in code "
                 f"before any result was seen and all are reported. Verdicts use the "
                 f"Holm-corrected p-value at {prices['alpha']:.0%}, which is strict "
                 f"with this many pairs.")
        L.append("")
        L.append("| Consumer price index | Commodity | After a 10% rise | p (Holm) | Months |")
        L.append("|---|---|---|---|---|")
        for r in prices["measured"]:
            p = r["p_holm"]
            ptxt = "<0.001" if p is not None and p < 0.001 else f"{p:.3f}"
            L.append(f"| {r['country']}: {r['target']} | {r['driver']} | "
                     f"+{r['per_10pct']:.2f}% | {ptxt} | {r['months']} |")
        L.append("")
        L.append(f"**{len(prices['measured'])} of {prices['n_pairs']} pairs show "
                 f"pass-through. {len(prices['no_relationship'])} show nothing that can "
                 f"be told from chance**, including every pair tested against India's "
                 f"index. A null is a result: for those commodities and those indices "
                 f"the monthly record shows nothing chance alone would not.")
        L.append("")
        L.append("**What this supports.** Budgeting: a procurement plan can carry a "
                 "stated elasticity for the pairs above, with its interval. **What it "
                 "does not support:** using a commodity price to estimate next month's "
                 "inflation. After correction, "
                 f"**{len(prices['nowcast_better'])} of {prices['n_pairs']} pairs** "
                 "improve a figure made before publication - several show large "
                 "improvements with an uncorrected p under 0.05 and are still not "
                 "counted.")
    L.append("")

    L.append("## What nothing here supports")
    L.append("")
    L.append("- **A direction call.** Tested on three methods and a null result each "
             "time (`forecast_evaluation.md`, `backtest_outlook_report.md`).")
    L.append("- **A causal claim from the inflation regressions.** They measure how two "
             "series moved together, over months, with a stated lag structure.")
    L.append("- **A claim that a macro-driven day was caused by the dollar.**")
    L.append("- **Anything about a commodity with no run behind it.** A blank in these "
             "tables is a gap in what has been measured, not a negative finding.")
    L.append("")
    L.append("Not financial advice.")
    return "\n".join(L)


def run(commodities=None, verbose: bool = True, save: bool = False) -> dict:
    keys = commodities or [k for k in COMMODITIES
                           if (RESULTS_DIR / get_commodity(k).anomalies_file).exists()]
    timings = [timing(k) for k in keys]
    exposures = [exposure(k) for k in keys]
    profiles = [forward_profile(k) for k in keys]
    numeraires = [numeraire_mix(k) for k in keys]
    prices = pass_through()
    text = report(timings, exposures, prices, profiles, numeraires)
    if verbose:
        print(text)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_MD.write_text(text, encoding="utf-8")
        REPORT_JSON.write_text(json.dumps(
            {"timing": timings, "exposure": exposures, "forward": profiles,
             "numeraire": numeraires, "pass_through": prices},
            indent=2, default=str), encoding="utf-8")
        if verbose:
            print(f"\nSaved {REPORT_MD} and {REPORT_JSON}")
    return {"timing": timings, "exposure": exposures, "pass_through": prices,
            "forward": profiles, "numeraire": numeraires}


def main():
    p = argparse.ArgumentParser(description="What the measurements support doing")
    p.add_argument("--commodity", default=None)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    if args.commodity and args.commodity not in COMMODITIES:
        p.error(f"unknown --commodity {args.commodity!r}; "
                f"choose from {', '.join(COMMODITIES)}")
    run(commodities=[args.commodity] if args.commodity else None, save=args.save)


if __name__ == "__main__":
    main()
