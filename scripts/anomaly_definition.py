"""
What counts as an anomaly, and what the current definition costs.

The detector flags a day when its move is unusual against that commodity's own
recent volatility. That is one clause of a definition, not the whole of one.
This module reports the three things the single clause leaves out, each
measured rather than argued, and changes NOTHING about the detector: every
evaluation figure in results/ was drawn at the current rule, so altering which
days are flagged would invalidate the answer key. These columns sit beside it,
as the factor model and the cross-commodity check do.

  1. EPISODES, NOT DAYS. The 10-day cumulative check tags every day inside a
     build-up, so one event becomes a run of flagged days. "12.7% of days" and
     "about one episode a month" describe the same detector; only the second
     is a fair summary. Reported with the shock / trend / both split, because
     the union of two checks reported as one rate hides which fired.

  2. WHAT THE BAR COSTS IN MONEY. A z-score is a statistical bar, not an
     economic one, and the same bar is a different event in each market: on
     this data the smallest flagged gold move is about 1%, the smallest
     flagged coffee move about 2.3%. A materiality clause - large enough that
     someone transacting would act - is set per commodity, as a multiple of
     that commodity's own median daily move, so it means the same thing to a
     buyer everywhere rather than the same thing to a statistician.

  3. WHETHER A FLAGGED DAY CHANGES WHAT FOLLOWS. The label-free test: if a
     flagged day marks a real disruption, the 20 trading days after it should
     be more volatile than the 20 days after an ordinary day. This needs no
     event list, no news, no answer key and no threshold argument. It is the
     one check here that no amount of selection can flatter.

Also reported: the empirically calibrated thresholds. Under a normal
distribution |z| >= 2 would fire on 4.55% of days. It does not, because
returns are fat-tailed, so "2 sigma" carries no false-positive rate and
quoting one is wrong. The table gives the z that fires on 5%, 2% and 1% of
days on THIS data, which is a rate that can be quoted.

What this module cannot do: estimate precision. Most flagged days sit outside
every cited event window and carry no label, so "how many of these are false
alarms" has no answer here. `--sample` writes a random sample of flagged days
to a CSV for a person to label, which is the only thing that would answer it.

Usage:
    python -m scripts.anomaly_definition
    python -m scripts.anomaly_definition --save
    python -m scripts.anomaly_definition --sample 60 --seed 7
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.commodities import COMMODITIES, get_commodity   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORT_MD = RESULTS_DIR / "anomaly_definition.md"
SAMPLE_CSV = RESULTS_DIR / "flagged_day_sample_to_label.csv"

EPISODE_GAP = 3          # clear trading days that end an episode
MATERIALITY_MULTIPLE = 3.0   # of the commodity's own median daily move
FORWARD_WINDOW = 20      # trading days after a flag, for the regime test
TARGET_RATES = (0.05, 0.02, 0.01)
NORMAL_TWO_SIGMA = 4.55  # per cent of days beyond 2 sigma under a normal law


def load(key: str) -> pd.DataFrame:
    path = RESULTS_DIR / get_commodity(key).anomalies_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.modeling.anomaly_detector "
            f"--commodity {key}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def usable(df: pd.DataFrame) -> pd.Series:
    """Days a percentage move is defined on and not an artefact of the series."""
    price = df["price"].astype(float)
    switch = (df["contract_switch"].fillna(False).astype(bool)
              if "contract_switch" in df.columns
              else pd.Series(False, index=df.index))
    move = price.pct_change() * 100
    return ((price > 0) & (price.shift() > 0) & ~switch
            & move.notna() & df["z_score"].notna())


def episodes(flagged: pd.Series, gap: int = EPISODE_GAP) -> list:
    """Runs of flagged days, merged across gaps shorter than `gap`.

    A build-up produces consecutive flags for one event. Counting days instead
    of episodes multiplies a single disruption by the length of the window
    that detected it.
    """
    rows = np.flatnonzero(flagged.to_numpy())
    if not len(rows):
        return []
    out, start, prev = [], rows[0], rows[0]
    for i in rows[1:]:
        if i - prev > gap:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


def calibration(df: pd.DataFrame) -> dict:
    """The z that fires on a stated share of days, read off this series.

    Under a normal distribution 2 sigma is 4.55% of days. The measured share
    is what matters, and the gap between them is the fat tail.
    """
    z = df["z_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return {}
    # Kurtosis is taken on the SIGNED z-score. On absolute values it measures
    # the shape of a folded distribution and is not comparable with the
    # textbook figure of 0 for a normal law.
    out = {"measured_at_2": float((z.abs() >= 2).mean()),
           "normal_at_2": NORMAL_TWO_SIGMA / 100.0,
           "excess_kurtosis": float(z.kurtosis())}
    for rate in TARGET_RATES:
        out[f"z_for_{rate:g}"] = float(np.quantile(z.abs(), 1 - rate))
    return out


def definitions(key: str) -> dict:
    """The current flag set beside two definitions that do not use a z-score."""
    df = load(key)
    ok = usable(df)
    price = df["price"].astype(float)
    move = (price.pct_change() * 100)
    flagged = df["anomaly_flag"].fillna(False).astype(bool) & ok
    kind = df["anomaly_type"].fillna("")
    daily = (df["z_score"].abs() >= 2) & ok

    median_move = float(move[ok].abs().median())
    bar = MATERIALITY_MULTIPLE * median_move
    material = (move.abs() >= bar) & ok
    both = daily & material

    eps = episodes(flagged)
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    return {
        "commodity": key,
        "scorable_days": int(ok.sum()),
        "flagged_days": int(flagged.sum()),
        "shock_only": int((kind == "shock").sum()),
        "trend_only": int((kind == "trend").sum()),
        "both_checks": int((kind == "shock+trend").sum()),
        "episodes": len(eps),
        "episodes_per_year": len(eps) / years if years else float("nan"),
        "median_episode_days": (float(np.median([b - a + 1 for a, b in eps]))
                                if eps else None),
        "median_daily_move_pct": median_move,
        "materiality_bar_pct": bar,
        # Over SHOCK days only. A trend-only day is flagged on its ten-day
        # cumulative move and its own daily move is often near zero, so the
        # smallest move across all flagged days is 0.00% and says nothing.
        "median_shock_move_pct": float(move[daily].abs().median()) if daily.any() else None,
        "smallest_shock_move_pct": float(move[daily].abs().min()) if daily.any() else None,
        "daily_check_days": int(daily.sum()),
        "material_days": int(material.sum()),
        "both_days": int(both.sum()),
        "material_not_flagged": int((material & ~flagged).sum()),
        "flagged_not_material": int((flagged & ~material).sum()),
        "jaccard_daily_material": (int((daily & material).sum()) / int((daily | material).sum())
                                   if int((daily | material).sum()) else None),
        "calibration": calibration(df),
    }


def event_recall(key: str) -> dict:
    """Cited events caught by each definition. A definition that misses
    documented events is not stricter, it is worse."""
    cfg = get_commodity(key)
    df = load(key)
    ok = usable(df)
    price = df["price"].astype(float)
    move = price.pct_change() * 100
    flagged = df["anomaly_flag"].fillna(False).astype(bool) & ok
    daily = (df["z_score"].abs() >= 2) & ok
    bar = MATERIALITY_MULTIPLE * float(move[ok].abs().median())
    material = (move.abs() >= bar) & ok
    counts = {"events": 0, "flagged": 0, "daily": 0, "material": 0, "both": 0}
    for event in cfg.known_events:
        window = (df["date"] >= event[1]) & (df["date"] <= event[2])
        if not window.any():
            continue
        counts["events"] += 1
        counts["flagged"] += bool(flagged[window].any())
        counts["daily"] += bool(daily[window].any())
        counts["material"] += bool(material[window].any())
        counts["both"] += bool((daily & material)[window].any())
    return {"commodity": key, **counts}


def regime_test(key: str, window: int = FORWARD_WINDOW) -> dict:
    """Does a flagged day change what follows? The label-free check.

    Median realised volatility of the `window` trading days AFTER a flagged
    day, against the same after any day, and against the same after a large
    move the detector did NOT flag. The third comparison is the demanding one:
    if a plain large move predicts a turbulent month better than the flag
    does, the flag is not identifying the days that matter.
    """
    df = load(key)
    ok = usable(df)
    price = df["price"].astype(float)
    log_ret = np.log(price.where(price > 0)).diff()
    forward = log_ret.shift(-1).rolling(window).std().shift(-(window - 1))
    flagged = df["anomaly_flag"].fillna(False).astype(bool) & ok
    move = (price.pct_change() * 100).abs()
    bar = MATERIALITY_MULTIPLE * float(move[ok].median())
    big_unflagged = (move >= bar) & ok & ~flagged

    def med(mask):
        values = forward[mask].dropna()
        return float(values.median()) if len(values) else None

    after_flag, baseline, after_big = med(flagged), med(ok), med(big_unflagged)
    return {
        "commodity": key, "window": window,
        "after_a_flag": after_flag, "after_any_day": baseline,
        "after_a_large_unflagged_move": after_big,
        "lift_over_baseline": (after_flag / baseline) if after_flag and baseline else None,
        "lift_over_large_move": (after_flag / after_big) if after_flag and after_big else None,
        "n_large_unflagged": int(big_unflagged.sum()),
    }


def sample_for_labelling(keys, n: int, seed: int) -> list:
    """A random sample of flagged days for a person to judge.

    Precision cannot be computed here: most flagged days carry no label. This
    writes the sampling frame that would let somebody produce one. Sampling is
    random within commodity rather than weighted by size, because weighting
    towards large moves is exactly how a precision figure gets flattered.
    """
    rng = random.Random(seed)
    rows = []
    per = max(1, n // max(1, len(keys)))
    for key in keys:
        df = load(key)
        ok = usable(df)
        price = df["price"].astype(float)
        move = price.pct_change() * 100
        flagged = df["anomaly_flag"].fillna(False).astype(bool) & ok
        idx = list(df.index[flagged])
        rng.shuffle(idx)
        for i in idx[:per]:
            rows.append({
                "commodity": key,
                "date": df.loc[i, "date"].date().isoformat(),
                "pct_move": round(float(move.iloc[i]), 2),
                "z_score": round(float(df.loc[i, "z_score"]), 2),
                "anomaly_type": df.loc[i, "anomaly_type"],
                "verdict_real_or_noise": "",
                "cause_if_known": "",
                "source_url": "",
                "labelled_by": "",
            })
    return rows


def report(defs: list, recalls: list, regimes: list) -> str:
    L = ["# What counts as an anomaly, and what the current definition costs", ""]
    L.append("Produced by `python -m scripts.anomaly_definition`. Nothing here changes "
             "the detector: every evaluation figure in `results/` was drawn at the "
             "current rule, so these columns sit beside it rather than replacing it.")
    L.append("")

    L.append("## 1. Episodes, not days")
    L.append("")
    L.append("The 10-day cumulative check tags every day inside a build-up, so one "
             "event becomes a run. Both columns describe the same detector; only the "
             "second is a fair summary of how often it speaks.")
    L.append("")
    L.append("| Commodity | Scorable days | Flagged days | Share | Shock only | "
             "Trend only | Both checks | Episodes | Per year |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in defs:
        share = 100 * d["flagged_days"] / d["scorable_days"] if d["scorable_days"] else 0
        L.append(f"| {get_commodity(d['commodity']).display_name} | {d['scorable_days']} | "
                 f"{d['flagged_days']} | {share:.1f}% | {d['shock_only']} | "
                 f"{d['trend_only']} | {d['both_checks']} | {d['episodes']} | "
                 f"{d['episodes_per_year']:.1f} |")
    L.append("")
    L.append("Say \"about one episode a month\", not \"one day in eight\". The second is "
             "true and misleading: it counts a ten-day build-up as ten anomalies.")
    L.append("")

    L.append("## 2. The bar is statistical, not economic")
    L.append("")
    L.append("A z-score asks whether a move is unusual for this market lately. It does "
             "not ask whether anyone would act on it, and the same bar is a different "
             "event in each market.")
    L.append("")
    L.append("| Commodity | Median daily move | Median move on a SHOCK day | "
             "Smallest shock move | "
             f"Materiality bar ({MATERIALITY_MULTIPLE:g}x median) |")
    L.append("|---|---|---|---|---|")
    for d in defs:
        L.append(f"| {get_commodity(d['commodity']).display_name} | "
                 f"{d['median_daily_move_pct']:.2f}% | "
                 f"{d['median_shock_move_pct']:.2f}% | "
                 f"{d['smallest_shock_move_pct']:.2f}% | "
                 f"{d['materiality_bar_pct']:.2f}% |")
    L.append("")
    L.append("The move columns are over days the DAILY check fired. A trend-only day is "
             "flagged on its ten-day cumulative move and its own daily move is often "
             "near zero, so a move statistic over all flagged days would be "
             "meaningless. That is the same conflation the episode table is about.")
    L.append("")
    L.append("| Commodity | Daily check | Material | Both | Material, not flagged | "
             "Flagged, not material | Overlap (Jaccard) |")
    L.append("|---|---|---|---|---|---|---|")
    for d in defs:
        j = "-" if d["jaccard_daily_material"] is None else f"{d['jaccard_daily_material']:.2f}"
        L.append(f"| {get_commodity(d['commodity']).display_name} | {d['daily_check_days']} | "
                 f"{d['material_days']} | {d['both_days']} | {d['material_not_flagged']} | "
                 f"{d['flagged_not_material']} | {j} |")
    L.append("")
    L.append("These are largely different sets of days. Requiring BOTH clauses - "
             "statistically unusual and large enough to act on - is the narrower "
             "definition, and the next table says what it costs in cited events.")
    L.append("")

    L.append("### Cited events caught, by definition")
    L.append("")
    L.append("| Commodity | Events | Current detector | Daily check only | "
             "Material only | Both |")
    L.append("|---|---|---|---|---|---|")
    for r in recalls:
        L.append(f"| {get_commodity(r['commodity']).display_name} | {r['events']} | "
                 f"{r['flagged']} | {r['daily']} | {r['material']} | {r['both']} |")
    L.append("")

    L.append("### Thresholds calibrated to this data, not to a normal distribution")
    L.append("")
    L.append(f"Under a normal law |z| >= 2 would fire on {NORMAL_TWO_SIGMA}% of days. It "
             f"does not, because returns are fat-tailed. Quote the measured column, "
             f"never the {NORMAL_TWO_SIGMA}%, and if a stated firing rate is wanted, "
             f"read the z off this table.")
    L.append("")
    L.append("| Commodity | Measured at \\|z\\|>=2 | Normal law says | z for 5% | "
             "z for 2% | z for 1% | Excess kurtosis |")
    L.append("|---|---|---|---|---|---|---|")
    for d in defs:
        c = d["calibration"]
        if not c:
            continue
        L.append(f"| {get_commodity(d['commodity']).display_name} | "
                 f"{100 * c['measured_at_2']:.2f}% | {NORMAL_TWO_SIGMA}% | "
                 f"{c['z_for_0.05']:.2f} | {c['z_for_0.02']:.2f} | "
                 f"{c['z_for_0.01']:.2f} | {c['excess_kurtosis']:.1f} |")
    L.append("")

    L.append("## 3. Does a flagged day change what follows?")
    L.append("")
    L.append(f"The label-free test. Median realised volatility of the {FORWARD_WINDOW} "
             f"trading days after a flagged day, against the same after any day, and "
             f"against the same after a large move the detector did NOT flag. No event "
             f"list, no news, no answer key, nothing selected by hand.")
    L.append("")
    L.append("| Commodity | After a flag | After any day | Lift | "
             "After a large unflagged move | Lift over that |")
    L.append("|---|---|---|---|---|---|")
    for g in regimes:
        lb = "-" if g["lift_over_baseline"] is None else f"{g['lift_over_baseline']:.2f}x"
        lm = "-" if g["lift_over_large_move"] is None else f"{g['lift_over_large_move']:.2f}x"
        L.append(f"| {get_commodity(g['commodity']).display_name} | "
                 f"{g['after_a_flag']:.5f} | {g['after_any_day']:.5f} | {lb} | "
                 f"{g['after_a_large_unflagged_move']:.5f} | {lm} |")
    L.append("")
    L.append("**Read the last column carefully.** A lift above 1 over the baseline says "
             "a flagged day is followed by a more turbulent month than an ordinary day. "
             "A lift below 1 in the final column says a plain large move predicts that "
             "turbulence BETTER than the flag does - which, where it happens, is "
             "evidence that the statistical bar is picking days that are unusual "
             "without being consequential. This is the strongest argument in the "
             "project for adding the materiality clause, and it is also the argument "
             "most likely to be made by an examiner, so it belongs in the report.")
    L.append("")

    L.append("## What this cannot tell you")
    L.append("")
    L.append("**Precision.** Most flagged days sit outside every cited event window and "
             "carry no label, so the share of flags that are nothing has no answer "
             "here and is not estimated. `--sample` writes a random sample of flagged "
             "days to a CSV for a person to judge; labelling it is the only thing that "
             "would produce a precision figure. Sampling is random within commodity, "
             "not weighted towards large moves, because weighting is how such a figure "
             "gets flattered.")
    L.append("")
    L.append("**Whether these definitions are better.** The materiality multiple here "
             "is a stated choice, not a fitted one. It was not tuned against the cited "
             "events, and the event-recall table shows what it costs rather than being "
             "adjusted until that cost disappeared.")
    return "\n".join(L)


def run(commodities=None, save: bool = False, verbose: bool = True,
        sample: int = 0, seed: int = 7) -> dict:
    keys = commodities or [k for k in COMMODITIES
                           if (RESULTS_DIR / get_commodity(k).anomalies_file).exists()]
    if not keys:
        raise SystemExit("no detections file in results/. Run the detector first.")
    defs = [definitions(k) for k in keys]
    recalls = [event_recall(k) for k in keys]
    regimes = [regime_test(k) for k in keys]
    text = report(defs, recalls, regimes)
    if verbose:
        print(text)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_MD.write_text(text, encoding="utf-8")
        if verbose:
            print(f"\nSaved {REPORT_MD}")
    if sample:
        rows = sample_for_labelling(keys, sample, seed)
        with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"Saved {SAMPLE_CSV}: {len(rows)} flagged days to label. "
                  f"Fill verdict_real_or_noise for each, with a source.")
    return {"definitions": defs, "event_recall": recalls, "regime": regimes}


def main():
    p = argparse.ArgumentParser(description="What counts as an anomaly")
    p.add_argument("--commodity", default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--sample", type=int, default=0,
                   help="write this many flagged days to a CSV for a person to label")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.commodity and args.commodity not in COMMODITIES:
        p.error(f"unknown --commodity {args.commodity!r}")
    run(commodities=[args.commodity] if args.commodity else None,
        save=args.save, sample=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
