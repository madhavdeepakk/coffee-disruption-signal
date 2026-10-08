"""
Volatility-adjusted anomaly detector - generalized to any commodity
configured in src/config/commodities.py (originally coffee-only; extended
to crude oil and wheat per the original proposal's "Build One Deeply,
Architect for Any" idea, S4). The detection MATH is identical across
commodities (it's price-series-agnostic by design); only the input file,
output paths, and validation events/control period change per commodity.

Design (per the revised proposal, section 3.2): NOT a fixed percentage
threshold ("flag anything over 15%"). Instead: for each day, compare the
day-over-day price change against a rolling window of recent volatility
(trailing 60-90 days), using a z-score. This means the same absolute move
is flagged differently depending on how turbulent the market has recently
been - a 10% move during a calm period is more anomalous than a 20% move
during an already-volatile stretch, and a fixed threshold can't tell the
difference.

A second check was added after wheat data exposed a limitation: the daily
z-score is a shock detector - it cannot see a move that builds up gradually
over several days without any single day standing out (wheat's 2012 US
drought ran the price up +12% over 5 trading days, none of which
individually cleared the z-score threshold). The second, independent check
is on the N-trading-day cumulative return (CUMULATIVE_WINDOW_DAYS=10),
z-scored against a rolling baseline of its own history - the same
adaptive-to-volatility principle, not a fixed % threshold. A day is flagged
if either check trips (anomaly_flag = daily_anomaly_flag OR
cumulative_anomaly_flag); anomaly_type records which one ("shock", "trend",
or "shock+trend"). CUMULATIVE_WINDOW_DAYS=10 was chosen for a generic
reason (the rough timescale a drought/export-ban/embargo's price impact
takes to build) before re-testing against any event, not fitted to the
wheat case, per the team plan's ban on tuning against test cases.

Output schema (extended from the team plan's original spec - date, price,
expected_value, rolling_std, z_score, anomaly_flag - to expose the new
check without breaking existing readers of those original columns):
    date, price, expected_value, rolling_std, z_score, daily_anomaly_flag,
    cumulative_return, cumulative_expected_value, cumulative_rolling_std,
    cumulative_z_score, cumulative_anomaly_flag, anomaly_flag, anomaly_type

Validation: cross-checked against cited historical events per commodity
(src/config/commodities.py's known_events) plus a control period. A missed
event or a false positive in a quiet control period is treated as a real
problem to fix. Coffee reaches 4/4 events detected with 0 false positives
in its control period; crude oil and wheat are held to the same standard.
The cumulative check was checked against real fetched data for all three to
confirm it does not create new control-period false positives.

Usage:
    python -m src.modeling.anomaly_detector                       # coffee (default)
    python -m src.modeling.anomaly_detector --commodity crude_oil
    python -m src.modeling.anomaly_detector --commodity wheat
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: don't require a display to save a PNG
import matplotlib.pyplot as plt
import pandas as pd

from src.config.commodities import COMMODITIES, get_commodity, load_contract_switches

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ROLLING_WINDOW_DAYS = 75  # within the 60-90 day range specified in the design
Z_SCORE_THRESHOLD = 2.0  # starting point, tunable - not tuned against any
                          # commodity's known events; that would be fitting the
                          # threshold to the test case, which the team plan's
                          # fallback rules forbid

# Secondary check, added after wheat data showed a missed case: the 2012 US
# drought never cleared a single-day z-score (max 1.83) even though the
# price ran up +12% over 5 trading days (884.5 on 2012-07-13 to 943.25 on
# 2012-07-20) then partly reversed, because wheat's trailing volatility
# baseline was already elevated going into that stretch. A day-over-day
# z-score is, by construction, a shock detector - it can't see a move that
# builds gradually over multiple days without any single day standing out.
# This adds a second, independent check: the N-trading-day cumulative
# return, z-scored against a rolling baseline of its own recent history
# (same adaptive-to-volatility principle as the daily check, not a fixed %
# threshold) - an anomaly can trip either check.
#
# CUMULATIVE_WINDOW_DAYS=10 (two trading weeks) was chosen before testing
# against any known event, for a generic reason: it's roughly the timescale
# over which a drought, export ban, or embargo's price impact typically
# builds, not fitted to the 2012 wheat case. Same threshold and same
# no-lookahead (shift(1)) discipline as the daily check.
CUMULATIVE_WINDOW_DAYS = 10


# Contract-switch days. The price series is a "front month" series: it follows
# the futures contract nearest to delivery and, when that contract expires,
# jumps to the next one. The two contracts trade at different prices, so on
# the switch day the series shows a move that no contract made. Six of the
# hundred sampled coffee anomalies were of this kind (the contract in active
# trading moved 0.8-4.1% on days the series shows 3.7-8.0%).
#
# data/labeling/<contract_switch_file> (read by load_contract_switches in
# src/config/commodities.py) lists the days that were checked
# against a market report of the day, with the move the active contract was
# reported to have made. On those days the detector uses the reported move
# instead of the jump in the series, both for the day itself and inside the
# rolling baselines that later days are compared against.
#
# This is a list of checked days, not a rule: only dates in the evaluation
# sample were checked. A rule would need prices per contract, which the price
# source does not give. A window around expiry would not do either - the
# July 2021 frost days fall inside one and were real.
def adjusted_prices(df: pd.DataFrame, reported_returns: dict) -> pd.Series:
    """The price series with each listed day's jump replaced by the reported
    move. Prices before a listed day are rescaled so that the step into that
    day equals the reported move; every other day-to-day change is left as it
    was. `df` must be sorted by date with a datetime `date` column."""
    price = df["price"].astype(float).copy()
    for date in sorted(reported_returns):
        at = df.index[df["date"] == pd.Timestamp(date)]
        if len(at) == 0 or at[0] == 0:
            continue
        i = int(at[0])
        jump = price.iloc[i] / price.iloc[i - 1]
        price.iloc[:i] *= jump / (1.0 + reported_returns[date])
    return price


def compute_anomalies(df: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS,
                       z_threshold: float = Z_SCORE_THRESHOLD,
                       cumulative_window: int = CUMULATIVE_WINDOW_DAYS,
                       reported_returns: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Moves are measured on `basis`: the price series itself, or, when
    # contract-switch days are given, the series with those jumps replaced by
    # the reported moves. The `price` column that is written out stays the
    # quoted price either way.
    reported_returns = reported_returns or {}
    basis = adjusted_prices(df, reported_returns) if reported_returns else df["price"]
    df["contract_switch"] = df["date"].dt.strftime("%Y-%m-%d").isin(set(reported_returns))

    # Day-over-day % return, not raw price - volatility of returns is the
    # standard way to measure "how turbulent has this been," not volatility
    # of price level (which trends with the price itself and isn't
    # comparable across different price regimes or commodities).
    df["return_pct"] = basis.pct_change()

    # Rolling mean/std of returns, computed only from data BEFORE today
    # (shift(1)) - using today's own return in its own baseline would be
    # circular and bias the z-score toward never flagging today as unusual.
    df["expected_value"] = df["return_pct"].rolling(window).mean().shift(1)
    df["rolling_std"] = df["return_pct"].rolling(window).std().shift(1)

    df["z_score"] = (df["return_pct"] - df["expected_value"]) / df["rolling_std"]
    df["daily_anomaly_flag"] = df["z_score"].abs() >= z_threshold

    # First `window` rows have no valid rolling stats (insufficient
    # history) - explicitly NOT flagged as anomalies (that would be an
    # artifact of missing data, not a real signal), and NaN z-scores are
    # left as NaN rather than coerced to a fake 0.
    df.loc[df["rolling_std"].isna(), "daily_anomaly_flag"] = False

    # --- Secondary check: N-day cumulative return, same adaptive-z-score
    # design, computed independently of the daily check above. pct_change
    # with periods=N gives the compounded return from N trading days ago
    # to today - exactly the "slow build-up" signal the daily check misses.
    df["cumulative_return"] = basis.pct_change(periods=cumulative_window)
    df["cumulative_expected_value"] = (
        df["cumulative_return"].rolling(window).mean().shift(1)
    )
    df["cumulative_rolling_std"] = (
        df["cumulative_return"].rolling(window).std().shift(1)
    )
    df["cumulative_z_score"] = (
        (df["cumulative_return"] - df["cumulative_expected_value"])
        / df["cumulative_rolling_std"]
    )
    df["cumulative_anomaly_flag"] = df["cumulative_z_score"].abs() >= z_threshold
    df.loc[df["cumulative_rolling_std"].isna(), "cumulative_anomaly_flag"] = False

    # Combined flag: either check catching it is enough - a real anomaly
    # doesn't have to announce itself in both a single day AND a 10-day
    # window to be real. anomaly_type keeps the two reasons distinguishable
    # for anyone inspecting *why* a given day was flagged.
    df["anomaly_flag"] = df["daily_anomaly_flag"] | df["cumulative_anomaly_flag"]
    df["anomaly_type"] = ""
    df.loc[df["daily_anomaly_flag"] & ~df["cumulative_anomaly_flag"], "anomaly_type"] = "shock"
    df.loc[~df["daily_anomaly_flag"] & df["cumulative_anomaly_flag"], "anomaly_type"] = "trend"
    df.loc[df["daily_anomaly_flag"] & df["cumulative_anomaly_flag"], "anomaly_type"] = "shock+trend"

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "price", "expected_value", "rolling_std", "z_score",
               "daily_anomaly_flag", "cumulative_return", "cumulative_expected_value",
               "cumulative_rolling_std", "cumulative_z_score", "cumulative_anomaly_flag",
               "anomaly_flag", "anomaly_type", "contract_switch"]]


def validate_against_all_events(df: pd.DataFrame, known_events: list, control_period: tuple) -> list:
    """
    Full validation: all known_events (should be flagged) plus control_period
    (should stay quiet). Returns a list of result dicts, printed by main()
    and also the basis for the commodity's validation markdown file.
    """
    results = []
    for label, start, end, source in known_events:
        window_df = df[(df["date"] >= start) & (df["date"] <= end)]
        flagged = window_df[window_df["anomaly_flag"]]
        results.append({
            "type": "event",
            "label": label,
            "source": source,
            "window": f"{start} to {end}",
            "trading_days": len(window_df),
            "flagged_count": len(flagged),
            "flagged_dates": flagged["date"].tolist(),
            "flagged_types": flagged["anomaly_type"].tolist(),
            "detected": len(flagged) > 0,
        })

    start, end, note = control_period
    window_df = df[(df["date"] >= start) & (df["date"] <= end)]
    flagged = window_df[window_df["anomaly_flag"]]
    results.append({
        "type": "control",
        "label": f"Control period ({start} to {end}): {note}",
        "source": "-",
        "window": f"{start} to {end}",
        "trading_days": len(window_df),
        "flagged_count": len(flagged),
        "flagged_dates": flagged["date"].tolist(),
        "flagged_types": flagged["anomaly_type"].tolist(),
        "detected": len(flagged) == 0,  # for a control, "detected" means correctly quiet
    })
    return results


def make_plot(df: pd.DataFrame, display_name: str, known_events: list, output_path: Path):
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["date"], plot_df["price"], color="#2563eb", linewidth=1,
            label=f"{display_name} price")

    # Distinguish the two ways a day can be flagged: a single-day shock
    # (daily z-score check) vs. a multi-day trend (cumulative-return check)
    # vs. both agreeing - different markers so it's visible at a glance
    # which detector caught which anomaly, not just that "something" was
    # flagged.
    shocks = plot_df[plot_df["anomaly_type"] == "shock"]
    trends = plot_df[plot_df["anomaly_type"] == "trend"]
    both = plot_df[plot_df["anomaly_type"] == "shock+trend"]
    ax.scatter(shocks["date"], shocks["price"], color="#dc2626", marker="o", s=25,
               zorder=5, label=f"Shock (daily |z| >= {Z_SCORE_THRESHOLD})")
    ax.scatter(trends["date"], trends["price"], color="#ea580c", marker="^", s=35,
               zorder=5, label=f"Trend ({CUMULATIVE_WINDOW_DAYS}-day cumulative |z| >= {Z_SCORE_THRESHOLD})")
    ax.scatter(both["date"], both["price"], color="#7f1d1d", marker="*", s=60,
               zorder=6, label="Shock + trend (both checks)")

    colors = ["#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4"]
    for i, (label, start, end, _source) in enumerate(known_events):
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   color=colors[i % len(colors)], alpha=0.15, label=label)

    ax.set_title(f"{display_name} price with volatility-adjusted anomaly flags")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def write_validation_md(display_name: str, all_results: list, md_path: Path):
    events_detected = sum(1 for r in all_results if r["type"] == "event" and r["detected"])
    events_total = sum(1 for r in all_results if r["type"] == "event")
    control = next(r for r in all_results if r["type"] == "control")

    lines = [
        f"# {display_name} — Anomaly Detector Validation",
        "",
        f"Rolling window: {ROLLING_WINDOW_DAYS} days. Z-score threshold: {Z_SCORE_THRESHOLD} "
        f"(chosen before this validation, not tuned against it). Two independent checks, "
        f"either one can flag a day: a daily shock check (day-over-day return z-score) and "
        f"a {CUMULATIVE_WINDOW_DAYS}-trading-day cumulative-return check (catches slower, "
        f"multi-day build-ups a single-day check can miss - see anomaly_detector.py's "
        f"module docstring for why this was added).",
        "",
        f"**Result: {events_detected}/{events_total} known real events had at least one day "
        f"flagged. Control period false positives: {control['flagged_count']}.**",
        "",
        "## Events",
        "",
    ]
    for r in all_results:
        if r["type"] != "event":
            continue
        status = "DETECTED" if r["detected"] else "MISSED"
        types = ", ".join(t for t in r["flagged_types"] if t) or "-"
        lines.append(f"- **{r['label']}** ({r['window']}) — {status}, "
                      f"{r['flagged_count']}/{r['trading_days']} days flagged "
                      f"(type: {types}). Source: {r['source']}")
    lines += [
        "",
        "## Control period",
        "",
        f"- **{control['label']}** — {control['flagged_count']}/{control['trading_days']} days "
        f"flagged ({'clean' if control['detected'] else 'FALSE POSITIVES FOUND'}).",
        "",
        "This control period is a candidate verified by this run's own numbers above, not "
        "assumed correct in advance - see src/config/commodities.py's note on this commodity's "
        "control_period for whether it was pre-verified or is still a first-pass candidate.",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved validation writeup to {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="coffee", choices=list(COMMODITIES.keys()))
    args = parser.parse_args()

    cfg = get_commodity(args.commodity)
    input_path = REPO_ROOT / "data" / "raw" / cfg.price_file
    output_csv_path = REPO_ROOT / "results" / cfg.anomalies_file
    output_plot_path = REPO_ROOT / "results" / f"{cfg.key}_anomaly_detection_plot.png"
    validation_md_path = REPO_ROOT / "results" / cfg.validation_md_file

    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found - run "
            f"'python -m src.modeling.fetch_price_data --commodity {cfg.key}' first."
        )
    raw = pd.read_csv(input_path)
    switches = load_contract_switches(cfg.key)
    result = compute_anomalies(raw, reported_returns=switches)
    if switches:
        print(f"[{cfg.display_name}] {len(switches)} contract-switch days measured by the "
              f"reported move, not the jump in the series: {', '.join(sorted(switches))}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False)
    print(f"[{cfg.display_name}] Saved {len(result)} rows to {output_csv_path}")

    total_flagged = int(result["anomaly_flag"].sum())
    total_valid = int(result["rolling_std"].notna().sum())
    shock_only = int((result["anomaly_type"] == "shock").sum())
    trend_only = int((result["anomaly_type"] == "trend").sum())
    both_count = int((result["anomaly_type"] == "shock+trend").sum())
    print(f"[{cfg.display_name}] Flagged {total_flagged} anomalies out of {total_valid} days "
          f"with sufficient rolling history ({100*total_flagged/total_valid:.1f}%) "
          f"[shock-only: {shock_only}, trend-only: {trend_only}, both: {both_count}]")

    print(f"\n--- [{cfg.display_name}] Validation against real events + control period ---")
    all_results = validate_against_all_events(result, cfg.known_events, cfg.control_period)
    for r in all_results:
        status = "DETECTED" if r["detected"] else ("MISSED" if r["type"] == "event" else "FALSE POSITIVES")
        print(f"\n  [{r['type'].upper()}] {r['label']}")
        print(f"    window: {r['window']}  source: {r['source']}")
        print(f"    flagged {r['flagged_count']}/{r['trading_days']} days -> {status}")
        if r["flagged_dates"]:
            print(f"    dates: {list(zip(r['flagged_dates'], r['flagged_types']))}")

    events_detected = sum(1 for r in all_results if r["type"] == "event" and r["detected"])
    events_total = sum(1 for r in all_results if r["type"] == "event")
    print(f"\n  Summary: {events_detected}/{events_total} real events showed at least one flag.")

    write_validation_md(cfg.display_name, all_results, validation_md_path)
    make_plot(result, cfg.display_name, cfg.known_events, output_plot_path)


if __name__ == "__main__":
    main()
