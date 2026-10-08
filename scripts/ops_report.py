"""
Operations report: what the model calls cost in tokens and time, and how
often the primary provider was unavailable.

Reads results/llm_call_log.csv, which the explainer appends to on every call
(src/rag/explainer.log_call). Offline; writes results/ops_report.md.

What it reports
  - calls by provider and model, and the share that fell back from the
    primary provider
  - failure rate (calls that ended in API_ERROR / PARSE_ERROR)
  - prompt, output and total tokens per call: median, 90th percentile, max
  - latency per call, for calls made after latency logging was added
  - cost per explanation, if you supply prices

Cost. The log records tokens, not money. To see what the same traffic would
cost at a given rate, pass prices in dollars per million tokens:

    python -m scripts.ops_report --price-in 0.30 --price-out 2.50

No price is assumed by default - look up the current rate for the model you
would deploy rather than trusting a number baked into a script.

Usage:
    python -m scripts.ops_report
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import rate, fmt_rate   # noqa: E402

LOG_PATH = REPO_ROOT / "results" / "llm_call_log.csv"
OUTPUT_MD_PATH = REPO_ROOT / "results" / "ops_report.md"

# "READ" is a completed blind reading (src/rag/blind_evidence.py): from pipeline
# version 4 it is the call the explain/refuse decision rests on.
EXPLANATION_DECISIONS = {"EXPLAINED", "INSUFFICIENT_EVIDENCE", "READ"}
FAILURE_DECISIONS = {"API_ERROR", "PARSE_ERROR"}
PRIMARY_PROVIDER = "google-gemini"


def true_provider(model: str, logged: str) -> str:
    """Older rows logged Groq-hosted models as 'openai' because the provider
    was guessed from the model name. Resolve it from the configured model
    lists instead."""
    from src.rag.explainer import _detect_provider
    return _detect_provider(model) if isinstance(model, str) and model else logged


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found - run the pipeline at least once first.")
    df = pd.read_csv(path)
    for col in ("prompt_tokens", "output_tokens", "total_tokens", "latency_ms"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["provider"] = [true_provider(m, p) for m, p in zip(df["model"], df["provider"])]
    return df


def _dist(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "median": float(s.median()),
            "p90": float(s.quantile(0.9)), "max": float(s.max())}


def summarize(df: pd.DataFrame, price_in: float = None, price_out: float = None) -> dict:
    n = len(df)
    failures = df["decision"].isin(FAILURE_DECISIONS)
    ok = df[~failures]
    explanations = df[df["decision"].isin(EXPLANATION_DECISIONS)]
    # Only the explanation path writes a row when a call fails (the briefing
    # and outlook log successes only), so a failure rate is only meaningful
    # over explanation attempts. Dividing by every logged call would dilute it
    # with call types whose failures are invisible.
    attempts = int(len(explanations) + failures.sum())

    by_model = (ok.groupby(["provider", "model"]).size()
                .sort_values(ascending=False).reset_index(name="calls"))
    out = {
        "n_calls": n,
        "first_call": str(df["timestamp_utc"].min())[:10],
        "last_call": str(df["timestamp_utc"].max())[:10],
        "failures": rate(int(failures.sum()), attempts),
        "fallback": rate(int((ok["provider"] != PRIMARY_PROVIDER).sum()), len(ok)),
        "by_model": by_model.to_dict("records"),
        "by_decision": df["decision"].value_counts().to_dict(),
        "explanation_calls": {
            "n": int(len(explanations)),
            "prompt_tokens": _dist(explanations["prompt_tokens"]),
            "output_tokens": _dist(explanations["output_tokens"]),
            "total_tokens": _dist(explanations["total_tokens"]),
            "latency_ms": _dist(explanations["latency_ms"]),
        },
        "all_calls_total_tokens": int(ok["total_tokens"].sum(skipna=True)),
    }
    if price_in is not None and price_out is not None and len(explanations):
        cost = (explanations["prompt_tokens"].fillna(0) * price_in
                + explanations["output_tokens"].fillna(0) * price_out) / 1_000_000
        out["cost"] = {"price_in": price_in, "price_out": price_out,
                       "per_explanation_median": float(cost.median()),
                       "per_explanation_p90": float(cost.quantile(0.9)),
                       "per_1000_explanations": float(1000 * cost.mean())}
    return out


def _tokens(d: dict) -> str:
    if not d.get("n"):
        return "not recorded"
    return f"median {d['median']:,.0f}, 90th percentile {d['p90']:,.0f}, max {d['max']:,.0f}"


def report(s: dict) -> str:
    e = s["explanation_calls"]
    L = ["# Operations Report", "",
         f"{s['n_calls']} model calls logged between {s['first_call']} and {s['last_call']} "
         f"(`results/llm_call_log.csv`).", "",
         "## Reliability", "",
         f"- Explanation attempts that failed (model unavailable or unparseable response): "
         f"{fmt_rate(s['failures'])}. Other call types log successes only, so they are "
         f"not in this figure.",
         f"- Successful calls answered by a fallback provider rather than the primary "
         f"({PRIMARY_PROVIDER}): {fmt_rate(s['fallback'])}", "",
         "| Provider | Model | Successful calls |", "|---|---|---|"]
    L += [f"| {r['provider']} | {r['model']} | {r['calls']} |" for r in s["by_model"]]
    L += ["", "Which model answered matters for evaluation: 2021-07-19 was refused by the "
          "primary model and explained by the fallback in a later run. Pipeline outputs "
          "record the model that produced them.", "",
          "## Explanation calls", "",
          f"{e['n']} calls that returned an explain/refuse decision.", "",
          f"- Prompt tokens: {_tokens(e['prompt_tokens'])}",
          f"- Output tokens: {_tokens(e['output_tokens'])}",
          f"- Total tokens (including any the provider counts as reasoning): "
          f"{_tokens(e['total_tokens'])}"]
    lat = e["latency_ms"]
    if lat.get("n"):
        L.append(f"- Latency: median {lat['median'] / 1000:.1f}s, 90th percentile "
                 f"{lat['p90'] / 1000:.1f}s, max {lat['max'] / 1000:.1f}s "
                 f"(over the {lat['n']} calls made since latency was logged)")
    else:
        L.append("- Latency: not recorded for these calls. Calls made from pipeline version 2 "
                 "onwards log it.")
    L.append("")

    L += ["## Cost", ""]
    cost = s.get("cost")
    if cost:
        L += [f"At ${cost['price_in']:.2f} per million input tokens and "
              f"${cost['price_out']:.2f} per million output tokens:", "",
              f"- Per explanation: median ${cost['per_explanation_median']:.4f}, "
              f"90th percentile ${cost['per_explanation_p90']:.4f}",
              f"- Per 1,000 explanations: ${cost['per_1000_explanations']:.2f}", "",
              "These are the prices passed on the command line applied to the logged token "
              "counts, not an invoice."]
    else:
        L += ["The log records tokens, not money. To see what this traffic would cost at a "
              "given rate, run `python -m scripts.ops_report --price-in X --price-out Y` with "
              "current prices in dollars per million tokens."]
    L += ["", "Model calls are not where the time goes. The GDELT client's own notes record "
          "20-25 second responses and frequent rate-limiting, which is why GDELT responses "
          "and extracted article text are cached on disk.", ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-in", type=float, default=None,
                        help="Dollars per million input tokens")
    parser.add_argument("--price-out", type=float, default=None,
                        help="Dollars per million output tokens")
    args = parser.parse_args()
    if (args.price_in is None) != (args.price_out is None):
        parser.error("pass both --price-in and --price-out, or neither")

    text = report(summarize(load_log(), args.price_in, args.price_out))
    OUTPUT_MD_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
