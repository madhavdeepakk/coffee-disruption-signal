"""
Grounded LLM explanation generation.

Design (team proposal S3.3 + plan Week 7): the LLM is ONLY ever called with
documents that already passed the relevance gate (src/rag/relevance_gate.py).
This module enforces that as a hard rule, not a convention - calling
generate_explanation() with zero accepted documents short-circuits to
INSUFFICIENT_EVIDENCE without ever calling the API. The gate is the safety
mechanism; this module never second-guesses a gate rejection by trying
anyway "just to see."

The prompt requires: every claim grounded in a specific cited document, no
causal claims beyond what the cited documents support, and a model-side
"insufficient evidence" fallback in addition to the gate's own
pre-filtering. This is defense in depth: the gate operates on retrieval
scores and never reads the text closely, so a document that scored well but
reads as vague on closer inspection can still trigger a refusal downstream
of the gate.

Per-call logging (team plan Week 6/8 requirement): every call appends a row
to results/llm_call_log.csv with timestamp, model, token counts - so
evaluation numbers are attributable to one known configuration, not
"whichever model answered that day."

Setup:
    pip install google-genai python-dotenv
    Put your Gemini API key in a file named .env in the repo root:
        GEMINI_API_KEY=your-key-here
    Do not commit .env to version control. If the key is ever exposed,
    regenerate it at aistudio.google.com.

Usage:
    python -m src.rag.explainer   (runs a self-test with synthetic + real data)
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.rag.evidence import pack_evidence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = REPO_ROOT / "results" / "llm_call_log.csv"

MODEL_NAME = "gemini-3.6-flash"  # primary model; if it 404s, check aistudio.google.com's
                                  # model list for the current default flash-tier model name

# Models to try in order. The call retries a transient "model busy" (503) on
# each, and falls through to the next model only if one is genuinely
# unavailable. Keep only models your API key can actually access - a name that
# 404s here is skipped, so listing dead models just wastes a step. Run
# `python -m src.rag.explainer --list-models` to see the names your key
# supports, then add valid alternates here (order = preference).
# Checked 6 October 2026: gemini-2.5-flash is closed to new keys and
# gemini-2.0-flash is retired, with Google's own 404 naming
# gemini-3.8-flash as the replacement. A retired name here is not
# harmless: with every alternate dead, a quota error on the primary
# left no working Gemini model at all.
MODEL_CANDIDATES = ["gemini-3.6-flash", "gemini-3.8-flash"]

PROVIDER = "google-gemini"

# ---------------------------------------------------------------------------
# Fallback provider: OpenAI-compatible API (Groq, OpenAI, Together, etc.)
# ---------------------------------------------------------------------------
# When Gemini quota is exhausted (RESOURCE_EXHAUSTED / 429 on all candidates),
# the system tries an OpenAI-compatible API if one is configured. Priority:
#   1. GROQ_API_KEY  -> api.groq.com  (free tier: 30 req/min, generous daily)
#   2. OPENAI_API_KEY -> api.openai.com (paid)
# Set the key in .env alongside GEMINI_API_KEY. The fallback uses the same
# prompt and JSON-mode constraint.
#
# max_prompt_tokens is the largest evidence prompt this provider is sent. The
# Groq free tier rejects any single request over 8,000 tokens (prompt plus
# reserved output) with a 413, which is what turned 2024-09-23 and 2024-12-02
# into SYSTEM_ERROR: 24-25 accepted documents at 1,500 characters each came to
# 8,400-11,000 tokens. With a budget set, the evidence is packed to fit (see
# src/rag/evidence.pack_evidence) instead of being sent whole and refused.
# None = no limit for that provider.
FALLBACK_CONFIGS = [
    {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "provider_name": "groq",
        "max_prompt_tokens": 5000,
    },
    {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini"],
        "provider_name": "openai",
        "max_prompt_tokens": None,
    },
]

# Gemini's context window is far larger than any evidence set this pipeline
# produces, so the primary provider is not budgeted.
GEMINI_MAX_PROMPT_TOKENS = None

# Output cap for explanation calls on the fallback providers, so that prompt
# + output stays under the per-request limit above. The explanation itself is
# a few hundred tokens, but the fallback models reason before answering and
# that counts as output: logged explanation calls ran up to about 1,300
# output tokens. The cap leaves room above that, because a response cut off
# mid-JSON is a parse error.
EXPLANATION_MAX_OUTPUT_TOKENS = 2000

# Explanations are generated at temperature 0 where the model supports it.
# The same date was explained by one model and refused by another across
# re-runs (2021-07-19), so removing sampling noise is the minimum needed to
# make a run repeatable. Different models can still disagree - model_used and
# provider are recorded per run.
EXPLANATION_TEMPERATURE = 0.0

# Google's guidance for the Gemini 3 family is to leave temperature at its
# default of 1.0: lower values "may lead to unexpected behavior, looping, or
# degraded performance". So the temperature above is NOT applied to those
# models, which means a Gemini 3 run is not exactly repeatable and a result
# that matters should be confirmed over more than one run.
GEMINI_FIXED_TEMPERATURE_PREFIXES = ("gemini-3",)

_PROVIDER_ALIASES = {"google-gemini": "gemini", "google": "gemini"}
DEFAULT_PROVIDER_ORDER = ["gemini", "groq", "openai"]


def provider_order() -> list:
    """Providers to try, in the order they are tried. Override with
    EXPLAINER_PROVIDERS, e.g. EXPLAINER_PROVIDERS=groq to run an experiment on
    one model family only (so results are not a mix of whichever provider
    happened to be up), or EXPLAINER_PROVIDERS=groq,gemini to change the
    order."""
    raw = os.environ.get("EXPLAINER_PROVIDERS", ",".join(DEFAULT_PROVIDER_ORDER))
    return normalize_providers(raw.split(",")) or list(DEFAULT_PROVIDER_ORDER)


def normalize_providers(names) -> list:
    """Lowercase, apply aliases (the call log writes 'google-gemini'), drop
    blanks and repeats, keep order."""
    out = []
    for name in names or []:
        name = _PROVIDER_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
        if name and name not in out:
            out.append(name)
    return out

load_dotenv(REPO_ROOT / ".env")


def check_providers(names=None) -> list:
    """The provider order that will be used, after checking that every name
    is a real provider and at least one of them has its API key set. Raises
    SystemExit with a plain message otherwise.

    Call this before doing any work. A mistyped name used to surface only at
    the model call, after several minutes of retrieval, and then for every
    date in a batch."""
    import difflib
    order = normalize_providers(names) if names else provider_order()
    keys = {"gemini": "GEMINI_API_KEY",
            **{cfg["provider_name"]: cfg["env_key"] for cfg in FALLBACK_CONFIGS}}
    unknown = [name for name in order if name not in keys]
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, list(keys), n=1)
            hints.append(f"'{name}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise SystemExit(f"Unknown model provider {', '.join(hints)}. "
                         f"Provider names are: {', '.join(keys)}.")
    if not any(os.environ.get(keys[name]) for name in order):
        raise SystemExit(f"No API key is set for {', '.join(order)}. Add "
                         f"{' or '.join(keys[name] for name in order)} to the .env file in "
                         f"the project root.")
    return order


_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY not set. Create a .env file in the repo root "
                "with: GEMINI_API_KEY=your-key-here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


# Prompt versions. v1 is the original four-rule prompt and is kept so the
# robustness harness (src/evaluation/robustness.py) can run both on the same
# evidence. v2 adds two rules, each written against a failure seen in stored
# runs that were labelled should-refuse:
#   rule 5 - 2023-09-20 (-5.6%) was "explained" from one article published six
#            days earlier ("sentiment likely persisted into the following
#            week"), and 2021-07-19 (-3.7%) from evidence that was 13-to-2
#            about prices RISING.
#   rule 6 - the 2021-07-19 explanation's only causal claim was that the drop
#            was "a short-term pull-back after the earlier rally", which is the
#            move described again, not a cause, and appeared in no document.
# v2 also tells the model how many TRADING days before the move each document
# was published. The first wording of rule 5 said "several days before" and
# the date line counted calendar days; on its first live run that made the
# model refuse 2024-09-23 (a Monday, labelled explainable) because the
# drought stories were from Friday - "3 days before". Rule 5 now names a
# threshold in trading days and says a weekend does not make news old.
# Whether v2 lowers the false-explanation rate WITHOUT also refusing dates it
# should explain is an empirical question the harness answers by running v1
# and v2 on the same evidence; it is not assumed here.
_PROMPT_HEAD = """You are explaining a detected coffee price anomaly using ONLY the retrieved news evidence provided below. This is a research/evaluation tool, not financial advice.

ANOMALY:
Date: {anomaly_date}
Price move: {price_description}

RETRIEVED EVIDENCE (already relevance-filtered - these documents passed a relevance gate):
{evidence_block}

RULES (all mandatory):
1. Every factual claim in your explanation must cite a specific document by its document_id, e.g. "(source: doc_abc123)".
2. Do not state or imply a causal link that isn't directly supported by the cited document's content. If a document only mentions a topic in passing without a clear causal claim, say so rather than inflating it.
3. If, after reading the evidence closely, you judge it is too vague, off-topic, or insufficient to support a genuine causal explanation (even though it passed the initial relevance gate), respond with insufficient_evidence: true and explain why in one sentence. Do not force an explanation from weak material.
4. Do not speculate beyond the provided documents. Do not use outside knowledge about coffee markets not present in the evidence.
"""

_PROMPT_RULES_V2 = """5. The evidence has to account for a move in THIS direction on THIS date. A document reporting prices moving the opposite way does not explain it. Each document's date line says how many trading days before the move it was published. News from the same day or the previous trading day (a weekend in between does not make it old) is timely and can explain the move. A document more than three trading days old that only describes background conditions does not, on its own, explain why the price moved on this date. If opposite-direction or stale background material is all the evidence offers, respond with insufficient_evidence: true.
6. Do not explain the move by re-describing it. "Profit-taking", "a pull-back after the rally" or "sentiment persisted" are acceptable only if a cited document itself gives that reason for this date.
"""

_PROMPT_TAIL = """
Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "insufficient_evidence": false,
  "explanation": "...",
  "citations": [{{"document_id": "...", "supports": "..."}}],
  "confidence": "high" | "medium" | "low"
}}
"""

PROMPT_TEMPLATES = {
    "v1": _PROMPT_HEAD + _PROMPT_TAIL,
    "v2": _PROMPT_HEAD + _PROMPT_RULES_V2 + _PROMPT_TAIL,
}
PROMPT_VERSION = "v2"

# Original name, kept for anything that imports it directly.
PROMPT_TEMPLATE = PROMPT_TEMPLATES["v1"]


CONTEXT_PROMPT_TEMPLATE = """You are summarizing the CURRENT news backdrop for coffee supply and prices, using ONLY the retrieved news evidence below. This is a research/evaluation tool, not financial advice.

AS OF DATE: {as_of_date}

RETRIEVED EVIDENCE (already relevance-filtered - these documents passed a relevance gate):
{evidence_block}

RULES (all mandatory):
1. Summarize, in 2-3 sentences, what recent news reports about factors currently affecting coffee supply and prices (weather, harvest, tariffs, demand, etc.).
2. Every factual claim must cite a specific document by its document_id, e.g. "(source: doc_abc123)".
3. Describe only what the documents say. Do NOT predict future prices, and do NOT claim these factors will cause any particular price move - this is context, not a forecast.
4. If the evidence is too vague or off-topic to describe the current situation, respond with insufficient_evidence: true and say why in one sentence. Do not force a summary from weak material.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "insufficient_evidence": false,
  "explanation": "...",
  "citations": [{{"document_id": "...", "supports": "..."}}],
  "confidence": "high" | "medium" | "low"
}}
"""


_STATUS_IN_MESSAGE_RE = re.compile(r"(?:error code:|status(?: code)?[:=]?|http)\s*(\d{3})\b",
                                   re.IGNORECASE)


def _http_status(exc: Exception):
    """The HTTP status of a provider error, or None.

    Read from the exception's own attribute where the client sets one
    (openai: status_code; google-genai: code). Only if there is none is the
    message searched, and then only for a status written as a status
    ("Error code: 413") - a bare "413" or "404" can just as well be a token
    count inside a rate-limit message, and treating that as "model not found"
    skips a model that only needed a few seconds' wait.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = _STATUS_IN_MESSAGE_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _is_permanent_model_error(exc: Exception) -> bool:
    """A 404 / 'model not found / no longer available' means the model name is
    wrong for this key - retrying it is pointless, so skip straight to the next
    candidate. A 503 'high demand', rate limit, or network blip is transient
    and worth retrying."""
    msg = str(exc).lower()
    return (_http_status(exc) == 404 or "not_found" in msg or "not found" in msg
            or "no longer available" in msg or "is not found" in msg
            or "does not exist" in msg)


def _is_quota_error(exc: Exception) -> bool:
    """True if the error is a quota/rate-limit exhaustion (not a transient
    busy signal). These won't resolve by retrying the same provider."""
    msg = str(exc).lower()
    return ("resource_exhausted" in msg or "quota" in msg
            or "rate limit" in msg or "rate_limit" in msg
            or "too many requests" in msg)


def _is_request_too_large(exc: Exception) -> bool:
    """A 413 / 'request too large' is about THIS prompt on THIS model's limit.
    Retrying the identical request can never succeed, so move on."""
    msg = str(exc).lower()
    return (_http_status(exc) == 413 or "request too large" in msg
            or "reduce your message size" in msg)


_RETRY_HINT_RE = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*s", re.IGNORECASE)


def _suggested_wait_seconds(exc: Exception, cap: float = 65.0):
    """Per-minute rate limits come back with 'Please try again in 12.5s'.
    Waiting that long (rather than a fixed 3-6s) is what lets a batch run get
    through a tokens-per-minute limit instead of failing every third call."""
    m = _RETRY_HINT_RE.search(str(exc))
    if not m:
        return None
    seconds = float(m.group(2)) + 60 * int(m.group(1) or 0)
    return min(seconds + 0.5, cap)


class _FallbackUsage:
    def __init__(self, usage):
        self.prompt_token_count = getattr(usage, "prompt_tokens", 0)
        self.candidates_token_count = getattr(usage, "completion_tokens", 0)
        self.total_token_count = getattr(usage, "total_tokens", 0)


class _FallbackResponse:
    """Wraps an OpenAI-style completion in the shape callers expect from the
    Gemini client (.text and .usage_metadata)."""
    def __init__(self, text, usage):
        self.text = text
        self.usage_metadata = usage


def _call_openai_compatible(prompt: str, config: dict, retries: int = 2,
                             backoff_seconds: int = 3, temperature: float = 0.3,
                             max_output_tokens: int = None):
    """Call an OpenAI-compatible API as a fallback. Tries each model in
    config['models'] in order — skips a model on 404 (not available for this
    key/tier) or 413 (prompt over that model's request limit), retries on
    transient errors. Returns (response, model) where response has .text and
    .usage_metadata attributes matching what callers expect from the Gemini
    response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"{config['env_key']} not set in .env")

    client = OpenAI(api_key=api_key, base_url=config["base_url"])

    models = config.get("models", [config.get("model", "")])
    last_exc = None

    for model in models:
        for attempt in range(retries + 1):
            try:
                kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a commodity market analyst. Respond with ONLY valid JSON, no other text."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                if max_output_tokens:
                    kwargs["max_tokens"] = max_output_tokens
                completion = client.chat.completions.create(**kwargs)
                resp_text = completion.choices[0].message.content
                usage = _FallbackUsage(completion.usage) if completion.usage else None
                return _FallbackResponse(resp_text, usage), model

            except Exception as exc:
                last_exc = exc
                if _is_permanent_model_error(exc):
                    print(f"    {model} not available, trying next...")
                    break  # skip to next model
                if _is_request_too_large(exc):
                    print(f"    {model} rejected the request as too large, trying next...")
                    break  # the same prompt will be rejected again
                if attempt < retries:
                    wait = _suggested_wait_seconds(exc)
                    time.sleep(wait if wait is not None else backoff_seconds * (attempt + 1))

    raise last_exc


def _materialize(prompt_or_builder, max_prompt_tokens):
    """A caller passes either a finished prompt string, or a builder
    f(max_prompt_tokens) -> (prompt, packing_report) so the evidence can be
    packed to each provider's own request limit."""
    if callable(prompt_or_builder):
        return prompt_or_builder(max_prompt_tokens)
    return prompt_or_builder, None


def _why_unavailable(exc) -> str:
    """One short phrase saying why a provider did not answer, for the
    fallback line. Falling back silently hides the difference between a
    spent daily quota (every later call will fall back too, so the run is
    really being answered by another model) and a one-off error."""
    if exc is None:
        return "no response"
    if _is_quota_error(exc):
        kind = "quota or rate limit reached"
    elif _is_permanent_model_error(exc):
        kind = "model not available"
    else:
        kind = type(exc).__name__
    status = _http_status(exc)
    return f"{kind}, HTTP {status}" if status else kind


def _most_telling(errors: list):
    """Of the errors from each model candidate, the one that says why the
    provider did not answer: a quota error if there was one, else the first
    candidate's. The last candidate's error is the least informative - the
    older names at the end of MODEL_CANDIDATES fail with "model not found"
    whatever stopped the model that was actually wanted."""
    for exc in errors:
        if _is_quota_error(exc):
            return exc
    return errors[0] if errors else None


def _try_gemini(prompt_or_builder, retries, backoff_seconds, temperature):
    """Try each Gemini candidate. Returns (result_dict | None, exception): the
    exception is the most telling one across the candidates (_most_telling)."""
    client = get_client()
    prompt, packing = _materialize(prompt_or_builder, GEMINI_MAX_PROMPT_TOKENS)
    last_exc = None
    errors = []
    for model in MODEL_CANDIDATES:
        config = {"response_mime_type": "application/json"}
        if temperature is not None and not model.startswith(GEMINI_FIXED_TEMPERATURE_PREFIXES):
            config["temperature"] = temperature
        for attempt in range(retries + 1):
            try:
                started = time.monotonic()
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config,
                )
                return {"response": resp, "model": model, "provider": "google-gemini",
                        "latency_ms": int(1000 * (time.monotonic() - started)),
                        "packing": packing}, None
            except Exception as exc:  # noqa: BLE001 - transient/availability errors vary by type
                last_exc = exc
                if _is_quota_error(exc):
                    break  # quota won't clear by retrying — try next model or fallback
                if _is_permanent_model_error(exc):
                    break  # wrong model name - don't retry it, move to the next candidate
                if attempt < retries:
                    time.sleep(backoff_seconds * (attempt + 1))
        if last_exc is not None and (not errors or errors[-1] is not last_exc):
            errors.append(last_exc)
    return None, _most_telling(errors)


def _call_model_ex(prompt_or_builder, retries: int = 3, backoff_seconds: int = 4,
                   temperature: float = None, max_output_tokens: int = None,
                   providers: list = None) -> dict:
    """
    Call an LLM for a JSON response and report which provider answered.

    Providers are tried in the order given by `providers` (default:
    provider_order(), i.e. Gemini, then each OpenAI-compatible provider in
    FALLBACK_CONFIGS). A provider with no API key set is skipped.

    Returns {"response", "model", "provider", "latency_ms", "packing"}.
    latency_ms is the wall-clock time of the call that succeeded (not the
    failed attempts before it). Raises the last error only if every provider
    fails.
    """
    order = normalize_providers(providers) if providers else provider_order()
    fallbacks = {cfg["provider_name"]: cfg for cfg in FALLBACK_CONFIGS}
    last_exc = None
    tried = []

    for name in order:
        if name == "gemini":
            if not os.environ.get("GEMINI_API_KEY"):
                continue
            tried.append(name)
            result, exc = _try_gemini(prompt_or_builder, retries, backoff_seconds, temperature)
            if result is not None:
                return result
            last_exc = exc or last_exc
            continue

        cfg = fallbacks.get(name)
        if cfg is None or not os.environ.get(cfg["env_key"]):
            continue
        if tried:
            print(f"  {tried[-1]} unavailable ({_why_unavailable(last_exc)}), trying {name} "
                  f"(models: {', '.join(cfg.get('models', []))})...")
        tried.append(name)
        prompt, packing = _materialize(prompt_or_builder, cfg.get("max_prompt_tokens"))
        try:
            started = time.monotonic()
            resp, model = _call_openai_compatible(
                prompt, cfg,
                temperature=0.3 if temperature is None else temperature,
                max_output_tokens=max_output_tokens,
            )
            return {"response": resp, "model": model, "provider": name,
                    "latency_ms": int(1000 * (time.monotonic() - started)),
                    "packing": packing}
        except Exception as fb_exc:  # noqa: BLE001
            print(f"  {name} also failed: {fb_exc}")
            last_exc = fb_exc

    if last_exc is None:
        raise SystemExit(
            f"No LLM provider is configured for {order}. Create a .env file in the repo "
            "root with GEMINI_API_KEY=your-key-here (and optionally GROQ_API_KEY / "
            "OPENAI_API_KEY as fallbacks). Provider names are: gemini, groq, openai."
        )
    raise last_exc


def provider_models() -> dict:
    """provider name -> the model names configured for it, in the order the
    pipeline tries them."""
    return {"gemini": list(MODEL_CANDIDATES),
            **{cfg["provider_name"]: list(cfg.get("models", [])) for cfg in FALLBACK_CONFIGS}}


def call_one_model(provider: str, model: str, prompt: str, temperature: float = None,
                   max_output_tokens: int = None) -> dict:
    """Call exactly this model, with no fallback to another model or
    provider, and raise whatever error it gives.

    _call_model_ex answers with whichever model has quota left, which is
    right for keeping a single run going and wrong for comparing models: a
    comparison column has to contain one model's answers and nothing else.

    Returns {"response", "model", "provider", "latency_ms"}.
    """
    provider = normalize_providers([provider])[0]
    started = time.monotonic()
    if provider == "gemini":
        config = {"response_mime_type": "application/json"}
        if temperature is not None and not model.startswith(GEMINI_FIXED_TEMPERATURE_PREFIXES):
            config["temperature"] = temperature
        response = get_client().models.generate_content(model=model, contents=prompt,
                                                        config=config)
        answered = model
    else:
        cfg = next((c for c in FALLBACK_CONFIGS if c["provider_name"] == provider), None)
        if cfg is None:
            raise ValueError(f"unknown provider {provider!r}; "
                             f"expected one of {', '.join(provider_models())}")
        response, answered = _call_openai_compatible(
            prompt, {**cfg, "models": [model]}, retries=1,
            temperature=0.3 if temperature is None else temperature,
            max_output_tokens=max_output_tokens)
    return {"response": response, "model": answered, "provider": provider,
            "latency_ms": int(1000 * (time.monotonic() - started))}


def _call_model(prompt: str, retries: int = 3, backoff_seconds: int = 4):
    """
    Call an LLM for a JSON response. Returns (response, model_used). Thin
    wrapper over _call_model_ex for callers that only need those two values
    (the daily briefing and the outlook).
    """
    info = _call_model_ex(prompt, retries=retries, backoff_seconds=backoff_seconds)
    return info["response"], info["model"]


def list_available_models() -> list:
    """Return the model names this API key can use (for choosing MODEL_CANDIDATES)."""
    client = get_client()
    names = []
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        # keep the ones that can generate content
        actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        if not actions or "generateContent" in actions:
            names.append(name.replace("models/", ""))
    return names


def _api_error_result(date_str: str, exc: Exception) -> dict:
    log_call(date_str, None, "API_ERROR")
    return {"decision": "API_ERROR", "explanation": None, "citations": [],
            "confidence": None,
            "reason": f"model_unavailable: {type(exc).__name__}: {exc}"[:300]}


def _published_line(doc: dict, anomaly_date: str) -> str:
    """'2024-09-20 (1 trading day before the move)' - so the model can see how
    fresh a document is instead of working it out from two dates. Trading
    days, not calendar days: Friday's news is the latest there is for a
    Monday move, and calling it "3 days before" invites the model to treat
    it as stale."""
    from src.rag.evidence import trading_days_before
    pub = doc.get("publication_date", "") or ""
    days = doc.get("trading_days_before_anomaly")
    if days is None and anomaly_date and pub:
        days = trading_days_before(pub, anomaly_date)
    if days is None:
        return pub
    if days == 0:
        return f"{pub} (same day as the move)"
    if days > 0:
        return f"{pub} ({days} trading day{'s' if days != 1 else ''} before the move)"
    return f"{pub} ({-days} trading day{'s' if days != -1 else ''} AFTER the move)"


def build_evidence_block(documents: list, excerpt_chars: int = 1500,
                         anomaly_date: str = None) -> str:
    """Render documents as the evidence section of the prompt.

    With anomaly_date=None this is the original format (prompt v1). With an
    anomaly date, each document's date line also says how many days before
    the move it was published, and a syndicated story says how many outlets
    carried it.
    """
    lines = []
    for doc in documents:
        date_line = (_published_line(doc, anomaly_date) if anomaly_date
                     else doc.get('publication_date', ''))
        entry = (
            f"- document_id: {doc.get('document_id')}\n"
            f"  title: {doc.get('title', '')}\n"
            f"  date: {date_line}\n"
            f"  url: {doc.get('url', '')}\n"
        )
        if anomaly_date and (doc.get("n_copies") or 1) > 1:
            entry += f"  syndicated: same story carried by {doc['n_copies']} outlets\n"
        entry += f"  text: {(doc.get('text') or doc.get('title') or '')[:excerpt_chars]}"
        lines.append(entry)
    return "\n".join(lines)


def _detect_provider(model: str) -> str:
    """Infer provider name from the model string for logging. Used only when
    the caller did not say which provider answered (older call sites). Checks
    the configured fallback model lists first: 'openai/gpt-oss-120b' is served
    by Groq, and matching on the substring 'gpt' used to log it as OpenAI."""
    for cfg in FALLBACK_CONFIGS:
        if model in cfg.get("models", []):
            return cfg["provider_name"]
    m = (model or "").lower()
    if "gemini" in m:
        return "google-gemini"
    if "llama" in m or "mixtral" in m or "qwen" in m or m.startswith("openai/"):
        return "groq"
    if "gpt" in m:
        return "openai"
    return "unknown"


LOG_COLUMNS = [
    "timestamp_utc", "anomaly_date", "model", "provider",
    "prompt_tokens", "output_tokens", "total_tokens", "decision",
    "latency_ms", "documents_in_prompt", "prompt_version",
]


def _migrate_log_header():
    """Add any new columns to an existing log in place. Rows written before a
    column existed get a blank for it, so the file stays one rectangular CSV
    instead of a mix of row lengths that a CSV reader would choke on."""
    try:
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return
    if not rows or rows[0] == LOG_COLUMNS:
        return
    old_header = rows[0]
    out = [LOG_COLUMNS]
    for row in rows[1:]:
        record = dict(zip(old_header, row))
        out.append([record.get(col, "") for col in LOG_COLUMNS])
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out)


def log_call(anomaly_date: str, usage_metadata, decision: str, model: str = MODEL_NAME,
             provider: str = None, latency_ms: int = None,
             documents_in_prompt: int = None, prompt_version: str = None):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    if not is_new:
        _migrate_log_header()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            anomaly_date,
            model,
            provider or _detect_provider(model),
            getattr(usage_metadata, "prompt_token_count", ""),
            getattr(usage_metadata, "candidates_token_count", ""),
            getattr(usage_metadata, "total_token_count", ""),
            decision,
            "" if latency_ms is None else latency_ms,
            "" if documents_in_prompt is None else documents_in_prompt,
            prompt_version or "",
        ])


def generate_explanation(anomaly_date: str, price_description: str, accepted_documents: list,
                         prompt_version: str = None, providers: list = None) -> dict:
    """
    Hard rule: no documents in = no API call, immediate INSUFFICIENT_EVIDENCE.
    This function trusts that `accepted_documents` already passed the gate -
    it does not re-run gating logic, it enforces the "zero docs -> no call"
    floor as a second, cheap safeguard against ever calling the LLM with
    nothing to ground it in.

    The evidence is packed to the answering provider's request limit (see
    pack_evidence). The result records which documents were actually in the
    prompt (evidence_document_ids) and the packing report, so a citation is
    later checked against what the model was shown, and a truncation is
    visible in the output rather than silent.

    prompt_version: "v1" or "v2" (default PROMPT_VERSION).
    providers: restrict/override the provider order for this call.
    """
    if not accepted_documents:
        return {
            "decision": "INSUFFICIENT_EVIDENCE",
            "explanation": None,
            "citations": [],
            "confidence": None,
            "reason": "no_accepted_documents_passed_to_explainer",
        }

    version = prompt_version or PROMPT_VERSION
    if version not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_version {version!r}; "
                         f"available: {sorted(PROMPT_TEMPLATES)}")
    template = PROMPT_TEMPLATES[version]
    # v1 is reproduced exactly as it was originally sent, including the
    # evidence block format, so a v1-vs-v2 comparison is not confounded by
    # anything except the prompt.
    block_anomaly_date = None if version == "v1" else anomaly_date

    def build(max_prompt_tokens):
        frame_tokens = int(len(template) / 3.2) + 60
        budget = None if max_prompt_tokens is None else max(max_prompt_tokens - frame_tokens, 500)
        packed, report = pack_evidence(accepted_documents, budget)
        report["document_ids"] = [d.get("document_id") for d in packed]
        prompt = template.format(
            anomaly_date=anomaly_date,
            price_description=price_description,
            evidence_block=build_evidence_block(
                packed, excerpt_chars=report["excerpt_chars"],
                anomaly_date=block_anomaly_date),
        )
        return prompt, report

    try:
        info = _call_model_ex(build, temperature=EXPLANATION_TEMPERATURE,
                              max_output_tokens=EXPLANATION_MAX_OUTPUT_TOKENS,
                              providers=providers)
    except Exception as exc:  # noqa: BLE001 - transient API/network error
        return _api_error_result(anomaly_date, exc)

    response, model_used = info["response"], info["model"]
    packing = info.get("packing") or {}
    run_info = {
        "model_used": model_used,
        "provider": info["provider"],
        "prompt_version": version,
        "latency_ms": info["latency_ms"],
        "evidence_document_ids": packing.get("document_ids", []),
        "evidence_packing": {k: v for k, v in packing.items() if k != "document_ids"},
    }
    log_kwargs = dict(model=model_used, provider=info["provider"],
                      latency_ms=info["latency_ms"],
                      documents_in_prompt=packing.get("n_packed"),
                      prompt_version=version)

    try:
        parsed = json.loads(response.text)
    except (ValueError, TypeError) as exc:
        # Model didn't return valid JSON despite the response_mime_type
        # constraint - treat as insufficient evidence rather than guessing
        # at a malformed response's meaning. Logged distinctly so a string
        # of these would be visible as "prompt/parsing needs work," not
        # silently swallowed.
        log_call(anomaly_date, getattr(response, "usage_metadata", None), "PARSE_ERROR",
                 **log_kwargs)
        return {
            "decision": "PARSE_ERROR",
            "explanation": None,
            "citations": [],
            "confidence": None,
            "reason": f"model_response_not_valid_json: {exc}",
            "raw_response": (response.text or "")[:500],
            **run_info,
        }

    if not isinstance(parsed, dict):
        log_call(anomaly_date, getattr(response, "usage_metadata", None), "PARSE_ERROR",
                 **log_kwargs)
        return {"decision": "PARSE_ERROR", "explanation": None, "citations": [],
                "confidence": None, "reason": "model_response_not_a_json_object",
                "raw_response": (response.text or "")[:500], **run_info}

    if parsed.get("insufficient_evidence"):
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        decision = "EXPLAINED"

    log_call(anomaly_date, getattr(response, "usage_metadata", None), decision, **log_kwargs)

    explanation = parsed.get("explanation")
    confidence = parsed.get("confidence")
    return {
        "decision": decision,
        "explanation": None if explanation is None else str(explanation),
        "citations": _clean_citations(parsed.get("citations")),
        "confidence": None if confidence is None else str(confidence),
        "reason": None if decision == "EXPLAINED" else "model_judged_evidence_insufficient",
        **run_info,
    }


def _clean_citations(raw) -> list:
    """The prompt asks for a list of {document_id, supports} objects. Models
    occasionally return a single object, or a list of bare ids. Everything
    downstream assumes dicts, so the shape is normalised here: an object
    becomes a one-item list, a bare string becomes {document_id: string}, and
    anything else is dropped."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"document_id": item.strip(), "supports": ""})
    return out


def generate_context_summary(as_of_date: str, accepted_documents: list) -> dict:
    """
    Summarize the CURRENT news backdrop for coffee from already-gated documents.
    Same safety discipline as generate_explanation: no documents in means no
    API call and an immediate INSUFFICIENT_EVIDENCE. The prompt forbids price
    prediction or causal claims about any forecast - this is context only.
    """
    if not accepted_documents:
        return {"decision": "INSUFFICIENT_EVIDENCE", "explanation": None,
                "citations": [], "confidence": None,
                "reason": "no_accepted_documents_passed_to_explainer"}

    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        as_of_date=as_of_date,
        evidence_block=build_evidence_block(accepted_documents),
    )
    try:
        response, model_used = _call_model(prompt)
    except Exception as exc:  # noqa: BLE001 - transient API/network error
        return _api_error_result(as_of_date, exc)
    try:
        parsed = json.loads(response.text)
    except (ValueError, TypeError) as exc:
        log_call(as_of_date, getattr(response, "usage_metadata", None), "PARSE_ERROR",
                 model=model_used)
        return {"decision": "PARSE_ERROR", "explanation": None, "citations": [],
                "confidence": None, "reason": f"model_response_not_valid_json: {exc}",
                "raw_response": response.text[:500]}

    decision = "INSUFFICIENT_EVIDENCE" if parsed.get("insufficient_evidence") else "EXPLAINED"
    log_call(as_of_date, getattr(response, "usage_metadata", None), decision, model=model_used)
    return {"decision": decision, "explanation": parsed.get("explanation"),
            "citations": parsed.get("citations", []), "confidence": parsed.get("confidence"),
            "reason": None if decision == "EXPLAINED" else "model_judged_evidence_insufficient",
            "model_used": model_used}


def _self_test():
    print(f"Testing explainer with model={MODEL_NAME} via {PROVIDER}\n")

    # Case 1: no documents - must short-circuit, no API call
    result = generate_explanation("2026-07-20", "16% single-day spike", [])
    print("Case 1 (no documents):", result["decision"], "- reason:", result["reason"])
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"

    # Case 2: real-shaped synthetic evidence, should produce a grounded explanation
    docs = [
        {
            "document_id": "test_doc_1",
            "title": "Frost concerns trigger sharp rally in coffee futures",
            "publication_date": "2026-07-06",
            "url": "https://example.com/coffee-frost",
            "text": (
                "Arabica coffee futures surged more than 16% on Monday after "
                "reports of frost damage to key growing regions in Brazil's "
                "Minas Gerais state, traders said, as concerns mounted over "
                "further supply disruption following an already tight harvest."
            ),
        }
    ]
    result = generate_explanation("2026-07-07", "Arabica up 16% in one session", docs)
    print("\nCase 2 (real-shaped evidence):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action="store_true",
                        help="List the model names this API key can use, then exit")
    args = parser.parse_args()
    if args.list_models:
        print("Models your API key can use for generateContent:")
        for name in list_available_models():
            print(f"  {name}")
        print("\nPut the ones you want into MODEL_CANDIDATES at the top of this file "
              "(primary first).")
    else:
        _self_test()
