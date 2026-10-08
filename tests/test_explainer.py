"""Explainer: prompt versions, per-provider packing, logging. No network -
the model call is replaced in every test."""
import csv
import types

import pytest

from src.rag import explainer as ex


def _docs(n, chars=3000):
    return [{"document_id": f"doc{i:02d}", "title": f"Coffee story {i}",
             "url": f"https://s{i}.example/a", "publication_date": "2024-09-20",
             "text": "palavra " * (chars // 8)} for i in range(n)]


def _fake_info(text, provider="groq", model="m", packing=None):
    resp = types.SimpleNamespace(text=text, usage_metadata=None)
    return {"response": resp, "model": model, "provider": provider,
            "latency_ms": 12, "packing": packing}


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "LOG_PATH", tmp_path / "llm_call_log.csv")


def test_no_documents_never_calls_the_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model must not be called with no evidence")
    monkeypatch.setattr(ex, "_call_model_ex", boom)
    out = ex.generate_explanation("2024-09-23", "+5%", [])
    assert out["decision"] == "INSUFFICIENT_EVIDENCE"


def test_v1_prompt_is_the_original_four_rule_prompt():
    assert "5. The evidence has to account" not in ex.PROMPT_TEMPLATES["v1"]
    assert "5. The evidence has to account" in ex.PROMPT_TEMPLATES["v2"]
    assert ex.PROMPT_TEMPLATE == ex.PROMPT_TEMPLATES["v1"]


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError):
        ex.generate_explanation("2024-09-23", "+5%", _docs(1), prompt_version="v9")


def test_evidence_is_packed_to_the_provider_budget(monkeypatch):
    seen = {}

    def fake_call(builder, **kwargs):
        prompt, packing = builder(5500)           # what a budgeted provider gets
        seen["prompt"], seen["kwargs"] = prompt, kwargs
        return _fake_info('{"insufficient_evidence": true, "explanation": "weak"}',
                          packing=packing)
    monkeypatch.setattr(ex, "_call_model_ex", fake_call)

    out = ex.generate_explanation("2024-09-23", "+5.1%", _docs(25))
    assert int(len(seen["prompt"]) / 3.2) <= 5200
    assert out["evidence_packing"]["truncated"] is True
    assert out["evidence_packing"]["n_input"] == 25
    assert len(out["evidence_document_ids"]) == out["evidence_packing"]["n_packed"]
    assert seen["kwargs"]["temperature"] == 0.0
    assert out["decision"] == "INSUFFICIENT_EVIDENCE"
    assert out["provider"] == "groq" and out["prompt_version"] == "v2"


def test_unbudgeted_provider_gets_every_document(monkeypatch):
    def fake_call(builder, **kwargs):
        prompt, packing = builder(None)
        return _fake_info('{"insufficient_evidence": false, "explanation": "x", '
                          '"citations": [], "confidence": "low"}',
                          provider="google-gemini", packing=packing)
    monkeypatch.setattr(ex, "_call_model_ex", fake_call)
    out = ex.generate_explanation("2024-09-23", "+5.1%", _docs(25))
    assert out["evidence_packing"]["n_packed"] == 25
    assert out["evidence_packing"]["truncated"] is False
    assert out["decision"] == "EXPLAINED"


def test_v2_evidence_block_states_how_old_each_document_is():
    doc = {"document_id": "a", "title": "t", "url": "u", "text": "x",
           "publication_date": "2023-09-14", "n_copies": 4}
    block = ex.build_evidence_block([doc], anomaly_date="2023-09-20")
    assert "4 trading days before the move" in block     # Thu 14th -> Wed 20th
    assert "carried by 4 outlets" in block
    # v1 format is untouched
    plain = ex.build_evidence_block([doc])
    assert "before the move" not in plain and "outlets" not in plain


def test_invalid_json_is_a_parse_error_not_an_explanation(monkeypatch):
    monkeypatch.setattr(ex, "_call_model_ex",
                        lambda b, **k: _fake_info("not json", packing=b(None)[1]))
    out = ex.generate_explanation("2024-09-23", "+5%", _docs(2))
    assert out["decision"] == "PARSE_ERROR"


def test_model_failure_is_an_api_error(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("Error code: 413 - Request too large")
    monkeypatch.setattr(ex, "_call_model_ex", fail)
    out = ex.generate_explanation("2024-09-23", "+5%", _docs(2))
    assert out["decision"] == "API_ERROR" and "413" in out["reason"]


def test_provider_is_logged_as_reported_not_guessed():
    # 'openai/gpt-oss-120b' is a Groq-hosted model; it used to be logged as OpenAI.
    assert ex._detect_provider("openai/gpt-oss-120b") == "groq"
    assert ex._detect_provider("gemini-2.5-flash") == "google-gemini"
    assert ex._detect_provider("gpt-4o-mini") == "openai"


def test_old_log_is_migrated_to_the_new_columns(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    log.write_text(
        "timestamp_utc,anomaly_date,model,provider,prompt_tokens,output_tokens,total_tokens,decision\n"
        "2026-09-12T00:20:07+00:00,2026-07-20,gemini-3.6-flash,google-gemini,431,76,1575,EXPLAINED\n",
        encoding="utf-8")
    monkeypatch.setattr(ex, "LOG_PATH", log)
    ex.log_call("2024-09-23", None, "EXPLAINED", model="m", provider="groq",
                latency_ms=850, documents_in_prompt=17, prompt_version="v2")
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["decision"] == "EXPLAINED" and rows[0]["latency_ms"] == ""
    assert rows[1]["latency_ms"] == "850" and rows[1]["provider"] == "groq"
    assert rows[1]["documents_in_prompt"] == "17"


def test_retry_hint_is_parsed_from_rate_limit_messages():
    exc = RuntimeError("Rate limit reached ... Please try again in 12.5s. Visit ...")
    assert ex._suggested_wait_seconds(exc) == 13.0
    assert ex._suggested_wait_seconds(RuntimeError("try again in 2m3s")) == 65.0  # capped
    assert ex._suggested_wait_seconds(RuntimeError("no hint")) is None


def test_request_too_large_is_recognised():
    assert ex._is_request_too_large(RuntimeError("Error code: 413 - Request too large for model"))
    assert not ex._is_request_too_large(RuntimeError("503 overloaded"))


def test_provider_order_can_be_restricted(monkeypatch):
    monkeypatch.setenv("EXPLAINER_PROVIDERS", "groq")
    assert ex.provider_order() == ["groq"]
    monkeypatch.setenv("EXPLAINER_PROVIDERS", "Groq, google-gemini, groq")
    assert ex.provider_order() == ["groq", "gemini"]         # alias, de-duplicated
    monkeypatch.delenv("EXPLAINER_PROVIDERS")
    assert ex.provider_order() == ["gemini", "groq", "openai"]


def _keys(monkeypatch, **present):
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        if present.get(name):
            monkeypatch.setenv(name, "test-value")
        else:
            monkeypatch.delenv(name, raising=False)


def test_providers_are_tried_in_the_order_given(monkeypatch):
    _keys(monkeypatch, GEMINI_API_KEY=True, GROQ_API_KEY=True)
    calls = []

    def fake_gemini(prompt_or_builder, retries, backoff, temperature):
        calls.append("gemini")
        return None, RuntimeError("503 overloaded")

    def fake_fallback(prompt, cfg, **kw):
        calls.append(cfg["provider_name"])
        return types.SimpleNamespace(text="{}", usage_metadata=None), cfg["models"][0]

    monkeypatch.setattr(ex, "_try_gemini", fake_gemini)
    monkeypatch.setattr(ex, "_call_openai_compatible", fake_fallback)

    info = ex._call_model_ex("p")                              # default: gemini first
    assert calls == ["gemini", "groq"] and info["provider"] == "groq"

    calls.clear()
    info = ex._call_model_ex("p", providers=["groq", "gemini"])
    assert calls == ["groq"] and info["provider"] == "groq"

    calls.clear()
    ex._call_model_ex("p", providers=["openai", "groq"])       # no OpenAI key: skipped
    assert calls == ["groq"]


def test_no_configured_provider_is_a_clear_exit(monkeypatch):
    _keys(monkeypatch)
    with pytest.raises(SystemExit) as err:
        ex._call_model_ex("p", providers=["groq"])
    assert "gemini, groq, openai" in str(err.value)


def test_temperature_is_left_alone_for_gemini_3_models(monkeypatch):
    sent = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            sent.append((model, dict(config)))
            return types.SimpleNamespace(text="{}", usage_metadata=None)

    monkeypatch.setattr(ex, "get_client", lambda: types.SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(ex, "MODEL_CANDIDATES", ["gemini-3.6-flash"])
    ex._try_gemini("p", 0, 0, 0.0)
    monkeypatch.setattr(ex, "MODEL_CANDIDATES", ["gemini-2.5-flash"])
    ex._try_gemini("p", 0, 0, 0.0)
    assert "temperature" not in sent[0][1]
    assert sent[1][1]["temperature"] == 0.0


def test_status_is_read_from_the_error_not_from_stray_digits():
    class ApiError(Exception):
        def __init__(self, message, status_code=None):
            super().__init__(message)
            self.status_code = status_code

    rate_limited = ApiError("Rate limit reached: Limit 8000, Used 7413, Requested 404. "
                            "Please try again in 4.2s", status_code=429)
    assert ex._http_status(rate_limited) == 429
    assert not ex._is_request_too_large(rate_limited)       # "7413" is not a 413
    assert not ex._is_permanent_model_error(rate_limited)   # "404" is a token count here
    assert ex._is_request_too_large(ApiError("too big", status_code=413))
    assert ex._is_permanent_model_error(ApiError("gone", status_code=404))
    assert ex._http_status(RuntimeError("Error code: 413 - Request too large")) == 413
    assert ex._http_status(RuntimeError("used 7413 tokens")) is None


def test_odd_citation_shapes_are_normalised():
    assert ex._clean_citations({"document_id": "a", "supports": "x"}) == \
        [{"document_id": "a", "supports": "x"}]
    assert ex._clean_citations(["a", {"document_id": "b"}, 7, ""]) == \
        [{"document_id": "a", "supports": ""}, {"document_id": "b"}]
    assert ex._clean_citations(None) == [] and ex._clean_citations("a") == []


def test_numeric_confidence_and_string_citations_do_not_break_the_run(monkeypatch):
    monkeypatch.setattr(ex, "_call_model_ex", lambda b, **k: _fake_info(
        '{"insufficient_evidence": false, "explanation": "x", '
        '"citations": ["doc00"], "confidence": 0.9}', packing=b(None)[1]))
    out = ex.generate_explanation("2024-09-23", "+5%", _docs(2))
    assert out["confidence"] == "0.9"
    assert out["citations"] == [{"document_id": "doc00", "supports": ""}]


def test_fridays_news_is_one_trading_day_before_a_monday_move():
    doc = {"document_id": "a", "title": "t", "url": "u", "text": "x",
           "publication_date": "2024-09-20"}                  # a Friday
    block = ex.build_evidence_block([doc], anomaly_date="2024-09-23")   # the Monday
    assert "1 trading day before the move" in block
    assert "weekend in between does not make it old" in ex.PROMPT_TEMPLATES["v2"]


def test_the_fallback_line_says_why_the_first_provider_did_not_answer():
    class Quota(Exception):
        code = 429

    assert ex._why_unavailable(Quota("429 RESOURCE_EXHAUSTED: quota exceeded for the day")) \
        == "quota or rate limit reached, HTTP 429"
    assert ex._why_unavailable(TimeoutError("timed out")) == "TimeoutError"
    assert ex._why_unavailable(None) == "no response"


def test_the_reason_reported_is_the_wanted_models_not_the_last_candidates():
    class ApiError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    quota = ApiError("429 RESOURCE_EXHAUSTED: daily quota exceeded", 429)
    gone = ApiError("404 NOT_FOUND: model is not found", 404)
    assert ex._most_telling([quota, gone, gone]) is quota
    assert ex._most_telling([gone, quota]) is quota
    busy = ApiError("503 UNAVAILABLE: high demand", 503)
    assert ex._most_telling([busy, gone]) is busy
    assert ex._most_telling([]) is None
    assert ex._why_unavailable(ex._most_telling([quota, gone])) == "quota or rate limit reached, HTTP 429"


def test_a_mistyped_provider_is_caught_before_any_work(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert ex.check_providers(["groq"]) == ["groq"]
    with pytest.raises(SystemExit) as err:
        ex.check_providers(["grq"])
    assert "did you mean 'groq'" in str(err.value)
    monkeypatch.setenv("EXPLAINER_PROVIDERS", "grq")
    with pytest.raises(SystemExit):
        ex.check_providers()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as err:
        ex.check_providers(["openai"])
    assert "OPENAI_API_KEY" in str(err.value)


def test_call_one_model_asks_only_that_model(monkeypatch):
    seen = []

    def fake(prompt, config, **kwargs):
        seen.append((config["models"], config["provider_name"], kwargs.get("retries")))
        return types.SimpleNamespace(text="{}", usage_metadata=None), config["models"][0]

    monkeypatch.setattr(ex, "_call_openai_compatible", fake)
    out = ex.call_one_model("groq", "qwen/qwen3.8-27b", "prompt", temperature=0.0)
    assert seen == [(["qwen/qwen3.8-27b"], "groq", 1)]
    assert out["model"] == "qwen/qwen3.8-27b" and out["provider"] == "groq"
    with pytest.raises(ValueError):
        ex.call_one_model("grq", "x", "prompt")
    assert "groq" in ex.provider_models() and "gemini" in ex.provider_models()
