"""Several models reading the same evidence: no fallback, same text, same rule."""
import json

import pytest

from scripts import compare_models as cm
from src.rag import explainer


def _reading(doc_id, direction, age=0, found=True, kind="report"):
    return {"document_id": doc_id, "trading_days_before": age, "direction": direction,
            "price_move": direction or "none", "pressure": "none", "kind": kind if direction else None,
            "cause": "a reason" if direction else None, "quote": "q" if direction else None,
            "quote_found": bool(direction) and found, "read": True}


def _run(date, direction, readings, model="model-a", read=True):
    result = {"decision": "x", "model_used": model if read else None}
    if read:
        result["evidence_document_ids"] = [r["document_id"] for r in readings]
        result["blind_evidence"] = {"readings": readings}
    return {"decision_mode": "blind", "anomaly": {"date": date, "direction": direction},
            "outcome": {"is_fault": False}, "explanation_result": result}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cm, "SHOWN_DIR", tmp_path / "shown")
    monkeypatch.setattr(cm, "READINGS_DIR", tmp_path / "readings")
    for d in ("results", "shown", "readings"):
        (tmp_path / d).mkdir()
    return tmp_path


def _store(dirs, model, date, readings):
    path = dirs / "readings" / cm.slug(model) / f"{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "readings": readings}))


def test_model_spec_and_slug():
    assert cm.parse_model("groq:qwen/qwen3.8-27b") == ("groq", "qwen/qwen3.8-27b")
    assert cm.slug("qwen/qwen3.8-27b") == "qwen_qwen3.8-27b"
    with pytest.raises(SystemExit):
        cm.parse_model("qwen")


def test_consensus_needs_most_models_and_a_found_quote():
    a = [_reading("d1", "up"), _reading("d2", "down"), _reading("d3", "up")]
    b = [_reading("d1", "up"), _reading("d2", "up"), _reading("d3", "up", found=False)]
    c = [_reading("d1", "up", kind="pressure"), _reading("d2", None), _reading("d3", None)]
    out = {r["document_id"]: r for r in cm.consensus_readings([a, b, c])}
    assert out["d1"]["direction"] == "up" and out["d1"]["kind"] == "report"   # 3 of 3, 2 say report
    assert out["d2"]["direction"] is None          # up, down, nothing: no majority
    assert out["d3"]["direction"] is None          # only one model had a quote that was found
    two = {r["document_id"]: r for r in cm.consensus_readings([a, b])}
    assert two["d1"]["direction"] == "up" and two["d2"]["direction"] is None  # 1 of 2 is not most


def test_kappa():
    E, R = cm.EXPLAINED, cm.REFUSED
    assert cm.kappa([(E, E), (R, R), (E, E), (R, R)]) == 1.0
    assert cm.kappa([(E, R), (R, E), (E, R), (R, E)]) == -1.0
    assert cm.kappa([(R, R), (R, R)]) is None            # both always refuse: undefined, not 1
    assert cm.kappa([]) is None


def test_comparison_uses_each_models_own_readings_and_the_same_rule(dirs):
    runs = {
        "2021-07-20": _run("2021-07-20", "up", [_reading("a", "up"), _reading("b", "up")]),
        "2021-07-30": _run("2021-07-30", "down", [_reading("c", "down")]),
        "2023-09-20": _run("2023-09-20", "down", [_reading("d", "up")]),
        "2020-03-03": _run("2020-03-03", "up", [], read=False),      # refused before any model
    }
    _store(dirs, "model-b", "2021-07-20", [_reading("a", "up"), _reading("b", None)])
    _store(dirs, "model-b", "2021-07-30", [_reading("c", "up")])        # reads it the other way
    _store(dirs, "model-b", "2023-09-20", [_reading("d", "up")])
    labels = {"2021-07-20": {"expected_outcome": "EXPLAIN"}, "2021-07-30": {"expected_outcome": "EXPLAIN"},
              "2023-09-20": {"expected_outcome": "REFUSE"}, "2020-03-03": {"expected_outcome": "REFUSE"}}

    c = cm.compare(runs, labels)
    assert c["models"] == ["model-a", "model-b"]
    assert c["n_read"] == 3 and c["n_decided_without_a_model"] == 1 and c["n_common"] == 3
    a, b = c["rows"]["model-a"], c["rows"]["model-b"]
    assert a["explained"]["k"] == 2 and b["explained"]["k"] == 1
    assert a["explain_labelled"]["k"] == 2 and b["explain_labelled"]["k"] == 1
    # the date no model was asked about counts as refused for everyone
    assert a["refuse_labelled"]["k"] == 2 and a["refuse_labelled"]["n"] == 2
    assert a["explained_either_direction"]["k"] == 0     # nothing explained both ways
    pair = c["pairs"][0]
    assert pair["dates"] == 3 and pair["same_decision"]["k"] == 2
    assert pair["same_direction_per_document"]["k"] == 2 and pair["same_direction_per_document"]["n"] == 4
    assert [d["date"] for d in c["disagreements"]] == ["2021-07-30"]
    # consensus of two needs both: 07-20 has "a" from both; 07-30 has no agreed direction
    assert c["rows"]["consensus"]["explained"]["k"] == 1

    report = cm.generate_report(c)
    assert "**consensus**" in report and "2021-07-30" in report and "Cohen's kappa" in report


def test_a_model_that_has_not_read_a_date_is_not_given_another_models_answer(dirs):
    runs = {"2021-07-20": _run("2021-07-20", "up", [_reading("a", "up")], model="model-a")}
    assert cm.readings_for("model-a", "2021-07-20", runs["2021-07-20"]) is not None
    assert cm.readings_for("model-b", "2021-07-20", runs["2021-07-20"]) is None
    c = cm.compare(runs, {}, models=["model-a", "model-b"])
    assert c["rows"]["model-b"]["dates_read"] == 0 and "consensus" not in c["rows"]


def test_read_calls_exactly_one_model_on_the_saved_text_and_resumes(dirs, monkeypatch, capsys):
    run = _run("2021-07-20", "up", [_reading("d1", "up")], model="model-a")
    (dirs / "results" / "pipeline_output_coffee_2021-07-20.json").write_text(json.dumps(run))
    run2 = _run("2021-07-22", "up", [_reading("d9", "up")], model="model-a")
    (dirs / "results" / "pipeline_output_coffee_2021-07-22.json").write_text(json.dumps(run2))
    (dirs / "shown" / "coffee_2021-07-20.json").write_text(json.dumps({
        "excerpt_chars": 1500, "documents": [{
            "document_id": "d1", "title": "Coffee surges as frost hits Brazil", "url": "https://a/1",
            "publication_date": "2021-07-20",
            "text": "Arabica futures surged after a severe frost damaged crops in Minas Gerais."}]}))
    # 2021-07-22 has no saved text: it cannot be read and must be reported, not guessed at

    calls = []

    class Response:
        usage_metadata = None
        text = json.dumps({"documents": [{
            "document_id": "d1", "price_move": "up", "pressure": "up", "cause": "frost damaged crops",
            "quote": "a severe frost damaged crops in Minas Gerais"}]})

    def call(provider, model, prompt, **kwargs):
        calls.append((provider, model, prompt))
        return {"response": Response(), "model": model, "provider": provider, "latency_ms": 3}

    monkeypatch.setattr(explainer, "call_one_model", call)
    monkeypatch.setattr(explainer, "check_providers", lambda names=None: names)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)

    cm.read_with("groq:model-b")
    assert [(p, m) for p, m, _ in calls] == [("groq", "model-b")]
    assert "NOT been told" in calls[0][2] and "severe frost" in calls[0][2]
    stored = cm.stored_reading("model-b", "2021-07-20")
    assert stored["readings"][0]["quote_found"] and stored["readings"][0]["direction"] == "up"
    assert "1 cannot be read" in capsys.readouterr().out

    cm.read_with("groq:model-b")                         # nothing left to do: no second call
    assert len(calls) == 1


def test_read_stops_cleanly_when_the_quota_runs_out(dirs, monkeypatch, capsys):
    for day in ("20", "21"):
        date = f"2021-07-{day}"
        run = _run(date, "up", [_reading("d1", "up")], model="model-a")
        (dirs / "results" / f"pipeline_output_coffee_{date}.json").write_text(json.dumps(run))
        (dirs / "shown" / f"coffee_{date}.json").write_text(json.dumps({
            "excerpt_chars": 1500, "documents": [{"document_id": "d1", "title": "t", "url": "u",
                                                  "publication_date": date, "text": "x"}]}))

    def call(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    monkeypatch.setattr(explainer, "call_one_model", call)
    monkeypatch.setattr(explainer, "check_providers", lambda names=None: names)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    cm.read_with("gemini:model-c")
    out = capsys.readouterr().out
    assert "out of quota after 0 date(s)" in out and "carries on" in out
    assert cm.stored_reading("model-c", "2021-07-20") is None


def test_shown_text_is_rebuilt_only_when_it_is_provably_the_same(dirs, monkeypatch):
    from src.rag import text_fetch
    cache = {"https://a/1": "y" * 3000}
    monkeypatch.setattr(text_fetch, "cached_article_text", lambda url: cache.get(url))
    run = _run("2021-07-20", "up", [_reading("d1", "up"), _reading("d2", "up")])
    run["explanation_result"]["evidence_packing"] = {"excerpt_chars": 800}
    run["sources"] = [
        {"document_id": "d1", "title": "T1", "url": "https://a/1", "publication_date": "2021-07-20",
         "text_chars": 3000},
        {"document_id": "d2", "title": "Headline only", "url": "https://news.google.com/x",
         "publication_date": "2021-07-20", "text_chars": 0}]

    pack = cm.shown_documents("2021-07-20", run)                 # nothing saved: rebuilt
    assert [d["document_id"] for d in pack["documents"]] == ["d1", "d2"]
    assert len(pack["documents"][0]["text"]) == 800 and pack["documents"][1]["text"] == ""

    cache["https://a/1"] = "y" * 2999                            # the cached text has changed
    assert cm.shown_documents("2021-07-20", run) is None
    del run["sources"][0]["text_chars"]                          # an older run with no length saved
    assert cm.shown_documents("2021-07-20", run) is None
