"""Citation audit: sheet building, quote verification, scoring."""
from scripts import audit_citations as ac
from src.evaluation.metrics import rate


def test_quote_must_really_be_in_the_document():
    doc = "Arabica coffee futures  surged more than 16% on Monday after reports of frost."
    assert ac.quote_in_document("surged more than 16% on Monday", doc)
    assert ac.quote_in_document("  Arabica coffee futures surged MORE than 16%  ", doc)   # spacing/case
    assert not ac.quote_in_document("prices fell 16% on Monday after frost", doc)
    assert not ac.quote_in_document("16%", doc)                    # too short to mean anything


def test_reexport_keeps_labels_already_entered():
    old = [{"anomaly_date": "2021-07-20", "document_id": "a", "claim": "frost lifted prices",
            "human_label": "supported", "notes": "clear", "judge_verdict": "supported",
            "judge_quote": "q", "judge_quote_found": "yes", "judge_model": "m"}]
    new = [{"anomaly_date": "2021-07-20", "document_id": "a", "claim": "frost lifted prices",
            "human_label": "", "notes": "", "judge_verdict": "", "judge_quote": "",
            "judge_quote_found": "", "judge_model": ""},
           {"anomaly_date": "2021-07-22", "document_id": "b", "claim": "new one",
            "human_label": "", "notes": "", "judge_verdict": "", "judge_quote": "",
            "judge_quote_found": "", "judge_model": ""}]
    merged = ac.merge_existing(new, old)
    assert merged[0]["human_label"] == "supported" and merged[0]["notes"] == "clear"
    assert merged[1]["human_label"] == ""


def test_sheet_rows_are_built_from_explained_runs_only(monkeypatch):
    monkeypatch.setattr(ac, "_cached_text", lambda url: "Body text about frost." if "a.example" in url else "")
    explained = {
        "anomaly": {"date": "2021-07-20", "pct_move": 6.7, "direction": "up"},
        "explanation_result": {"decision": "EXPLAINED", "citations": [
            {"document_id": "doc_aaa", "supports": "Frost lifted prices."},
            {"document_id": "bbb", "supports": "Supply fears grew."}]},
        "sources": [{"document_id": "aaa", "title": "Frost hits Brazil", "url": "https://a.example/1",
                     "publication_date": "2021-07-20"},
                    {"document_id": "bbb", "title": "Supply fears", "url": "https://b.example/1",
                     "publication_date": "2021-07-19"}],
        "faithfulness_report": {"per_citation": [
            {"document_id": "aaa", "flag": "ok", "support_score": 0.9,
             "support_rank": 1, "support_rank_of": 8}]},
    }
    refused = {"anomaly": {"date": "2021-07-30"},
               "explanation_result": {"decision": "INSUFFICIENT_EVIDENCE", "citations": []}}
    rows = ac.build_rows([explained, refused], index={})
    assert len(rows) == 2
    assert rows[0]["document_id"] == "aaa" and rows[0]["text_available"] == "body"
    assert rows[0]["price_move"] == "+6.7% (up)" and rows[0]["auto_support_rank"] == "1/8"
    assert rows[1]["text_available"] == "title_only"
    assert rows[1]["text_shown_to_model"] == "Supply fears"


def _labelled(human, auto="ok", judge="", text="body"):
    return {"human_label": human, "auto_flag": auto, "judge_verdict": judge,
            "text_available": text, "anomaly_date": "d", "claim": "c", "document_title": "t"}


def test_scoring_reports_support_rate_and_agreement():
    rows = [
        _labelled("supported", judge="supported"),
        _labelled("supported", judge="supported"),
        _labelled("Partial ", judge="partial"),                 # case/space tolerant
        _labelled("unsupported", judge="unverified"),           # judge caught it
        _labelled("unsupported", judge="supported", text="title_only"),   # judge missed it
        _labelled("", judge="supported"),                       # unlabelled: ignored
        _labelled("maybe"),                                     # unrecognised: ignored, reported
    ]
    s = ac.score_rows(rows)
    assert s["n_citations"] == 7 and s["n_labelled"] == 5
    assert s["supported"] == rate(2, 5) and s["unsupported"] == rate(2, 5)
    assert s["unrecognised_labels"] == ["maybe"]
    # the pipeline flag said "ok" for all five, and only two were supported
    assert s["auto_flag_ok_and_supported"] == rate(2, 5)
    assert s["unsupported_caught_by_auto_flag"] == rate(0, 2)
    assert s["judge"]["agreement"] == rate(4, 5)
    assert s["judge"]["unsupported_caught"] == rate(1, 2)
    assert s["judge"]["unverified_quotes"] == 1
    assert s["by_text_available"]["body"] == rate(2, 4)
    assert "Unsupported citations" in ac.report(s, rows)


def test_cohen_kappa():
    assert ac.cohen_kappa([("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]) == 1.0
    assert ac.cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
    assert ac.cohen_kappa([]) is None
    assert ac.cohen_kappa([("a", "a"), ("a", "a")]) is None     # no variation: undefined


def test_judge_downgrades_a_verdict_whose_quote_is_not_in_the_document(monkeypatch):
    import types
    from src.rag import explainer
    row = {"claim": "Frost lifted prices", "document_title": "Frost hits Brazil",
           "text_shown_to_model": "Arabica futures surged after frost hit Minas Gerais on Monday."}

    def answer(text):
        return lambda *a, **k: {"response": types.SimpleNamespace(text=text), "model": "judge-m"}

    monkeypatch.setattr(explainer, "_call_model_ex", answer(
        '{"verdict": "supported", "quote": "Arabica futures surged after frost hit Minas Gerais"}'))
    assert ac.judge_row(row)["judge_verdict"] == "supported"

    monkeypatch.setattr(explainer, "_call_model_ex", answer(
        '{"verdict": "supported", "quote": "coffee prices rose 20% because of the frost damage"}'))
    out = ac.judge_row(row)
    assert out["judge_verdict"] == "unverified" and out["judge_quote_found"] == "no"
