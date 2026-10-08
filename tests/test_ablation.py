"""Ablation study: every configuration must gate the same scored documents."""
import json

from src.evaluation import ablation_study as ab


def _docs(date):
    # Two strong semantic matches (one a syndicated copy), one keyword match,
    # one weak document, one dated after the anomaly.
    return [
        {"document_id": "a", "title": "Café dispara com geada", "url": "https://a.example/1",
         "publication_date": date, "retrieval_score": 0.0, "semantic_score": 0.88},
        {"document_id": "b", "title": "Café dispara com geada", "url": "https://b.example/1",
         "publication_date": date, "retrieval_score": 0.0, "semantic_score": 0.87},
        {"document_id": "c", "title": "Coffee frost drought supply", "url": "https://c.example/1",
         "publication_date": date, "retrieval_score": 0.55, "semantic_score": 0.80},
        {"document_id": "d", "title": "Food festival", "url": "https://d.example/1",
         "publication_date": date, "retrieval_score": 0.02, "semantic_score": 0.70},
        {"document_id": "e", "title": "Next day wrap", "url": "https://e.example/1",
         "publication_date": "2099-01-01", "retrieval_score": 0.0, "semantic_score": 0.90},
    ]


def test_all_gate_configurations_see_the_same_documents(tmp_path, monkeypatch):
    calls = []

    def fake_load(date, **kwargs):
        calls.append((date, tuple(sorted(kwargs.items()))))
        return _docs(date)

    monkeypatch.setattr(ab, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ab, "cached_dates", lambda: ["2021-07-20", "2021-07-30"])
    monkeypatch.setattr(ab, "load_scored_documents", fake_load)
    labels = {"2021-07-20": {"expected_outcome": "EXPLAIN"},
              "2021-07-30": {"expected_outcome": "REFUSE"}}
    monkeypatch.setattr(ab, "load_labels", lambda: labels)

    out = ab.run_ablation(verbose=False)
    s = out["summary"]

    # one baseline load per date, plus one per preparation configuration
    assert len(calls) == 2 * (1 + len(ab.PREP_CONFIGS))
    for config in ab.GATE_CONFIGS:
        assert s[config]["avg_retrieved"] == 5.0          # same input for every gate row

    assert s["full_pipeline"]["avg_accepted"] == 4.0      # a, b, c, e
    assert s["semantic_only"]["avg_accepted"] == 3.0      # a, b, e
    assert s["keyword_only"]["avg_accepted"] == 1.0       # c
    assert s["no_gate"]["avg_accepted"] == 5.0            # everything, including the weak one
    assert s["semantic_0.80"]["avg_accepted"] == 4.0
    assert s["semantic_0.90"]["avg_accepted"] == 2.0      # c (keyword) and e
    assert s["full_pipeline"]["avg_distinct_stories"] == 3.0   # a and b are one story
    assert s["full_pipeline"]["accepted_after_date"]["k"] == 2  # e, on both dates

    assert s["full_pipeline"]["passes_explain_labelled"]["n"] == 1
    assert s["full_pipeline"]["blocks_refuse_labelled"] == ab.rate(0, 1)

    saved = json.loads((tmp_path / "ablation_study.json").read_text(encoding="utf-8"))
    assert saved["like_for_like"] is True
    # keys the dashboard reads
    assert {"explain_rate", "avg_retrieved", "avg_accepted",
            "avg_best_score"} <= set(saved["summary"]["full_pipeline"])
    report = (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")
    assert "Every row gates the same scored documents" in report


def test_refuses_to_run_without_semantic_scores(tmp_path, monkeypatch):
    import pytest
    docs = [{k: v for k, v in d.items() if k != "semantic_score"} for d in _docs("2021-07-20")]
    monkeypatch.setattr(ab, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ab, "cached_dates", lambda: ["2021-07-20"])
    monkeypatch.setattr(ab, "load_scored_documents", lambda date, **k: docs)
    monkeypatch.setattr(ab, "load_labels", lambda: {})
    with pytest.raises(SystemExit):
        ab.run_ablation(verbose=False)
    out = ab.run_ablation(verbose=False, allow_keyword_only=True)
    assert out["semantic_scores_available"] is False
    assert "Semantic scores were not available" in \
        (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")


def test_findings_are_computed_from_the_decisions():
    results = []
    for date, full, kw in (("d1", "EXPLAIN", "INSUFFICIENT_EVIDENCE"), ("d2", "EXPLAIN", "EXPLAIN")):
        for config, decision in (("full_pipeline", full), ("keyword_only", kw),
                                 ("semantic_only", full), ("no_gate", full)):
            results.append({"date": date, "config": config, "gate_decision": decision,
                            "n_retrieved": 10, "n_accepted": 2, "n_distinct_stories": 2,
                            "n_accepted_after_date": 0, "best_score": 0.9})
    notes = ab.findings(ab.summarize(results, {}), results)
    assert any("Removing the semantic channel changes the gate decision on 1 of 2" in n for n in notes)
    assert any("Removing the keyword channel changes the gate decision on none" in n for n in notes)
