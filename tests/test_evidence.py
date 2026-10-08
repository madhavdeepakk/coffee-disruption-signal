"""Evidence preparation: window filter, syndicated-copy collapsing, packing."""
from src.rag import evidence as ev


def _doc(i, title, date="2024-09-20", text="", url=None, **extra):
    return {"document_id": f"d{i}", "title": title, "publication_date": date,
            "text": text, "url": url or f"https://site{i}.example/a", **extra}


# --- window filter ---------------------------------------------------------

def test_filter_drops_documents_published_after_the_anomaly():
    docs = [_doc(1, "a", "2022-05-11"), _doc(2, "b", "2022-05-12"), _doc(3, "c", "2022-05-03")]
    kept, dropped = ev.filter_to_window(docs, "2022-05-11", days_before=10)
    assert [d["document_id"] for d in kept] == ["d1", "d3"]
    assert dropped == [{"document_id": "d2", "publication_date": "2022-05-12",
                        "reason": "published_after_anomaly"}]


def test_filter_drops_documents_older_than_the_window():
    docs = [_doc(1, "a", "2022-04-30"), _doc(2, "b", "2022-05-01")]
    kept, dropped = ev.filter_to_window(docs, "2022-05-11", days_before=10)
    assert [d["document_id"] for d in kept] == ["d2"]
    assert dropped[0]["reason"] == "published_before_window"


def test_filter_keeps_but_marks_undated_documents():
    docs = [_doc(1, "a", "")]
    kept, dropped = ev.filter_to_window(docs, "2022-05-11", days_before=10)
    assert len(kept) == 1 and dropped == []
    assert kept[0]["publication_date_unknown"] is True


def test_days_before_anomaly():
    assert ev.days_before_anomaly("2023-09-14", "2023-09-20") == 6
    assert ev.days_before_anomaly("2023-09-20", "2023-09-20") == 0
    assert ev.days_before_anomaly("2023-09-21", "2023-09-20") == -1
    assert ev.days_before_anomaly("", "2023-09-20") is None


def test_recency_summary_counts_fresh_documents():
    docs = [{"days_before_anomaly": 0}, {"days_before_anomaly": 2},
            {"days_before_anomaly": 6}, {"days_before_anomaly": None}]
    s = ev.recency_summary(docs, fresh_within_days=2)
    assert s["n_dated"] == 3 and s["n_fresh"] == 2
    assert s["newest_days_before"] == 0 and s["median_days_before"] == 2


# --- syndicated copies -----------------------------------------------------

def test_same_headline_on_different_sites_collapses_to_one():
    docs = [
        _doc(1, "Brazil drought punishes coffee farms and threatens to push prices even higher"),
        _doc(2, "Brazil drought punishes coffee farms and threatens to push prices even higher - Yahoo"),
        _doc(3, "BRAZIL DROUGHT PUNISHES COFFEE FARMS, AND THREATENS TO PUSH PRICES EVEN HIGHER"),
        _doc(4, "Coffee prices post moderate losses on rain forecasts"),
    ]
    out = ev.collapse_syndicated(docs)
    assert len(out) == 2
    assert out[0]["n_copies"] == 3 and out[1]["n_copies"] == 1
    assert len(out[0]["duplicate_urls"]) == 2
    assert len(out[0]["syndication_domains"]) == 3


def test_opposite_headlines_are_not_collapsed():
    # Near-identical strings that say opposite things must stay separate.
    docs = [_doc(1, "Coffee Prices Settle Higher on Brazil Weather"),
            _doc(2, "Coffee Prices Settle Lower on Brazil Weather")]
    assert len(ev.collapse_syndicated(docs)) == 2


def test_same_body_under_different_headlines_collapses():
    body = ("Arabica coffee futures surged on Monday after a severe drought in Brazil "
            "damaged crops across Minas Gerais, traders said, with prices reaching their "
            "highest level in more than a decade as supply concerns mounted.")
    docs = [_doc(1, "Coffee soars as drought hits Brazil", text=body),
            _doc(2, "Drought sends coffee to decade high", text=body + " More follows."),
            _doc(3, "Unrelated story", text="A completely different article about "
                                             "shipping container shortages at ports worldwide.")]
    out = ev.collapse_syndicated(docs)
    assert [d["n_copies"] for d in out] == [2, 1]


def test_representative_prefers_the_copy_with_text():
    docs = [_doc(1, "Same wire story here"),
            _doc(2, "Same wire story here", text="has the body text")]
    out = ev.collapse_syndicated(docs)
    assert len(out) == 1 and out[0]["document_id"] == "d2"


def test_collapsing_twice_keeps_the_copy_count():
    docs = [_doc(1, "Same wire story here"), _doc(2, "Same wire story here"),
            _doc(3, "A different story entirely")]
    once = ev.collapse_syndicated(docs, use_text=False)
    twice = ev.collapse_syndicated(once, use_text=True)
    assert [d["n_copies"] for d in twice] == [2, 1]


# --- packing ---------------------------------------------------------------

def _long_docs(n, chars=4000):
    return [{"document_id": f"d{i}", "title": f"title {i}", "url": f"https://s{i}.example/x",
             "text": "word " * (chars // 5)} for i in range(n)]


def test_no_budget_sends_everything_at_full_excerpt():
    docs = _long_docs(25)
    packed, rep = ev.pack_evidence(docs, None)
    assert len(packed) == 25
    assert rep["excerpt_chars"] == ev.EXCERPT_STEPS[0] and rep["truncated"] is False


def test_excerpt_is_shortened_before_any_document_is_dropped():
    docs = _long_docs(20)
    budget = 20 * 900 // 3  # too small for 1500 chars each, fine for a shorter excerpt
    packed, rep = ev.pack_evidence(docs, budget)
    assert len(packed) == 20, "should keep every document and shorten the excerpt instead"
    assert rep["excerpt_chars"] < ev.EXCERPT_STEPS[0]
    assert rep["dropped_document_ids"] == [] and rep["truncated"] is True
    assert rep["estimated_tokens"] <= budget


def test_documents_are_dropped_from_the_end_when_shortening_is_not_enough():
    docs = _long_docs(40)
    packed, rep = ev.pack_evidence(docs, 1500)
    assert 3 <= len(packed) < 40
    assert [d["document_id"] for d in packed] == [f"d{i}" for i in range(len(packed))]
    assert rep["dropped_document_ids"][0] == "d39"  # lowest priority goes first
    assert rep["estimated_tokens"] <= 1500


def test_packing_never_goes_below_the_minimum():
    docs = _long_docs(5)
    packed, rep = ev.pack_evidence(docs, 10, min_documents=3)
    assert len(packed) == 3


def test_titles_in_other_scripts_are_not_merged_on_a_shared_number():
    # A Latin-only word pattern reduced both of these to "2024".
    docs = [_doc(1, "В 2024 году кофе подорожал из-за засухи в Бразилии"),
            _doc(2, "Цены на пшеницу в 2024 году упали после рекордного урожая")]
    assert len(ev.collapse_syndicated(docs)) == 2


def test_a_short_title_is_not_enough_to_call_two_documents_the_same():
    docs = [_doc(1, "Coffee prices"), _doc(2, "Coffee prices")]
    assert len(ev.collapse_syndicated(docs)) == 2
    assert ev.title_key("Coffee prices") == ""


def test_doubled_publisher_suffix_still_matches_the_plain_headline():
    docs = [_doc(1, "Brazil Drought Punishes Coffee Farms - AP News - AP News"),
            _doc(2, "Brazil drought punishes coffee farms"),
            _doc(3, "Brazil drought punishes coffee farms - AP News")]
    out = ev.collapse_syndicated(docs)
    assert len(out) == 1 and out[0]["n_copies"] == 3


def test_token_estimate_counts_non_latin_text_more_heavily():
    latin = "coffee prices rose sharply today"          # 32 characters
    cyrillic = "кофе подорожал на десять процент"        # 32 characters
    assert ev.estimate_tokens(latin) < 15
    assert ev.estimate_tokens(cyrillic) > 2 * ev.estimate_tokens(latin)
    assert ev.estimate_tokens("café arábica dispara") < 10   # accented Latin is still Latin


def test_identical_text_on_the_same_site_is_boilerplate_not_syndication():
    # e.g. a consent page or redirect stub returned for every link on one host
    stub = ("Before you continue to the site please review how cookies and data are used "
            "to deliver and maintain services and to measure audience engagement today.")
    docs = [_doc(1, "Coffee prices jump on frost fears in Brazil", text=stub,
                 url="https://news.example/articles/1"),
            _doc(2, "Vietnam robusta exports fall for a third month", text=stub,
                 url="https://news.example/articles/2"),
            _doc(3, "Colombia growers warn of a smaller harvest", text=stub,
                 url="https://other.example/a")]
    out = ev.collapse_syndicated(docs)
    # 1 and 2 share a site, so they stay apart; 3 is the same text elsewhere and folds into 1
    assert [d["document_id"] for d in out] == ["d1", "d2"]
    assert out[0]["n_copies"] == 2 and out[1]["n_copies"] == 1


def test_trading_days_ignore_the_weekend():
    assert ev.trading_days_before("2024-09-20", "2024-09-23") == 1    # Fri -> Mon
    assert ev.trading_days_before("2024-09-21", "2024-09-23") == 1    # Sat -> Mon
    assert ev.trading_days_before("2024-09-23", "2024-09-23") == 0
    assert ev.trading_days_before("2024-09-16", "2024-09-23") == 5    # Mon -> Mon
    assert ev.trading_days_before("2024-09-24", "2024-09-23") == -1
    assert ev.trading_days_before("", "2024-09-23") is None


def test_recency_counts_fridays_news_as_fresh_for_a_monday_move():
    docs = ev.annotate_recency([{"publication_date": "2024-09-20"},
                                {"publication_date": "2024-09-13"}], "2024-09-23")
    assert docs[0]["days_before_anomaly"] == 3 and docs[0]["trading_days_before_anomaly"] == 1
    s = ev.recency_summary(docs, fresh_within_days=2)
    assert s["n_fresh"] == 1 and s["newest_days_before"] == 1
