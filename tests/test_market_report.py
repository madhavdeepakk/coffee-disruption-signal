"""Choosing what the blind reading is shown. The headlines below are real ones
from the first run on the labelled dates (pipeline version 8), where the
reading was shown the wrong documents."""
from src.rag import market_report as mr


def _doc(i, title, age=0, sem=0.82):
    return {"document_id": f"d{i}", "title": title, "age": age, "semantic_score": sem}


REUTERS = "SOFTS - Coffee prices jump nearly 7 % in New York after stocks drawn"
BLOOMBERG = "Coffee tumbles most since 2008 as Brazil frost concerns ease"
JUNK = ["The Sprudge Guide To Coffee In Kyoto For 2024 - sprudge.com",
        "Lavazza unveils new ¡Tierra! Organic Coffee collection - BeanScene",
        "Spinelli Coffee reportedly ceasing operations in Singapore - World Coffee Portal",
        "Project Café East Asia 2024 - World Coffee Portal",
        "Is there value in new water products designed specifically for coffee? - perfectdailygrind.com",
        "Farmer Boys To Test Specialty Coffee - RestaurantNews.com"]


def test_market_reports_score_above_coffee_shop_news():
    score = lambda title: mr.market_report_score({"title": title})  # noqa: E731
    assert score(REUTERS) >= 6
    assert score(BLOOMBERG) >= mr.REPORT_SCORE
    assert all(score(t) < mr.REPORT_SCORE for t in JUNK)
    # other languages the retrieved documents were in
    for title in ("Café e açúcar recuam em meio à venda generalizada de petróleo e ações",
                  "Commodities agrícolas têm forte queda em Nova York",
                  "Giá cà phê hôm nay 1 / 12 : Đồng loạt tăng mạnh , Robusta thêm gần trăm USD / tấn",
                  "Café fecha em forte alta com problemas de embarques e frio no Brasil",
                  "Les prix à terme du café Arabica affichent la plus forte hausse depuis 10 ans"):
        assert score(title) >= mr.REPORT_SCORE, title
    assert score("") == 0 and mr.market_report_score({}) == 0


def test_the_score_does_not_know_which_way_prices_moved():
    pairs = [("Coffee prices jump 7% in New York", "Coffee prices plunge 7% in New York"),
             ("Café sobe 5% na bolsa de Nova York", "Café cai 5% na bolsa de Nova York"),
             ("Giá cà phê tăng mạnh", "Giá cà phê giảm mạnh"),
             ("Arabica futures settle higher", "Arabica futures settle lower")]
    for up, down in pairs:
        assert mr.market_report_score({"title": up}) == mr.market_report_score({"title": down})


def test_the_report_that_was_left_unread_now_comes_first():
    # 2023-11-30: these were all candidates from the day; the first twelve by
    # the old ordering (similarity to a supply-shock phrase) left Reuters out.
    docs = [_doc(i, t, sem=0.83 + i / 1000) for i, t in enumerate(JUNK)] + [_doc(99, REUTERS, sem=0.816)]
    order = mr.reading_order(docs, lambda d: d["age"], lambda d: d["semantic_score"])
    assert order[0]["title"] == REUTERS


def test_yesterdays_market_report_is_read_before_todays_shop_news_but_after_todays_reports():
    today_report = _doc(1, "Coffee prices settle lower on rain forecasts for Brazil", age=0)
    yesterday_report = _doc(2, "Arabica futures surge 5% on frost fears", age=1)
    today_junk = _doc(3, JUNK[0], age=0, sem=0.95)
    order = mr.reading_order([today_junk, yesterday_report, today_report],
                             lambda d: d["age"], lambda d: d["semantic_score"])
    assert [d["document_id"] for d in order] == ["d1", "d2", "d3"]
    # the same documents in any arrival order come out the same
    again = mr.reading_order([yesterday_report, today_report, today_junk],
                             lambda d: d["age"], lambda d: d["semantic_score"])
    assert [d["document_id"] for d in again] == ["d1", "d2", "d3"]


def test_a_document_seen_before_the_exchange_opens_belongs_to_the_session_before():
    # 2023-11-30: the report of the 29th's close was first seen at 02:00 UTC on
    # the 30th and counted against the 30th's rise.
    assert mr.session_date("20231130T020000Z") == "2023-11-29"
    assert mr.session_date("20231130T074500Z") == "2023-11-29"
    assert mr.session_date("20231130T080000Z") == "2023-11-30"
    assert mr.session_date("20231130T204500Z") == "2023-11-30"      # the Reuters closing report
    assert mr.session_date("20240101T030000Z") == "2023-12-31"      # across a year end
    assert mr.session_date("") is None and mr.session_date("not a time") is None
    assert mr.session_date(None) is None


SOFTS_REPORT = (
    "Raw sugar futures on ICE climbed to their highest in nearly six years on Tuesday as concerns "
    "over tight supplies persisted, while arabica coffee also gained. "
    + "Dealers said the sugar market remained focused on the pace of the harvest in India. " * 7
    + "March arabica coffee rose 5% to $1.7905 per lb. Dealers said the market was driven by a rise "
      "in some Brazilian physical coffee premiums to their highest in a decade. "
    + "London cocoa was little changed. " * 10)


def test_the_excerpt_reaches_the_passage_about_coffee():
    assert "premiums" not in SOFTS_REPORT[:mr.OPENING_CHARS]    # what version 8 showed the model
    focused = mr.focus_text(SOFTS_REPORT)
    assert "premiums" in focused[:mr.OPENING_CHARS]             # though the opening already names coffee
    first_sentence = SOFTS_REPORT.split(". ")[0] + "."
    assert focused.startswith(first_sentence)                   # the opening sentence is kept whole
    # the sentence with the move and the one after it, which has the reason, come up together
    assert "rose 5% to $1.7905 per lb. Dealers said the market was driven by" in focused
    # every piece is copied exactly, so a quote from it is a quote from the document
    for piece in focused.split(mr.SEPARATOR):
        assert piece in SOFTS_REPORT


def test_an_article_about_coffee_is_never_rearranged():
    # shaped like the report of 2024-10-07: the sentence with the move and the
    # one with the reason do not use the word coffee
    article = ("International coffee prices remain driven by crop conditions in Brazil, the largest "
               "arabica exporter. The contract ended Monday in New York down 4.93%, the lowest since "
               "September 6. Many factors led to the fall, the most prominent being the prospect of "
               "rain in producing areas. " + "Regions such as Mogiana were badly hit by the drought. " * 20
               + "Brazil is expected to export more than 45 million bags of coffee, up 10%. ")
    assert mr.focus_text(article) == article
    dense = ("Arabica coffee futures surged 6% on Tuesday after a severe frost hit Minas Gerais. "
             "The frost was the worst since 1994, growers said. ") * 20
    assert mr.focus_text(dense) == dense
    # a round-up with nothing more about coffee further down is left alone too
    sugar_only = SOFTS_REPORT.split("March arabica coffee rose")[0] + "London cocoa was little changed. " * 10
    assert mr.focus_text(sugar_only) == sugar_only
    assert mr.focus_text("") == "" and mr.focus_text(None) == ""
    assert mr.focus_text("Coffee rose.") == "Coffee rose."


def test_places_are_kept_for_the_days_event_stories():
    reports = [_doc(i, f"Arabica futures surge {i}% in New York", age=1, sem=0.9) for i in range(20)]
    frost = _doc(100, "Frosts stain Brazil coffee belt, growers see a third of fields hit", age=0, sem=0.88)
    shops = [_doc(200 + i, f"New coffee shop opens in town {i}", age=0, sem=0.81) for i in range(5)]
    order = mr.reading_order(reports + shops + [frost], lambda d: d["age"],
                             lambda d: d["semantic_score"], limit=12)
    head = order[:12]
    assert sum(1 for d in head if d["age"] == 0) == mr.KEPT_FOR_THE_DAY
    assert frost in head                                        # the most relevant of the day's others
    assert len(order) == 26 and len({d["document_id"] for d in order}) == 26   # nothing lost or doubled
    # with room to spare nothing needs keeping: reports first, then the rest
    few = mr.reading_order(reports[:3] + [frost], lambda d: d["age"], lambda d: d["semantic_score"], limit=12)
    assert [d["document_id"] for d in few] == ["d0", "d1", "d2", "d100"]


def test_shop_prices_are_not_market_reports():
    score = lambda title: mr.market_report_score({"title": title})  # noqa: E731
    assert score("U.S. Coffee Prices Soar at the Supermarket") < mr.REPORT_SCORE
    assert score("Coffee prices spike more than 20% in latest consumer report") < mr.REPORT_SCORE
    assert score("Coffee Prices Surge in Greece: Cappuccino Freddo Now Costs Up to 5.50 a cup") < mr.REPORT_SCORE
    # the exchange in the headline keeps it a market report whatever else it says
    assert score("Arabica futures jump 5% as consumers brace for higher prices") >= mr.REPORT_SCORE


def test_headlines_in_arabic_korean_and_turkish_are_scored_either_way():
    rising = {"title": "العقود الآجلة للقهوة تواصل الارتفاع مع تقلبات عنيفة للطقس"}
    falling = {"title": "العقود الآجلة للقهوة تواصل الانخفاض مع تحسن الطقس"}
    assert mr.market_report_score(rising) >= mr.REPORT_SCORE
    assert mr.market_report_score(rising) == mr.market_report_score(falling)   # no direction favoured
    assert mr.market_report_score({"title": "أسعار القهوة ترتفع 5 %"}) == \
        mr.market_report_score({"title": "أسعار القهوة تنخفض 5 %"}) >= mr.REPORT_SCORE
    assert mr.market_report_score({"title": "커피 선물 가격 급등"}) == \
        mr.market_report_score({"title": "커피 선물 가격 급락"}) >= mr.REPORT_SCORE
    assert mr.market_report_score({"title": "Kahve fiyatları yükseldi"}) == \
        mr.market_report_score({"title": "Kahve fiyatları geriledi"}) >= mr.REPORT_SCORE
    # a bank is not coffee, and a story with no price in it is not a report
    assert mr.market_report_score({"title": "البنك المركزي يرفع الفائدة"}) < mr.REPORT_SCORE
    assert mr.market_report_score({"title": "커피 전문점 새 메뉴 출시"}) < mr.REPORT_SCORE
    assert mr.COFFEE.search("للقهوة") and mr.COFFEE.search("커피")
