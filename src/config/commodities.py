"""
Per-commodity configuration for the anomaly detector, RAG retrieval, and
dashboard.

Added when the project expanded beyond coffee (the revised proposal's
single-commodity scope) to also cover crude oil and wheat, and later gold and
silver, per the original
proposal's "Build One Deeply, Architect for Any" idea (S4): coffee stays the
fully-validated reference commodity, and this module is the generic config
layer the other two plug into, rather than copy-pasted per-commodity
scripts.

Every event window and control period below is a real, cited historical
event, researched rather than guessed. Each control_period is treated as a
candidate until verified against the real fetched price series for that
commodity - see each config's control_period note.

Usage:
    from src.config.commodities import get_commodity, COMMODITIES
    cfg = get_commodity("crude_oil")
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

CONTRACT_SWITCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "labeling"


@dataclass
class CommodityConfig:
    key: str
    display_name: str
    ticker: str  # Yahoo Finance ticker
    gdelt_queries: list  # narrow, ANDed query strings - see pipeline.py's docstring on why
    # a single broad AND-of-many-terms query under-retrieves
    semantic_reference_query: str  # single broad query text embedded and compared
    # against every retrieved document for semantic scoring (src/rag/vector_store.py).
    # Unlike gdelt_queries above (kept narrow, for retrieval variety), this one is
    # broad: "what would explain an anomaly in this commodity"
    known_events: list  # list of (label, start_date, end_date, citation) tuples
    control_period: tuple  # (start_date, end_date, note)
    history_start_date: str  # earliest date to fetch from Yahoo Finance - must be
    # earlier than this commodity's earliest known_event/control_period start date
    # (plus a rolling-window buffer), or that window silently has 0 trading days
    # and any "MISSED"/"clean" result for it is meaningless, not a real test. A
    # single global 2018-01-01 start date is correct for coffee (earliest event
    # 2021) but gives wheat's 2010 and 2012 events, and both new commodities' 2017
    # control period, zero-row windows - which is why the start date lives per
    # commodity here.
    price_file: str  # filename under data/raw/
    anomalies_file: str  # filename under results/
    validation_md_file: str  # filename under results/
    # Queries sent over the last few days only (pipeline.RECENT_WINDOW_DAYS_BEFORE)
    # instead of the full retrieval window: searches for coverage of the move
    # itself, where the twenty results a query returns should all be recent.
    gdelt_recent_queries: list = field(default_factory=list)
    # The same searches as one request each. GDELT answers roughly one request
    # in five (see results/EVALUATION.md, section 9b), so the pipeline asks for
    # these two instead of every query above; the queries above are then only
    # read from the cache, where an earlier run already paid for them.
    #   gdelt_combined_query         over the full retrieval window
    #   gdelt_combined_recent_query  over the recent days only
    #   gdelt_combined_day_query     over the day of the move alone, so that the
    #                                day's market reports do not compete for a
    #                                request's 250 results with a week of
    #                                other coverage
    gdelt_combined_query: str = ""
    gdelt_combined_recent_query: str = ""
    gdelt_combined_day_query: str = ""
    # What the blind reading is told the watched prices are, and which prices
    # are not them, so that a report about another grade or a local price is
    # not taken as a report about this series (src/rag/blind_evidence.py).
    # Empty: a wording that fits any futures market is used.
    watched_market: str = ""
    other_markets: str = ""
    # File under data/labeling/ listing the days on which the price series
    # switched futures contract and so shows a move no contract made (see
    # src/modeling/anomaly_detector.py). Empty: none have been checked.
    contract_switch_file: str = ""


COMMODITIES = {
    "coffee": CommodityConfig(
        key="coffee",
        display_name="Coffee (Arabica)",
        ticker="KC=F",
        # Supply-shock topics, over the full window. These are stories about
        # prices rising: with only these, the evidence for a day prices FELL
        # started out against the move.
        gdelt_queries=[
            "coffee Brazil frost", "coffee Brazil drought", "coffee Minas Gerais",
            "coffee tariff", "coffee export Vietnam", "coffee supply shipping",
        ],
        # Not tied to a direction, over the recent days only: market coverage
        # either way, the daily market report, and the mirror image of frost
        # and drought (rain, a good harvest).
        gdelt_recent_queries=[
            "coffee prices arabica", "coffee futures", "coffee prices settle",
            "coffee harvest rain",
        ],
        gdelt_combined_query=(
            'coffee (frost OR drought OR "Minas Gerais" OR tariff OR Vietnam OR shipping)'),
        gdelt_combined_recent_query=(
            "coffee (prices OR futures OR arabica OR settle OR harvest OR rain)"),
        gdelt_combined_day_query=(
            "coffee (futures OR arabica OR robusta OR prices OR settle OR market)"),
        semantic_reference_query="coffee Brazil frost drought price disruption supply",
        known_events=[
            ("2021 Brazil drought+frost", "2021-06-01", "2021-08-15",
             "multiple news sources - see results/anomaly_detector_validation.md"),
            ("2024 Brazil drought+Typhoon Yagi", "2024-08-15", "2024-10-15",
             "see results/anomaly_detector_validation.md"),
            ("2025 sharp spike+reversal", "2025-08-15", "2025-09-20",
             "identified from price data itself, not a news date - see validation.md"),
            ("2026 frost concerns (16%+ single-day surge)", "2026-07-01", "2026-07-15",
             "cocoaintel.com, July 2026"),
        ],
        control_period=("2019-01-01", "2019-03-31",
                         "VERIFIED quiet in real data - 0/61 days flagged (see "
                         "results/anomaly_detector_validation.md)."),
        history_start_date="2018-01-01",  # earliest need: control period 2019-01-01
        price_file="coffee_prices.csv",
        anomalies_file="anomaly_detections.csv",
        watched_market="arabica coffee futures in New York",
        other_markets="robusta coffee, or prices paid locally in one country",
        contract_switch_file="contract_switch_dates.csv",
        validation_md_file="anomaly_detector_validation.md",
    ),
    "crude_oil": CommodityConfig(
        key="crude_oil",
        display_name="Crude Oil (WTI)",
        ticker="CL=F",
        gdelt_queries=["oil price OPEC", "oil price Russia Ukraine", "crude oil supply"],
        gdelt_recent_queries=["oil prices", "crude futures", "oil prices settle"],
        gdelt_combined_query=(
            'oil (OPEC OR Russia OR sanctions OR inventories OR Iran OR "supply cut")'),
        gdelt_combined_recent_query=(
            "oil (prices OR futures OR crude OR Brent OR WTI OR settle)"),
        gdelt_combined_day_query=(
            "crude (futures OR Brent OR WTI OR prices OR settle OR barrel)"),
        semantic_reference_query="crude oil price OPEC supply disruption geopolitical shock",
        known_events=[
            ("2014 OPEC no-cut decision", "2014-11-24", "2014-12-05",
             "OPEC declined to cut output at its Nov 27, 2014 meeting; oil fell to a "
             "4-year low - Washington Post ('OPEC decides not to cut oil production, "
             "sending crude prices to 4-year low', Nov 27 2014), CNBC"),
            ("2020 Saudi-Russia price war", "2020-03-06", "2020-03-13",
             "OPEC+ talks collapsed Mar 6 2020; Saudi Aramco announced discounts/output "
             "hike; WTI fell ~25% on Mar 9 2020 ('Black Monday') - Wikipedia '2020 "
             "Russia-Saudi Arabia oil price war', CNBC, Forbes"),
            ("2020 negative WTI price", "2020-04-15", "2020-04-22",
             "WTI May futures settled at -$37.63/barrel on Apr 20 2020 amid a "
             "storage-capacity crunch - EIA, CFTC interim report, Forbes"),
            ("2022 Russia invades Ukraine", "2022-02-23", "2022-03-09",
             "Russia invaded Ukraine Feb 24 2022; oil surged past $100/barrel and kept "
             "rising into early March - EIA, Yahoo Finance"),
            ("2024 Red Sea shipping attacks", "2024-01-01", "2024-02-29",
             "Added after the cumulative-return check (see anomaly_detector.py) flagged a "
             "gradual price rise here that a first choice of control period (2024 Q1) had "
             "missed. Cause: Houthi attacks on Red Sea oil tankers forced longer shipping "
             "routes and pushed prices up through Jan-Feb 2024 - EIA (cited in "
             "databoks.katadata.co.id, 'Global Oil Prices Creep Upward in Early 2024'): "
             "Brent +4.4% and WTI +3.8% month-on-month in Feb 2024. A slow grind, not a "
             "single-day shock - no individual day's z-score cleared 1.7, which is the "
             "failure mode the cumulative check exists to catch."),
        ],
        control_period=("2023-07-01", "2023-09-30",
                         "Verified quiet in real data - 0/63 days flagged under both checks "
                         "(shock + cumulative). Two earlier candidates were rejected against "
                         "real data: 2017-01-01/2017-03-31 (contained a -5.4% "
                         "EIA-inventory-driven shock on 2017-03-08), and 2024-01-01/"
                         "2024-03-31 (contained the gradual 2024 Red Sea-driven rise above, "
                         "which the cumulative check flagged and which was moved to its own "
                         "known_event rather than left as a mislabeled control-period false "
                         "positive)."),
        history_start_date="2013-01-01",  # earliest need: 2014-11-24 OPEC event,
        # plus ~1 year of buffer so the 75-day rolling window has real history by then
        price_file="crude_oil_prices.csv",
        anomalies_file="crude_oil_anomaly_detections.csv",
        watched_market="WTI crude oil futures in New York",
        other_markets="petrol or diesel prices at the pump, or the shares of oil companies",
        validation_md_file="crude_oil_anomaly_detector_validation.md",
    ),
    "wheat": CommodityConfig(
        key="wheat",
        display_name="Wheat (Chicago SRW)",
        ticker="ZW=F",
        gdelt_queries=["wheat export ban Russia", "wheat drought", "wheat Ukraine war"],
        gdelt_recent_queries=["wheat prices", "wheat futures", "wheat prices settle"],
        gdelt_combined_query=(
            'wheat (drought OR "export ban" OR Russia OR Ukraine OR harvest OR tariff)'),
        gdelt_combined_recent_query=(
            "wheat (prices OR futures OR CBOT OR Chicago OR settle OR bushel)"),
        gdelt_combined_day_query=(
            "wheat (futures OR CBOT OR Chicago OR prices OR settle OR market)"),
        semantic_reference_query="wheat export ban drought war price disruption supply",
        known_events=[
            ("2010 Russia export ban", "2010-08-03", "2010-08-16",
             "Russia announced a wheat export ban Aug 5 2010 (effective Aug 15) after a "
             "severe drought - France24 ('Ban on wheat exports sends global prices "
             "skyrocketing'), NPR, Aug 2010"),
            ("2012 US drought", "2012-06-20", "2012-08-10",
             "Worst US drought since 1956. Window widened from an initial 2012-07-15 start "
             "(drawn from two CNN article dates) to capture the earlier onset of the rally - "
             "the S&P GSCI grains rally 'since mid-June' "
             "(Business Standard, 'Commodities enter bull market after drought hits crops', "
             "Aug 23 2012) and corn/soybean prices 'rallied sharply beginning in July 2012... "
             "as drought conditions unfolded', peaking Aug 10 2012 (farmdocdaily.illinois.edu, "
             "May 2013) - CNN ('Corn, soybean prices shoot up as drought worsens', Jul 19 2012; "
             "'U.S. drought drives up food prices worldwide', Aug 9 2012)"),
            ("2022 Russia invades Ukraine", "2022-02-23", "2022-03-09",
             "Chicago wheat futures hit a record high in early March 2022 after the "
             "Feb 24 invasion - farmpolicynews.illinois.edu, AgFax, Mar 2022"),
            ("2023 Black Sea grain deal collapse", "2023-07-17", "2023-07-28",
             "Russia terminated the Black Sea Grain Initiative ~Jul 17 2023; wheat "
             "prices jumped in the following days - CNN, World Economic Forum, Jul 2023"),
        ],
        control_period=("2022-04-01", "2022-06-30",
                         "Verified quiet in real data - 0/62 days flagged (the only fully "
                         "clean calendar quarter found across all quarters from 2009-2025 for "
                         "wheat, which is a more volatile series than coffee - most quarters "
                         "have 1-8 flagged days). An earlier candidate (2017-01-01 to "
                         "2017-03-31) was rejected after testing against the real fetched "
                         "price series showed it was not quiet: it contained a one-day jump on "
                         "2017-03-15 (+5.8%, z=3.54, the quarter's most extreme day)."),
        history_start_date="2009-01-01",  # earliest need: 2010-08-03 export-ban
        # event, plus ~1 year of buffer so the 75-day rolling window has real history by then
        price_file="wheat_prices.csv",
        anomalies_file="wheat_anomaly_detections.csv",
        watched_market="Chicago wheat futures",
        other_markets="the price of flour or bread, or wheat prices paid locally in one country",
        validation_md_file="wheat_anomaly_detector_validation.md",
    ),
    "gold": CommodityConfig(
        key="gold",
        display_name="Gold (COMEX)",
        ticker="GC=F",
        # Gold moves on interest rates, the dollar and fear, not on a harvest:
        # the topics are the ones a gold market report gives as its reason.
        gdelt_queries=["gold price Federal Reserve", "gold price dollar", "gold safe haven"],
        gdelt_recent_queries=["gold prices", "gold futures", "gold prices settle"],
        gdelt_combined_query=(
            'gold (Fed OR "interest rates" OR dollar OR inflation OR "safe haven" OR "central bank")'),
        gdelt_combined_recent_query=(
            "gold (prices OR futures OR bullion OR settle OR ounce OR spot)"),
        gdelt_combined_day_query=(
            "gold (futures OR bullion OR prices OR settle OR spot OR market)"),
        semantic_reference_query="gold price Federal Reserve dollar safe haven bullion selloff rally",
        known_events=[
            ("2013 April gold crash", "2013-04-12", "2013-04-17",
             "Gold fell about 9% on Apr 15 2013, its largest one-day fall in 30 years - "
             "Bloomberg ('Gold's Worst Plunge Since 1983 Ends 12-Year Bull Run: Timeline', "
             "Apr 17 2013), CoinNews ('Gold Dives 9.3%, Silver Tumbles 11.3%', Apr 15 2013)"),
            ("2016 Brexit referendum", "2016-06-23", "2016-06-28",
             "The UK voted to leave the EU on Jun 23 2016; gold rose close to 5% on "
             "Jun 24 as investors moved to safe havens - widely reported, e.g. Reuters "
             "and CNBC market reports of Jun 24 2016"),
            ("2020 pandemic dash for cash", "2020-03-09", "2020-03-27",
             "Gold was sold with everything else as investors raised cash in the "
             "second week of March 2020, then rebounded sharply from Mar 23-24 after "
             "the Federal Reserve announced unlimited asset purchases - Reuters and "
             "CNBC precious-metals reports, March 2020"),
            ("2020 August reversal from the record", "2020-08-07", "2020-08-13",
             "After a record above $2,000 an ounce, gold fell about 4.6% on Aug 11 "
             "2020, its largest one-day fall in seven years, as bond yields rose - "
             "CNBC and Reuters market reports of Aug 11 2020"),
            ("2025 October reversal from the record", "2025-10-20", "2025-10-23",
             "Gold fell more than 5% on Oct 21 2025, its steepest one-day fall since "
             "2013 - Axios ('Gold and silver plunge after recent surge to record "
             "highs', Oct 21 2025), InvestmentNews ('Gold, silver post steepest "
             "declines in over 13 years as rally stalls')"),
            ("2026 January metals crash", "2026-01-28", "2026-02-04",
             "Gold fell about 11% on Jan 30 2026 after a parabolic run to a record; "
             "reports tied the selling to the news about the next Federal Reserve "
             "chair, a stronger dollar and forced liquidation - Morningstar ('Why Did "
             "Gold and Silver Prices Plunge?'), Seeking Alpha ('Gold And Silver's "
             "Most Volatile Day')"),
        ],
        control_period=("2026-04-01", "2026-06-30",
                         "Verified quiet in real data - 0 days flagged under both checks, the "
                         "only fully clean calendar quarter from 2012 to 2026 for gold. It "
                         "follows the January-March 2026 turmoil, so 'quiet' here is relative "
                         "to a trailing window that had just absorbed very large moves; the "
                         "next-best quarters (2023 Q2, 2025 Q1) each have one flagged day."),
        history_start_date="2012-01-01",  # earliest need: the April 2013 crash, plus
        # more than a year so the 75-day rolling window has real history by then
        price_file="gold_prices.csv",
        anomalies_file="gold_anomaly_detections.csv",
        watched_market="gold futures in New York (COMEX)",
        other_markets=("silver or other metals, jewellery or retail gold prices in one "
                       "country, or the shares of gold miners"),
        validation_md_file="gold_anomaly_detector_validation.md",
    ),
    "silver": CommodityConfig(
        key="silver",
        display_name="Silver (COMEX)",
        ticker="SI=F",
        gdelt_queries=["silver price Federal Reserve", "silver price dollar",
                       "silver industrial demand"],
        gdelt_recent_queries=["silver prices", "silver futures", "silver prices settle"],
        gdelt_combined_query=(
            'silver (Fed OR "interest rates" OR dollar OR margin OR "industrial demand" OR squeeze)'),
        gdelt_combined_recent_query=(
            "silver (prices OR futures OR bullion OR settle OR ounce OR spot)"),
        gdelt_combined_day_query=(
            "silver (futures OR bullion OR prices OR settle OR spot OR market)"),
        semantic_reference_query="silver price Federal Reserve dollar margin bullion selloff rally",
        known_events=[
            ("2011 May margin-hike collapse", "2011-05-02", "2011-05-06",
             "Silver fell by more than a quarter in a week after the CME raised "
             "margin requirements several times - The Motley Fool ('Did Higher "
             "Margins Kill Commodities?', May 11 2011), InvestmentNews ('Silver "
             "investors holding a falling knife')"),
            ("2011 September plunge", "2011-09-22", "2011-09-26",
             "Silver fell about 18% on Sep 23 2011 as the CME raised margins again - "
             "tradingnrg.com ('Gold and Silver Plummeted | CME Margin Hike Fueled "
             "Drop', Sep 23 2011)"),
            ("2013 April metals crash", "2013-04-12", "2013-04-17",
             "Silver fell about 11% on Apr 15 2013 alongside gold's largest fall in 30 "
             "years - CoinNews ('Gold Dives 9.3%, Silver Tumbles 11.3%; Both at "
             "Multi-Year Lows', Apr 15 2013)"),
            ("2020 pandemic dash for cash", "2020-03-09", "2020-03-27",
             "Silver fell to an 11-year low in mid-March 2020 as investors raised "
             "cash, then rebounded - Reuters precious-metals reports, March 2020"),
            ("2021 retail silver squeeze", "2021-01-28", "2021-02-03",
             "Silver jumped to an eight-year high on Feb 1 2021 on a retail buying "
             "campaign, then fell about 10% on Feb 2 after the CME raised margins by "
             "17.9% - Reuters ('Silver slides from 8-year peak as retail frenzy "
             "fizzles out'), Bloomberg ('Silver's Selloff Accelerates With Reddit "
             "Frenzy Unraveling', Feb 2 2021)"),
            ("2026 January metals crash", "2026-01-26", "2026-02-04",
             "Silver fell about 30% on Jan 30 2026, its largest one-day fall on "
             "record, after a parabolic run - Morningstar ('Why Did Gold and Silver "
             "Prices Plunge?'), SD Bullion ('Biggest Drop in Intraday Price History')"),
        ],
        control_period=("2026-04-01", "2026-06-30",
                         "Verified quiet in real data - 0 days flagged under both checks, the "
                         "only fully clean calendar quarter from 2010 to 2026 for silver. As "
                         "for gold it follows the January-March 2026 turmoil; the next-best "
                         "quarters (2012 Q1, 2017 Q1, 2021 Q2, 2024 Q1) each have one flagged "
                         "day."),
        history_start_date="2010-01-01",  # earliest need: the May 2011 collapse, plus
        # more than a year so the 75-day rolling window has real history by then
        price_file="silver_prices.csv",
        anomalies_file="silver_anomaly_detections.csv",
        watched_market="silver futures in New York (COMEX)",
        other_markets=("gold or other metals, retail silver coins and bars, or the "
                       "shares of silver miners"),
        validation_md_file="silver_anomaly_detector_validation.md",
    ),
}


def get_commodity(key: str) -> CommodityConfig:
    if key not in COMMODITIES:
        raise ValueError(f"Unknown commodity '{key}'. Available: {list(COMMODITIES.keys())}")
    return COMMODITIES[key]


def load_contract_switches(key: str = "coffee") -> dict:
    """{date: the move the contract in active trading made that day, as a
    fraction} for the days on which a commodity's price series switched
    futures contract. Only rows marked EXCLUDE count. Empty when the commodity
    has no list. See src/modeling/anomaly_detector.py for what is done with it."""
    name = get_commodity(key).contract_switch_file
    path = CONTRACT_SWITCH_DIR / name if name else None
    if path is None or not path.exists():
        return {}
    switches = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("verdict") or "").strip().upper() != "EXCLUDE":
                continue
            try:
                switches[row["date"].strip()] = float(row["reported_pct"]) / 100.0
            except (KeyError, TypeError, ValueError):
                continue
    return switches
