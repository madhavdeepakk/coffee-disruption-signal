"""
Per-commodity configuration for the anomaly detector, RAG retrieval, and
dashboard.

Added when the project expanded beyond coffee (the revised proposal's
single-commodity scope) to also cover crude oil and wheat, per the original
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
        validation_md_file="crude_oil_anomaly_detector_validation.md",
    ),
    "wheat": CommodityConfig(
        key="wheat",
        display_name="Wheat (Chicago SRW)",
        ticker="ZW=F",
        gdelt_queries=["wheat export ban Russia", "wheat drought", "wheat Ukraine war"],
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
        validation_md_file="wheat_anomaly_detector_validation.md",
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
