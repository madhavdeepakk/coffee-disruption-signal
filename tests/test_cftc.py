"""Offline tests for the CFTC fixes: parsing, history accumulation, percentile."""
import pandas as pd
from src.data import cftc


DISAGG_FIXTURE = (
    "Market_and_Exchange_Names,CFTC_Contract_Market_Code,Report_Date_as_YYYY-MM-DD,"
    "Open_Interest_All,M_Money_Positions_Long_All,M_Money_Positions_Short_All,"
    "Prod_Merc_Positions_Long_All,Prod_Merc_Positions_Short_All\n"
    "COFFEE C - ICE FUTURES U.S.,083731,2024-01-02,200000,60000,20000,50000,70000\n"
    "WHEAT - CHICAGO BOARD OF TRADE,001602,2024-01-02,300000,1,1,1,1\n"
)


def test_parse_disaggregated_filters_coffee():
    df = cftc._parse_disaggregated(DISAGG_FIXTURE)
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["noncommercial_long"]) == 60000
    assert int(row["noncommercial_short"]) == 20000
    assert int(row["open_interest"]) == 200000


def test_merge_cache_dedupes_and_accumulates(tmp_path, monkeypatch):
    cache = tmp_path / "cftc_cot.csv"
    monkeypatch.setattr(cftc, "CACHE_PATH", cache)
    wk1 = cftc._parse_disaggregated(DISAGG_FIXTURE)
    merged = cftc._merge_cache(wk1)
    merged.to_csv(cache, index=False)
    # a second, later week
    wk2 = cftc._parse_disaggregated(
        DISAGG_FIXTURE.replace("2024-01-02", "2024-01-09").replace("60000", "90000"))
    merged2 = cftc._merge_cache(wk2)
    assert len(merged2) == 2  # two distinct weeks accumulated
    # re-merging the same week does not duplicate
    again = cftc._merge_cache(wk2)
    assert len(again) == 2


def test_percentile_is_meaningful_with_history():
    # 10 weeks, net rising each week -> latest should sit at the top percentile.
    rows = []
    for i in range(10):
        rows.append({
            "report_date": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i),
            "open_interest": 200000,
            "noncommercial_long": 40000 + i * 5000,
            "noncommercial_short": 20000,
            "commercial_long": 0, "commercial_short": 0,
        })
    df = pd.DataFrame(rows)
    res = cftc.analyze_positioning(df)
    assert res["net_spec_percentile"] >= 80  # not the single-row default of 50
    assert res["net_speculative"] == 40000 + 9 * 5000 - 20000
