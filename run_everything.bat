@echo off
REM ==========================================================================
REM Everything that can still be run without a person reading anything.
REM
REM ORDER MATTERS, and the order is not "all at once". The news API answers
REM about one request in five and asks for at most one request every five
REM seconds. Three commodities fetching at the same time send three times the
REM requests into the same limit, get more of them rejected, and all three
REM finish later than one would have. So the keyless work runs first and
REM fast, and the news-fetching work runs in ONE chain after it.
REM
REM Run this from a terminal you can leave alone. The last stage is hours.
REM It is safe to stop and run again: every answer is cached as it arrives.
REM ==========================================================================
cd /d D:\CAPSTONE

echo.
echo ====== STAGE 1: keyless, offline, minutes =================================
echo.

REM The numeraire test: did the move survive translation into euro, yen,
REM yuan and rupees, or did the dollar move and the commodity not? This has
REM never been run. It needs no key and no network.
py -m src.analysis.numeraire --commodity coffee    --flagged --save
py -m src.analysis.numeraire --commodity crude_oil --flagged --save
py -m src.analysis.numeraire --commodity wheat     --flagged --save
py -m src.analysis.numeraire --commodity gold      --flagged --save
py -m src.analysis.numeraire --commodity silver    --flagged --save

REM One card per labelled day, every layer side by side, no overall verdict.
py -m src.evaluation.anomaly_card --commodity coffee --labelled --save

REM Rebuild the strategy report now that the numeraire layer exists.
py -m src.analysis.strategy --save

REM Rebuild the threshold sweep and the cross-commodity check from current data.
py -m scripts.threshold_sweep --save
py -m src.evaluation.cross_commodity

echo.
echo ====== STAGE 2: the long chain, one commodity at a time ===================
echo.

REM Prefetch each commodity's quiet-day news first, with enough rounds to
REM finish, then run it. Doing the fetch separately means the progress is
REM visible instead of buried in a log.
py -m scripts.fill_placebo_news --commodity gold   --rounds 40
py -m scripts.run_commodity_eval --commodity gold   --providers groq

py -m scripts.fill_placebo_news --commodity wheat  --rounds 40
py -m scripts.run_commodity_eval --commodity wheat  --providers groq

py -m scripts.fill_placebo_news --commodity silver --rounds 40
py -m scripts.run_commodity_eval --commodity silver --providers groq

echo.
echo ====== STAGE 3: rebuild everything downstream =============================
echo.
py -m scripts.run_commodity_eval --all --report-only
py -m src.analysis.strategy --save
py -m pytest -q

echo.
echo ====== DONE ===============================================================
echo Check results\multi_commodity_report.md : every commodity should now have
echo BOTH columns filled in. A commodity still showing "not run yet" in the
echo invented-move column had quiet days whose news never arrived - run
echo    py -m scripts.fill_placebo_news --commodity NAME --rounds 40
echo again, then
echo    py -m scripts.run_commodity_eval --commodity NAME --providers groq
echo.
pause
