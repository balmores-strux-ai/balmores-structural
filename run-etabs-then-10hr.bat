@echo off
REM ======================================================================
REM 1) Generate up to 2500 NEW real ETABS models (toward 100k roadmap)
REM 2) Skip short inline train — run full 10-hour learning on the CSV
REM ======================================================================
REM Requires: ETABS installed, licensed. Close other ETABS instances.
REM Repeat this batch (or run overnight) many times to grow the dataset.
REM At 2500/run: ~40 runs to reach 100,000 rows from zero (plus failures).
REM ======================================================================
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q

set ETABS_TARGET_TOTAL=100000
set ETABS_MAX_NEW_PER_RUN=2500
set ETABS_SKIP_TRAIN=1

echo.
echo Step 1/2: ETABS data collection (max 2500 new rows this run^)...
python scripts/etabs_connect_test.py
if errorlevel 1 (
  echo ETABS connection failed.
  pause
  exit /b 1
)
python scripts/etabs_brain_full.py
if errorlevel 1 (
  echo ETABS pipeline failed.
  pause
  exit /b 1
)

echo.
echo Step 2/2: 10-hour brain training (keep PC awake^)...
set ETABS_SKIP_TRAIN=
python scripts/train_10hr.py --hours 10

echo.
echo Done. Push model: git add backend/models/*.pt backend/data/*.csv ^&^& git commit -m "ETABS+10h" ^&^& git push
pause
