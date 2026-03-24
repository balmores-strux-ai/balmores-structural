@echo off
REM ETABS-based training: generate 1000 real parametric models in ETABS,
REM append to teacher CSV, then train the brain.
REM
REM REQUIREMENTS:
REM   - ETABS installed and licensed
REM   - Close other ETABS instances before running
REM
cd /d "%~dp0backend"

set ETABS_TARGET_TOTAL=100000
set ETABS_MAX_NEW_PER_RUN=2500

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate

echo Installing dependencies (comtypes, psutil, torch, pandas)...
pip install -r requirements.txt -q

echo.
echo ======================================================================
echo ETABS TRAINING - batch toward 100,000-row roadmap (max 2500 new this run)
echo ======================================================================
echo.
echo This will:
echo   1. Connect to ETABS
echo   2. Add up to 2500 NEW rows (ETABS_MAX_NEW_PER_RUN), until CSV hits 100k total
echo   3. Run analysis, extract results, append CSV
echo   4. Train the brain (short pass) and save .pt
echo.
echo For ETABS + 10-hour learning in one go, use: run-etabs-then-10hr.bat
echo Env: ETABS_TARGET_TOTAL, ETABS_MAX_NEW_PER_RUN, ETABS_SKIP_TRAIN=1
echo.
echo Ensure ETABS is installed. The script will start ETABS if needed.
echo.
echo Running ETABS connection test first...
python scripts/etabs_connect_test.py
if errorlevel 1 (
  echo.
  echo Connection test FAILED. Fix ETABS connection before training.
  pause
  exit /b 1
)
echo.
echo Connection OK. Starting full ETABS training...
echo.
python scripts/etabs_brain_full.py

echo.
echo Done. Check backend/data/etabs_parametric_structural_teacher.csv
echo and backend/models/etabs_parametric_structural_brain.pt
pause
