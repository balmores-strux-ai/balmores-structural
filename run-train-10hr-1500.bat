@echo off
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q
echo ============================================================
echo BALMORES STRUCTURAL - 10 HOUR training
echo Target: current CSV rows + 1500 augmented samples (conservative noise)
echo.
echo If you already appended 1500 REAL ETABS rows to the CSV, use
echo   run-train-10hr.bat   instead (no synthetic top-up).
echo ============================================================
python scripts/train_10hr.py --hours 10 --augment-extra 1500
echo.
pause
