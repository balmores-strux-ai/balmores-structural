@echo off
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q
echo ============================================================
echo BALMORES STRUCTURAL - 10 HOUR extended training
echo Cosine warm-restarts, EMA, physics weights, checkpoints every 30 min
echo Keep this PC awake. Close only after it finishes.
echo ============================================================
python scripts/train_10hr.py --hours 10
echo.
pause
