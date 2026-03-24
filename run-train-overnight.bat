@echo off
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q
echo ============================================================
echo BALMORES STRUCTURAL - Overnight Training until 7:30 AM
echo Physics-informed, ETABS-level accuracy target
echo ============================================================
python scripts/train_overnight.py --until 07:30
echo.
echo Training complete. Model saved.
echo Run: git add backend/models/*.pt backend/data/*.csv
echo      git commit -m "Physics-trained brain (overnight)"
echo      git push
pause
