@echo off
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q
echo BALMORES STRUCTURAL - Physics-informed brain training (5000+ models)
echo Using: StructuralBrain3Res, weighted physics targets, 800 epochs
echo.
python scripts/train_brain.py --epochs 800 --patience 150
echo.
echo Done. Brain saved to backend/models/etabs_parametric_structural_brain.pt
echo.
echo To deploy: git add backend/models/*.pt backend/data/*.csv
echo            git commit -m "Physics-upgraded brain (5000+)"
echo            git push
pause
