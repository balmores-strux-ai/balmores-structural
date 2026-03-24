"""
ETABS-based training: Generate 1000 real parametric models in ETABS,
extract results, append to CSV, then train the brain.

REQUIREMENTS:
  - ETABS installed and licensed
  - Python with: comtypes, numpy, pandas, torch, psutil (optional)
  - Run from project root or backend/

USAGE:
  cd c:\Users\dell\Desktop\balmores-strux-ai
  "C:\Program Files\Python38\python.exe" backend\scripts\run_etabs_train.py

  Or with venv:
  cd backend && .venv\Scripts\activate
  pip install comtypes
  python scripts/run_etabs_train.py

This script delegates to your FULL etabs_brain.py.
If you have the complete ETABS COM script (with connect_to_etabs,
build_one_etabs_model, run_analysis_and_extract, collect_dataset),
place it as backend/scripts/etabs_brain_full.py and this launcher
will run it with NEW_SAMPLES=1000.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND = SCRIPT_DIR.parent
PROJECT = BACKEND.parent

# Target: add 1000 new ETABS models to existing dataset
NEW_SAMPLES = 1000

# Full ETABS script with COM
ETABS_FULL_SCRIPT = SCRIPT_DIR / "etabs_brain_full.py"


def main():
    full_script = ETABS_FULL_SCRIPT
    if not full_script.exists():
        print("=" * 70)
        print("ETABS TRAINING LAUNCHER")
        print("=" * 70)
        print()
        print("To generate 1000 real ETABS models and train:")
        print()
        print("1. You need the FULL etabs_brain script that:")
        print("   - Connects to ETABS via COM (comtypes.client)")
        print("   - Builds parametric steel frame models (NewSteelDeck)")
        print("   - Runs analysis, extracts FrameForce, BaseReact, JointDispl")
        print("   - Appends rows to etabs_parametric_structural_teacher.csv")
        print("   - Trains StructuralBrain and saves .pt")
        print()
        print("2. Save that script as:")
        print("   backend/scripts/etabs_brain_full.py")
        print()
        print("3. Configure it for 1000 new samples:")
        print("   TARGET_TOTAL_MODELS = existing_rows + 1000  (e.g. 4100)")
        print("   APPEND_DATASET = True")
        print()
        print("4. Run:")
        print('   "C:\\Program Files\\Python38\\python.exe" backend\\scripts\\run_etabs_train.py')
        print()
        print("5. Ensure ETABS is installed. Close other ETABS instances.")
        print()
        print("See backend/scripts/ETABS_BRAIN_ENHANCEMENTS.md for the")
        print("enhancement guide. The full script was pasted in an earlier session.")
        print("=" * 70)
        sys.exit(1)

    # Run the full script with env override for new samples
    import os
    env = os.environ.copy()
    env["ETABS_MAX_NEW_PER_RUN"] = str(NEW_SAMPLES)

    proc = subprocess.run(
        [sys.executable, str(full_script)],
        cwd=str(BACKEND),
        env=env,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
