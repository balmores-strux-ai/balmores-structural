# Structural analysis bundle (PyNite)

This folder is a **snapshot of PyNite** components used for **structural / finite-element analysis** and **design-oriented examples**, copied from the upstream `Pynite-main/` tree in this repository.

## Contents

| Path | Purpose |
|------|--------|
| **`Pynite/`** | Core library: 3D frames, members, plates/quads, loads, linear & P-Δ analysis, visualization helpers, reporting. |
| **`Examples/`** | Runnable scripts (beams, frames, high-rise, P-Δ, modal, shear walls, etc.). |
| **`Testing/`** | Pytest suite that exercises analysis features (regression / validation). |
| **`LICENSE`** | PyNite license (upstream). |
| **`requirements.txt`** | Upstream Python dependencies for running PyNite + examples/tests. |
| **`setup.py`** | Upstream package metadata. |
| **`PYNITE_UPSTREAM_README.md`** | Original PyNite README. |

## Not copied here

Documentation site, Jupyter derivations, CI configs, and archived one-off files remain under `Pynite-main/` only. If you need them, use that folder or clone [PyNite](https://github.com/JWock82/Pynite) directly.

## Using this copy in Python

Add this directory’s parent to `sys.path`, then import as usual:

```python
import sys
from pathlib import Path
root = Path(__file__).resolve().parent / "structural_analysis_pynite"
sys.path.insert(0, str(root))
from Pynite import FEModel3D
```

The Balmores backend currently points at `Pynite-main/` in `backend/app/pynite_fea.py`; you can switch `_PYNITE_ROOT` to this folder if you want a single “analysis-only” vendored tree.

## Syncing from `Pynite-main`

After upgrading PyNite upstream, re-run the same copy commands from the repo root (PowerShell):

```powershell
Copy-Item -Recurse -Force Pynite-main\Pynite structural_analysis_pynite\Pynite
Copy-Item -Recurse -Force Pynite-main\Examples structural_analysis_pynite\Examples
Copy-Item -Recurse -Force Pynite-main\Testing structural_analysis_pynite\Testing
Copy-Item -Force Pynite-main\LICENSE, Pynite-main\requirements.txt, Pynite-main\setup.py structural_analysis_pynite\
```
