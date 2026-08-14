# KdV single-soliton example (PINN and Con-PINN)

This folder currently contains **one** paper case:

- PINN
- Con-PINN

Remaining cases, cPINN, and study scripts will be added **upon publication**.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

## Run (paper settings, 5 seeds)

```bash
cd src
python kdv1.py --seeds 0 1 2 3 4 --methods pinn conpinn --out_dir ../results_kdv1
```

`--methods all` also runs PINN and Con-PINN.

Quick check (small net, few epochs):

```bash
python kdv1.py --seeds 0 --methods pinn conpinn --tiny
```

Outputs go to `results_kdv1/` (`models/`, `runs/`, `results_raw.csv`).
