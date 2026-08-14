# Con-PINN: Conservation-Enhanced Physics-Informed Neural Networks for Dispersive PDEs

[![Status](https://img.shields.io/badge/example-PINN%20%2B%20Con--PINN%20(KdV1)-green)](#example-currently-available)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository accompanies the paper:

> Jain M. Francis, Chandhini G, Rakesh Kumar, Nagaiah Chamakuri.  
> **Conservation-Enhanced PINN for dispersive PDEs.**
> (Under Review)

**Corresponding author:** Chandhini G ([chandhini@nitk.edu.in](mailto:chandhini@nitk.edu.in))

---

## Example currently available

This repository currently includes **one** paper case: KdV single soliton, with **PINN** and **Con-PINN** only. Remaining cases, the cPINN baseline, and the study scripts will be added **upon publication**.

```bash
pip install -r requirements.txt
cd src
python kdv1.py --seeds 0 1 2 3 4 --methods pinn conpinn --out_dir ../results_kdv1
```

See [`src/README.md`](src/README.md) for a short smoke test (`--tiny`) and output layout.

Until the full release, this repository also provides a stable citation URL: [github.com/Jainmf/Con-PINN](https://github.com/Jainmf/Con-PINN).

---

## What this work does

Standard Physics-Informed Neural Networks (PINNs) enforce the PDE residual and the initial/boundary conditions, but they do not automatically preserve physical invariants. **Con-PINN** adds mass, momentum, and energy conservation as extra terms in the training loss, evaluated by trapezoidal quadrature on collocation points, while remaining mesh-free.

The paper compares Con-PINN with a standard PINN and, where applicable, a domain-decomposition conservative PINN baseline (cPINN).

### Test cases

| Case | PDE | Domain |
|------|-----|--------|
| KdV, single soliton | Korteweg–de Vries | \(x \in [0, 2]\), \(t \in [0, 3]\) |
| KdV, double soliton | Korteweg–de Vries | \(x \in [0, 2]\), \(t \in [0, 3]\) |
| Equal Width (EW) | Sobolev-type | \(x \in [0, 30]\), \(t \in [0, 10]\) |
| Zakharov system | Coupled complex Langmuir / ion-acoustic waves | \(x \in [-32, 32]\), \(t \in [0, 10]\) |

Additional studies in the paper: initial-condition noise sensitivity, temporal extrapolation, computational scaling, and a hyperparameter ablation (depth, width, activation, loss weights \(\lambda_{\mathrm{IC}}\), \(\lambda_{\mathrm{BC}}\), \(\lambda_{\mathrm{Con}}\)).

Main results use **five independent random seeds** (`0`–`4`).

---

## What will be released later

After publication this repository will also include:

- Training scripts for the remaining test cases, and the cPINN baseline
- Scripts for the noise, extrapolation, scaling, and ablation studies
- Plotting / table scripts used to generate the paper figures

Current layout:

```text
con-pinn/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
└── src/
    ├── kdv1.py          # KdV single soliton: PINN and Con-PINN
    └── common.py
```

---

## Environment used in the paper

Reported training environment (see Section 5 of the manuscript):

| Item | Value |
|------|--------|
| Python | 3.8.5 |
| PyTorch | 1.11.0+cu102 |
| CUDA | 10.2 |
| GPU | NVIDIA Tesla V100-SXM2-16GB |
| CPU | 40 logical cores |
| OS | Linux |

Typical architecture for the scalar cases: **5 hidden layers, 100 neurons, `tanh`**. Zakharov uses **5 × 80**. Optimisation is **Adam followed by L-BFGS**. Conservation integrals use the composite trapezoidal rule.

---

## Citation

Please cite the paper if you use this work:

```bibtex
@article{francis2026conpinn,
  title  = {Conservation-Enhanced {PINN} for dispersive {PDEs}},
  author = {Francis, Jain M. and G, Chandhini and Kumar, Rakesh and Chamakuri, Nagaiah},
  year   = {2026}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff). Add the journal name, `year`, `doi`, and pages after the article is published.

---

## License

This repository is released under the [MIT License](LICENSE).

---

## Contact

- Jain M. Francis — [jainmfrancis.197ma002@nitk.edu.in](mailto:jainmfrancis.197ma002@nitk.edu.in), [jainmfrancis@gmail.com](mailto:jainmfrancis@gmail.com)
- Chandhini G (corresponding) — [chandhini@nitk.edu.in](mailto:chandhini@nitk.edu.in)
- Rakesh Kumar — [rakesh.kumar@mahindrauniversity.edu.in](mailto:rakesh.kumar@mahindrauniversity.edu.in)
- Nagaiah Chamakuri — [nagaiah.chamakuri@iisertvm.ac.in](mailto:nagaiah.chamakuri@iisertvm.ac.in)
