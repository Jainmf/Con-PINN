"""KdV example 1 (single soliton): PINN and Con-PINN.

Equation: u_t + eps u u_x + mu u_xxx = 0
  python kdv1.py --seeds 0 1 2 3 4 --methods pinn conpinn
"""
import os
import sys

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import time
import math
import argparse

import numpy as np
import torch

from common import DNN, set_seed, sech, lhs, device, DTYPE, trapz_mean, trapz_mean_batch,    plot_loss, plot_solution_comparison, plot_conservation_comparison,    reset_memory_tracking, get_memory_mb, count_parameters, write_run_report, save_loss_history_csv,    scalar_result_rows, save_grid_npz, save_curves_npz

eps, mu, c = 1.0, 4.84e-4, 0.3
A = 0.5 * math.sqrt(eps * c / mu)
B = 0.5 * eps * c * math.sqrt(eps * c / mu)
D = -6.0

I1 = 6 * c / A
I2 = 12 * c ** 2 / A
I3 = 144 * c ** 3 / (5 * A) - 144 * A * c ** 2 * mu / (5 * eps)

x_min, x_max, t_min, t_max = 0.0, 2.0, 0.0, 3.0
CONSERVED_TIMES = [0.5, 1.5, 3.0]

def exact(x, t):
    temp = A * x + D - B * t
    return 3 * c * sech(temp) ** 2

class PINN:
    """Set use_conservation=True for Con-PINN, False for plain PINN."""

    def __init__(self, use_conservation: bool, seed: int,
                 epochs_adam=10000, lbfgs_max_iter=10000, lbfgs_steps=1,
                 n_layer=5, n_node=100, N_0=1000, N_bc=1000, N_f=40000,
                 N_cx=1000, N_ct=10, lam_i=10.0, lam_b=10.0,
                 lam_c=1.0, con_warmup=0, con_relative=False, ic_noise_std=0.0,
                 activation="tanh"):
        """use_conservation=True for Con-PINN. ic_noise_std perturbs the IC (noise study)."""
        set_seed(seed)
        self.use_conservation = use_conservation
        self.epochs_adam = epochs_adam
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_steps = lbfgs_steps
        self.n_layer = n_layer
        self.n_node = n_node
        self.lam_c = lam_c
        self.con_warmup = con_warmup
        self.con_relative = con_relative
        self.activation_name = (activation if isinstance(activation, str)
                                 else activation.__class__.__name__)

        x_pts = lambda n: np.random.uniform(x_min, x_max, (n, 1))

        x0 = x_pts(N_0)
        t0 = np.zeros((N_0, 1))
        u0 = exact(x0, t0)
        if ic_noise_std > 0:
            u0 = u0 + np.random.normal(0.0, ic_noise_std, size=u0.shape)
        self.xt_0 = torch.tensor(np.hstack([x0, t0]), dtype=DTYPE).to(device)
        self.u_0 = torch.tensor(u0, dtype=DTYPE).to(device)

        t_bc = np.linspace(t_min, t_max, N_bc).reshape(-1, 1)
        xt_bc = np.vstack([np.hstack([np.full_like(t_bc, x_min), t_bc]),
                            np.hstack([np.full_like(t_bc, x_max), t_bc])])
        self.xt_bc = torch.tensor(xt_bc, dtype=DTYPE).to(device)
        self.u_bc = torch.zeros(xt_bc.shape[0], 1, dtype=DTYPE).to(device)

        x_f = x_min + (x_max - x_min) * lhs(1, N_f)
        t_f = t_min + (t_max - t_min) * lhs(1, N_f)
        xt_f = np.vstack([np.hstack([x_f, t_f]), xt_bc, np.hstack([x0, t0])])

        self.xt_f = torch.tensor(xt_f, dtype=DTYPE, requires_grad=True).to(device)

        self.N_cx, self.N_ct = N_cx, N_ct
        x_c = np.linspace(x_min, x_max, N_cx).reshape(-1, 1)
        t_c = np.linspace(t_min, t_max, N_ct)

        xc_rep = np.tile(x_c, (N_ct, 1))
        tc_rep = np.repeat(t_c, N_cx).reshape(-1, 1)
        self.xt_c_all = torch.tensor(np.hstack([xc_rep, tc_rep]), dtype=DTYPE,
                                      requires_grad=True).to(device)
        self.xdiff = x_max - x_min
        self._ones_f = torch.ones(self.xt_f.shape[0], 1, dtype=DTYPE, device=device)
        self._ones_c = torch.ones(self.xt_c_all.shape[0], 1, dtype=DTYPE, device=device)

        if isinstance(activation, str):
            act_map = {
                "tanh": torch.nn.Tanh,
                "sigmoid": torch.nn.Sigmoid,
                "softplus": torch.nn.Softplus,
                "swish": torch.nn.SiLU,
                "silu": torch.nn.SiLU,
                "relu": torch.nn.ReLU,
            }
            key = activation.lower()
            if key not in act_map:
                raise ValueError(f"Unknown activation '{activation}'. "
                                 f"Choose from {sorted(act_map)}")
            act = act_map[key]()
        else:
            act = activation
        self.net = DNN(dim_in=2, dim_out=1, n_layer=n_layer, n_node=n_node,
                        ub=np.array([x_max, t_max]), lb=np.array([x_min, t_min]),
                        activation=act).to(DTYPE).to(device)
        self.adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(), lr=1.0, max_iter=lbfgs_max_iter, max_eval=None,
            tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50, line_search_fn="strong_wolfe")
        self.lam_i = torch.tensor(lam_i, device=device)
        self.lam_b = torch.tensor(lam_b, device=device)

        self.losses = {"ic": [], "bc": [], "con": [], "pde": [], "loss": []}
        self.iter = 0

    def predict(self, xt):
        return self.net(xt)

    def bc_loss(self):
        return torch.mean((self.predict(self.xt_bc) - self.u_bc) ** 2)

    def ic_loss(self):
        return torch.mean((self.predict(self.xt_0) - self.u_0) ** 2)

    def pde_loss(self):
        xt = self.xt_f
        u = self.predict(xt)
        grad_u = torch.autograd.grad(u, xt, grad_outputs=self._ones_f, create_graph=True)[0]
        u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=self._ones_f, create_graph=True)[0][:, 0:1]
        u_xxx = torch.autograd.grad(u_xx, xt, grad_outputs=self._ones_f, create_graph=True)[0][:, 0:1]
        res = u_t + eps * u * u_x + mu * u_xxx
        return torch.mean(res ** 2)

    def con_loss(self):
        """Batched over all N_ct slices (one forward + one u_x grad)."""
        xt = self.xt_c_all
        u = self.predict(xt)
        u_x = torch.autograd.grad(u, xt, grad_outputs=self._ones_c, create_graph=True)[0][:, 0:1]
        u = u.reshape(self.N_ct, self.N_cx)
        u_x = u_x.reshape(self.N_ct, self.N_cx)
        i1 = self.xdiff * trapz_mean_batch(u)
        i2 = self.xdiff * trapz_mean_batch(u ** 2)
        i3 = self.xdiff * trapz_mean_batch(u ** 3 - (3 * mu / eps) * u_x ** 2)
        if self.con_relative:
            err = ((i1 - I1) / I1) ** 2 + ((i2 - I2) / I2) ** 2 + ((i3 - I3) / I3) ** 2
        else:
            err = (i1 - I1) ** 2 + (i2 - I2) ** 2 + (i3 - I3) ** 2
        return err.mean()

    def closure(self):
        self.adam.zero_grad(set_to_none=True)
        self.lbfgs.zero_grad(set_to_none=True)
        mse_bc, mse_ic, mse_pde = self.bc_loss(), self.ic_loss(), self.pde_loss()
        if self.use_conservation:
            mse_con = self.con_loss()
        else:
            mse_con = torch.zeros((), dtype=DTYPE, device=device)
        loss = self.lam_i * mse_ic + self.lam_b * mse_bc + mse_pde
        if self.use_conservation and self.iter >= self.con_warmup:
            loss = loss + self.lam_c * mse_con
        loss.backward()
        for k, v in zip(["bc", "ic", "pde", "con", "loss"], [mse_bc, mse_ic, mse_pde, mse_con, loss]):
            self.losses[k].append(v.detach().cpu().item())
        self.iter += 1
        return loss

    def train(self, verbose_every=0):
        t0 = time.time()
        for i in range(self.epochs_adam):
            self.closure()
            self.adam.step()
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"  [adam {i+1}/{self.epochs_adam}] loss={self.losses['loss'][-1]:.4e} "
                      f"elapsed={time.time()-t0:.1f}s")

        for _ in range(self.lbfgs_steps):
            self.lbfgs.step(self.closure)
        if verbose_every:
            print(f"  Done. Total time {time.time()-t0:.1f}s")

    def eval_l2(self, n_points=2000, t_range=None):
        """t_range: optional (t_lo, t_hi) to evaluate OUTSIDE the training
        window [t_min, t_max] -- e.g. t_range=(t_max, 2*t_max) for a
        temporal-extrapolation study. exact() is a plain closed-form
        function of (x,t) with no domain restriction, so this is valid;
        accuracy is of course expected to degrade the network was never
        trained to be accurate there."""
        t_lo, t_hi = t_range if t_range is not None else (t_min, t_max)
        x = np.random.uniform(x_min, x_max, (n_points, 1))
        t = np.random.uniform(t_lo, t_hi, (n_points, 1))
        u_star = exact(x, t)
        xt = torch.tensor(np.hstack([x, t]), dtype=DTYPE).to(device)
        with torch.no_grad():
            u_pred = self.predict(xt).cpu().numpy()
        return float(np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star))

    def eval_conservation_errors(self, times):
        x_c = np.linspace(x_min, x_max, 2000).reshape(-1, 1)
        out = {}
        for t in times:
            xt = torch.tensor(np.hstack([x_c, t * np.ones_like(x_c)]), dtype=DTYPE,
                               requires_grad=True).to(device)
            u = self.predict(xt)
            u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u))[0][:, 0:1]
            i1p = (self.xdiff * trapz_mean(u)).item()
            i2p = (self.xdiff * trapz_mean(u ** 2)).item()
            i3p = (self.xdiff * trapz_mean(u ** 3 - (3 * mu / eps) * u_x ** 2)).item()
            out[t] = {"mass": abs(i1p - I1), "momentum": abs(i2p - I2), "energy": abs(i3p - I3)}
        return out

    def eval_conservation_curve(self, n_times=50, t_start=None, t_end=None):
        """t_start/t_end: optional override of the evaluation window.
        Defaults to [t_min, t_max] (the training window) exactly as
        before; pass t_end > t_max for a temporal-extrapolation curve."""
        x_c = np.linspace(x_min, x_max, 2000).reshape(-1, 1)
        times = np.linspace(t_start if t_start is not None else t_min,
                             t_end if t_end is not None else t_max, n_times)
        mass, momentum, energy = [], [], []
        for t in times:
            xt = torch.tensor(np.hstack([x_c, t * np.ones_like(x_c)]), dtype=DTYPE,
                               requires_grad=True).to(device)
            u = self.predict(xt)
            u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u))[0][:, 0:1]
            mass.append((self.xdiff * trapz_mean(u)).item())
            momentum.append((self.xdiff * trapz_mean(u ** 2)).item())
            energy.append((self.xdiff * trapz_mean(u ** 3 - (3 * mu / eps) * u_x ** 2)).item())
        return {"times": times, "mass": np.array(mass), "momentum": np.array(momentum),
                "energy": np.array(energy)}

    def eval_grid(self, nx=101, nt=101):
        x1, t1 = np.linspace(x_min, x_max, nx), np.linspace(t_min, t_max, nt)
        x_grid, t_grid = np.meshgrid(x1, t1)
        xt = np.hstack([x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)])
        with torch.no_grad():
            u_pred = self.predict(torch.tensor(xt, dtype=DTYPE).to(device)).cpu().numpy().reshape(x_grid.shape)
        return x_grid, t_grid, u_pred, exact(x_grid, t_grid)

DEFAULT_PINN_KWARGS = dict(n_layer=5, n_node=100, N_0=1000, N_bc=1000, N_f=40000, N_cx=1000, N_ct=10)

def main(seeds, methods, epochs_adam, lbfgs_max_iter, lbfgs_steps,
         out_dir, verbose_every=0, lam_c=1.0, con_warmup=0, con_relative=False,
         pinn_kwargs=None):
    pinn_kwargs = pinn_kwargs or {}
    eff_pinn = {**DEFAULT_PINN_KWARGS, **pinn_kwargs}
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    rows = []

    for seed in seeds:
        curves, grids = {}, {}

        for method, use_con in [("PINN", False), ("ConPINN", True)]:
            if method not in methods:
                continue
            reset_memory_tracking()
            model = PINN(use_conservation=use_con, seed=seed, epochs_adam=epochs_adam,
                         lbfgs_max_iter=lbfgs_max_iter, lbfgs_steps=lbfgs_steps,
                         lam_c=lam_c, con_warmup=con_warmup, con_relative=con_relative, **pinn_kwargs)
            print(f"[kdv1 | {method} | seed={seed}] training...")
            tic = time.time()
            model.train(verbose_every=verbose_every)
            train_time = time.time() - tic
            mem = get_memory_mb()
            l2 = model.eval_l2()
            con_err = model.eval_conservation_errors(CONSERVED_TIMES)
            curves[method] = model.eval_conservation_curve()
            grids[method] = model.eval_grid()
            torch.save(model.net.state_dict(), os.path.join(out_dir, "models", f"{method}_seed{seed}.pt"))
            plot_loss(model.losses, os.path.join(out_dir, "plots", f"{method}_seed{seed}_loss.png"),
                       title=f"kdv1 / {method} / seed={seed}")

            tag = f"{method}_seed{seed}"
            n_params = count_parameters(model.net)
            sections = {
                "Architecture": {"Hidden layers (n_layer)": eff_pinn["n_layer"],
                                  "Neurons per layer (n_node)": eff_pinn["n_node"],
                                  "Trainable parameters": n_params,
                                  "Network outputs": "1 (u)"},
                "Training Config": {"Method": method, "Adam epochs": epochs_adam,
                                     "LBFGS max_iter (per call)": lbfgs_max_iter,
                                     "LBFGS steps (# of .step() calls)": lbfgs_steps,
                                     "lam_i (IC weight)": 10.0, "lam_b (BC weight)": 10.0,
                                     "lam_c (conservation weight)": lam_c if use_con else "N/A (plain PINN)",
                                     "con_warmup (epochs before con loss on)": con_warmup if use_con else "N/A",
                                     "con_relative (per-term normalization)": con_relative if use_con else "N/A"},
                "Data Budget": {"IC points (N_0)": eff_pinn["N_0"], "BC points (N_bc)": eff_pinn["N_bc"],
                                 "Collocation points (N_f)": eff_pinn["N_f"],
                                 "Conservation check: spatial (N_cx)": eff_pinn["N_cx"],
                                 "Conservation check: time slices (N_ct)": eff_pinn["N_ct"]},
                "Results": {"Train time (s)": train_time, "Relative L2 error": l2,
                             **{f"|{qty} error| @ t={t}": v for t, errs in con_err.items() for qty, v in errs.items()}},
                "Memory": {"Peak CPU RSS (MB)": mem["cpu_rss_mb"] if mem["cpu_rss_mb"] is not None else "N/A (psutil not installed)",
                            "Peak GPU allocated (MB)": mem["gpu_peak_mb"] if mem["gpu_peak_mb"] is not None else "N/A (no GPU)"},
            }
            write_run_report(os.path.join(out_dir, "runs", tag + ".txt"),
                              problem="kdv1", method=method, seed=seed, sections=sections)
            save_loss_history_csv(model.losses, os.path.join(out_dir, "runs", tag + "_loss_history.csv"))

            for t, errs in con_err.items():
                for qty, val in errs.items():
                    rows.append({"problem": "kdv1", "method": method, "seed": seed, "time": t,
                                 "quantity": qty, "abs_error": val, "relative_l2_error": l2, "train_time_s": train_time,
                                 "peak_cpu_mb": mem["cpu_rss_mb"], "peak_gpu_mb": mem["gpu_peak_mb"]})
            rows.append({"problem": "kdv1", "method": method, "seed": seed, "time": "overall",
                         "quantity": "L2_error", "abs_error": l2, "relative_l2_error": l2, "train_time_s": train_time,
                         "peak_cpu_mb": mem["cpu_rss_mb"], "peak_gpu_mb": mem["gpu_peak_mb"]})
            rows.extend(scalar_result_rows("kdv1", method, seed, train_time, mem))
            print(f"  L2={l2:.4e}  time={train_time:.1f}s  params={n_params}  "
                  f"mem={mem['cpu_rss_mb']:.0f}MB" if mem["cpu_rss_mb"] is not None else
                  f"  L2={l2:.4e}  time={train_time:.1f}s  params={n_params}")

        if len(grids) >= 1:
            any_grid = next(iter(grids.values()))
            x_grid, t_grid, _, u_exact = any_grid
            preds = {m: g[2] for m, g in grids.items()}
            save_grid_npz(os.path.join(out_dir, "data", f"seed{seed}_solution.npz"),
                          x_grid, t_grid, preds, u_exact)
            plot_solution_comparison(x_grid, t_grid, preds, u_exact,
                                      os.path.join(out_dir, "plots", f"seed{seed}_solution_comparison.png"),
                                      title=f"kdv1 / seed={seed} -- solution ({'+'.join(preds.keys())})")

        if len(curves) >= 1:
            save_curves_npz(os.path.join(out_dir, "data", f"seed{seed}_conservation.npz"), curves)
            if len(curves) >= 2:
                for qty, true_val in [("mass", I1), ("momentum", I2), ("energy", I3)]:
                    plot_conservation_comparison(curves, qty, true_val,
                                                  os.path.join(out_dir, "plots", f"seed{seed}_conservation_{qty}.png"),
                                                  title=f"kdv1 / seed={seed} -- {qty} ({'+'.join(curves.keys())})")

    csv_path = os.path.join(out_dir, "results_raw.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")
    print(f"Plots saved under: {out_dir}/plots/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", type=str, nargs="+", default=["all"],
                     choices=["pinn", "conpinn", "all"])
    ap.add_argument("--epochs_adam", type=int, default=10000)
    ap.add_argument("--lbfgs_max_iter", type=int, default=10000)
    ap.add_argument("--lbfgs_steps", type=int, default=1,
                     help="Number of L-BFGS step() calls (default 1).")
    ap.add_argument("--out_dir", type=str, default="results_kdv1")
    ap.add_argument("--print_every", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="Fast correctness check only.")
    ap.add_argument("--cpu", action="store_true",
                     help="Force CPU (handled before CUDA init via sys.argv). "
                          "Use when the GPU driver reports OOM even for tiny runs.")
    ap.add_argument("--lam_c", type=float, default=1.0,
                     help="Weight on Con-PINN's conservation loss term. Default 1.0 "
                          "matches the original (unweighted) behavior.")
    ap.add_argument("--con_warmup", type=int, default=0,
                     help="Delay turning on the conservation loss until this many "
                          "closure() calls have passed. Default 0 = on from the start "
                          "(original behavior).")
    ap.add_argument("--con_relative", action="store_true",
                     help="Normalize each conservation term (mass/momentum/energy) by "
                          "its own target value, instead of using raw squared error. "
                          "Off by default (original behavior).")
    args = ap.parse_args()

    name_map = {"pinn": "PINN", "conpinn": "ConPINN"}
    methods = ["PINN", "ConPINN"] if "all" in args.methods else [name_map[m] for m in args.methods]

    if args.tiny:
        args.epochs_adam, args.lbfgs_max_iter = 20, 20
        pinn_kwargs = dict(n_layer=2, n_node=10, N_0=50, N_bc=50, N_f=200, N_cx=50, N_ct=5)
    else:
        pinn_kwargs = {}

    main(seeds=args.seeds, methods=methods, epochs_adam=args.epochs_adam,
         lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_steps=args.lbfgs_steps,
         out_dir=args.out_dir, verbose_every=args.print_every,
         lam_c=args.lam_c, con_warmup=args.con_warmup, con_relative=args.con_relative,
         pinn_kwargs=pinn_kwargs)
