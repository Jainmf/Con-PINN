"""Shared PINN building blocks: network, sampling, quadrature, I/O, and plots.
Used by the KdV, Equal Width, and Zakharov training scripts."""
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    import psutil
    _process = psutil.Process()
except ImportError:
    psutil = None
    _process = None

def reset_memory_tracking():
    """Call once, right before building the model / starting training."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_memory_mb():
    """Returns {'cpu_rss_mb': float|None, 'gpu_peak_mb': float|None}.
    cpu_rss_mb is the process's CURRENT resident set size (needs the
    `psutil` package; None if it isn't installed -- this is the one
    thing here with an optional dependency, everything else only needs
    torch/numpy/matplotlib). gpu_peak_mb is the PEAK CUDA memory
    allocated since the last reset_memory_tracking() call -- None if no
    GPU. On CPU-only machines gpu_peak_mb is always None; that's not
    a bug, there's no GPU to measure."""
    out = {"cpu_rss_mb": None, "gpu_peak_mb": None}
    if _process is not None:
        out["cpu_rss_mb"] = _process.memory_info().rss / (1024 ** 2)
    if torch.cuda.is_available():
        out["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return out

def count_parameters(*modules):
    """Total trainable parameter count across one or more nn.Modules
    (pass model.net for PINN / Con-PINN)."""
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)

def scalar_result_rows(problem, method, seed, train_time, mem):
    """Builds extra rows (for the same results_raw.csv the *_experiments
    scripts already write) with quantity='train_time_s' / 'peak_cpu_mb' /
    'peak_gpu_mb', time='overall'. Extend your `rows` list with these
    alongside the usual conservation-error and L2-error rows -- doing so
    means statistical_analysis.py's existing paired t-test / Wilcoxon /
    CI / Cohen's d machinery automatically ALSO tests whether training
    time and memory differ significantly between methods across seeds,
    with no changes needed to that script."""
    out = [{"problem": problem, "method": method, "seed": seed, "time": "overall",
            "quantity": "train_time_s", "abs_error": train_time}]
    if mem.get("cpu_rss_mb") is not None:
        out.append({"problem": problem, "method": method, "seed": seed, "time": "overall",
                     "quantity": "peak_cpu_mb", "abs_error": mem["cpu_rss_mb"]})
    if mem.get("gpu_peak_mb") is not None:
        out.append({"problem": problem, "method": method, "seed": seed, "time": "overall",
                     "quantity": "peak_gpu_mb", "abs_error": mem["gpu_peak_mb"]})
    return out

def get_system_info():
    """Hardware/software environment for a run -- reviewers commonly ask
    for this and it's easy to forget: Python/torch/CUDA versions, GPU
    model (if any), CPU core count, OS. Computed once per call (cheap)."""
    import platform
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "OS": f"{platform.system()} {platform.release()}",
        "CPU cores (logical)": os.cpu_count(),
        "Device used": str(device),
    }
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["CUDA version"] = torch.version.cuda
    else:
        info["GPU"] = "None (ran on CPU)"
    return info

def write_run_report(path, problem, method, seed, sections, notes=None):
    """Writes a plain-text summary of one training run: system info,
    architecture, training config, data budget, results, and memory use,
    all in one place per (problem, method, seed) -- e.g.
    results_kdv1/runs/PINN_seed0.txt.

    sections: dict of {"Section Title": {key: value, ...}}, written in
    the order given (Python dicts preserve insertion order). A "System"
    section (Python/PyTorch/CUDA versions, GPU model, CPU cores) is
    added automatically at the top of every report -- reviewers commonly
    ask for this and it's easy to forget to log per-run.
    notes: optional list of strings, e.g. caveats specific to this run.
    """
    all_sections = {"System": get_system_info(), **sections}
    with open(path, "w") as f:
        header = f"RUN REPORT: {problem} / {method} / seed={seed}"
        f.write(header + "\n" + "=" * len(header) + "\n\n")
        for title, kv in all_sections.items():
            f.write(title + "\n" + "-" * len(title) + "\n")
            for k, v in kv.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.6g}\n")
                else:
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
        if notes:
            f.write("Notes\n-----\n")
            for n in notes:
                f.write(f"  - {n}\n")

def save_loss_history_csv(losses_dict, path):
    """Writes the per-epoch loss curves (whatever keys losses_dict has,
    e.g. 'loss','pde','ic','bc','con') to a CSV, one column per key."""
    import csv
    keys = [k for k in losses_dict if len(losses_dict[k]) > 0]
    if not keys:
        return
    n = len(losses_dict[keys[0]])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n):
            writer.writerow([i] + [losses_dict[k][i] for k in keys])

def set_seed(seed: int):
    """Seeds every RNG a training run touches. Called at the START of each
    PINN __init__, before the network is built or any data is
    sampled -- so 'seed=0' and 'seed=1' genuinely give two different,
    independent runs. (Old code called torch.manual_seed(123) once, at
    network.py's *import* time, so this never happened before.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sech(x):
    return 1.0 / np.cosh(x)

def lhs(n, samples):
    """Classic Latin Hypercube Sampling. Same algorithm as the old
    Utils.py::_lhsclassic (the only variant any script actually used --
    Utils.py's 'maximin'/'correlation' criteria and StepLambdaScheduler
    were dead code, never imported by any *.py file, so dropped here)."""
    cut = np.linspace(0, 1, samples + 1)
    u = np.random.rand(samples, n)
    a, b = cut[:samples], cut[1:samples + 1]
    rd = np.zeros_like(u)
    for j in range(n):
        rd[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rd)
    for j in range(n):
        H[:, j] = rd[np.random.permutation(samples), j]
    return H

class _Layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        return self.activation(x) if self.activation is not None else x

class DNN(nn.Module):
    """Feed-forward network with min-max input scaling. Same architecture
    as the old network.py::DNN (Xavier-uniform weights, zero bias)."""

    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(_Layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(_Layer(n_node, n_node, activation))
        self.net.append(_Layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        for layer in self.net:
            x = layer(x)
        return x

def trapz_mean(f):
    """Returns the PROPER trapezoidal-rule domain average of f, sampled at
    N uniformly spaced points INCLUDING both endpoints (e.g. via
    linspace(x_min, x_max, N)) -- a drop-in, still-fully-differentiable
    replacement for f.mean() wherever it's used to approximate
    (1/width)*integral(f dx) (then multiplied by xdiff to get the
    integral itself, same calling convention as before: `xdiff *
    trapz_mean(f)` instead of `xdiff * f.mean()`).

    WHY THIS MATTERS: f.mean() = sum(f)/N, but the domain has only N-1
    INTERVALS (width = (N-1)*dx), not N -- so `xdiff * f.mean()` is
    systematically biased low by an exact factor of (N-1)/N, i.e. a
    -1/N relative error, for ANY integrand, regardless of how accurate
    f itself is. For N=4000 that's -0.025% -- small, but NOT zero, and
    NOT something more training fixes, since it's a property of the
    quadrature rule itself, not of the network. This was caught by
    noticing conservation error stayed nonzero even at t=0, exactly
    where IC supervision should make it negligible -- confirmed by
    computing the SAME -1/N discrepancy independently for the exact
    solution (a hypothetical "perfect" network would show it too).

    Proper trapezoidal average = [sum(f) - 0.5*f[0] - 0.5*f[-1]] / (N-1).
    f can be any shape; averages over the FIRST dimension (the one that
    must correspond to the uniformly-spaced sample axis)."""
    f = f.flatten()
    n = f.shape[0]
    return (f.sum() - 0.5 * f[0] - 0.5 * f[-1]) / (n - 1)

def trapz_mean_batch(f):
    """Batched trapezoidal domain average. f shape (B, N) -> (B,).
    Same rule as trapz_mean, applied independently to each row -- used by
    the batched conservation loss (one forward for all time slices)."""
    n = f.shape[-1]
    return (f.sum(dim=-1) - 0.5 * f[:, 0] - 0.5 * f[:, -1]) / (n - 1)

def save_grid_npz(path, x_grid, t_grid, preds_by_method, u_exact=None):
    """Persists eval_grid()-style data: x_grid, t_grid, one array per
    method, and (if available) u_exact. Loads back with load_grid_npz."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"x_grid": x_grid, "t_grid": t_grid, "_methods": np.array(list(preds_by_method.keys()))}
    for m, arr in preds_by_method.items():
        payload[f"pred__{m}"] = arr
    if u_exact is not None:
        payload["u_exact"] = u_exact
    np.savez_compressed(path, **payload)
    print(f"  saved grid npz -> {path}")

def load_grid_npz(path):
    """Returns (x_grid, t_grid, preds_by_method, u_exact_or_None)."""
    d = np.load(path, allow_pickle=False)
    methods = [str(m) for m in d["_methods"]]
    preds = {m: d[f"pred__{m}"] for m in methods}
    u_exact = d["u_exact"] if "u_exact" in d.files else None
    return d["x_grid"], d["t_grid"], preds, u_exact

def save_curves_npz(path, curves_by_method):
    """Persists eval_conservation_curve()-style data: for each method, a
    dict with 'times','mass','momentum','energy' arrays. Loads back with
    load_curves_npz."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"_methods": np.array(list(curves_by_method.keys()))}
    for m, curve in curves_by_method.items():
        for k, v in curve.items():
            payload[f"{m}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)
    print(f"  saved curves npz -> {path}")

def load_curves_npz(path):
    """Returns curves_by_method dict, same shape eval_conservation_curve() returns."""
    d = np.load(path, allow_pickle=False)
    methods = [str(m) for m in d["_methods"]]
    out = {}
    for m in methods:
        prefix = f"{m}__"
        out[m] = {key[len(prefix):]: d[key] for key in d.files if key.startswith(prefix)}
    return out

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.labelweight": "medium",
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#333333",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

DEFAULT_COLORS = {
    "PINN": "#D55E00", "pinn": "#D55E00",
    "ConPINN": "#0072B2", "conpinn": "#0072B2",
    "Exact": "#000000", "exact": "#000000",
}

DEFAULT_LINESTYLES = {
    "PINN": "-", "pinn": "-",
    "ConPINN": (0, (6, 2)), "conpinn": (0, (6, 2)),
    "Exact": "-", "exact": "-",
}
_FALLBACK_LINESTYLES = ["-", (0, (6, 2)), (0, (1, 1.3)), (0, (4, 1.5, 1, 1.5)), "-."]

def _resolve_colors(methods, colors=None):
    """colors: optional dict {method: matplotlib color}. Falls back to
    DEFAULT_COLORS for PINN/ConPINN, then a print-safe tab10 subset
    for anything else."""
    colors = colors or {}
    tab10 = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))
    out = {}
    for i, m in enumerate(methods):
        out[m] = colors.get(m, DEFAULT_COLORS.get(m, tab10[i]))
    return out

def _resolve_linestyles(methods, linestyles=None):
    linestyles = linestyles or {}
    out = {}
    for i, m in enumerate(methods):
        out[m] = linestyles.get(m, DEFAULT_LINESTYLES.get(m, _FALLBACK_LINESTYLES[i % len(_FALLBACK_LINESTYLES)]))
    return out

def _clean_axes(ax):
    """Remove top/right spines and lighten the remaining ones -- standard
    academic-figure look, avoids the 'boxed-in' default matplotlib style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

def _suffixed_path(path, suffix):
    """Inserts _suffix before the file extension, e.g.
    _suffixed_path('plots/seed0_solution.png', 't0p99') ->
    'plots/seed0_solution_t0p99.png'. Used to turn one multi-panel figure
    into several separate single-panel files, one path per panel."""
    stem, ext = os.path.splitext(path)
    return f"{stem}_{suffix}{ext}"

def plot_loss(losses_dict, path, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    label_map = {"loss": r"Total, $\mathcal{L}$", "pde": r"PDE, $\mathcal{L}_{\mathrm{PDE}}$",
                 "ic": r"Initial, $\mathcal{L}_{\mathrm{IC}}$", "bc": r"Boundary, $\mathcal{L}_{\mathrm{BC}}$",
                 "con": r"Conservation, $\mathcal{L}_{\mathrm{Con}}$", "flux": r"Flux, $\mathcal{L}_{\mathrm{Flux}}$",
                 "match": r"Match, $\mathcal{L}_{\mathrm{Match}}$"}
    for key, label in label_map.items():
        if key in losses_dict and len(losses_dict[key]) > 0:
            ax.plot(losses_dict[key], label=label, linewidth=1.6, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(loc="upper right")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_solution_heatmap(x_grid, t_grid, Z, path, title=None, cbar_label=r"$u(x,t)$"):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.pcolormesh(x_grid, t_grid, Z, cmap="viridis", shading="auto")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=13)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_solution_comparison(x_grid, t_grid, preds_by_method, u_exact, path, title=None, n_slices=4,
                              colors=None, linestyles=None, exact_color=None, figsize=(6.5, 5),
                              xlabel=r"$x$", ylabel=r"$u(x,t)$"):
    """u(x) at a handful of fixed times, all requested methods overlaid
    against Exact (if available). Saves ONE SEPARATE FILE PER TIME SLICE
    (no subplot grid) -- `path` is used as a base name, e.g. passing
    '.../seed0_solution.png' with n_slices=4 produces 4 files:
    '.../seed0_solution_t0p00.png', '..._t0p99.png', '..._t1p98.png',
    '..._t3p00.png'. Returns the list of paths actually written.
    colors / linestyles: optional {method: value} overrides -- pass these
    when re-plotting from saved .npz data if you want a different look
    without retraining (see replot_example.py)."""
    nt = t_grid.shape[0]
    slice_idx = np.linspace(0, nt - 1, n_slices).astype(int)
    methods = list(preds_by_method.keys())
    rcolors = _resolve_colors(methods, colors)
    rstyles = _resolve_linestyles(methods, linestyles)
    exact_color = exact_color or DEFAULT_COLORS["Exact"]

    written = []
    for idx in slice_idx:
        t_val = t_grid[idx, 0]
        fig, ax = plt.subplots(figsize=figsize)
        if u_exact is not None:
            ax.plot(x_grid[idx, :], u_exact[idx, :], "-", color=exact_color, linewidth=2.6,
                    alpha=0.55, label="Exact", zorder=1)
        for method in methods:
            ax.plot(x_grid[idx, :], preds_by_method[method][idx, :], linestyle=rstyles[method],
                    color=rcolors[method], linewidth=2.2, label=method, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        _clean_axes(ax)
        fig.tight_layout()
        out_path = _suffixed_path(path, f"t{t_val:.2f}".replace(".", "p").replace("-", "m"))
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)
    return written

def plot_conservation_comparison(curves_by_method, quantity, true_value, path, title=None, vline=None,
                                  vline_label=None, colors=None, linestyles=None,
                                  quantity_symbol=None, figsize=(7, 5.3)):
    """Saves TWO SEPARATE FILES (no stacked subplot): value-vs-time for
    every method + the true constant, and |deviation|-vs-time (log scale).
    `path` is used as a base name, e.g. passing '.../seed0_cons_mass.png'
    produces '.../seed0_cons_mass_value.png' and
    '.../seed0_cons_mass_deviation.png'. Returns (value_path, deviation_path).
    vline: optional x-position to mark with a vertical dashed line, e.g.
    the training window's t_max in a temporal-extrapolation plot -- makes
    the in-domain vs extrapolation regions visually obvious. The legend is
    always placed ABOVE the axes (not loc="best") specifically so it can
    never collide with the vline's text label, regardless of curve shape.
    colors / linestyles: optional {method: value} overrides (see
    plot_solution_comparison / replot_example.py).
    quantity_symbol: optional LaTeX symbol (e.g. r"$I_1$") to prefer over
    the plain quantity name on the y-axis, e.g. "Mass, $I_1$"."""
    methods = list(curves_by_method.keys())
    rcolors = _resolve_colors(methods, colors)
    rstyles = _resolve_linestyles(methods, linestyles)
    ylab = quantity_symbol or quantity.capitalize()
    n_legend_items = len(methods) + 1

    fig, ax_val = plt.subplots(figsize=figsize)
    for method in methods:
        curve = curves_by_method[method]
        ax_val.plot(curve["times"], curve[quantity], label=method, linewidth=2.2,
                    color=rcolors[method], linestyle=rstyles[method])
    ax_val.axhline(true_value, color=DEFAULT_COLORS["Exact"], linestyle=(0, (8, 3)),
                   linewidth=1.8, alpha=0.7, label="Exact", zorder=0)
    if vline is not None:
        ax_val.axvline(vline, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
        y_lo, y_hi = ax_val.get_ylim()
        ax_val.text(vline, y_lo + 0.03 * (y_hi - y_lo), vline_label or r"$t_{\max}$",
                    fontsize=10, ha="right", va="bottom", rotation=90, color="#666666")
    ax_val.set_xlabel(r"Time, $t$")
    ax_val.set_ylabel(ylab)
    ax_val.legend(loc="best", ncol=1, frameon=True)
    _clean_axes(ax_val)
    fig.tight_layout()
    value_path = _suffixed_path(path, "value")
    fig.savefig(value_path)
    plt.close(fig)

    fig, ax_dev = plt.subplots(figsize=figsize)
    for method in methods:
        curve = curves_by_method[method]
        ax_dev.plot(curve["times"], np.abs(curve[quantity] - true_value), label=method,
                    linewidth=2.2, color=rcolors[method], linestyle=rstyles[method])
    if vline is not None:
        ax_dev.axvline(vline, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
    ax_dev.set_yscale("log")
    ax_dev.set_xlabel(r"Time, $t$")
    ax_dev.set_ylabel(rf"$|$Deviation in {quantity}$|$")
    ax_dev.legend(loc="best", ncol=1, frameon=True)
    _clean_axes(ax_dev)
    fig.tight_layout()
    deviation_path = _suffixed_path(path, "deviation")
    fig.savefig(deviation_path)
    plt.close(fig)

    return value_path, deviation_path

def plot_metric_vs_param(x_values, y_by_method, path, title=None, xlabel="", ylabel="",
                          logx=False, logy=True, yerr_by_method=None, colors=None, linestyles=None):
    """Generic 'metric vs swept parameter' line plot, one line per method,
    optional error bars (e.g. std over seeds). Used by
    noise_sensitivity_study.py (x=noise level), scaling_study.py
    (x=N_f or N_ct), and ablation_summary.py (x=layers/neurons/lambda)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = list(y_by_method.keys())
    rcolors = _resolve_colors(methods, colors)
    rstyles = _resolve_linestyles(methods, linestyles)
    for method in methods:
        yerr = (yerr_by_method or {}).get(method)
        ax.errorbar(x_values, y_by_method[method], yerr=yerr, marker="o", markersize=6,
                    capsize=4, capthick=1.4, linewidth=2.2, linestyle=rstyles[method],
                    color=rcolors[method], label=method)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
