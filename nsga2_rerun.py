"""
NSGA-II rerun for the NSTC EV battery project (113-2813-C-130-080-E)

What this script does differently from calu_nsga.py:
  1. Convergence is recorded from the actual run (pymoo save_history=True),
     not typed in by hand.
  2. The GNN "prediction error" objective is removed: it compared an
     untrained network branch (direct_fc) against a hard-coded 0.9.
  3. Two versions of the problem are run side by side:
       - "original": same formulas and dataset-derived bounds as calu_nsga.py
       - "fixed":    each efficiency factor clipped at 0, and C-rate bounds
                     set to the defaults already written in calu_nsga.py
                     (charge 0.1-2, discharge 0.1-3)
  4. Every setting is repeated over 10 random seeds.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

POP, NGEN, SEEDS = 50, 50, range(10)


def dataset_bounds(folder="fix"):
    """Reproduce the bound calculation in calu_nsga.py."""
    means = [pd.read_csv(f).mean(numeric_only=True) for f in glob.glob(f"{folder}/*.csv")]
    tmin = min(m["Battery Temperature_min"] for m in means) - 5
    tmax = max(m["Battery Temperature_max"] for m in means) + 5
    cmin = min(m["Battery Current_min"] for m in means)
    cmax = max(m["Battery Current_max"] for m in means)
    socmin = min(m["min_SoC"] for m in means) / 100
    socmax = max(m["max_SoC"] for m in means) / 100
    xl = np.array([0.1, 0.1, max(15.0, tmin), max(0.1, socmin)])
    xu = np.array([max(2.0, abs(min(0, cmin)) / 10), max(3.0, abs(max(0, cmax)) / 10),
                   min(45.0, tmax), min(0.9, socmax)])
    return xl, xu


def degradation(dod, temp_c):
    """f1 in calu_nsga.py (the unused calendar-aging term is dropped)."""
    k1, k2 = 0.0133, 0.0039
    return (k1 * dod**2 + 0.2 * dod) * (1 + k2 * (temp_c - 25.0))


def efficiency(cr, dr, temp_c, fixed):
    """f2 in calu_nsga.py. In the original, two negative factors can multiply
    into a positive number, which the optimizer exploits."""
    t = 1 - 0.004 * np.abs(temp_c - 25.0)
    c = 1 - 0.03 * cr**2
    d = 1 - 0.025 * dr**2
    if fixed:
        t, c, d = np.clip(t, 0, None), np.clip(c, 0, None), np.clip(d, 0, None)
    return np.clip(0.95 * t * c * d, 0.6, 0.98)


class BatteryProblem(Problem):
    def __init__(self, xl, xu, fixed):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=3, xl=xl, xu=xu)
        self.fixed = fixed

    def _evaluate(self, x, out, *args, **kwargs):
        cr, dr, temp, dod = x.T
        out["F"] = np.column_stack([degradation(dod, temp),
                                    -efficiency(cr, dr, temp, self.fixed)])
        out["G"] = np.column_stack([temp - 40.0, dod - 0.9,
                                    np.where(temp > 35, cr - 1.0, -1.0)])


def run(problem, seed):
    res = minimize(problem, NSGA2(pop_size=POP), ("n_gen", NGEN),
                   seed=seed, save_history=True, verbose=False)
    hv = HV(ref_point=np.array([0.25, -0.55]))
    n_nd = [len(h.opt) for h in res.history]
    hvs = [hv(h.opt.get("F")) for h in res.history]
    return res, np.array(n_nd), np.array(hvs)


if __name__ == "__main__":
    xl_data, xu_data = dataset_bounds()
    xl_def, xu_def = np.array([0.1, 0.1, 15.0, 0.1]), np.array([2.0, 3.0, 45.0, 0.9])
    settings = {"original": BatteryProblem(xl_data, xu_data, fixed=False),
                "fixed": BatteryProblem(xl_def, xu_def, fixed=True)}
    print("Dataset-derived bounds (original):", np.round(xl_data, 2), np.round(xu_data, 2))

    rows, curves, fronts = [], {}, {}
    for name, prob in settings.items():
        nd_all, hv_all = [], []
        for s in SEEDS:
            res, nd, hv = run(prob, s)
            nd_all.append(nd); hv_all.append(hv)
            rows.append({"setting": name, "seed": s, "final_nondominated": nd[-1],
                         "final_hv": hv[-1],
                         "gen_hv_99pct": int(np.argmax(hv >= 0.99 * hv[-1])) + 1})
            if s == 0:
                fronts[name] = (res.X, res.F)
        curves[name] = (np.array(nd_all), np.array(hv_all))

    df = pd.DataFrame(rows)
    df.to_csv("nsga2_rerun_seeds.csv", index=False)
    print(df.groupby("setting")[["final_nondominated", "final_hv", "gen_hv_99pct"]]
            .agg(["mean", "min", "max"]).round(4))
    for name, (X, F) in fronts.items():
        i = np.argmax(-F[:, 1])
        print(f"\n[{name}] seed 0: {len(X)} Pareto solutions")
        print("  best-efficiency solution (charge, discharge, temp, DoD):", np.round(X[i], 2),
              " efficiency:", round(-F[i, 1], 3))
        print("  temp range on front: %.1f-%.1f C, DoD range: %.2f-%.2f"
              % (X[:, 2].min(), X[:, 2].max(), X[:, 3].min(), X[:, 3].max()))

    gens = np.arange(1, NGEN + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, color in [("original", "tab:gray"), ("fixed", "tab:blue")]:
        nd, hv = curves[name]
        ax[0].plot(gens, nd.mean(0), color=color, label=name)
        ax[0].fill_between(gens, nd.min(0), nd.max(0), color=color, alpha=0.2)
        ax[1].plot(gens, hv.mean(0), color=color, label=name)
        ax[1].fill_between(gens, hv.min(0), hv.max(0), color=color, alpha=0.2)
    ax[0].set(title="Non-dominated solutions per generation (10 seeds)", xlabel="Generation",
              ylabel="Count")
    ax[1].set(title="Hypervolume per generation (10 seeds)", xlabel="Generation", ylabel="HV")
    X, F = fronts["fixed"]
    sc = ax[2].scatter(F[:, 0], -F[:, 1], c=X[:, 2], cmap="viridis")
    fig.colorbar(sc, ax=ax[2], label="Temperature (C)")
    ax[2].set(title="Pareto front, fixed version (seed 0)", xlabel="Degradation (lower is better)",
              ylabel="Efficiency (higher is better)")
    for a in ax[:2]:
        a.legend(); a.grid(alpha=0.3)
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("nsga2_rerun.png", dpi=150)
    print("\nSaved nsga2_rerun.png and nsga2_rerun_seeds.csv")
