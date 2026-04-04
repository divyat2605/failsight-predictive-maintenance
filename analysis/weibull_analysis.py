"""
Reliability Analysis: Weibull failure distribution, MTTF, hazard curves
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_PROCESSED_DIR, REPORTS_DIR

try:
    from reliability.Fitters import Fit_Weibull_2P
    from reliability.Probability_plotting import plot_points
    HAS_RELIABILITY = True
except ImportError:
    HAS_RELIABILITY = False
    print("reliability package not installed. Using scipy fallback.")
    from scipy.stats import weibull_min
    from scipy.optimize import curve_fit


def get_failure_cycles(df: pd.DataFrame) -> pd.Series:
    """Extract cycle at failure (max cycle per unit)."""
    return df.groupby("unit")["cycle"].max()


def fit_weibull_scipy(failure_times: np.ndarray):
    """Fallback: fit Weibull using scipy."""
    shape, loc, scale = weibull_min.fit(failure_times, floc=0)
    return shape, scale


def compute_mttf(shape: float, scale: float) -> float:
    """MTTF = scale * Gamma(1 + 1/shape)"""
    from scipy.special import gamma
    return scale * gamma(1 + 1 / shape)


def plot_hazard_curve(shape: float, scale: float, max_cycles: int = 400, subset: str = "FD001"):
    """Plot hazard (failure rate) curve h(t) = (shape/scale) * (t/scale)^(shape-1)"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    t = np.linspace(1, max_cycles, 500)
    h = (shape / scale) * (t / scale) ** (shape - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Weibull Reliability Analysis — {subset}", fontsize=14, fontweight="bold")

    # Hazard curve
    axes[0].plot(t, h, color="#e63946", linewidth=2)
    axes[0].set_title("Hazard Rate h(t)")
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("Failure Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=h.mean(), color="gray", linestyle="--", label=f"Mean h(t) = {h.mean():.4f}")
    axes[0].legend()

    # Reliability curve R(t) = exp(-(t/scale)^shape)
    R = np.exp(-(t / scale) ** shape)
    axes[1].plot(t, R, color="#2a9d8f", linewidth=2)
    axes[1].set_title("Reliability R(t) = P(no failure by cycle t)")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Reliability")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="50% survival")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, f"weibull_{subset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weibull plot saved: {out}")
    return out


def run_weibull_analysis(subset: str = "FD001") -> dict:
    path = os.path.join(DATA_PROCESSED_DIR, f"{subset}_train.parquet")
    if not os.path.exists(path):
        print(f"Data not found for {subset}")
        return {}

    df = pd.read_parquet(path)
    failure_times = get_failure_cycles(df).values.astype(float)

    if HAS_RELIABILITY:
        wbf = Fit_Weibull_2P(failures=failure_times, show_probability_plot=False, print_results=False)
        shape = wbf.beta
        scale = wbf.alpha
    else:
        shape, scale = fit_weibull_scipy(failure_times)

    mttf = compute_mttf(shape, scale)
    plot_path = plot_hazard_curve(shape, scale, subset=subset)

    results = {
        "subset": subset,
        "weibull_shape_beta": round(shape, 4),
        "weibull_scale_alpha": round(scale, 4),
        "mttf_cycles": round(mttf, 2),
        "n_units": len(failure_times),
        "mean_failure_cycle": round(failure_times.mean(), 2),
        "plot_path": plot_path
    }

    print(f"\n[{subset}] Weibull Results:")
    for k, v in results.items():
        if k != "plot_path":
            print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        run_weibull_analysis(subset)