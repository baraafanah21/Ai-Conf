"""Statistical analysis of the raw results from all four systems.

Reads results/raw/*.csv, then for each comparison in the priority matrix
applies exactly the protocol of Rahman et al. 2024 (arXiv:2408.02825):

    - Percentage change:  P = (mean_other - mean_baseline) / |mean_baseline| * 100
    - Mann-Whitney U test (two-sided, alpha = 0.05)
    - Cliff's delta effect size, with the paper's magnitude bands
      (negligible <= 0.147 < small <= 0.33 < medium <= 0.474 < large)
    - A difference counts as significant ONLY if p < alpha AND the effect
      size is not negligible.

Each (comparison, metric) cell is classified in the style of the paper's
Tables 3/4: Zero variability / Non-zero, insignificant / Non-zero, significant.

Outputs (in results/analysis/):
    comparison_results.csv   full stats for every comparison x metric
    summary_table.md         Table 3/4-style classification + headline numbers
    pct_change.png           percentage-change chart (poster-ready)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FILES = {
    "ubuntu": "ubuntu_raw.csv",
    "win11_machine1": "win11_m1_raw.csv",
    "fedora": "fedora_raw.csv",
    "win11_machine2": "win11_m2_raw.csv",
}

# (baseline, other, description) — the priority matrix from CLAUDE.md.
COMPARISONS = [
    ("ubuntu", "win11_machine1", "Ubuntu vs Win11 (machine 1, identical hardware)"),
    ("fedora", "win11_machine2", "Fedora vs Win11 (machine 2, identical hardware)"),
    ("win11_machine1", "win11_machine2", "Win11 vs Win11 (across machines)"),
    ("ubuntu", "fedora", "Ubuntu vs Fedora (across machines)"),
]

METRICS = {
    "accuracy": ("Accuracy", "performance"),
    "f1_macro": ("Macro F1", "performance"),
    "per_image_ms": ("Inference time (ms/image)", "processing time"),
    "total_time_s": ("Total run time (s)", "processing time"),
    "peak_rss_mb": ("Peak RAM (MB)", "expense"),
    "cpu_percent": ("CPU utilization (%)", "expense"),
}

ZERO = "Zero variability"
INSIG = "Non-zero, insignificant"
SIG = "Non-zero, significant"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """delta = P(x > y) - P(x < y), computed from the Mann-Whitney U statistic."""
    n1, n2 = len(x), len(y)
    u1 = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * u1 / (n1 * n2) - 1.0


def delta_magnitude(d: float) -> str:
    ad = abs(d)
    if ad <= 0.147:
        return "negligible"
    if ad <= 0.33:
        return "small"
    if ad <= 0.474:
        return "medium"
    return "large"


def compare(base: np.ndarray, other: np.ndarray, alpha: float) -> dict:
    if np.unique(np.concatenate([base, other])).size == 1:
        return {"pct_change": 0.0, "p_value": np.nan, "cliffs_delta": 0.0,
                "magnitude": "negligible", "category": ZERO}
    mean_b, mean_o = base.mean(), other.mean()
    pct = (mean_o - mean_b) / abs(mean_b) * 100.0 if mean_b != 0 else np.nan
    p = mannwhitneyu(base, other, alternative="two-sided").pvalue
    d = cliffs_delta(other, base)  # positive delta = other stochastically larger
    mag = delta_magnitude(d)
    significant = (p < alpha) and (mag != "negligible")
    return {"pct_change": pct, "p_value": p, "cliffs_delta": d,
            "magnitude": mag, "category": SIG if significant else INSIG}


def make_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Percentage-change bars, one panel per comparison; significant bars in
    the accent blue, non-significant in neutral gray (poster-ready, light surface)."""
    BLUE, GRAY, INK, MUTED = "#2a78d6", "#b8b7ae", "#1a1a19", "#6b6a63"
    comparisons = df["comparison"].unique()
    ncols = 2
    nrows = int(np.ceil(len(comparisons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    for ax_idx, comp in enumerate(comparisons):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[df["comparison"] == comp]
        labels = [METRICS[m][0] for m in sub["metric"]]
        vals = sub["pct_change"].to_numpy(dtype=float)
        sig = (sub["category"] == SIG).to_numpy()
        colors = [BLUE if s else GRAY for s in sig]

        y = np.arange(len(labels))
        ax.barh(y, vals, height=0.55, color=colors, zorder=3)
        ax.axvline(0, color=INK, linewidth=1, zorder=4)
        # Pad both sides so direct labels land inside the axes, clear of tick labels.
        finite = vals[np.isfinite(vals)]
        vmin, vmax = min(0.0, finite.min()), max(0.0, finite.max())
        pad = (vmax - vmin) * 0.18 or 1.0
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_yticks(y, labels=labels, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_title(comp, fontsize=10, color=INK, loc="left", pad=10)
        ax.set_xlabel("% change vs baseline", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color="#e8e7e0", linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d0cfc6")
        # Direct labels on the significant bars only.
        for yi, (v, s) in enumerate(zip(vals, sig)):
            if s and np.isfinite(v):
                ha = "left" if v >= 0 else "right"
                off = max(abs(vals[np.isfinite(vals)]).max(), 1e-9) * 0.02
                ax.text(v + (off if v >= 0 else -off), yi, f"{v:+.1f}%",
                        va="center", ha=ha, fontsize=8, color=INK)

    for ax_idx in range(len(comparisons), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    fig.legend(handles, ["Significant (p < 0.05, non-negligible effect)", "Not significant"],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("OS-induced change in DL inference metrics (physical machines)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path, alpha: float) -> None:
    lines = ["# Summary — OS-Induced Variability on Physical Machines", ""]
    lines += [f"Protocol: Mann-Whitney U (alpha = {alpha}) + Cliff's delta; "
              "significant = p < alpha AND effect size non-negligible "
              "(identical to Rahman et al. 2024).", ""]

    # Table 3/4-style classification: rows = metric, columns = comparison.
    comps = df["comparison"].unique()
    lines.append("## Classification per metric and comparison")
    lines.append("")
    header = "| Metric | " + " | ".join(comps) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(comps) + 1))
    for metric, (label, _group) in METRICS.items():
        row = [label]
        for comp in comps:
            cell = df[(df["comparison"] == comp) & (df["metric"] == metric)]
            row.append(cell["category"].iloc[0] if len(cell) else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Headline numbers: share of comparisons significant, per metric group.
    lines.append("## Headline numbers (share of comparisons with significant change)")
    lines.append("")
    for group in ("performance", "processing time", "expense"):
        metrics = [m for m, (_l, g) in METRICS.items() if g == group]
        sub = df[df["metric"].isin(metrics)]
        if len(sub) == 0:
            continue
        n_sig = (sub["category"] == SIG).sum()
        lines.append(f"- **{group}**: {n_sig}/{len(sub)} comparison-metric cells significant "
                     f"({100.0 * n_sig / len(sub):.0f}%)")
    lines.append("")
    lines.append("Reference point: Rahman et al. 2024 found significant processing-time "
                 "change in 96-100% of projects but performance change in only ~20-23%, "
                 "on virtualized CI environments.")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out-dir", default="results/analysis")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for label, fname in FILES.items():
        path = raw_dir / fname
        if path.exists():
            data[label] = pd.read_csv(path)
            print(f"loaded {label}: {len(data[label])} runs ({path})")
        else:
            print(f"missing {path} — comparisons involving '{label}' will be skipped")

    rows = []
    for base, other, desc in COMPARISONS:
        if base not in data or other not in data:
            continue
        for metric in METRICS:
            b = data[base][metric].to_numpy(dtype=float)
            o = data[other][metric].to_numpy(dtype=float)
            stats = compare(b, o, args.alpha)
            rows.append({
                "comparison": desc, "baseline": base, "other": other, "metric": metric,
                "mean_baseline": b.mean(), "mean_other": o.mean(),
                "std_baseline": b.std(ddof=1), "std_other": o.std(ddof=1),
                "n_baseline": len(b), "n_other": len(o), **stats,
            })

    if not rows:
        raise SystemExit("No comparisons possible — need at least two raw CSV files.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_results.csv", index=False)
    write_summary(df, out_dir / "summary_table.md", args.alpha)
    make_chart(df, out_dir / "pct_change.png")

    print(f"\nWrote {out_dir / 'comparison_results.csv'}")
    print(f"Wrote {out_dir / 'summary_table.md'}")
    print(f"Wrote {out_dir / 'pct_change.png'}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n", df[["comparison", "metric", "pct_change", "p_value",
                        "cliffs_delta", "magnitude", "category"]].to_string(index=False))


if __name__ == "__main__":
    main()
