# Exploratory EU x AU decision grids (3x3): mean SMOTE G-mean gain per cell.
# Usage: python3 make_heatmap.py results/real_gains.csv manuscript/figures
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
r = pd.read_csv(sys.argv[1]); out = sys.argv[2]
g = r.gain_SMOTE_GMEAN
ab, cuts = pd.qcut(r.AUsb_bal, 3, labels=["low", "medium", "high"], retbins=True)
order = ["high", "medium", "low"]
COL = {"no": "#F4C7C3", "maybe": "#FBE3A6", "yes": "#BFE3BF"}
LAB = {"no": "do not\noversample", "maybe": "uncertain", "yes": "oversample"}
SPEC = [("EUsb_min", [0.17, 0.27], "Epistemic measure $\\mathrm{EU}^{\\mathrm{sb}}_{\\min}$", "(a)"),
        ("EUsb_diff", [0.06, 0.20], "Epistemic difference measure $\\mathrm{EU}^{\\mathrm{sb}}_{\\Delta}$", "(b)")]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.4))
for ax, (m, c, xl, tag) in zip(axes, SPEC):
    labs = [f"$\\leq${c[0]:.2f}", f"{c[0]:.2f} to {c[1]:.2f}", f"$>${c[1]:.2f}"]
    eb = pd.cut(r[m], [-1, c[0], c[1], 9], labels=labs)
    D = pd.DataFrame({"g": g, "w": g > 0.02, "e": eb, "a": ab})
    M = D.groupby(["a", "e"], observed=False).agg(mean=("g", "mean"), share=("w", "mean"), n=("g", "size"))
    mean = M["mean"].unstack().loc[order]; share = M["share"].unstack().loc[order]; n = M["n"].unstack().loc[order]
    for i in range(3):
        for j in range(3):
            sh = share.values[i, j]; k = "no" if sh < 0.25 else ("yes" if sh > 0.75 else "maybe")
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=COL[k], edgecolor="black", lw=0.8))
            ax.text(j, i - 0.08, LAB[k], ha="center", va="center", fontsize=10.5, fontweight="bold")
            ax.text(j, i + 0.30, f"gain {mean.values[i, j]:.2f}, {sh*100:.0f}% worth\n(n = {int(n.values[i, j])})", ha="center", va="center", fontsize=7.5)
    ax.set_xlim(-.5, 2.5); ax.set_ylim(2.5, -.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(order)
    ax.set_xlabel(f"{tag} {xl}"); ax.set_ylabel("Aleatoric measure $\\mathrm{AU}^{\\mathrm{sb}}_{\\mathrm{bal}}$")
    for sp in ax.spines.values(): sp.set_visible(False)
    share.round(2).to_csv(f"{out}/heatmap_share_{m}.csv"); mean.round(3).to_csv(f"{out}/heatmap_mean_{m}.csv"); n.to_csv(f"{out}/heatmap_n_{m}.csv")
    print(m); print(share.round(2)); print(mean.round(3)); print(n)
fig.tight_layout(); fig.savefig(f"{out}/fig_real_heatmap.pdf")
