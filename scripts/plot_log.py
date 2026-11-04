"""Plot CE loss and grad-norm from a distill training log.
Usage: python scripts/plot_log.py train.log out.png [--window 20]
Parses lines like:  step 12 lr 5.0e-05 gn 8.68 {'ce': 2.55} 173s
"""
import argparse, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (one per measure)

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("out"); ap.add_argument("--window", type=int, default=20)
args = ap.parse_args()

rows = [(int(m[1]), float(m[2]), float(m[3])) for l in open(args.log)
        if (m := re.search(r"step (\d+) .*?gn ([\d.]+) .*?'ce': ([\d.]+)", l))]
step, gn, ce = (np.array(x) for x in zip(*rows))


def smooth(y, w):  # trailing mean (only uses past steps); shorter window at the start
    return np.array([y[max(0, i - w + 1): i + 1].mean() for i in range(len(y))])


plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK})
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor=SURFACE,
                         gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
for ax, y, color, title, ylab in (
        (axes[0], ce, BLUE, "Distillation loss (cross-entropy on assistant tokens)", "CE"),
        (axes[1], gn, ORANGE, "Gradient norm (before clipping at 1.0)", "grad norm")):
    ax.set_facecolor(SURFACE)
    ax.plot(step, y, color=color, lw=1, alpha=0.30)
    ax.plot(step, smooth(y, args.window), color=color, lw=2.2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylab, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    ax.text(0.99, 0.95, f"{args.window}-step mean at end: {smooth(y, args.window)[-1]:.2f}",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
axes[1].set_xlabel("optimizer step", color=INK2)
fig.text(0.01, -0.01, f"Thick line = trailing {args.window}-step mean; faint line = per-step value "
         "(logged from the last micro-batch of each step only, so noisy).", color=INK2, fontsize=8.5, ha="left")
fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
n = args.window
print(f"CE  first {n}: {ce[:n].mean():.2f}   last {n}: {ce[-n:].mean():.2f}")
print(f"gn  first {n}: {gn[:n].mean():.2f}   last {n}: {gn[-n:].mean():.2f}   spikes>6.5 at steps {step[gn > 6.5].tolist()}")
