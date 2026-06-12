"""
Plot TinyOPAConvNet training history from opa_tiny.history.json.
Output: report/training_curves.png
"""

import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "backend" / "checkpoints" / "opa_tiny.history.json"
OUTPUT_PATH = ROOT / "report" / "training_curves.png"

with open(HISTORY_PATH, encoding="utf-8") as f:
    history = json.load(f)

epochs     = [d["epoch"]                        for d in history]
train_loss = [d["train"]["loss"]                for d in history]
val_loss   = [d["val"]["loss"]                  for d in history]
train_acc  = [d["train"]["accuracy"]            for d in history]
val_acc    = [d["val"]["accuracy"]              for d in history]
train_f1   = [d["train"]["f1"]                  for d in history]
val_f1     = [d["val"]["f1"]                    for d in history]
train_bal  = [d["train"]["balanced_accuracy"]   for d in history]
val_bal    = [d["val"]["balanced_accuracy"]     for d in history]

fig = plt.figure(figsize=(14, 9))
fig.suptitle("TinyOPAConvNet Training Curves (12 Epochs)", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

METRICS = [
    (gs[0, 0], train_loss,  val_loss,  "Cross-Entropy Loss",  "Loss"),
    (gs[0, 1], train_acc,   val_acc,   "Accuracy",            "Accuracy"),
    (gs[1, 0], train_f1,    val_f1,    "F1 Score",            "F1"),
    (gs[1, 1], train_bal,   val_bal,   "Balanced Accuracy",   "Balanced Acc"),
]

for spec, tr, va, title, ylabel in METRICS:
    ax = fig.add_subplot(spec)
    ax.plot(epochs, tr, "o-", color="#2563eb", linewidth=1.8, markersize=4, label="Train")
    ax.plot(epochs, va, "s-", color="#dc2626", linewidth=1.8, markersize=4, label="Val")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(epochs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.45)
    # annotate best val point
    best_idx = (va.index(min(va)) if ylabel == "Loss" else va.index(max(va)))
    ax.annotate(
        f"best\n{va[best_idx]:.4f}",
        xy=(epochs[best_idx], va[best_idx]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=7, color="#dc2626",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8),
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to: {OUTPUT_PATH}")
plt.show()