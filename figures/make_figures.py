"""Generate the result figures for the Llama-3-8B summary.

Numbers come straight from the 300-split headline / score-ablation on
results/graded_claims.jsonl (see results_summary_llama3-8b.md). Re-run with:
    python figures/make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OVER, UNDER, NEUTRAL = "#d1495b", "#2a9d8f", "#577590"   # over-budget, safe, neutral
BUDGET = 0.05

# ---------------------------------------------------------------- Figure 1
# Global vs severity-aware on the dangerous tier: risk (left), retention (right).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
labels = ["Single global\nthreshold", "Severity-aware\n(ours)"]

risk = [0.100, 0.046]
bars = ax1.bar(labels, risk, color=[OVER, UNDER], width=0.6)
ax1.axhline(BUDGET, ls="--", color="black", lw=1.2)
ax1.text(1.42, BUDGET + 0.002, "budget 0.05", ha="right", va="bottom", fontsize=9)
ax1.set_title("Dangerous-tier hallucination risk", fontsize=11, weight="bold")
ax1.set_ylabel("risk (lower is safer)")
ax1.set_ylim(0, 0.12)
for b, v in zip(bars, risk):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=10, weight="bold")

ret = [0.968, 0.837]
bars = ax2.bar(labels, ret, color=[NEUTRAL, UNDER], width=0.6)
ax2.set_title("Dangerous-tier retention", fontsize=11, weight="bold")
ax2.set_ylabel("true claims kept (higher is better)")
ax2.set_ylim(0, 1.05)
for b, v in zip(bars, ret):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, weight="bold")

fig.suptitle("Severity-aware control restores the safety guarantee on dangerous claims",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig_main_dangerous.png", dpi=150)
print("wrote figures/fig_main_dangerous.png")

# ---------------------------------------------------------------- Figure 2
# Score ablation: validity holds for every score; retention tracks score quality.
scores = ["P(true)\nAUROC .87", "NLI\nAUROC .71", "SeqLogProb\nAUROC .59", "Random\nAUROC .51"]
abl_risk = [0.046, 0.047, 0.047, 0.047]
abl_ret  = [0.837, 0.384, 0.257, 0.408]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

bars = ax1.bar(scores, abl_risk, color=UNDER, width=0.6)
ax1.axhline(BUDGET, ls="--", color="black", lw=1.2)
ax1.text(3.45, BUDGET + 0.0015, "budget 0.05", ha="right", va="bottom", fontsize=9)
ax1.set_title("Validity holds for every score\n(dangerous risk stays within budget)", fontsize=11, weight="bold")
ax1.set_ylabel("dangerous-tier risk")
ax1.set_ylim(0, 0.08)
for b, v in zip(bars, abl_risk):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9)

bars = ax2.bar(scores, abl_ret, color=[UNDER, NEUTRAL, NEUTRAL, OVER], width=0.6)
ax2.set_title("Retention tracks score quality\n(a better score keeps more truth)", fontsize=11, weight="bold")
ax2.set_ylabel("dangerous-tier retention")
ax2.set_ylim(0, 1.0)
for b, v in zip(bars, abl_ret):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

fig.suptitle("Safety is guaranteed for any score; usefulness depends on the score",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig_ablation.png", dpi=150)
print("wrote figures/fig_ablation.png")
