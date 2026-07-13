"""
Plots from the real 150-iteration Build 2 production run (production_run.py),
using the logged iteration_log.csv -- nothing here is synthetic.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("production_run/iteration_log.csv")
zones = range(5)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. FI convergence
ax = axes[0, 0]
ax.plot(df["iter"], df["global_max_fi"], "-", color="darkred", label="Global max FI", linewidth=1.5)
ax.plot(df["iter"], df["global_mean_fi"], "-", color="orange", label="Global mean FI", linewidth=1.5)
ax.axhline(0.8, color="k", linestyle="--", label="Target FI = 0.8")
ax.set_xlabel("Iteration")
ax.set_ylabel("Tsai-Wu Failure Index")
ax.set_title("VAM-FSD Convergence (150 real MYSTRAN solves)")
ax.legend()
ax.grid(alpha=0.3)

# 2. Mass
ax = axes[0, 1]
ax.plot(df["iter"], df["mass_kg"], "-", color="steelblue", linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Structural mass [kg]")
ax.set_title("Mass Convergence")
ax.grid(alpha=0.3)

# 3. Thickness scale per zone
ax = axes[1, 0]
for z in zones:
    ax.plot(df["iter"], df[f"zone{z}_thickness_scale"], label=f"Zone {z}", linewidth=1.2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Thickness scale multiplier")
ax.set_title("Per-Zone Continuous FSD Thickness Resizing")
ax.legend(ncol=5, fontsize=8)
ax.grid(alpha=0.3)

# 4. Ply count per zone
ax = axes[1, 1]
for z in zones:
    ax.step(df["iter"], df[f"zone{z}_total_plies"], where="post", label=f"Zone {z}", linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Total plies (all angles)")
ax.set_title("Per-Zone Discrete Ply-Count Search")
ax.set_ylim(3.5, 5.5)
ax.legend(ncol=5, fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Build 2 Production Run: SU2 pressure -> MYSTRAN -> VAM-FSD, 150 real iterations, 0 failures", fontsize=13)
plt.tight_layout()
plt.savefig("renders/08_production_run_convergence.png", dpi=150)
print("Saved renders/08_production_run_convergence.png")

# Zoomed view of the first 10 iterations (where the interesting transient is)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
early = df[df["iter"] < 10]
axes2[0].plot(early["iter"], early["global_max_fi"], "o-", color="darkred")
axes2[0].axhline(0.8, color="k", linestyle="--", label="Target")
axes2[0].set_xlabel("Iteration")
axes2[0].set_ylabel("Global max FI")
axes2[0].set_title("Early transient: FI (log scale)")
axes2[0].set_yscale("log")
axes2[0].legend()
axes2[0].grid(alpha=0.3)

for z in zones:
    axes2[1].step(early["iter"], early[f"zone{z}_total_plies"], where="post", label=f"Zone {z}")
axes2[1].set_xlabel("Iteration")
axes2[1].set_ylabel("Total plies")
axes2[1].set_title("Ply addition: all 5 zones add 1 ply at iter 1")
axes2[1].legend(ncol=5, fontsize=8)
axes2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("renders/09_production_run_early_transient.png", dpi=150)
print("Saved renders/09_production_run_early_transient.png")

# Summary printout
print("\n--- Summary ---")
print(f"Iterations: {len(df)}")
print(f"Final mass: {df['mass_kg'].iloc[-1]:.4f} kg")
print(f"Final global max FI: {df['global_max_fi'].iloc[-1]:.6f}")
print(f"Final per-zone max FI: {[round(df[f'zone{z}_max_fi'].iloc[-1], 4) for z in zones]}")
print(f"Final thickness scale per zone: {[round(df[f'zone{z}_thickness_scale'].iloc[-1], 4) for z in zones]}")
print(f"Final ply count per zone: {[df[f'zone{z}_total_plies'].iloc[-1] for z in zones]}")
print(f"Total wall time: {df['wall_time_s'].sum():.1f}s, avg {df['wall_time_s'].mean():.3f}s/iter")
