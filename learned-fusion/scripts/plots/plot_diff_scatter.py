import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Prepare data
# -------------------------
cols = ["model", "score_fusion_type", "run_time_mean", "num_params"]
df_sub = df[cols].copy()

# Force numeric
df_sub[["run_time_mean", "num_params"]] = df_sub[["run_time_mean", "num_params"]].apply(
    pd.to_numeric, errors="coerce"
)

# Drop invalid rows
df_sub = df_sub.dropna(subset=["run_time_mean", "num_params"])

# Pivot so fusion types become columns
pivot = df_sub.pivot_table(
    index="model", columns="score_fusion_type", values="run_time_mean", aggfunc="mean"
)

# One num_params value per model
num_params = df_sub.groupby("model")["num_params"].first()

# Keep only models with all needed fusion types
valid_models = pivot.dropna(subset=["learned", "default", "random"]).index

pivot = pivot.loc[valid_models]
num_params = num_params.loc[valid_models]

# -------------------------
# Compute differences
# -------------------------
delta_ld = pivot["learned"] - pivot["default"]
delta_lr = pivot["learned"] - pivot["random"]

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    num_params,
    delta_ld,
    label="learned − default",
    color="tab:blue",
    s=70,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)

ax.scatter(
    num_params,
    delta_lr,
    label="learned − random",
    color="tab:orange",
    s=70,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)

# Reference line
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters")
ax.set_ylabel("Δ Inference Runtime (s)")
ax.set_title("Runtime Difference vs Model Size")

ax.legend()

plt.tight_layout()
plt.show()
