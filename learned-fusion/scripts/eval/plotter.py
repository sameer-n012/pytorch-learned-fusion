import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "output/eval"
OUTPUT_DIR = "output/plots"


def plot_models_bar():
    df = pd.read_csv(f"{DATA_DIR}/evaluation_results_final.csv")

    model_names = [
        "microsoft/deberta-v3-base",
        "microsoft/deberta-v3-large",
        "microsoft/deberta-v2-xlarge",
    ]

    pretty_model_names = [
        "DeBERTa-v3 Base",
        "DeBERTa-v3 Large",
        "DeBERTa-v2 XLarge",
    ]

    fusion_types = ["default", "learned", "random"]
    colors = ["tab:blue", "tab:green", "tab:red"]

    df_sub = df[df["model"].isin(model_names)].copy()

    num_cols = ["compile_time", "run_time_mean", "run_time_se"]
    df_sub[num_cols] = df_sub[num_cols].apply(pd.to_numeric, errors="coerce")

    df_sub["model"] = pd.Categorical(
        df_sub["model"], categories=model_names, ordered=True
    )
    df_sub["score_fusion_type"] = pd.Categorical(
        df_sub["score_fusion_type"], categories=fusion_types, ordered=True
    )

    df_sub = df_sub.sort_values(["model", "score_fusion_type"])

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (fusion, color) in enumerate(zip(fusion_types, colors)):
        data = df_sub[df_sub["score_fusion_type"] == fusion]
        ax.bar(
            x + i * width - width,
            data["run_time_mean"] * 1000,
            width,
            yerr=data["run_time_se"] * 1000,
            capsize=4,
            label=fusion,
            color=color,
        )

    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("Inference Latency by Model and Score Fusion Type")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_model_names)
    ax.legend(title="Score Fusion Type")

    # plt.grid()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_inference.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (fusion, color) in enumerate(zip(fusion_types, colors)):
        data = df_sub[df_sub["score_fusion_type"] == fusion]
        ax.bar(
            x + i * width - width,
            data["compile_time"],
            width,
            label=fusion,
            color=color,
        )

    ax.set_ylabel("Compile Time (s)")
    ax.set_title("Compile Time by Model and Score Fusion Type")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_model_names)
    ax.legend(title="Score Fusion Type")

    # plt.grid()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_compile.png")
    plt.show()


def plot_models_scatter():
    df = pd.read_csv(f"{DATA_DIR}/evaluation_results_final.csv")

    df_sub = df[df["score_fusion_type"] == "learned"].copy()

    num_cols = ["run_time_mean", "num_params"]
    df_sub[num_cols] = df_sub[num_cols].apply(pd.to_numeric, errors="coerce")

    df_sub = df_sub.dropna(subset=num_cols)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df_sub["num_params"] * 1e6,
        df_sub["run_time_mean"] * 1000,
        s=60,
        alpha=0.75,
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("Learned Fusion Inference Time vs Model Size")

    ax.set_xscale("log")

    ax.tick_params(axis="both", which="major", labelsize=10)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learned_scatter.png")
    plt.show()


def plot_diff_scatter():
    df = pd.read_csv(f"{DATA_DIR}/evaluation_results_final.csv")

    cols = ["model", "score_fusion_type", "run_time_mean", "num_params"]
    df_sub = df[cols].copy()

    df_sub[["run_time_mean", "num_params"]] = df_sub[
        ["run_time_mean", "num_params"]
    ].apply(pd.to_numeric, errors="coerce")

    df_sub = df_sub.dropna(subset=["run_time_mean", "num_params"])

    pivot = df_sub.pivot_table(
        index="model",
        columns="score_fusion_type",
        values="run_time_mean",
        aggfunc="mean",
    )

    num_params = df_sub.groupby("model")["num_params"].first()

    valid_models = pivot.dropna(subset=["learned", "default", "random"]).index

    pivot = pivot.loc[valid_models]
    num_params = num_params.loc[valid_models]

    delta_ld = pivot["learned"] - pivot["default"]
    delta_lr = pivot["learned"] - pivot["random"]

    print(
        delta_ld.nsmallest(
            5,
            keep="first",
        )
        * 1000
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        num_params * 1e6,
        delta_ld * 1e3,
        label="Learned Fusion − Default Fusion",
        color="tab:blue",
        s=70,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.scatter(
        num_params * 1e6,
        delta_lr * 1e3,
        label="Learned Fusion − Random Fusion",
        color="tab:red",
        s=70,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Inference Runtime Difference (ms)")
    ax.set_title("Runtime Difference vs Model Size")

    ax.legend()

    plt.tight_layout()
    # plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/diff_scatter.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_models_bar()
    plot_models_scatter()
    plot_diff_scatter()

    df = pd.read_csv(f"{DATA_DIR}/evaluation_results_final.csv")
