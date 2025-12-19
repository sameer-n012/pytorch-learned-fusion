import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "output/eval"
OUTPUT_DIR = "output/plots"


def main():
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

    plt.grid()

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

    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_compile.png")
    plt.show()


if __name__ == "__main__":
    main()
