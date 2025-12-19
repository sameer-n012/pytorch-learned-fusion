import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "output/eval"
OUTPUT_DIR = "output/plots"


def main():
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
    ax.set_title("Learned Fusion Compile Time vs Model Size")

    ax.set_xscale("log")

    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learned_scatter.png")
    plt.show()


if __name__ == "__main__":
    main()
