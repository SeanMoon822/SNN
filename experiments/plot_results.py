import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bar(df, col, ylabel, title, outpath):
    plt.figure(figsize=(6, 4))
    plt.bar(df.index.str.upper(), df[col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    csv_path = Path("results_summary.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "results_summary.csv not found. Run:\n"
            "  python3 -m experiments.run_all_and_analyze\n"
            "first to generate it."
        )

    df = pd.read_csv(csv_path, index_col=0)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Core bar plots
    plot_bar(df, "test_acc_pct", "Test Accuracy (%)",
             "Accuracy Comparison Across Coding Schemes",
             out_dir / "fig_accuracy.png")

    plot_bar(df, "mean_latency_t", "Mean Latency (timesteps)",
             "Inference Latency Comparison",
             out_dir / "fig_latency.png")

    plot_bar(df, "spikes_per_sample", "Spikes per Sample",
             "Energy Proxy (Spike Count)",
             out_dir / "fig_spikes.png")

    # Robustness (Gaussian)
    plt.figure(figsize=(6, 4))
    sigmas = [0.05, 0.10, 0.20]
    for scheme in df.index:
        y = [
            df.loc[scheme, "gauss_acc_0.05"],
            df.loc[scheme, "gauss_acc_0.1"],
            df.loc[scheme, "gauss_acc_0.2"],
        ]
        plt.plot(sigmas, y, marker="o", label=scheme.upper())

    plt.xlabel("Gaussian Noise σ")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness to Gaussian Noise")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_robustness_gaussian.png", dpi=200)
    plt.close()

    # Robustness (Salt & Pepper)
    plt.figure(figsize=(6, 4))
    ps = [0.01, 0.05, 0.10]
    for scheme in df.index:
        y = [
            df.loc[scheme, "sp_acc_0.01"],
            df.loc[scheme, "sp_acc_0.05"],
            df.loc[scheme, "sp_acc_0.1"],
        ]
        plt.plot(ps, y, marker="o", label=scheme.upper())

    plt.xlabel("Salt & Pepper Probability p")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness to Salt & Pepper Noise")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_robustness_saltpepper.png", dpi=200)
    plt.close()

    # Accuracy vs Energy tradeoff
    plt.figure(figsize=(6, 4))
    for scheme in df.index:
        plt.scatter(df.loc[scheme, "spikes_per_sample"],
                    df.loc[scheme, "test_acc_pct"],
                    s=90, label=scheme.upper())

    plt.xlabel("Spikes per Sample")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Energy Trade-off")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_acc_vs_energy.png", dpi=200)
    plt.close()

    print("Saved figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
