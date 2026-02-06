import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models.snn_emnist import SNN_EMNIST
from train.train_emnist import train_one_epoch
from train.evaluate import evaluate, add_gaussian_noise, add_salt_pepper


# Data

def get_emnist_loaders(batch_size=64, val_ratio=0.15, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),  # maps [0,1] -> [-1,1]
    ])

    full_train = torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=False, download=True, transform=transform
    )

    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader



# Plot helpers

def zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col = out[c].to_numpy(dtype=float)
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if np.isfinite(sd) and sd > 0:
            out[c] = (out[c] - mu) / sd
        else:
            out[c] = np.nan
    return out


def heatmap(df: pd.DataFrame, title: str, outfile: str, normalize: bool = True):
    data = zscore_cols(df) if normalize else df

    fig, ax = plt.subplots(figsize=(1.2 * data.shape[1] + 3, 0.9 * data.shape[0] + 2))
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(data.index))

    raw = df.to_numpy(dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(raw[i, j]):
                ax.text(j, i, f"{raw[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)



# Run one coding scheme

def run_one(
    coding: str,
    train_loader,
    val_loader,
    test_loader,
    device,
    time_steps=10,
    hidden_dim=256,
    num_classes=47,
    tau_out=2.0,
    lr=1e-3,
    epochs=24,
):
    model = SNN_EMNIST(
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        coding=coding,
        tau_out=tau_out,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_m = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{coding.upper()}] Epoch {epoch:02d}/{epochs} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% "
              f"lat={val_m['mean_latency_t']:.2f} spk={val_m['spikes_per_sample']:.1f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Clean test
    test_loss, test_acc, test_m = evaluate(model, test_loader, criterion, device, return_latency_dist=True)

    row = {
        "coding": coding,
        "test_loss": test_loss,
        "test_acc_pct": test_acc * 100.0,
        "mean_latency_t": float(test_m["mean_latency_t"]),
        "spikes_per_sample": float(test_m["spikes_per_sample"]),
        "throughput_sps": float(test_m["throughput_sps"]),
    }
    if "median_latency_t" in test_m:
        row["median_latency_t"] = float(test_m["median_latency_t"])
    if "early_decision_rate" in test_m:
        row["early_decision_rate"] = float(test_m["early_decision_rate"])

    # Robustness (store noisy acc and drops)
    clean_acc_pct = row["test_acc_pct"]

    for sig in [0.05, 0.10, 0.20]:
        _, acc_n, _m_n = evaluate(
            model, test_loader, criterion, device,
            noise_fn=lambda x, s=sig: add_gaussian_noise(x, s, clamp_min=-1.0, clamp_max=1.0),
        )
        acc_n_pct = acc_n * 100.0
        row[f"gauss_acc_{sig}"] = acc_n_pct
        row[f"gauss_drop_{sig}"] = clean_acc_pct - acc_n_pct

    for p in [0.01, 0.05, 0.10]:
        _, acc_n, _m_n = evaluate(
            model, test_loader, criterion, device,
            noise_fn=lambda x, pp=p: add_salt_pepper(x, pp, low=-1.0, high=1.0),
        )
        acc_n_pct = acc_n * 100.0
        row[f"sp_acc_{p}"] = acc_n_pct
        row[f"sp_drop_{p}"] = clean_acc_pct - acc_n_pct

    return row


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparams
    batch_size = 64
    lr = 1e-3
    epochs = 24
    time_steps = 10
    hidden_dim = 256
    num_classes = 47
    tau_out = 2.0

    train_loader, val_loader, test_loader = get_emnist_loaders(batch_size=batch_size)

    rows = []
    for coding in ["rate", "ttfs", "phase", "burst"]:
        print("\n" + "=" * 70)
        print("CODING =", coding)
        rows.append(run_one(
            coding, train_loader, val_loader, test_loader, device,
            time_steps=time_steps, hidden_dim=hidden_dim, num_classes=num_classes,
            tau_out=tau_out, lr=lr, epochs=epochs
        ))

    df = pd.DataFrame(rows).set_index("coding")
    df.to_csv("results_summary.csv")
    print("\nSaved results_summary.csv")

    # Heatmaps
    core_cols = ["test_acc_pct", "mean_latency_t", "spikes_per_sample", "throughput_sps"]
    heatmap(df[core_cols], "Core Metrics (raw labels; colors normalized)", "heatmap_core_metrics.png", normalize=True)

    drop_cols = ["gauss_drop_0.05", "gauss_drop_0.1", "gauss_drop_0.2",
                 "sp_drop_0.01", "sp_drop_0.05", "sp_drop_0.1"]
    heatmap(df[drop_cols], "Robustness: Accuracy Drop vs Clean (lower is better)", "heatmap_robustness_drops.png", normalize=True)

    print("Saved heatmaps:")
    print(" - heatmap_core_metrics.png")
    print(" - heatmap_robustness_drops.png")
    print("\nPreview:")
    print(df[core_cols + drop_cols].round(3))


if __name__ == "__main__":
    main()
