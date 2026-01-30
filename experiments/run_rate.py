import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models.snn_emnist import SNN_EMNIST
from train.train_emnist import train_one_epoch
from train.evaluate import evaluate


def get_emnist_loaders(batch_size=64, val_ratio=0.15, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    full_train = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )

    test_set = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters 
    batch_size = 64
    learning_rate = 1e-3
    epochs = 24

    time_steps = 10
    hidden_dim = 256
    num_classes = 47
    tau_out = 2.0

    train_loader, val_loader, test_loader = get_emnist_loaders(batch_size=batch_size)

    model = SNN_EMNIST(
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        coding="rate",
        tau_out=tau_out,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_state = None

    print("\nTraining SNN with coding = rate")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[RATE] Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Load best model based on validation accuracy
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[RATE] Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
