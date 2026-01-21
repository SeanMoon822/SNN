import torch
import torch.nn as nn
import torch.optim as optim
from models.snn_emnist import SNN_EMNIST
from train.train_emnist import train_one_epoch
from train.evaluate import evaluate
from data_loader import get_emnist_loaders   # optional helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SNN_EMNIST(coding="rate").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)