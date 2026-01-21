def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total