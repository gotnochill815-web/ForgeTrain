import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleCNN
from data import get_dataloaders


def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train():
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    ckpt_dir = cfg["checkpoint_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        start = time.time()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

        train_elapsed = time.time() - start
        avg_loss = running_loss / total_samples
        throughput = total_samples / train_elapsed

        acc = evaluate(model, test_loader, device)

        path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), path)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.2f}% | "
            f"Train Time: {train_elapsed:.2f}s | "
            f"Samples/s: {throughput:.2f} | "
            f"Workers: 1"
        )

    print("Training complete.")


if __name__ == "__main__":
    train()