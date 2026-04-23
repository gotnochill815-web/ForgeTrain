import csv
from datetime import datetime
import os
import time
import yaml
import torch
import csv
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity

from models import get_model
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

def log_metrics(row):
    os.makedirs("logs", exist_ok=True)

    file_path = "logs/train_metrics.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model",
                "epoch",
                "loss",
                "accuracy",
                "train_time",
                "samples_per_sec",
                "device"
            ])

        writer.writerow(row)

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

    model_name = cfg["model"]
    print("Using model:", model_name)

    model = get_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------- PROFILER BLOCK --------
    print("\nRunning profiler on first 10 batches...\n")

    model.train()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True
    ) as prof:

        for step, (images, labels) in enumerate(train_loader):
            if step >= 10:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15
    ))

    print("\nStarting normal training...\n")

    # -------- NORMAL TRAINING --------
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
        log_metrics([
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    model_name,
    epoch + 1,
    round(avg_loss, 4),
    round(acc, 2),
    round(train_elapsed, 2),
    round(throughput, 2),
    str(device)
])

    print("Training complete.")


if __name__ == "__main__":
    train()