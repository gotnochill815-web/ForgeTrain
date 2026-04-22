import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity

from models import SimpleCNN
from data import get_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 128

    train_loader, _ = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\nRunning profiler on first 10 batches...\n")

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
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

    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20
        )
    )


if __name__ == "__main__":
    main()