import os
import time
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import SimpleCNN
from data import get_dataloaders


MASTER_IP = "172.16.0.80"
MASTER_PORT = "29511"


def train_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_IP
    os.environ["MASTER_PORT"] = MASTER_PORT

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}",
        rank=rank,
        world_size=world_size
    )

    batch_size = 64
    epochs = 3

    train_loader, _ = get_dataloaders(batch_size=batch_size)

    sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
    train_loader.dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2,
    persistent_workers=True
)

    model = SimpleCNN()
    model = DDP(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        start = time.time()
        total_loss = 0.0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {total_loss/total:.4f} | "
                f"Time: {time.time()-start:.2f}s | "
                f"Workers: {world_size}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    world_size = 2

    spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )