import multiprocessing
import os
import torch.distributed as dist
from torch.multiprocessing import spawn


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    print(f"Worker {rank} initialized")

    dist.destroy_process_group()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    spawn(
        worker,
        args=(2,),
        nprocs=2,
        join=True
    )