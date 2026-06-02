import time

import torch

from torch.utils.data import DataLoader, TensorDataset

device = "cuda"

x = torch.randn(10000, 128, 5)

dataset = TensorDataset(x)

def benchmark(pin_memory):

    loader = DataLoader(

        dataset,

        batch_size=256,

        shuffle=True,

        num_workers=4,

        pin_memory=pin_memory,

    )

    torch.cuda.synchronize()

    start = time.time()

    for (batch,) in loader:

        batch = batch.to(device, non_blocking=True)

        # fake compute

        y = batch * 2

    torch.cuda.synchronize()

    return time.time() - start

print("pin_memory=False:", benchmark(False))

print("pin_memory=True :", benchmark(True))