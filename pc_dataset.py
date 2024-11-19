from torch.utils.data import Dataset
import os
import numpy as np
import torch


class PointCloudDataset(Dataset):
    def __init__(self, data_dir, N, M=1):
        self.files = [os.path.join(
            data_dir, f"{i}/{i:06}.txt") for i in range(min(M, 546))]
        self.N = N
        self.mu, self.sigma = None, None  # Normalization parameters
        self.data = []
        for fname in self.files:
            new_data = np.loadtxt(fname, usecols=(0, 1, 2), skiprows=2)
            # if the data is too long, remove the extra points
            if len(new_data) > self.N:
                # print how many points are removed
                print("remove", len(new_data) - self.N)
                new_data = new_data[:self.N]
            else:  # randomly sample the data rows to make it N
                print("add", self.N - len(new_data))
                new_data = np.vstack(
                    [new_data, new_data[np.random.choice(len(new_data), self.N - len(new_data))]])
            assert len(new_data) == self.N, "Data length must be N"
            self.data.append(new_data)

        self.all_points = np.vstack(self.data)
        self.mu, self.sigma = self.compute_normalization()
        self.data = [(d - self.mu) / self.sigma for d in self.data]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def compute_normalization(self):
        self.mu = np.mean(self.all_points, axis=0)
        self.sigma = np.std(self.all_points, axis=0)
        return self.mu, self.sigma
