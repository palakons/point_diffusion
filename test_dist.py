import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
# make 10 x 10 fig
fig, axs = plt.subplots(10, 10, figsize=(20, 20))
# iterate over each subplot
for i, j in itertools.product(range(10), range(10)):
    data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([5, i+1])),
        torch.distributions.Normal(torch.tensor(
            [-4., 4.]), torch.tensor([1., j+1]))
    )

    dataset = data_distribution.sample(torch.Size([1000, 1]))
    # sns.histplot(dataset[:, 0])
    axs[i, j].hist(dataset[:, 0].numpy(), bins=50, color='blue', alpha=0.7)
    axs[i, j].set_title(f"i={i+1}, j={j+1}")

# save plot
plt.savefig('/home/palakons/from_scratch/data_distribution.png')
