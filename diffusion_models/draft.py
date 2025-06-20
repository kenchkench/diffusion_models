# Synthetic labeled data: 2D Gaussians


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def generate_data(n=1000):
    x1 = torch.randn(n, 2) * 0.2 + torch.tensor([-2., 0.])
    x2 = torch.randn(n, 2) * 0.2 + torch.tensor([2., 0.])
    x = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x, y


x, y = generate_data()


print(y.shape)
