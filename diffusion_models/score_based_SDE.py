import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

# Generate toy dataset
data, _ = make_moons(n_samples=1000, noise=0.1)
data = torch.tensor(data, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define noise schedule for VE-SDE
sigma_min = 0.01
sigma_max = 50.0
def sigma(t):
    return sigma_min * (sigma_max / sigma_min) ** t

# Score network
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x, t):
        t_embed = t.view(-1, 1).expand(-1, x.size(1))
        return self.net(torch.cat([x, t_embed], dim=1))

model = ScoreNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train with denoising score matching
for epoch in range(10):
    for batch in dataloader:
        x0 = batch[0]
        t = torch.rand(x0.size(0), 1)
        sig = sigma(t)
        noise = torch.randn_like(x0) * sig
        xt = x0 + noise
        target = -noise / (sig ** 2)
        pred = model(xt, t)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Sampling (reverse-time SDE)
@torch.no_grad()
def sample(model, steps=500, n=500):
    x = torch.randn(n, 2) * sigma_max
    ts = torch.linspace(1., 1e-3, steps)
    dt = -1. / steps
    for t in ts:
        t_tensor = torch.full((n, 1), t)
        g = sigma(t_tensor)
        score = model(x, t_tensor)
        z = torch.randn_like(x)
        # Reverses an SDE using Euler-Maruyama to generate samples from noise
        x = x + (g ** 2) * score * dt + g * np.sqrt(-dt) * z
    return x

# Generate samples
samples = sample(model).numpy()

# Plot
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Training Data")
plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
plt.subplot(1, 2, 2)
plt.title("Generated Samples")
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6)
plt.tight_layout()
plt.show()
