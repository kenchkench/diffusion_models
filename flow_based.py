import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Toy dataset: 2D moons
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=10000, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)

# DataLoader
loader = torch.utils.data.DataLoader(X, batch_size=256, shuffle=True)

# Affine coupling layer (RealNVP-style)
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        self.translate = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if not reverse:
            s = self.scale(x1)
            t = self.translate(x1)
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            s = self.scale(x1)
            t = self.translate(x1)
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
        return torch.cat([x1, y2], dim=1), log_det

# Full model
class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(2, 128),
            CouplingLayer(2, 128)
        ])
    
    def f(self, x):
        log_det = torch.zeros(x.size(0))
        for i, layer in enumerate(self.layers):
            x, ld = layer(x, reverse=False)
            log_det += ld
        return x, log_det

    def f_inv(self, z):
        for i, layer in reversed(list(enumerate(self.layers))):
            z, _ = layer(z, reverse=True)
        return z

# Instantiate
model = FlowModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    for x_batch in loader:
        z, log_det = model.f(x_batch)
        log_prob = -0.5 * z.pow(2).sum(dim=1) - np.log(2 * np.pi)
        loss = -(log_prob + log_det).mean() # this is to maximize P(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Sampling
@torch.no_grad()
def sample(model, n=1000):
    z = torch.randn(n, 2)
    x = model.f_inv(z)
    return x.numpy()

# Plot generated samples
samples = sample(model)
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=2)
plt.title("Generated Samples")
plt.show()
