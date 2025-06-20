

#  Denoising Diffusion Implicit Models


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Create dataset
data, _ = make_moons(n_samples=1000, noise=0.1)
data = torch.tensor(data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(data), batch_size=128, shuffle=True)

# Define model
class SimpleNet(nn.Module):
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
        t = t.unsqueeze(1).float() / T
        t = t.expand(-1, x.shape[1])
        x_in = torch.cat([x, t], dim=1)
        return self.net(x_in)

# Diffusion schedule
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Model setup
model = SimpleNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5):
    for batch in dataloader:
        x0 = batch[0]
        t = torch.randint(0, T, (x0.size(0),))
        noise = torch.randn_like(x0)
        alpha_t = alphas_cumprod[t].unsqueeze(1)
        xt = x0 * alpha_t.sqrt() + noise * (1 - alpha_t).sqrt()
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# DDIM Sampling
@torch.no_grad()
def ddim_sample(model, n=500, steps=20):
    x = torch.randn(n, 2)
    step_indices = torch.linspace(T - 1, 0, steps, dtype=torch.long)

    for i in range(steps - 1):
        t = step_indices[i]
        t_next = step_indices[i + 1]
        t_batch = torch.full((n,), t, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_next = alphas_cumprod[t_next]

        pred_noise = model(x, t_batch)
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

        # Deterministic update, no noise added
        x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * pred_noise

    return x


# Generate samples
samples = ddim_sample(model)

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
plt.title("DDIM Generated Samples")
plt.show()
