import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Data
X, _ = make_moons(n_samples=1000, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)

# Noise schedule
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Model
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        t_embed = t.unsqueeze(1).float() / T
        return self.net(x + t_embed)

model = ScoreNet()
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
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")


# DDPM Sampling (stochastic)
@torch.no_grad()
def ddpm_sample(model, n=500):
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        alpha_bar_t = alphas_cumprod[t]
        beta = betas[t]

        pred_noise = model(x, t_batch)
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

        noise = torch.randn_like(x) if t > 0 else 0
        x = (1 / alphas[t].sqrt()) * (x - beta / (1 - alpha_bar_t).sqrt() * pred_noise) + beta.sqrt() * noise
    return x

# Plot
samples = ddpm_sample(model)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label='Original')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Generated')
plt.legend()
plt.title("Original vs Diffusion Generated")
plt.axis("equal")
plt.show()
