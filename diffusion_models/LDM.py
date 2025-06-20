
# Latent Diffusion Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =======================
# 1. Autoencoder (toy)
# =======================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# =======================
# 2. Score model (latent)
# =======================
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# =======================
# 3. Diffusion functions
# =======================
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(z0, t, noise):
    alpha_bar = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return z0 * alpha_bar.sqrt() + noise * (1 - alpha_bar).sqrt()

# =======================
# 4. Dataset
# =======================
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =======================
# 5. Training
# =======================
ae = Autoencoder().to(device)
score_model = ScoreNet().to(device)
optimizer = torch.optim.Adam(list(ae.parameters()) + list(score_model.parameters()), lr=1e-3)

for epoch in range(5):
    for x, _ in dataloader:
        x = x.to(device)
        z0 = ae.encode(x)  # Encode to latent
        t = torch.randint(0, T, (x.size(0),), device=device)
        noise = torch.randn_like(z0)
        zt = q_sample(z0, t, noise)
        pred_noise = score_model(zt)
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# =======================
# 6. Sampling
# =======================
@torch.no_grad()
def sample(n=10):
    z = torch.randn(n, 4, 7, 7).to(device)  # latent shape
    for t in reversed(range(T)):
        beta = betas[t]
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(z) if t > 0 else 0
        pred_noise = score_model(z)
        z = (1 / alphas[t].sqrt()) * (z - beta / (1 - alpha_bar).sqrt() * pred_noise) + beta.sqrt() * noise
    x_gen = ae.decode(z).cpu()
    return x_gen

samples = sample(10)
grid = torchvision.utils.make_grid(samples, nrow=5)
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()




