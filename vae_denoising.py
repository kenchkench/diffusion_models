import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class DVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- ENCODER -----
        # Input: 784 (flattened 28x28 image) → Output: 400 (hidden features)
        self.fc1 = nn.Linear(784, 400)

        # Latent space: from 400 → 20 (mean vector)
        self.fc_mu = nn.Linear(400, 20)

        # Latent space: from 400 → 20 (log variance vector)
        self.fc_logvar = nn.Linear(400, 20)

        # ----- DECODER -----
        # Decode: from latent 20 → 400 (hidden layer)
        self.fc2 = nn.Linear(20, 400)

        # Reconstruct: from hidden 400 → 784 (output image vector)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        # Encode input into hidden representation
        h = F.relu(self.fc1(x))
        # Project to mean and log variance of latent distribution
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * logvar)       # convert log variance to std
        eps = torch.randn_like(std)         # sample epsilon from N(0, 1)
        return mu + eps * std               # return sampled z

    def decode(self, z):
        # Decode the latent vector back to image
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))   # output in range [0, 1]

    def forward(self, x):
        # End-to-end forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Add noise to images
def add_noise(x, factor=0.3):
    return torch.clamp(x + factor * torch.randn_like(x), 0., 1.)

# Load MNIST data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train (1 epoch demo)
model.train()
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.view(-1, 784).to(device)
    noisy = add_noise(data)
    recon, mu, logvar = model(noisy)
    loss = vae_loss(recon, data, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_idx == 100: break  # limit for speed

# Prediction
model.eval()
sample = next(iter(train_loader))[0][0].view(-1, 784).to(device)
noisy_sample = add_noise(sample)
with torch.no_grad():
    recon, _, _ = model(noisy_sample)

# Show clean, noisy, and reconstructed
def show(img, title):
    plt.imshow(img.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(9, 3))
show(sample, "Clean")
plt.subplot(1, 3, 2)
show(noisy_sample, "Noisy")
plt.subplot(1, 3, 3)
show(recon, "Reconstructed")
plt.tight_layout()
plt.show()
