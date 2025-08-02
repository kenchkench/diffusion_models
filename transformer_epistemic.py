import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ========== Time Series Segment Definitions ==========
seq_len = 24          # Past segment length
pred_len = 12         # Future to predict
total_len = seq_len + pred_len
num_features = 1

# ========== Model & Training Settings ==========
batch_size = 16
d_model = 32
latent_dim = 16
epochs = 10

# ========== Dummy Time Series Data ==========
def generate_sine_data(n_samples=200):
    x = np.linspace(0, 4 * np.pi, total_len)
    data = [np.sin(x + np.random.rand() * 2 * np.pi) for _ in range(n_samples)]
    data = np.array(data)
    return data[..., None]  # shape (n_samples, total_len, 1)

data = generate_sine_data()
x_enc = data[:, :seq_len, :]
x_dec_target = data[:, seq_len:, :]

dataset = TensorDataset(torch.Tensor(x_enc), torch.Tensor(x_dec_target))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== Transformer-based VAE ==========
class TimeSeriesVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * num_features, d_model),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, d_model)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, pred_len * num_features),
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        out = self.decoder(h)
        return out.view(-1, pred_len, num_features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

# ========== Loss Function ==========
def vae_loss(pred, target, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(pred, target, reduction='mean')
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + kl, recon_loss, kl

# ========== Training Loop ==========
model = TimeSeriesVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        out, mu, logvar = model(xb)
        loss, recon_loss, kl = vae_loss(out, yb, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

# ========== Dummy Prediction ==========
model.eval()
with torch.no_grad():
    example = torch.Tensor(data[0:1, :seq_len, :])  # shape (1, seq_len, 1)
    pred, mu, logvar = model(example)
    print("Predicted future sequence:\n", pred.squeeze().numpy())
