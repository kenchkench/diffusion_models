import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ========== Time Series Segment Definitions ==========
seq_len = 24          # Length of past segment used for prediction
pred_len = 12         # Length of future segment to predict
total_len = seq_len + pred_len

# ========== Model & Training Settings ==========
batch_size = 16
d_model = 32
latent_dim = 16
nhead = 4
num_layers = 2
epochs = 10

# ========== Synthetic Data Generation ==========
def generate_sine_data(n_samples=200):
    x = np.linspace(0, 4 * np.pi, total_len)
    data = [np.sin(x + np.random.rand() * 2 * np.pi) for _ in range(n_samples)]
    data = np.array(data, dtype=np.float32)
    return data[:, :seq_len], data[:, seq_len:]

x_data, y_data = generate_sine_data()
x_data = torch.tensor(x_data).unsqueeze(-1)  # shape: [batch, seq_len, 1]
y_data = torch.tensor(y_data).unsqueeze(-1)  # shape: [batch, pred_len, 1]
dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== Model Definition ==========
class VariationalTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, latent_dim, nhead, num_layers, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len + pred_len, d_model))

        self.encoder_fc = nn.Linear(d_model, 2 * latent_dim)
        self.latent_to_memory = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.pred_len = pred_len
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x):
        x_emb = self.input_proj(x) + self.pos_enc[:, :seq_len]
        pooled = x_emb.mean(dim=1)
        mu, logvar = self.encoder_fc(pooled).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, memory):
        tgt = torch.zeros((memory.size(0), self.pred_len, memory.size(-1)), device=memory.device)
        tgt = tgt + self.pos_enc[:, seq_len:seq_len + self.pred_len]
        dec_out = self.decoder(tgt.transpose(0, 1), memory.unsqueeze(0)).transpose(0, 1)
        return self.output_proj(dec_out)

    def forward(self, src):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        memory = self.latent_to_memory(z)
        output = self.decode(memory)
        return output, mu, logvar

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

# ========== Training ==========
model = VariationalTimeSeriesTransformer(1, d_model, latent_dim, nhead, num_layers, pred_len)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        out, mu, logvar = model(src)
        recon_loss = F.mse_loss(out, tgt)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + 0.001 * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# ========== Prediction ==========
model.eval()
with torch.no_grad():
    sample_input = x_data[:1]
    prediction, _, _ = model(sample_input)
    print("Predicted shape:", prediction.shape)  # [1, pred_len, 1]
    print("Predicted values:", prediction.squeeze().numpy())
