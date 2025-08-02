import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ====== Settings ======
seq_len = 24
label_len = 12
pred_len = 12
d_model = 32
num_features = 5
batch_size = 16
epochs = 5

# ====== Generate Dummy Data ======
def generate_dummy_data(n_samples=200):
    data = np.sin(np.linspace(0, 10, seq_len + pred_len))[None, :]  # basic pattern
    data = data + np.random.randn(n_samples, seq_len + pred_len) * 0.1
    data = np.expand_dims(data, -1)  # [B, T, 1]
    data = np.repeat(data, num_features, axis=-1).astype(np.float32)
    return data[:, :seq_len], data[:, seq_len:]

x_data, y_data = generate_dummy_data()
x_tensor = torch.tensor(x_data)
y_tensor = torch.tensor(y_data)

loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size)

# ====== Model ======
class AleatoricTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.query_embed = nn.Parameter(torch.randn(label_len + pred_len, d_model))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.out_mean = nn.Linear(d_model, num_features)
        self.out_logvar = nn.Linear(d_model, num_features)

    def forward(self, x):
        x = self.input_proj(x)              # [B, seq_len, d_model]
        x = x.permute(1, 0, 2)              # [seq_len, B, d_model]
        memory = self.encoder(x)            # [seq_len, B, d_model]

        tgt = self.query_embed.unsqueeze(1).repeat(1, x.shape[1], 1)  # [label_len+pred_len, B, d_model]
        out = self.decoder(tgt, memory)     # [label_len+pred_len, B, d_model]
        out = out.permute(1, 0, 2)          # [B, label_len+pred_len, d_model]
        mean = self.out_mean(out)           # [B, label_len+pred_len, num_features]
        logvar = self.out_logvar(out)
        return mean, logvar

# ====== Loss Function ======
def aleatoric_loss(mean, logvar, target):
    mean = mean[:, -pred_len:]        # only last pred_len
    logvar = logvar[:, -pred_len:]
    target = target                   # [B, pred_len, num_features]
    precision = torch.exp(-logvar)
    loss = precision * (target - mean) ** 2 + logvar
    return loss.mean()

# ====== Training ======
model = AleatoricTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        mean, logvar = model(xb)
        loss = aleatoric_loss(mean, logvar, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# ====== Prediction ======
model.eval()
with torch.no_grad():
    test_input = x_tensor[:1]  # one sample
    pred_mean, pred_logvar = model(test_input)
    forecast = pred_mean[:, -pred_len:]
    print("Forecast Mean:\n", forecast)
    print("Forecast LogVar:\n", pred_logvar[:, -pred_len:])
