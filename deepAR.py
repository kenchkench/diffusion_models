import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data: noisy sine wave
np.random.seed(42)
T_total = 150
data = np.sin(np.linspace(0, 4 * np.pi, T_total)) + np.random.normal(0, 0.1, size=T_total)

# DeepAR parameters
context_length = 30
prediction_length = 14

# Prepare training samples
X = []
Y = []
for i in range(T_total - context_length - prediction_length):
    X.append(data[i : i + context_length])
    Y.append(data[i + context_length : i + context_length + prediction_length])
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Define DeepAR-style model
class DeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=40):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu = nn.Linear(hidden_size, 1)
        self.sigma = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        mu = self.mu(out)
        sigma = torch.exp(self.sigma(out))  # Ensure positivity
        return mu, sigma

# Training
model = DeepAR()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = lambda mu, sigma, y: 0.5 * torch.log(2 * np.pi * sigma**2) + (y - mu)**2 / (2 * sigma**2)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    x_input = X.unsqueeze(-1)  # [B, T, 1]
    y_target = Y.unsqueeze(-1)  # [B, T, 1]
    mu, sigma = model(x_input)
    mu = mu[:, -prediction_length:, :]
    sigma = sigma[:, -prediction_length:, :]
    loss = loss_fn(mu, sigma, y_target).mean()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Forecast last prediction_length points
model.eval()
with torch.no_grad():
    last_context = torch.tensor(data[-context_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    mu, sigma = model(last_context)
    mu_pred = mu[0, -prediction_length:, 0].numpy()
    sigma_pred = sigma[0, -prediction_length:, 0].numpy()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(data, label="Original Data")
future_x = np.arange(len(data), len(data) + prediction_length)
plt.plot(future_x, mu_pred, label="Forecast (mean)")
plt.fill_between(future_x, mu_pred - 2 * sigma_pred, mu_pred + 2 * sigma_pred, alpha=0.3, label="Confidence Interval")
plt.legend()
plt.title("Basic DeepAR Forecast")
plt.grid(True)
plt.show()