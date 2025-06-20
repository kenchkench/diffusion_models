


# Denoising score matching

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist, batch_size=128, shuffle=True)

# Define score network
class ScoreNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

# Denoising Score Matching loss
def dsm_loss(score_model, x, sigma):
    noise = torch.randn_like(x) * sigma
    x_tilde = x + noise
    score_est = score_model(x_tilde)
    score_true = -(x_tilde - x) / (sigma ** 2)
    return ((score_est - score_true) ** 2).sum(dim=1).mean()

# Training
device = torch.device("cpu")
model = ScoreNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
sigma = 0.3
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for x, _ in train_loader:
        x = x.view(-1, 784).to(device)
        loss = dsm_loss(model, x, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
