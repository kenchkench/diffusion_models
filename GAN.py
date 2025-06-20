import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Generator: input noise -> output fake data
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# Discriminator: input data -> output real/fake probability
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Initialize models
G = Generator()
D = Discriminator()

# Optimizers
g_opt = optim.Adam(G.parameters(), lr=0.01)
d_opt = optim.Adam(D.parameters(), lr=0.01)

# Loss
criterion = nn.BCELoss()

# Training loop
for epoch in range(1000):
    for _ in range(10):
        # === Train Discriminator ===
        # Real data from N(4,1)
        real_data = torch.randn(32, 1) * 1.0 + 4.0
        real_labels = torch.ones(32, 1)

        # Fake data from Generator
        noise = torch.randn(32, 1)
        fake_data = G(noise)
        fake_labels = torch.zeros(32, 1)

        # Discriminator loss
        d_loss_real = criterion(D(real_data), real_labels)
        d_loss_fake = criterion(D(fake_data.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # === Train Generator ===
        noise = torch.randn(32, 1)
        fake_data = G(noise)
        g_loss = criterion(D(fake_data), real_labels)  # Fool the discriminator

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")


# Generate new samples
with torch.no_grad():
    test_noise = torch.randn(1000, 1)
    gen_samples = G(test_noise).numpy()

# Plot
plt.hist(gen_samples, bins=50, alpha=0.7, label="Generated")
plt.axvline(4, color='r', linestyle='--', label="Target mean")
plt.legend()
plt.title("Generated Samples After Training")
plt.show()
