'''
classifier guided diffusion model
https://arxiv.org/pdf/2105.05233
'''



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# For simplicity, generate 2D toy data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 100  # number of diffusion steps
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 128),  # x and time
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, t):
        t_emb = t.unsqueeze(1).float() / T
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)  # predict noise

#  Classifier (2 classes)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # logits for 2 classes
        )
    
    def forward(self, x):
        return self.net(x)


# Synthetic labeled data: 2D Gaussians
def generate_data(n=1000):
    x1 = torch.randn(n, 2) * 0.2 + torch.tensor([-2., 0.])
    x2 = torch.randn(n, 2) * 0.2 + torch.tensor([2., 0.])
    x = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x.to(device), y.to(device)

# Models
denoise_model = DenoiseNet().to(device)
classifier = Classifier().to(device)

# Optimizers
opt_denoise = torch.optim.Adam(denoise_model.parameters(), lr=1e-3)
opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# Training denoiser
for epoch in range(5):
    x0, _ = generate_data()
    t = torch.randint(0, T, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].unsqueeze(1)
    xt = x0 * alpha_t.sqrt() + noise * (1 - alpha_t).sqrt()
    
    pred_noise = denoise_model(xt, t)
    loss = F.mse_loss(pred_noise, noise)

    opt_denoise.zero_grad()
    loss.backward()
    opt_denoise.step()
    print(f"Epoch {epoch+1} Denoiser Loss: {loss.item():.4f}")

# Training classifier on clean data
for epoch in range(10):
    x0, y = generate_data()
    logits = classifier(x0)
    loss = F.cross_entropy(logits, y)
    opt_classifier.zero_grad()
    loss.backward()
    opt_classifier.step()
    print(f"Epoch {epoch+1} Classifier Loss: {loss.item():.4f}")


# @torch.no_grad()
def guided_sample(n=100, target_class=1, guidance_scale=2.5):
    x = torch.randn(n, 2).to(device)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long, device=device)
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        x.requires_grad_(True)
        pred_noise = denoise_model(x, t_batch)

        # Classifier guidance
        logits = classifier(x)
        log_prob = F.log_softmax(logits, dim=1)
        selected = log_prob[:, target_class].sum()
        grad = torch.autograd.grad(selected, x)[0]

        # Modify noise prediction
        guided_noise = pred_noise - guidance_scale * beta * grad

        noise = torch.randn_like(x) if t > 0 else 0
        x = (1 / alphas[t].sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * guided_noise) + beta.sqrt() * noise
        x = x.detach()  # Detach from graph for next step
    return x.cpu()

samples = guided_sample(n=1000, target_class=1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.title("Classifier-Guided Samples (Class 1)")
plt.axis('equal')
plt.show()
