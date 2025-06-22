import torch
import torch.nn.functional as F

class FlowAttention(torch.nn.Module):
    def __init__(self, dim, proj_phi):
        super().__init__()
        self.proj_phi_q = proj_phi(dim)
        self.proj_phi_k = proj_phi(dim)
        self.scale = dim ** -0.5

    def forward(self, Q, K, V, mask=None):
        phi_q = self.proj_phi_q(Q)     # [B, N, D]
        phi_k = self.proj_phi_k(K)     # [B, M, D]

        # Compute outgoing flow normalization per source:
        O = phi_k / (phi_k.sum(dim=1, keepdim=True) + 1e-6)
        # Compute incoming flow normalization per sink:
        I = phi_q / (phi_q.sum(dim=2, keepdim=True) + 1e-6)

        V_b = F.softmax((O * V), dim=1)
        A = torch.bmm(I, torch.bmm(phi_k.transpose(-2,-1), V_b))
        R = torch.sigmoid((phi_q.sum(dim=2, keepdim=True))) * A
        return R

class FlowformerLayer(torch.nn.Module):
    def __init__(self, dim, proj_phi, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = FlowAttention(dim, proj_phi)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_dim, dim),
            torch.nn.Dropout(dropout)
        )
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x2 = self.attn(x, x, x)
        x = x + x2; x = self.norm1(x)
        x2 = self.mlp(x)
        return x + x2 if self.norm2 is None else self.norm2(x + x2)

class FlowformerTS(torch.nn.Module):
    def __init__(self, input_dim=1, dim=64, depth=4, proj_phi=torch.relu, mlp_dim=128):
        super().__init__()
        self.embed = torch.nn.Linear(input_dim, dim)
        self.layers = torch.nn.ModuleList([
            FlowformerLayer(dim, proj_phi, mlp_dim) for _ in range(depth)
        ])
        self.pred = torch.nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, 1]
        x = self.embed(x)  # → [B, T, D]
        for l in self.layers:
            x = l(x)
        return self.pred(x[:, -1, :])  # predict next value

# Training loop
model = FlowformerTS()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    for batch in dataloader:  # provide (input_seq, target_val)
        x, y = batch  # x: [B, T, 1], y: [B, 1]
        y_hat = model(x)
        loss = criterion(y_hat, y)
        opt.zero_grad(); loss.backward(); opt.step()
