import torch
import torch.nn as nn
import time
from mct5.engine import MCT5
from mct5.config import MCT5Config

# Setup 10k batch size
batch_size = 10000
D = 64
X = torch.randn(batch_size, D)

# Baseline MLP (3 layers)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP()
# Warmup
mlp(X)
t0 = time.time()
for _ in range(10):
    mlp(X)
t_mlp = (time.time() - t0) / 10

# MCT5
config = MCT5Config(D=D, r=16, n_classes=2, input_dim=D)
mct = MCT5(config)
mct.initialize()

# Warmup
mct.forward(X)
t0 = time.time()
for _ in range(10):
    mct.forward(X)
t_mct = (time.time() - t0) / 10

print(f"MLP (3 layer D={D}): {t_mlp * 1000:.2f} ms / pass")
print(f"MCT5 (D={D}, r=16):  {t_mct * 1000:.2f} ms / pass")
print(f"Ratio: {t_mct / t_mlp:.2f}x")
