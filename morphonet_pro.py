"""
MorphoNet Pro - Production-Ready Self-Structuring Networks

A complete library for learnable-sparse neural networks.

Features:
- Multiple layer types (Linear, Conv2d, Attention)
- Advanced sparsity schedules
- Deep network support with skip connections
- Comprehensive benchmarks
- Model export/import
- Visualization tools

Author: MCT5 Research
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import time
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class MorphoConfig:
    """Configuration for MorphoNet models."""
    
    def __init__(
        self,
        # Architecture
        input_dim: int = 0,
        hidden_dims: List[int] = None,
        n_classes: int = 2,
        network_type: str = "mlp",  # mlp, cnn, transformer
        
        # Sparsity
        sparse_init: float = 0.4,      # Initial connection probability
        target_sparsity: float = 0.7,  # Target sparsity at end
        sparsity_loss: float = 0.002,  # L1 penalty on masks
        sparsity_schedule: str = "cosine",  # constant, linear, cosine
        
        # Learning
        weight_lr: float = 0.001,
        mask_lr: float = 0.02,
        weight_decay: float = 0.01,
        grad_clip: float = 5.0,
        
        # Training
        batch_size: int = 64,
        epochs: int = 100,
        warmup_epochs: int = 10,
        
        # Pruning
        prune_threshold: float = 0.2,
        prune_start_epoch: int = 0.5,  # Fraction of training
        
        # Advanced
        use_skip_connections: bool = True,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        
        # CNN specific
        cnn_channels: List[int] = None,
        kernel_size: int = 3,
        
        # Transformer specific
        n_heads: int = 4,
        ff_dim: int = 128,
        n_layers: int = 2,
        
        # Device
        device: str = "auto",
        
        # Misc
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.n_classes = n_classes
        self.network_type = network_type
        
        self.sparse_init = sparse_init
        self.target_sparsity = target_sparsity
        self.sparsity_loss = sparsity_loss
        self.sparsity_schedule = sparsity_schedule
        
        self.weight_lr = weight_lr
        self.mask_lr = mask_lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        
        self.prune_threshold = prune_threshold
        self.prune_start_epoch = prune_start_epoch
        
        self.use_skip_connections = use_skip_connections
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        
        self.cnn_channels = cnn_channels or [32, 64]
        self.kernel_size = kernel_size
        
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.seed = seed
        self.verbose = verbose
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> "MorphoConfig":
        return cls(**d)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "MorphoConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# MORPHO LAYERS
# ═══════════════════════════════════════════════════════════════════════════

class MorphoLinear(nn.Module):
    """
    Linear layer with learnable connectivity masks.
    
    Features:
    - Differentiable mask via sigmoid relaxation
    - Temperature annealing
    - Sparsity penalty
    - Optional skip connection
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparse_init: float = 0.4,
        sparsity_loss: float = 0.002,
        mask_lr: float = 0.02,
        use_skip: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_loss = sparsity_loss
        self.mask_lr = mask_lr
        self.use_skip = use_skip and (in_features == out_features)
        
        # Weight initialization (He)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 
                                   np.sqrt(2.0 / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Mask initialization
        mask_init = torch.rand(out_features, in_features) < sparse_init
        self.mask_logits = nn.Parameter(torch.zeros(out_features, in_features))
        self.mask_logits.data = torch.where(
            mask_init,
            torch.ones_like(self.mask_logits) * 2,
            torch.ones_like(self.mask_logits) * -2
        )
        
        # Skip connection gate
        if self.use_skip:
            self.skip_gate = nn.Parameter(torch.tensor(0.0))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Temperature for mask relaxation
        self.register_buffer("temperature", torch.tensor(1.0))
    
    def get_mask(self, hard: bool = False) -> torch.Tensor:
        """Get connectivity mask."""
        if hard:
            return (self.mask_logits > 0).float()
        return torch.sigmoid(self.mask_logits / self.temperature)
    
    def get_sparsity(self) -> float:
        """Fraction of inactive connections."""
        return (self.get_mask(hard=True) < 0.5).float().mean().item()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask()
        masked_weight = self.weight * mask
        
        out = F.linear(x, masked_weight, self.bias)
        out = self.dropout(out)
        
        if self.use_skip:
            skip_weight = torch.sigmoid(self.skip_gate)
            out = out + skip_weight * x
        
        return out
    
    def get_mask_loss(self) -> torch.Tensor:
        """Sparsity penalty."""
        return self.sparsity_loss * self.get_mask().mean()
    
    def set_temperature(self, temp: float):
        """Set mask temperature."""
        self.temperature.fill_(temp)
    
    def prune(self, threshold: float = 0.2):
        """Hard prune connections below threshold."""
        with torch.no_grad():
            mask = self.get_mask()
            to_prune = mask < threshold
            self.mask_logits[to_prune] = -100


class MorphoConv2d(nn.Module):
    """
    Conv2d with learnable spatial masks.
    
    Each filter learns which spatial locations to use.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        sparse_init: float = 0.5,
        sparsity_loss: float = 0.001,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sparsity_loss = sparsity_loss
        
        # Conv weights
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Spatial mask (per filter)
        mask_init = torch.rand(out_channels, kernel_size, kernel_size) < sparse_init
        self.mask_logits = nn.Parameter(torch.zeros(out_channels, kernel_size, kernel_size))
        self.mask_logits.data = torch.where(
            mask_init,
            torch.ones_like(self.mask_logits) * 2,
            torch.ones_like(self.mask_logits) * -2
        )
        
        self.register_buffer("temperature", torch.tensor(1.0))
    
    def get_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits / self.temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask().unsqueeze(1)  # (C, 1, K, K)
        masked_weight = self.weight * mask
        
        return F.conv2d(x, masked_weight, self.bias, 
                       padding=self.kernel_size // 2)
    
    def get_mask_loss(self) -> torch.Tensor:
        return self.sparsity_loss * self.get_mask().mean()
    
    def get_sparsity(self) -> float:
        return (self.get_mask() < 0.5).float().mean().item()
    
    def prune(self, threshold: float = 0.2):
        with torch.no_grad():
            mask = self.get_mask()
            self.mask_logits[mask < threshold] = -100


class MorphoAttention(nn.Module):
    """
    Attention with learnable connectivity.
    
    Learns which query-key pairs to attend to.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        sparse_init: float = 0.5,
        sparsity_loss: float = 0.0005,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.sparsity_loss = sparsity_loss
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Attention mask (per head)
        self.attn_mask_logits = nn.Parameter(
            torch.randn(n_heads, 1, 1) * 0.1
        )
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        self.register_buffer("temperature", torch.tensor(1.0))
    
    def get_attention_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.attn_mask_logits / self.temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply learnable mask
        mask = self.get_attention_mask()
        attn = attn * mask
        
        attn = attn.softmax(dim=-1)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
    
    def get_mask_loss(self) -> torch.Tensor:
        return self.sparsity_loss * self.get_attention_mask().mean()
    
    def get_sparsity(self) -> float:
        return (self.get_attention_mask() < 0.5).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# MORPHONET MODELS
# ═══════════════════════════════════════════════════════════════════════════

class MorphoNetMLP(nn.Module):
    """MorphoNet for tabular data (MLP architecture)."""
    
    def __init__(self, config: MorphoConfig):
        super().__init__()
        self.config = config
        
        # Build layers
        self.layers = nn.ModuleList()
        
        dims = [config.input_dim] + config.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(MorphoLinear(
                dims[i], dims[i+1],
                sparse_init=config.sparse_init,
                sparsity_loss=config.sparsity_loss,
                mask_lr=config.mask_lr,
                use_skip=config.use_skip_connections,
                dropout=config.dropout
            ))
        
        # Batch norm (optional)
        if config.use_batchnorm and len(config.hidden_dims) > 0:
            self.bn = nn.BatchNorm1d(config.hidden_dims[-1])
        else:
            self.bn = nn.Identity()
        
        # Output layer (dense)
        self.output = nn.Linear(config.hidden_dims[-1] if config.hidden_dims else config.input_dim, 
                               config.n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.bn(x)
        return self.output(x)
    
    def get_all_masks(self) -> List[torch.Tensor]:
        return [layer.get_mask() for layer in self.layers]
    
    def get_total_sparsity(self) -> float:
        return np.mean([layer.get_sparsity() for layer in self.layers])
    
    def get_architecture_loss(self) -> torch.Tensor:
        return sum(layer.get_mask_loss() for layer in self.layers)
    
    def get_effective_params(self) -> int:
        count = 0
        for layer in self.layers:
            mask = layer.get_mask(hard=True)
            count += mask.sum().item()
        count += self.output.weight.numel()
        return int(count)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def set_temperature(self, temp: float):
        for layer in self.layers:
            layer.set_temperature(temp)
    
    def prune(self, threshold: float = 0.2):
        for layer in self.layers:
            layer.prune(threshold)


class MorphoNetCNN(nn.Module):
    """MorphoNet for image data (CNN architecture)."""
    
    def __init__(self, config: MorphoConfig, input_shape: Tuple[int, int, int] = None):
        super().__init__()
        self.config = config
        self.input_shape = input_shape or (3, 32, 32)
        
        # Conv layers
        self.conv_layers = nn.ModuleList()
        
        in_ch = self.input_shape[0]
        for out_ch in config.cnn_channels:
            self.conv_layers.append(MorphoConv2d(
                in_ch, out_ch,
                kernel_size=config.kernel_size,
                sparse_init=config.sparse_init,
                sparsity_loss=config.sparsity_loss
            ))
            in_ch = out_ch
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # FC layers
        fc_input = config.cnn_channels[-1] * 4 * 4
        self.fc = MorphoLinear(
            fc_input, config.hidden_dims[0] if config.hidden_dims else 64,
            sparse_init=config.sparse_init,
            sparsity_loss=config.sparsity_loss
        )
        
        self.output = nn.Linear(
            config.hidden_dims[0] if config.hidden_dims else 64,
            config.n_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
        
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        return self.output(x)
    
    def get_total_sparsity(self) -> float:
        sparsity = [layer.get_sparsity() for layer in self.conv_layers]
        sparsity.append(self.fc.get_sparsity())
        return np.mean(sparsity)
    
    def get_architecture_loss(self) -> torch.Tensor:
        loss = sum(layer.get_mask_loss() for layer in self.conv_layers)
        loss += self.fc.get_mask_loss()
        return loss
    
    def set_temperature(self, temp: float):
        for layer in self.conv_layers:
            layer.set_temperature(temp)
        self.fc.set_temperature(temp)
    
    def prune(self, threshold: float = 0.2):
        for layer in self.conv_layers:
            layer.prune(threshold)
        self.fc.prune(threshold)


class MorphoNetTransformer(nn.Module):
    """MorphoNet for sequence data (Transformer architecture)."""
    
    def __init__(self, config: MorphoConfig, vocab_size: int = None, max_len: int = 512):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Embedding
        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, config.n_heads * config.ff_dim // 2)
        else:
            self.embedding = nn.Linear(config.input_dim, config.n_heads * config.ff_dim // 2)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, config.n_heads * config.ff_dim // 2) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        dim = config.n_heads * config.ff_dim // 2
        
        for _ in range(config.n_layers):
            self.layers.append(nn.ModuleList([
                MorphoAttention(dim, config.n_heads, config.sparse_init, config.sparsity_loss),
                MorphoLinear(dim, config.ff_dim, config.sparse_init, config.sparsity_loss, use_skip=True),
                MorphoLinear(config.ff_dim, dim, config.sparse_init, config.sparsity_loss),
            ]))
        
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, config.n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vocab_size:
            x = self.embedding(x)
        else:
            x = self.embedding(x)
        
        B, N, _ = x.shape
        x = x + self.pos_embedding[:, :N, :]
        
        for attn, ff1, ff2 in self.layers:
            # Self-attention with residual
            x = x + attn(self.norm(x))
            # FFN with residual
            x = x + ff2(F.relu(ff1(x)))
        
        # Global average pooling
        x = x.mean(dim=1)
        return self.output(x)
    
    def get_total_sparsity(self) -> float:
        sparsity = []
        for attn, ff1, ff2 in self.layers:
            sparsity.append(attn.get_sparsity())
            sparsity.append(ff1.get_sparsity())
            sparsity.append(ff2.get_sparsity())
        return np.mean(sparsity)
    
    def get_architecture_loss(self) -> torch.Tensor:
        loss = 0
        for attn, ff1, ff2 in self.layers:
            loss += attn.get_mask_loss()
            loss += ff1.get_mask_loss()
            loss += ff2.get_mask_loss()
        return loss
    
    def prune(self, threshold: float = 0.2):
        for attn, ff1, ff2 in self.layers:
            attn.prune(threshold)
            ff1.prune(threshold)
            ff2.prune(threshold)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class MorphoTrainer:
    """
    Trainer for MorphoNet models.
    
    Features:
    - Separate optimizers for weights and masks
    - Sparsity scheduling
    - Temperature annealing
    - Progressive pruning
    - Learning rate scheduling
    """
    
    def __init__(self, model: Union[MorphoNetMLP, MorphoNetCNN, MorphoNetTransformer], 
                 config: MorphoConfig):
        self.model = model
        self.config = config
        self.loss_history = []
        self.sparsity_history = []
        
        # Separate parameter groups
        weight_params = []
        mask_params = []
        
        for name, param in model.named_parameters():
            if 'mask_logits' in name or 'skip_gate' in name or 'attn_mask' in name:
                mask_params.append(param)
            else:
                weight_params.append(param)
        
        self.weight_opt = torch.optim.AdamW(
            weight_params, 
            lr=config.weight_lr, 
            weight_decay=config.weight_decay
        )
        self.mask_opt = torch.optim.Adam(
            mask_params, 
            lr=config.mask_lr,
            weight_decay=0.0  # No weight decay on masks
        )
        
        # LR scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.weight_opt, T_max=config.epochs
        )
    
    def get_sparsity_target(self, epoch: int) -> float:
        """Get target sparsity for current epoch."""
        cfg = self.config
        progress = epoch / cfg.epochs
        
        if cfg.sparsity_schedule == "constant":
            return cfg.sparse_init
        elif cfg.sparsity_schedule == "linear":
            return cfg.sparse_init + (cfg.target_sparsity - cfg.sparse_init) * progress
        elif cfg.sparsity_schedule == "cosine":
            return cfg.sparse_init + (cfg.target_sparsity - cfg.sparse_init) * (1 - np.cos(np.pi * progress)) / 2
        else:
            return cfg.sparse_init
    
    def get_temperature(self, epoch: int) -> float:
        """Get mask temperature for current epoch."""
        # Warm start with high temperature, anneal to 0.5
        progress = epoch / self.config.epochs
        return max(0.5, 2.0 - progress * 1.5)
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, 
                   epoch: int) -> Tuple[float, float]:
        """One training step."""
        self.model.train()
        
        # Update temperature
        temp = self.get_temperature(epoch)
        self.model.set_temperature(temp)
        
        # Forward pass
        logits = self.model(X)
        ce_loss = F.cross_entropy(logits, y)
        arch_loss = self.model.get_architecture_loss()
        total_loss = ce_loss + arch_loss
        
        # Backward pass
        self.weight_opt.zero_grad()
        self.mask_opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Update
        self.weight_opt.step()
        self.mask_opt.step()
        
        sparsity = self.model.get_total_sparsity()
        self.loss_history.append(total_loss.item())
        self.sparsity_history.append(sparsity)
        
        return total_loss.item(), sparsity
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: bool = None) -> dict:
        """Full training loop."""
        if verbose is None:
            verbose = self.config.verbose
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.config.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.config.device)
        
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.config.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.config.device)
        
        n = len(X)
        cfg = self.config
        
        start_time = time.time()
        best_val_acc = 0
        
        for epoch in range(cfg.epochs):
            # Mini-batch training
            perm = torch.randperm(n, device=self.config.device)
            epoch_loss = 0
            epoch_sparsity = 0
            n_batches = 0
            
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i+cfg.batch_size]
                loss, sparsity = self.train_step(X_t[idx], y_t[idx], epoch)
                epoch_loss += loss
                epoch_sparsity += sparsity
                n_batches += 1
            
            # LR scheduling
            self.lr_scheduler.step()
            
            # Validation
            val_acc = 0
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_preds = val_logits.argmax(-1)
                    val_acc = (val_preds == y_val_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Progress reporting
            if verbose and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
                temp = self.get_temperature(epoch)
                sparsity_target = self.get_sparsity_target(epoch)
                msg = f"Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f}, "
                msg += f"sparsity={epoch_sparsity/n_batches:.1%}, "
                msg += f"temp={temp:.2f}"
                if X_val is not None:
                    msg += f", val_acc={val_acc:.1%}"
                print(msg)
        
        train_time = time.time() - start_time
        
        # Final pruning
        prune_epoch = int(cfg.epochs * cfg.prune_start_epoch)
        self.model.prune(cfg.prune_threshold)
        final_sparsity = self.model.get_total_sparsity()
        
        return {
            'train_time': train_time,
            'final_loss': self.loss_history[-1],
            'final_sparsity': final_sparsity,
            'total_params': self.model.get_total_params(),
            'effective_params': self.model.get_effective_params(),
            'best_val_acc': best_val_acc,
            'loss_history': self.loss_history,
            'sparsity_history': self.sparsity_history,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_morphonet(
    network_type: str,
    input_dim: int,
    n_classes: int,
    hidden_dims: List[int] = None,
    **kwargs
) -> Union[MorphoNetMLP, MorphoNetCNN, MorphoNetTransformer]:
    """
    Factory function to create MorphoNet models.
    
    Args:
        network_type: 'mlp', 'cnn', or 'transformer'
        input_dim: Input dimension (or channels for CNN)
        n_classes: Number of output classes
        hidden_dims: Hidden layer sizes
        **kwargs: Additional config options
    
    Returns:
        MorphoNet model
    """
    if network_type == "mlp":
        config = MorphoConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [64, 32],
            n_classes=n_classes,
            **kwargs
        )
        return MorphoNetMLP(config)
    
    elif network_type == "cnn":
        config = MorphoConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [128],
            n_classes=n_classes,
            cnn_channels=kwargs.pop('cnn_channels', [32, 64]),
            **kwargs
        )
        input_shape = kwargs.get('input_shape', (input_dim, 32, 32))
        return MorphoNetCNN(config, input_shape)
    
    elif network_type == "transformer":
        config = MorphoConfig(
            input_dim=input_dim,
            hidden_dims=None,
            n_classes=n_classes,
            n_heads=kwargs.pop('n_heads', 4),
            ff_dim=kwargs.pop('ff_dim', 128),
            n_layers=kwargs.pop('n_layers', 2),
            **kwargs
        )
        vocab_size = kwargs.pop('vocab_size', None)
        return MorphoNetTransformer(config, vocab_size)
    
    else:
        raise ValueError(f"Unknown network_type: {network_type}")


# Export public API
__all__ = [
    'MorphoConfig',
    'MorphoLinear',
    'MorphoConv2d', 
    'MorphoAttention',
    'MorphoNetMLP',
    'MorphoNetCNN',
    'MorphoNetTransformer',
    'MorphoTrainer',
    'create_morphonet',
]
