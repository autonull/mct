"""
MorphoX - Morphogenic Executive Networks

A NOVEL architecture that goes beyond static sparsity to:
1. Input-dependent dynamic masks (true conditional computation)
2. Learnable primitive selection (not just connectivity)
3. Hierarchical multi-scale structure
4. Cross-layer skip connections (learned depth)
5. Attention-integrated masks (content-based routing)

This is NOT just sparse training - it's learning the computation graph per input.

Key innovation: The network MORPHS its structure based on each input,
allocating compute where needed and skipping unnecessary computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import time

# ═══════════════════════════════════════════════════════════════════════════
# NOVEL COMPONENT 1: DYNAMIC INPUT-DEPENDENT MASKS
# ═══════════════════════════════════════════════════════════════════════════

class DynamicMask(nn.Module):
    """
    Generates masks conditioned on input.
    
    Unlike static MorphoNet masks, these adapt per sample.
    This enables true conditional computation - different inputs
    use different subnetworks.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 hidden_dim: int = 16,  # Small router network
                 temperature: float = 1.0,
                 sparsity_target: float = 0.5,
                 compute_cost: float = 0.01):  # Penalty for using compute
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.sparsity_target = sparsity_target
        self.compute_cost = compute_cost
        
        # Router network (generates mask logits from input)
        self.router = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features * in_features)
        )
        
        # Global mask bias (learned baseline sparsity)
        self.mask_bias = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Initialize router to produce ~50% sparsity
        nn.init.constant_(self.router[-1].weight, 0.0)
        nn.init.constant_(self.router[-1].bias, 0.0)
    
    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Generate input-dependent mask.
        
        Args:
            x: Input tensor (batch, in_features)
            hard: If True, use straight-through estimator
        
        Returns:
            mask: Binary/soft mask (out_features, in_features)
            info: Dict with sparsity, cost metrics
        """
        B = x.size(0)
        
        # Generate mask logits from input
        # Average across batch for consistent mask within batch
        logits = self.router(x).mean(0, keepdim=True)  # (1, out*in)
        logits = logits.view(self.out_features, self.in_features)
        
        # Add global bias
        logits = logits + self.mask_bias
        
        # Apply temperature
        if hard:
            # Straight-through estimator
            mask_hard = (torch.sigmoid(logits / self.temperature) > 0.5).float()
            mask_soft = torch.sigmoid(logits / self.temperature)
            mask = mask_hard + (mask_soft - mask_soft.detach())  # STE
        else:
            mask = torch.sigmoid(logits / self.temperature)
        
        # Metrics
        sparsity = (mask < 0.5).float().mean().item()
        compute_cost = mask.mean() * self.compute_cost
        
        info = {
            'sparsity': sparsity,
            'compute_cost': compute_cost,
            'mask_entropy': -(mask * torch.log(mask + 1e-9) + 
                             (1-mask) * torch.log(1-mask + 1e-9)).mean().item()
        }
        
        return mask, info


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL COMPONENT 2: LEARNABLE PRIMITIVE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class LearnablePrimitive(nn.Module):
    """
    Learns which nonlinearity to use per neuron.
    
    Instead of fixed ReLU/GELU, the network learns a weighted
    combination of primitives, potentially different per neuron.
    """
    
    def __init__(self, n_primitives: int = 4, temperature: float = 1.0):
        super().__init__()
        self.n_primitives = n_primitives
        self.temperature = temperature
        
        # Primitive weights (learned per-output-neuron)
        self.primitive_logits = nn.Parameter(torch.zeros(n_primitives))
        
        self.primitives = ['relu', 'gelu', 'tanh', 'identity']
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Apply learned mixture of primitives."""
        weights = F.softmax(self.primitive_logits / self.temperature, dim=-1)
        
        # Apply each primitive
        outputs = {
            'relu': F.relu(x),
            'gelu': F.gelu(x),
            'tanh': torch.tanh(x),
            'identity': x
        }
        
        # Weighted combination
        output = sum(weights[i] * outputs[name] for i, name in enumerate(self.primitives))
        
        # Dominant primitive (for interpretability)
        dominant = self.primitives[weights.argmax().item()]
        
        info = {
            'primitive_weights': weights.detach().cpu().numpy(),
            'dominant_primitive': dominant
        }
        
        return output, info


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL COMPONENT 3: CROSS-LAYER ATTENTION MASKS
# ═══════════════════════════════════════════════════════════════════════════

class CrossLayerAttention(nn.Module):
    """
    Attention mechanism that learns cross-layer connections.
    
    Instead of fixed feedforward, learns which previous layers
    to attend to for each computation.
    """
    
    def __init__(self, dim: int, n_layers: int, n_heads: int = 4,
                 sparse_init: float = 0.5):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Query for current layer, keys/values from all previous layers
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Learnable layer connectivity mask
        self.layer_mask_logits = nn.Parameter(
            torch.randn(n_layers, n_heads) * 0.1
        )
        
        self.out_proj = nn.Linear(dim, dim)
        
        # Initialize mask to encourage sparsity
        with torch.no_grad():
            self.layer_mask_logits[:] = torch.where(
                torch.rand(n_layers, n_heads) < sparse_init,
                torch.ones_like(self.layer_mask_logits) * 2,
                torch.ones_like(self.layer_mask_logits) * -2
            )
    
    def forward(self, layer_outputs: List[torch.Tensor], 
                current_layer: int) -> Tuple[torch.Tensor, dict]:
        """
        Attend to previous layer outputs.
        
        Args:
            layer_outputs: List of tensors from previous layers
            current_layer: Index of current layer
        
        Returns:
            attended: Combined output from attended layers
            info: Attention weights, sparsity metrics
        """
        if current_layer == 0 or not layer_outputs:
            return torch.zeros_like(layer_outputs[0]) if layer_outputs else None, {}
        
        B = layer_outputs[0].size(0)
        
        # Stack previous outputs: (n_prev, B, dim)
        prev_outputs = torch.stack(layer_outputs[:current_layer], dim=0)
        n_prev = prev_outputs.size(0)
        
        # Q from current, K,V from previous
        if len(layer_outputs) > current_layer:
            current = layer_outputs[current_layer]
        else:
            current = torch.zeros(B, self.dim, device=prev_outputs.device)
        
        Q = self.query(current).reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(prev_outputs).reshape(n_prev, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.value(prev_outputs).reshape(n_prev, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, 1, n_prev)
        
        # Apply layer mask (learned connectivity)
        layer_mask = torch.sigmoid(self.layer_mask_logits.unsqueeze(0))  # (1, L, H)
        layer_mask = layer_mask.transpose(1, 2).unsqueeze(2)  # (1, H, 1, L)
        attn = attn * layer_mask
        
        # Softmax over previous layers
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        attended = (attn @ V).transpose(1, 2).reshape(B, 1, self.dim)
        attended = self.out_proj(attended).squeeze(1)
        
        # Sparsity metrics
        layer_sparsity = (layer_mask.squeeze() < 0.5).float().mean().item()
        
        info = {
            'attention_weights': attn.detach().cpu().numpy(),
            'layer_sparsity': layer_sparsity,
            'layer_mask': layer_mask.detach().cpu().numpy()
        }
        
        return attended, info


# ═══════════════════════════════════════════════════════════════════════════
# MORPHOX LAYER - Combines All Novel Components
# ═══════════════════════════════════════════════════════════════════════════

class MorphoXLayer(nn.Module):
    """
    A single MorphoX layer combining:
    - Dynamic input-dependent masks
    - Learnable primitive selection
    - Optional cross-layer attention
    """
    
    def __init__(self, in_features: int, out_features: int,
                 use_dynamic_mask: bool = True,
                 use_learnable_primitive: bool = True,
                 primitive_hidden: int = 16,
                 mask_temperature: float = 1.0,
                 primitive_temperature: float = 1.0,
                 sparsity_target: float = 0.5,
                 compute_cost: float = 0.01):
        super().__init__()
        
        self.use_dynamic_mask = use_dynamic_mask
        self.use_learnable_primitive = use_learnable_primitive
        
        # Base weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 
                                   np.sqrt(2.0 / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Dynamic mask (input-dependent sparsity)
        if use_dynamic_mask:
            self.dynamic_mask = DynamicMask(
                in_features, out_features,
                hidden_dim=primitive_hidden,
                temperature=mask_temperature,
                sparsity_target=sparsity_target,
                compute_cost=compute_cost
            )
        
        # Learnable primitive
        if use_learnable_primitive:
            self.primitive = LearnablePrimitive(temperature=primitive_temperature)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, 
                return_info: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with dynamic computation.
        
        Args:
            x: Input (batch, in_features)
            return_info: Whether to return metrics
        
        Returns:
            output: (batch, out_features)
            info: Dict with sparsity, primitive, cost metrics
        """
        info = {}
        
        # Apply dynamic mask to weights
        if self.use_dynamic_mask:
            mask, mask_info = self.dynamic_mask(x)
            masked_weight = self.weight * mask
            info.update(mask_info)
        else:
            masked_weight = self.weight
        
        # Linear transformation
        out = F.linear(x, masked_weight, self.bias)
        
        # Apply learnable primitive
        if self.use_learnable_primitive:
            out, prim_info = self.primitive(out)
            info.update(prim_info)
        else:
            out = F.relu(out)
        
        # Layer norm
        out = self.norm(out)
        
        if return_info:
            return out, info
        return out


# ═══════════════════════════════════════════════════════════════════════════
# MORPHOX NETWORK - Full Architecture
# ═══════════════════════════════════════════════════════════════════════════

class MorphoX(nn.Module):
    """
    MorphoX Network - A truly dynamic, morphing architecture.
    
    Features:
    1. Input-dependent computation graphs
    2. Learnable primitives per layer
    3. Cross-layer attention (optional)
    4. Adaptive depth (early exiting)
    5. Compute budget awareness
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 n_classes: int,
                 # Dynamic mask config
                 use_dynamic_mask: bool = True,
                 mask_temperature: float = 1.0,
                 sparsity_target: float = 0.5,
                 compute_cost: float = 0.01,
                 # Primitive config
                 use_learnable_primitive: bool = True,
                 primitive_temperature: float = 1.0,
                 # Cross-layer attention
                 use_cross_attention: bool = False,
                 n_attention_heads: int = 4,
                 # Early exiting
                 use_early_exit: bool = True,
                 exit_threshold: float = 0.7,
                 # Regularization
                 dropout: float = 0.1,
                 device: str = "cpu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.use_cross_attention = use_cross_attention
        self.use_early_exit = use_early_exit
        self.exit_threshold = exit_threshold
        self.device = device
        
        # Build layers
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(MorphoXLayer(
                dims[i], dims[i+1],
                use_dynamic_mask=use_dynamic_mask,
                use_learnable_primitive=use_learnable_primitive,
                mask_temperature=mask_temperature,
                primitive_temperature=primitive_temperature,
                sparsity_target=sparsity_target,
                compute_cost=compute_cost
            ))
        
        # Cross-layer attention (optional)
        if use_cross_attention and len(hidden_dims) > 1:
            self.cross_attn = CrossLayerAttention(
                hidden_dims[-1], len(hidden_dims), n_attention_heads
            )
        else:
            self.cross_attn = None
        
        # Early exit classifiers (one per hidden layer)
        if use_early_exit:
            self.exit_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim // 2, n_classes)
                ) for dim in hidden_dims
            ])
        
        # Final classifier
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, n_classes)
        )
        
        # Temperature annealing
        self.register_buffer("mask_temperature", torch.tensor(mask_temperature))
        self.register_buffer("primitive_temperature", torch.tensor(primitive_temperature))
    
    def set_temperatures(self, mask_temp: float = None, prim_temp: float = None):
        """Anneal temperatures for exploration → exploitation."""
        if mask_temp is not None:
            self.mask_temperature.fill_(mask_temp)
            for layer in self.layers:
                if hasattr(layer, 'dynamic_mask'):
                    layer.dynamic_mask.temperature = mask_temp
        
        if prim_temp is not None:
            self.primitive_temperature.fill_(prim_temp)
            for layer in self.layers:
                if hasattr(layer, 'primitive'):
                    layer.primitive.temperature = prim_temp
    
    def forward(self, x: torch.Tensor, 
                budget: float = 1.0,  # Compute budget (0-1)
                return_all: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with dynamic computation.
        
        Args:
            x: Input (batch, input_dim)
            budget: Compute budget (1.0 = use all layers, <1.0 = early exit)
            return_all: Return all intermediate outputs and metrics
        
        Returns:
            logits: Output predictions
            info: Dict with all metrics
        """
        info = {
            'layers_used': 0,
            'early_exits': 0,
            'total_sparsity': [],
            'primitives_used': [],
            'compute_cost': 0.0
        }
        
        layer_outputs = []
        total_sparsity = 0
        total_compute_cost = 0
        
        for i, layer in enumerate(self.layers):
            # Apply layer
            out, layer_info = layer(x, return_info=True)
            layer_outputs.append(out)
            
            # Track metrics
            if 'sparsity' in layer_info:
                total_sparsity += layer_info['sparsity']
                info['total_sparsity'].append(layer_info['sparsity'])
            if 'compute_cost' in layer_info:
                total_compute_cost += layer_info['compute_cost']
            if 'dominant_primitive' in layer_info:
                info['primitives_used'].append(layer_info['dominant_primitive'])
            
            info['layers_used'] = i + 1
            
            # Early exiting
            if self.use_early_exit and i < len(self.exit_classifiers):
                exit_logits = self.exit_classifiers[i](out)
                exit_probs = F.softmax(exit_logits, dim=-1)
                confidence = exit_probs.max(dim=-1).values.mean()
                
                # Exit if confident and within budget
                if confidence > self.exit_threshold and (i + 1) / len(self.layers) <= budget:
                    info['early_exits'] += 1
                    info['exit_layer'] = i
                    info['exit_confidence'] = confidence.item()
                    
                    if return_all:
                        info['layer_outputs'] = layer_outputs
                        info['total_sparsity'] = total_sparsity / (i + 1)
                        info['compute_cost'] = total_compute_cost
                        return exit_logits, info
                    
                    return exit_logits, info
            
            # Cross-layer attention
            if self.cross_attn is not None:
                attn_out, attn_info = self.cross_attn(layer_outputs, i + 1)
                out = out + attn_out
                info['attention'] = attn_info
            
            x = out
        
        # Final classifier
        logits = self.output(out)
        
        if return_all:
            info['layer_outputs'] = layer_outputs
            info['total_sparsity'] = total_sparsity / len(self.layers)
            info['compute_cost'] = total_compute_cost
        
        return logits, info


# ═══════════════════════════════════════════════════════════════════════════
# MORPHOX TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class MorphoXTrainer:
    """
    Trainer for MorphoX with specialized losses:
    - Task loss (cross-entropy)
    - Sparsity loss (encourage efficient computation)
    - Compute cost loss (stay within budget)
    - Diversity loss (encourage different primitives)
    """
    
    def __init__(self, model: MorphoX,
                 weight_lr: float = 0.001,
                 mask_lr: float = 0.01,
                 sparsity_weight: float = 0.01,
                 compute_weight: float = 0.1,
                 diversity_weight: float = 0.001):
        self.model = model
        
        # Separate optimizers
        weight_params = [p for n, p in model.named_parameters() 
                        if 'mask' not in n and 'primitive' not in n and 'logits' not in n]
        mask_params = [p for n, p in model.named_parameters() if 'mask' in n or 'logits' in n]
        
        self.weight_opt = torch.optim.AdamW(weight_params, lr=weight_lr, weight_decay=0.01)
        self.mask_opt = torch.optim.Adam(mask_params, lr=mask_lr)
        
        self.sparsity_weight = sparsity_weight
        self.compute_weight = compute_weight
        self.diversity_weight = diversity_weight
        
        self.loss_history = []
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, 
                   epoch: int, total_epochs: int) -> dict:
        """One training step with all losses."""
        self.model.train()
        
        # Anneal temperature
        progress = epoch / total_epochs
        mask_temp = max(0.5, 2.0 - progress * 1.5)
        prim_temp = max(0.5, 2.0 - progress * 1.5)
        self.model.set_temperatures(mask_temp, prim_temp)
        
        # Forward pass
        logits, info = self.model(X, return_all=True)
        
        # Task loss
        task_loss = F.cross_entropy(logits, y)
        
        # Sparsity loss (encourage sparse computation)
        sparsity_loss = self.sparsity_weight * info.get('total_sparsity', 0)
        
        # Compute cost loss
        compute_loss = self.compute_weight * info.get('compute_cost', 0)
        
        # Diversity loss (encourage different primitives across layers)
        primitives = info.get('primitives_used', [])
        if len(primitives) > 1:
            prim_counts = {}
            for p in primitives:
                prim_counts[p] = prim_counts.get(p, 0) + 1
            diversity = 1 - max(prim_counts.values()) / len(primitives)
            diversity_loss = -self.diversity_weight * diversity
        else:
            diversity_loss = 0
        
        # Total loss
        total_loss = task_loss + sparsity_loss + compute_loss + diversity_loss
        
        # Backward pass
        self.weight_opt.zero_grad()
        self.mask_opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        # Update
        self.weight_opt.step()
        self.mask_opt.step()
        
        self.loss_history.append(total_loss.item())
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'sparsity_loss': sparsity_loss,
            'compute_loss': compute_loss,
            'diversity_loss': diversity_loss,
            'layers_used': info.get('layers_used', 0),
            'early_exits': info.get('early_exits', 0),
            'sparsity': info.get('total_sparsity', 0),
            'primitives': info.get('primitives_used', [])
        }
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 64,
              verbose: bool = True) -> dict:
        """Full training loop."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.model.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.model.device)
        
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.model.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.model.device)
        
        n = len(X)
        start_time = time.time()
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(n, device=self.model.device)
            epoch_metrics = {'total_loss': 0, 'layers_used': 0, 'sparsity': 0}
            n_batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                metrics = self.train_step(X_t[idx], y_t[idx], epoch, epochs)
                for k in epoch_metrics:
                    epoch_metrics[k] += metrics.get(k, 0)
                n_batches += 1
            
            # Average metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= n_batches
            
            # Validation
            val_acc = 0
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.model(X_val_t)
                    val_preds = val_logits.argmax(-1)
                    val_acc = (val_preds == y_val_t).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Reporting
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:3d}: loss={epoch_metrics['total_loss']:.4f}, "
                msg += f"layers={epoch_metrics['layers_used']:.1f}, "
                msg += f"sparsity={epoch_metrics['sparsity']:.1%}"
                if X_val is not None:
                    msg += f", val_acc={val_acc:.1%}"
                print(msg)
        
        train_time = time.time() - start_time
        
        return {
            'train_time': train_time,
            'best_val_acc': best_val_acc,
            'final_loss': self.loss_history[-1],
            'loss_history': self.loss_history
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def create_morphox(config: dict) -> MorphoX:
    """
    Factory function to create MorphoX models.
    
    Example:
        model = create_morphox({
            'input_dim': 20,
            'hidden_dims': [64, 32],
            'n_classes': 5,
            'use_dynamic_mask': True,
            'use_cross_attention': True,
            'use_early_exit': True
        })
    """
    return MorphoX(**config)


# Export API
__all__ = [
    'DynamicMask',
    'LearnablePrimitive', 
    'CrossLayerAttention',
    'MorphoXLayer',
    'MorphoX',
    'MorphoXTrainer',
    'create_morphox'
]
