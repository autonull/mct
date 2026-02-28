"""
MCT5 Unified Engine

Top-level API integrating all components into a cohesive learning system.

Usage:
    from mct5_unified import MCT5, MCT5Config, LearningMode
    
    config = MCT5Config(learning_mode=LearningMode.HYBRID)
    model = MCT5(config)
    model.initialize()
    
    for X, y in dataloader:
        loss = model.train_batch(X, y)
    
    predictions = model.predict(X_test)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time

from .config import MCT5Config, LearningMode
from .types import GraphState, Node, NodeType, ActiveRecord
from .primitives import Primitive
from .residue import HolographicResidue
from .forward import ForwardExecutor
from .learning import LearningEngine, AutogradLearning
from .structural import StructuralEvolution


class MCT5(nn.Module):
    """
    MCT5 Unified - Morphogenic Compute Topology v5.
    
    A self-structuring, continuously-learning compute graph with:
    - Hybrid learning (autograd + dual-signal)
    - Intelligent structural evolution
    - Holographic context memory
    - Anytime inference
    
    Example:
        >>> config = MCT5Config(D=96, r=24, learning_mode=LearningMode.HYBRID)
        >>> model = MCT5(config)
        >>> model.initialize()
        >>> 
        >>> # Training
        >>> for epoch in range(100):
        >>>     for X_batch, y_batch in dataloader:
        >>>         loss = model.train_batch(X_batch, y_batch)
        >>> 
        >>> # Inference
        >>> predictions = model.predict(X_test)
        >>> accuracy = model.score(X_test, y_test)
    """
    
    def __init__(self, config: Optional[MCT5Config] = None):
        super().__init__()
        
        self.cfg = config or MCT5Config()
        cfg = self.cfg
        
        # Set random seeds if specified
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        
        self.device = cfg.device_torch
        
        # ═══════════════════════════════════════════════════════════════════
        # CORE COMPONENTS
        # ═══════════════════════════════════════════════════════════════════
        
        # Graph state
        self.state = GraphState(D=cfg.D, r=cfg.r, device=cfg.device)
        
        # Holographic residue
        self.residue = HolographicResidue(
            D=cfg.D,
            max_nodes=cfg.max_nodes,
            phi_max=cfg.phi_max,
            omega=cfg.omega,
            decay_rate=cfg.residue_decay,
            device=cfg.device
        )
        
        # Forward executor
        self.forward_executor = ForwardExecutor(
            self.state,
            self.residue,
            lambda_async=cfg.lambda_async,
            residue_boost_scale=cfg.residue_boost_scale
        )
        
        # Learning engine
        self.learner = LearningEngine(
            self.state,
            self.residue,
            learning_mode=cfg.learning_mode,
            eta_W=cfg.eta_W,
            eta_S=cfg.eta_S,
            grad_clip=cfg.grad_clip,
            weight_decay=cfg.weight_decay,
            lambda_contrastive=cfg.lambda_contrastive,
            lambda_retrograde=cfg.lambda_retrograde,
            goodness_threshold=cfg.goodness_threshold,
            goodness_scale_input=cfg.goodness_scale_input,
            spectral_clamp_min=cfg.spectral_clamp_min,
            spectral_clamp_max=cfg.spectral_clamp_max,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_eps,
            W_max=cfg.W_max,
            ema_loss_alpha=cfg.ema_loss_alpha,
        )
        
        # Structural evolution
        self.evolver = StructuralEvolution(
            self.state,
            self.residue,
            sigma_mut=cfg.sigma_mut,
            sigma_mut_min=cfg.sigma_mut_min,
            sigma_mut_max=cfg.sigma_mut_max,
            K=cfg.K,
            tau_lateral=cfg.tau_lateral,
            quadratic_spawn_bias=cfg.quadratic_spawn_bias,
            adaptive_mutation=cfg.adaptive_mutation,
            adaptation_sensitivity=cfg.adaptation_sensitivity,
            prune_threshold=cfg.prune_threshold,
            min_nodes=cfg.min_nodes,
            lateral_wiring=cfg.lateral_wiring,
            ensure_nonlinearity=cfg.ensure_nonlinearity,
            device=cfg.device
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # CLASSIFICATION HEAD
        # ═══════════════════════════════════════════════════════════════════
        
        # Output projection: D → n_classes
        self.classifier = nn.Linear(cfg.D, cfg.n_classes, device=cfg.device)
        
        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZERS
        # ═══════════════════════════════════════════════════════════════════
        
        self.opt_W = None  # For weights (A, B, bias, classifier)
        self.opt_S = None  # For routing signatures (S)
        
        # ═══════════════════════════════════════════════════════════════════
        # TRAINING STATE
        # ═══════════════════════════════════════════════════════════════════
        
        self.step_count = 0
        self.register_buffer("ema_loss", torch.tensor(1.0))
        self.prev_ema_loss = 1.0
        self._steps_since_evolve = 0
        
        # Metrics history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def initialize(self, primitive_hidden: str = None):
        """
        Build the minimal starter graph.
        
        Args:
            primitive_hidden: Default hidden primitive (default: from config)
        """
        prim = primitive_hidden or self.cfg.init_hidden_primitive
        self.evolver.initialize_graph(prim)
        self._build_optimizers()
        
        if self.cfg.verbose:
            print(f"MCT5 initialized with {len(self.state.nodes)} nodes")
    
    def _build_optimizers(self):
        """Create or rebuild optimizers after structural changes."""
        # Collect parameters
        w_params = [self.classifier.weight, self.classifier.bias]
        s_params = []
        
        for node in self.state.hidden_nodes() + self.state.input_nodes() + self.state.output_nodes():
            w_params.extend([node.A, node.B, node.bias])
            s_params.append(node.S)
        
        # AdamW with weight decay
        self.opt_W = torch.optim.AdamW(
            w_params,
            lr=self.cfg.eta_W,
            weight_decay=self.cfg.weight_decay,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_eps
        )
        
        self.opt_S = torch.optim.AdamW(
            s_params,
            lr=self.cfg.eta_S,
            weight_decay=self.cfg.weight_decay,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_eps
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # FORWARD PASS
    # ═══════════════════════════════════════════════════════════════════════
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MCT5 graph.
        
        Args:
            X: Input tensor (B, input_dim) or (input_dim,)
        
        Returns:
            Logits tensor (B, n_classes) or (n_classes,)
        """
        # Ensure 2D
        single_sample = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            single_sample = True
        
        # Pad/truncate to D dimensions
        X = self._embed_input(X)
        
        # Execute graph
        graph_out = self.forward_executor.forward(
            X,
            t_budget=self.cfg.t_budget,
            lambda_tau=self.cfg.lambda_tau
        )
        
        # Classify
        logits = self.classifier(graph_out)
        
        if single_sample:
            logits = logits.squeeze(0)
        
        return logits
    
    def _embed_input(self, X: torch.Tensor) -> torch.Tensor:
        """Pad or truncate input to D dimensions."""
        B, current_dim = X.shape
        D = self.cfg.D
        
        if current_dim >= D:
            return X[:, :D]
        
        padding = torch.zeros(B, D - current_dim, device=X.device)
        return torch.cat([X, padding], dim=1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_step(self, X: np.ndarray, y: int) -> float:
        """
        Train on a single sample (online learning).
        
        Args:
            X: Input features (input_dim,) or will be converted
            y: Integer class label
        
        Returns:
            Loss value
        """
        self.train()
        
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if X.dim() == 0:
            X = X.unsqueeze(0)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        y_tensor = torch.tensor([y], dtype=torch.long, device=self.device)
        
        # Zero gradients
        self.opt_W.zero_grad()
        self.opt_S.zero_grad()
        
        # Forward pass
        logits = self.forward(X)
        
        # Compute loss
        loss = F.cross_entropy(logits, y_tensor)
        
        # Backward pass (for autograd/hybrid modes)
        if self.cfg.learning_mode in [LearningMode.AUTOGRAD, LearningMode.HYBRID]:
            loss.backward()
            
            # Gradient clipping
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.cfg.grad_clip
                )
            
            # Step optimizers
            self.opt_W.step()
            self.opt_S.step()
        
        # Dual-signal learning (for dual_signal/hybrid modes)
        if self.cfg.learning_mode in [LearningMode.DUAL_SIGNAL, LearningMode.HYBRID]:
            # Get graph output before classifier
            with torch.no_grad():
                X_embed = self._embed_input(X)
                graph_out = self.forward_executor.forward(X_embed)
            
            # Compute target
            Y_star = torch.zeros_like(graph_out)
            Y_star[0, y % self.cfg.n_classes] = 1.0
            
            # Learn
            output_nodes = self.state.output_nodes()
            if output_nodes:
                self.learner.learn(
                    graph_out, Y_star,
                    output_nodes[0].id,
                    is_positive=True,
                    labels=y_tensor
                )
        
        # Update tracking
        loss_val = loss.item()
        self.ema_loss = self.cfg.ema_loss_alpha * self.ema_loss + (1 - self.cfg.ema_loss_alpha) * loss_val
        self.step_count += 1
        self._steps_since_evolve += 1
        self.loss_history.append(loss_val)
        
        # Structural evolution
        if self._steps_since_evolve >= self.cfg.evolve_interval:
            self._run_evolution()
            self._steps_since_evolve = 0
        
        self.state.tick()
        
        return loss_val
    
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Train on a batch.
        
        Args:
            X_batch: Input features (batch_size, input_dim)
            y_batch: Integer class labels (batch_size,)
        
        Returns:
            Mean loss over batch
        """
        self.train()
        
        # Convert to tensors
        X = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_batch, dtype=torch.long, device=self.device)
        
        # Zero gradients
        self.opt_W.zero_grad()
        self.opt_S.zero_grad()
        
        # Forward pass
        logits = self.forward(X)
        
        # Compute loss
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        if self.cfg.learning_mode in [LearningMode.AUTOGRAD, LearningMode.HYBRID]:
            loss.backward()
            
            # Gradient clipping
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.cfg.grad_clip
                )
            
            # Step optimizers
            self.opt_W.step()
            self.opt_S.step()
        
        # Dual-signal learning for each sample in batch
        if self.cfg.learning_mode in [LearningMode.DUAL_SIGNAL, LearningMode.HYBRID]:
            with torch.no_grad():
                X_embed = self._embed_input(X)
                graph_out = self.forward_executor.forward(X_embed)
            
            output_nodes = self.state.output_nodes()
            if output_nodes:
                # Create one-hot targets
                Y_star = torch.zeros_like(graph_out)
                for i, label in enumerate(y_batch):
                    Y_star[i, label % self.cfg.n_classes] = 1.0
                
                self.learner.learn(
                    graph_out, Y_star,
                    output_nodes[0].id,
                    is_positive=True,
                    labels=y
                )
        
        # Update tracking
        loss_val = loss.item()
        self.ema_loss = self.cfg.ema_loss_alpha * self.ema_loss + (1 - self.cfg.ema_loss_alpha) * loss_val
        self.step_count += 1
        self._steps_since_evolve += 1
        self.loss_history.append(loss_val)
        
        # Structural evolution
        if self._steps_since_evolve >= self.cfg.evolve_interval:
            self._run_evolution()
            self._steps_since_evolve = 0
        
        self.state.tick()
        
        return loss_val
    
    def _run_evolution(self):
        """Execute structural evolution step."""
        # Prune unhealthy nodes
        pruned = self.evolver.prune()
        
        # Get max tension edge
        max_edge = self.learner.get_max_tension_edge()
        
        # Evolve
        stats = self.evolver.evolve(
            pruned=pruned,
            max_tension_edge=max_edge,
            ema_loss=float(self.ema_loss.item()),
            ema_loss_prev=self.prev_ema_loss
        )
        
        # Rebuild optimizers if structure changed
        if pruned or stats["spawned"] > 0:
            self._build_optimizers()
        
        self.prev_ema_loss = float(self.ema_loss.item())
        
        if self.cfg.verbose and (pruned or stats["spawned"] > 0):
            print(f"  Evolution: pruned={len(pruned)}, spawned={stats['spawned']}, "
                  f"lateral={stats['lateral_edges']}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Predicted class labels
        """
        self.eval()
        
        single_sample = False
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
            single_sample = True
        
        with torch.no_grad():
            logits = self.forward(X)
            preds = torch.argmax(logits, dim=-1)
        
        if single_sample:
            return preds[0].cpu().numpy()
        return preds.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Class probabilities (batch_size, n_classes)
        """
        self.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(X)
            probs = F.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy.
        
        Args:
            X: Input features
            y: True labels
        
        Returns:
            Accuracy score
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def reset_sequence(self):
        """Reset holographic residue at sequence boundary."""
        self.forward_executor.reset_context()
    
    def reset_state(self):
        """Reset all training state (for fresh start with same architecture)."""
        self.step_count = 0
        self.ema_loss = torch.tensor(1.0)
        self.prev_ema_loss = 1.0
        self._steps_since_evolve = 0
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.residue.reset()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STATISTICS & INSPECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Graph stats
        graph_stats = self.state.get_stats()
        
        # Learning stats
        learning_stats = self.learner.get_stats()
        
        # Evolution stats
        evolve_stats = self.evolver.get_stats()
        
        # Residue stats
        residue_stats = self.residue.get_stats()
        
        return {
            "total_params": total_params,
            "ema_loss": float(self.ema_loss.item()),
            "step_count": self.step_count,
            "graph": graph_stats,
            "learning": learning_stats,
            "evolution": evolve_stats,
            "residue": residue_stats,
            "config": {
                "D": self.cfg.D,
                "r": self.cfg.r,
                "learning_mode": self.cfg.learning_mode.name,
                "t_budget": self.cfg.t_budget,
            }
        }
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("  MCT5 Model Statistics")
        print("=" * 60)
        print(f"  Parameters:     {stats['total_params']:,}")
        print(f"  EMA Loss:       {stats['ema_loss']:.4f}")
        print(f"  Steps:          {stats['step_count']}")
        print(f"  Learning Mode:  {stats['config']['learning_mode']}")
        print()
        print("  Graph:")
        print(f"    Nodes:        {stats['graph']['total_nodes']} "
              f"(input={stats['graph']['input_nodes']}, "
              f"hidden={stats['graph']['hidden_nodes']}, "
              f"output={stats['graph']['output_nodes']})")
        print(f"    Edges:        {stats['graph']['total_edges']}")
        print(f"    Avg Health:   {stats['graph']['avg_rho']:.2f}")
        print(f"    Primitives:   {stats['graph']['primitives']}")
        print()
        print("  Evolution:")
        print(f"    Total Pruned: {stats['evolution']['total_pruned']}")
        print(f"    Total Spawned:{stats['evolution']['total_spawned']}")
        print(f"    σ_mut:        {stats['evolution']['current_sigma_mut']:.4f}")
        print()
        print("  Residue:")
        print(f"    Norm:         {stats['residue']['norm']:.3f}")
        print(f"    Active Bases: {stats['residue']['active_bases']}")
        print("=" * 60 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: File path (.pt or .pth)
        """
        state_dict = {
            "model_state": self.state_dict(),
            "config": self.cfg.to_dict(),
            "graph_state": {
                "nodes": len(self.state.nodes),
                "edges_out": self.state.edges_out,
                "edges_in": self.state.edges_in,
                "next_id": self.state.next_id,
            },
            "training": {
                "step_count": self.step_count,
                "ema_loss": float(self.ema_loss.item()),
                "loss_history": self.loss_history,
            },
            "residue_stats": self.residue.get_stats(),
        }
        torch.save(state_dict, path)
    
    def load(self, path: str):
        """
        Load model from file.
        
        Args:
            path: File path (.pt or .pth)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        self.cfg = MCT5Config.from_dict(checkpoint["config"])
        
        # Load state dict
        self.load_state_dict(checkpoint["model_state"])
        
        # Restore graph structure
        self.state.edges_out = checkpoint["graph_state"]["edges_out"]
        self.state.edges_in = checkpoint["graph_state"]["edges_in"]
        self.state.next_id = checkpoint["graph_state"]["next_id"]
        
        # Restore training state
        self.step_count = checkpoint["training"]["step_count"]
        self.ema_loss = torch.tensor(checkpoint["training"]["ema_loss"])
        self.loss_history = checkpoint["training"]["loss_history"]
        
        # Rebuild optimizers
        self._build_optimizers()
    
    # ═══════════════════════════════════════════════════════════════════════
    # REPRESENTATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def extra_repr(self) -> str:
        return f"D={self.cfg.D}, r={self.cfg.r}, mode={self.cfg.learning_mode.name}"
