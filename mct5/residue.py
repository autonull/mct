import torch

class HolographicResidue:
    """
    Complex-valued phase-rotation memory (Holographic Resonance).
    R ∈ ℂᴰ, encoding structural sequence history.
    """
    def __init__(self, max_nodes: int, D: int, phi_max: float = 4.0, device: str = "cpu"):
        self.D = D
        self.phi_max = phi_max
        self.device = device
        
        self.R = torch.zeros(D, dtype=torch.complex64, device=device)
        self.B = torch.zeros((max_nodes, D), dtype=torch.float32, device=device)
        self.active_indices = []
        self.next_idx = 0

    def register_node(self, node_id: int):
        """Assign orthogonal basis vector strictly using Gram-Schmidt."""
        if self.next_idx >= self.B.size(0):
            # Grow buffer if needed
            new_B = torch.zeros((self.B.size(0) * 2, self.D), dtype=torch.float32, device=self.device)
            new_B[:self.B.size(0)] = self.B
            self.B = new_B

        idx = self.next_idx
        v = torch.randn(self.D, device=self.device)
        
        # Gram-Schmidt against all active vectors (slow but stable enough for setup/mutations)
        if self.active_indices:
            active_B = self.B[self.active_indices]
            projs = (active_B @ v).unsqueeze(1) * active_B # (num_active, D)
            v = v - projs.sum(dim=0)
            
        norm = torch.norm(v)
        if norm < 1e-5:
            v = torch.randn(self.D, device=self.device)
            v = v / torch.norm(v)
        else:
            v = v / norm
            
        self.B[idx] = v
        self.active_indices.append(idx)
        self.next_idx += 1
        return idx

    def release_node(self, internal_idx: int):
        if internal_idx in self.active_indices:
            self.active_indices.remove(internal_idx)

    def reset_sequence(self):
        self.R.zero_()

    def inject_ghost(self, internal_idx: int, rho: float, omega: float = 1.0, t: float = 0.0):
        """R += (ρ / √D) · B_idx · e^(iωt)"""
        if internal_idx < 0 or internal_idx >= self.B.size(0):
            return
            
        basis = self.B[internal_idx].to(torch.complex64)
        phase = torch.complex(torch.cos(torch.tensor(omega * t)), torch.sin(torch.tensor(omega * t))).to(self.device)
        contrib = (rho / torch.sqrt(torch.tensor(self.D, dtype=torch.float32))) * basis * phase
        
        self.R = self.R + contrib
        self._prune_norm()

    def decode(self, internal_idx: int) -> torch.Tensor:
        """Re(⟨B_idx, R⟩)"""
        if internal_idx < 0 or internal_idx >= self.B.size(0):
            return torch.tensor(0.0, device=self.device)
            
        basis = self.B[internal_idx].to(torch.complex64)
        return torch.real(torch.dot(basis, self.R))

    def _prune_norm(self):
        """Keeps trace norm below phi_max."""
        norm = torch.norm(self.R)
        if norm > self.phi_max:
            # Soft thresholding down to phi_max
            scale = self.phi_max / norm
            self.R = self.R * scale
