"""
Centered Kernel Alignment (CKA) Analysis
"""
import torch
import numpy as np
from typing import Optional

def gram_linear(x):
    """Compute Gram matrix for linear kernel"""
    return x @ x.t()

def gram_rbf(x, threshold=1.0):
    """Compute Gram matrix for RBF kernel"""
    dot_products = x @ x.t()
    sq_norms = (x**2).sum(dim=1).view(-1, 1)
    sq_dists = sq_norms + sq_norms.t() - 2 * dot_products
    # Use median distance heuristic for bandwidth
    # sigma = torch.median(sq_dists)
    # return torch.exp(-sq_dists / (2 * sigma))
    return torch.exp(-sq_dists / (2 * threshold**2))

def center_gram(k, n):
    """Center the Gram matrix"""
    h = torch.eye(n, device=k.device) - (1.0 / n) * torch.ones(n, n, device=k.device)
    return h @ k @ h

def cka(gram_x, gram_y, debiased=False):
    """Compute CKA given two Gram matrices"""
    # Note: debiased CKA is strictly better but standard CKA is more common
    # For simplicity we implement standard CKA here
    
    # Center matrices
    n = gram_x.shape[0]
    gx_c = center_gram(gram_x, n)
    gy_c = center_gram(gram_y, n)

    # Compute CKA
    # trace(GX * GY) / sqrt(trace(GX * GX) * trace(GY * GY))
    scaled_hsic = torch.sum(gx_c * gy_c)
    norm_x = torch.sqrt(torch.sum(gx_c * gx_c))
    norm_y = torch.sqrt(torch.sum(gy_c * gy_c))
    
    return scaled_hsic / (norm_x * norm_y)

class CKAAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device

    def linear_cka(self, features_x: torch.Tensor, features_y: torch.Tensor) -> float:
        """
        Compute Linear CKA between two feature matrices.
        Shape: [num_samples, num_features]
        """
        x = features_x.to(self.device).float()
        y = features_y.to(self.device).float()
        
        # Center features first (optional but often done)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        
        gram_x = gram_linear(x)
        gram_y = gram_linear(y)
        
        return cka(gram_x, gram_y).item()

    def kernel_cka(self, features_x: torch.Tensor, features_y: torch.Tensor) -> float:
        """
        Compute RBF Kernel CKA.
        """
        x = features_x.to(self.device).float()
        y = features_y.to(self.device).float()
        
        gram_x = gram_rbf(x)
        gram_y = gram_rbf(y)
        
        return cka(gram_x, gram_y).item()
