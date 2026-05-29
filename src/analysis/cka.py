"""
Centered Kernel Alignment (CKA) Analysis

Provides both low-level CKA computation on feature matrices and
high-level model-to-model CKA measurement.
"""
import torch
import numpy as np
from typing import Optional, List
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

def gram_linear(x):
    """Compute Gram matrix for linear kernel"""
    return x @ x.t()

def gram_rbf(x, threshold=1.0):
    """Compute Gram matrix for RBF kernel"""
    dot_products = x @ x.t()
    sq_norms = (x**2).sum(dim=1).view(-1, 1)
    sq_dists = sq_norms + sq_norms.t() - 2 * dot_products
    return torch.exp(-sq_dists / (2 * threshold**2))

def center_gram(k, n):
    """Center the Gram matrix"""
    h = torch.eye(n, device=k.device) - (1.0 / n) * torch.ones(n, n, device=k.device)
    return h @ k @ h

def cka(gram_x, gram_y, debiased=False):
    """Compute CKA given two Gram matrices"""
    # debiased CKA is strictly better but standard CKA is more common
    # For simplicity we implement standard CKA here
    
    # Center matrices
    n = gram_x.shape[0]
    gx_c = center_gram(gram_x, n)
    gy_c = center_gram(gram_y, n)

    # Compute CKA
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


def compute_model_cka(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 128,
    n_samples: int = 30,
) -> float:
    """
    Compute CKA between two models on a set of texts.

    - Converts activations to float32 (avoids fp16 NaN)
    - Centers representations before Gram matrix computation
    - Handles dimension mismatches via truncation

    Args:
        model_a, model_b: Models to compare
        tokenizer: Shared tokenizer
        texts: Input texts for representation collection
        device: Device string
        max_length: Max sequence length
        n_samples: Max number of texts to use

    Returns:
        CKA similarity score in [0, 1].
    """
    model_a.eval()
    model_b.eval()

    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size
    common_dim = min(dim_a, dim_b)

    all_a, all_b = [], []

    with torch.no_grad():
        for text in tqdm(texts[:n_samples], desc="CKA", leave=False):
            if len(text) < 20:
                continue

            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)

            out_a = model_a(inputs.input_ids, output_hidden_states=True)
            out_b = model_b(inputs.input_ids, output_hidden_states=True)

            # Mean pool over sequence, convert to float32
            h_a = out_a.hidden_states[-1].float().mean(dim=1)  # [1, dim_a]
            h_b = out_b.hidden_states[-1].float().mean(dim=1)  # [1, dim_b]

            # Truncate to common dimension
            h_a = h_a[:, :common_dim]
            h_b = h_b[:, :common_dim]

            all_a.append(h_a.cpu())
            all_b.append(h_b.cpu())

    if len(all_a) < 3:
        return 0.0

    X = torch.cat(all_a)  # [N, common_dim]
    Y = torch.cat(all_b)  # [N, common_dim]

    analyzer = CKAAnalyzer(device='cpu')
    return analyzer.linear_cka(X, Y)
