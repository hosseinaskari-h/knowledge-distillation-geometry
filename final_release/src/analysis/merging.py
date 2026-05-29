"""
Model Merging & Compatibility Module

Consolidates weight merging and representation stitching functions.
Used for validating that geometric synchronization (cos θ) predicts
functional compatibility.
"""
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .perplexity import compute_perplexity


def weight_merge(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    alpha: float = 0.5,
) -> PreTrainedModel:
    """
    Merge two models by averaging their weights.

    For same-architecture models (e.g., GPT-2-Medium + DialoGPT-Medium),
    this performs standard weight interpolation:
        merged_param = alpha * param_a + (1 - alpha) * param_b

    For different-sized models within the same family, parameters are
    matched by name and merged where shapes match. Missing parameters
    are kept from model_a (the template).

    Args:
        model_a: Primary model (used as template for merged model)
        model_b: Secondary model
        alpha: Interpolation weight. 1.0 = all model_a, 0.0 = all model_b.

    Returns:
        Merged model (deep copy of model_a with merged weights).
    """
    merged = deepcopy(model_a)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state = merged.state_dict()

    merged_count = 0
    skipped_count = 0

    for key in merged_state.keys():
        if key in state_b and state_a[key].shape == state_b[key].shape:
            # Same shape: direct interpolation
            merged_state[key] = alpha * state_a[key] + (1 - alpha) * state_b[key]
            merged_count += 1
        elif key in state_b:
            # Shape mismatch: try zero-padding smaller into larger
            param_a = state_a[key]
            param_b = state_b[key]

            if len(param_a.shape) == len(param_b.shape) and all(
                sb <= sa for sb, sa in zip(param_b.shape, param_a.shape)
            ):
                # B fits inside A: zero-pad B to A's shape
                padded_b = torch.zeros_like(param_a)
                slices = tuple(slice(0, s) for s in param_b.shape)
                padded_b[slices] = param_b
                merged_state[key] = alpha * param_a + (1 - alpha) * padded_b
                merged_count += 1
            else:
                # Can't align: keep model_a's parameter
                merged_state[key] = param_a
                skipped_count += 1
        else:
            # Not in model_b: keep model_a's parameter
            skipped_count += 1

    merged.load_state_dict(merged_state)
    print(f"  Merged {merged_count} params, skipped {skipped_count}")
    return merged


def evaluate_weight_merge(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_texts: List[str],
    alphas: List[float] = [0.3, 0.5, 0.7],
    device: str = 'cuda',
) -> Dict:
    """
    Full weight merge evaluation pipeline.

    Merges two models at multiple alpha values and reports perplexity
    degradation for each.

    Args:
        model_a: Primary model
        model_b: Secondary model
        tokenizer: Tokenizer (shared between both models)
        test_texts: Evaluation texts
        alphas: List of interpolation weights to test
        device: Device string

    Returns:
        Dict with per-alpha results including PPL and degradation %.
    """
    print(f"  Evaluating Model A baseline...")
    ppl_a = compute_perplexity(model_a, tokenizer, test_texts, device)
    print(f"  PPL(A) = {ppl_a:.2f}")

    results = {
        'ppl_a': ppl_a,
        'merge_results': {},
    }

    for alpha_val in alphas:
        print(f"  Merging at alpha={alpha_val}...")
        merged = weight_merge(model_a, model_b, alpha=alpha_val)
        merged = merged.to(device)

        ppl_merged = compute_perplexity(merged, tokenizer, test_texts, device)
        degradation = (ppl_merged - ppl_a) / ppl_a * 100

        results['merge_results'][str(alpha_val)] = {
            'ppl_merged': ppl_merged,
            'degradation_pct': degradation,
        }
        print(f"  PPL(merged@{alpha_val}) = {ppl_merged:.2f} ({degradation:+.1f}%)")

        del merged
        torch.cuda.empty_cache()

    return results


def compute_cos_theta(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 128,
) -> Tuple[float, float]:
    """
    Measure average cosine similarity between hidden states of two models.

    This is the paper's primary metric for geometric synchronization.

    Args:
        model_a, model_b: Models to compare
        tokenizer: Shared tokenizer
        texts: Input texts
        device: Device
        max_length: Max sequence length

    Returns:
        (mean_cos_theta, std_cos_theta)
    """
    model_a.eval()
    model_b.eval()

    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size

    cos_scores = []

    with torch.no_grad():
        for text in texts[:20]:
            if len(text) < 10:
                continue

            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)

            out_a = model_a(inputs.input_ids, output_hidden_states=True)
            out_b = model_b(inputs.input_ids, output_hidden_states=True)

            h_a = out_a.hidden_states[-1].float()
            h_b = out_b.hidden_states[-1].float()

            # Align dimensions
            if dim_a != dim_b:
                min_dim = min(dim_a, dim_b)
                h_a_flat = h_a[:, :, :min_dim].reshape(-1)
                h_b_flat = h_b[:, :, :min_dim].reshape(-1)
            else:
                h_a_flat = h_a.reshape(-1)
                h_b_flat = h_b.reshape(-1)

            cos = torch.nn.functional.cosine_similarity(
                h_a_flat, h_b_flat, dim=0
            ).item()
            cos_scores.append(cos)

    return float(np.mean(cos_scores)), float(np.std(cos_scores))
