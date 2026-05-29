#Perplexity Computation Module

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 512,
    show_progress: bool = True,
) -> float:
    """
    Compute perplexity of a language model on a set of texts.

    Uses standard cross-entropy loss averaged over all tokens,
    then exponentiated.

    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        texts: List of text strings to evaluate
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length for tokenization
        show_progress: Whether to show tqdm progress bar

    Returns:
        Perplexity (float). Lower is better.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        iterator = tqdm(texts, desc="Computing PPL", leave=False) if show_progress else texts
        for text in iterator:
            if len(text) < 10:
                continue

            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)

            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            num_tokens = inputs.input_ids.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def compute_stitched_perplexity(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 512,
    show_progress: bool = True,
) -> float:
    """
    Compute perplexity of representation-stitched output.

    Takes hidden states from both models, averages them, and
    decodes through model_a's LM head.

    Args:
        model_a: Primary model (provides LM head for decoding)
        model_b: Secondary model (provides hidden states)
        tokenizer: Shared tokenizer
        texts: Evaluation texts
        device: Device string
        max_length: Max sequence length
        show_progress: Show progress bar

    Returns:
        Stitched perplexity (float).
    """
    model_a.eval()
    model_b.eval()

    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size

    ppl_scores = []

    with torch.no_grad():
        iterator = tqdm(texts, desc="Stitching", leave=False) if show_progress else texts
        for text in iterator:
            if len(text) < 10:
                continue

            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)

            out_a = model_a(inputs.input_ids, output_hidden_states=True)
            out_b = model_b(inputs.input_ids, output_hidden_states=True)

            h_a = out_a.hidden_states[-1]
            h_b = out_b.hidden_states[-1]

            # Align dimensions if needed
            if dim_a != dim_b:
                if dim_b > dim_a:
                    h_b = h_b[:, :, :dim_a]
                else:
                    padding = torch.zeros(
                        h_b.shape[0], h_b.shape[1], dim_a - dim_b,
                        device=device, dtype=h_b.dtype,
                    )
                    h_b = torch.cat([h_b, padding], dim=-1)

            # Stitch: average hidden states
            h_stitched = (h_a + h_b) / 2.0

            # Decode via model_a's LM head
            logits = model_a.lm_head(h_stitched)

            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            ppl = torch.exp(loss).item()

            if ppl < 1e10:  # Sanity check
                ppl_scores.append(ppl)

    if len(ppl_scores) == 0:
        return float('inf')

    return float(np.mean(ppl_scores))
