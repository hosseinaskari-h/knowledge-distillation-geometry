"""
Comprehensive Model Compatibility Validation

Tests BOTH:
1. Weight Merging (for compatible architectures)
2. Representation Stitching (for incompatible architectures)

This validates paper's predictions for both scenarios.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from copy import deepcopy

# ============================================================================
# PART 1: WEIGHT MERGING (Same-Architecture Compatible Pairs)
# ============================================================================

def merge_models_with_projection(model_a, model_b, alpha=0.5):
    """
    Merge models via weight averaging, handling dimension mismatches.
    
    For GPT-2 (768) + GPT-2-Medium (1024):
    - Project smaller model to larger dimension
    - Then merge
    """
    merged_model = deepcopy(model_a)  # Use larger model as template
    
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state = merged_model.state_dict()
    
    for key in merged_state.keys():
        if key not in state_a or key not in state_b:
            # Keep model_a's parameter if missing in b
            merged_state[key] = state_a[key]
            continue
            
        param_a = state_a[key]
        param_b = state_b[key]
        
        # Handle shape mismatches: Embed param_b (Source) into param_a (Target) shape
        if param_a.shape != param_b.shape:
            if len(param_a.shape) != len(param_b.shape):
                 # Rank mismatch - cannot merge
                 merged_state[key] = param_a
                 continue
                 
            # Check if B fits inside A
            if all(sb <= sa for sb, sa in zip(param_b.shape, param_a.shape)):
                # Create container of shape A
                new_b = torch.zeros_like(param_a)
                # Copy B into it (Top-Left alignment)
                slices = tuple(slice(0, s) for s in param_b.shape)
                new_b[slices] = param_b
                param_b = new_b
            else:
                # B is larger in some dim? Cannot merge into A without truncation (undesirable)
                # Or A is Source and B is Target? 
                # The caller ensures A is Target (Larger).
                # But if dimensions are mixed (A wider, B taller), skip.
                print(f"Warning: Cannot align {key} {param_b.shape} -> {param_a.shape}, keeping Target")
                merged_state[key] = param_a
                continue

        # Merge
        merged_state[key] = alpha * param_a + (1 - alpha) * param_b
    
    merged_model.load_state_dict(merged_state)
    return merged_model

def compute_perplexity(model, tokenizer, texts, device='cuda', max_length=512):
    """Compute perplexity on texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL", leave=False):
            if len(text) < 10:
                continue
            
            inputs = tokenizer(text, return_tensors="pt", 
                             truncation=True, max_length=max_length).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            num_tokens = inputs.input_ids.numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)

def test_weight_merging(model_a_name, model_b_name, test_texts, device='cuda'):
    """
    Test weight merging for compatible architectures.
    Returns None if models are incompatible.
    """
    print(f"\n[WEIGHT MERGE] Testing: {model_a_name} + {model_b_name}")
    
    # Load
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device)
    model_a.eval()
    model_b.eval()
    
    # Check compatibility
    if tokenizer_a.vocab_size != AutoTokenizer.from_pretrained(model_b_name).vocab_size:
        print(f"  [X] Incompatible: different vocabularies")
        return None
    
    # Check if same family (can attempt merge)
    family_a = model_a.config.model_type
    family_b = model_b.config.model_type
    
    if family_a != family_b:
        print(f"  [X] Incompatible: different families ({family_a} vs {family_b})")
        return None
    
    # Evaluate individuals
    print("  Evaluating Model A...")
    ppl_a = compute_perplexity(model_a, tokenizer_a, test_texts, device)
    
    print("  Evaluating Model B...")
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    ppl_b = compute_perplexity(model_b, tokenizer_b, test_texts, device)
    
    # Merge
    print("  Merging weights...")
    try:
        # Use larger model as template
        if model_a.config.hidden_size >= model_b.config.hidden_size:
            merged = merge_models_with_projection(model_a, model_b)
        else:
            merged = merge_models_with_projection(model_b, model_a)
    except Exception as e:
        print(f"  [X] Merge failed: {e}")
        return None
    
    # Evaluate merged
    print("  Evaluating merged model...")
    ppl_merged = compute_perplexity(merged, tokenizer_a, test_texts, device)
    
    # Metrics
    avg_individual = (ppl_a + ppl_b) / 2
    degradation = (ppl_merged - avg_individual) / avg_individual * 100
    success = degradation < 50.0  # Generous threshold
    
    del model_a, model_b, merged
    torch.cuda.empty_cache()
    
    return {
        'method': 'weight_merge',
        'ppl_a': float(ppl_a),
        'ppl_b': float(ppl_b),
        'ppl_merged': float(ppl_merged),
        'avg_individual': float(avg_individual),
        'degradation_pct': float(degradation),
        'success': success
    }

# ============================================================================
# PART 2: REPRESENTATION STITCHING (Cross-Architecture Incompatible Pairs)
# ============================================================================

def test_representation_stitching(model_a_name, model_b_name, test_texts, device='cuda'):
    """
    Test representation compatibility via stitching.
    h_merged = (h_a + h_b) / 2
    """
    print(f"\n[REP STITCH] Testing: {model_a_name} + {model_b_name}")
    
    # Load
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device)
    model_a.eval()
    model_b.eval()
    
    # Baseline: model A alone
    print("  Evaluating Model A (baseline)...")
    ppl_baseline = compute_perplexity(model_a, tokenizer_a, test_texts, device)
    
    # Stitched evaluation
    print("  Evaluating stitched representations...")
    ppl_scores = []
    
    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size
    
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Stitching", leave=False):
            if len(text) < 10:
                continue
            
            inputs = tokenizer_a(text, return_tensors="pt", 
                               truncation=True, max_length=512).to(device)
            
            # Get hidden states
            out_a = model_a(inputs.input_ids, output_hidden_states=True)
            out_b = model_b(inputs.input_ids, output_hidden_states=True)
            
            h_a = out_a.hidden_states[-1]  # Last layer
            h_b = out_b.hidden_states[-1]
            
            # Align dimensions
            if dim_a != dim_b:
                if dim_b > dim_a:
                    h_b = h_b[:, :, :dim_a]
                else:
                    padding = torch.zeros(h_b.shape[0], h_b.shape[1], 
                                        dim_a - dim_b, device=device)
                    h_b = torch.cat([h_b, padding], dim=-1)
            
            # Stitch
            h_stitched = (h_a + h_b) / 2.0
            
            # Decode via A's head
            logits = model_a.lm_head(h_stitched)
            
            # Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            ppl_scores.append(torch.exp(loss).item())
    
    ppl_stitched = np.mean(ppl_scores)
    
    # Metrics
    degradation = (ppl_stitched - ppl_baseline) / ppl_baseline * 100
    success = degradation < 50.0
    
    del model_a, model_b
    torch.cuda.empty_cache()
    
    return {
        'method': 'representation_stitch',
        'ppl_baseline': float(ppl_baseline),
        'ppl_stitched': float(ppl_stitched),
        'degradation_pct': float(degradation),
        'success': success
    }

# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPATIBILITY VALIDATION")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load test data
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_texts = [item['text'] for item in dataset if len(item['text']) > 100][:30]
        print(f"Loaded {len(test_texts)} test samples from WikiText-2\n")
    except:
        test_texts = ["The quick brown fox jumps over the lazy dog. " * 20] * 10
        print(f"Using fallback test data\n")
    
    # Define test configurations
    configs = [
        {
            'name': 'High Sync - Same Arch',
            'model_a': 'gpt2',
            'model_b': 'gpt2',  # Same model, will test identity
            'cos_theta': 1.000,
            'method': 'weight_merge',
            'expected': 'SUCCESS'
        },
        {
            'name': 'Low Sync - Cross Scale',
            'model_a': 'gpt2',
            'model_b': 'gpt2-medium',
            'cos_theta': 0.11,
            'method': 'weight_merge',
            'expected': 'FAILURE'
        },
        {
            'name': 'High Sync - Distillation',
            'model_a': 'gpt2',
            'model_b': 'distilgpt2',
            'cos_theta': 0.998,
            'method': 'representation_stitch',  # Can't weight-merge different layers
            'expected': 'SUCCESS'
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"TEST: {config['name']}")
        print(f"Paper cos theta: {config['cos_theta']}")
        print(f"Method: {config['method']}")
        print(f"Expected: {config['expected']}")
        print(f"{'='*80}")
        
        # Run appropriate test
        if config['method'] == 'weight_merge':
            result = test_weight_merging(
                config['model_a'], config['model_b'], test_texts, device
            )
        else:
            result = test_representation_stitching(
                config['model_a'], config['model_b'], test_texts, device
            )
        
        if result is not None:
            result.update({
                'test_name': config['name'],
                'model_a': config['model_a'],
                'model_b': config['model_b'],
                'cos_theta': config['cos_theta'],
                'expected': config['expected']
            })
            results.append(result)
            
            # Print summary
            print(f"\n  Results:")
            if result['method'] == 'weight_merge':
                print(f"    Model A PPL:    {result['ppl_a']:.2f}")
                print(f"    Model B PPL:    {result['ppl_b']:.2f}")
                print(f"    Merged PPL:     {result['ppl_merged']:.2f}")
                print(f"    Avg Individual: {result['avg_individual']:.2f}")
            else:
                print(f"    Baseline PPL:   {result['ppl_baseline']:.2f}")
                print(f"    Stitched PPL:   {result['ppl_stitched']:.2f}")
            
            print(f"    Degradation:    {result['degradation_pct']:.1f}%")
            print(f"    Success:        {'YES' if result['success'] else 'NO'}")
            
            match = (result['success'] and config['expected']=='SUCCESS') or \
                   (not result['success'] and config['expected']=='FAILURE')
            print(f"    Prediction OK:  {'[OK]' if match else '[X]'}")
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Test':<30} | {'Method':<10} | {'cos theta':<7} | {'Degrad%':<8} | {'Match'}")
    print("-" * 80)
    
    for r in results:
        match = (r['success'] and r['expected']=='SUCCESS') or \
               (not r['success'] and r['expected']=='FAILURE')
        method_short = 'Weight' if r['method'] == 'weight_merge' else 'Stitch'
        print(f"{r['test_name']:<30} | {method_short:<10} | {r['cos_theta']:<7.3f} | "
              f"{r['degradation_pct']:<8.1f} | {'[OK]' if match else '[X]'}")
    
    print(f"{'='*80}\n")
    
    # Save with custom encoder for numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (bool, np.bool_)): # Fix bool error
                return bool(obj)
            return super(NpEncoder, self).default(obj)

    with open('validation_results_comprehensive.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print("Results saved to: validation_results_comprehensive.json")
    
    # Verdict
    matches = sum(1 for r in results 
                  if (r['success'] and r['expected']=='SUCCESS') or 
                     (not r['success'] and r['expected']=='FAILURE'))
    
    print(f"\nVERDICT: {matches}/{len(results)} predictions validated")
    
    if matches == len(results):
        print("[OK] All predictions VALIDATED")
    elif matches >= len(results) * 0.7:
        print("~ Predictions PARTIALLY validated")
    else:
        print("[X] Predictions NOT validated")

if __name__ == '__main__':
    main()
