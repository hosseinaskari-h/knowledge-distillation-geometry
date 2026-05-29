#!/usr/bin/env python3
"""
 Experiment Runner for Coupled Transformer Dynamics

CLI-based interface for running all paper experiments without the GUI.
Uses the existing modular architecture in src/.

Usage:
    python run_experiments.py --experiment dynamics --models gpt2 distilgpt2
    python run_experiments.py --experiment merge --models gpt2-medium microsoft/DialoGPT-medium
    python run_experiments.py --experiment cka --models gpt2,distilgpt2 gpt2-medium,microsoft/DialoGPT-medium
    python run_experiments.py --experiment cos_theta --models gpt2 distilgpt2
    python run_experiments.py --experiment full
    python run_experiments.py --experiment reproduction
    python run_experiments.py --list-configs
"""

import os
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.agents.hf_agent import HuggingFaceAgent
from src.training.vector_dynamics import (
    VectorDynamicsTrainer,
    DirectCoupling,
    BlendedCoupling,
    DynamicsTrajectory,
    analyze_trajectory,
    create_coupling_strategy,
)
from src.analysis.cka import CKAAnalyzer, compute_model_cka
from src.analysis.perplexity import compute_perplexity, compute_stitched_perplexity
from src.analysis.merging import (
    weight_merge,
    evaluate_weight_merge,
    compute_cos_theta,
)

logger = logging.getLogger(__name__)


# TERMINAL

class Colors:
    """ANSI color codes for terminal output."""
    # Check if terminal supports colors
    ENABLED = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() or os.environ.get('FORCE_COLOR')

    RESET   = '\033[0m'  if ENABLED else ''
    BOLD    = '\033[1m'  if ENABLED else ''
    DIM     = '\033[2m'  if ENABLED else ''

    # Colors
    GREEN   = '\033[92m' if ENABLED else ''
    CYAN    = '\033[96m' if ENABLED else ''
    YELLOW  = '\033[93m' if ENABLED else ''
    RED     = '\033[91m' if ENABLED else ''
    BLUE    = '\033[94m' if ENABLED else ''
    MAGENTA = '\033[95m' if ENABLED else ''
    WHITE   = '\033[97m' if ENABLED else ''


def cprint(msg: str, color: str = '', bold: bool = False):
    """Print with color."""
    prefix = (Colors.BOLD if bold else '') + color
    suffix = Colors.RESET if prefix else ''
    print(f"{prefix}{msg}{suffix}")


def header(msg: str):
    """Print a major section header."""
    cprint(f"\n{msg}", Colors.GREEN, bold=True)


def subheader(msg: str):
    """Print a sub-section header."""
    cprint(f"\n# {msg}", Colors.CYAN, bold=True)


def info(msg: str):
    """Print info message."""
    cprint(f"  {msg}", Colors.WHITE)


def success(msg: str):
    """Print success message."""
    cprint(f"  [+] {msg}", Colors.GREEN)


def warn(msg: str):
    """Print warning message."""
    cprint(f"  [!] {msg}", Colors.YELLOW)


def error(msg: str):
    """Print error message."""
    cprint(f"  [x] {msg}", Colors.RED, bold=True)


def result_line(label: str, value: str):
    """Print a result key-value pair."""
    cprint(f"  {label}: ", Colors.DIM, bold=False)
    # Print on same line by using end=''
    sys.stdout.write(f"{Colors.BOLD}{Colors.WHITE}{value}{Colors.RESET}\n")


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds colors based on level."""
    LEVEL_COLORS = {
        logging.DEBUG:    Colors.DIM,
        logging.INFO:     Colors.CYAN,
        logging.WARNING:  Colors.YELLOW,
        logging.ERROR:    Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, '')
        msg = super().format(record)
        return f"{color}{msg}{Colors.RESET}"


def setup_logging(verbose: bool = False):
    """Configure logging with colors, writing to stdout instead of stderr."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create handler that writes to stdout (not stderr = no red in terminals)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy library loggers
    for noisy in ['urllib3', 'filelock', 'fsspec', 'datasets', 'huggingface_hub']:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# CONFIGURATION

# Paper experiment configurations (Legacy)
PAPER_CONFIGS = {
    "identity": {
        "name": "Identity (GPT-2 + GPT-2)",
        "model_a": "gpt2",
        "model_b": "gpt2",
        "expected_cos": 1.0,
        "coupling": "direct",
    },
    "distillation": {
        "name": "Distillation (GPT-2 + DistilGPT-2)",
        "model_a": "gpt2",
        "model_b": "distilgpt2",
        "expected_cos": 0.998,
        "coupling": "direct",
    },
    "finetuning_small": {
        "name": "Fine-tuning (GPT-2 + DialoGPT-Small)",
        "model_a": "gpt2",
        "model_b": "microsoft/DialoGPT-small",
        "expected_cos": 0.95,
        "coupling": "direct",
    },
    "finetuning_medium": {
        "name": "Fine-tuning (GPT-2-Med + DialoGPT-Med)",
        "model_a": "gpt2-medium",
        "model_b": "microsoft/DialoGPT-medium",
        "expected_cos": 0.949,
        "coupling": "direct",
    },
    "cross_scale": {
        "name": "Cross-Scale (GPT-2 + GPT-2-Medium)",
        "model_a": "gpt2",
        "model_b": "gpt2-medium",
        "expected_cos": 0.11,
        "coupling": "direct",
    },
    "cross_family": {
        "name": "Cross-Family (GPT-2 + BERT)",
        "model_a": "gpt2",
        "model_b": "bert-base-uncased",
        "expected_cos": 0.0,
        "coupling": "direct",
    },
}

# Full 41-Configuration Suite (Table 8 in Paper)
REPRODUCTION_CONFIGS = []

# 1. Identity (GPT-2 + GPT-2) - 8 configs
for init in ['random', 'same', 'text']:
    REPRODUCTION_CONFIGS.append({
        'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'direct', 'init': init, 'force_proj': False
    })
for alpha in [0.5, 0.3, 0.7]:
    REPRODUCTION_CONFIGS.append({
        'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'blended', 'alpha': alpha, 'init': 'random', 'force_proj': False
    })
for decay in [0.1, 0.3]:
    REPRODUCTION_CONFIGS.append({
        'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'decay', 'decay': decay, 'init': 'random', 'force_proj': False
    })

# 2. Distillation Self (DistilGPT-2 + DistilGPT-2) - 3 configs
REPRODUCTION_CONFIGS.append({'model_a': 'distilgpt2', 'model_b': 'distilgpt2', 'coupling': 'direct', 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'distilgpt2', 'model_b': 'distilgpt2', 'coupling': 'blended', 'alpha': 0.5, 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'distilgpt2', 'model_b': 'distilgpt2', 'coupling': 'decay', 'decay': 0.1, 'init': 'random', 'force_proj': False})

# 3. Projection Control (GPT-2 + GPT-2 Forced) - 2 configs
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'direct', 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'direct', 'init': 'text', 'force_proj': True})

# 4. Encoder - 2 configs
REPRODUCTION_CONFIGS.append({'model_a': 'bert-base-uncased', 'model_b': 'distilbert-base-uncased', 'coupling': 'direct', 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'bert-base-uncased', 'model_b': 'bert-base-uncased', 'coupling': 'direct', 'init': 'random', 'force_proj': False})

# 5. Fine-tuning - 2 configs
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2-medium', 'model_b': 'microsoft/DialoGPT-medium', 'coupling': 'direct', 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2-medium', 'model_b': 'gpt2-medium', 'coupling': 'direct', 'init': 'random', 'force_proj': False})

# 6. Cross-Scale (Implicit Proj) - 5 configs
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2-medium', 'coupling': 'direct', 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2-medium', 'coupling': 'blended', 'alpha': 0.5, 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2-large', 'coupling': 'direct', 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'gpt2-large', 'coupling': 'blended', 'alpha': 0.5, 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2-medium', 'model_b': 'gpt2-large', 'coupling': 'direct', 'init': 'random', 'force_proj': True})

# 7. Distillation Pair - 3 configs
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'distilgpt2', 'coupling': 'direct', 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'distilgpt2', 'coupling': 'blended', 'alpha': 0.5, 'init': 'random', 'force_proj': False})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'distilgpt2', 'coupling': 'decay', 'decay': 0.1, 'init': 'random', 'force_proj': False})

# 8. Cross-Architecture - 5 configs
# Note: Using GPT-Neo-125M or similar small one if 1.3B is too big? Paper used 1.3B.
# We will use 'EleutherAI/gpt-neo-1.3B' as per paper.
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'EleutherAI/gpt-neo-1.3B', 'coupling': 'direct', 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'EleutherAI/gpt-neo-1.3B', 'coupling': 'blended', 'alpha': 0.5, 'init': 'random', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'EleutherAI/gpt-neo-1.3B', 'coupling': 'blended', 'alpha': 0.5, 'init': 'text', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2', 'model_b': 'EleutherAI/gpt-neo-1.3B', 'coupling': 'decay', 'decay': 0.1, 'init': 'text', 'force_proj': True})
REPRODUCTION_CONFIGS.append({'model_a': 'gpt2-medium', 'model_b': 'EleutherAI/gpt-neo-1.3B', 'coupling': 'direct', 'init': 'random', 'force_proj': True})

# 9. Phase Transition Sweep - 11 configs
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # Skip duplicates handled in Group 1/2? No, we run them as part of sweep logic.
    REPRODUCTION_CONFIGS.append({
        'model_a': 'gpt2', 'model_b': 'gpt2', 'coupling': 'blended', 'alpha': alpha, 'init': 'random', 'force_proj': False,
        'group': 'phase_sweep' # Marker to distinguish
    })


# HELPER FUNCTIONS

class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/torch types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


def load_test_texts(n_samples: int = 30) -> list:
    """Load WikiText-2 test samples."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item['text'] for item in dataset if len(item['text']) > 100][:n_samples]
        print(f"Loaded {len(texts)} test samples from WikiText-2")
        return texts
    except Exception as e:
        print(f"Dataset load error: {e}")
        fallback = ["The quick brown fox jumps over the lazy dog. " * 20] * 10
        print(f"Using fallback test data ({len(fallback)} samples)")
        return fallback


def save_results(results: dict, output_dir: str, experiment_name: str):
    """Save results to JSON with timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"\nResults saved to: {filepath}")
    return filepath


def load_model_pair(model_a_name: str, model_b_name: str, device: str):
    """Load a pair of models and their tokenizer."""
    print(f"\nLoading models...")
    print(f"  Model A: {model_a_name}")
    print(f"  Model B: {model_b_name}")

    dtype = torch.float16 if device == 'cuda' else torch.float32

    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_name, torch_dtype=dtype
    ).to(device)
    model_a.eval()

    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_name, torch_dtype=dtype
    ).to(device)
    model_b.eval()

 
    tokenizer = AutoTokenizer.from_pretrained(model_a_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model A dim: {model_a.config.hidden_size}")
    print(f"  Model B dim: {model_b.config.hidden_size}")
    print(f"  Device: {device}, dtype: {dtype}")

    return model_a, model_b, tokenizer


# EXPERIMENT RUNNERS

def run_coupled_dynamics(
    model_a_name: str,
    model_b_name: str,
    device: str = 'cuda',
    num_steps: int = 1000,
    seeds: list = [42, 43, 44],
    coupling_type: str = 'direct',
    alpha: float = 0.5,
    decay: float = 0.1,
    force_projection: bool = False,
    init_mode: str = 'random',
    output_dir: str = 'data/results',
) -> dict:
    """
    Run coupled dynamics experiment.

    Uses the VectorDynamicsTrainer from src/training/vector_dynamics.py
    with HuggingFaceAgent from src/agents/hf_agent.py.
    """
    header("COUPLED DYNAMICS EXPERIMENT")
    info(f"Models: {model_a_name} <-> {model_b_name}")
    info(f"Coupling: {coupling_type} (alpha={alpha}, decay={decay}, proj={force_projection})")
    info(f"Init: {init_mode}")

    # Create agents
    agent_a = HuggingFaceAgent("Agent_A", {
        'model': model_a_name,
        'device': device,
        'generation': {'max_new_tokens': 20},
    })
    agent_b = HuggingFaceAgent("Agent_B", {
        'model': model_b_name,
        'device': device,
        'generation': {'max_new_tokens': 20},
    })

    # Create coupling strategy
    config = {
        'coupling': coupling_type,
        'alpha': alpha,
        'decay': decay,
        'force_projection': force_projection
    }
    coupling = create_coupling_strategy(coupling_type, config)

    # Trainer config
    trainer_config = {
        'coupling': coupling_type,
        'model_a': model_a_name,
        'model_b': model_b_name,
        **config
    }

    all_results = []

    for seed in seeds:
        cprint(f"\n--- Seed {seed} ---", Colors.MAGENTA, bold=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        trainer = VectorDynamicsTrainer(agent_a, agent_b, coupling, trainer_config)

        init_text = "The study of consciousness reveals profound questions about the nature of mind."

        trajectory = trainer.run(
            num_steps=num_steps,
            init_mode=init_mode,
            init_text_a=init_text if init_mode == 'text' else None,
            init_text_b=init_text if init_mode == 'text' else None,
        )

        # Analyze trajectory
        analysis = analyze_trajectory(trajectory)

        result = {
            'seed': seed,
            'init_mode': init_mode,
            'model_a': model_a_name,
            'model_b': model_b_name,
            'coupling': coupling_type,
            'alpha': alpha,
            'decay': decay,
            'force_projection': force_projection,
            'num_steps': num_steps,
            'final_cos_theta': trajectory.states[-1].cosine_similarity if trajectory.states else None,
            'sync_index': analysis.get('synchronization_index'),
            'lyapunov': analysis.get('largest_lyapunov'),
            'attractor_dim': analysis.get('attractor_dimension'),
            # 'cos_trajectory': [s.cosine_similarity for s in trajectory.states], # Too large for full reproduction log
        }
        all_results.append(result)

        if trajectory.states:
            success(f"Final cos θ: {trajectory.states[-1].cosine_similarity:.4f}")
            info(f"Sync index: {analysis.get('synchronization_index', 'N/A')}")

    # Summary
    final_cos_values = [r['final_cos_theta'] for r in all_results if r['final_cos_theta'] is not None]
    summary = {
        'experiment': 'coupled_dynamics',
        'model_a': model_a_name,
        'model_b': model_b_name,
        'config': {
            'coupling': coupling_type,
            'init': init_mode,
            'alpha': alpha,
            'decay': decay,
            'force_proj': force_projection,
        },
        'mean_final_cos': float(np.mean(final_cos_values)) if final_cos_values else None,
        'std_final_cos': float(np.std(final_cos_values)) if final_cos_values else None,
        'runs': all_results,
    }

    success(f"SUMMARY: cos θ = {summary['mean_final_cos']:.4f} ± {summary['std_final_cos']:.4f}")

    # Cleanup
    del agent_a, agent_b
    torch.cuda.empty_cache()

    return summary


def run_weight_merge_experiment(
    model_a_name: str,
    model_b_name: str,
    device: str = 'cuda',
    alphas: list = [0.3, 0.5, 0.7],
    output_dir: str = 'data/results',
) -> dict:
    """Run weight merging experiment."""
    header("WEIGHT MERGING EXPERIMENT")

    model_a, model_b, tokenizer = load_model_pair(model_a_name, model_b_name, device)
    test_texts = load_test_texts()

    results = evaluate_weight_merge(model_a, model_b, tokenizer, test_texts, alphas, device)
    results['model_a'] = model_a_name
    results['model_b'] = model_b_name
    results['experiment'] = 'weight_merge'

    del model_a, model_b
    torch.cuda.empty_cache()

    save_results(results, output_dir, 'merge')
    return results


def run_cka_experiment(
    model_pairs: list,
    device: str = 'cuda',
    output_dir: str = 'data/results',
) -> dict:
    """
    Run CKA comparison across multiple model pairs.

    Args:
        model_pairs: List of (model_a_name, model_b_name) tuples
    """
    header("CKA COMPARISON TABLE")

    test_texts = load_test_texts()
    results = []

    for model_a_name, model_b_name in model_pairs:
        cprint(f"\n--- {model_a_name} + {model_b_name} ---", Colors.MAGENTA, bold=True)
        model_a, model_b, tokenizer = load_model_pair(model_a_name, model_b_name, device)

        cka_score = compute_model_cka(model_a, model_b, tokenizer, test_texts, device)
        cos_mean, cos_std = compute_cos_theta(model_a, model_b, tokenizer, test_texts, device)

        results.append({
            'pair': f"{model_a_name} + {model_b_name}",
            'cka': cka_score,
            'cos_theta': cos_mean,
            'cos_theta_std': cos_std,
        })

        success(f"CKA = {cka_score:.4f}")
        success(f"cos θ = {cos_mean:.4f} ± {cos_std:.4f}")

        del model_a, model_b
        torch.cuda.empty_cache()

    # Summary table
    cprint(f"\n{'Pair':<45} | {'CKA':<8} | {'cos θ':<10}", Colors.WHITE, bold=True)
    for r in results:
        cprint(f"{r['pair']:<45} | {r['cka']:<8.4f} | {r['cos_theta']:<10.4f}", Colors.WHITE)

    summary = {'experiment': 'cka_table', 'pairs': results}
    save_results(summary, output_dir, 'cka')
    return summary


def run_cos_theta_experiment(
    model_a_name: str,
    model_b_name: str,
    device: str = 'cuda',
    output_dir: str = 'data/results',
) -> dict:
    """Run cos θ measurement between two models."""
    header("COS THETA MEASUREMENT")

    model_a, model_b, tokenizer = load_model_pair(model_a_name, model_b_name, device)
    test_texts = load_test_texts()

    cos_mean, cos_std = compute_cos_theta(model_a, model_b, tokenizer, test_texts, device)

    success(f"cos θ = {cos_mean:.4f} ± {cos_std:.4f}")

    results = {
        'experiment': 'cos_theta',
        'model_a': model_a_name,
        'model_b': model_b_name,
        'cos_theta': cos_mean,
        'cos_theta_std': cos_std,
    }

    del model_a, model_b
    torch.cuda.empty_cache()

    save_results(results, output_dir, 'cos_theta')
    return results


def run_reproduction_suite(
    device: str = 'cuda',
    seeds: list = [42, 43, 44],
    num_steps: int = 1000,
    output_dir: str = 'data/reproduction',
) -> dict:
    """
    Run the full 123-experiment reproduction suite.
    """
    header("FULL PAPER REPRODUCTION SUITE (123 Experiments)")
    info(f"Target: {len(REPRODUCTION_CONFIGS)} configurations x {len(seeds)} seeds = {len(REPRODUCTION_CONFIGS)*len(seeds)} total runs")
    
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []

    for i, cfg in enumerate(REPRODUCTION_CONFIGS):
        cprint(f"\n[{i+1}/{len(REPRODUCTION_CONFIGS)}] Config: {cfg['model_a']} + {cfg['model_b']} ({cfg['coupling']}, {cfg['init']})", Colors.BLUE, bold=True)
        
        try:
            summary = run_coupled_dynamics(
                model_a_name=cfg['model_a'],
                model_b_name=cfg['model_b'],
                device=device,
                num_steps=num_steps,
                seeds=seeds,
                coupling_type=cfg['coupling'],
                alpha=cfg.get('alpha', 0.5),
                decay=cfg.get('decay', 0.1),
                force_projection=cfg.get('force_proj', False),
                init_mode=cfg['init'],
                output_dir=output_dir,
            )
            all_summaries.append(summary)
        except Exception as e:
            error(f"Failed to run config {i+1}: {e}")
            all_summaries.append({'error': str(e), 'config': cfg})

        # Save incremental progress
        save_results({'configs_completed': i+1, 'latest': summary}, output_dir, f'reproduction_progress')

    # Final Compilation
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'results': all_summaries
    }
    path = save_results(full_report, output_dir, 'final_reproduction_report')
    success(f"Reproduction suite complete. Saved to {path}")
    return full_report


def run_full_validation(
    device: str = 'cuda',
    seeds: list = [42, 43, 44],
    num_steps: int = 1000,
    output_dir: str = 'data/results',
) -> dict:
    """Run the legacy validation suite."""
    header("FULL PAPER VALIDATION (LEGACY)")

    
    all_results = {}
    

    
    keys = ["identity", "distillation", "finetuning_small", "finetuning_medium", "cross_scale"]
    
    dynamics_results = {}
    for key in keys:
        cfg = PAPER_CONFIGS[key]
        print(f"Running {key}...")
        res = run_coupled_dynamics(
            cfg['model_a'], cfg['model_b'],
            device=device, num_steps=num_steps, seeds=seeds,
            coupling_type=cfg['coupling']
        )
        dynamics_results[key] = res
        
    all_results['dynamics'] = dynamics_results
    
    # CKA
    cka_res = run_cka_experiment([
        ("gpt2", "gpt2"), ("gpt2", "distilgpt2"), 
        ("gpt2-medium", "microsoft/DialoGPT-medium"), ("gpt2", "gpt2-medium")
    ], device, output_dir)
    all_results['cka'] = cka_res
    
    save_results(all_results, output_dir, 'full_validation')
    return all_results


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Terminal Experiment Runner for Coupled Transformer Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--experiment', '-e',
        choices=['dynamics', 'merge', 'cka', 'cos_theta', 'full', 'reproduction', 'repro'],
        help='Experiment to run',
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Model names.',
    )
    parser.add_argument(
        '--device', '-d',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (default: cuda if available)',
    )
    parser.add_argument(
        '--steps', '-s',
        type=int, default=1000,
        help='Number of dynamics steps (default: 1000)',
    )
    parser.add_argument(
        '--seeds',
        type=str, default='42,43,44',
        help='Comma-separated seeds',
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float, default=0.5,
        help='Blending alpha',
    )
    parser.add_argument(
        '--decay',
        type=float, default=0.1,
        help='Decay beta',
    )
    parser.add_argument(
        '--coupling',
        choices=['direct', 'blended', 'projected', 'decay'],
        default='direct',
        help='Coupling strategy',
    )
    parser.add_argument(
        '--init',
        choices=['random', 'text', 'same'],
        default='random',
        help='Initialization mode',
    )
    parser.add_argument(
        '--force-projection',
        action='store_true',
        help='Force projection layers even if dims match',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data/results',
        help='Output directory',
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List predefined paper experiment configs',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging with colors (writes to stdout, not stderr)
    setup_logging(verbose=args.verbose)

    # List configs
    if args.list_configs:
        print("\nPredefined Paper Configurations:")
        for key, cfg in PAPER_CONFIGS.items():
            print(f"  {key:<20} {cfg['name']}")
        return

    if args.experiment is None:
        print("Error: --experiment is required")
        return

    seeds = [int(s) for s in args.seeds.split(',')]

    # Resolve config
    # ... arg parsing logic ...
    
    if args.experiment == 'reproduction' or args.experiment == 'repro':
        run_reproduction_suite(
            device=args.device,
            seeds=seeds,
            num_steps=args.steps,
            output_dir=args.output_dir
        )
        return

    # Check requires two models for simple modes
    if args.experiment in ['dynamics', 'merge', 'cos_theta'] and (not args.models or len(args.models) < 2):
         print("Error: --models requires two model names")
         return
         
    model_a = args.models[0] if args.models else None
    model_b = args.models[1] if args.models and len(args.models)>1 else None

    # Run experiment
    if args.experiment == 'dynamics':
        run_coupled_dynamics(
            model_a, model_b,
            device=args.device,
            num_steps=args.steps,
            seeds=seeds,
            coupling_type=args.coupling,
            alpha=args.alpha,
            decay=args.decay,
            force_projection=args.force_projection,
            init_mode=args.init,
            output_dir=args.output_dir,
        )

    elif args.experiment == 'merge':
        run_weight_merge_experiment(
            model_a, model_b,
            device=args.device,
            output_dir=args.output_dir,
        )

    elif args.experiment == 'cka':
        pairs = []
        if args.models:
            for pair_str in args.models:
                parts = pair_str.split(',')
                if len(parts) == 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
        else:
             pairs = [("gpt2", "gpt2"), ("gpt2", "distilgpt2")]
        run_cka_experiment(pairs, device=args.device, output_dir=args.output_dir)

    elif args.experiment == 'cos_theta':
        run_cos_theta_experiment(
            model_a, model_b,
            device=args.device,
            output_dir=args.output_dir,
        )

    elif args.experiment == 'full':
        run_full_validation(
            device=args.device,
            seeds=seeds,
            num_steps=args.steps,
            output_dir=args.output_dir,
        )

if __name__ == '__main__':
    main()
