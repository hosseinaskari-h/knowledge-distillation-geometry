# Knowledge Distillation Geometry - Rvised

This folder contains the complete, validated codebase for the paper "Knowledge Distillation Preserves Representational Geometry".

## Contents

- `src/`: Core library for vector dynamics and analysis.
- `paper/`: LaTeX source for the paper (`main.tex`).
- `run_experiments.py`: CLI runner for all experiments.
- `requirements.txt`: Python dependencies.

## Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Reproduce Paper Results**:
    Run the full suite (123 experiments) as described in Appendix B.1:
    ```bash
    python run_experiments.py --experiment reproduction --steps 100
    ```
    Results will be saved to `data/reproduction/`.


