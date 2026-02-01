# Emergent Protocols: Replication Hub

This directory contains the full source code and interactive GUI for replicating the findings in:
**"Knowledge Distillation Preserves Representational Geometry"**

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Replication GUI:
   ```bash
   python emergent_gui.py
   ```

## Features

- **Vector Dynamics:** Visualize coupled transformer dynamics in real-time.
- **Merging Validation:** Interactive validation of weight merging and stitching.
- **Batch Replication:** One-click execution of all 123 experiments from the paper.
- **Embeddings:** 3D visualization of the representation manifolds.

## Structure

- `emergent_gui.py`: Main entry point.
- `src/`: Core logic (Agents, Training, Analysis).
- `configs/`: Experiment configuration files.
