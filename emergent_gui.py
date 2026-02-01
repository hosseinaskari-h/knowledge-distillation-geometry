#!/usr/bin/env python3
"""
PAPER REPLICATION GUI
Interactive visualization and control for Coupled Transformer Dynamics experiments.
Supports:
- Vector Dynamics Trajectory Analysis
- Model Merging & Stitching Validation
- Dimensionality & Geometric Analysis
"""

import sys
import os
import warnings
import logging

# Suppress repetitive warnings
warnings.filterwarnings('ignore', message='.*cumsum_cuda_kernel does not have a deterministic.*')
warnings.filterwarnings('ignore', message='.*Setting `pad_token_id` to `eos_token_id`.*')

# Suppress transformers logging spam
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
import threading
import queue
import webbrowser
import tempfile
import time
import io
import traceback

# Core GUI
import dearpygui.dearpygui as dpg

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# 3D visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Stats
from scipy import stats
from scipy.spatial.distance import cdist, cosine

# System monitoring
import psutil
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
try:
    from src.data.loader import ExperimentLoader
    from src.data.storage import ExperimentStorage
    from src.analysis.dimensional import DimensionalAnalyzer
    from src.analysis.information import InformationAnalyzer
    from src.analysis.geometric import GeometricAnalyzer
    from src.analysis.temporal import TemporalAnalyzer
    from src.analysis.statistical import StatisticalTests
    import matplotlib.pyplot as plt
    HAS_PROJECT = True
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    HAS_PROJECT = False

# Optional torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentInfo:
    """Metadata for experiment listing"""
    name: str
    path: Path
    episode_count: int = 0
    timestamp: str = ""
    config: Dict = field(default_factory=dict)
    has_embeddings: bool = False
    total_reward: float = 0.0
    avg_reward: float = 0.0


@dataclass
class AnalysisResult:
    """Results from analysis"""
    experiment_name: str
    dimensional: Dict = field(default_factory=dict)
    information: Dict = field(default_factory=dict)
    geometric: Dict = field(default_factory=dict)
    temporal: Dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class TrainingState:
    """Current training state"""
    is_running: bool = False
    current_episode: int = 0
    total_episodes: int = 0
    current_reward: float = 0.0
    avg_reward: float = 0.0
    rewards_history: List[float] = field(default_factory=list)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL LOG HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class GUILogHandler:
    """Captures log output for GUI display"""

    def __init__(self, max_lines: int = 1000):
        self.log_queue = queue.Queue()
        self.max_lines = max_lines
        self.lines = []

    def write(self, text: str):
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] {text.strip()}"
            self.lines.append(line)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
            self.log_queue.put(line)

    def flush(self):
        pass

    def get_all(self) -> str:
        return "\n".join(self.lines)

    def get_new(self) -> List[str]:
        new_lines = []
        while not self.log_queue.empty():
            try:
                new_lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return new_lines


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GUI CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class EmergentProtocolsGUI:
    """Main GUI application for emergent communication research"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.experiments_dir = self.base_dir / "data" / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.base_dir / "configs" / "experiments"

        # State
        self.experiments: List[ExperimentInfo] = []
        self.current_experiment: Optional[ExperimentInfo] = None
        self.current_analysis: Optional[AnalysisResult] = None
        self.training_state = TrainingState()
        self.comparison_experiments: List[ExperimentInfo] = []

        # Caches
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._projection_cache: Dict[str, np.ndarray] = {}
        self._loader_cache: Dict[str, ExperimentLoader] = {}

        # Threading
        self.training_thread: Optional[threading.Thread] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.stop_training = threading.Event()
        self.plot_queue = queue.Queue()

        # Logging
        self.log_handler = GUILogHandler()

        # Performance monitoring
        self._perf_last_update = 0
        self._perf_update_interval = 1.0  # Update every 1 second

        # Theme colors (scientific paper aesthetic)
        self.colors = {
            'bg': (252, 252, 255),
            'bg_dark': (245, 245, 250),
            'panel': (248, 248, 252),
            'text': (30, 30, 40),
            'text_dim': (100, 100, 120),
            'accent': (50, 100, 180),
            'accent_light': (100, 140, 200),
            'success': (60, 160, 80),
            'warning': (200, 150, 50),
            'error': (200, 60, 60),
            'highlight': (220, 100, 80),
            'plot_colors': [
                (31, 119, 180),   # blue
                (255, 127, 14),   # orange
                (44, 160, 44),    # green
                (214, 39, 40),    # red
                (148, 103, 189),  # purple
                (140, 86, 75),    # brown
                (227, 119, 194),  # pink
                (127, 127, 127),  # gray
            ]
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def scan_experiments(self) -> List[ExperimentInfo]:
        """Scan for experiments"""
        experiments = []

        if not self.experiments_dir.exists():
            return experiments

        for exp_dir in sorted(self.experiments_dir.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue

            try:
                info = ExperimentInfo(
                    name=exp_dir.name,
                    path=exp_dir,
                )

                # Check for config
                config_path = exp_dir / "config.yaml"
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path) as f:
                            info.config = yaml.safe_load(f)
                    except:
                        pass

                # Check for embeddings
                emb_path = exp_dir / "embeddings.h5"
                info.has_embeddings = emb_path.exists()

                # Check for metadata
                meta_path = exp_dir / "metadata.db"
                if meta_path.exists():
                    import sqlite3
                    try:
                        conn = sqlite3.connect(meta_path)
                        cursor = conn.execute("SELECT COUNT(*) FROM episodes")
                        info.episode_count = cursor.fetchone()[0]

                        cursor = conn.execute("SELECT AVG(total_reward) FROM episodes")
                        result = cursor.fetchone()[0]
                        info.avg_reward = result if result else 0.0

                        conn.close()
                    except:
                        pass

                # Get timestamp from directory name or modification time
                info.timestamp = datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

                experiments.append(info)

            except Exception as e:
                self.log(f"Error scanning {exp_dir}: {e}")

        self.experiments = experiments
        return experiments

    def load_experiment_data(self, exp: ExperimentInfo) -> Optional[ExperimentLoader]:
        """Load experiment data for analysis"""
        if exp.name in self._loader_cache:
            return self._loader_cache[exp.name]

        try:
            loader = ExperimentLoader(exp.path)
            self._loader_cache[exp.name] = loader
            return loader
        except Exception as e:
            self.log(f"Error loading experiment: {e}")
            return None

    def get_embeddings(self, exp: ExperimentInfo, max_samples: int = 10000) -> Optional[np.ndarray]:
        """Get embeddings for experiment"""
        cache_key = f"{exp.name}_{max_samples}"
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]

        loader = self.load_experiment_data(exp)
        if loader is None:
            return None

        try:
            embeddings = loader.get_all_embeddings()
            # Convert to float32 to avoid overflow issues in numpy linalg operations
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings = embeddings[indices]
            self._embeddings_cache[cache_key] = embeddings
            return embeddings
        except Exception as e:
            self.log(f"Error getting embeddings: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def run_full_analysis(self, exp: ExperimentInfo) -> Optional[AnalysisResult]:
        """Run complete analysis on experiment"""
        embeddings = self.get_embeddings(exp)
        if embeddings is None or len(embeddings) < 10:
            self.log("Not enough embeddings for analysis")
            return None

        result = AnalysisResult(
            experiment_name=exp.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        try:
            # Dimensional analysis
            self.log("Running dimensional analysis...")
            dim_analyzer = DimensionalAnalyzer(embeddings)
            result.dimensional = dim_analyzer.full_analysis()

            # Information analysis
            self.log("Running information analysis...")
            info_analyzer = InformationAnalyzer(embeddings)
            result.information = info_analyzer.full_analysis()

            # Geometric analysis
            self.log("Running geometric analysis...")
            geom_analyzer = GeometricAnalyzer(embeddings)
            result.geometric = geom_analyzer.full_analysis()

            # Temporal analysis
            self.log("Running temporal analysis...")
            temp_analyzer = TemporalAnalyzer(embeddings)
            result.temporal = temp_analyzer.full_analysis()

            self.log("Analysis complete!")
            return result

        except Exception as e:
            self.log(f"Analysis error: {e}")
            traceback.print_exc()
            return None

    def run_analysis_async(self, exp: ExperimentInfo, callback: Callable):
        """Run analysis in background thread"""
        def worker():
            result = self.run_full_analysis(exp)
            callback(result)

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════════

    def start_training(self, config_name: str):
        """Start training in background"""
        if self.training_state.is_running:
            self.log("Training already running!")
            return

        self.stop_training.clear()
        self.training_state = TrainingState(is_running=True)

        def training_worker():
            try:
                self.log(f"Starting training with config: {config_name}")

                # Import training components
                from src.agents.registry import create_agent
                from src.environments.registry import create_environment
                from src.training.rewards.registry import create_reward_function
                from src.training.rl_trainer import MARLTrainer
                from src.data.storage import ExperimentStorage
                from src.utils.reproducibility import set_seed
                from omegaconf import OmegaConf

                # Load config
                config_path = self.configs_dir / f"{config_name}.yaml"
                if not config_path.exists():
                    self.log(f"Config not found: {config_path}")
                    return

                cfg = OmegaConf.load(config_path)

                # Setup
                seed = cfg.experiment.get('seed', 42)
                set_seed(seed)

                experiment_name = cfg.experiment.name
                experiment_dir = self.experiments_dir / experiment_name
                experiment_dir.mkdir(parents=True, exist_ok=True)

                self.log(f"Experiment directory: {experiment_dir}")

                # Initialize components
                self.log("Loading agents...")
                agents = {}
                for name, agent_cfg in cfg.agents.items():
                    agents[name] = create_agent(name, dict(agent_cfg))
                    self.log(f"  Loaded: {name} ({agent_cfg.model})")

                self.log("Creating environment...")
                environment = create_environment(dict(cfg.environment))

                self.log("Creating reward function...")
                reward_cfg = dict(cfg.training.reward)
                first_agent = list(agents.values())[0]
                reward_cfg['device'] = first_agent.device
                reward_cfg['hidden_dim'] = first_agent.embedding_dim
                reward_fn = create_reward_function(reward_cfg)

                self.log("Initializing storage...")
                storage = ExperimentStorage(experiment_dir)

                self.log("Creating trainer...")
                trainer = MARLTrainer(agents, environment, reward_fn, dict(cfg.training))

                # Check for checkpoint resume
                start_episode = 0
                resume_enabled = dpg.get_value("custom_resume_enabled") if dpg.does_item_exist("custom_resume_enabled") else False

                if resume_enabled:
                    resume_exp = dpg.get_value("custom_resume_experiment")
                    resume_ckpt = dpg.get_value("custom_resume_checkpoint")

                    if resume_exp and resume_exp != "(none)" and resume_ckpt and resume_ckpt != "(none)":
                        checkpoint_dir = self.experiments_dir / resume_exp / "checkpoints"
                        trainer_ckpt_path = checkpoint_dir / f"{resume_ckpt}.pt"

                        if trainer_ckpt_path.exists():
                            self.log(f"Loading checkpoint: {resume_ckpt}")

                            # Load trainer state
                            trainer.load_checkpoint(trainer_ckpt_path)
                            start_episode = trainer.episode_counter
                            self.log(f"  Trainer state loaded (episode {start_episode})")

                            # Load agent weights
                            for agent_name, agent in agents.items():
                                # Try to find matching agent checkpoint
                                agent_ckpt_name = resume_ckpt.replace("checkpoint_", f"agent_{agent_name}_")
                                agent_ckpt_path = checkpoint_dir / f"{agent_ckpt_name}.pt"
                                if agent_ckpt_path.exists():
                                    agent.load_checkpoint(str(agent_ckpt_path))
                                    self.log(f"  Agent {agent_name} weights loaded")
                                else:
                                    self.log(f"  Warning: No checkpoint for agent {agent_name}")

                            self.log(f"Resuming from episode {start_episode}")
                        else:
                            self.log(f"Checkpoint not found: {trainer_ckpt_path}")
                    else:
                        self.log("Starting fresh (no valid checkpoint selected)")
                else:
                    self.log("Starting fresh (pretrained weights)")

                # Training loop - use GUI input, fallback to config
                gui_episodes = dpg.get_value("training_episodes")
                config_episodes = cfg.training.get('num_episodes', cfg.environment.get('num_episodes', 1000))
                num_episodes = gui_episodes if gui_episodes and gui_episodes > 0 else config_episodes
                self.training_state.total_episodes = num_episodes

                # Get save interval from config (default to 1 = save every episode)
                save_interval = cfg.training.get('save_interval', 1)

                if start_episode > 0:
                    self.log(f"Resuming training from episode {start_episode}: {num_episodes} more episodes")
                else:
                    self.log(f"Starting training: {num_episodes} episodes (saving every {save_interval})")

                for episode_idx in range(start_episode, start_episode + num_episodes):
                    if self.stop_training.is_set():
                        self.log("Training stopped by user")
                        break

                    # Train one episode
                    episode, metrics = trainer.train_episode()

                    # Update state
                    self.training_state.current_episode = episode_idx + 1 - start_episode
                    self.training_state.current_reward = metrics['total_reward']
                    self.training_state.rewards_history.append(metrics['total_reward'])

                    # Calculate running average
                    if len(self.training_state.rewards_history) >= 100:
                        self.training_state.avg_reward = np.mean(
                            self.training_state.rewards_history[-100:]
                        )

                    # Save based on interval
                    if save_interval == 1 or episode_idx % save_interval == 0:
                        storage.save_episode(episode, metrics)

                    # Log periodically
                    if episode_idx % 100 == 0:
                        self.log(
                            f"Episode {episode_idx}: reward={metrics['total_reward']:.3f}, "
                            f"avg={self.training_state.avg_reward:.3f}"
                        )

                    # Save checkpoint every 500 episodes
                    checkpoint_interval = cfg.training.get('checkpoint_interval', 500)
                    if checkpoint_interval > 0 and (episode_idx + 1) % checkpoint_interval == 0:
                        checkpoint_dir = experiment_dir / "checkpoints"
                        checkpoint_dir.mkdir(exist_ok=True)
                        checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode_idx + 1}.pt"

                        # Save trainer state
                        trainer.save_checkpoint(checkpoint_path)

                        # Save agent models
                        for agent_name, agent in agents.items():
                            agent_path = checkpoint_dir / f"agent_{agent_name}_ep{episode_idx + 1}.pt"
                            agent.save_checkpoint(str(agent_path))

                        self.log(f"Checkpoint saved at episode {episode_idx + 1}")

                # Save final checkpoint when done
                checkpoint_dir = experiment_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                final_ep = start_episode + self.training_state.current_episode
                checkpoint_path = checkpoint_dir / f"checkpoint_ep{final_ep}_final.pt"
                trainer.save_checkpoint(checkpoint_path)
                for agent_name, agent in agents.items():
                    agent_path = checkpoint_dir / f"agent_{agent_name}_ep{final_ep}_final.pt"
                    agent.save_checkpoint(str(agent_path))
                self.log(f"Final checkpoint saved at episode {final_ep}")

                self.log("Training complete!")

            except Exception as e:
                self.log(f"Training error: {e}")
                traceback.print_exc()

            finally:
                self.training_state.is_running = False

        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()

    def stop_training_request(self):
        """Request training stop"""
        self.stop_training.set()
        self.log("Stop requested...")

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION (Plotly)
    # ═══════════════════════════════════════════════════════════════════════════

    def create_embedding_3d_plot(self, exp: ExperimentInfo,
                                  method: str = "pca",
                                  color_by: str = "time",
                                  max_points: int = 5000) -> Optional[go.Figure]:
        """Create 3D embedding visualization"""
        embeddings = self.get_embeddings(exp, max_points)
        if embeddings is None or len(embeddings) < 10:
            return None

        cache_key = f"{exp.name}_{method}_{len(embeddings)}"

        if cache_key not in self._projection_cache:
            try:
                if method == "pca":
                    reducer = PCA(n_components=3)
                    coords = reducer.fit_transform(embeddings)
                elif method == "tsne":
                    perplexity = min(30, len(embeddings) - 1)
                    reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                elif method == "umap" and HAS_UMAP:
                    n_neighbors = min(15, len(embeddings) - 1)
                    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                else:
                    return None

                self._projection_cache[cache_key] = coords
            except Exception as e:
                self.log(f"Projection error: {e}")
                return None
        else:
            coords = self._projection_cache[cache_key]

        # Determine colors
        if color_by == "time":
            colors = np.arange(len(coords))
            colorscale = 'Viridis'
            colorbar_title = 'Time Step'
        elif color_by == "norm":
            colors = np.linalg.norm(embeddings, axis=1)
            colorscale = 'Plasma'
            colorbar_title = 'Embedding Norm'
        elif color_by == "density":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(10, len(coords))).fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            colors = 1 / (distances.mean(axis=1) + 0.01)
            colorscale = 'Hot'
            colorbar_title = 'Local Density'
        else:
            colors = np.arange(len(coords))
            colorscale = 'Viridis'
            colorbar_title = 'Index'

        # Ensure float64 for plotly JSON serialization
        coords = coords.astype(np.float64)
        colors = np.asarray(colors, dtype=np.float64)

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            z=coords[:, 2].tolist(),
            mode='markers',
            marker=dict(
                size=3,
                color=colors.tolist(),
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title),
                opacity=0.7
            ),
            hoverinfo='text',
            hovertext=[f"Index: {i}<br>Norm: {np.linalg.norm(embeddings[i]):.3f}"
                      for i in range(len(coords))]
        )])

        fig.update_layout(
            title=dict(
                text=f"Embedding Space ({method.upper()}) - {exp.name}",
                font=dict(size=16, family="Segoe UI, sans-serif")
            ),
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
                bgcolor='rgb(250, 250, 252)',
            ),
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=50, b=0),
            font=dict(family="Segoe UI, sans-serif")
        )

        return fig

    def create_metrics_dashboard(self, exp: ExperimentInfo) -> Optional[go.Figure]:
        """Create metrics dashboard with multiple subplots"""
        loader = self.load_experiment_data(exp)
        if loader is None:
            return None

        metrics = loader.get_metrics()
        if not metrics:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Reward', 'Average Reward', 'Episode Length', 'Reward Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )

        # Reward over time
        if 'total_reward' in metrics:
            rewards = metrics['total_reward']
            fig.add_trace(
                go.Scatter(y=rewards, mode='lines', name='Total Reward',
                          line=dict(color='rgb(50,100,180)', width=1)),
                row=1, col=1
            )
            # Moving average
            if len(rewards) > 100:
                ma = np.convolve(rewards, np.ones(100)/100, mode='valid')
                fig.add_trace(
                    go.Scatter(y=ma, mode='lines', name='MA(100)',
                              line=dict(color='rgb(200,60,60)', width=2)),
                    row=1, col=1
                )

        # Average reward
        if 'avg_reward' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['avg_reward'], mode='lines', name='Avg Reward',
                          line=dict(color='rgb(60,160,80)', width=1)),
                row=1, col=2
            )

        # Episode length
        if 'num_turns' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['num_turns'], mode='lines', name='Turns',
                          line=dict(color='rgb(148,103,189)', width=1)),
                row=2, col=1
            )

        # Reward distribution
        if 'total_reward' in metrics:
            fig.add_trace(
                go.Histogram(x=metrics['total_reward'], name='Reward Dist',
                            marker_color='rgb(50,100,180)'),
                row=2, col=2
            )

        fig.update_layout(
            title=f"Training Metrics - {exp.name}",
            height=700,
            showlegend=True,
            paper_bgcolor='white'
        )

        return fig

    def create_dimensional_analysis_plot(self, analysis: AnalysisResult) -> Optional[go.Figure]:
        """Create dimensional analysis visualization"""
        if not analysis.dimensional:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'PCA Explained Variance',
                'Intrinsic Dimension Estimates',
                'Participation Ratio',
                'Dimension Comparison'
            )
        )

        # PCA variance
        if 'pca' in analysis.dimensional:
            pca = analysis.dimensional['pca']
            if 'explained_variance_ratio' in analysis.dimensional:
                var_ratio = analysis.dimensional['explained_variance_ratio'][:20]
                cumvar = np.cumsum(var_ratio)
                fig.add_trace(
                    go.Bar(y=var_ratio, name='Variance Ratio'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=cumvar, mode='lines+markers', name='Cumulative',
                              line=dict(color='red')),
                    row=1, col=1
                )

        # Dimension estimates comparison
        dims = {
            'MLE': analysis.dimensional.get('mle_dimension', 0),
            'Correlation': analysis.dimensional.get('correlation_dimension', 0),
            'PCA (95%)': analysis.dimensional.get('pca', {}).get('effective_dim_95', 0),
        }
        fig.add_trace(
            go.Bar(x=list(dims.keys()), y=list(dims.values()),
                   name='Dimension Estimates',
                   marker_color='rgb(50,100,180)'),
            row=1, col=2
        )

        # Add thesis reference line
        fig.add_hline(y=6.0, line_dash="dash", line_color="red",
                     annotation_text="Thesis finding: 5.5-6.5",
                     row=1, col=2)

        fig.update_layout(
            title=f"Dimensional Analysis - {analysis.experiment_name}",
            height=600,
            showlegend=True,
            paper_bgcolor='white'
        )

        return fig

    def create_geometric_analysis_plot(self, analysis: AnalysisResult) -> Optional[go.Figure]:
        """Create geometric analysis visualization"""
        if not analysis.geometric:
            return None

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Ricci Curvature Distribution', 'Local Density', 'Clustering')
        )

        # Ricci curvature
        if 'ricci_curvature' in analysis.geometric:
            ricci = analysis.geometric['ricci_curvature']
            if 'curvatures' in ricci and len(ricci['curvatures']) > 0:
                fig.add_trace(
                    go.Histogram(x=ricci['curvatures'], name='Ricci Curvature',
                                marker_color='rgb(50,100,180)'),
                    row=1, col=1
                )
                # Add thesis reference
                fig.add_vline(x=0.514, line_dash="dash", line_color="red",
                             annotation_text="Thesis: 0.514", row=1, col=1)

        # Density stats
        if 'density' in analysis.geometric:
            density = analysis.geometric['density']
            fig.add_trace(
                go.Bar(
                    x=['Mean', 'Std', 'Min', 'Max'],
                    y=[density.get('mean', 0), density.get('std', 0),
                       density.get('min', 0), density.get('max', 0)],
                    name='Density Stats',
                    marker_color='rgb(60,160,80)'
                ),
                row=1, col=2
            )

        # Clustering silhouette
        if 'clustering' in analysis.geometric:
            clustering = analysis.geometric['clustering']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=clustering.get('silhouette', 0),
                    title={'text': "Silhouette Score"},
                    gauge={'axis': {'range': [-1, 1]},
                           'bar': {'color': "rgb(50,100,180)"},
                           'threshold': {'line': {'color': "red", 'width': 2},
                                        'value': 0.5}}
                ),
                row=1, col=3
            )

        fig.update_layout(
            title=f"Geometric Analysis - {analysis.experiment_name}",
            height=400,
            paper_bgcolor='white'
        )

        return fig

    def create_comparison_plot(self, experiments: List[ExperimentInfo]) -> Optional[go.Figure]:
        """Create comparison plot between experiments"""
        if len(experiments) < 2:
            return None

        # Collect analysis results
        results = []
        for exp in experiments:
            embeddings = self.get_embeddings(exp, max_samples=5000)
            if embeddings is not None and len(embeddings) > 10:
                dim_analyzer = DimensionalAnalyzer(embeddings)
                dim_result = dim_analyzer.full_analysis()
                geom_analyzer = GeometricAnalyzer(embeddings)
                geom_result = geom_analyzer.full_analysis()
                results.append({
                    'name': exp.name,
                    'mle_dim': dim_result.get('mle_dimension', 0),
                    'ricci': geom_result.get('ricci_curvature', {}).get('mean', 0),
                    'episodes': exp.episode_count,
                    'avg_reward': exp.avg_reward
                })

        if len(results) < 2:
            return None

        # Create comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Intrinsic Dimension',
                'Ricci Curvature',
                'Episode Count',
                'Average Reward'
            )
        )

        names = [r['name'][:20] for r in results]

        # Dimension comparison
        fig.add_trace(
            go.Bar(x=names, y=[r['mle_dim'] for r in results],
                   marker_color='rgb(50,100,180)'),
            row=1, col=1
        )
        fig.add_hline(y=6.0, line_dash="dash", line_color="red", row=1, col=1)

        # Ricci comparison
        fig.add_trace(
            go.Bar(x=names, y=[r['ricci'] for r in results],
                   marker_color='rgb(60,160,80)'),
            row=1, col=2
        )
        fig.add_hline(y=0.514, line_dash="dash", line_color="red", row=1, col=2)

        # Episodes
        fig.add_trace(
            go.Bar(x=names, y=[r['episodes'] for r in results],
                   marker_color='rgb(148,103,189)'),
            row=2, col=1
        )

        # Reward
        fig.add_trace(
            go.Bar(x=names, y=[r['avg_reward'] for r in results],
                   marker_color='rgb(200,60,60)'),
            row=2, col=2
        )

        fig.update_layout(
            title="Experiment Comparison",
            height=600,
            showlegend=False,
            paper_bgcolor='white'
        )

        return fig

    def show_plot_in_browser(self, fig: go.Figure):
        """Open plot in browser"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            fig.write_html(f.name)
            webbrowser.open(f'file://{f.name}')

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════════════

    def log(self, message: str):
        """Log message to GUI"""
        self.log_handler.write(message)
        print(message)

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAR PYGUI SETUP
    # ═══════════════════════════════════════════════════════════════════════════

    def setup_theme(self):
        """Create research paper aesthetic theme with readable text"""
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Window background
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (252, 252, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (248, 248, 252), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (252, 252, 255), category=dpg.mvThemeCat_Core)

                # Text - dark for readability on light backgrounds
                dpg.add_theme_color(dpg.mvThemeCol_Text, (30, 30, 40), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (140, 140, 150), category=dpg.mvThemeCat_Core)

                # Menu bar - light background for dark text readability
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (235, 238, 245), category=dpg.mvThemeCat_Core)

                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 100, 170), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 120, 190), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 90, 160), category=dpg.mvThemeCat_Core)

                # Tabs - light backgrounds for dark text readability
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (215, 220, 235), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (190, 200, 225), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (250, 250, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (225, 228, 238), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (245, 246, 252), category=dpg.mvThemeCat_Core)

                # Headers - light for readability
                dpg.add_theme_color(dpg.mvThemeCol_Header, (200, 215, 240), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (180, 200, 230), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (170, 190, 225), category=dpg.mvThemeCat_Core)

                # Frame
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (240, 240, 245), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (230, 230, 240), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (220, 225, 235), category=dpg.mvThemeCat_Core)

                # Title bar - lighter so text is readable
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (200, 210, 230), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (180, 195, 220), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (220, 225, 235), category=dpg.mvThemeCat_Core)

                # Plot
                dpg.add_theme_color(dpg.mvThemeCol_PlotLines, (60, 100, 170), category=dpg.mvThemeCat_Core)

                # Separator
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (180, 185, 200), category=dpg.mvThemeCat_Core)

                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (245, 245, 250), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (180, 190, 210), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (160, 175, 200), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (140, 160, 190), category=dpg.mvThemeCat_Core)

                # Rounding
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4, category=dpg.mvThemeCat_Core)

                # Padding
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8, category=dpg.mvThemeCat_Core)

        dpg.bind_theme(self.global_theme)

    def setup_windows(self):
        """Create all GUI windows"""
        with dpg.window(tag="primary_window"):
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Refresh Experiments", callback=self._refresh_experiments)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Export Analysis Report", callback=self._export_report)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

                with dpg.menu(label="View"):
                    dpg.add_menu_item(label="3D Embedding Plot", callback=self._show_3d_plot)
                    dpg.add_menu_item(label="Metrics Dashboard", callback=self._show_metrics_dashboard)
                    dpg.add_menu_item(label="Dimensional Analysis", callback=self._show_dimensional_plot)
                    dpg.add_menu_item(label="Geometric Analysis", callback=self._show_geometric_plot)

                with dpg.menu(label="Analysis"):
                    dpg.add_menu_item(label="Run Full Analysis", callback=self._run_analysis)
                    dpg.add_menu_item(label="Compare Experiments", callback=self._show_comparison)

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="About", callback=self._show_about)

            # Header with table layout for left/right alignment
            with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False,
                          borders_outerH=False, borders_outerV=False):
                dpg.add_table_column(width_stretch=True)  # Left side (expandable)
                dpg.add_table_column(width_fixed=True)    # Right side (fixed)

                with dpg.table_row():
                    # Left side - title and status
                    with dpg.group(horizontal=True):
                        dpg.add_text("PAPER REPLICATION: COUPLED DYNAMICS", color=(50, 100, 180))
                        dpg.add_spacer(width=30)
                        dpg.add_text("", tag="status_text", color=(100, 100, 120))
                        dpg.add_spacer(width=30)
                        dpg.add_text("", tag="training_status", color=(60, 160, 80))

                    # Right side - performance monitors
                    with dpg.group(horizontal=True):
                        dpg.add_text("CPU:", color=(100, 100, 120))
                        dpg.add_text("---%", tag="perf_cpu", color=(80, 140, 200))
                        dpg.add_spacer(width=15)
                        dpg.add_text("RAM:", color=(100, 100, 120))
                        dpg.add_text("----", tag="perf_ram", color=(80, 140, 200))
                        dpg.add_spacer(width=15)
                        dpg.add_text("GPU:", color=(100, 100, 120))
                        dpg.add_text("---", tag="perf_gpu", color=(80, 200, 120))
                        dpg.add_spacer(width=15)
                        dpg.add_text("VRAM:", color=(100, 100, 120))
                        dpg.add_text("----", tag="perf_vram", color=(80, 200, 120))
                        dpg.add_spacer(width=5)  # Right margin to align with content below

            dpg.add_separator()

            # Main content with sidebar
            with dpg.group(horizontal=True):

                # Left sidebar - Experiment browser
                with dpg.child_window(width=300, height=-150, tag="sidebar"):
                    self._setup_sidebar()

                # Main content area with tabs
                with dpg.child_window(width=-1, height=-150, tag="main_content"):
                    with dpg.tab_bar(tag="main_tabs"):
                        with dpg.tab(label="Vector Dynamics"):
                            self._setup_vector_dynamics_tab()
                        with dpg.tab(label="Merging Validation"):
                            self._setup_validation_tab()
                        with dpg.tab(label="Analysis Results"):
                            self._setup_analysis_tab()
                        with dpg.tab(label="Embeddings"):
                            self._setup_embeddings_tab()
                        with dpg.tab(label="Comparison"):
                            self._setup_comparison_tab()
                        with dpg.tab(label="Weight Analysis"):
                            self._setup_weight_analysis_tab()

            # Bottom panel - Terminal log
            dpg.add_separator()
            with dpg.child_window(height=-1, tag="terminal_panel"):
                self._setup_terminal()

    def _setup_sidebar(self):
        """Setup experiment browser sidebar"""
        dpg.add_text("EXPERIMENTS", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Refresh", callback=self._refresh_experiments, width=80)
            dpg.add_button(label="Open Folder", callback=self._open_experiments_folder, width=100)

        dpg.add_spacer(height=5)
        dpg.add_input_text(hint="Search...", width=-1, tag="exp_search", callback=self._filter_experiments)
        dpg.add_spacer(height=10)

        # Experiment list
        with dpg.child_window(height=-50, tag="exp_list_container"):
            dpg.add_text("Loading...", tag="exp_list_placeholder")

        # Quick actions
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load", callback=self._load_selected_experiment, width=70)
            dpg.add_button(label="Analyze", callback=self._run_analysis, width=70)
            dpg.add_button(label="Compare", callback=self._add_to_comparison, width=70)

    def _setup_dashboard_tab(self):
        """Setup dashboard overview tab"""
        dpg.add_text("DASHBOARD", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # Summary cards
            with dpg.child_window(width=250, height=120):
                dpg.add_text("Total Experiments", color=(100, 100, 120))
                dpg.add_text("0", tag="dash_total_exp", color=(50, 100, 180))
                dpg.add_spacer(height=10)
                dpg.add_text("Total Episodes", color=(100, 100, 120))
                dpg.add_text("0", tag="dash_total_episodes", color=(50, 100, 180))

            dpg.add_spacer(width=20)

            with dpg.child_window(width=250, height=120):
                dpg.add_text("Current Experiment", color=(100, 100, 120))
                dpg.add_text("None", tag="dash_current_exp", color=(50, 100, 180))
                dpg.add_spacer(height=10)
                dpg.add_text("Avg Reward", color=(100, 100, 120))
                dpg.add_text("N/A", tag="dash_avg_reward", color=(60, 160, 80))

            dpg.add_spacer(width=20)

            with dpg.child_window(width=250, height=120):
                dpg.add_text("Thesis Comparison", color=(100, 100, 120))
                dpg.add_text("Dimension: N/A (target: 5.5-6.5)", tag="dash_dimension", color=(100, 100, 120))
                dpg.add_text("Ricci: N/A (target: 0.514)", tag="dash_ricci", color=(100, 100, 120))
                dpg.add_text("Capacity: N/A (target: 2.71 bits)", tag="dash_capacity", color=(100, 100, 120))

        dpg.add_separator()
        dpg.add_text("Quick Actions", color=(50, 100, 180))

        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Baseline Replication", callback=lambda: self.start_training("baseline_replication"), width=200)
            dpg.add_spacer(width=10)
            dpg.add_button(label="Run Reward Ablation", callback=lambda: self.start_training("reward_ablation"), width=200)
            dpg.add_spacer(width=10)
            dpg.add_button(label="View 3D Embeddings", callback=self._show_3d_plot, width=200)

    def _setup_training_tab(self):
        """Setup training configuration and monitoring tab"""
        dpg.add_text("TRAINING", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # Config selection
            with dpg.child_window(width=350, height=-1):
                dpg.add_text("Configuration", color=(50, 100, 180))
                dpg.add_separator()

                dpg.add_text("Select Config:", color=(100, 100, 120))
                configs = self._get_available_configs()
                dpg.add_combo(configs, default_value=configs[0] if configs else "",
                             tag="training_config", width=-1)

                dpg.add_spacer(height=10)
                dpg.add_text("Episodes:", color=(100, 100, 120))
                dpg.add_input_int(default_value=1000, tag="training_episodes", width=-1)

                dpg.add_spacer(height=10)
                dpg.add_text("Seed:", color=(100, 100, 120))
                dpg.add_input_int(default_value=42, tag="training_seed", width=-1)

                dpg.add_spacer(height=20)
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start Training", callback=self._start_training_clicked,
                                  width=130, tag="btn_start_training")
                    dpg.add_button(label="Stop", callback=self.stop_training_request,
                                  width=80, tag="btn_stop_training")

                dpg.add_spacer(height=20)
                dpg.add_text("Status:", color=(100, 100, 120))
                dpg.add_text("Idle", tag="training_status_detail", wrap=300)

            # Training monitor
            with dpg.child_window(width=-1, height=-1):
                dpg.add_text("Training Monitor", color=(50, 100, 180))
                dpg.add_separator()

                # Progress
                dpg.add_text("Progress:", color=(100, 100, 120))
                dpg.add_progress_bar(default_value=0.0, tag="training_progress", width=-1)
                dpg.add_text("0 / 0 episodes", tag="training_progress_text", color=(100, 100, 120))

                dpg.add_spacer(height=10)

                # Live metrics
                with dpg.group(horizontal=True):
                    dpg.add_text("Current Reward:", color=(100, 100, 120))
                    dpg.add_text("0.000", tag="live_reward", color=(60, 160, 80))
                    dpg.add_spacer(width=30)
                    dpg.add_text("Avg Reward (100):", color=(100, 100, 120))
                    dpg.add_text("0.000", tag="live_avg_reward", color=(60, 160, 80))

                dpg.add_spacer(height=20)

                # Reward plot
                dpg.add_text("Reward History", color=(50, 100, 180))
                with dpg.plot(label="Rewards", height=300, width=-1, tag="training_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Episode", tag="train_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Reward", tag="train_y")
                    # Pre-create line series to avoid flickering
                    dpg.add_line_series([], [], label="Reward", parent="train_y", tag="reward_line_series")

    def _setup_analysis_tab(self):
        """Setup analysis results tab"""
        dpg.add_text("ANALYSIS RESULTS", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Full Analysis", callback=self._run_analysis, width=150)
            dpg.add_spacer(width=10)
            dpg.add_button(label="Export Report", callback=self._export_report, width=150)

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Dimensional results
            with dpg.child_window(width=350, height=250):
                dpg.add_text("Dimensional Analysis", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("", tag="analysis_dimensional", wrap=330, color=(60, 60, 70))

            dpg.add_spacer(width=10)

            # Information results
            with dpg.child_window(width=350, height=250):
                dpg.add_text("Information Analysis", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("", tag="analysis_information", wrap=330, color=(60, 60, 70))

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Geometric results
            with dpg.child_window(width=350, height=250):
                dpg.add_text("Geometric Analysis", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("", tag="analysis_geometric", wrap=330, color=(60, 60, 70))

            dpg.add_spacer(width=10)

            # Temporal results
            with dpg.child_window(width=350, height=250):
                dpg.add_text("Temporal Analysis", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("", tag="analysis_temporal", wrap=330, color=(60, 60, 70))

        dpg.add_spacer(height=10)

        # Thesis comparison
        with dpg.child_window(width=-1, height=-1):
            dpg.add_text("Comparison to Thesis Findings", color=(50, 100, 180))
            dpg.add_separator()
            dpg.add_text("", tag="thesis_comparison", wrap=-1, color=(60, 60, 70))

    def _setup_embeddings_tab(self):
        """Setup embeddings visualization tab"""
        dpg.add_text("EMBEDDINGS VISUALIZATION", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Reduction Method:", color=(100, 100, 120))
            dpg.add_radio_button(["PCA", "t-SNE", "UMAP"] if HAS_UMAP else ["PCA", "t-SNE"],
                                tag="emb_reduction", horizontal=True, default_value="PCA")
            dpg.add_spacer(width=30)
            dpg.add_text("Color By:", color=(100, 100, 120))
            dpg.add_radio_button(["Time", "Norm", "Density"], tag="emb_color",
                                horizontal=True, default_value="Time")
            dpg.add_spacer(width=30)
            dpg.add_text("Max Points:", color=(100, 100, 120))
            dpg.add_input_int(default_value=5000, tag="emb_max_points", width=100)

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Generate 3D Plot", callback=self._show_3d_plot, width=150)
            dpg.add_spacer(width=10)
            dpg.add_button(label="Export HTML", callback=self._export_3d_html, width=150)

        dpg.add_separator()
        dpg.add_text("Embedding Statistics", color=(50, 100, 180))
        dpg.add_text("", tag="emb_stats", wrap=-1, color=(60, 60, 70))

    def _setup_comparison_tab(self):
        """Setup experiment comparison tab"""
        dpg.add_text("EXPERIMENT COMPARISON", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # Selected experiments
            with dpg.child_window(width=300, height=-1):
                dpg.add_text("Selected for Comparison", color=(50, 100, 180))
                dpg.add_separator()
                with dpg.child_window(height=-50, tag="comparison_list"):
                    dpg.add_text("No experiments selected", tag="comparison_placeholder",
                                color=(140, 140, 150))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Clear All", callback=self._clear_comparison, width=100)
                    dpg.add_button(label="Compare", callback=self._show_comparison, width=100)

            # Comparison results
            with dpg.child_window(width=-1, height=-1):
                dpg.add_text("Comparison Results", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("", tag="comparison_results", wrap=-1, color=(60, 60, 70))

                dpg.add_spacer(height=10)
                dpg.add_button(label="Open Comparison Plot", callback=self._open_comparison_plot, width=200)

    def _setup_data_browser_tab(self):
        """Setup data browser tab"""
        dpg.add_text("DATA BROWSER", color=(50, 100, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Episode Range:", color=(100, 100, 120))
            dpg.add_input_int(default_value=0, tag="browse_start", width=80)
            dpg.add_text("to", color=(100, 100, 120))
            dpg.add_input_int(default_value=100, tag="browse_end", width=80)
            dpg.add_button(label="Load Episodes", callback=self._load_episodes_range, width=120)

        dpg.add_separator()

        with dpg.child_window(height=-1, tag="data_browser_content"):
            dpg.add_text("Select an experiment and load episodes to browse data",
                        tag="data_browser_placeholder", color=(140, 140, 150))

    def _setup_custom_run_tab(self):
        """Setup custom run configuration generator tab"""
        dpg.add_text("CUSTOM RUN GENERATOR", color=(50, 100, 180))
        dpg.add_separator()
        dpg.add_text("Create custom experiment configurations with your choice of models and parameters.",
                     wrap=700, color=(80, 80, 90))
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Left panel - Configuration
            with dpg.child_window(width=450, height=-50):
                dpg.add_text("Configuration", color=(50, 100, 180))
                dpg.add_separator()

                dpg.add_text("Config Name:", color=(100, 100, 120))
                dpg.add_input_text(default_value="my_custom_run", tag="custom_config_name", width=-1)
                dpg.add_spacer(height=10)

                dpg.add_text("Agent A Model:", color=(50, 100, 180))
                agent_a_models = self._get_available_models()
                dpg.add_combo(agent_a_models, default_value=agent_a_models[0] if agent_a_models else "gpt2",
                             tag="custom_agent_a_model", width=-1)
                dpg.add_spacer(height=5)

                dpg.add_text("Agent B Model:", color=(50, 100, 180))
                dpg.add_combo(agent_a_models, default_value=agent_a_models[1] if len(agent_a_models) > 1 else "distilgpt2",
                             tag="custom_agent_b_model", width=-1)
                dpg.add_spacer(height=15)

                dpg.add_text("Training Parameters", color=(50, 100, 180))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text("Episodes:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=10000, tag="custom_episodes", width=150)

                with dpg.group(horizontal=True):
                    dpg.add_text("Max Turns:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=20, tag="custom_max_turns", width=150)

                with dpg.group(horizontal=True):
                    dpg.add_text("Learning Rate:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.0001, tag="custom_lr", width=150, format="%.6f")

                with dpg.group(horizontal=True):
                    dpg.add_text("Seed:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=42, tag="custom_seed", width=150)

                with dpg.group(horizontal=True):
                    dpg.add_text("Save Interval:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=1, tag="custom_save_interval", width=150)
                    dpg.add_text("(1=every episode)", color=(80, 80, 90))

                with dpg.group(horizontal=True):
                    dpg.add_text("Checkpoint Interval:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=500, tag="custom_checkpoint_interval", width=150)
                    dpg.add_text("(0=disabled)", color=(80, 80, 90))

                dpg.add_spacer(height=15)
                dpg.add_text("Resume from Checkpoint", color=(50, 100, 180))
                dpg.add_separator()

                dpg.add_checkbox(label="Resume from existing checkpoint", tag="custom_resume_enabled",
                                callback=self._on_resume_toggle)
                dpg.add_text("Starting from: Pretrained (fresh weights)", tag="custom_starting_from",
                            color=(80, 200, 120))

                with dpg.group(tag="resume_options_group"):
                    dpg.add_text("Select Experiment:", color=(100, 100, 120))
                    experiments = self._get_experiments_with_checkpoints()
                    dpg.add_combo(experiments, default_value=experiments[0] if experiments else "(none)",
                                 tag="custom_resume_experiment", width=-1,
                                 callback=self._on_resume_experiment_changed)

                    dpg.add_text("Select Checkpoint:", color=(100, 100, 120))
                    dpg.add_combo(["(none)"], default_value="(none)",
                                 tag="custom_resume_checkpoint", width=-1)

                dpg.configure_item("resume_options_group", show=False)

                dpg.add_spacer(height=15)
                dpg.add_text("Reward Function", color=(50, 100, 180))
                dpg.add_separator()
                reward_types = ["layered", "pure_embedding"]
                dpg.add_combo(reward_types, default_value="layered", tag="custom_reward_type", width=-1)

                dpg.add_spacer(height=10)
                dpg.add_text("Reward Weights:", color=(100, 100, 120))

                with dpg.group(horizontal=True):
                    dpg.add_text("Info:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.4, tag="custom_weight_info", width=80, format="%.2f")
                    dpg.add_text("Geom:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.3, tag="custom_weight_geom", width=80, format="%.2f")
                    dpg.add_text("Dyn:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.3, tag="custom_weight_dyn", width=80, format="%.2f")

            dpg.add_spacer(width=20)

            # Right panel - Prompts and Preview
            with dpg.child_window(width=-1, height=-50):
                dpg.add_text("Conversation Prompts", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("Initial prompts (one per line):", color=(100, 100, 120))

                default_prompts = "The nature of consciousness is\nIn the beginning there was\nConsider the following proposition:\nWhat exists beyond the boundaries of\nThe relationship between mind and"

                dpg.add_input_text(multiline=True, default_value=default_prompts,
                                   tag="custom_prompts", width=-1, height=150)

                dpg.add_spacer(height=15)
                dpg.add_text("Generation Parameters", color=(50, 100, 180))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text("Max Tokens:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=50, tag="custom_max_tokens", width=100)
                    dpg.add_text("Temperature:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.8, tag="custom_temperature", width=100, format="%.2f")

                with dpg.group(horizontal=True):
                    dpg.add_text("Top-p:", color=(100, 100, 120))
                    dpg.add_input_float(default_value=0.9, tag="custom_top_p", width=100, format="%.2f")
                    dpg.add_text("Top-k:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=50, tag="custom_top_k", width=100)

                dpg.add_spacer(height=15)
                dpg.add_text("Config Preview", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_input_text(multiline=True, readonly=True, tag="custom_config_preview",
                                   width=-1, height=180)

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview Config", callback=self._preview_custom_config, width=150)
            dpg.add_button(label="Save Config", callback=self._save_custom_config, width=150)
            dpg.add_button(label="Load Existing", callback=self._load_custom_config, width=150)
            dpg.add_spacer(width=20)
            dpg.add_text("Saved:", color=(100, 100, 120))
            saved_configs = self._get_saved_custom_configs()
            dpg.add_combo(saved_configs, default_value=saved_configs[0] if saved_configs else "",
                         tag="saved_configs_list", width=200)
            dpg.add_button(label="Delete", callback=self._delete_custom_config, width=80)

    def _setup_vector_dynamics_tab(self):
        """Setup Vector Dynamics tab"""
        dpg.add_text("VECTOR DYNAMICS", color=(50, 100, 180))
        dpg.add_separator()
        dpg.add_text("Coupled Dynamics: Direct embedding exchange.",
                     wrap=700, color=(80, 80, 90))
        dpg.add_text("Investigate coupled neural manifolds in hidden state space.",
                     wrap=700, color=(100, 80, 120))
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Left panel - Configuration
            with dpg.child_window(width=400, height=-50):
                dpg.add_text("Configuration", color=(50, 100, 180))
                dpg.add_separator()

                dpg.add_text("Run Name:", color=(100, 100, 120))
                dpg.add_input_text(default_value="vector_dynamics_run", tag="vd_run_name", width=-1)
                dpg.add_spacer(height=10)

                dpg.add_text("Agent A Model:", color=(50, 100, 180))
                agent_models = self._get_available_models()
                dpg.add_combo(agent_models, default_value=agent_models[0] if agent_models else "gpt2",
                             tag="vd_agent_a_model", width=-1)

                dpg.add_text("Agent B Model:", color=(50, 100, 180))
                dpg.add_combo(agent_models, default_value=agent_models[0] if agent_models else "gpt2",
                             tag="vd_agent_b_model", width=-1)
                dpg.add_spacer(height=10)

                dpg.add_text("Device:", color=(50, 100, 180))
                # Check what's available
                cuda_available = HAS_TORCH and torch.cuda.is_available()
                device_options = ["cuda (GPU - faster)", "cpu (RAM - slower)"] if cuda_available else ["cpu (RAM)"]
                default_device = device_options[0]
                dpg.add_combo(device_options, default_value=default_device, tag="vd_device", width=-1)
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    dpg.add_text(f"GPU: {gpu_name}", color=(80, 160, 80))
                else:
                    dpg.add_text("CUDA not available", color=(200, 100, 100))
                dpg.add_spacer(height=15)

                dpg.add_text("Dynamics Parameters", color=(50, 100, 180))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text("Steps:", color=(100, 100, 120))
                    dpg.add_input_int(default_value=1000, tag="vd_num_steps", width=150)

                dpg.add_spacer(height=10)
                dpg.add_text("Coupling Strategy:", color=(100, 100, 120))
                coupling_strategies = ["direct", "blended", "decay"]
                dpg.add_combo(coupling_strategies, default_value="direct",
                             tag="vd_coupling_strategy", width=-1,
                             callback=self._on_vd_coupling_changed)

                # Coupling-specific parameters
                with dpg.group(tag="vd_blended_params", show=False):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Blend Alpha:", color=(100, 100, 120))
                        dpg.add_slider_float(default_value=0.5, min_value=0.0, max_value=1.0,
                                            tag="vd_blend_alpha", width=150, format="%.2f")
                    dpg.add_text("(0=pure exchange, 1=isolated)", color=(80, 80, 90))

                with dpg.group(tag="vd_decay_params", show=False):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Decay Factor:", color=(100, 100, 120))
                        dpg.add_slider_float(default_value=0.1, min_value=0.0, max_value=0.5,
                                            tag="vd_decay_factor", width=150, format="%.2f")
                    dpg.add_text("(memory of own previous state)", color=(80, 80, 90))

                dpg.add_spacer(height=10)
                dpg.add_text("Initialization:", color=(100, 100, 120))
                init_modes = ["random", "same", "text", "zero"]
                dpg.add_combo(init_modes, default_value="random", tag="vd_init_mode", width=-1,
                             callback=self._on_vd_init_mode_changed)

                with dpg.group(tag="vd_text_init_params", show=False):
                    dpg.add_text("Init Text A:", color=(100, 100, 120))
                    dpg.add_input_text(default_value="The nature of consciousness", tag="vd_init_text_a", width=-1)
                    dpg.add_text("Init Text B:", color=(100, 100, 120))
                    dpg.add_input_text(default_value="In the beginning", tag="vd_init_text_b", width=-1)

                dpg.add_spacer(height=15)
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Run Dynamics", callback=self._run_vector_dynamics,
                                  width=150, height=40, tag="btn_run_vd")
                    dpg.add_button(label="Stop", callback=self._stop_vector_dynamics,
                                  width=80, height=40, tag="btn_stop_vd")

                dpg.add_spacer(height=10)
                dpg.add_text("Status:", color=(100, 100, 120))
                dpg.add_text("Ready", tag="vd_status", color=(80, 200, 120))

                dpg.add_spacer(height=5)
                dpg.add_progress_bar(default_value=0.0, tag="vd_progress", width=-1)
                dpg.add_text("0 / 0 steps", tag="vd_progress_text", color=(100, 100, 120))

            dpg.add_spacer(width=20)

            # Right panel - Results and Visualization
            with dpg.child_window(width=-1, height=-50):
                dpg.add_text("Dynamics Analysis", color=(50, 100, 180))
                dpg.add_separator()

                # Quick metrics
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=200, height=100):
                        dpg.add_text("Synchronization", color=(100, 100, 120))
                        dpg.add_text("--", tag="vd_sync_index", color=(60, 160, 80))
                        dpg.add_text("", tag="vd_sync_type", color=(80, 80, 90))

                    with dpg.child_window(width=200, height=100):
                        dpg.add_text("Lyapunov Exp.", color=(100, 100, 120))
                        dpg.add_text("--", tag="vd_lyapunov", color=(60, 160, 80))
                        dpg.add_text("", tag="vd_dynamics_type", color=(80, 80, 90))

                    with dpg.child_window(width=200, height=100):
                        dpg.add_text("Attractor Dim.", color=(100, 100, 120))
                        dpg.add_text("--", tag="vd_attractor_dim", color=(60, 160, 80))

                    with dpg.child_window(width=200, height=100):
                        dpg.add_text("Final Cosine Sim", color=(100, 100, 120))
                        dpg.add_text("--", tag="vd_final_cosine", color=(60, 160, 80))

                dpg.add_spacer(height=10)

                # Live plots
                with dpg.group(horizontal=True):
                    # Cosine similarity over time
                    with dpg.plot(label="Cosine Similarity", height=200, width=400, tag="vd_cosine_plot"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Step", tag="vd_cos_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Cosine Sim", tag="vd_cos_y")
                        dpg.set_axis_limits("vd_cos_y", -1.0, 1.0)
                        dpg.add_line_series([], [], label="Similarity", parent="vd_cos_y", tag="vd_cos_line")

                    # Norms over time
                    with dpg.plot(label="State Norms", height=200, width=400, tag="vd_norm_plot"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Step", tag="vd_norm_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Norm", tag="vd_norm_y")
                        dpg.add_line_series([], [], label="Agent A", parent="vd_norm_y", tag="vd_norm_a_line")
                        dpg.add_line_series([], [], label="Agent B", parent="vd_norm_y", tag="vd_norm_b_line")

                dpg.add_spacer(height=10)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="View 3D Trajectory", callback=self._show_vd_3d_trajectory, width=150)
                    dpg.add_button(label="Export Results", callback=self._export_vd_results, width=150)
                    dpg.add_button(label="Full Analysis", callback=self._run_vd_full_analysis, width=150)

                dpg.add_spacer(height=10)
                dpg.add_text("Interpretation", color=(50, 100, 180))
                dpg.add_separator()
                dpg.add_text("Run a dynamics experiment to see results...", tag="vd_interpretation",
                            wrap=500, color=(80, 80, 90))

        # Batch Experiments Section
        dpg.add_spacer(height=20)
        dpg.add_separator()

        with dpg.collapsing_header(label="Batch Experiments", default_open=False):
            dpg.add_text("Automated execution of experimental phases defined in the paper.",
                        wrap=700, color=(80, 80, 90))
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                # Left - Experiment Selection
                with dpg.child_window(width=500, height=350):
                    dpg.add_text("Experiment Sets", color=(50, 100, 180))
                    dpg.add_separator()

                    # Predefined experiment sets using only cached models
                    experiment_sets = [
                        "Full Replication Suite",
                        "Phase 1.1: GPT-2 + GPT-2 (Same Model)",
                        "Phase 1.2: DistilGPT-2 + DistilGPT-2",
                        "Phase 2: Same Arch Different Scale",
                        "Phase 3: GPT-2 + DistilGPT-2 (Same Dim)",
                        "Phase 4: Cross-Scale (GPT-2 + Neo-1.3B)",
                        "Phase 5: Coupling Strength Sweep",
                        "Custom Selection",
                    ]
                    dpg.add_combo(experiment_sets, default_value=experiment_sets[0],
                                 tag="batch_experiment_set", width=-1,
                                 callback=self._on_batch_set_changed)

                    dpg.add_spacer(height=10)
                    dpg.add_text("Experiments to run:", color=(100, 100, 120))

                    # Table showing experiments
                    with dpg.child_window(height=180, border=True):
                        dpg.add_text("", tag="batch_experiment_list", wrap=450)

                    dpg.add_spacer(height=10)
                    dpg.add_text("Custom Models (for Custom Selection):", color=(100, 100, 120))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Model A:", color=(80, 80, 90))
                        cached_models = self._get_cached_models()
                        dpg.add_combo(cached_models, default_value=cached_models[0] if cached_models else "gpt2",
                                     tag="batch_custom_model_a", width=180)
                        dpg.add_text("Model B:", color=(80, 80, 90))
                        dpg.add_combo(cached_models, default_value=cached_models[1] if len(cached_models) > 1 else "gpt2",
                                     tag="batch_custom_model_b", width=180)

                dpg.add_spacer(width=20)

                # Right - Run Controls & Results
                with dpg.child_window(width=-1, height=350):
                    dpg.add_text("Batch Run Settings", color=(50, 100, 180))
                    dpg.add_separator()

                    with dpg.group(horizontal=True):
                        dpg.add_text("Steps per experiment:", color=(100, 100, 120))
                        dpg.add_input_int(default_value=1000, tag="batch_steps", width=100)

                    with dpg.group(horizontal=True):
                        dpg.add_text("Seeds per config:", color=(100, 100, 120))
                        dpg.add_input_int(default_value=1, tag="batch_seeds", width=100, min_value=1, max_value=5)

                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Run Batch", callback=self._run_batch_experiments,
                                      width=120, height=35, tag="btn_run_batch")
                        dpg.add_button(label="Stop Batch", callback=self._stop_batch_experiments,
                                      width=100, height=35)

                    dpg.add_spacer(height=10)
                    dpg.add_text("Batch Status:", color=(100, 100, 120))
                    dpg.add_text("Ready", tag="batch_status", color=(80, 200, 120))
                    dpg.add_progress_bar(default_value=0.0, tag="batch_progress", width=-1)
                    dpg.add_text("0 / 0 experiments", tag="batch_progress_text", color=(100, 100, 120))

                    dpg.add_spacer(height=10)
                    dpg.add_text("Results Summary", color=(50, 100, 180))
                    dpg.add_separator()

                    with dpg.child_window(height=120, border=True):
                        dpg.add_text("", tag="batch_results_summary", wrap=350)

                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Export All Results", callback=self._export_batch_results, width=130)
                        dpg.add_button(label="Open Results Folder", callback=self._open_batch_results_folder, width=130)

        # Initialize batch experiment list
        self._update_batch_experiment_list()

    # Vector Dynamics callbacks
    def _on_vd_coupling_changed(self, sender, app_data):
        """Show/hide coupling-specific parameters"""
        dpg.configure_item("vd_blended_params", show=(app_data == "blended"))
        dpg.configure_item("vd_decay_params", show=(app_data == "decay"))

    def _on_vd_init_mode_changed(self, sender, app_data):
        """Show/hide text init parameters"""
        dpg.configure_item("vd_text_init_params", show=(app_data == "text"))

    def _run_vector_dynamics(self):
        """Run vector dynamics experiment in background thread"""
        if hasattr(self, '_vd_running') and self._vd_running:
            self.log("Vector dynamics already running")
            return

        def run_dynamics():
            try:
                self._vd_running = True
                self._vd_stop_requested = False
                dpg.set_value("vd_status", "Loading models...")
                dpg.configure_item("vd_status", color=(200, 200, 80))

                # Import here to avoid loading at startup
                from src.agents.hf_agent import HuggingFaceAgent
                from src.training.vector_dynamics import (
                    VectorDynamicsTrainer,
                    create_coupling_strategy,
                    analyze_trajectory,
                )

                # Get config from GUI
                model_a = dpg.get_value("vd_agent_a_model")
                model_b = dpg.get_value("vd_agent_b_model")
                num_steps = dpg.get_value("vd_num_steps")
                coupling_name = dpg.get_value("vd_coupling_strategy")
                init_mode = dpg.get_value("vd_init_mode")

                # Get device selection
                device_selection = dpg.get_value("vd_device")
                device = 'cuda' if 'cuda' in device_selection.lower() else 'cpu'

                self.log(f"Initializing agents: {model_a}, {model_b} on {device.upper()}")

                # Create agents
                agent_a = HuggingFaceAgent("agent_a", {
                    'model': model_a,
                    'device': device,
                    'generation': {'max_new_tokens': 20},
                })
                agent_b = HuggingFaceAgent("agent_b", {
                    'model': model_b,
                    'device': device,
                    'generation': {'max_new_tokens': 20},
                })

                # Check for dimension mismatch
                if agent_a.embedding_dim != agent_b.embedding_dim:
                    self.log(f"NOTE: Dimension mismatch ({agent_a.embedding_dim} vs {agent_b.embedding_dim}). Using projection layers.")

                # Create coupling strategy
                coupling_config = {'hidden_dim': agent_a.embedding_dim, 'device': device}
                if coupling_name == 'blended':
                    coupling_config['alpha'] = dpg.get_value("vd_blend_alpha")
                elif coupling_name == 'decay':
                    coupling_config['decay'] = dpg.get_value("vd_decay_factor")

                coupling = create_coupling_strategy(coupling_name, coupling_config)

                # Create trainer
                trainer = VectorDynamicsTrainer(agent_a, agent_b, coupling, {
                    'record_every': max(1, num_steps // 500),  # ~500 data points
                    'compute_metrics_every': max(1, num_steps // 100),
                })

                dpg.set_value("vd_status", "Running dynamics...")

                # Lists for live plotting
                self._vd_steps = []
                self._vd_cosines = []
                self._vd_norms_a = []
                self._vd_norms_b = []

                def update_callback(step, state):
                    if self._vd_stop_requested:
                        return
                    # Update progress
                    progress = step / num_steps
                    dpg.set_value("vd_progress", progress)
                    dpg.set_value("vd_progress_text", f"{step} / {num_steps} steps")

                    # Record for plotting
                    if state.cosine_similarity is not None:
                        self._vd_steps.append(step)
                        self._vd_cosines.append(state.cosine_similarity)
                        self._vd_norms_a.append(state.state_a_norm or 0)
                        self._vd_norms_b.append(state.state_b_norm or 0)

                        # Update plots (limit frequency)
                        if len(self._vd_steps) % 5 == 0:
                            dpg.set_value("vd_cos_line", [self._vd_steps, self._vd_cosines])
                            dpg.set_value("vd_norm_a_line", [self._vd_steps, self._vd_norms_a])
                            dpg.set_value("vd_norm_b_line", [self._vd_steps, self._vd_norms_b])
                            dpg.fit_axis_data("vd_cos_x")
                            dpg.fit_axis_data("vd_norm_x")
                            dpg.fit_axis_data("vd_norm_y")

                # Get init text if needed
                init_text_a = dpg.get_value("vd_init_text_a") if init_mode == "text" else None
                init_text_b = dpg.get_value("vd_init_text_b") if init_mode == "text" else None

                # Run!
                self.log(f"Starting vector dynamics: {num_steps} steps, coupling={coupling_name}, init={init_mode}")
                trajectory = trainer.run(
                    num_steps=num_steps,
                    init_mode=init_mode,
                    init_text_a=init_text_a,
                    init_text_b=init_text_b,
                    callback=update_callback,
                )

                if self._vd_stop_requested:
                    dpg.set_value("vd_status", "Stopped")
                    dpg.configure_item("vd_status", color=(200, 80, 80))
                    self.log("Vector dynamics stopped by user")
                    return

                # Analyze results
                dpg.set_value("vd_status", "Analyzing...")
                self.log("Analyzing trajectory...")
                results = analyze_trajectory(trajectory)

                # Store for later use
                self._vd_trajectory = trajectory
                self._vd_results = results

                # Update GUI with results
                dpg.set_value("vd_sync_index", f"{results.get('synchronization_index', 0):.4f}")
                dpg.set_value("vd_sync_type", results.get('synchronization_type', ''))
                dpg.set_value("vd_lyapunov", f"{results.get('lyapunov_exponent', 0):.4f}")
                dpg.set_value("vd_dynamics_type", results.get('dynamics_type', ''))
                dpg.set_value("vd_attractor_dim", f"{results.get('attractor_dimension', 0):.2f}")
                dpg.set_value("vd_final_cosine", f"{results.get('final_cosine_similarity', 0):.4f}")

                # Generate interpretation
                interpretation = self._generate_vd_interpretation(results)
                dpg.set_value("vd_interpretation", interpretation)

                dpg.set_value("vd_progress", 1.0)
                dpg.set_value("vd_progress_text", f"{num_steps} / {num_steps} steps (complete)")
                dpg.set_value("vd_status", "Complete")
                dpg.configure_item("vd_status", color=(80, 200, 120))
                self.log("Vector dynamics run complete")

            except Exception as e:
                self.log(f"Error in vector dynamics: {e}")
                import traceback
                traceback.print_exc()
                dpg.set_value("vd_status", f"Error: {str(e)[:50]}")
                dpg.configure_item("vd_status", color=(200, 80, 80))
            finally:
                self._vd_running = False

        self._vd_thread = threading.Thread(target=run_dynamics, daemon=True)
        self._vd_thread.start()

    def _stop_vector_dynamics(self):
        """Stop running vector dynamics"""
        self._vd_stop_requested = True
        self.log("Stopping vector dynamics...")

    def _generate_vd_interpretation(self, results: dict) -> str:
        """Generate human-readable interpretation of dynamics results"""
        lines = []

        # Synchronization interpretation
        sync = results.get('synchronization_index', 0)
        sync_type = results.get('synchronization_type', 'none')
        if sync_type == 'strong':
            lines.append("SYNCHRONIZATION: Strong coupling detected. The agents are moving in lockstep through vector space.")
        elif sync_type == 'moderate':
            lines.append("SYNCHRONIZATION: Moderate coupling. Agents show correlated but not identical dynamics.")
        elif sync_type == 'weak':
            lines.append("SYNCHRONIZATION: Weak coupling. Some correlation exists but agents maintain independence.")
        else:
            lines.append("SYNCHRONIZATION: No significant coupling detected. Agents evolve independently.")

        # Lyapunov interpretation
        lyap = results.get('lyapunov_exponent', 0)
        dyn_type = results.get('dynamics_type', '')
        if dyn_type == 'chaotic':
            lines.append(f"\nDYNAMICS: Chaotic (Lyapunov={lyap:.3f}). Nearby trajectories diverge exponentially.")
        elif dyn_type == 'converging':
            lines.append(f"\nDYNAMICS: Converging (Lyapunov={lyap:.3f}). System is settling toward an attractor.")
        else:
            lines.append(f"\nDYNAMICS: Periodic/quasi-periodic (Lyapunov={lyap:.3f}). Structured but not chaotic.")

        # Attractor interpretation
        attr_dim = results.get('attractor_dimension', 0)
        if attr_dim > 0:
            lines.append(f"\nATTRACTOR: Estimated dimension {attr_dim:.2f}.")
            if attr_dim < 3:
                lines.append("Low-dimensional attractor suggests simple structured dynamics.")
            elif attr_dim < 10:
                lines.append("Moderate-dimensional attractor - rich but bounded complexity.")
            else:
                lines.append("High-dimensional attractor - complex dynamics in the coupled system.")

        # Cosine similarity interpretation
        final_cos = results.get('final_cosine_similarity', 0)
        initial_cos = results.get('mean_cosine_similarity', 0)
        if final_cos > 0.9:
            lines.append(f"\nCONVERGENCE: Agents converged to nearly identical states (cos={final_cos:.3f}).")
        elif final_cos < -0.5:
            lines.append(f"\nDIVERGENCE: Agents evolved to opposite orientations (cos={final_cos:.3f}).")
        else:
            lines.append(f"\nFINAL STATE: Agents maintain distinct but related states (cos={final_cos:.3f}).")

        return "\n".join(lines)

    def _show_vd_3d_trajectory(self):
        """Show 3D visualization of the trajectory"""
        if not hasattr(self, '_vd_trajectory') or self._vd_trajectory is None:
            self.log("No trajectory data available. Run dynamics first.")
            return

        try:
            from sklearn.decomposition import PCA

            # Get joint trajectory
            joint = self._vd_trajectory.get_joint_trajectory().numpy()

            # PCA to 3D
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(joint)

            # Create 3D plot
            fig = go.Figure()

            # Color by time
            colors = np.linspace(0, 1, len(coords_3d))

            fig.add_trace(go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='lines+markers',
                marker=dict(
                    size=2,
                    color=colors,
                    colorscale='Viridis',
                    colorbar=dict(title='Time'),
                ),
                line=dict(width=1, color='rgba(100,100,100,0.5)'),
                name='Trajectory',
            ))

            # Mark start and end
            fig.add_trace(go.Scatter3d(
                x=[coords_3d[0, 0]],
                y=[coords_3d[0, 1]],
                z=[coords_3d[0, 2]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Start',
            ))
            fig.add_trace(go.Scatter3d(
                x=[coords_3d[-1, 0]],
                y=[coords_3d[-1, 1]],
                z=[coords_3d[-1, 2]],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='End',
            ))

            fig.update_layout(
                title=f'Vector Dynamics Trajectory (PCA 3D)<br>Explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                ),
                width=900,
                height=700,
            )

            # Save and open
            import tempfile
            import webbrowser
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                fig.write_html(f.name)
                webbrowser.open(f'file://{f.name}')

            self.log("Opened 3D trajectory visualization in browser")

        except Exception as e:
            self.log(f"Error creating 3D visualization: {e}")

    def _export_vd_results(self):
        """Export vector dynamics results to JSON"""
        if not hasattr(self, '_vd_results') or self._vd_results is None:
            self.log("No results to export. Run dynamics first.")
            return

        try:
            run_name = dpg.get_value("vd_run_name")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.base_dir / "data" / "experiments" / f"{run_name}_{timestamp}_results.json"

            # Convert numpy types to Python types
            results_clean = {}
            for k, v in self._vd_results.items():
                if isinstance(v, (np.floating, np.integer)):
                    results_clean[k] = float(v)
                else:
                    results_clean[k] = v

            results_clean['config'] = {
                'model_a': dpg.get_value("vd_agent_a_model"),
                'model_b': dpg.get_value("vd_agent_b_model"),
                'num_steps': dpg.get_value("vd_num_steps"),
                'coupling': dpg.get_value("vd_coupling_strategy"),
                'init_mode': dpg.get_value("vd_init_mode"),
            }

            with open(filename, 'w') as f:
                json.dump(results_clean, f, indent=2)

            self.log(f"Exported results to {filename}")

        except Exception as e:
            self.log(f"Error exporting results: {e}")

    def _run_vd_full_analysis(self):
        """Run comprehensive analysis on the trajectory"""
        if not hasattr(self, '_vd_trajectory') or self._vd_trajectory is None:
            self.log("No trajectory data. Run dynamics first.")
            return

        self.log("Running full trajectory analysis...")

        def run_analysis():
            try:
                from src.training.vector_dynamics import (
                    compute_synchronization_index,
                    compute_lyapunov_exponent,
                    compute_attractor_dimension,
                )
                from scipy import stats
                from scipy.fft import fft, fftfreq
                from sklearn.decomposition import PCA

                trajectory = self._vd_trajectory
                states_a, states_b = trajectory.to_tensors()
                joint = trajectory.get_joint_trajectory().numpy()

                # Handle dimension mismatch for analysis
                dim_a, dim_b = states_a.shape[1], states_b.shape[1]
                dim_mismatch = dim_a != dim_b
                if dim_mismatch:
                    min_dim = min(dim_a, dim_b)
                    states_a_common = states_a[:, :min_dim]
                    states_b_common = states_b[:, :min_dim]
                    self.log(f"Dimension mismatch ({dim_a} vs {dim_b}), using first {min_dim} dims for comparison")
                else:
                    states_a_common = states_a
                    states_b_common = states_b

                results = {}

                # === 1. Basic trajectory statistics ===
                self.log("Computing trajectory statistics...")
                results['num_steps'] = len(states_a)
                results['embedding_dim_a'] = dim_a
                results['embedding_dim_b'] = dim_b

                # Norms over time
                norms_a = np.linalg.norm(states_a.numpy(), axis=1)
                norms_b = np.linalg.norm(states_b.numpy(), axis=1)
                results['norm_a_mean'] = float(np.mean(norms_a))
                results['norm_a_std'] = float(np.std(norms_a))
                results['norm_b_mean'] = float(np.mean(norms_b))
                results['norm_b_std'] = float(np.std(norms_b))

                # === 2. Synchronization analysis ===
                self.log("Analyzing synchronization...")
                results['synchronization_index'] = compute_synchronization_index(trajectory)

                # Cosine similarity over time (using common dimensions)
                cos_sims = []
                for i in range(len(states_a_common)):
                    a, b = states_a_common[i].numpy(), states_b_common[i].numpy()
                    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    cos_sims.append(cos)
                results['cosine_mean'] = float(np.mean(cos_sims))
                results['cosine_std'] = float(np.std(cos_sims))
                results['cosine_trend'] = float(cos_sims[-1] - cos_sims[0])

                # === 3. Dynamical systems analysis ===
                self.log("Computing Lyapunov exponent...")
                results['lyapunov_exponent'] = compute_lyapunov_exponent(trajectory)

                self.log("Estimating attractor dimension...")
                results['attractor_dimension'] = compute_attractor_dimension(trajectory)

                # === 4. Velocity analysis ===
                self.log("Analyzing velocity dynamics...")
                vel_a = np.diff(states_a.numpy(), axis=0)
                vel_b = np.diff(states_b.numpy(), axis=0)
                vel_norms_a = np.linalg.norm(vel_a, axis=1)
                vel_norms_b = np.linalg.norm(vel_b, axis=1)

                results['velocity_a_mean'] = float(np.mean(vel_norms_a))
                results['velocity_b_mean'] = float(np.mean(vel_norms_b))

                # Acceleration
                acc_a = np.diff(vel_a, axis=0)
                acc_b = np.diff(vel_b, axis=0)
                results['acceleration_a_mean'] = float(np.mean(np.linalg.norm(acc_a, axis=1)))
                results['acceleration_b_mean'] = float(np.mean(np.linalg.norm(acc_b, axis=1)))

                # === 5. Spectral analysis ===
                self.log("Running spectral analysis...")
                # Project to first few PCs for spectral analysis
                pca = PCA(n_components=min(10, joint.shape[1]))
                joint_pca = pca.fit_transform(joint)

                # FFT on first PC
                n = len(joint_pca)
                yf = fft(joint_pca[:, 0])
                xf = fftfreq(n, 1)[:n//2]
                power = 2.0/n * np.abs(yf[0:n//2])

                # Find dominant frequency
                dominant_freq_idx = np.argmax(power[1:]) + 1  # Skip DC component
                results['dominant_frequency'] = float(xf[dominant_freq_idx])
                results['spectral_entropy'] = float(stats.entropy(power + 1e-10))

                # === 6. PCA variance explained ===
                results['pca_variance_explained_3d'] = float(sum(pca.explained_variance_ratio_[:3]))
                results['pca_variance_explained_10d'] = float(sum(pca.explained_variance_ratio_))

                # === 7. Stationarity test ===
                self.log("Testing stationarity...")
                # Split trajectory in half and compare distributions
                half = len(joint_pca) // 2
                first_half = joint_pca[:half, 0]
                second_half = joint_pca[half:, 0]
                ks_stat, ks_pval = stats.ks_2samp(first_half, second_half)
                results['stationarity_ks_stat'] = float(ks_stat)
                results['stationarity_ks_pval'] = float(ks_pval)
                results['is_stationary'] = ks_pval > 0.05

                # === 8. Recurrence analysis (simplified) ===
                self.log("Computing recurrence metrics...")
                # Sample for efficiency
                sample_size = min(200, len(joint))
                indices = np.linspace(0, len(joint)-1, sample_size, dtype=int)
                sampled = joint[indices]

                # Pairwise distances
                from sklearn.metrics import pairwise_distances
                dists = pairwise_distances(sampled)
                threshold = np.percentile(dists, 10)  # 10% recurrence rate
                recurrence_matrix = dists < threshold
                results['recurrence_rate'] = float(np.mean(recurrence_matrix))

                # Determinism (diagonal lines in recurrence plot)
                diag_lengths = []
                for k in range(-len(sampled)+1, len(sampled)):
                    diag = np.diag(recurrence_matrix, k)
                    # Count consecutive True values
                    in_line = False
                    line_len = 0
                    for val in diag:
                        if val:
                            in_line = True
                            line_len += 1
                        else:
                            if in_line and line_len > 1:
                                diag_lengths.append(line_len)
                            in_line = False
                            line_len = 0
                    if in_line and line_len > 1:
                        diag_lengths.append(line_len)

                if diag_lengths:
                    results['determinism'] = float(sum(diag_lengths) / (np.sum(recurrence_matrix) + 1e-10))
                    results['mean_diagonal_length'] = float(np.mean(diag_lengths))
                else:
                    results['determinism'] = 0.0
                    results['mean_diagonal_length'] = 0.0

                # === Generate visualization ===
                self.log("Generating visualization...")
                fig = make_subplots(
                    rows=3, cols=3,
                    subplot_titles=(
                        'Cosine Similarity', 'State Norms', 'Velocity Norms',
                        'PCA Trajectory (2D)', 'Power Spectrum', 'Recurrence Plot',
                        'Phase Portrait (PC1 vs PC2)', 'Norm Phase Space', 'Velocity Correlation'
                    ),
                    specs=[
                        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'heatmap'}],
                        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                    ]
                )

                steps = list(range(len(cos_sims)))

                # Row 1
                fig.add_trace(go.Scatter(x=steps, y=cos_sims, name='Cosine Sim'), row=1, col=1)
                fig.add_trace(go.Scatter(x=steps, y=norms_a, name='Norm A'), row=1, col=2)
                fig.add_trace(go.Scatter(x=steps, y=norms_b, name='Norm B'), row=1, col=2)
                fig.add_trace(go.Scatter(x=list(range(len(vel_norms_a))), y=vel_norms_a, name='Vel A'), row=1, col=3)
                fig.add_trace(go.Scatter(x=list(range(len(vel_norms_b))), y=vel_norms_b, name='Vel B'), row=1, col=3)

                # Row 2
                colors = np.linspace(0, 1, len(joint_pca))
                fig.add_trace(go.Scatter(x=joint_pca[:, 0], y=joint_pca[:, 1],
                                        mode='markers', marker=dict(size=3, color=colors, colorscale='Viridis'),
                                        name='Trajectory'), row=2, col=1)
                fig.add_trace(go.Scatter(x=xf[1:len(power)//4], y=power[1:len(power)//4], name='Power'), row=2, col=2)
                fig.add_trace(go.Heatmap(z=recurrence_matrix.astype(float), colorscale='Blues', showscale=False), row=2, col=3)

                # Row 3
                # Phase portrait
                fig.add_trace(go.Scatter(x=joint_pca[:-1, 0], y=joint_pca[1:, 0],
                                        mode='markers', marker=dict(size=2, color=colors[:-1], colorscale='Viridis'),
                                        name='PC1 Phase'), row=3, col=1)
                # Norm phase space
                fig.add_trace(go.Scatter(x=norms_a[:-1], y=norms_a[1:],
                                        mode='markers', marker=dict(size=2), name='Norm A'), row=3, col=2)
                fig.add_trace(go.Scatter(x=norms_b[:-1], y=norms_b[1:],
                                        mode='markers', marker=dict(size=2), name='Norm B'), row=3, col=2)
                # Velocity correlation
                fig.add_trace(go.Scatter(x=vel_norms_a, y=vel_norms_b, mode='markers',
                                        marker=dict(size=2), name='Vel Correlation'), row=3, col=3)

                fig.update_layout(height=900, width=1200, title_text="Vector Dynamics Full Analysis", showlegend=False)

                # Save and open
                import tempfile
                import webbrowser
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    fig.write_html(f.name)
                    webbrowser.open(f'file://{f.name}')

                # Store results
                self._vd_full_results = results

                # Log summary
                self.log("=" * 50)
                self.log("FULL ANALYSIS RESULTS")
                self.log("=" * 50)
                self.log(f"Trajectory: {results['num_steps']} steps, dims=({results['embedding_dim_a']}, {results['embedding_dim_b']})")
                self.log(f"Synchronization Index: {results['synchronization_index']:.4f}")
                self.log(f"Lyapunov Exponent: {results['lyapunov_exponent']:.4f}")
                self.log(f"Attractor Dimension: {results['attractor_dimension']:.2f}")
                self.log(f"Cosine Similarity: {results['cosine_mean']:.4f} +/- {results['cosine_std']:.4f}")
                self.log(f"Dominant Frequency: {results['dominant_frequency']:.4f}")
                self.log(f"Recurrence Rate: {results['recurrence_rate']:.4f}")
                self.log(f"Determinism: {results['determinism']:.4f}")
                self.log(f"Stationary: {results['is_stationary']} (p={results['stationarity_ks_pval']:.4f})")
                self.log(f"PCA Variance (3D): {results['pca_variance_explained_3d']*100:.1f}%")
                self.log("=" * 50)
                self.log("Full analysis complete - visualization opened in browser")

            except Exception as e:
                self.log(f"Error in full analysis: {e}")
                import traceback
                traceback.print_exc()

        # Run in background thread
        threading.Thread(target=run_analysis, daemon=True).start()

    # =========================================================================
    # Batch Experiment Methods
    # =========================================================================

    def _get_cached_models(self) -> list:
        """Get list of locally cached HuggingFace models"""
        import os
        from pathlib import Path

        cached = []

        # Check HuggingFace cache directory
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for model_dir in hf_cache.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("models--"):
                    # Extract model name from directory name
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    cached.append(model_name)

        # If no cached models found, return defaults
        if not cached:
            return ["gpt2", "gpt2-medium", "distilgpt2"]

        # Sort by model name for consistent ordering
        return sorted(cached)

    def _get_batch_experiments(self, set_name: str) -> list:
        """Get list of experiments for a given experiment set"""
        # Define experiments using only cached models
        # Format: (model_a, model_b, coupling, init, alpha_or_decay)

        if "Full Replication Suite" in set_name:
            # Combine all phases
            all_experiments = []
            all_experiments.extend(self._get_batch_experiments("Phase 1.1"))
            all_experiments.extend(self._get_batch_experiments("Phase 1.2"))
            all_experiments.extend(self._get_batch_experiments("Phase 1.3"))
            all_experiments.extend(self._get_batch_experiments("Phase 1.4"))
            all_experiments.extend(self._get_batch_experiments("Phase 1.5"))
            all_experiments.extend(self._get_batch_experiments("Phase 2"))
            all_experiments.extend(self._get_batch_experiments("Phase 3"))
            all_experiments.extend(self._get_batch_experiments("Phase 4"))
            all_experiments.extend(self._get_batch_experiments("Phase 5"))
            return all_experiments

        elif "Phase 1.1" in set_name:
            # GPT-2 + GPT-2 same model experiments
            return [
                ("gpt2", "gpt2", "direct", "random", None),
                ("gpt2", "gpt2", "direct", "same", None),
                ("gpt2", "gpt2", "direct", "text", None),
                ("gpt2", "gpt2", "blended", "random", 0.5),
                ("gpt2", "gpt2", "blended", "random", 0.3),
                ("gpt2", "gpt2", "blended", "random", 0.7),
                ("gpt2", "gpt2", "decay", "random", 0.1),
                ("gpt2", "gpt2", "decay", "random", 0.3),
            ]

        elif "Phase 1.2" in set_name:
            # DistilGPT-2 experiments
            return [
                ("distilgpt2", "distilgpt2", "direct", "random", None),
                ("distilgpt2", "distilgpt2", "blended", "random", 0.5),
                ("distilgpt2", "distilgpt2", "decay", "random", 0.1),
            ]

        elif "Phase 1.3" in set_name:
            # Identity Projection Control
            # Tests if projection layer itself breaks synchronization
            return [
                ("gpt2", "gpt2", "direct", "random", "force_projection"),
                ("gpt2", "gpt2", "direct", "text", "force_projection"),
            ]

        elif "Phase 1.4" in set_name:
            # BERT Family (Encoder-only)
            # Generalization test beyond GPT family
            return [
                ("bert-base-uncased", "distilbert-base-uncased", "direct", "random", None),
                ("bert-base-uncased", "bert-base-uncased", "direct", "random", None),
            ]

        elif "Phase 1.5" in set_name:
            # Fine-tuning Shift (Dialogue vs Base)
            # Tests if task-specific tuning moves the manifold
            return [
                ("gpt2-medium", "microsoft/DialoGPT-medium", "direct", "random", None),
                ("gpt2-medium", "gpt2-medium", "direct", "random", None),
            ]

        elif "Phase 2" in set_name:
            # Same arch different scale
            return [
                ("gpt2", "gpt2-medium", "direct", "random", None),
                ("gpt2", "gpt2-medium", "blended", "random", 0.5),
                ("gpt2", "gpt2-large", "direct", "random", None),
                ("gpt2", "gpt2-large", "blended", "random", 0.5),
                ("gpt2-medium", "gpt2-large", "direct", "random", None),
            ]

        elif "Phase 3" in set_name:
            # Same dim different model (768): GPT-2 vs DistilGPT-2
            # Tests if same dimension but different training leads to sync
            return [
                ("gpt2", "distilgpt2", "direct", "random", None),
                ("gpt2", "distilgpt2", "blended", "random", 0.5),
                ("gpt2", "distilgpt2", "decay", "random", 0.1),
            ]

        elif "Phase 4" in set_name:
            # Cross-scale cross-architecture
            return [
                ("gpt2", "EleutherAI/gpt-neo-1.3B", "direct", "random", None),
                ("gpt2", "EleutherAI/gpt-neo-1.3B", "blended", "random", 0.5),
                ("gpt2", "EleutherAI/gpt-neo-1.3B", "blended", "text", 0.5),
                ("gpt2", "EleutherAI/gpt-neo-1.3B", "decay", "text", 0.1),
                ("gpt2-medium", "EleutherAI/gpt-neo-1.3B", "direct", "random", None),
            ]

        elif "Phase 5" in set_name:
            # Coupling strength sweep
            experiments = []
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                experiments.append(("gpt2", "gpt2", "blended", "random", alpha))
            return experiments

        elif "Custom" in set_name:
            # Custom selection - use GUI values
            model_a = dpg.get_value("batch_custom_model_a")
            model_b = dpg.get_value("batch_custom_model_b")
            return [
                (model_a, model_b, "direct", "random", None),
                (model_a, model_b, "blended", "random", 0.5),
                (model_a, model_b, "decay", "random", 0.1),
            ]

        return []

    def _update_batch_experiment_list(self):
        """Update the experiment list display"""
        set_name = dpg.get_value("batch_experiment_set") if dpg.does_item_exist("batch_experiment_set") else "Phase 1.1"
        experiments = self._get_batch_experiments(set_name)

        # Format for display
        lines = []
        for i, exp in enumerate(experiments[:15]):  # Limit display to 15
            model_a, model_b, coupling, init, param = exp
            # Shorten model names for display
            ma = model_a.split("/")[-1][:12]
            mb = model_b.split("/")[-1][:12]
            param_str = f" ({param})" if param is not None else ""
            lines.append(f"{i+1}. {ma} + {mb} | {coupling}{param_str} | {init}")

        if len(experiments) > 15:
            lines.append(f"... and {len(experiments) - 15} more")

        dpg.set_value("batch_experiment_list", "\n".join(lines) if lines else "No experiments defined")

    def _on_batch_set_changed(self, sender, app_data):
        """Callback when experiment set changes"""
        self._update_batch_experiment_list()

    def _run_batch_experiments(self):
        """Run all experiments in the selected set"""
        if hasattr(self, '_batch_running') and self._batch_running:
            self.log("Batch experiments already running")
            return

        def run_batch():
            try:
                self._batch_running = True
                self._batch_stop_requested = False
                self._batch_results = []

                from src.agents.hf_agent import HuggingFaceAgent
                from src.training.vector_dynamics import (
                    VectorDynamicsTrainer,
                    create_coupling_strategy,
                    analyze_trajectory,
                )
                import json
                from datetime import datetime

                set_name = dpg.get_value("batch_experiment_set")
                experiments = self._get_batch_experiments(set_name)
                num_steps = dpg.get_value("batch_steps")
                num_seeds = dpg.get_value("batch_seeds")

                total_runs = len(experiments) * num_seeds
                dpg.set_value("batch_status", f"Running 0/{total_runs}...")
                dpg.configure_item("batch_status", color=(200, 200, 80))

                device_selection = dpg.get_value("vd_device")
                device = 'cuda' if 'cuda' in device_selection.lower() else 'cpu'

                # Setup incremental save file
                results_dir = self.base_dir / "data" / "batch_results"
                results_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                set_name_clean = set_name.replace(" ", "_").replace(":", "").replace("/", "-")
                self._batch_save_file = results_dir / f"batch_{set_name_clean}_{timestamp}.json"
                self.log(f"Results will be saved incrementally to: {self._batch_save_file}")

                # Cache loaded models to avoid reloading
                # NOTE: This is safe in Branch 2 because models are FROZEN (eval mode, no_grad)
                # We're not training - just passing embeddings through fixed transformers
                model_cache = {}
                run_idx = 0

                for exp in experiments:
                    if self._batch_stop_requested:
                        break

                    model_a_name, model_b_name, coupling_name, init_mode, param = exp

                    for seed in range(num_seeds):
                        if self._batch_stop_requested:
                            break

                        run_idx += 1
                        dpg.set_value("batch_status", f"Running {run_idx}/{total_runs}...")
                        dpg.set_value("batch_progress", run_idx / total_runs)
                        dpg.set_value("batch_progress_text", f"{run_idx} / {total_runs} experiments")

                        try:
                            # Load or get cached models
                            if model_a_name not in model_cache:
                                self.log(f"Loading model: {model_a_name}")
                                model_cache[model_a_name] = HuggingFaceAgent(f"agent_{model_a_name}", {
                                    'model': model_a_name,
                                    'device': device,
                                    'generation': {'max_new_tokens': 20},
                                })

                            if model_b_name not in model_cache:
                                self.log(f"Loading model: {model_b_name}")
                                model_cache[model_b_name] = HuggingFaceAgent(f"agent_{model_b_name}", {
                                    'model': model_b_name,
                                    'device': device,
                                    'generation': {'max_new_tokens': 20},
                                })

                            agent_a = model_cache[model_a_name]
                            agent_b = model_cache[model_b_name]

                            # Create coupling strategy
                            coupling_config = {'hidden_dim': agent_a.embedding_dim, 'device': device}
                            if coupling_name == 'blended' and param is not None:
                                coupling_config['alpha'] = param
                            elif coupling_name == 'decay' and param is not None:
                                coupling_config['decay'] = param
                            elif coupling_name == 'direct' and param == 'force_projection':
                                coupling_config['force_projection'] = True

                            coupling = create_coupling_strategy(coupling_name, coupling_config)

                            # Create trainer
                            trainer = VectorDynamicsTrainer(agent_a, agent_b, coupling, {
                                'record_every': max(1, num_steps // 100),
                                'compute_metrics_every': max(1, num_steps // 50),
                            })

                            # Set random seed
                            import torch
                            import numpy as np
                            torch.manual_seed(seed + 42)
                            np.random.seed(seed + 42)

                            # Run experiment
                            trajectory = trainer.run(
                                num_steps=num_steps,
                                init_mode=init_mode,
                                init_text_a="The nature of consciousness",
                                init_text_b="In the beginning",
                            )

                            # Analyze
                            results = analyze_trajectory(trajectory)
                            results['experiment'] = {
                                'model_a': model_a_name,
                                'model_b': model_b_name,
                                'coupling': coupling_name,
                                'init_mode': init_mode,
                                'param': param,
                                'seed': seed,
                                'num_steps': num_steps,
                            }

                            self._batch_results.append(results)

                            # Log progress
                            cos_sim = results.get('final_cosine_similarity', 0)
                            sync_idx = results.get('synchronization_index', 0)
                            self.log(f"  [{run_idx}/{total_runs}] {model_a_name.split('/')[-1]} + {model_b_name.split('/')[-1]} ({coupling_name}): cos={cos_sim:.3f}, sync={sync_idx:.3f}")

                            # INCREMENTAL SAVE after each experiment
                            self._save_batch_results_incremental()

                        except Exception as e:
                            self.log(f"  Error in experiment {run_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Still save what we have so far on error
                            self._save_batch_results_incremental()

                # Update summary
                if self._batch_results:
                    summary_lines = []
                    summary_lines.append(f"Completed {len(self._batch_results)} experiments")
                    summary_lines.append("")

                    # Group by coupling type
                    by_coupling = {}
                    for r in self._batch_results:
                        c = r['experiment']['coupling']
                        if c not in by_coupling:
                            by_coupling[c] = []
                        by_coupling[c].append(r)

                    for coupling_type, results in by_coupling.items():
                        cos_vals = [r['final_cosine_similarity'] for r in results if r['final_cosine_similarity'] is not None]
                        if cos_vals:
                            avg_cos = sum(cos_vals) / len(cos_vals)
                            summary_lines.append(f"{coupling_type}: avg cos_sim = {avg_cos:.3f} (n={len(cos_vals)})")

                    dpg.set_value("batch_results_summary", "\n".join(summary_lines))

                    # Save results to JSON
                    self._save_batch_results()

                if self._batch_stop_requested:
                    dpg.set_value("batch_status", "Stopped")
                    dpg.configure_item("batch_status", color=(200, 80, 80))
                else:
                    dpg.set_value("batch_status", "Complete")
                    dpg.configure_item("batch_status", color=(80, 200, 120))
                    dpg.set_value("batch_progress", 1.0)

                self.log(f"Batch experiments complete: {len(self._batch_results)} results saved")

            except Exception as e:
                self.log(f"Error in batch experiments: {e}")
                import traceback
                traceback.print_exc()
                dpg.set_value("batch_status", f"Error: {str(e)[:50]}")
                dpg.configure_item("batch_status", color=(200, 80, 80))
            finally:
                self._batch_running = False

        self._batch_thread = threading.Thread(target=run_batch, daemon=True)
        self._batch_thread.start()

    def _stop_batch_experiments(self):
        """Stop batch experiment run"""
        self._batch_stop_requested = True
        self.log("Stopping batch experiments...")

    def _save_batch_results_incremental(self):
        """Save batch results incrementally (after each experiment)"""
        import json

        if not hasattr(self, '_batch_results') or not self._batch_results:
            return

        if not hasattr(self, '_batch_save_file'):
            # Should not happen if flow is correct, but fallback
            results_dir = self.base_dir / "data" / "batch_results"
            results_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._batch_save_file = results_dir / f"batch_results_{timestamp}.json"

        try:
            # Save JSON
            with open(self._batch_save_file, 'w') as f:
                json.dump(self._batch_results, f, indent=2)
            
            # Save CSV
            self._save_batch_results_csv()
            
        except Exception as e:
            self.log(f"Error saving batch results: {e}")

    def _save_batch_results_csv(self):
        """Save batch results to CSV file for paper"""
        if not hasattr(self, '_batch_results') or not self._batch_results:
            return

        import csv
        
        # Use same filename base as JSON but with .csv extension
        if hasattr(self, '_batch_save_file'):
            csv_file = self._batch_save_file.with_suffix('.csv')
        else:
            return

        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    'Model A', 'Model B', 'Coupling', 'Init Mode', 'Param', 'Seed',
                    'Final Cosine Sim', 'Sync Index', 'Lyapunov Exp',
                    'Attractor Dim', 'CKA Linear', 'CKA Kernel'
                ])
                
                for r in self._batch_results:
                    exp = r.get('experiment', {})
                    writer.writerow([
                        exp.get('model_a', ''),
                        exp.get('model_b', ''),
                        exp.get('coupling', ''),
                        exp.get('init_mode', ''),
                        exp.get('param', ''),
                        exp.get('seed', ''),
                        f"{r.get('final_cosine_similarity', 0):.4f}",
                        f"{r.get('synchronization_index', 0):.4f}",
                        f"{r.get('lyapunov_exponent', 0):.4f}",
                        f"{r.get('attractor_dimension', 0):.4f}",
                        f"{r.get('cka_linear', 0):.4f}",
                        f"{r.get('cka_kernel', 0):.4f}"
                    ])
            # Only log on first write or periodically to avoid spamming? 
            # Actually, logging every time is fine for "Incremental"
            # self.log(f"CSV summary updated: {csv_file.name}")
        except Exception as e:
            self.log(f"Error saving CSV: {e}")

    def _save_batch_results(self):
        """Save batch results to JSON file"""
        import json
        from datetime import datetime

        if not hasattr(self, '_batch_results') or not self._batch_results:
            return

        # Create results directory
        results_dir = self.base_dir / "data" / "batch_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        set_name = dpg.get_value("batch_experiment_set").replace(" ", "_").replace(":", "").replace("/", "-")
        filename = results_dir / f"batch_{set_name}_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for r in self._batch_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, type(None), dict, list)):
                    sr[k] = v
                else:
                    sr[k] = str(v)
            serializable_results.append(sr)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.log(f"Batch results saved to: {filename}")

    def _export_batch_results(self):
        """Export batch results to JSON"""
        if not hasattr(self, '_batch_results') or not self._batch_results:
            self.log("No batch results to export. Run batch experiments first.")
            return
        self._save_batch_results()
        self.log("Results exported!")

    def _open_batch_results_folder(self):
        """Open the batch results folder"""
        import subprocess
        import os

        results_dir = self.base_dir / "data" / "batch_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if os.name == 'nt':  # Windows
            os.startfile(str(results_dir))
        elif os.name == 'posix':  # macOS/Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(results_dir)])

        self.log(f"Opened results folder: {results_dir}")

    def _setup_validation_tab(self):
        """Setup Model Merging Validation tab"""
        dpg.add_text("MERGING VALIDATION", color=(50, 100, 180))
        dpg.add_separator()
        dpg.add_text("Validate paper predictions: High geometric synchronization should predict successful model merging.",
                     wrap=700, color=(80, 80, 90))
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Left Panel: Configuration
            with dpg.child_window(width=400, height=-50):
                dpg.add_text("Test Configuration", color=(50, 100, 180))
                dpg.add_separator()

                dpg.add_text("Paper Scenarios:", color=(100, 100, 120))
                scenarios = [
                    "Custom",
                    "Paper: High Sync (GPT-2 + DistilGPT-2)",
                    "Paper: Low Sync (GPT-2 + GPT-2-Medium)",
                    "Control: Identity (GPT-2 + GPT-2)"
                ]
                dpg.add_combo(scenarios, default_value=scenarios[0], tag="val_scenario",
                             callback=self._on_val_scenario_changed, width=-1)
                
                dpg.add_spacer(height=10)
                dpg.add_text("Models:", color=(100, 100, 120))
                models = self._get_available_models()
                dpg.add_combo(models, default_value="gpt2", tag="val_model_a", width=-1)
                dpg.add_combo(models, default_value="distilgpt2", tag="val_model_b", width=-1)

                dpg.add_spacer(height=10)
                dpg.add_text("Validation Method:", color=(100, 100, 120))
                dpg.add_radio_button(["Weight Merging", "Representation Stitching"], 
                                   default_value="Weight Merging", tag="val_method")
                
                dpg.add_spacer(height=10)
                dpg.add_text("Evaluation:", color=(100, 100, 120))
                dpg.add_checkbox(label="Use WikiText-2 (Slow)", tag="val_use_wikitext", default_value=False)
                dpg.add_input_int(label="Test Samples", default_value=10, tag="val_samples", width=100)

                dpg.add_spacer(height=20)
                dpg.add_button(label="Run Validation", callback=self._run_merging_validation, 
                              width=-1, height=40)
                
                dpg.add_spacer(height=10)
                dpg.add_text("Status:", color=(100, 100, 120))
                dpg.add_text("Ready", tag="val_status", color=(80, 200, 120))
                dpg.add_progress_bar(default_value=0.0, tag="val_progress", width=-1)

            dpg.add_spacer(width=20)

            # Right Panel: Results
            with dpg.child_window(width=-1, height=-50):
                dpg.add_text("Validation Results", color=(50, 100, 180))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    with dpg.child_window(width=200, height=120):
                        dpg.add_text("Perplexity (Model A)", color=(100, 100, 120))
                        dpg.add_text("--", tag="val_ppl_a", color=(50, 100, 180))
                        dpg.add_spacer(height=10)
                        dpg.add_text("Perplexity (Model B)", color=(100, 100, 120))
                        dpg.add_text("--", tag="val_ppl_b", color=(50, 100, 180))

                    with dpg.child_window(width=200, height=120):
                        dpg.add_text("Merged Perplexity", color=(100, 100, 120))
                        dpg.add_text("--", tag="val_ppl_merged", color=(60, 160, 80))
                        dpg.add_spacer(height=10)
                        dpg.add_text("Degradation", color=(100, 100, 120))
                        dpg.add_text("--", tag="val_degradation", color=(200, 60, 60))

                dpg.add_spacer(height=10)
                dpg.add_text("Interpretation:", color=(100, 100, 120))
                dpg.add_text("", tag="val_interpretation", wrap=-1, color=(80, 80, 90))
                
                dpg.add_spacer(height=10)
                dpg.add_text("Detailed Log:", color=(100, 100, 120))
                dpg.add_input_text(multiline=True, readonly=True, tag="val_log", width=-1, height=-1)

    def _on_val_scenario_changed(self, sender, app_data):
        """Preset selection for validation"""
        if "High Sync" in app_data:
            dpg.set_value("val_model_a", "gpt2")
            dpg.set_value("val_model_b", "distilgpt2")
            dpg.set_value("val_method", "Representation Stitching")
        elif "Low Sync" in app_data:
            dpg.set_value("val_model_a", "gpt2")
            dpg.set_value("val_model_b", "gpt2-medium")
            dpg.set_value("val_method", "Weight Merging")
        elif "Identity" in app_data:
            dpg.set_value("val_model_a", "gpt2")
            dpg.set_value("val_model_b", "gpt2")
            dpg.set_value("val_method", "Weight Merging")

    def _run_merging_validation(self):
        """Run the merging/stitching validation logic"""
        if hasattr(self, '_val_running') and self._val_running:
            return

        def validation_worker():
            self._val_running = True
            try:
                import torch
                import numpy as np
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from copy import deepcopy
                
                model_a_name = dpg.get_value("val_model_a")
                model_b_name = dpg.get_value("val_model_b")
                method = dpg.get_value("val_method")
                use_wikitext = dpg.get_value("val_use_wikitext")
                num_samples = dpg.get_value("val_samples")
                
                dpg.set_value("val_status", "Loading models...")
                dpg.configure_item("val_status", color=(200, 200, 80))
                dpg.set_value("val_log", f"Starting validation: {model_a_name} + {model_b_name} via {method}\n")
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Load models
                tokenizer = AutoTokenizer.from_pretrained(model_a_name)
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                
                model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device).eval()
                model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device).eval()
                
                # Prepare data
                dpg.set_value("val_status", "Preparing data...")
                texts = []
                if use_wikitext:
                    try:
                        from datasets import load_dataset
                        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                        texts = [x['text'] for x in ds if len(x['text']) > 100][:num_samples]
                    except:
                        texts = ["The quick brown fox jumps over the lazy dog. " * 20] * num_samples
                else:
                    texts = ["The quick brown fox jumps over the lazy dog. " * 20] * num_samples

                def compute_ppl(model, desc):
                    dpg.set_value("val_status", f"Evaluating {desc}...")
                    total_loss = 0
                    total_tokens = 0
                    for i, text in enumerate(texts):
                        dpg.set_value("val_progress", (i / len(texts)) * 0.3 + (0.3 if 'B' in desc else 0))
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                        with torch.no_grad():
                            out = model(**inputs, labels=inputs.input_ids)
                            total_loss += out.loss.item() * inputs.input_ids.numel()
                            total_tokens += inputs.input_ids.numel()
                    return np.exp(total_loss / total_tokens)

                # Baseline PPL
                ppl_a = compute_ppl(model_a, "Model A")
                dpg.set_value("val_ppl_a", f"{ppl_a:.2f}")
                
                ppl_b = compute_ppl(model_b, "Model B")
                dpg.set_value("val_ppl_b", f"{ppl_b:.2f}")
                
                # Merge
                dpg.set_value("val_status", "Merging...")
                dpg.set_value("val_progress", 0.7)
                
                merged_ppl = 0.0
                
                if method == "Weight Merging":
                    try:
                        merged = deepcopy(model_a)
                        sd_a = model_a.state_dict()
                        sd_b = model_b.state_dict()
                        for k in sd_a:
                            if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                                sd_a[k] = (sd_a[k] + sd_b[k]) / 2.0
                        merged.load_state_dict(sd_a)
                        merged_ppl = compute_ppl(merged, "Merged Model")
                    except Exception as e:
                        dpg.set_value("val_log", dpg.get_value("val_log") + f"\nMerge Error: {e}\n")
                        merged_ppl = float('inf')
                        
                else: # Representation Stitching
                    dpg.set_value("val_status", "Stitching...")
                    losses = []
                    for i, text in enumerate(texts):
                        dpg.set_value("val_progress", 0.7 + (i/len(texts))*0.3)
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                        with torch.no_grad():
                            out_a = model_a(inputs.input_ids, output_hidden_states=True)
                            out_b = model_b(inputs.input_ids, output_hidden_states=True)
                            
                            h_a = out_a.hidden_states[-1]
                            h_b = out_b.hidden_states[-1]
                            
                            min_dim = min(h_a.shape[-1], h_b.shape[-1])
                            h_merged = (h_a[..., :min_dim] + h_b[..., :min_dim]) / 2.0
                            
                            if h_a.shape[-1] > min_dim:
                                padding = torch.zeros(*h_merged.shape[:-1], h_a.shape[-1]-min_dim, device=device)
                                h_merged = torch.cat([h_merged, padding], dim=-1)
                                
                            logits = model_a.lm_head(h_merged)
                            
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs.input_ids[..., 1:].contiguous()
                            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            losses.append(loss.item())
                    merged_ppl = np.exp(np.mean(losses))

                dpg.set_value("val_ppl_merged", f"{merged_ppl:.2f}")
                
                avg_base = (ppl_a + ppl_b) / 2
                degrad = (merged_ppl - avg_base) / avg_base * 100
                dpg.set_value("val_degradation", f"{degrad:.1f}%")
                
                if degrad < 50:
                    interp = "SUCCESS: Models are geometrically compatible."
                    dpg.configure_item("val_interpretation", color=(60, 160, 80))
                else:
                    interp = "FAILURE: Models have incompatible geometries."
                    dpg.configure_item("val_interpretation", color=(200, 60, 60))
                
                dpg.set_value("val_interpretation", interp)
                dpg.set_value("val_status", "Complete")
                dpg.configure_item("val_status", color=(80, 200, 120))
                dpg.set_value("val_progress", 1.0)
                
            except Exception as e:
                import traceback
                dpg.set_value("val_status", "Error")
                dpg.configure_item("val_status", color=(200, 60, 60))
                dpg.set_value("val_log", dpg.get_value("val_log") + f"\nError: {e}\n{traceback.format_exc()}")
            finally:
                self._val_running = False

        import threading
        threading.Thread(target=validation_worker, daemon=True).start()

    def _setup_weight_analysis_tab(self):
        """Setup weight diff analysis tab for comparing trained vs pretrained models"""
        dpg.add_text("WEIGHT ANALYSIS", color=(50, 100, 180))
        dpg.add_separator()
        dpg.add_text("Compare trained model weights against original pretrained weights to understand what changed.",
                     wrap=700, color=(80, 80, 90))
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Left panel - Selection and Controls
            with dpg.child_window(width=350, height=-50):
                dpg.add_text("Model Selection", color=(50, 100, 180))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text("Experiment:", color=(100, 100, 120))
                    dpg.add_button(label="Refresh", callback=self._refresh_weight_experiments, width=60)
                experiments = self._get_experiments_with_checkpoints()
                dpg.add_combo(experiments, default_value=experiments[0] if experiments else "(none)",
                             tag="weight_analysis_experiment", width=-1,
                             callback=self._on_weight_analysis_experiment_changed)

                dpg.add_text("Checkpoint:", color=(100, 100, 120))
                dpg.add_combo(["(none)"], default_value="(none)",
                             tag="weight_analysis_checkpoint", width=-1)

                dpg.add_text("Agent:", color=(100, 100, 120))
                dpg.add_combo(["agent_a", "agent_b"], default_value="agent_a",
                             tag="weight_analysis_agent", width=-1)

                dpg.add_spacer(height=15)
                dpg.add_button(label="Load & Analyze", callback=self._run_weight_analysis,
                              width=-1, height=40)

                dpg.add_spacer(height=15)
                dpg.add_text("Analysis Status:", color=(100, 100, 120))
                dpg.add_text("Ready", tag="weight_analysis_status", color=(80, 200, 120))

                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_text("Quick Stats", color=(50, 100, 180))
                dpg.add_spacer(height=5)

                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False,
                              borders_innerV=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=150)
                    dpg.add_table_column(width_stretch=True)

                    with dpg.table_row():
                        dpg.add_text("Total Parameters:", color=(100, 100, 120))
                        dpg.add_text("--", tag="weight_stat_total_params")
                    with dpg.table_row():
                        dpg.add_text("Changed Params:", color=(100, 100, 120))
                        dpg.add_text("--", tag="weight_stat_changed_params")
                    with dpg.table_row():
                        dpg.add_text("Total L2 Delta:", color=(100, 100, 120))
                        dpg.add_text("--", tag="weight_stat_l2_delta")
                    with dpg.table_row():
                        dpg.add_text("Avg Change %:", color=(100, 100, 120))
                        dpg.add_text("--", tag="weight_stat_avg_change")
                    with dpg.table_row():
                        dpg.add_text("Most Changed:", color=(100, 100, 120))
                        dpg.add_text("--", tag="weight_stat_most_changed")

            dpg.add_spacer(width=20)

            # Right panel - Analysis Results
            with dpg.child_window(width=-1, height=-50):
                with dpg.tab_bar(tag="weight_analysis_tabs"):
                    with dpg.tab(label="Weight-Level"):
                        dpg.add_text("Weight-Level Analysis", color=(50, 100, 180))
                        dpg.add_text("Layer-by-layer weight magnitude changes, distributions, and SVD analysis.",
                                    color=(80, 80, 90))
                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Layer Magnitudes", callback=self._show_layer_magnitudes, width=140)
                            dpg.add_button(label="Weight Distributions", callback=self._show_weight_distributions, width=140)
                            dpg.add_button(label="SVD Analysis", callback=self._show_svd_analysis, width=140)

                        dpg.add_spacer(height=10)
                        dpg.add_input_text(multiline=True, readonly=True, tag="weight_level_results",
                                          width=-1, height=300)

                    with dpg.tab(label="Structural"):
                        dpg.add_text("Structural Analysis", color=(50, 100, 180))
                        dpg.add_text("Attention head changes, early vs late layers, embedding shifts.",
                                    color=(80, 80, 90))
                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Attention Heads", callback=self._show_attention_head_analysis, width=140)
                            dpg.add_button(label="Layer Depth", callback=self._show_layer_depth_analysis, width=140)
                            dpg.add_button(label="Embedding Shifts", callback=self._show_embedding_shifts, width=140)

                        dpg.add_spacer(height=10)
                        dpg.add_input_text(multiline=True, readonly=True, tag="structural_results",
                                          width=-1, height=300)

                    with dpg.tab(label="Functional"):
                        dpg.add_text("Functional Analysis", color=(50, 100, 180))
                        dpg.add_text("Compare outputs on same prompts, perplexity changes, attention patterns.",
                                    color=(80, 80, 90))
                        dpg.add_spacer(height=10)

                        dpg.add_text("Test Prompt:", color=(100, 100, 120))
                        dpg.add_input_text(default_value="The nature of consciousness is",
                                          tag="weight_func_test_prompt", width=-1)

                        dpg.add_spacer(height=10)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Compare Outputs", callback=self._show_output_comparison, width=140)
                            dpg.add_button(label="Perplexity", callback=self._show_perplexity_analysis, width=140)
                            dpg.add_button(label="Attention Patterns", callback=self._show_attention_patterns, width=140)

                        dpg.add_spacer(height=10)
                        dpg.add_input_text(multiline=True, readonly=True, tag="functional_results",
                                          width=-1, height=300)

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Export Full Report", callback=self._export_weight_report, width=150)
            dpg.add_button(label="Open All Plots", callback=self._open_all_weight_plots, width=150)

    def _refresh_weight_experiments(self, sender=None, app_data=None):
        """Refresh the experiment dropdown in weight analysis"""
        experiments = self._get_experiments_with_checkpoints()
        dpg.configure_item("weight_analysis_experiment", items=experiments)
        if experiments and experiments[0] != "(none)":
            dpg.set_value("weight_analysis_experiment", experiments[0])
            # Also update checkpoints
            checkpoints = self._get_checkpoints_for_experiment(experiments[0])
            dpg.configure_item("weight_analysis_checkpoint", items=checkpoints)
            if checkpoints and checkpoints[0] != "(none)":
                dpg.set_value("weight_analysis_checkpoint", checkpoints[-1])
        self.log(f"Found {len(experiments)} experiments with checkpoints")

    def _on_weight_analysis_experiment_changed(self, sender, app_data):
        """Update checkpoint dropdown when experiment changes"""
        experiment_name = app_data
        if experiment_name and experiment_name != "(none)":
            checkpoints = self._get_checkpoints_for_experiment(experiment_name)
            dpg.configure_item("weight_analysis_checkpoint", items=checkpoints)
            if checkpoints and checkpoints[0] != "(none)":
                dpg.set_value("weight_analysis_checkpoint", checkpoints[-1])
        else:
            dpg.configure_item("weight_analysis_checkpoint", items=["(none)"])

    def _run_weight_analysis(self, sender=None, app_data=None):
        """Load models and run weight analysis"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        experiment = dpg.get_value("weight_analysis_experiment")
        checkpoint = dpg.get_value("weight_analysis_checkpoint")
        agent = dpg.get_value("weight_analysis_agent")

        if experiment == "(none)" or checkpoint == "(none)":
            self.log("Please select an experiment and checkpoint")
            return

        dpg.set_value("weight_analysis_status", "Loading...")
        dpg.configure_item("weight_analysis_status", color=(255, 180, 80))
        self.log(f"Loading weight analysis for {experiment}/{checkpoint}/{agent}")

        def analysis_worker():
            try:
                # Find the config to get model name
                checkpoint_dir = self.experiments_dir / experiment / "checkpoints"
                agent_checkpoint = checkpoint_dir / f"{checkpoint.replace('checkpoint_', f'agent_{agent}_')}.pt"

                if not agent_checkpoint.exists():
                    self.log(f"Agent checkpoint not found: {agent_checkpoint}")
                    dpg.set_value("weight_analysis_status", "Error: Checkpoint not found")
                    dpg.configure_item("weight_analysis_status", color=(255, 80, 80))
                    return

                # Load checkpoint to get model name
                ckpt_data = torch.load(agent_checkpoint, map_location='cpu')
                model_name = ckpt_data.get('config', {}).get('model', 'gpt2')

                self.log(f"Loading pretrained model: {model_name}")

                # Load pretrained (original) model
                pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
                pretrained_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

                # Load trained model
                trained_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
                trained_model.load_state_dict(ckpt_data['model_state'])
                trained_state = trained_model.state_dict()

                # Store for analysis
                self._weight_analysis_data = {
                    'pretrained_state': pretrained_state,
                    'trained_state': trained_state,
                    'model_name': model_name,
                    'pretrained_model': pretrained_model,
                    'trained_model': trained_model,
                    'tokenizer': AutoTokenizer.from_pretrained(model_name),
                }

                # Compute basic stats
                total_params = 0
                changed_params = 0
                total_l2_delta = 0.0
                layer_deltas = {}

                for name in pretrained_state:
                    if name in trained_state:
                        p_weights = pretrained_state[name].float()
                        t_weights = trained_state[name].float()
                        delta = (t_weights - p_weights).abs()

                        num_params = p_weights.numel()
                        total_params += num_params

                        # Count significantly changed params (> 1e-6)
                        changed = (delta > 1e-6).sum().item()
                        changed_params += changed

                        # L2 norm of delta
                        l2 = torch.norm(t_weights - p_weights).item()
                        total_l2_delta += l2

                        # Store per-layer delta
                        layer_deltas[name] = {
                            'l2': l2,
                            'mean_abs': delta.mean().item(),
                            'max_abs': delta.max().item(),
                            'changed_ratio': changed / num_params if num_params > 0 else 0
                        }

                # Find most changed layer
                most_changed = max(layer_deltas.items(), key=lambda x: x[1]['l2'])

                # Store layer deltas
                self._weight_analysis_data['layer_deltas'] = layer_deltas

                # Update UI
                dpg.set_value("weight_stat_total_params", f"{total_params:,}")
                dpg.set_value("weight_stat_changed_params", f"{changed_params:,}")
                dpg.set_value("weight_stat_l2_delta", f"{total_l2_delta:.4f}")
                avg_change = (changed_params / total_params * 100) if total_params > 0 else 0
                dpg.set_value("weight_stat_avg_change", f"{avg_change:.2f}%")
                dpg.set_value("weight_stat_most_changed", most_changed[0][:30])

                dpg.set_value("weight_analysis_status", "Analysis Ready")
                dpg.configure_item("weight_analysis_status", color=(80, 200, 120))
                self.log("Weight analysis complete!")

            except Exception as e:
                self.log(f"Weight analysis error: {e}")
                import traceback
                traceback.print_exc()
                dpg.set_value("weight_analysis_status", f"Error: {str(e)[:30]}")
                dpg.configure_item("weight_analysis_status", color=(255, 80, 80))

        import threading
        threading.Thread(target=analysis_worker, daemon=True).start()

    def _show_layer_magnitudes(self, sender=None, app_data=None):
        """Show layer-by-layer magnitude changes"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        layer_deltas = self._weight_analysis_data.get('layer_deltas', {})

        # Build results text
        results = "LAYER MAGNITUDE ANALYSIS\n"
        results += "=" * 60 + "\n\n"
        results += f"{'Layer Name':<45} {'L2 Delta':>12}\n"
        results += "-" * 60 + "\n"

        # Sort by L2 delta
        sorted_layers = sorted(layer_deltas.items(), key=lambda x: x[1]['l2'], reverse=True)

        for name, data in sorted_layers[:30]:  # Top 30
            short_name = name[:44] if len(name) > 44 else name
            results += f"{short_name:<45} {data['l2']:>12.6f}\n"

        dpg.set_value("weight_level_results", results)

        # Create Plotly chart
        import plotly.graph_objects as go

        # Group by layer type
        layer_types = {}
        for name, data in layer_deltas.items():
            if 'attn' in name.lower() or 'attention' in name.lower():
                layer_type = 'Attention'
            elif 'mlp' in name.lower() or 'fc' in name.lower() or 'dense' in name.lower():
                layer_type = 'MLP/FFN'
            elif 'embed' in name.lower() or 'wte' in name.lower() or 'wpe' in name.lower():
                layer_type = 'Embeddings'
            elif 'ln' in name.lower() or 'norm' in name.lower():
                layer_type = 'LayerNorm'
            else:
                layer_type = 'Other'

            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += data['l2']

        fig = go.Figure(data=[
            go.Bar(x=list(layer_types.keys()), y=list(layer_types.values()),
                  marker_color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B'])
        ])
        fig.update_layout(
            title="Weight Changes by Layer Type",
            xaxis_title="Layer Type",
            yaxis_title="Total L2 Delta",
            template="plotly_dark"
        )
        self.show_plot_in_browser(fig)

    def _show_weight_distributions(self, sender=None, app_data=None):
        """Show weight distribution changes"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        pretrained = self._weight_analysis_data['pretrained_state']
        trained = self._weight_analysis_data['trained_state']

        # Collect all weights
        pre_weights = []
        train_weights = []
        delta_weights = []

        for name in pretrained:
            if name in trained:
                p = pretrained[name].float().flatten().numpy()
                t = trained[name].float().flatten().numpy()
                # Sample to avoid memory issues
                if len(p) > 10000:
                    idx = np.random.choice(len(p), 10000, replace=False)
                    p = p[idx]
                    t = t[idx]
                pre_weights.extend(p)
                train_weights.extend(t)
                delta_weights.extend(t - p)

        pre_weights = np.array(pre_weights)
        train_weights = np.array(train_weights)
        delta_weights = np.array(delta_weights)

        # Results text
        results = "WEIGHT DISTRIBUTION ANALYSIS\n"
        results += "=" * 60 + "\n\n"
        results += f"{'Metric':<25} {'Pretrained':>15} {'Trained':>15}\n"
        results += "-" * 60 + "\n"
        results += f"{'Mean':<25} {np.mean(pre_weights):>15.6f} {np.mean(train_weights):>15.6f}\n"
        results += f"{'Std':<25} {np.std(pre_weights):>15.6f} {np.std(train_weights):>15.6f}\n"
        results += f"{'Min':<25} {np.min(pre_weights):>15.6f} {np.min(train_weights):>15.6f}\n"
        results += f"{'Max':<25} {np.max(pre_weights):>15.6f} {np.max(train_weights):>15.6f}\n"
        results += f"\nDelta Statistics:\n"
        results += f"{'Mean Delta':<25} {np.mean(delta_weights):>15.6f}\n"
        results += f"{'Std Delta':<25} {np.std(delta_weights):>15.6f}\n"
        results += f"{'Max Abs Delta':<25} {np.max(np.abs(delta_weights)):>15.6f}\n"

        dpg.set_value("weight_level_results", results)

        # Create histogram
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Weight Values", "Weight Deltas"))

        fig.add_trace(go.Histogram(x=pre_weights, name="Pretrained", opacity=0.7, nbinsx=100), row=1, col=1)
        fig.add_trace(go.Histogram(x=train_weights, name="Trained", opacity=0.7, nbinsx=100), row=1, col=1)
        fig.add_trace(go.Histogram(x=delta_weights, name="Delta", nbinsx=100, marker_color='orange'), row=1, col=2)

        fig.update_layout(
            title="Weight Distribution Comparison",
            template="plotly_dark",
            barmode='overlay'
        )
        self.show_plot_in_browser(fig)

    def _show_svd_analysis(self, sender=None, app_data=None):
        """Show SVD analysis of weight matrices"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import numpy as np
        import plotly.graph_objects as go

        pretrained = self._weight_analysis_data['pretrained_state']
        trained = self._weight_analysis_data['trained_state']

        results = "SVD ANALYSIS (Singular Value Changes)\n"
        results += "=" * 60 + "\n\n"

        svd_data = []

        for name in pretrained:
            if name in trained and len(pretrained[name].shape) == 2:
                p = pretrained[name].float()
                t = trained[name].float()

                # Only analyze reasonably sized matrices
                if p.shape[0] > 10 and p.shape[1] > 10 and p.numel() < 1000000:
                    try:
                        # Compute SVD
                        _, s_pre, _ = torch.linalg.svd(p, full_matrices=False)
                        _, s_train, _ = torch.linalg.svd(t, full_matrices=False)

                        # Compare top singular values
                        top_k = min(10, len(s_pre))
                        sv_change = torch.norm(s_train[:top_k] - s_pre[:top_k]).item()

                        # Effective rank (number of significant singular values)
                        threshold = 0.01 * s_pre[0].item()
                        eff_rank_pre = (s_pre > threshold).sum().item()
                        eff_rank_train = (s_train > threshold).sum().item()

                        svd_data.append({
                            'name': name,
                            'sv_change': sv_change,
                            'eff_rank_pre': eff_rank_pre,
                            'eff_rank_train': eff_rank_train,
                            'rank_change': eff_rank_train - eff_rank_pre
                        })
                    except Exception:
                        pass

        # Sort by singular value change
        svd_data.sort(key=lambda x: x['sv_change'], reverse=True)

        results += f"{'Layer':<35} {'SV Change':>10} {'Rank Delta':>10}\n"
        results += "-" * 60 + "\n"

        for item in svd_data[:20]:
            short_name = item['name'][:34]
            results += f"{short_name:<35} {item['sv_change']:>10.4f} {item['rank_change']:>+10d}\n"

        dpg.set_value("weight_level_results", results)

        # Plot
        if svd_data:
            names = [d['name'][-30:] for d in svd_data[:15]]
            changes = [d['sv_change'] for d in svd_data[:15]]

            fig = go.Figure(data=[go.Bar(x=names, y=changes, marker_color='#9C27B0')])
            fig.update_layout(
                title="Singular Value Changes by Layer",
                xaxis_title="Layer",
                yaxis_title="Top-k SV Change",
                template="plotly_dark",
                xaxis_tickangle=-45
            )
            self.show_plot_in_browser(fig)

    def _show_attention_head_analysis(self, sender=None, app_data=None):
        """Analyze changes in attention heads"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import numpy as np
        import plotly.graph_objects as go

        pretrained = self._weight_analysis_data['pretrained_state']
        trained = self._weight_analysis_data['trained_state']
        layer_deltas = self._weight_analysis_data.get('layer_deltas', {})

        results = "ATTENTION HEAD ANALYSIS\n"
        results += "=" * 60 + "\n\n"

        attn_layers = {}
        for name, data in layer_deltas.items():
            if 'attn' in name.lower() or 'attention' in name.lower():
                # Extract layer number
                import re
                match = re.search(r'\.(\d+)\.', name)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num not in attn_layers:
                        attn_layers[layer_num] = {'total_l2': 0, 'components': []}
                    attn_layers[layer_num]['total_l2'] += data['l2']
                    attn_layers[layer_num]['components'].append((name, data['l2']))

        results += f"{'Layer':>10} {'Total L2 Delta':>15} {'Top Component':<35}\n"
        results += "-" * 60 + "\n"

        for layer_num in sorted(attn_layers.keys()):
            layer_data = attn_layers[layer_num]
            top_comp = max(layer_data['components'], key=lambda x: x[1])
            results += f"{layer_num:>10} {layer_data['total_l2']:>15.6f} {top_comp[0][-34:]:<35}\n"

        dpg.set_value("structural_results", results)

        # Plot
        if attn_layers:
            layers = sorted(attn_layers.keys())
            deltas = [attn_layers[l]['total_l2'] for l in layers]

            fig = go.Figure(data=[go.Bar(x=[f"Layer {l}" for l in layers], y=deltas,
                                        marker_color='#2196F3')])
            fig.update_layout(
                title="Attention Layer Changes",
                xaxis_title="Layer",
                yaxis_title="Total L2 Delta",
                template="plotly_dark"
            )
            self.show_plot_in_browser(fig)

    def _show_layer_depth_analysis(self, sender=None, app_data=None):
        """Compare early vs late layer changes"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import re
        import plotly.graph_objects as go

        layer_deltas = self._weight_analysis_data.get('layer_deltas', {})

        # Group by layer depth
        layer_depths = {}
        for name, data in layer_deltas.items():
            match = re.search(r'\.h\.(\d+)\.|\.layers\.(\d+)\.|\.block\.(\d+)\.', name)
            if match:
                layer_num = int(next(g for g in match.groups() if g is not None))
                if layer_num not in layer_depths:
                    layer_depths[layer_num] = 0
                layer_depths[layer_num] += data['l2']

        if not layer_depths:
            self.log("Could not parse layer structure")
            return

        results = "LAYER DEPTH ANALYSIS\n"
        results += "=" * 60 + "\n\n"

        max_layer = max(layer_depths.keys())
        early_sum = sum(v for k, v in layer_depths.items() if k < max_layer / 3)
        mid_sum = sum(v for k, v in layer_depths.items() if max_layer / 3 <= k < 2 * max_layer / 3)
        late_sum = sum(v for k, v in layer_depths.items() if k >= 2 * max_layer / 3)

        results += f"Early Layers (0-{int(max_layer/3)}):     {early_sum:.6f}\n"
        results += f"Middle Layers ({int(max_layer/3)}-{int(2*max_layer/3)}):  {mid_sum:.6f}\n"
        results += f"Late Layers ({int(2*max_layer/3)}-{max_layer}):    {late_sum:.6f}\n\n"

        results += f"{'Layer':>10} {'L2 Delta':>15}\n"
        results += "-" * 30 + "\n"
        for layer_num in sorted(layer_depths.keys()):
            results += f"{layer_num:>10} {layer_depths[layer_num]:>15.6f}\n"

        dpg.set_value("structural_results", results)

        # Plot
        layers = sorted(layer_depths.keys())
        deltas = [layer_depths[l] for l in layers]

        fig = go.Figure(data=[go.Scatter(x=layers, y=deltas, mode='lines+markers',
                                        line=dict(color='#4CAF50', width=2),
                                        marker=dict(size=8))])
        fig.update_layout(
            title="Weight Changes by Layer Depth",
            xaxis_title="Layer Number",
            yaxis_title="Total L2 Delta",
            template="plotly_dark"
        )
        self.show_plot_in_browser(fig)

    def _show_embedding_shifts(self, sender=None, app_data=None):
        """Analyze embedding layer changes"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import numpy as np
        import plotly.graph_objects as go

        pretrained = self._weight_analysis_data['pretrained_state']
        trained = self._weight_analysis_data['trained_state']
        tokenizer = self._weight_analysis_data['tokenizer']

        results = "EMBEDDING SHIFT ANALYSIS\n"
        results += "=" * 60 + "\n\n"

        # Find embedding layers
        embed_keys = [k for k in pretrained.keys() if 'embed' in k.lower() or 'wte' in k.lower()]

        for key in embed_keys:
            if key in trained:
                p_emb = pretrained[key].float()
                t_emb = trained[key].float()

                # Per-token embedding change
                token_deltas = torch.norm(t_emb - p_emb, dim=1).numpy()

                results += f"Layer: {key}\n"
                results += f"  Shape: {list(p_emb.shape)}\n"
                results += f"  Mean Delta: {np.mean(token_deltas):.6f}\n"
                results += f"  Max Delta: {np.max(token_deltas):.6f}\n"

                # Find most changed tokens
                top_k = 20
                top_indices = np.argsort(token_deltas)[-top_k:][::-1]

                results += f"\n  Top {top_k} Most Changed Tokens:\n"
                for idx in top_indices:
                    try:
                        token = tokenizer.decode([idx])
                        results += f"    {idx:>6}: '{token:<15}' delta={token_deltas[idx]:.6f}\n"
                    except Exception:
                        results += f"    {idx:>6}: <unknown>       delta={token_deltas[idx]:.6f}\n"

                # Plot histogram
                fig = go.Figure(data=[go.Histogram(x=token_deltas, nbinsx=100, marker_color='#FF9800')])
                fig.update_layout(
                    title=f"Token Embedding Changes ({key})",
                    xaxis_title="L2 Delta",
                    yaxis_title="Count",
                    template="plotly_dark"
                )
                self.show_plot_in_browser(fig)

        dpg.set_value("structural_results", results)

    def _show_output_comparison(self, sender=None, app_data=None):
        """Compare outputs from pretrained vs trained model"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch

        prompt = dpg.get_value("weight_func_test_prompt")
        pretrained_model = self._weight_analysis_data['pretrained_model']
        trained_model = self._weight_analysis_data['trained_model']
        tokenizer = self._weight_analysis_data['tokenizer']

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrained_model = pretrained_model.to(device)
        trained_model = trained_model.to(device)

        results = "OUTPUT COMPARISON\n"
        results += "=" * 60 + "\n\n"
        results += f"Prompt: {prompt}\n\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # Generate from pretrained
            pre_output = pretrained_model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=tokenizer.pad_token_id
            )
            pre_text = tokenizer.decode(pre_output[0], skip_special_tokens=True)

            # Generate from trained
            train_output = trained_model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=tokenizer.pad_token_id
            )
            train_text = tokenizer.decode(train_output[0], skip_special_tokens=True)

        results += "PRETRAINED OUTPUT:\n"
        results += "-" * 40 + "\n"
        results += pre_text + "\n\n"

        results += "TRAINED OUTPUT:\n"
        results += "-" * 40 + "\n"
        results += train_text + "\n"

        dpg.set_value("functional_results", results)
        self.log("Output comparison complete")

    def _show_perplexity_analysis(self, sender=None, app_data=None):
        """Compare perplexity between models"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import torch.nn.functional as F

        prompt = dpg.get_value("weight_func_test_prompt")
        pretrained_model = self._weight_analysis_data['pretrained_model']
        trained_model = self._weight_analysis_data['trained_model']
        tokenizer = self._weight_analysis_data['tokenizer']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrained_model = pretrained_model.to(device).eval()
        trained_model = trained_model.to(device).eval()

        # Test prompts
        test_prompts = [
            prompt,
            "The relationship between mind and body",
            "In the beginning there was",
            "What is the meaning of existence",
            "The future of artificial intelligence"
        ]

        results = "PERPLEXITY ANALYSIS\n"
        results += "=" * 60 + "\n\n"
        results += f"{'Prompt':<40} {'Pretrained':>10} {'Trained':>10}\n"
        results += "-" * 60 + "\n"

        pre_perps = []
        train_perps = []

        for test_prompt in test_prompts:
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                # Pretrained perplexity
                pre_out = pretrained_model(**inputs, labels=inputs.input_ids)
                pre_perp = torch.exp(pre_out.loss).item()
                pre_perps.append(pre_perp)

                # Trained perplexity
                train_out = trained_model(**inputs, labels=inputs.input_ids)
                train_perp = torch.exp(train_out.loss).item()
                train_perps.append(train_perp)

            short_prompt = test_prompt[:39]
            results += f"{short_prompt:<40} {pre_perp:>10.2f} {train_perp:>10.2f}\n"

        results += "-" * 60 + "\n"
        results += f"{'AVERAGE':<40} {sum(pre_perps)/len(pre_perps):>10.2f} {sum(train_perps)/len(train_perps):>10.2f}\n"

        if sum(train_perps) < sum(pre_perps):
            results += "\nTrained model has LOWER perplexity (more confident)\n"
        else:
            results += "\nTrained model has HIGHER perplexity (less confident)\n"

        dpg.set_value("functional_results", results)

        # Plot
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(name='Pretrained', x=[p[:20] for p in test_prompts], y=pre_perps, marker_color='#2196F3'),
            go.Bar(name='Trained', x=[p[:20] for p in test_prompts], y=train_perps, marker_color='#4CAF50')
        ])
        fig.update_layout(
            title="Perplexity Comparison",
            xaxis_title="Prompt",
            yaxis_title="Perplexity",
            template="plotly_dark",
            barmode='group'
        )
        self.show_plot_in_browser(fig)

    def _show_attention_patterns(self, sender=None, app_data=None):
        """Compare attention patterns between models"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import torch
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        prompt = dpg.get_value("weight_func_test_prompt")
        pretrained_model = self._weight_analysis_data['pretrained_model']
        trained_model = self._weight_analysis_data['trained_model']
        tokenizer = self._weight_analysis_data['tokenizer']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrained_model = pretrained_model.to(device).eval()
        trained_model = trained_model.to(device).eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        results = "ATTENTION PATTERN ANALYSIS\n"
        results += "=" * 60 + "\n\n"
        results += f"Prompt: {prompt}\n"
        results += f"Tokens: {tokens}\n\n"

        with torch.no_grad():
            pre_out = pretrained_model(**inputs, output_attentions=True)
            train_out = trained_model(**inputs, output_attentions=True)

        # Compare attention patterns
        pre_attn = pre_out.attentions  # tuple of [batch, heads, seq, seq]
        train_attn = train_out.attentions

        attn_diffs = []
        for layer_idx, (pre_a, train_a) in enumerate(zip(pre_attn, train_attn)):
            diff = (train_a - pre_a).abs().mean().item()
            attn_diffs.append(diff)
            results += f"Layer {layer_idx}: Attention diff = {diff:.6f}\n"

        results += f"\nMost changed layer: {np.argmax(attn_diffs)} (diff={max(attn_diffs):.6f})\n"

        dpg.set_value("functional_results", results)

        # Visualize attention for most changed layer
        most_changed_layer = np.argmax(attn_diffs)

        pre_attention = pre_attn[most_changed_layer][0, 0].cpu().numpy()  # First head
        train_attention = train_attn[most_changed_layer][0, 0].cpu().numpy()

        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=(f"Pretrained (Layer {most_changed_layer})",
                                         f"Trained (Layer {most_changed_layer})"))

        fig.add_trace(go.Heatmap(z=pre_attention, x=tokens, y=tokens, colorscale='Blues'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=train_attention, x=tokens, y=tokens, colorscale='Greens'), row=1, col=2)

        fig.update_layout(
            title=f"Attention Patterns - Layer {most_changed_layer} Head 0",
            template="plotly_dark"
        )
        self.show_plot_in_browser(fig)

    def _export_weight_report(self, sender=None, app_data=None):
        """Export full weight analysis report"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        import json
        from datetime import datetime

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self._weight_analysis_data['model_name'],
            'layer_deltas': {k: v for k, v in self._weight_analysis_data['layer_deltas'].items()}
        }

        report_path = self.base_dir / "weight_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"Report exported to {report_path}")

    def _open_all_weight_plots(self, sender=None, app_data=None):
        """Open all weight analysis plots"""
        if not hasattr(self, '_weight_analysis_data'):
            self.log("Run analysis first")
            return

        self._show_layer_magnitudes()
        self._show_weight_distributions()
        self._show_layer_depth_analysis()
        self.log("Opened all plots")

    def _get_available_models(self):
        """Get list of available HuggingFace models"""
        models = [
            "gpt2", "distilgpt2", "gpt2-medium", "gpt2-large",
            "EleutherAI/gpt-neo-125m", "EleutherAI/gpt-neo-1.3B",
            "facebook/opt-125m", "facebook/opt-350m",
            "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium",
        ]
        try:
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for model_dir in cache_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("models--"):
                        model_name = model_dir.name.replace("models--", "").replace("--", "/")
                        if model_name not in models:
                            models.append(model_name)
        except Exception:
            pass
        return models

    def _get_saved_custom_configs(self):
        """Get list of saved custom configurations"""
        configs = []
        custom_dir = self.base_dir / "configs" / "custom"
        if custom_dir.exists():
            for f in custom_dir.glob("*.yaml"):
                configs.append(f.stem)
        return configs if configs else ["(none)"]

    def _get_experiments_with_checkpoints(self):
        """Get list of experiments that have checkpoint files"""
        experiments = []
        if self.experiments_dir.exists():
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir():
                    checkpoint_dir = exp_dir / "checkpoints"
                    if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pt")):
                        experiments.append(exp_dir.name)
        return experiments if experiments else ["(none)"]

    def _get_checkpoints_for_experiment(self, experiment_name: str):
        """Get list of checkpoints for a specific experiment"""
        checkpoints = []
        checkpoint_dir = self.experiments_dir / experiment_name / "checkpoints"
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
                checkpoints.append(f.stem)
        return checkpoints if checkpoints else ["(none)"]

    def _on_resume_toggle(self, sender, app_data):
        """Handle resume checkbox toggle"""
        enabled = app_data
        dpg.configure_item("resume_options_group", show=enabled)

        if enabled:
            # Update starting from text
            exp = dpg.get_value("custom_resume_experiment")
            ckpt = dpg.get_value("custom_resume_checkpoint")
            if exp != "(none)" and ckpt != "(none)":
                dpg.set_value("custom_starting_from", f"Starting from: Checkpoint ({ckpt})")
                dpg.configure_item("custom_starting_from", color=(255, 180, 80))
            else:
                dpg.set_value("custom_starting_from", "Starting from: Select checkpoint...")
                dpg.configure_item("custom_starting_from", color=(255, 180, 80))
        else:
            dpg.set_value("custom_starting_from", "Starting from: Pretrained (fresh weights)")
            dpg.configure_item("custom_starting_from", color=(80, 200, 120))

    def _on_resume_experiment_changed(self, sender, app_data):
        """Handle experiment selection change for resume"""
        experiment_name = app_data
        if experiment_name and experiment_name != "(none)":
            checkpoints = self._get_checkpoints_for_experiment(experiment_name)
            dpg.configure_item("custom_resume_checkpoint", items=checkpoints)
            if checkpoints and checkpoints[0] != "(none)":
                dpg.set_value("custom_resume_checkpoint", checkpoints[-1])  # Default to latest
                dpg.set_value("custom_starting_from", f"Starting from: Checkpoint ({checkpoints[-1]})")
            else:
                dpg.set_value("custom_resume_checkpoint", "(none)")
                dpg.set_value("custom_starting_from", "Starting from: No checkpoints found")
        else:
            dpg.configure_item("custom_resume_checkpoint", items=["(none)"])
            dpg.set_value("custom_resume_checkpoint", "(none)")

    def _preview_custom_config(self):
        """Generate and preview the custom config YAML"""
        config = self._generate_custom_config_yaml()
        dpg.set_value("custom_config_preview", config)
        self.log("Config preview generated")

    def _generate_custom_config_yaml(self):
        """Generate YAML content from current settings"""
        name = dpg.get_value("custom_config_name")
        agent_a = dpg.get_value("custom_agent_a_model")
        agent_b = dpg.get_value("custom_agent_b_model")
        episodes = dpg.get_value("custom_episodes")
        max_turns = dpg.get_value("custom_max_turns")
        lr = dpg.get_value("custom_lr")
        seed = dpg.get_value("custom_seed")
        reward_type = dpg.get_value("custom_reward_type")
        weight_info = dpg.get_value("custom_weight_info")
        weight_geom = dpg.get_value("custom_weight_geom")
        weight_dyn = dpg.get_value("custom_weight_dyn")
        prompts_raw = dpg.get_value("custom_prompts")
        prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
        max_tokens = dpg.get_value("custom_max_tokens")
        temperature = dpg.get_value("custom_temperature")
        top_p = dpg.get_value("custom_top_p")
        top_k = dpg.get_value("custom_top_k")
        save_interval = dpg.get_value("custom_save_interval")
        checkpoint_interval = dpg.get_value("custom_checkpoint_interval")

        prompts_yaml = "\n".join([f'    - "{p}"' for p in prompts])

        return f'''# Custom Experiment: {name}
experiment:
  name: {name}

agents:
  agent_a:
    type: hf_causal
    model_name: {agent_a}
    device: cuda
    generation:
      max_new_tokens: {max_tokens}
      temperature: {temperature}
      top_p: {top_p}
      top_k: {top_k}
      do_sample: true

  agent_b:
    type: hf_causal
    model_name: {agent_b}
    device: cuda
    generation:
      max_new_tokens: {max_tokens}
      temperature: {temperature}
      top_p: {top_p}
      top_k: {top_k}
      do_sample: true

environment:
  type: freeform
  max_turns: {max_turns}
  prompts:
{prompts_yaml}

training:
  num_episodes: {episodes}
  lr: {lr}
  gamma: 0.99
  value_coef: 0.5
  entropy_coef: 0.01
  checkpoint_interval: {checkpoint_interval}
  save_interval: {save_interval}

reward:
  type: {reward_type}
  weights:
    information: {weight_info}
    geometric: {weight_geom}
    dynamics: {weight_dyn}

seed: {seed}
'''

    def _save_custom_config(self):
        """Save the custom configuration to a YAML file"""
        name = dpg.get_value("custom_config_name")
        if not name:
            self.log("Error: Please enter a config name")
            return

        custom_dir = self.base_dir / "configs" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        main_config_dir = Path(__file__).parent / "configs" / "experiments"
        main_config_dir.mkdir(parents=True, exist_ok=True)

        yaml_content = self._generate_custom_config_yaml()

        custom_path = custom_dir / f"{name}.yaml"
        with open(custom_path, 'w') as f:
            f.write(yaml_content)

        main_path = main_config_dir / f"{name}.yaml"
        with open(main_path, 'w') as f:
            f.write(yaml_content)

        self.log(f"Saved config: {name}")
        self._refresh_saved_configs()
        self._refresh_training_configs()

    def _refresh_saved_configs(self):
        """Refresh the saved configs dropdown"""
        configs = self._get_saved_custom_configs()
        dpg.configure_item("saved_configs_list", items=configs)

    def _refresh_training_configs(self):
        """Refresh the training tab config dropdown"""
        configs = self._get_available_configs()
        dpg.configure_item("training_config", items=configs)

    def _load_custom_config(self):
        """Load a saved custom configuration"""
        selected = dpg.get_value("saved_configs_list")
        if not selected or selected == "(none)":
            self.log("No config selected")
            return

        config_path = self.base_dir / "configs" / "custom" / f"{selected}.yaml"
        if not config_path.exists():
            self.log(f"Config not found: {config_path}")
            return

        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'experiment' in config:
                dpg.set_value("custom_config_name", config['experiment'].get('name', selected))
            if 'agents' in config:
                if 'agent_a' in config['agents']:
                    dpg.set_value("custom_agent_a_model", config['agents']['agent_a'].get('model_name', 'gpt2'))
                if 'agent_b' in config['agents']:
                    dpg.set_value("custom_agent_b_model", config['agents']['agent_b'].get('model_name', 'distilgpt2'))
            if 'training' in config:
                dpg.set_value("custom_episodes", config['training'].get('num_episodes', 10000))
                dpg.set_value("custom_lr", config['training'].get('lr', 0.0001))
            if 'environment' in config:
                dpg.set_value("custom_max_turns", config['environment'].get('max_turns', 20))
                if 'prompts' in config['environment']:
                    dpg.set_value("custom_prompts", "\n".join(config['environment']['prompts']))
            if 'reward' in config:
                dpg.set_value("custom_reward_type", config['reward'].get('type', 'layered'))
            if 'seed' in config:
                dpg.set_value("custom_seed", config['seed'])

            self.log(f"Loaded config: {selected}")
            self._preview_custom_config()
        except Exception as e:
            self.log(f"Error loading config: {e}")

    def _delete_custom_config(self):
        """Delete a saved custom configuration"""
        selected = dpg.get_value("saved_configs_list")
        if not selected or selected == "(none)":
            return

        custom_path = self.base_dir / "configs" / "custom" / f"{selected}.yaml"
        main_path = Path(__file__).parent / "configs" / "experiments" / f"{selected}.yaml"

        try:
            if custom_path.exists():
                custom_path.unlink()
            if main_path.exists():
                main_path.unlink()
            self.log(f"Deleted config: {selected}")
            self._refresh_saved_configs()
            self._refresh_training_configs()
        except Exception as e:
            self.log(f"Error deleting: {e}")


    def _setup_terminal(self):
        """Setup terminal log panel"""
        with dpg.group(horizontal=True):
            dpg.add_text("TERMINAL LOG", color=(50, 100, 180))
            dpg.add_spacer(width=20)
            dpg.add_button(label="Clear", callback=self._clear_terminal, width=60)
            dpg.add_button(label="Copy", callback=self._copy_terminal, width=60)

        dpg.add_separator()
        dpg.add_input_text(multiline=True, readonly=True, tag="terminal_output",
                          width=-1, height=-1, default_value="Emergent Protocols GUI initialized\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════

    def _refresh_experiments(self, sender=None, app_data=None):
        """Refresh experiment list"""
        self.scan_experiments()
        self._update_experiment_list()
        self._update_dashboard()

    def _open_experiments_folder(self, sender=None, app_data=None):
        """Open experiments folder in file explorer"""
        import subprocess
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', str(self.experiments_dir)])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(self.experiments_dir)])
        else:
            subprocess.Popen(['xdg-open', str(self.experiments_dir)])

    def _filter_experiments(self, sender, app_data):
        """Filter experiment list"""
        self._update_experiment_list(filter_term=app_data)

    def _update_experiment_list(self, filter_term: str = ""):
        """Update experiment list display"""
        if dpg.does_item_exist("exp_list_placeholder"):
            dpg.delete_item("exp_list_placeholder")

        for child in dpg.get_item_children("exp_list_container", slot=1):
            dpg.delete_item(child)

        filter_term = filter_term.lower()

        for i, exp in enumerate(self.experiments):
            if filter_term and filter_term not in exp.name.lower():
                continue

            with dpg.group(parent="exp_list_container", tag=f"exp_item_{i}"):
                with dpg.group(horizontal=True):
                    dpg.add_selectable(label=exp.name[:25], tag=f"exp_sel_{i}",
                                       callback=self._on_experiment_select, user_data=i, width=180)
                    if exp.has_embeddings:
                        dpg.add_text("[3D]", color=(60, 160, 80))

                dpg.add_text(f"  Episodes: {exp.episode_count} | Avg: {exp.avg_reward:.2f}",
                            color=(100, 100, 120))
                dpg.add_separator()

    def _on_experiment_select(self, sender, app_data, user_data):
        """Handle experiment selection"""
        idx = user_data
        for i in range(len(self.experiments)):
            if dpg.does_item_exist(f"exp_sel_{i}"):
                dpg.set_value(f"exp_sel_{i}", i == idx)

    def _get_selected_experiment(self) -> Optional[ExperimentInfo]:
        """Get currently selected experiment"""
        for i in range(len(self.experiments)):
            if dpg.does_item_exist(f"exp_sel_{i}") and dpg.get_value(f"exp_sel_{i}"):
                return self.experiments[i]
        return None

    def _load_selected_experiment(self, sender=None, app_data=None):
        """Load selected experiment"""
        exp = self._get_selected_experiment()
        if exp:
            self.current_experiment = exp
            dpg.set_value("status_text", f"Loaded: {exp.name}")
            self._update_dashboard()
            self.log(f"Loaded experiment: {exp.name}")

    def _run_analysis(self, sender=None, app_data=None):
        """Run analysis on current experiment"""
        exp = self.current_experiment or self._get_selected_experiment()
        if exp is None:
            self.log("No experiment selected")
            return

        self.log(f"Starting analysis for {exp.name}...")
        dpg.set_value("status_text", f"Analyzing: {exp.name}")

        def on_complete(result):
            if result:
                self.current_analysis = result
                self._update_analysis_display()
                self.log("Analysis complete!")
            else:
                self.log("Analysis failed")
            dpg.set_value("status_text", f"Loaded: {exp.name}")

        self.run_analysis_async(exp, on_complete)

    def _update_analysis_display(self):
        """Update analysis display"""
        if self.current_analysis is None:
            return

        result = self.current_analysis

        # Dimensional
        dim_text = ""
        if result.dimensional:
            dim_text += f"MLE Intrinsic Dimension: {result.dimensional.get('mle_dimension', 0):.2f}\n"
            dim_text += f"Correlation Dimension: {result.dimensional.get('correlation_dimension', 0):.2f}\n"
            pca = result.dimensional.get('pca', {})
            dim_text += f"PCA Effective Dim (95%): {pca.get('effective_dim_95', 0)}\n"
            dim_text += f"Participation Ratio: {pca.get('participation_ratio', 0):.2f}\n"
        dpg.set_value("analysis_dimensional", dim_text)

        # Information
        info_text = ""
        if result.information:
            info_text += f"Channel Capacity: {result.information.get('channel_capacity', 0):.2f} bits\n"
            info_text += f"Entropy Rate: {result.information.get('entropy_rate', 0):.3f}\n"
            mi = result.information.get('mutual_information', {})
            info_text += f"MI (lag=1): {mi.get('lag1_mean', 0):.3f}\n"
            info_text += f"MI (lag=2): {mi.get('lag2_mean', 0):.3f}\n"
        dpg.set_value("analysis_information", info_text)

        # Geometric
        geom_text = ""
        if result.geometric:
            ricci = result.geometric.get('ricci_curvature', {})
            geom_text += f"Ricci Curvature (mean): {ricci.get('mean', 0):.3f}\n"
            geom_text += f"Ricci Curvature (std): {ricci.get('std', 0):.3f}\n"
            geom_text += f"Fractal Dimension: {result.geometric.get('fractal_dimension', 0):.2f}\n"
            clustering = result.geometric.get('clustering', {})
            geom_text += f"Silhouette Score: {clustering.get('silhouette', 0):.3f}\n"
        dpg.set_value("analysis_geometric", geom_text)

        # Temporal
        temp_text = ""
        if result.temporal:
            vel = result.temporal.get('velocity', {})
            temp_text += f"Mean Velocity: {vel.get('mean', 0):.3f}\n"
            acf = result.temporal.get('autocorrelation', {})
            temp_text += f"Decorrelation Time: {acf.get('decorrelation_time', 0)}\n"
            stat = result.temporal.get('stationarity', {})
            temp_text += f"Stationary: {stat.get('is_stationary', 'N/A')}\n"
        dpg.set_value("analysis_temporal", temp_text)

        # Thesis comparison
        thesis_text = "COMPARISON TO THESIS FINDINGS:\n\n"
        thesis_text += "Metric          | This Experiment | Thesis Finding\n"
        thesis_text += "-" * 55 + "\n"

        mle_dim = result.dimensional.get('mle_dimension', 0) if result.dimensional else 0
        thesis_text += f"Intrinsic Dim   | {mle_dim:15.2f} | 5.5 - 6.5\n"

        ricci_mean = result.geometric.get('ricci_curvature', {}).get('mean', 0) if result.geometric else 0
        thesis_text += f"Ricci Curvature | {ricci_mean:15.3f} | 0.514\n"

        capacity = result.information.get('channel_capacity', 0) if result.information else 0
        thesis_text += f"Channel Capacity| {capacity:15.2f} | 2.71 bits\n"

        fractal = result.geometric.get('fractal_dimension', 0) if result.geometric else 0
        thesis_text += f"Fractal Dim     | {fractal:15.2f} | 2.646\n"

        dpg.set_value("thesis_comparison", thesis_text)

        # Update dashboard
        dpg.set_value("dash_dimension", f"Dimension: {mle_dim:.2f} (target: 5.5-6.5)")
        dpg.set_value("dash_ricci", f"Ricci: {ricci_mean:.3f} (target: 0.514)")
        dpg.set_value("dash_capacity", f"Capacity: {capacity:.2f} bits (target: 2.71)")

    def _update_dashboard(self):
        """Update dashboard display"""
        total_episodes = sum(exp.episode_count for exp in self.experiments)
        dpg.set_value("dash_total_exp", str(len(self.experiments)))
        dpg.set_value("dash_total_episodes", str(total_episodes))

        if self.current_experiment:
            dpg.set_value("dash_current_exp", self.current_experiment.name[:30])
            dpg.set_value("dash_avg_reward", f"{self.current_experiment.avg_reward:.3f}")

    def _get_available_configs(self) -> List[str]:
        """Get available training configs"""
        configs = []
        if self.configs_dir.exists():
            for f in self.configs_dir.glob("*.yaml"):
                configs.append(f.stem)
        return configs if configs else ["baseline_replication"]

    def _start_training_clicked(self, sender=None, app_data=None):
        """Handle start training button"""
        config_name = dpg.get_value("training_config")
        if config_name:
            self.start_training(config_name)

    def _add_to_comparison(self, sender=None, app_data=None):
        """Add experiment to comparison list"""
        exp = self._get_selected_experiment()
        if exp and exp not in self.comparison_experiments:
            self.comparison_experiments.append(exp)
            self._update_comparison_list()
            self.log(f"Added {exp.name} to comparison")

    def _clear_comparison(self, sender=None, app_data=None):
        """Clear comparison list"""
        self.comparison_experiments = []
        self._update_comparison_list()

    def _update_comparison_list(self):
        """Update comparison list display"""
        if dpg.does_item_exist("comparison_placeholder"):
            dpg.delete_item("comparison_placeholder")

        for child in dpg.get_item_children("comparison_list", slot=1):
            dpg.delete_item(child)

        if not self.comparison_experiments:
            dpg.add_text("No experiments selected", tag="comparison_placeholder",
                        parent="comparison_list", color=(140, 140, 150))
            return

        for i, exp in enumerate(self.comparison_experiments):
            with dpg.group(parent="comparison_list", horizontal=True):
                dpg.add_text(f"{i+1}. {exp.name[:25]}", color=(60, 60, 70))
                dpg.add_button(label="X", callback=lambda s, a, e=exp: self._remove_from_comparison(e),
                              width=25)

    def _remove_from_comparison(self, exp: ExperimentInfo):
        """Remove experiment from comparison"""
        if exp in self.comparison_experiments:
            self.comparison_experiments.remove(exp)
            self._update_comparison_list()

    def _show_comparison(self, sender=None, app_data=None):
        """Show comparison results"""
        if len(self.comparison_experiments) < 2:
            self.log("Need at least 2 experiments for comparison")
            return

        self.log("Running comparison analysis...")

        results_text = "COMPARISON RESULTS\n\n"
        results_text += "Experiment".ljust(25) + "Episodes".rjust(10) + "Avg Reward".rjust(12) + "\n"
        results_text += "-" * 50 + "\n"

        for exp in self.comparison_experiments:
            results_text += f"{exp.name[:24].ljust(25)}{str(exp.episode_count).rjust(10)}{exp.avg_reward:12.3f}\n"

        dpg.set_value("comparison_results", results_text)

    def _open_comparison_plot(self, sender=None, app_data=None):
        """Open comparison plot in browser"""
        fig = self.create_comparison_plot(self.comparison_experiments)
        if fig:
            self.show_plot_in_browser(fig)

    def _show_3d_plot(self, sender=None, app_data=None):
        """Show 3D embedding plot"""
        exp = self.current_experiment or self._get_selected_experiment()
        if exp is None:
            self.log("No experiment selected")
            return

        method = dpg.get_value("emb_reduction").lower().replace("-", "")
        color_by = dpg.get_value("emb_color").lower()
        max_points = dpg.get_value("emb_max_points")

        self.log(f"Generating 3D plot for {exp.name}...")
        fig = self.create_embedding_3d_plot(exp, method, color_by, max_points)
        if fig:
            self.show_plot_in_browser(fig)
        else:
            self.log("Could not generate 3D plot")

    def _show_metrics_dashboard(self, sender=None, app_data=None):
        """Show metrics dashboard"""
        exp = self.current_experiment or self._get_selected_experiment()
        if exp is None:
            self.log("No experiment selected")
            return

        fig = self.create_metrics_dashboard(exp)
        if fig:
            self.show_plot_in_browser(fig)

    def _show_dimensional_plot(self, sender=None, app_data=None):
        """Show dimensional analysis plot"""
        if self.current_analysis is None:
            self.log("Run analysis first")
            return

        fig = self.create_dimensional_analysis_plot(self.current_analysis)
        if fig:
            self.show_plot_in_browser(fig)

    def _show_geometric_plot(self, sender=None, app_data=None):
        """Show geometric analysis plot"""
        if self.current_analysis is None:
            self.log("Run analysis first")
            return

        fig = self.create_geometric_analysis_plot(self.current_analysis)
        if fig:
            self.show_plot_in_browser(fig)

    def _export_3d_html(self, sender=None, app_data=None):
        """Export 3D plot as HTML"""
        exp = self.current_experiment or self._get_selected_experiment()
        if exp is None:
            return

        method = dpg.get_value("emb_reduction").lower().replace("-", "")
        color_by = dpg.get_value("emb_color").lower()

        fig = self.create_embedding_3d_plot(exp, method, color_by)
        if fig:
            filepath = self.experiments_dir / f"{exp.name}_3d.html"
            fig.write_html(str(filepath))
            self.log(f"Exported to {filepath}")

    def _export_report(self, sender=None, app_data=None):
        """Export analysis report"""
        if self.current_analysis is None:
            self.log("Run analysis first")
            return

        result = self.current_analysis
        report = f"""# Emergent Protocols Analysis Report

## Experiment: {result.experiment_name}
**Generated:** {result.timestamp}

## Dimensional Analysis

- **MLE Intrinsic Dimension:** {result.dimensional.get('mle_dimension', 0):.2f}
- **Correlation Dimension:** {result.dimensional.get('correlation_dimension', 0):.2f}
- **PCA Effective Dimension (95%):** {result.dimensional.get('pca', {}).get('effective_dim_95', 0)}

## Information Analysis

- **Channel Capacity:** {result.information.get('channel_capacity', 0):.2f} bits
- **Entropy Rate:** {result.information.get('entropy_rate', 0):.3f}

## Geometric Analysis

- **Ricci Curvature (mean):** {result.geometric.get('ricci_curvature', {}).get('mean', 0):.3f}
- **Fractal Dimension:** {result.geometric.get('fractal_dimension', 0):.2f}

## Comparison to Thesis Findings

| Metric | This Experiment | Thesis Finding |
|--------|-----------------|----------------|
| Intrinsic Dimension | {result.dimensional.get('mle_dimension', 0):.2f} | 5.5 - 6.5 |
| Ricci Curvature | {result.geometric.get('ricci_curvature', {}).get('mean', 0):.3f} | 0.514 |
| Channel Capacity | {result.information.get('channel_capacity', 0):.2f} bits | 2.71 bits |
| Fractal Dimension | {result.geometric.get('fractal_dimension', 0):.2f} | 2.646 |

---
*Generated by Emergent Protocols Research GUI*
"""
        filepath = self.experiments_dir / f"{result.experiment_name}_report.md"
        with open(filepath, 'w') as f:
            f.write(report)
        self.log(f"Report exported to {filepath}")

    def _load_episodes_range(self, sender=None, app_data=None):
        """Load episodes in range for browsing"""
        exp = self.current_experiment or self._get_selected_experiment()
        if exp is None:
            self.log("No experiment selected")
            return

        start = dpg.get_value("browse_start")
        end = dpg.get_value("browse_end")

        self.log(f"Loading episodes {start}-{end}...")

        try:
            loader = self.load_experiment_data(exp)
            if loader is None:
                self.log("Failed to load experiment data")
                return

            episode_ids = loader.storage.get_all_episode_ids()
            if not episode_ids:
                self.log("No episodes found")
                return

            # Clamp range
            start_idx = max(0, start)
            end_idx = min(len(episode_ids), end)

            if start_idx >= end_idx:
                self.log(f"Invalid range. Available: 0-{len(episode_ids)}")
                return

            selected_ids = episode_ids[start_idx:end_idx]

            # Build display content
            content_lines = [f"Loaded {len(selected_ids)} episodes ({start_idx}-{end_idx} of {len(episode_ids)})\n"]

            for ep_id in selected_ids[:20]:  # Show first 20
                try:
                    episode = loader.storage.load_episode(ep_id)
                    content_lines.append(f"\n{'='*60}")
                    content_lines.append(f"Episode: {ep_id}")
                    content_lines.append(f"Turns: {len(episode.turns)}")
                    content_lines.append(f"Reward: {episode.metadata.get('total_reward', 0.0):.4f}")
                    content_lines.append("-" * 40)

                    for turn in episode.turns[:5]:  # First 5 turns
                        text_preview = turn.output.text[:100].replace('\n', ' ')
                        if len(turn.output.text) > 100:
                            text_preview += "..."
                        content_lines.append(f"  [{turn.agent_name}]: {text_preview}")

                    if len(episode.turns) > 5:
                        content_lines.append(f"  ... ({len(episode.turns) - 5} more turns)")

                except Exception as e:
                    content_lines.append(f"\nError loading {ep_id}: {e}")

            if len(selected_ids) > 20:
                content_lines.append(f"\n... and {len(selected_ids) - 20} more episodes")

            # Clear existing content
            for child in dpg.get_item_children("data_browser_content", slot=1):
                dpg.delete_item(child)

            # Add new content
            dpg.add_text("\n".join(content_lines), parent="data_browser_content", wrap=700)
            self.log(f"Loaded {len(selected_ids)} episodes")

        except Exception as e:
            self.log(f"Error loading episodes: {e}")
            traceback.print_exc()

    def _clear_terminal(self, sender=None, app_data=None):
        """Clear terminal log"""
        self.log_handler.lines = []
        dpg.set_value("terminal_output", "")

    def _copy_terminal(self, sender=None, app_data=None):
        """Copy terminal contents"""
        import pyperclip
        try:
            pyperclip.copy(self.log_handler.get_all())
            self.log("Copied to clipboard")
        except:
            pass

    def _show_about(self, sender=None, app_data=None):
        """Show about dialog"""
        with dpg.window(label="About", width=450, height=350, modal=True,
                       tag="about_popup", on_close=lambda: dpg.delete_item("about_popup")):
            dpg.add_text("COUPLED DYNAMICS REPLICATION SUITE", color=(50, 100, 180))
            dpg.add_separator()
            dpg.add_text("""
Diagnostic platform for investigating the geometric properties of 
Large Language Model (LLM) representation manifolds.

Based on the research:
"Knowledge Distillation Preserves Representational Geometry: 
Evidence from Coupled Transformer Dynamics and Attractor Analysis"

Core Capabilities:
- Coupled Dynamics Testing: Direct hidden state exchange.
- Geometric Analysis: Lyapunov exponents and attractor dimensions.
- Model Compatibility Validation: Weight merging and stitching evaluation.
- Dimensionality Reduction: PCA/t-SNE/UMAP visualization of manifolds.

Version 1.0 (Academic Replication)
            """, wrap=430)

    def _update_terminal(self):
        """Update terminal with new log messages"""
        new_lines = self.log_handler.get_new()
        if new_lines:
            current = dpg.get_value("terminal_output")
            new_text = current + "\n".join(new_lines) + "\n"
            # Keep last 500 lines
            lines = new_text.split("\n")
            if len(lines) > 500:
                lines = lines[-500:]
            dpg.set_value("terminal_output", "\n".join(lines))

    def _update_training_display(self):
        """Update training display"""
        if self.training_state.is_running:
            progress = self.training_state.current_episode / max(1, self.training_state.total_episodes)
            dpg.set_value("training_progress", progress)
            dpg.set_value("training_progress_text",
                         f"{self.training_state.current_episode} / {self.training_state.total_episodes} episodes")
            dpg.set_value("live_reward", f"{self.training_state.current_reward:.3f}")
            dpg.set_value("live_avg_reward", f"{self.training_state.avg_reward:.3f}")
            dpg.set_value("training_status", "Training...")
            dpg.set_value("training_status_detail", f"Episode {self.training_state.current_episode}")

            # Update plot (use set_value to avoid flickering)
            if len(self.training_state.rewards_history) > 1:
                dpg.set_value("reward_line_series", [
                    list(range(len(self.training_state.rewards_history))),
                    self.training_state.rewards_history
                ])
                dpg.fit_axis_data("train_x")
                dpg.fit_axis_data("train_y")
        else:
            dpg.set_value("training_status", "")
            dpg.set_value("training_status_detail", "Idle")

    def _update_performance_monitors(self):
        """Update CPU, RAM, GPU stats"""
        current_time = time.time()
        if current_time - self._perf_last_update < self._perf_update_interval:
            return
        self._perf_last_update = current_time

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_color = (80, 200, 120) if cpu_percent < 70 else (200, 150, 50) if cpu_percent < 90 else (200, 60, 60)
            dpg.set_value("perf_cpu", f"{cpu_percent:4.0f}%")
            dpg.configure_item("perf_cpu", color=cpu_color)

            # RAM usage
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_percent = ram.percent
            ram_color = (80, 200, 120) if ram_percent < 70 else (200, 150, 50) if ram_percent < 90 else (200, 60, 60)
            dpg.set_value("perf_ram", f"{ram_used_gb:.1f}G")
            dpg.configure_item("perf_ram", color=ram_color)

            # GPU stats (if available)
            if HAS_TORCH and torch.cuda.is_available():
                try:
                    gpu_util = 0
                    vram_used = 0
                    vram_total = 0

                    # Try to get GPU utilization via nvidia-smi through torch
                    device = torch.cuda.current_device()
                    vram_used = torch.cuda.memory_allocated(device) / (1024**3)
                    vram_reserved = torch.cuda.memory_reserved(device) / (1024**3)

                    # Get total VRAM
                    props = torch.cuda.get_device_properties(device)
                    vram_total = props.total_memory / (1024**3)

                    vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
                    vram_color = (80, 200, 120) if vram_percent < 70 else (200, 150, 50) if vram_percent < 90 else (200, 60, 60)

                    dpg.set_value("perf_gpu", "ON")
                    dpg.configure_item("perf_gpu", color=(80, 200, 120))
                    dpg.set_value("perf_vram", f"{vram_used:.1f}G")
                    dpg.configure_item("perf_vram", color=vram_color)
                except Exception:
                    dpg.set_value("perf_gpu", "ERR")
                    dpg.set_value("perf_vram", "---")
            else:
                dpg.set_value("perf_gpu", "N/A")
                dpg.configure_item("perf_gpu", color=(100, 100, 120))
                dpg.set_value("perf_vram", "N/A")
                dpg.configure_item("perf_vram", color=(100, 100, 120))

        except Exception:
            pass  # Don't crash on monitoring errors

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self):
        """Main application loop"""
        dpg.create_context()

        # Setup
        self.setup_theme()
        self.setup_windows()

        # Initial data load
        self.scan_experiments()
        self._update_experiment_list()
        # self._update_dashboard()

        # Viewport
        dpg.create_viewport(title="Emergent Protocols Research GUI", width=1600, height=1000)
        dpg.setup_dearpygui()
        dpg.set_primary_window("primary_window", True)

        # Show and run
        dpg.show_viewport()

        # Main loop with updates
        while dpg.is_dearpygui_running():
            self._update_terminal()
            # self._update_training_display()
            self._update_performance_monitors()
            # self._check_for_plot_updates()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()




# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gui = EmergentProtocolsGUI()
    gui.run()
