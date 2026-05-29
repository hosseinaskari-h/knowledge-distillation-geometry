"""
Checkpointing utilities
"""
import torch
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading checkpoints"""

    def __init__(self,
                 checkpoint_dir: Path,
                 max_checkpoints: int = 5,
                 keep_best: bool = True):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Whether to always keep the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best

        self.checkpoints: List[Dict] = []
        self.best_metric = None
        self.best_checkpoint = None

        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """Load existing checkpoint history"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_metric = data.get('best_metric')
                self.best_checkpoint = data.get('best_checkpoint')

    def _save_checkpoint_history(self):
        """Save checkpoint history"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'checkpoints': self.checkpoints,
                'best_metric': self.best_metric,
                'best_checkpoint': self.best_checkpoint,
            }, f, indent=2)

    def save(self,
             state: Dict[str, Any],
             episode: int,
             metric: Optional[float] = None,
             is_best: bool = False):
        """
        Save checkpoint

        Args:
            state: State dict to save
            episode: Episode number
            metric: Optional metric value for tracking best
            is_best: Force marking as best
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_ep{episode:06d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename

        # Add metadata
        state['_checkpoint_meta'] = {
            'episode': episode,
            'timestamp': timestamp,
            'metric': metric,
        }

        # Save
        torch.save(state, filepath)
        logger.info(f"Saved checkpoint: {filename}")

        # Update history
        checkpoint_info = {
            'filename': filename,
            'episode': episode,
            'timestamp': timestamp,
            'metric': metric,
        }
        self.checkpoints.append(checkpoint_info)

        # Check if best
        if metric is not None:
            if self.best_metric is None or metric > self.best_metric or is_best:
                self.best_metric = metric
                self.best_checkpoint = filename

                # Create symlink/copy for best
                best_path = self.checkpoint_dir / "best.pt"
                if best_path.exists():
                    best_path.unlink()
                shutil.copy(filepath, best_path)
                logger.info(f"New best checkpoint: {metric:.4f}")

        # Cleanup old checkpoints
        self._cleanup()
        self._save_checkpoint_history()

    def _cleanup(self):
        """Remove old checkpoints keeping max_checkpoints"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by episode (oldest first)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x['episode'])

        # Determine which to remove
        to_remove = []
        for cp in sorted_checkpoints[:-self.max_checkpoints]:
            if self.keep_best and cp['filename'] == self.best_checkpoint:
                continue
            to_remove.append(cp)

        # Remove files and update list
        for cp in to_remove:
            filepath = self.checkpoint_dir / cp['filename']
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Removed old checkpoint: {cp['filename']}")
            self.checkpoints.remove(cp)

    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint

        Args:
            filename: Specific filename to load, or None for latest

        Returns:
            Loaded state dict
        """
        if filename is None:
            # Load latest
            if not self.checkpoints:
                raise ValueError("No checkpoints available")
            filename = self.checkpoints[-1]['filename']

        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        state = torch.load(filepath)
        logger.info(f"Loaded checkpoint: {filename}")

        return state

    def load_best(self) -> Dict[str, Any]:
        """Load best checkpoint"""
        if self.best_checkpoint is None:
            raise ValueError("No best checkpoint available")

        return self.load(self.best_checkpoint)

    def get_latest_episode(self) -> Optional[int]:
        """Get episode number of latest checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]['episode']

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist"""
        return len(self.checkpoints) > 0
