"""
HuggingFace transformers agent implementation
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import transformers.utils.import_utils
import transformers.modeling_utils
from typing import Optional, Tuple
import logging

# MONKEYPATCH: Disable PyTorch version check for pickle vulnerability
# The user explicitly requested to ignore this security check for local offline testing.
if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.core.agent import AgentOutput
from .base import BaseAgent

logger = logging.getLogger(__name__)


class HuggingFaceAgent(BaseAgent):
    """Agent using HuggingFace transformers"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        model_name = config['model']
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading model {model_name} for agent {name}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            ).to(device)
        except ValueError as e:
            # Fallback for encoder models (BERT, DistilBERT)
            if "Unrecognized configuration class" in str(e):
                logger.info(f"AutoModelForCausalLM failed for {model_name}, trying AutoModel (likely encoder model)")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                ).to(device)
            else:
                raise e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._device = device
        self.model.eval()

        # Generation config
        self.gen_config = config.get('generation', {})
        self.save_hidden_states = config.get('save_hidden_states', True)
        self.save_attentions = config.get('save_attentions', False)

        logger.info(f"Agent {name} ready on {device}")

    @property
    def device(self) -> str:
        return self._device

    def generate(self, prompt: str, **kwargs) -> AgentOutput:
        """Generate response to prompt"""
        max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - self.gen_config.get('max_new_tokens', 50)
        ).to(self._device)

        gen_kwargs = {
            **self.gen_config,
            **kwargs,
            'output_hidden_states': self.save_hidden_states,
            'output_attentions': self.save_attentions,
            'return_dict_in_generate': True,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode generated text
        generated_ids = outputs.sequences[0]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract embeddings from last hidden state
        if self.save_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # hidden_states is tuple of tuples: (step_0, step_1, ...)
            # Each step is tuple: (layer_0, layer_1, ...)
            # We want last step, last layer
            last_step_states = outputs.hidden_states[-1]  # Last generation step
            last_layer = last_step_states[-1]  # Last layer
            embeddings = last_layer[0, -1, :]  # [hidden_dim]
        else:
            # Fallback: get embeddings by forward pass
            embeddings = self._get_embeddings_fallback(generated_ids)

        self._increment_generation()

        return AgentOutput(
            text=text,
            tokens=generated_ids,
            embeddings=embeddings,
            logits=outputs.scores if hasattr(outputs, 'scores') else None,
            hidden_states=outputs.hidden_states if self.save_hidden_states else None,
            attentions=outputs.attentions if self.save_attentions else None,
            metadata={
                'model': self.config['model'],
                'generation_count': self.generation_count,
            }
        )

    def generate_with_log_prob(self, prompt: str, **kwargs) -> Tuple[AgentOutput, torch.Tensor]:
        """Generate with log probability for RL"""
        max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - self.gen_config.get('max_new_tokens', 50)
        ).to(self._device)

        gen_kwargs = {
            **self.gen_config,
            **kwargs,
            'output_hidden_states': self.save_hidden_states,
            'output_attentions': self.save_attentions,
            'return_dict_in_generate': True,
            'output_scores': True,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Calculate log probabilities
        generated_ids = outputs.sequences[0]
        scores = outputs.scores  # tuple of [batch, vocab_size]

        # Get log probs of generated tokens
        log_probs = []
        input_len = len(inputs.input_ids[0])
        for i, score in enumerate(scores):
            if input_len + i < len(generated_ids):
                token_id = generated_ids[input_len + i]
                log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                log_probs.append(log_prob)

        if len(log_probs) > 0:
            total_log_prob = torch.stack(log_probs).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=self._device)

        # Create output
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.save_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_step_states = outputs.hidden_states[-1]
            last_layer = last_step_states[-1]
            embeddings = last_layer[0, -1, :]
        else:
            embeddings = self._get_embeddings_fallback(generated_ids)

        self._increment_generation()

        output = AgentOutput(
            text=text,
            tokens=generated_ids,
            embeddings=embeddings,
            logits=scores[-1] if scores else None,
            hidden_states=outputs.hidden_states if self.save_hidden_states else None,
            attentions=outputs.attentions if self.save_attentions else None,
            metadata={
                'model': self.config['model'],
                'generation_count': self.generation_count,
            }
        )

        return output, total_log_prob

    def _get_embeddings_fallback(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings via forward pass (fallback)"""
        with torch.no_grad():
            outputs = self.model(token_ids.unsqueeze(0), output_hidden_states=True)
            # Last layer, last token
            return outputs.hidden_states[-1][0, -1, :]

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings without generation"""
        max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self._device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Return last layer, last token
            return outputs.hidden_states[-1][0, -1, :]

    def save_checkpoint(self, path: str):
        """Save agent state"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'generation_count': self.generation_count,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint for {self.name} to {path}")

    def load_checkpoint(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self._device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.generation_count = checkpoint.get('generation_count', 0)
        logger.info(f"Loaded checkpoint for {self.name} from {path}")

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embedding space"""
        return self.model.config.hidden_size

    # =========================================================================
    # BRANCH 2: Pure Vector Dynamics Methods
    # =========================================================================

    def forward_from_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Pass a hidden state through transformer layers WITHOUT tokenization.

        Implementation of coupled dynamics: direct hidden state exchange.
        Bypasses tokenization/decoding bottleneck.

        Args:
            hidden_state: [hidden_dim] or [1, hidden_dim] or [1, seq_len, hidden_dim]

        Returns:
            output_state: [hidden_dim] - the transformed hidden state
        """
        # Ensure proper shape: [batch, seq_len, hidden_dim]
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.to(self._device)

        # Match the model's dtype (important for mixed precision)
        model_dtype = next(self.model.parameters()).dtype
        hidden_state = hidden_state.to(dtype=model_dtype)

        with torch.no_grad():
            # Access transformer blocks directly (works for GPT-2, GPT-Neo, OPT, etc.)
            if hasattr(self.model, 'transformer'):
                # GPT-2 has .h, DistilBERT has .layer
                if hasattr(self.model.transformer, 'h'):
                    # GPT-2 style
                    blocks = self.model.transformer.h
                    ln_f = self.model.transformer.ln_f
                elif hasattr(self.model.transformer, 'layer'):
                    # DistilBERT style (when loaded as AutoModel)
                    blocks = self.model.transformer.layer
                    ln_f = lambda x: x # DistilBERT doesn't expose final LN in the same way
                else:
                    raise NotImplementedError(f"Unknown transformer structure for {type(self.model)}")
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                # OPT style
                blocks = self.model.model.decoder.layers
                ln_f = self.model.model.decoder.final_layer_norm
            elif hasattr(self.model, 'gpt_neox'):
                # GPT-NeoX style
                blocks = self.model.gpt_neox.layers
                ln_f = self.model.gpt_neox.final_layer_norm
            elif hasattr(self.model, 'bert'):
                # BERT style
                blocks = self.model.bert.encoder.layer
                ln_f = lambda x: x
            elif hasattr(self.model, 'distilbert'):
                # DistilBERT style (wrapped)
                blocks = self.model.distilbert.transformer.layer
                ln_f = lambda x: x
            elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                 # Generic encoder (BERT-like when loaded as AutoModel might just have .encoder)
                 blocks = self.model.encoder.layer
                 ln_f = lambda x: x
            else:
                raise NotImplementedError(f"Model architecture not supported for forward_from_hidden: {type(self.model)}")

            # Pass through each transformer block
            hidden = hidden_state
            for block in blocks:
                # Most blocks return tuple (hidden_states, ...)
                block_output = block(hidden)
                if isinstance(block_output, tuple):
                    hidden = block_output[0]
                else:
                    hidden = block_output

            # Apply final layer norm
            hidden = ln_f(hidden)

        # Return last position, squeezed to [hidden_dim]
        return hidden[0, -1, :]

    def get_random_hidden_state(self) -> torch.Tensor:
        """
        Generate a random hidden state for initializing vector dynamics.

        Samples from a distribution similar to actual model activations.
        """
        # Use a standard normal scaled to typical activation magnitudes
        hidden = torch.randn(self.embedding_dim, device=self._device) * 0.1
        return hidden

    def get_hidden_from_text(self, text: str) -> torch.Tensor:
        """
        Get initial hidden state from text (useful for seeding dynamics).

        Bridges text-based initialization with vector dynamics evolution.
        """
        return self.get_embeddings(text)
