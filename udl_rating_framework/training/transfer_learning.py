"""
CTM-Aware Transfer Learning for UDL Rating Framework.

Implements transfer learning that leverages CTM's unique temporal processing
and synchronization mechanisms. Focuses on transferring learned temporal
dynamics, neuron-level patterns, and synchronization strategies rather than
just static representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        BertModel,
        GPT2Model,
        RobertaModel,
        PreTrainedModel,
        PreTrainedTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary
from ..core.representation import UDLRepresentation
from .training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class CTMTransferConfig:
    """
    Configuration for CTM-aware transfer learning.

    Focuses on transferring temporal dynamics and synchronization patterns
    rather than just static representations.
    """

    # Source CTM configuration
    source_model_path: Optional[str] = None
    freeze_source_ctm: bool = True

    # Transfer strategy
    transfer_method: str = "temporal_dynamics"  # 'temporal_dynamics', 'synchronization_patterns', 'neuron_level_models'

    # Temporal transfer configuration
    transfer_iterations: bool = True  # Transfer learned iteration patterns
    transfer_memory_patterns: bool = True  # Transfer neuron-level memory patterns
    transfer_synchronization: bool = True  # Transfer synchronization strategies

    # Fine-tuning configuration
    fine_tune_nlms: bool = False  # Whether to fine-tune neuron-level models
    fine_tune_synapses: bool = True  # Whether to fine-tune synapse models
    fine_tune_synchronization: bool = (
        True  # Whether to fine-tune synchronization parameters
    )

    # Training configuration
    adaptation_epochs: int = 15
    fine_tune_epochs: int = 25
    adaptation_lr: float = 1e-4
    fine_tune_lr: float = 1e-5

    # CTM-specific parameters
    preserve_temporal_structure: bool = True  # Maintain temporal processing structure
    adapt_memory_length: bool = False  # Whether to adapt memory length for new task

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pretrained_model_name": self.pretrained_model_name,
            "freeze_pretrained": self.freeze_pretrained,
            "fine_tune_layers": self.fine_tune_layers,
            "adaptation_method": self.adaptation_method,
            "adapter_dim": self.adapter_dim,
            "projection_dim": self.projection_dim,
            "pretrain_epochs": self.pretrain_epochs,
            "fine_tune_epochs": self.fine_tune_epochs,
            "pretrain_lr": self.pretrain_lr,
            "fine_tune_lr": self.fine_tune_lr,
        }


class AdapterLayer(nn.Module):
    """
    Adapter layer for efficient transfer learning.

    Implements the adapter architecture from "Parameter-Efficient Transfer Learning for NLP"
    """

    def __init__(self, input_dim: int, adapter_dim: int, dropout: float = 0.1):
        """
        Initialize adapter layer.

        Args:
            input_dim: Input dimension
            adapter_dim: Adapter bottleneck dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Initialize weights to near-zero for residual connection
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        residual = x

        # Adapter transformation
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)

        # Residual connection
        return residual + x


class PretrainedFeatureExtractor(nn.Module):
    """
    Feature extractor using pre-trained language models.

    Extracts contextualized representations from pre-trained models
    and adapts them for UDL rating tasks.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze_pretrained: bool = True,
        fine_tune_layers: int = 2,
        adaptation_method: str = "feature_extraction",
        adapter_dim: int = 64,
        projection_dim: int = 256,
    ):
        """
        Initialize pre-trained feature extractor.

        Args:
            model_name: Name of pre-trained model
            freeze_pretrained: Whether to freeze pre-trained weights
            fine_tune_layers: Number of top layers to fine-tune
            adaptation_method: Method for adaptation
            adapter_dim: Adapter dimension
            projection_dim: Output projection dimension
        """
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for transfer learning. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.adaptation_method = adaptation_method

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze pre-trained parameters
        if freeze_pretrained:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

            # Unfreeze top layers for fine-tuning
            if fine_tune_layers > 0:
                if hasattr(self.pretrained_model, "encoder"):
                    # BERT-like models
                    layers = self.pretrained_model.encoder.layer
                    for layer in layers[-fine_tune_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                elif hasattr(self.pretrained_model, "transformer"):
                    # GPT-like models
                    layers = self.pretrained_model.transformer.h
                    for layer in layers[-fine_tune_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True

        # Get hidden dimension
        self.hidden_dim = self.pretrained_model.config.hidden_size

        # Adaptation layers
        if adaptation_method == "adapter":
            # Add adapter layers
            self.adapters = nn.ModuleList(
                [
                    AdapterLayer(self.hidden_dim, adapter_dim)
                    for _ in range(fine_tune_layers)
                ]
            )

        # Projection layer to match CTM input dimension
        self.projection = nn.Linear(self.hidden_dim, projection_dim)

        logger.info(f"Initialized feature extractor with {model_name}")
        logger.info(f"Hidden dim: {self.hidden_dim}, Projection dim: {projection_dim}")

    def tokenize_udl(
        self, udl_text: str, max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize UDL text using pre-trained tokenizer.

        Args:
            udl_text: UDL source text
            max_length: Maximum sequence length

        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            udl_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from pre-trained model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Extracted features [batch, seq_len, projection_dim]
        """
        # Get pre-trained model outputs
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Apply adapters if using adapter method
        if self.adaptation_method == "adapter" and hasattr(self, "adapters"):
            for adapter in self.adapters:
                hidden_states = adapter(hidden_states)

        # Project to target dimension
        features = self.projection(hidden_states)  # [batch, seq_len, projection_dim]

        return features


class CTMTransferLearningModel(nn.Module):
    """
    CTM model with transfer learning that leverages temporal dynamics.

    Transfers learned temporal processing patterns, synchronization strategies,
    and neuron-level model behaviors between tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        transfer_config: CTMTransferConfig,
        ctm_config: Dict[str, Any],
    ):
        """
        Initialize transfer learning CTM.

        Args:
            vocab_size: Size of UDL vocabulary
            transfer_config: Transfer learning configuration
            ctm_config: CTM model configuration
        """
        super().__init__()

        self.transfer_config = transfer_config
        self.vocab_size = vocab_size

        # Pre-trained feature extractor
        self.feature_extractor = PretrainedFeatureExtractor(
            model_name=transfer_config.pretrained_model_name,
            freeze_pretrained=transfer_config.freeze_pretrained,
            fine_tune_layers=transfer_config.fine_tune_layers,
            adaptation_method=transfer_config.adaptation_method,
            adapter_dim=transfer_config.adapter_dim,
            projection_dim=transfer_config.projection_dim,
        )

        # Update CTM config to use projected features
        ctm_config = ctm_config.copy()
        ctm_config["d_input"] = transfer_config.projection_dim

        # CTM model (without embedding layer since we use pre-trained features)
        self.ctm = UDLRatingCTM(vocab_size=vocab_size, **ctm_config)

        # Replace CTM embedding with identity (we use pre-trained features)
        self.ctm.embedding = nn.Identity()

        logger.info("Initialized transfer learning CTM")

    def forward(
        self, udl_texts: List[str], max_length: int = 512, track: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """
        Forward pass with transfer learning.

        Args:
            udl_texts: List of UDL source texts
            max_length: Maximum sequence length
            track: Whether to return tracking information

        Returns:
            Tuple of (ratings, certainties, synch_out, tracking_data)
        """
        device = next(self.parameters()).device

        # Tokenize UDL texts
        batch_inputs = []
        for text in udl_texts:
            inputs = self.feature_extractor.tokenize_udl(text, max_length)
            batch_inputs.append(inputs)

        # Stack batch inputs
        input_ids = torch.stack(
            [inputs["input_ids"].squeeze(0) for inputs in batch_inputs]
        ).to(device)
        attention_mask = torch.stack(
            [inputs["attention_mask"].squeeze(0) for inputs in batch_inputs]
        ).to(device)

        # Extract pre-trained features
        features = self.feature_extractor(
            input_ids, attention_mask
        )  # [batch, seq_len, d_input]

        # Process through CTM (bypass embedding)
        return self.ctm.forward(features, track=track)

    def forward_with_token_ids(
        self, token_ids: torch.Tensor, track: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """
        Forward pass with token IDs (for compatibility).

        Args:
            token_ids: Token IDs [batch, seq_len]
            track: Whether to return tracking information

        Returns:
            Tuple of (ratings, certainties, synch_out, tracking_data)
        """
        # This is a fallback method - in practice, you'd convert token_ids back to text
        # For now, just use the original CTM forward pass
        return self.ctm.forward(token_ids, track=track)


class CTMTransferLearningTrainer:
    """
    CTM-aware transfer learning trainer.

    Implements transfer learning that leverages CTM's unique temporal
    processing and synchronization mechanisms.
    """

    def __init__(
        self,
        vocab_size: int,
        transfer_config: CTMTransferConfig,
        ctm_config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize transfer learning trainer.

        Args:
            vocab_size: Size of UDL vocabulary
            transfer_config: Transfer learning configuration
            ctm_config: CTM configuration
            device: Device for training
        """
        self.vocab_size = vocab_size
        self.transfer_config = transfer_config
        self.ctm_config = ctm_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create model
        self.model = CTMTransferLearningModel(vocab_size, transfer_config, ctm_config)
        self.model.to(self.device)

        logger.info(f"Initialized transfer learning trainer with device: {self.device}")

    def pretrain_on_language_modeling(
        self, text_corpus: List[str], num_epochs: int = 10, batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        Pre-train on language modeling task.

        Args:
            text_corpus: Corpus of text for pre-training
            num_epochs: Number of pre-training epochs
            batch_size: Batch size

        Returns:
            Pre-training history
        """
        logger.info(f"Starting language modeling pre-training for {num_epochs} epochs")

        # Create optimizer for pre-training
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.transfer_config.pretrain_lr
        )

        # Language modeling loss
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.model.feature_extractor.tokenizer.pad_token_id
        )

        history = {"pretrain_loss": []}

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            # Process corpus in batches
            for i in range(0, len(text_corpus), batch_size):
                batch_texts = text_corpus[i : i + batch_size]

                # Tokenize batch
                inputs = self.model.feature_extractor.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass through feature extractor only
                features = self.model.feature_extractor(
                    inputs["input_ids"], inputs["attention_mask"]
                )

                # Language modeling loss (predict next token)
                # This is simplified - in practice, you'd implement proper MLM or CLM
                shift_features = features[:, :-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()

                # Project features to vocabulary size for prediction
                vocab_proj = nn.Linear(features.size(-1), self.vocab_size).to(
                    self.device
                )
                logits = vocab_proj(shift_features)

                loss = criterion(
                    logits.view(-1, self.vocab_size), shift_labels.view(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            history["pretrain_loss"].append(avg_loss)

            logger.info(
                f"Pre-training epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}"
            )

        logger.info("Completed language modeling pre-training")
        return history

    def fine_tune_on_udl_rating(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics: List[Any],
        aggregator: Any,
        num_epochs: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune on UDL rating task.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            metrics: Quality metrics
            aggregator: Metric aggregator
            num_epochs: Number of fine-tuning epochs

        Returns:
            Fine-tuning history
        """
        logger.info(f"Starting UDL rating fine-tuning for {num_epochs} epochs")

        # Create training pipeline
        pipeline = TrainingPipeline(
            model=self.model,
            metrics=metrics,
            aggregator=aggregator,
            learning_rate=self.transfer_config.fine_tune_lr,
            device=self.device,
        )

        # Fine-tune model
        history = pipeline.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
        )

        logger.info("Completed UDL rating fine-tuning")
        return history

    def train_with_transfer_learning(
        self,
        text_corpus: Optional[List[str]],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics: List[Any],
        aggregator: Any,
    ) -> Dict[str, Any]:
        """
        Complete transfer learning training pipeline.

        Args:
            text_corpus: Optional text corpus for pre-training
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            metrics: Quality metrics
            aggregator: Metric aggregator

        Returns:
            Complete training history
        """
        complete_history = {}

        # Stage 1: Pre-training (optional)
        if text_corpus:
            pretrain_history = self.pretrain_on_language_modeling(
                text_corpus, self.transfer_config.pretrain_epochs
            )
            complete_history.update(pretrain_history)

        # Stage 2: Fine-tuning
        finetune_history = self.fine_tune_on_udl_rating(
            train_dataloader,
            val_dataloader,
            metrics,
            aggregator,
            self.transfer_config.fine_tune_epochs,
        )
        complete_history.update(finetune_history)

        return complete_history

    def save_model(self, filepath: str):
        """
        Save transfer learning model.

        Args:
            filepath: Path to save model
        """
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "transfer_config": self.transfer_config.to_dict(),
            "ctm_config": self.ctm_config,
            "vocab_size": self.vocab_size,
        }

        torch.save(save_data, filepath)
        logger.info(f"Saved transfer learning model to {filepath}")

    @classmethod
    def load_model(
        cls, filepath: str, device: Optional[torch.device] = None
    ) -> "CTMTransferLearningTrainer":
        """
        Load transfer learning model.

        Args:
            filepath: Path to model file
            device: Device for loading

        Returns:
            Loaded trainer
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        transfer_config = CTMTransferConfig(**checkpoint["transfer_config"])

        trainer = cls(
            vocab_size=checkpoint["vocab_size"],
            transfer_config=transfer_config,
            ctm_config=checkpoint["ctm_config"],
            device=device,
        )

        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded transfer learning model from {filepath}")
        return trainer


def create_ctm_transfer_learning_model(
    vocab_size: int,
    source_model_path: Optional[str] = None,
    ctm_config: Optional[Dict[str, Any]] = None,
    transfer_config: Optional[CTMTransferConfig] = None,
) -> CTMTransferLearningModel:
    """
    Convenience function to create transfer learning model.

    Args:
        vocab_size: Size of UDL vocabulary
        pretrained_model_name: Name of pre-trained model
        ctm_config: CTM configuration
        transfer_config: Transfer learning configuration

    Returns:
        Transfer learning CTM model
    """
    if transfer_config is None:
        transfer_config = CTMTransferConfig(source_model_path=source_model_path)

    if ctm_config is None:
        ctm_config = {"d_model": 256, "iterations": 20, "n_synch_out": 32, "heads": 8}

    return CTMTransferLearningModel(vocab_size, transfer_config, ctm_config)
