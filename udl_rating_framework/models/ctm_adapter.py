"""
CTM Model Adapter for UDL Rating.

Adapts the Continuous Thought Machine architecture for UDL quality prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import sys
import os

# Add the models directory to the path to import CTM
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from ctm import ContinuousThoughtMachine
from ..core.representation import UDLRepresentation, Token, TokenType


class UDLTokenVocabulary:
    """
    Vocabulary for UDL tokens.
    
    Maps tokens to integer indices for embedding lookup.
    """
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0
        
        # Add special tokens
        self._add_token('<PAD>')  # Padding token
        self._add_token('<UNK>')  # Unknown token
        self._add_token('<BOS>')  # Beginning of sequence
        self._add_token('<EOS>')  # End of sequence
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary."""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]
    
    def add_tokens_from_udl(self, udl: UDLRepresentation):
        """Add all tokens from a UDL to the vocabulary."""
        for token in udl.get_tokens():
            if token.type != TokenType.EOF:
                self._add_token(token.text)
    
    def token_to_index(self, token: str) -> int:
        """Convert token to index."""
        return self.token_to_id.get(token, self.token_to_id['<UNK>'])
    
    def index_to_token(self, index: int) -> str:
        """Convert index to token."""
        return self.id_to_token.get(index, '<UNK>')
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)


class UDLRatingCTM(nn.Module):
    """
    CTM model adapted for UDL quality prediction.
    
    Architecture:
    1. Token Embedding: E: Token → ℝᵈ
    2. CTM Core: Processes sequence with T iterations
    3. Synchronization: Extracts S(t) at each iteration
    4. Rating Head: Maps final S(T) → [0,1]
    
    Mathematical Definition:
    Given a UDL represented as token sequence (t₁, t₂, ..., tₙ):
    1. Embed: xᵢ = E(tᵢ) ∈ ℝᵈ
    2. Process: S(T) = CTM(x₁, x₂, ..., xₙ)
    3. Rate: q = σ(W·S(T) + b) ∈ [0,1]
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 d_input: int = 64,
                 iterations: int = 20,
                 n_synch_out: int = 32,
                 heads: int = 8,
                 n_synch_action: int = 16,
                 synapse_depth: int = 3,
                 memory_length: int = 10,
                 deep_nlms: bool = True,
                 memory_hidden_dims: int = 128,
                 do_layernorm_nlm: bool = False,
                 dropout: float = 0.1,
                 neuron_select_type: str = 'random-pairing',
                 n_random_pairing_self: int = 0,
                 **ctm_kwargs):
        """
        Initialize CTM for UDL rating.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Core dimensionality of CTM latent space
            d_input: Dimensionality of projected attention outputs
            iterations: Number of internal 'thought' ticks
            n_synch_out: Number of neurons for output synchronization
            heads: Number of attention heads
            n_synch_action: Number of neurons for action synchronization
            synapse_depth: Depth of synapse model
            memory_length: History length for Neuron-Level Models
            deep_nlms: Use deeper NLMs if True
            memory_hidden_dims: Hidden dimension size for deep NLMs
            do_layernorm_nlm: Apply LayerNorm within NLMs
            dropout: Dropout rate
            neuron_select_type: Neuron selection strategy
            n_random_pairing_self: Number of self-pairing neurons
            **ctm_kwargs: Additional CTM arguments
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_input = d_input
        self.iterations = iterations
        self.n_synch_out = n_synch_out
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_input, padding_idx=0)
        
        # Initialize CTM with appropriate parameters
        self.ctm = ContinuousThoughtMachine(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='none',  # No visual backbone for text
            positional_embedding_type='none',  # No positional embedding when no backbone
            out_dims=1,  # Single quality score output
            prediction_reshaper=[-1],  # For certainty computation
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            **ctm_kwargs
        )
        
        # Rating head that maps synchronization to quality score
        self.rating_head = nn.Linear(n_synch_out, 1)
        
        # Sigmoid activation to ensure output in [0,1]
        self.sigmoid = nn.Sigmoid()
        
        # Store parameters for property testing
        self.embedding_dim = d_input
    
    def forward(self, token_ids: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            token_ids: [batch, seq_len] token indices
            track: Whether to return tracking information
        
        Returns:
            ratings: [batch, 1] quality scores in [0,1]
            certainties: [batch, 2] certainty scores
            synch_out: [batch, n_synch_out] synchronization representation (if track=True)
        """
        # Embed tokens: [batch, seq_len] -> [batch, seq_len, d_input]
        x = self.embedding(token_ids)
        
        # CTM expects 4D input for visual processing, but we have text
        # We need to bypass the visual processing and go directly to the recurrent loop
        # Let's create a custom forward pass for text
        
        B, seq_len, d_input = x.shape
        device = x.device
        
        # Initialize CTM state
        state_trace = self.ctm.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.ctm.start_activated_state.unsqueeze(0).expand(B, -1)
        
        # Prepare storage for outputs
        predictions = torch.empty(B, self.ctm.out_dims, self.ctm.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.ctm.iterations, device=device, dtype=torch.float32)
        
        # Initialize synchronization parameters
        decay_alpha_action, decay_beta_action = None, None
        self.ctm.decay_params_action.data = torch.clamp(self.ctm.decay_params_action, 0, 15)
        self.ctm.decay_params_out.data = torch.clamp(self.ctm.decay_params_out, 0, 15)
        r_action = torch.exp(-self.ctm.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.ctm.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        _, decay_alpha_out, decay_beta_out = self.ctm.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        
        # Prepare text features for attention (treat as key-value pairs)
        kv = x  # [batch, seq_len, d_input]
        
        # Tracking variables
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            attention_tracking = []
        
        # Recurrent loop
        for stepi in range(self.ctm.iterations):
            # Calculate synchronization for input data interaction
            synchronisation_action, decay_alpha_action, decay_beta_action = self.ctm.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # Interact with data via attention
            q = self.ctm.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.ctm.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            
            # Apply synapses
            state = self.ctm.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            
            # Apply neuron-level models
            activated_state = self.ctm.trace_processor(state_trace)
            
            # Calculate synchronization for output predictions
            synchronisation_out, decay_alpha_out, decay_beta_out = self.ctm.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Get predictions and certainties
            current_prediction = self.ctm.output_projector(synchronisation_out)
            current_certainty = self.ctm.compute_certainty(current_prediction)
            
            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty
            
            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
        
        # Extract final prediction: [batch, 1, T] -> [batch, 1]
        final_pred = predictions[:, :, -1]
        final_cert = certainties[:, :, -1]
        
        # Apply sigmoid to ensure [0,1] range
        ratings = self.sigmoid(final_pred)
        
        if track:
            synch_tracking = (np.array(synch_out_tracking), np.array(synch_action_tracking))
            return predictions, certainties, synch_tracking, np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        else:
            return ratings, final_cert, synchronisation_out
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension for property testing."""
        return self.embedding_dim
    
    def get_synch_out_dim(self) -> int:
        """Return synchronization output dimension."""
        return self.n_synch_out
    
    def tokenize_udl(self, udl: UDLRepresentation, vocab: UDLTokenVocabulary, max_length: int = 512) -> torch.Tensor:
        """
        Convert UDL to token indices.
        
        Args:
            udl: UDL representation
            vocab: Token vocabulary
            max_length: Maximum sequence length
            
        Returns:
            Token indices tensor [seq_len]
        """
        tokens = udl.get_tokens()
        
        # Convert to indices
        indices = [vocab.token_to_index('<BOS>')]
        for token in tokens:
            if token.type != TokenType.EOF and len(indices) < max_length - 1:
                indices.append(vocab.token_to_index(token.text))
        indices.append(vocab.token_to_index('<EOS>'))
        
        # Pad or truncate to max_length
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([vocab.token_to_index('<PAD>')] * (max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)


def create_udl_rating_model(vocab_size: int, **kwargs) -> UDLRatingCTM:
    """
    Factory function to create UDL rating model with default parameters.
    
    Args:
        vocab_size: Size of token vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        Initialized UDLRatingCTM model
    """
    default_params = {
        'd_model': 256,
        'd_input': 64,
        'iterations': 20,
        'n_synch_out': 32,
        'heads': 8,
        'n_synch_action': 16,
        'synapse_depth': 3,
        'memory_length': 10,
        'deep_nlms': True,
        'memory_hidden_dims': 128,
        'dropout': 0.1,
        'neuron_select_type': 'random-pairing',
    }
    
    # Override defaults with provided kwargs
    default_params.update(kwargs)
    
    return UDLRatingCTM(vocab_size=vocab_size, **default_params)