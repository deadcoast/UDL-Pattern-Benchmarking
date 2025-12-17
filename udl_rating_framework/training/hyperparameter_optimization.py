"""
Hyperparameter Optimization for CTM Training.

Provides automated hyperparameter tuning using various optimization strategies
including grid search, random search, and Bayesian optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .training_pipeline import TrainingPipeline, UDLDataset
from ..models.ctm_adapter import UDLRatingCTM, create_udl_rating_model
from ..core.metrics.base import QualityMetric
from ..core.aggregation import MetricAggregator

logger = logging.getLogger(__name__)


@dataclass
class CTMHyperparameterSpace:
    """
    Defines the CTM-specific hyperparameter search space.
    
    Focuses on parameters unique to Continuous Thought Machines:
    - Synchronization mechanisms
    - Neuron-level model configurations
    - Temporal processing parameters
    - Memory and attention dynamics
    """
    # Core CTM architecture parameters
    d_model: Any = field(default_factory=lambda: [128, 256, 512])
    d_input: Any = field(default_factory=lambda: [32, 64, 128])
    iterations: Any = field(default_factory=lambda: [10, 15, 20, 25, 30])  # Temporal processing depth
    
    # Synchronization parameters (CTM-specific)
    n_synch_out: Any = field(default_factory=lambda: [16, 32, 64, 128])
    n_synch_action: Any = field(default_factory=lambda: [8, 16, 32, 64])
    neuron_select_type: Any = field(default_factory=lambda: ['first-last', 'random', 'random-pairing'])
    n_random_pairing_self: Any = field(default_factory=lambda: [0, 1, 2, 3, 4])  # For random-pairing
    
    # Neuron-level model parameters (CTM-specific)
    memory_length: Any = field(default_factory=lambda: [3, 5, 10, 15, 20])  # NLM history length
    deep_nlms: Any = field(default_factory=lambda: [True, False])  # Deep vs shallow NLMs
    memory_hidden_dims: Any = field(default_factory=lambda: [64, 128, 256])  # NLM hidden dimensions
    do_layernorm_nlm: Any = field(default_factory=lambda: [False])  # Usually False for CTM
    
    # Synapse and processing parameters
    synapse_depth: Any = field(default_factory=lambda: [2, 3, 4, 5])  # U-Net depth
    heads: Any = field(default_factory=lambda: [4, 8, 16])  # Attention heads
    
    # Regularization
    dropout: Any = field(default_factory=lambda: (0.0, 0.3))
    dropout_nlm: Any = field(default_factory=lambda: (0.0, 0.2))  # NLM-specific dropout
    
    # Training parameters
    learning_rate: Any = field(default_factory=lambda: (1e-5, 1e-2))
    batch_size: Any = field(default_factory=lambda: [8, 16, 32])
    alpha: Any = field(default_factory=lambda: (0.5, 0.9))  # Loss weighting
    beta: Any = field(default_factory=lambda: (0.1, 0.5))   # Loss weighting
    weight_decay: Any = field(default_factory=lambda: (1e-6, 1e-3))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterSpace':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationResult:
    """
    Results from hyperparameter optimization.
    """
    best_params: Dict[str, Any]
    best_score: float
    best_model_path: Optional[str]
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    optimization_time: float
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model_path': self.best_model_path,
            'optimization_history': self.optimization_history,
            'total_trials': self.total_trials,
            'optimization_time': self.optimization_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)


class CTMHyperparameterOptimizer:
    """
    CTM-aware hyperparameter optimizer.
    
    Optimizes CTM-specific parameters with understanding of their interactions:
    - Synchronization parameter relationships
    - Memory length vs iteration count trade-offs
    - Neuron selection strategy impacts
    - Temporal processing optimization
    """
    
    def __init__(self,
                 vocab_size: int,
                 metrics: List[QualityMetric],
                 aggregator: MetricAggregator,
                 param_space: CTMHyperparameterSpace,
                 device: Optional[torch.device] = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            vocab_size: Size of token vocabulary
            metrics: List of quality metrics
            aggregator: Metric aggregator
            param_space: Hyperparameter search space
            device: Device for training
        """
        self.vocab_size = vocab_size
        self.metrics = metrics
        self.aggregator = aggregator
        self.param_space = param_space
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimization_history = []
        
        logger.info(f"Initialized hyperparameter optimizer with device: {self.device}")
    
    def _create_model_and_pipeline(self, params: Dict[str, Any]) -> Tuple[UDLRatingCTM, TrainingPipeline]:
        """
        Create model and training pipeline with given parameters.
        
        Args:
            params: Hyperparameter dictionary
            
        Returns:
            Tuple of (model, training_pipeline)
        """
        # Extract model parameters
        model_params = {
            'vocab_size': self.vocab_size,
            'd_model': params['d_model'],
            'd_input': params['d_input'],
            'iterations': params['iterations'],
            'n_synch_out': params['n_synch_out'],
            'heads': params['heads'],
            'n_synch_action': params['n_synch_action'],
            'synapse_depth': params['synapse_depth'],
            'memory_length': params['memory_length'],
            'memory_hidden_dims': params['memory_hidden_dims'],
            'dropout': params['dropout']
        }
        
        # Create model
        model = create_udl_rating_model(**model_params)
        
        # Extract training parameters
        training_params = {
            'model': model,
            'metrics': self.metrics,
            'aggregator': self.aggregator,
            'alpha': params['alpha'],
            'beta': params.get('beta', 1.0 - params['alpha']),
            'learning_rate': params['learning_rate'],
            'device': self.device
        }
        
        # Create training pipeline
        pipeline = TrainingPipeline(**training_params)
        
        # Set weight decay if specified
        if 'weight_decay' in params:
            pipeline.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        
        return model, pipeline
    
    def _evaluate_hyperparameters(self,
                                  params: Dict[str, Any],
                                  train_dataloader: DataLoader,
                                  val_dataloader: DataLoader,
                                  num_epochs: int = 20) -> float:
        """
        Evaluate a set of hyperparameters.
        
        Args:
            params: Hyperparameter dictionary
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Validation score (higher is better)
        """
        try:
            # Create model and pipeline
            model, pipeline = self._create_model_and_pipeline(params)
            
            # Train model
            history = pipeline.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=num_epochs,
                checkpoint_dir=None,  # Don't save checkpoints during optimization
                save_every=num_epochs + 1  # Never save
            )
            
            # Return best validation correlation (higher is better)
            if history['val_correlation']:
                score = max(history['val_correlation'])
            else:
                # Fallback to negative loss if no validation correlation
                score = -min(history['val_loss'])
            
            # Record trial
            trial_result = {
                'params': params.copy(),
                'score': score,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'best_val_correlation': score if history['val_correlation'] else None
            }
            
            self.optimization_history.append(trial_result)
            
            logger.info(f"Trial completed: Score={score:.4f}, Params={params}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating hyperparameters {params}: {e}")
            # Return very low score for failed trials
            return -1000.0
    
    def grid_search(self,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    num_epochs: int = 20,
                    max_trials: Optional[int] = None) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs per trial
            max_trials: Maximum number of trials (None for exhaustive)
            
        Returns:
            Optimization results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting grid search optimization")
        
        # Generate all parameter combinations
        param_names = []
        param_values = []
        
        for name, value in self.param_space.to_dict().items():
            param_names.append(name)
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (int, float)):
                # Continuous parameter - sample discrete values
                if isinstance(value[0], int):
                    param_values.append(list(range(value[0], value[1] + 1)))
                else:
                    param_values.append(np.linspace(value[0], value[1], 5).tolist())
            elif isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))
        
        if max_trials and len(all_combinations) > max_trials:
            # Randomly sample combinations
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_trials]
        
        logger.info(f"Evaluating {len(all_combinations)} parameter combinations")
        
        best_score = -float('inf')
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            # Ensure alpha + beta = 1 if both are specified
            if 'alpha' in params and 'beta' in params:
                total = params['alpha'] + params['beta']
                params['alpha'] = params['alpha'] / total
                params['beta'] = params['beta'] / total
            
            score = self._evaluate_hyperparameters(params, train_dataloader, val_dataloader, num_epochs)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            logger.info(f"Grid search progress: {i+1}/{len(all_combinations)}, "
                       f"Current best score: {best_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model_path=None,
            optimization_history=self.optimization_history.copy(),
            total_trials=len(all_combinations),
            optimization_time=optimization_time
        )
        
        logger.info(f"Grid search completed in {optimization_time:.2f}s. "
                   f"Best score: {best_score:.4f}")
        
        return result
    
    def random_search(self,
                      train_dataloader: DataLoader,
                      val_dataloader: DataLoader,
                      num_trials: int = 50,
                      num_epochs: int = 20) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_trials: Number of random trials
            num_epochs: Number of training epochs per trial
            
        Returns:
            Optimization results
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting random search optimization with {num_trials} trials")
        
        best_score = -float('inf')
        best_params = None
        
        for trial in range(num_trials):
            # Sample random parameters
            params = {}
            
            for name, value in self.param_space.to_dict().items():
                if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (int, float)):
                    # Continuous parameter - check if it's a valid range
                    if value[0] < value[1]:
                        if isinstance(value[0], int):
                            params[name] = np.random.randint(value[0], value[1] + 1)
                        else:
                            params[name] = np.random.uniform(value[0], value[1])
                    else:
                        # If range is invalid (low >= high), use the first value
                        params[name] = value[0]
                elif isinstance(value, list):
                    params[name] = np.random.choice(value)
                else:
                    params[name] = value
            
            # Ensure alpha + beta = 1
            if 'alpha' in params and 'beta' in params:
                total = params['alpha'] + params['beta']
                params['alpha'] = params['alpha'] / total
                params['beta'] = params['beta'] / total
            
            score = self._evaluate_hyperparameters(params, train_dataloader, val_dataloader, num_epochs)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            logger.info(f"Random search progress: {trial+1}/{num_trials}, "
                       f"Current best score: {best_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model_path=None,
            optimization_history=self.optimization_history.copy(),
            total_trials=num_trials,
            optimization_time=optimization_time
        )
        
        logger.info(f"Random search completed in {optimization_time:.2f}s. "
                   f"Best score: {best_score:.4f}")
        
        return result
    
    def bayesian_optimization(self,
                             train_dataloader: DataLoader,
                             val_dataloader: DataLoader,
                             num_trials: int = 100,
                             num_epochs: int = 20) -> OptimizationResult:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_trials: Number of optimization trials
            num_epochs: Number of training epochs per trial
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. "
                            "Install with: pip install optuna")
        
        import time
        start_time = time.time()
        
        logger.info(f"Starting Bayesian optimization with {num_trials} trials")
        
        def objective(trial):
            # Sample parameters using Optuna
            params = {}
            
            for name, value in self.param_space.to_dict().items():
                if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (int, float)):
                    # Continuous parameter
                    if isinstance(value[0], int):
                        params[name] = trial.suggest_int(name, value[0], value[1])
                    else:
                        params[name] = trial.suggest_float(name, value[0], value[1])
                elif isinstance(value, list):
                    params[name] = trial.suggest_categorical(name, value)
                else:
                    params[name] = value
            
            # Ensure alpha + beta = 1
            if 'alpha' in params and 'beta' in params:
                total = params['alpha'] + params['beta']
                params['alpha'] = params['alpha'] / total
                params['beta'] = params['beta'] / total
            
            return self._evaluate_hyperparameters(params, train_dataloader, val_dataloader, num_epochs)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trials)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_model_path=None,
            optimization_history=self.optimization_history.copy(),
            total_trials=num_trials,
            optimization_time=optimization_time
        )
        
        logger.info(f"Bayesian optimization completed in {optimization_time:.2f}s. "
                   f"Best score: {study.best_value:.4f}")
        
        return result
    
    def optimize(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 method: str = 'random',
                 num_trials: int = 50,
                 num_epochs: int = 20,
                 **kwargs) -> OptimizationResult:
        """
        Perform hyperparameter optimization.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            method: Optimization method ('grid', 'random', 'bayesian')
            num_trials: Number of trials
            num_epochs: Number of training epochs per trial
            **kwargs: Additional method-specific arguments
            
        Returns:
            Optimization results
        """
        if method == 'grid':
            return self.grid_search(train_dataloader, val_dataloader, num_epochs, num_trials)
        elif method == 'random':
            return self.random_search(train_dataloader, val_dataloader, num_trials, num_epochs)
        elif method == 'bayesian':
            return self.bayesian_optimization(train_dataloader, val_dataloader, num_trials, num_epochs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")


def create_ctm_param_space() -> CTMHyperparameterSpace:
    """
    Create CTM-specific hyperparameter search space.
    
    Returns:
        CTM-focused hyperparameter space
    """
    return CTMHyperparameterSpace()


def create_focused_ctm_param_space(focus: str = 'synchronization') -> CTMHyperparameterSpace:
    """
    Create focused CTM hyperparameter space for specific aspects.
    
    Args:
        focus: Focus area ('synchronization', 'temporal', 'memory', 'all')
        
    Returns:
        Focused CTM hyperparameter space
    """
    if focus == 'synchronization':
        return CTMHyperparameterSpace(
            # Focus on synchronization parameters
            n_synch_out=[16, 32, 64, 128],
            n_synch_action=[8, 16, 32, 64],
            neuron_select_type=['first-last', 'random', 'random-pairing'],
            n_random_pairing_self=[0, 1, 2, 3, 4, 5],
            # Keep other parameters more constrained
            d_model=[256],
            iterations=[20],
            memory_length=[10]
        )
    elif focus == 'temporal':
        return CTMHyperparameterSpace(
            # Focus on temporal processing
            iterations=[10, 15, 20, 25, 30, 35],
            memory_length=[3, 5, 10, 15, 20, 25],
            deep_nlms=[True, False],
            # Keep synchronization parameters constrained
            n_synch_out=[32],
            n_synch_action=[16],
            neuron_select_type=['random-pairing']
        )
    elif focus == 'memory':
        return CTMHyperparameterSpace(
            # Focus on neuron-level model memory
            memory_length=[3, 5, 8, 10, 15, 20, 25],
            memory_hidden_dims=[32, 64, 128, 256, 512],
            deep_nlms=[True, False],
            do_layernorm_nlm=[False, True],
            # Keep other parameters constrained
            iterations=[20],
            n_synch_out=[32]
        )
    else:  # 'all'
        return CTMHyperparameterSpace()


def optimize_ctm_hyperparameters(vocab_size: int,
                                 metrics: List[QualityMetric],
                                 aggregator: MetricAggregator,
                                 train_dataloader: DataLoader,
                                 val_dataloader: DataLoader,
                                 method: str = 'random',
                                 num_trials: int = 50,
                                 param_space: Optional[CTMHyperparameterSpace] = None,
                                 **kwargs) -> OptimizationResult:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        vocab_size: Size of token vocabulary
        metrics: List of quality metrics
        aggregator: Metric aggregator
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        method: Optimization method
        num_trials: Number of trials
        param_space: Hyperparameter search space (default if None)
        **kwargs: Additional arguments
        
    Returns:
        Optimization results
    """
    if param_space is None:
        param_space = create_ctm_param_space()
    
    optimizer = CTMHyperparameterOptimizer(
        vocab_size=vocab_size,
        metrics=metrics,
        aggregator=aggregator,
        param_space=param_space
    )
    
    return optimizer.optimize(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        method=method,
        num_trials=num_trials,
        **kwargs
    )