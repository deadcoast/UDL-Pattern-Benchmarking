"""
Active Learning for UDL Rating Framework.

Implements active learning strategies to improve training data selection
and model performance with minimal labeling effort.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary
from ..core.representation import UDLRepresentation
from .training_pipeline import TrainingPipeline, UDLDataset

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """
    Configuration for active learning.
    """
    # Query strategy
    query_strategy: str = 'uncertainty_sampling'  # 'uncertainty_sampling', 'diversity_sampling', 'hybrid'
    
    # Sampling parameters
    initial_pool_size: int = 100
    query_batch_size: int = 20
    max_iterations: int = 10
    
    # Uncertainty sampling parameters
    uncertainty_method: str = 'entropy'  # 'entropy', 'margin', 'least_confident'
    
    # Diversity sampling parameters
    diversity_method: str = 'kmeans'  # 'kmeans', 'core_set', 'random'
    n_clusters: int = 10
    
    # Hybrid parameters
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3
    
    # Training parameters
    retrain_epochs: int = 10
    validation_split: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_strategy': self.query_strategy,
            'initial_pool_size': self.initial_pool_size,
            'query_batch_size': self.query_batch_size,
            'max_iterations': self.max_iterations,
            'uncertainty_method': self.uncertainty_method,
            'diversity_method': self.diversity_method,
            'n_clusters': self.n_clusters,
            'uncertainty_weight': self.uncertainty_weight,
            'diversity_weight': self.diversity_weight,
            'retrain_epochs': self.retrain_epochs,
            'validation_split': self.validation_split
        }


class QueryStrategy(ABC):
    """
    Abstract base class for active learning query strategies.
    """
    
    @abstractmethod
    def select_samples(self,
                      model: UDLRatingCTM,
                      unlabeled_pool: Dataset,
                      labeled_indices: List[int],
                      n_samples: int) -> List[int]:
        """
        Select samples for labeling.
        
        Args:
            model: Trained model
            unlabeled_pool: Pool of unlabeled samples
            labeled_indices: Indices of already labeled samples
            n_samples: Number of samples to select
            
        Returns:
            List of selected sample indices
        """
        pass


class CTMUncertaintySampling(QueryStrategy):
    """
    CTM-aware uncertainty sampling strategy.
    
    Selects samples based on CTM-specific uncertainty signals:
    - Synchronization instability
    - Neuron activation diversity
    - Temporal processing uncertainty
    """
    
    def __init__(self, 
                 method: str = 'synchronization_entropy', 
                 device: Optional[torch.device] = None,
                 use_temporal_analysis: bool = True):
        """
        Initialize CTM uncertainty sampling.
        
        Args:
            method: CTM uncertainty method ('synchronization_entropy', 'neuron_diversity', 'temporal_instability')
            device: Device for computation
            use_temporal_analysis: Whether to analyze temporal dynamics
        """
        self.method = method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_temporal_analysis = use_temporal_analysis
    
    def select_samples(self,
                      model: UDLRatingCTM,
                      unlabeled_pool: Dataset,
                      labeled_indices: List[int],
                      n_samples: int) -> List[int]:
        """
        Select samples with highest CTM-specific uncertainty.
        
        Args:
            model: Trained CTM model
            unlabeled_pool: Pool of unlabeled samples
            labeled_indices: Indices of already labeled samples
            n_samples: Number of samples to select
            
        Returns:
            List of selected sample indices
        """
        model.eval()
        
        # Get unlabeled indices
        all_indices = set(range(len(unlabeled_pool)))
        labeled_set = set(labeled_indices)
        unlabeled_indices = list(all_indices - labeled_set)
        
        if len(unlabeled_indices) <= n_samples:
            return unlabeled_indices
        
        # Compute CTM-specific uncertainties for unlabeled samples
        uncertainties = []
        
        with torch.no_grad():
            for idx in unlabeled_indices:
                token_ids, _ = unlabeled_pool[idx]
                token_ids = token_ids.unsqueeze(0).to(self.device)
                
                # Get model predictions with tracking
                predictions, certainties, synch_out, tracking_data = model(token_ids, track=self.use_temporal_analysis)
                
                # Compute CTM-specific uncertainty
                if self.method == 'synchronization_entropy':
                    uncertainty = self._compute_synchronization_uncertainty(synch_out)
                    
                elif self.method == 'neuron_diversity':
                    uncertainty = self._compute_neuron_diversity_uncertainty(tracking_data)
                    
                elif self.method == 'temporal_instability':
                    uncertainty = self._compute_temporal_uncertainty(tracking_data)
                    
                else:
                    # Fallback to traditional entropy
                    pred_prob = predictions.squeeze().item()
                    probs = torch.tensor([pred_prob, 1 - pred_prob])
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    uncertainty = entropy.item()
                
                uncertainties.append((idx, uncertainty))
        
        # Sort by uncertainty (descending) and select top n_samples
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in uncertainties[:n_samples]]
        
        logger.info(f"Selected {len(selected_indices)} samples using CTM {self.method} uncertainty sampling")
        return selected_indices
    
    def _compute_synchronization_uncertainty(self, synch_out: torch.Tensor) -> float:
        """
        Compute uncertainty based on synchronization entropy.
        
        Args:
            synch_out: Synchronization output [batch, synch_dim]
            
        Returns:
            Synchronization-based uncertainty score
        """
        if synch_out is None:
            return 0.5  # Neutral uncertainty
        
        # Normalize synchronization values to probabilities
        synch_probs = torch.softmax(synch_out.flatten(), dim=0)
        
        # Compute Shannon entropy
        entropy = -torch.sum(synch_probs * torch.log(synch_probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = torch.log(torch.tensor(len(synch_probs), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy.item()
    
    def _compute_neuron_diversity_uncertainty(self, tracking_data) -> float:
        """
        Compute uncertainty based on neuron activation diversity.
        
        Args:
            tracking_data: CTM tracking data
            
        Returns:
            Neuron diversity-based uncertainty score
        """
        if tracking_data is None:
            return 0.5  # Neutral uncertainty
        
        # Get final neuron activations
        post_activations = tracking_data.post_activations  # [iterations, batch, neurons]
        final_activations = post_activations[-1, 0, :]  # [neurons] - final iteration, first batch
        
        # Compute activation entropy across neurons
        activation_probs = torch.softmax(torch.tensor(final_activations), dim=0)
        neuron_entropy = -torch.sum(activation_probs * torch.log(activation_probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(final_activations))
        normalized_entropy = neuron_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy.item()
    
    def _compute_temporal_uncertainty(self, tracking_data) -> float:
        """
        Compute uncertainty based on temporal instability.
        
        Args:
            tracking_data: CTM tracking data
            
        Returns:
            Temporal instability-based uncertainty score
        """
        if tracking_data is None:
            return 0.5  # Neutral uncertainty
        
        # Analyze synchronization stability over time
        synch_out = tracking_data.synch_out  # [iterations, batch, synch_dim]
        
        if synch_out.shape[0] <= 1:
            return 0.0  # No temporal variation
        
        # Compute temporal variance (instability)
        temporal_variance = np.var(synch_out, axis=0)  # [batch, synch_dim]
        mean_instability = np.mean(temporal_variance)
        
        # Normalize instability to [0, 1] range
        # Higher instability = higher uncertainty
        normalized_instability = min(1.0, mean_instability)
        
        return normalized_instability


class DiversitySampling(QueryStrategy):
    """
    Diversity sampling strategy.
    
    Selects diverse samples to maximize coverage of the feature space.
    """
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 10, device: Optional[torch.device] = None):
        """
        Initialize diversity sampling.
        
        Args:
            method: Diversity method ('kmeans', 'core_set', 'random')
            n_clusters: Number of clusters for k-means
            device: Device for computation
        """
        self.method = method
        self.n_clusters = n_clusters
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def select_samples(self,
                      model: UDLRatingCTM,
                      unlabeled_pool: Dataset,
                      labeled_indices: List[int],
                      n_samples: int) -> List[int]:
        """
        Select diverse samples.
        
        Args:
            model: Trained model
            unlabeled_pool: Pool of unlabeled samples
            labeled_indices: Indices of already labeled samples
            n_samples: Number of samples to select
            
        Returns:
            List of selected sample indices
        """
        # Get unlabeled indices
        all_indices = set(range(len(unlabeled_pool)))
        labeled_set = set(labeled_indices)
        unlabeled_indices = list(all_indices - labeled_set)
        
        if len(unlabeled_indices) <= n_samples:
            return unlabeled_indices
        
        if self.method == 'random':
            # Random sampling as baseline
            return random.sample(unlabeled_indices, n_samples)
        
        # Extract features for diversity computation
        features = self._extract_features(model, unlabeled_pool, unlabeled_indices)
        
        if self.method == 'kmeans':
            return self._kmeans_sampling(features, unlabeled_indices, n_samples)
        elif self.method == 'core_set':
            return self._core_set_sampling(features, unlabeled_indices, labeled_indices, n_samples)
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")
    
    def _extract_features(self, 
                         model: UDLRatingCTM, 
                         dataset: Dataset, 
                         indices: List[int]) -> np.ndarray:
        """
        Extract features from model for diversity computation.
        
        Args:
            model: Trained model
            dataset: Dataset
            indices: Sample indices
            
        Returns:
            Feature matrix [n_samples, feature_dim]
        """
        model.eval()
        features = []
        
        with torch.no_grad():
            for idx in indices:
                token_ids, _ = dataset[idx]
                token_ids = token_ids.unsqueeze(0).to(self.device)
                
                # Get synchronization representation as features
                _, _, synch_out = model(token_ids)
                if synch_out is not None:
                    features.append(synch_out.squeeze().cpu().numpy())
                else:
                    # Fallback: use embedding of first token
                    embedded = model.embedding(token_ids)
                    features.append(embedded.mean(dim=1).squeeze().cpu().numpy())
        
        return np.array(features)
    
    def _kmeans_sampling(self, 
                        features: np.ndarray, 
                        indices: List[int], 
                        n_samples: int) -> List[int]:
        """
        K-means based diversity sampling.
        
        Args:
            features: Feature matrix
            indices: Sample indices
            n_samples: Number of samples to select
            
        Returns:
            Selected sample indices
        """
        # Perform k-means clustering
        n_clusters = min(self.n_clusters, len(indices))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select samples closest to cluster centers
        selected_indices = []
        samples_per_cluster = n_samples // n_clusters
        remaining_samples = n_samples % n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = [indices[i] for i in range(len(indices)) if cluster_mask[i]]
            
            if len(cluster_indices) == 0:
                continue
            
            # Number of samples to select from this cluster
            n_cluster_samples = samples_per_cluster
            if cluster_id < remaining_samples:
                n_cluster_samples += 1
            
            n_cluster_samples = min(n_cluster_samples, len(cluster_indices))
            
            # Find samples closest to cluster center
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_indices = np.argsort(distances)[:n_cluster_samples]
            
            selected_indices.extend([cluster_indices[i] for i in closest_indices])
        
        logger.info(f"Selected {len(selected_indices)} samples using k-means diversity sampling")
        return selected_indices
    
    def _core_set_sampling(self, 
                          features: np.ndarray, 
                          unlabeled_indices: List[int], 
                          labeled_indices: List[int], 
                          n_samples: int) -> List[int]:
        """
        Core-set based diversity sampling.
        
        Selects samples that maximize minimum distance to already labeled samples.
        
        Args:
            features: Feature matrix for unlabeled samples
            unlabeled_indices: Unlabeled sample indices
            labeled_indices: Labeled sample indices
            n_samples: Number of samples to select
            
        Returns:
            Selected sample indices
        """
        if not labeled_indices:
            # If no labeled samples, use random selection
            return random.sample(unlabeled_indices, min(n_samples, len(unlabeled_indices)))
        
        # Get features for labeled samples (simplified - would need to extract these too)
        # For now, use greedy selection based on distance to unlabeled samples
        
        selected_indices = []
        remaining_indices = unlabeled_indices.copy()
        remaining_features = features.copy()
        
        for _ in range(min(n_samples, len(remaining_indices))):
            if len(selected_indices) == 0:
                # Select first sample randomly
                idx = random.randint(0, len(remaining_indices) - 1)
            else:
                # Select sample with maximum minimum distance to selected samples
                selected_features = np.array([features[unlabeled_indices.index(idx)] 
                                            for idx in selected_indices])
                
                min_distances = []
                for feat in remaining_features:
                    distances = np.linalg.norm(selected_features - feat, axis=1)
                    min_distances.append(np.min(distances))
                
                idx = np.argmax(min_distances)
            
            selected_indices.append(remaining_indices[idx])
            remaining_indices.pop(idx)
            remaining_features = np.delete(remaining_features, idx, axis=0)
        
        logger.info(f"Selected {len(selected_indices)} samples using core-set diversity sampling")
        return selected_indices


class HybridSampling(QueryStrategy):
    """
    Hybrid sampling strategy combining uncertainty and diversity.
    """
    
    def __init__(self, 
                 uncertainty_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 uncertainty_method: str = 'entropy',
                 diversity_method: str = 'kmeans',
                 device: Optional[torch.device] = None):
        """
        Initialize hybrid sampling.
        
        Args:
            uncertainty_weight: Weight for uncertainty component
            diversity_weight: Weight for diversity component
            uncertainty_method: Uncertainty sampling method
            diversity_method: Diversity sampling method
            device: Device for computation
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        
        self.uncertainty_sampler = UncertaintySampling(uncertainty_method, device)
        self.diversity_sampler = DiversitySampling(diversity_method, device=device)
    
    def select_samples(self,
                      model: UDLRatingCTM,
                      unlabeled_pool: Dataset,
                      labeled_indices: List[int],
                      n_samples: int) -> List[int]:
        """
        Select samples using hybrid approach.
        
        Args:
            model: Trained model
            unlabeled_pool: Pool of unlabeled samples
            labeled_indices: Indices of already labeled samples
            n_samples: Number of samples to select
            
        Returns:
            List of selected sample indices
        """
        # Calculate number of samples for each strategy
        n_uncertainty = int(n_samples * self.uncertainty_weight)
        n_diversity = n_samples - n_uncertainty
        
        selected_indices = []
        
        # Select uncertain samples
        if n_uncertainty > 0:
            uncertain_indices = self.uncertainty_sampler.select_samples(
                model, unlabeled_pool, labeled_indices, n_uncertainty
            )
            selected_indices.extend(uncertain_indices)
        
        # Select diverse samples (excluding already selected)
        if n_diversity > 0:
            current_labeled = labeled_indices + selected_indices
            diverse_indices = self.diversity_sampler.select_samples(
                model, unlabeled_pool, current_labeled, n_diversity
            )
            selected_indices.extend(diverse_indices)
        
        logger.info(f"Selected {len(selected_indices)} samples using hybrid sampling "
                   f"({n_uncertainty} uncertain, {n_diversity} diverse)")
        
        return selected_indices


class ActiveLearner:
    """
    Active learning framework for UDL rating.
    
    Implements iterative active learning with various query strategies.
    """
    
    def __init__(self,
                 model: UDLRatingCTM,
                 unlabeled_dataset: Dataset,
                 metrics: List[Any],
                 aggregator: Any,
                 config: ActiveLearningConfig,
                 oracle_fn: Optional[Callable] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize active learner.
        
        Args:
            model: CTM model for training
            unlabeled_dataset: Pool of unlabeled UDL samples
            metrics: Quality metrics for ground truth computation
            aggregator: Metric aggregator
            config: Active learning configuration
            oracle_fn: Function to get ground truth labels (if None, uses metrics)
            device: Device for computation
        """
        self.model = model
        self.unlabeled_dataset = unlabeled_dataset
        self.metrics = metrics
        self.aggregator = aggregator
        self.config = config
        self.oracle_fn = oracle_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize query strategy
        if config.query_strategy == 'uncertainty_sampling':
            self.query_strategy = UncertaintySampling(config.uncertainty_method, device)
        elif config.query_strategy == 'diversity_sampling':
            self.query_strategy = DiversitySampling(config.diversity_method, config.n_clusters, device)
        elif config.query_strategy == 'hybrid':
            self.query_strategy = HybridSampling(
                config.uncertainty_weight, config.diversity_weight,
                config.uncertainty_method, config.diversity_method, device
            )
        else:
            raise ValueError(f"Unknown query strategy: {config.query_strategy}")
        
        # Active learning state
        self.labeled_indices = []
        self.learning_history = []
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Initialized active learner with {config.query_strategy} strategy")
    
    def _get_ground_truth(self, udl_representation: UDLRepresentation) -> float:
        """
        Get ground truth label for UDL.
        
        Args:
            udl_representation: UDL representation
            
        Returns:
            Ground truth quality score
        """
        if self.oracle_fn:
            return self.oracle_fn(udl_representation)
        else:
            # Use mathematical metrics
            metric_values = {}
            for i, metric in enumerate(self.metrics):
                metric_name = f"metric_{i}"
                metric_values[metric_name] = metric.compute(udl_representation)
            
            return self.aggregator.aggregate(metric_values)
    
    def _create_training_dataset(self, indices: List[int]) -> UDLDataset:
        """
        Create training dataset from labeled indices.
        
        Args:
            indices: Labeled sample indices
            
        Returns:
            Training dataset
        """
        # Extract UDL representations for labeled samples
        udl_representations = []
        for idx in indices:
            _, udl_repr = self.unlabeled_dataset[idx]
            udl_representations.append(udl_repr)
        
        # Create vocabulary (simplified - would need proper vocab management)
        vocab = UDLTokenVocabulary()
        for udl in udl_representations:
            vocab.add_tokens_from_udl(udl)
        
        return UDLDataset(udl_representations, vocab)
    
    def initialize_labeled_pool(self) -> List[int]:
        """
        Initialize labeled pool with random samples.
        
        Returns:
            Initial labeled indices
        """
        n_samples = min(self.config.initial_pool_size, len(self.unlabeled_dataset))
        initial_indices = random.sample(range(len(self.unlabeled_dataset)), n_samples)
        
        self.labeled_indices = initial_indices
        
        logger.info(f"Initialized labeled pool with {len(initial_indices)} samples")
        return initial_indices
    
    def active_learning_iteration(self) -> Dict[str, Any]:
        """
        Perform one iteration of active learning.
        
        Returns:
            Iteration results
        """
        logger.info(f"Starting active learning iteration {len(self.learning_history) + 1}")
        
        # Select new samples to label
        new_indices = self.query_strategy.select_samples(
            self.model,
            self.unlabeled_dataset,
            self.labeled_indices,
            self.config.query_batch_size
        )
        
        # Add to labeled pool
        self.labeled_indices.extend(new_indices)
        
        # Create training dataset
        train_dataset = self._create_training_dataset(self.labeled_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Split for validation
        val_size = int(len(self.labeled_indices) * self.config.validation_split)
        if val_size > 0:
            val_indices = random.sample(self.labeled_indices, val_size)
            val_dataset = self._create_training_dataset(val_indices)
            val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        else:
            val_dataloader = None
        
        # Retrain model
        pipeline = TrainingPipeline(
            model=self.model,
            metrics=self.metrics,
            aggregator=self.aggregator,
            device=self.device
        )
        
        training_history = pipeline.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=self.config.retrain_epochs
        )
        
        # Record iteration results
        iteration_result = {
            'iteration': len(self.learning_history) + 1,
            'labeled_pool_size': len(self.labeled_indices),
            'new_samples': len(new_indices),
            'training_history': training_history,
            'final_train_loss': training_history['train_loss'][-1],
            'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None
        }
        
        self.learning_history.append(iteration_result)
        
        logger.info(f"Completed iteration {iteration_result['iteration']}: "
                   f"Pool size={iteration_result['labeled_pool_size']}, "
                   f"Train loss={iteration_result['final_train_loss']:.4f}")
        
        return iteration_result
    
    def run_active_learning(self) -> List[Dict[str, Any]]:
        """
        Run complete active learning process.
        
        Returns:
            List of iteration results
        """
        logger.info(f"Starting active learning with {self.config.max_iterations} iterations")
        
        # Initialize labeled pool
        self.initialize_labeled_pool()
        
        # Run active learning iterations
        for iteration in range(self.config.max_iterations):
            if len(self.labeled_indices) >= len(self.unlabeled_dataset):
                logger.info("All samples have been labeled. Stopping active learning.")
                break
            
            self.active_learning_iteration()
        
        logger.info(f"Completed active learning with {len(self.learning_history)} iterations")
        return self.learning_history
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """
        Get learning curve data.
        
        Returns:
            Learning curve data
        """
        pool_sizes = [result['labeled_pool_size'] for result in self.learning_history]
        train_losses = [result['final_train_loss'] for result in self.learning_history]
        val_losses = [result['final_val_loss'] for result in self.learning_history 
                     if result['final_val_loss'] is not None]
        
        return {
            'pool_sizes': pool_sizes,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_state(self, filepath: str):
        """
        Save active learning state.
        
        Args:
            filepath: Path to save state
        """
        state = {
            'config': self.config.to_dict(),
            'labeled_indices': self.labeled_indices,
            'learning_history': self.learning_history,
            'model_state_dict': self.model.state_dict()
        }
        
        torch.save(state, filepath)
        logger.info(f"Saved active learning state to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load active learning state.
        
        Args:
            filepath: Path to state file
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.labeled_indices = state['labeled_indices']
        self.learning_history = state['learning_history']
        self.model.load_state_dict(state['model_state_dict'])
        
        logger.info(f"Loaded active learning state from {filepath}")


def create_active_learner(model: UDLRatingCTM,
                         unlabeled_dataset: Dataset,
                         metrics: List[Any],
                         aggregator: Any,
                         query_strategy: str = 'uncertainty_sampling',
                         **config_kwargs) -> ActiveLearner:
    """
    Convenience function to create active learner.
    
    Args:
        model: CTM model
        unlabeled_dataset: Unlabeled dataset
        metrics: Quality metrics
        aggregator: Metric aggregator
        query_strategy: Query strategy name
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Active learner instance
    """
    config = ActiveLearningConfig(query_strategy=query_strategy, **config_kwargs)
    
    return ActiveLearner(
        model=model,
        unlabeled_dataset=unlabeled_dataset,
        metrics=metrics,
        aggregator=aggregator,
        config=config
    )