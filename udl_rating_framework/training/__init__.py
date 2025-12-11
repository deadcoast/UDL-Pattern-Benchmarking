"""
Training module for UDL Rating Framework.

Contains training pipeline and CTM-aware machine learning enhancements that
leverage the unique architecture of Continuous Thought Machines, including:
- Synchronization-based uncertainty quantification
- CTM ensemble methods with synchronization diversity
- CTM-specific hyperparameter optimization
- Neuron-level active learning
- Temporal dynamics transfer learning
"""

# CTM-aware active learning
from .active_learning import (
    ActiveLearner,
    ActiveLearningConfig,
    CTMUncertaintySampling,
    DiversitySampling,
    HybridSampling,
    QueryStrategy,
    create_active_learner,
)

# CTM-aware ensemble methods
from .ensemble_methods import (
    CTMEnsembleTrainer,
    EnsembleMember,
    EnsemblePredictor,
    create_bootstrap_ensemble,
)

# CTM-aware hyperparameter optimization
from .hyperparameter_optimization import (
    CTMHyperparameterOptimizer,
    CTMHyperparameterSpace,
    OptimizationResult,
    create_ctm_param_space,
    create_focused_ctm_param_space,
)
from .training_pipeline import TrainingPipeline

# CTM-aware transfer learning
from .transfer_learning import (
    CTMTransferConfig,
    CTMTransferLearningTrainer,
    PretrainedFeatureExtractor,
    create_ctm_transfer_learning_model,
)

# CTM-aware uncertainty quantification
from .uncertainty_quantification import (
    CalibrationAnalyzer,
    DeepEnsembleUncertainty,
    NeuronLevelUncertainty,
    SynchronizationUncertainty,
    UncertaintyAwarePredictor,
    UncertaintyEstimate,
    UncertaintyQuantifier,
    VariationalInference,
    bootstrap_confidence_intervals,
    create_uncertainty_quantifier,
)

__all__ = [
    # Core training
    "TrainingPipeline",
    # CTM-aware hyperparameter optimization
    "CTMHyperparameterOptimizer",
    "CTMHyperparameterSpace",
    "OptimizationResult",
    "create_ctm_param_space",
    "create_focused_ctm_param_space",
    # CTM-aware ensemble methods
    "EnsemblePredictor",
    "EnsembleMember",
    "CTMEnsembleTrainer",
    "create_bootstrap_ensemble",
    # CTM-aware transfer learning
    "CTMTransferConfig",
    "CTMTransferLearningTrainer",
    "PretrainedFeatureExtractor",
    "create_ctm_transfer_learning_model",
    # CTM-aware active learning
    "ActiveLearner",
    "ActiveLearningConfig",
    "QueryStrategy",
    "CTMUncertaintySampling",
    "DiversitySampling",
    "HybridSampling",
    "create_active_learner",
    # CTM-aware uncertainty quantification
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "UncertaintyAwarePredictor",
    "SynchronizationUncertainty",
    "NeuronLevelUncertainty",
    "DeepEnsembleUncertainty",
    "VariationalInference",
    "CalibrationAnalyzer",
    "create_uncertainty_quantifier",
    "bootstrap_confidence_intervals",
]
