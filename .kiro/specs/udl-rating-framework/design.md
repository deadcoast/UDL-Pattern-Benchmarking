# Design Document: UDL Rating Framework

## Overview

The UDL Rating Framework is a mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs). Built on the Continuous Thought Machine (CTM) architecture, the framework combines formal language theory, information theory, and graph theory to produce objective, reproducible quality assessments.

The system operates in two modes:

1. **Mathematical Mode**: Computes quality metrics directly using formal algorithms based on language structure analysis
2. **Learning Mode**: Trains a CTM model to approximate the mathematical metrics, enabling faster inference and pattern recognition

The core innovation is the rigorous mathematical foundation: every rating is traceable to formal computations with proven properties, eliminating subjective assessments.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UDL Rating Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Input      │      │  Mathematical │      │   Output     │  │
│  │  Processing  │─────▶│    Engine     │─────▶│  Generation  │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  File        │      │  Metric      │      │  Report      │  │
│  │  Discovery   │      │  Computation │      │  Generator   │  │
│  │  & Parsing   │      │  Module      │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │              ┌───────┴───────┐              │          │
│         │              │               │              │          │
│         │              ▼               ▼              │          │
│         │      ┌──────────────┐ ┌──────────────┐     │          │
│         │      │  Consistency │ │ Completeness │     │          │
│         │      │    Metric    │ │    Metric    │     │          │
│         │      └──────────────┘ └──────────────┘     │          │
│         │              │               │              │          │
│         │              ▼               ▼              │          │
│         │      ┌──────────────┐ ┌──────────────┐     │          │
│         │      │Expressiveness│ │  Structural  │     │          │
│         │      │    Metric    │ │  Coherence   │     │          │
│         │      └──────────────┘ └──────────────┘     │          │
│         │                      │                      │          │
│         │                      ▼                      │          │
│         │              ┌──────────────┐              │          │
│         └─────────────▶│  CTM Model   │──────────────┘          │
│                        │  (Optional)  │                         │
│                        └──────────────┘                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

1. **Input Layer**: File discovery, parsing, and UDL representation construction
2. **Mathematical Engine**: Core metric computation with formal algorithms
3. **CTM Layer** (Optional): Neural approximation for fast inference
4. **Output Layer**: Rating aggregation, confidence computation, and report generation

## Components and Interfaces

### 1. UDL Representation

**Purpose**: Formal representation of UDL structure for mathematical analysis

**Interface**:
```python
class UDLRepresentation:
    """
    Formal representation of a UDL as a multi-level structure.
    
    Mathematical Definition:
    A UDL is represented as a tuple U = (T, G, S, R) where:
    - T: Set of tokens (terminal symbols)
    - G: Grammar graph G = (V, E) with vertices V (non-terminals) and edges E (production rules)
    - S: Semantic mapping S: T → Semantics
    - R: Set of constraints/rules
    """
    
    def __init__(self, source_text: str, file_path: str):
        """Parse UDL from source text."""
        pass
    
    def get_tokens(self) -> List[Token]:
        """Return token sequence T."""
        pass
    
    def get_grammar_graph(self) -> nx.DiGraph:
        """Return grammar as directed graph G = (V, E)."""
        pass
    
    def get_semantic_map(self) -> Dict[Token, Semantics]:
        """Return semantic mapping S."""
        pass
    
    def get_constraints(self) -> Set[Constraint]:
        """Return constraint set R."""
        pass
    
    def to_ast(self) -> AST:
        """Convert to abstract syntax tree representation."""
        pass
```

### 2. Metric Base Class

**Purpose**: Abstract interface for all quality metrics with mathematical contracts

**Interface**:
```python
from abc import ABC, abstractmethod
from typing import Tuple

class QualityMetric(ABC):
    """
    Abstract base class for quality metrics.
    
    Mathematical Contract:
    Each metric must define a function f: UDL_Space → [0,1]
    with the following properties:
    1. Boundedness: ∀u ∈ UDL_Space, 0 ≤ f(u) ≤ 1
    2. Determinism: f(u₁) = f(u₂) if u₁ = u₂
    3. Computability: f must terminate in polynomial time
    """
    
    @abstractmethod
    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute metric value.
        
        Returns:
            float in [0, 1]
        """
        pass
    
    @abstractmethod
    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties.
        
        Returns:
            Dict with keys: 'bounded', 'monotonic', 'additive', 'continuous'
        """
        pass
    
    def verify_boundedness(self, udl: UDLRepresentation) -> bool:
        """Verify 0 ≤ compute(udl) ≤ 1."""
        value = self.compute(udl)
        return 0.0 <= value <= 1.0
    
    def verify_determinism(self, udl: UDLRepresentation, trials: int = 10) -> bool:
        """Verify same input produces same output."""
        values = [self.compute(udl) for _ in range(trials)]
        return all(v == values[0] for v in values)
```

### 3. Consistency Metric

**Purpose**: Measure internal coherence using graph-theoretic analysis

**Mathematical Definition**:
```
Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)

Where:
- Contradictions: Pairs of rules that cannot both be satisfied
- Cycles: Circular dependencies in grammar graph
- Rules: Total number of production rules
```

**Interface**:
```python
class ConsistencyMetric(QualityMetric):
    """
    Measures rule coherence using graph analysis.
    
    Algorithm:
    1. Build dependency graph from grammar rules
    2. Detect cycles using DFS (O(V + E))
    3. Identify contradictions using SAT solver
    4. Normalize by total rule count
    """
    
    def compute(self, udl: UDLRepresentation) -> float:
        """Compute consistency score."""
        pass
    
    def detect_cycles(self, graph: nx.DiGraph) -> List[List[Node]]:
        """Find all cycles in grammar graph."""
        pass
    
    def find_contradictions(self, rules: Set[Rule]) -> Set[Tuple[Rule, Rule]]:
        """Identify contradictory rule pairs."""
        pass
```

### 4. Completeness Metric

**Purpose**: Measure coverage of language constructs

**Mathematical Definition**:
```
Completeness(U) = |Defined_Constructs| / |Required_Constructs|

Where:
- Defined_Constructs: Set of implemented language features
- Required_Constructs: Set of expected features for the language class
```

**Interface**:
```python
class CompletenessMetric(QualityMetric):
    """
    Measures construct coverage using set theory.
    
    Algorithm:
    1. Extract defined constructs from grammar
    2. Determine required constructs based on language type
    3. Compute coverage ratio
    """
    
    def compute(self, udl: UDLRepresentation) -> float:
        """Compute completeness score."""
        pass
    
    def extract_defined_constructs(self, udl: UDLRepresentation) -> Set[Construct]:
        """Extract D = {c | c is defined in U}."""
        pass
    
    def get_required_constructs(self, language_type: str) -> Set[Construct]:
        """Return R = required constructs for language type."""
        pass
```

### 5. Expressiveness Metric

**Purpose**: Measure language power using formal language theory

**Mathematical Definition**:
```
Expressiveness(U) = (Chomsky_Level + Complexity_Score) / 2

Where:
- Chomsky_Level ∈ {0, 0.33, 0.67, 1.0} for Type-3, Type-2, Type-1, Type-0
- Complexity_Score: Normalized Kolmogorov complexity approximation
```

**Interface**:
```python
class ExpressivenessMetric(QualityMetric):
    """
    Measures language power using Chomsky hierarchy.
    
    Algorithm:
    1. Classify grammar into Chomsky hierarchy
    2. Approximate Kolmogorov complexity via compression
    3. Combine scores with equal weighting
    """
    
    def compute(self, udl: UDLRepresentation) -> float:
        """Compute expressiveness score."""
        pass
    
    def classify_chomsky_level(self, grammar: Grammar) -> int:
        """Return Chomsky type: 0 (unrestricted), 1 (context-sensitive), 
        2 (context-free), 3 (regular)."""
        pass
    
    def approximate_kolmogorov_complexity(self, udl: UDLRepresentation) -> float:
        """Approximate K(U) using compression ratio."""
        pass
```

### 6. Structural Coherence Metric

**Purpose**: Measure organizational quality using information theory

**Mathematical Definition**:
```
Structural_Coherence(U) = 1 - H(G) / H_max

Where:
- H(G): Shannon entropy of grammar graph structure
- H_max: Maximum possible entropy (log₂|V|)
- Lower entropy indicates better organization
```

**Interface**:
```python
class StructuralCoherenceMetric(QualityMetric):
    """
    Measures organization using Shannon entropy.
    
    Algorithm:
    1. Compute degree distribution of grammar graph
    2. Calculate Shannon entropy H = -Σ p(d) log₂ p(d)
    3. Normalize by maximum entropy
    """
    
    def compute(self, udl: UDLRepresentation) -> float:
        """Compute structural coherence score."""
        pass
    
    def compute_shannon_entropy(self, graph: nx.DiGraph) -> float:
        """Calculate H(G) = -Σ p_i log₂(p_i)."""
        pass
    
    def compute_modularity(self, graph: nx.DiGraph) -> float:
        """Calculate Newman modularity Q."""
        pass
```

### 7. Metric Aggregator

**Purpose**: Combine individual metrics into overall quality score

**Mathematical Definition**:
```
Q(U) = Σᵢ wᵢ · mᵢ(U)

Where:
- wᵢ: Weight for metric i (Σwᵢ = 1, wᵢ ≥ 0)
- mᵢ: Individual metric function
- Q: Overall quality score ∈ [0, 1]
```

**Interface**:
```python
class MetricAggregator:
    """
    Combines metrics using weighted sum.
    
    Properties:
    1. Boundedness: If all mᵢ ∈ [0,1] and Σwᵢ=1, then Q ∈ [0,1]
    2. Monotonicity: If mᵢ increases, Q increases (for wᵢ > 0)
    3. Linearity: Q is linear in each metric
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize with metric weights.
        
        Args:
            weights: Dict mapping metric names to weights
        
        Requires:
            sum(weights.values()) == 1.0
            all(w >= 0 for w in weights.values())
        """
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values())
        self.weights = weights
    
    def aggregate(self, metric_values: Dict[str, float]) -> float:
        """
        Compute Q = Σ wᵢ · mᵢ.
        
        Returns:
            Overall quality score in [0, 1]
        """
        return sum(self.weights[name] * value 
                   for name, value in metric_values.items())
```

### 8. Confidence Calculator

**Purpose**: Compute certainty of quality assessment

**Mathematical Definition**:
```
C = 1 - H(p) / H_max

Where:
- H(p) = -Σ pᵢ log(pᵢ): Shannon entropy of prediction distribution
- H_max = log(n): Maximum entropy for n classes
- C ∈ [0, 1]: Confidence score
```

**Interface**:
```python
class ConfidenceCalculator:
    """
    Computes confidence from prediction entropy.
    
    Properties:
    1. Boundedness: 0 ≤ C ≤ 1
    2. Monotonicity: Lower entropy → higher confidence
    3. Calibration: C should match empirical accuracy
    """
    
    def compute_confidence(self, prediction_probs: np.ndarray) -> float:
        """
        Compute C = 1 - H(p)/H_max.
        
        Args:
            prediction_probs: Probability distribution over classes
        
        Returns:
            Confidence score in [0, 1]
        """
        entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10))
        max_entropy = np.log(len(prediction_probs))
        return 1.0 - (entropy / max_entropy)
```

### 9. CTM Model Adapter

**Purpose**: Adapt CTM architecture for UDL rating

**Interface**:
```python
class UDLRatingCTM(nn.Module):
    """
    CTM model adapted for UDL quality prediction.
    
    Architecture:
    1. Token Embedding: E: Token → ℝᵈ
    2. CTM Core: Processes sequence with T iterations
    3. Synchronization: Extracts S(t) at each iteration
    4. Rating Head: Maps final S(T) → [0,1]
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 d_input: int = 64,
                 iterations: int = 20,
                 n_synch_out: int = 32,
                 **ctm_kwargs):
        """Initialize CTM for UDL rating."""
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_input)
        self.ctm = ContinuousThoughtMachine(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            n_synch_out=n_synch_out,
            backbone_type='none',
            positional_embedding_type='custom-rotational-1d',
            **ctm_kwargs
        )
        self.rating_head = nn.Linear(n_synch_out, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            token_ids: [batch, seq_len] token indices
        
        Returns:
            ratings: [batch, 1] quality scores in [0,1]
            certainties: [batch, 2] certainty scores
        """
        # Embed tokens
        x = self.embedding(token_ids)  # [batch, seq_len, d_input]
        
        # Process through CTM
        predictions, certainties, synch = self.ctm(x)  # predictions: [batch, 1, T]
        
        # Extract final prediction
        final_pred = predictions[:, :, -1]  # [batch, 1]
        final_cert = certainties[:, :, -1]  # [batch, 2]
        
        # Apply sigmoid to ensure [0,1]
        ratings = self.sigmoid(final_pred)
        
        return ratings, final_cert
```

### 10. Training Pipeline

**Purpose**: Train CTM to approximate mathematical metrics

**Interface**:
```python
class TrainingPipeline:
    """
    Trains CTM model to approximate mathematical metrics.
    
    Loss Function:
    L = α · L_rating + β · L_confidence
    
    Where:
    - L_rating = MSE(predicted, ground_truth)
    - L_confidence = Calibration_Loss(confidence, accuracy)
    """
    
    def __init__(self,
                 model: UDLRatingCTM,
                 metrics: List[QualityMetric],
                 aggregator: MetricAggregator,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        """Initialize training pipeline."""
        self.model = model
        self.metrics = metrics
        self.aggregator = aggregator
        self.alpha = alpha
        self.beta = beta
    
    def compute_ground_truth(self, udl: UDLRepresentation) -> float:
        """Compute ground truth using mathematical metrics."""
        metric_values = {
            metric.__class__.__name__: metric.compute(udl)
            for metric in self.metrics
        }
        return self.aggregator.aggregate(metric_values)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    def evaluate(self, test_set: Dataset) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
```

## Data Models

### UDL Token

```python
@dataclass
class Token:
    """Represents a single token in UDL."""
    text: str
    type: TokenType  # KEYWORD, IDENTIFIER, OPERATOR, LITERAL, etc.
    position: int
    line: int
    column: int
```

### Grammar Rule

```python
@dataclass
class GrammarRule:
    """Represents a production rule in grammar."""
    lhs: str  # Left-hand side (non-terminal)
    rhs: List[str]  # Right-hand side (sequence of symbols)
    constraints: List[Constraint]
    metadata: Dict[str, Any]
```

### Quality Report

```python
@dataclass
class QualityReport:
    """Complete quality assessment report."""
    overall_score: float  # Q ∈ [0, 1]
    confidence: float  # C ∈ [0, 1]
    metric_scores: Dict[str, float]  # Individual metric values
    metric_formulas: Dict[str, str]  # LaTeX formulas
    computation_trace: List[ComputationStep]  # Step-by-step trace
    error_bounds: Dict[str, Tuple[float, float]]  # (lower, upper) bounds
    timestamp: datetime
    udl_file: str
```

### Computation Step

```python
@dataclass
class ComputationStep:
    """Single step in computation trace."""
    step_number: int
    operation: str  # Description of operation
    formula: str  # LaTeX formula
    inputs: Dict[str, Any]
    output: Any
    intermediate_values: Dict[str, float]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Metric Specification Conformance

*For any* UDL and any implemented metric, the computed value must match the mathematical specification exactly (within numerical precision ε = 1e-6).

**Validates: Requirements 1.3**

**Rationale**: This ensures that the implementation faithfully represents the mathematical definitions. We can test this by computing metrics on known UDL examples and comparing against hand-calculated values.

### Property 2: Metric Boundedness

*For any* UDL and any quality metric m, the computed value must satisfy 0 ≤ m(UDL) ≤ 1.

**Validates: Requirements 1.4, 3.7**

**Rationale**: All metrics must be normalized to [0,1] for comparability. This is a fundamental mathematical property that must hold for all inputs.

### Property 3: Metric Determinism

*For any* UDL, computing the same metric multiple times must produce identical results.

**Validates: Requirements 1.4, 6.5**

**Rationale**: Metrics must be deterministic functions - same input always produces same output. This is essential for reproducibility.

### Property 4: Aggregation Correctness

*For any* set of metric values {m₁, m₂, ..., mₙ} and weights {w₁, w₂, ..., wₙ} where Σwᵢ = 1, the aggregated score Q must equal Σ(wᵢ · mᵢ) and satisfy 0 ≤ Q ≤ 1.

**Validates: Requirements 1.5, 3.8**

**Rationale**: The aggregation function must correctly implement the weighted sum formula and preserve boundedness.

### Property 5: Confidence Formula Correctness

*For any* prediction distribution p, the confidence C must equal 1 - H(p)/H_max where H(p) = -Σ pᵢ log(pᵢ) and H_max = log(n).

**Validates: Requirements 1.6, 5.4**

**Rationale**: Confidence must be computed exactly according to the entropy-based formula to ensure mathematical correctness.

### Property 6: File Discovery Completeness

*For any* directory structure containing UDL files with supported extensions, the system must discover all such files.

**Validates: Requirements 2.1**

**Rationale**: The file discovery mechanism must not miss any valid UDL files in the directory tree.

### Property 7: Graceful Error Handling

*For any* directory containing at least one unreadable file, the system must continue processing remaining files and log the error.

**Validates: Requirements 2.3**

**Rationale**: Errors in individual files should not prevent processing of other files.

### Property 8: Result Aggregation

*For any* set of processed UDL files, the system must produce a summary report containing results for all successfully processed files.

**Validates: Requirements 2.4**

**Rationale**: All successful processing results must be included in the final report.

### Property 9: Consistency Metric Correctness

*For any* UDL with grammar graph G, the consistency score must be computed as 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1).

**Validates: Requirements 3.2**

**Rationale**: The consistency metric must follow its mathematical definition exactly.

### Property 10: Completeness Metric Correctness

*For any* UDL, the completeness score must equal |Defined_Constructs| / |Required_Constructs|.

**Validates: Requirements 3.3**

**Rationale**: The completeness metric must implement the set-theoretic coverage ratio.

### Property 11: Expressiveness Metric Correctness

*For any* UDL, the expressiveness score must be computed using Chomsky hierarchy classification and complexity measures as specified.

**Validates: Requirements 3.4**

**Rationale**: The expressiveness metric must follow the formal language theory-based computation.

### Property 12: Structural Coherence Metric Correctness

*For any* UDL with grammar graph G, the structural coherence must be computed as 1 - H(G)/H_max where H(G) is Shannon entropy.

**Validates: Requirements 3.5**

**Rationale**: The structural coherence metric must use information-theoretic measures as specified.

### Property 13: Tokenization Consistency

*For any* UDL string, tokenization must produce the same token sequence on repeated calls.

**Validates: Requirements 4.1**

**Rationale**: Tokenization must be deterministic to ensure reproducible results.

### Property 14: Embedding Dimensionality

*For any* token, the embedding must map to a vector in ℝᵈ where d is the specified embedding dimension.

**Validates: Requirements 4.2**

**Rationale**: Embeddings must have the correct dimensionality for CTM processing.

### Property 15: Loss Function Correctness

*For any* training batch, the loss must be computed as L = α·L_rating + β·L_confidence where L_rating is MSE and L_confidence is calibration loss.

**Validates: Requirements 4.3**

**Rationale**: The training loss must follow the specified formula exactly.

### Property 16: Ground Truth Consistency

*For any* UDL, the ground truth quality score computed for training must equal the score computed by the mathematical metrics.

**Validates: Requirements 4.5**

**Rationale**: Training labels must match the mathematical metric computations to ensure the model learns the correct function.

### Property 17: Model Approximation Accuracy

*For any* test UDL, the absolute difference between model prediction and true metric value must be less than error bound ε.

**Validates: Requirements 4.8**

**Rationale**: The learned model must approximate the mathematical metrics within acceptable error.

### Property 18: Independent Metric Computation

*For any* UDL, each quality metric must be computed independently without side effects on other metrics.

**Validates: Requirements 5.1**

**Rationale**: Metrics must be independent to ensure modularity and correctness.

### Property 19: Synchronization Extraction

*For any* UDL input to the CTM, synchronization representations S(t) must be extracted at each iteration t ∈ [1, T].

**Validates: Requirements 5.2**

**Rationale**: The CTM must produce synchronization representations at all time steps for proper functioning.

### Property 20: Overall Score Computation

*For any* UDL, the overall quality score Q must be computed using the aggregation function with all intermediate values traceable.

**Validates: Requirements 5.3**

**Rationale**: The overall score must follow the defined aggregation procedure.

### Property 21: Per-Metric Reporting

*For any* UDL, all individual metric scores must be included in the output report.

**Validates: Requirements 5.5**

**Rationale**: Users must have access to all component scores, not just the aggregate.

### Property 22: Uncertainty Estimation

*For any* computed metric, an uncertainty estimate or error bound must be provided.

**Validates: Requirements 5.8**

**Rationale**: All measurements should include uncertainty quantification for scientific rigor.

### Property 23: New Metric Property Validation

*For any* newly added metric, automated tests must verify that it satisfies boundedness, determinism, and tractability properties.

**Validates: Requirements 6.3, 6.5**

**Rationale**: New metrics must be validated to ensure they meet the mathematical contracts.

### Property 24: Activation Recording

*For any* UDL processed with tracking enabled, neuron activations a_i(t) must be recorded for all neurons i and iterations t.

**Validates: Requirements 7.1**

**Rationale**: Tracking mode must capture complete activation traces for analysis.

### Property 25: Synchronization Matrix Recording

*For any* UDL processed with tracking enabled, synchronization matrices S(t) must be recorded at all time steps.

**Validates: Requirements 7.2**

**Rationale**: Tracking mode must capture synchronization evolution for analysis.

### Property 26: Attention Weight Normalization

*For any* recorded attention weights α_ij(t), the sum over j must equal 1: Σ_j α_ij(t) = 1.

**Validates: Requirements 7.3**

**Rationale**: Attention weights must form a valid probability distribution.

### Property 27: Activation Statistics

*For any* activation trace, statistical measures (mean, variance, spectral properties) must be computable.

**Validates: Requirements 7.4**

**Rationale**: The system must support quantitative analysis of activation patterns.

### Property 28: Synchronization Evolution Metrics

*For any* synchronization trace, evolution metrics (stability, convergence) must be computable.

**Validates: Requirements 7.5**

**Rationale**: The system must support analysis of how synchronization evolves over time.

### Property 29: Consistent Rating Procedures

*For any* two UDLs processed by the system, the rating computation procedure must be identical.

**Validates: Requirements 8.1**

**Rationale**: All UDLs must be evaluated using the same methodology for fair comparison.

### Property 30: Pairwise Difference Computation

*For any* pair of UDLs (i, j), the difference Δ_ij = Q_i - Q_j must be computed correctly.

**Validates: Requirements 8.2**

**Rationale**: Comparisons require accurate difference calculations.

### Property 31: Statistical Significance Testing

*For any* comparison between UDLs, statistical tests (t-test, Wilcoxon) must be applied and p-values reported.

**Validates: Requirements 8.3**

**Rationale**: Comparisons must include statistical significance to avoid false conclusions.

### Property 32: Effect Size Computation

*For any* comparison between UDLs, Cohen's d effect size must be computed to quantify the magnitude of difference.

**Validates: Requirements 8.4**

**Rationale**: Statistical significance alone is insufficient; effect size quantifies practical importance.

### Property 33: Ranking with Confidence Intervals

*For any* set of UDLs, a ranking must be produced with confidence intervals for each position.

**Validates: Requirements 8.5**

**Rationale**: Rankings should include uncertainty to reflect measurement precision.

### Property 34: Chunking Context Preservation

*For any* large UDL requiring chunking, the sliding window must preserve context with overlap (stride s < window w).

**Validates: Requirements 9.3**

**Rationale**: Chunking must not lose information at boundaries.

### Property 35: Parse Error Reporting

*For any* malformed UDL input, the system must report parse errors with line numbers and expected tokens.

**Validates: Requirements 9.4**

**Rationale**: Error messages must be informative to help users fix issues.

### Property 36: Input Format Validation

*For any* input file, format validation must occur before processing begins.

**Validates: Requirements 9.5**

**Rationale**: Early validation prevents wasted computation on invalid inputs.

### Property 37: Correlation Reporting

*For any* evaluation, Pearson and Spearman correlations must be reported with 95% confidence intervals.

**Validates: Requirements 10.2**

**Rationale**: Evaluation must include both parametric and non-parametric correlation measures with uncertainty.

### Property 38: Calibration Error Computation

*For any* evaluation, the expected calibration error ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n) must be computed.

**Validates: Requirements 10.3**

**Rationale**: Confidence scores must be calibrated to match empirical accuracy.

### Property 39: Error Distribution Analysis

*For any* evaluation, error distributions must be analyzed with normality tests (Shapiro-Wilk).

**Validates: Requirements 10.4**

**Rationale**: Understanding error distributions helps identify systematic biases.

### Property 40: Bootstrap Confidence Intervals

*For any* performance metric, bootstrap confidence intervals must be computed with B ≥ 1000 samples.

**Validates: Requirements 10.5**

**Rationale**: Bootstrap provides robust uncertainty estimates without distributional assumptions.


## Error Handling

### Error Categories

1. **Input Errors**
   - File not found
   - Unsupported file format
   - Malformed UDL syntax
   - Empty or invalid directory

2. **Computation Errors**
   - Numerical overflow/underflow
   - Division by zero in metric computation
   - Graph algorithm failures (e.g., cycle detection on invalid graphs)
   - Memory exhaustion on very large UDLs

3. **Model Errors**
   - Model checkpoint not found
   - Incompatible model version
   - CUDA out of memory
   - Inference timeout

4. **Validation Errors**
   - Metric property violation (e.g., value outside [0,1])
   - Weight sum not equal to 1
   - Invalid configuration parameters

### Error Handling Strategy

```python
class UDLRatingError(Exception):
    """Base exception for UDL rating framework."""
    pass

class InputError(UDLRatingError):
    """Errors related to input processing."""
    pass

class ComputationError(UDLRatingError):
    """Errors during metric computation."""
    pass

class ValidationError(UDLRatingError):
    """Errors from validation checks."""
    pass

def safe_metric_compute(metric: QualityMetric, 
                       udl: UDLRepresentation) -> Tuple[Optional[float], Optional[str]]:
    """
    Safely compute metric with error handling.
    
    Returns:
        (value, error_message) where value is None if computation failed
    """
    try:
        value = metric.compute(udl)
        
        # Validate boundedness
        if not (0.0 <= value <= 1.0):
            return None, f"Metric {metric.__class__.__name__} produced out-of-bounds value: {value}"
        
        return value, None
        
    except ZeroDivisionError as e:
        return None, f"Division by zero in {metric.__class__.__name__}: {str(e)}"
    
    except MemoryError as e:
        return None, f"Memory exhausted computing {metric.__class__.__name__}: {str(e)}"
    
    except Exception as e:
        return None, f"Unexpected error in {metric.__class__.__name__}: {str(e)}"
```

### Graceful Degradation

When errors occur, the system should:

1. **Log the error** with full context (file, metric, stack trace)
2. **Continue processing** other files/metrics when possible
3. **Report partial results** with clear indication of what failed
4. **Provide fallback values** (e.g., NaN or None) for failed computations
5. **Include error summary** in final report

## Testing Strategy

### Unit Testing

**Scope**: Individual components and functions

**Key Test Cases**:

1. **Metric Computation Tests**
   - Test each metric on hand-crafted UDL examples with known correct values
   - Test boundary cases (empty UDL, minimal UDL, maximal UDL)
   - Test metric properties (boundedness, determinism)
   - Example: Test that consistency metric returns 1.0 for a UDL with no contradictions or cycles

2. **Aggregation Tests**
   - Test weighted sum with various weight configurations
   - Test that aggregation preserves boundedness
   - Test edge cases (all weights on one metric, equal weights)

3. **Confidence Calculation Tests**
   - Test entropy computation on known distributions
   - Test that confidence is in [0,1]
   - Test extreme cases (uniform distribution, delta distribution)

4. **Parsing Tests**
   - Test tokenization on various UDL syntaxes
   - Test grammar graph construction
   - Test error handling for malformed input

5. **File Discovery Tests**
   - Test recursive directory traversal
   - Test filtering by file extension
   - Test handling of symbolic links and permissions

### Property-Based Testing

**Scope**: Universal properties that should hold across all inputs

**Property Testing Library**: We will use `hypothesis` for Python, which provides:
- Automatic test case generation
- Shrinking of failing examples
- Stateful testing capabilities

**Configuration**: Each property test should run a minimum of 100 iterations to ensure thorough coverage.

**Key Properties to Test**:

1. **Metric Boundedness** (Property 2)
   - Generate random UDL representations
   - Verify all metrics produce values in [0,1]

2. **Metric Determinism** (Property 3)
   - Generate random UDL representations
   - Compute metric multiple times
   - Verify all results are identical

3. **Aggregation Correctness** (Property 4)
   - Generate random metric values and weights
   - Verify Q = Σ(wᵢ · mᵢ) and Q ∈ [0,1]

4. **Confidence Formula** (Property 5)
   - Generate random probability distributions
   - Verify C = 1 - H(p)/H_max

5. **Tokenization Consistency** (Property 13)
   - Generate random UDL strings
   - Tokenize multiple times
   - Verify identical token sequences

6. **Attention Weight Normalization** (Property 26)
   - Generate random attention weights
   - Verify Σ_j α_ij(t) = 1 for all i, t

### Integration Testing

**Scope**: End-to-end workflows

**Key Test Scenarios**:

1. **Complete Rating Pipeline**
   - Input: Directory with multiple UDL files
   - Process: Full rating computation
   - Output: Verify report contains all expected fields

2. **Training Pipeline**
   - Input: Synthetic UDL dataset with ground truth
   - Process: Train CTM model for 10 epochs
   - Output: Verify loss decreases and validation metrics improve

3. **Comparison Workflow**
   - Input: Multiple UDL versions
   - Process: Compute ratings and statistical comparisons
   - Output: Verify rankings and p-values are computed

4. **Error Recovery**
   - Input: Directory with mix of valid and invalid files
   - Process: Attempt to process all files
   - Output: Verify valid files are processed and errors are logged

### Performance Testing

**Scope**: Computational efficiency and scalability

**Key Benchmarks**:

1. **Metric Computation Time**
   - Measure time to compute each metric on UDLs of varying sizes
   - Verify O(n) or O(n log n) complexity as specified

2. **Model Inference Time**
   - Measure CTM inference time on sequences of varying lengths
   - Verify acceptable latency (< 1 second for typical UDLs)

3. **Memory Usage**
   - Monitor memory consumption during processing
   - Verify no memory leaks over extended runs

4. **Batch Processing Throughput**
   - Measure files processed per second
   - Verify linear scaling with number of files

### Validation Testing

**Scope**: Mathematical correctness verification

**Approach**:

1. **Hand-Calculated Examples**
   - Create 10-20 UDL examples with hand-calculated metric values
   - Verify system produces identical results (within ε = 1e-6)

2. **Property Verification**
   - For each metric, verify mathematical properties hold
   - Example: Verify consistency metric is monotonic in number of contradictions

3. **Cross-Validation**
   - Compare metric values computed by different implementations
   - Verify agreement within numerical precision

4. **Regression Testing**
   - Maintain suite of test cases with known outputs
   - Verify outputs remain stable across code changes

### Test Organization

```
tests/
├── unit/
│   ├── test_metrics.py
│   ├── test_aggregation.py
│   ├── test_confidence.py
│   ├── test_parsing.py
│   └── test_file_discovery.py
├── property/
│   ├── test_metric_properties.py
│   ├── test_aggregation_properties.py
│   ├── test_confidence_properties.py
│   └── test_tokenization_properties.py
├── integration/
│   ├── test_rating_pipeline.py
│   ├── test_training_pipeline.py
│   ├── test_comparison_workflow.py
│   └── test_error_recovery.py
├── performance/
│   ├── benchmark_metrics.py
│   ├── benchmark_inference.py
│   └── benchmark_batch_processing.py
├── validation/
│   ├── test_hand_calculated.py
│   ├── test_mathematical_properties.py
│   └── test_regression.py
└── fixtures/
    ├── sample_udls/
    ├── expected_outputs/
    └── test_data_generators.py
```

## Mathematical Framework Document

As specified in Requirement 1, a separate mathematical framework document will be created that provides:

1. **Formal Definitions**
   - UDL representation space
   - Each quality metric with explicit formulas
   - Aggregation function
   - Confidence measure

2. **Mathematical Proofs**
   - Metric boundedness proofs
   - Aggregation correctness proof
   - Confidence calibration properties

3. **Complexity Analysis**
   - Time complexity for each metric
   - Space complexity for data structures
   - Overall system complexity

4. **Worked Examples**
   - Step-by-step calculations for sample UDLs
   - Verification of properties
   - Edge case analysis

5. **Literature References**
   - Formal language theory foundations
   - Information theory references
   - Graph theory algorithms

This document will be created as `docs/mathematical_framework.pdf` with full LaTeX source in `docs/mathematical_framework.tex`.

## Implementation Notes

### Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+
- **Graph Analysis**: NetworkX
- **Numerical Computing**: NumPy, SciPy
- **Statistical Testing**: SciPy.stats, statsmodels
- **Property Testing**: Hypothesis
- **Parsing**: ANTLR4 or Lark parser
- **Visualization**: Matplotlib, Seaborn
- **Documentation**: Sphinx with LaTeX math support

### Code Organization

```
udl_rating_framework/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── representation.py      # UDLRepresentation class
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py            # QualityMetric base class
│   │   ├── consistency.py     # ConsistencyMetric
│   │   ├── completeness.py    # CompletenessMetric
│   │   ├── expressiveness.py  # ExpressivenessMetric
│   │   └── structural.py      # StructuralCoherenceMetric
│   ├── aggregation.py         # MetricAggregator
│   └── confidence.py          # ConfidenceCalculator
├── models/
│   ├── __init__.py
│   ├── ctm_adapter.py         # UDLRatingCTM
│   └── training.py            # TrainingPipeline
├── io/
│   ├── __init__.py
│   ├── file_discovery.py      # Directory traversal
│   ├── parsing.py             # UDL parsing
│   └── reporting.py           # Report generation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   ├── statistical_tests.py  # Statistical comparisons
│   └── visualization.py       # Result visualization
├── utils/
│   ├── __init__.py
│   ├── math_utils.py          # Mathematical utilities
│   ├── graph_utils.py         # Graph algorithms
│   └── validation.py          # Input validation
└── cli/
    ├── __init__.py
    ├── rate.py                # Rating command
    ├── train.py               # Training command
    ├── compare.py             # Comparison command
    └── evaluate.py            # Evaluation command
```

### Development Workflow

1. **Phase 1: Mathematical Foundation**
   - Implement UDL representation
   - Implement all metrics with unit tests
   - Verify mathematical properties

2. **Phase 2: Core Pipeline**
   - Implement file discovery and parsing
   - Implement aggregation and confidence
   - Implement reporting

3. **Phase 3: CTM Integration**
   - Adapt CTM architecture
   - Implement training pipeline
   - Train initial models

4. **Phase 4: Evaluation & Refinement**
   - Implement evaluation utilities
   - Run comprehensive tests
   - Refine based on results

5. **Phase 5: Documentation & Examples**
   - Write mathematical framework document
   - Create tutorial notebooks
   - Generate API documentation

### Performance Optimization

1. **Caching**: Cache parsed UDL representations to avoid re-parsing
2. **Parallelization**: Use multiprocessing for batch file processing
3. **Lazy Evaluation**: Compute metrics only when needed
4. **Efficient Data Structures**: Use sparse matrices for large grammar graphs
5. **GPU Acceleration**: Use CUDA for CTM inference when available

### Extensibility Points

1. **Custom Metrics**: Users can subclass `QualityMetric` to add new metrics
2. **Custom Parsers**: Support for additional UDL formats via parser plugins
3. **Custom Aggregation**: Alternative aggregation functions beyond weighted sum
4. **Custom Reports**: Templated report generation with user-defined formats

## Deployment Considerations

### Installation

```bash
pip install udl-rating-framework
```

### Usage Examples

**Basic Rating**:
```bash
udl-rate --directory ./my_udls --output report.json
```

**Training**:
```bash
udl-train --dataset ./training_data --epochs 50 --output model.pt
```

**Comparison**:
```bash
udl-compare --files udl_v1.dsl udl_v2.dsl --output comparison.html
```

### Configuration

Configuration via YAML file:

```yaml
metrics:
  consistency:
    weight: 0.3
  completeness:
    weight: 0.25
  expressiveness:
    weight: 0.25
  structural_coherence:
    weight: 0.2

model:
  use_ctm: true
  checkpoint: ./models/udl_rating_ctm.pt
  
output:
  format: json
  include_trace: true
  precision: 6
```

## Future Enhancements

1. **Interactive Visualization**: Web-based dashboard for exploring results
2. **Incremental Analysis**: Track quality changes over time
3. **Recommendation System**: Suggest improvements based on low metric scores
4. **Multi-Language Support**: Extend beyond UDLs to general programming languages
5. **Distributed Computing**: Support for large-scale batch processing
6. **Active Learning**: Iteratively improve model with user feedback
7. **Explainability**: Generate natural language explanations of ratings
8. **Benchmark Suite**: Curated collection of UDLs for standardized evaluation
