# Implementation Plan

- [x] 1. Set up project structure and mathematical foundation
  - Create directory structure for the UDL rating framework
  - Set up Python package with proper __init__.py files
  - Configure development dependencies (PyTorch, NetworkX, NumPy, SciPy, Hypothesis)
  - Create mathematical framework document template
  - _Requirements: 1.1, 1.2, 11.1_

- [x] 1.1 Write unit tests for project setup
  - Test package imports
  - Test dependency availability
  - _Requirements: 11.7_

- [x] 2. Implement UDL representation and parsing
  - Create Token dataclass with text, type, position, line, column fields
  - Create GrammarRule dataclass with lhs, rhs, constraints, metadata
  - Implement UDLRepresentation class with token extraction, grammar graph construction, and AST conversion
  - Implement basic tokenizer for UDL text
  - Implement grammar graph builder using NetworkX
  - _Requirements: 1.2, 2.2, 4.1, 9.1_

- [x] 2.1 Write property test for tokenization consistency
  - **Property 13: Tokenization Consistency**
  - **Validates: Requirements 4.1**
  - Generate random UDL strings
  - Verify tokenization produces identical results on repeated calls
  - _Requirements: 4.1_

- [x] 2.2 Write unit tests for UDL representation
  - Test token extraction on sample UDLs
  - Test grammar graph construction
  - Test AST conversion
  - _Requirements: 1.2, 2.2_

- [x] 3. Implement metric base class and validation framework
  - Create QualityMetric abstract base class with compute(), get_formula(), get_properties() methods
  - Implement verify_boundedness() method
  - Implement verify_determinism() method
  - Create metric registry for plugin architecture
  - _Requirements: 1.3, 1.4, 6.1, 6.4_

- [x] 3.1 Write property test for metric boundedness
  - **Property 2: Metric Boundedness**
  - **Validates: Requirements 1.4, 3.7**
  - Generate random UDL representations
  - Verify all metrics produce values in [0,1]
  - _Requirements: 1.4, 3.7_

- [x] 3.2 Write property test for metric determinism
  - **Property 3: Metric Determinism**
  - **Validates: Requirements 1.4, 6.5**
  - Generate random UDL representations
  - Compute metrics multiple times
  - Verify identical results
  - _Requirements: 1.4, 6.5_

- [x] 4. Implement Consistency Metric
  - Implement ConsistencyMetric class extending QualityMetric
  - Implement cycle detection using DFS on grammar graph
  - Implement contradiction detection using constraint analysis
  - Implement consistency score computation: 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
  - Document mathematical formula in LaTeX
  - _Requirements: 1.3, 3.1, 3.2_

- [x] 4.1 Write property test for consistency metric correctness
  - **Property 9: Consistency Metric Correctness**
  - **Validates: Requirements 3.2**
  - Generate UDLs with known cycle/contradiction counts
  - Verify formula: 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
  - _Requirements: 3.2_

- [x] 4.2 Write unit tests for consistency metric
  - Test on UDL with no contradictions/cycles (expect 1.0)
  - Test on UDL with known contradictions
  - Test on UDL with known cycles
  - _Requirements: 3.2_

- [x] 5. Implement Completeness Metric
  - Implement CompletenessMetric class extending QualityMetric
  - Implement extract_defined_constructs() method
  - Implement get_required_constructs() method based on language type
  - Implement completeness score computation: |Defined| / |Required|
  - Document mathematical formula in LaTeX
  - _Requirements: 1.3, 3.1, 3.3_

- [x] 5.1 Write property test for completeness metric correctness
  - **Property 10: Completeness Metric Correctness**
  - **Validates: Requirements 3.3**
  - Generate UDLs with varying construct coverage
  - Verify formula: |Defined_Constructs| / |Required_Constructs|
  - _Requirements: 3.3_

- [x] 5.2 Write unit tests for completeness metric
  - Test on fully complete UDL (expect 1.0)
  - Test on partially complete UDL
  - Test on minimal UDL
  - _Requirements: 3.3_

- [x] 6. Implement Expressiveness Metric
  - Implement ExpressivenessMetric class extending QualityMetric
  - Implement Chomsky hierarchy classification algorithm
  - Implement Kolmogorov complexity approximation via compression
  - Implement expressiveness score computation: (Chomsky_Level + Complexity_Score) / 2
  - Document mathematical formula in LaTeX
  - _Requirements: 1.3, 3.1, 3.4_

- [x] 6.1 Write property test for expressiveness metric correctness
  - **Property 11: Expressiveness Metric Correctness**
  - **Validates: Requirements 3.4**
  - Generate UDLs of different Chomsky types
  - Verify Chomsky classification is correct
  - Verify complexity approximation is reasonable
  - _Requirements: 3.4_

- [x] 6.2 Write unit tests for expressiveness metric
  - Test on regular grammar (Type-3)
  - Test on context-free grammar (Type-2)
  - Test on context-sensitive grammar (Type-1)
  - _Requirements: 3.4_

- [x] 7. Implement Structural Coherence Metric
  - Implement StructuralCoherenceMetric class extending QualityMetric
  - Implement Shannon entropy computation for grammar graph
  - Implement graph modularity calculation
  - Implement structural coherence score: 1 - H(G) / H_max
  - Document mathematical formula in LaTeX
  - _Requirements: 1.3, 3.1, 3.5_

- [x] 7.1 Write property test for structural coherence metric correctness
  - **Property 12: Structural Coherence Metric Correctness**
  - **Validates: Requirements 3.5**
  - Generate grammar graphs with varying entropy
  - Verify formula: 1 - H(G)/H_max where H(G) is Shannon entropy
  - _Requirements: 3.5_

- [x] 7.2 Write unit tests for structural coherence metric
  - Test on highly organized graph (low entropy)
  - Test on random graph (high entropy)
  - Test entropy calculation correctness
  - _Requirements: 3.5_

- [x] 8. Implement metric aggregation and confidence calculation
  - Create MetricAggregator class with weight validation
  - Implement aggregate() method: Q = Σ(wᵢ · mᵢ)
  - Create ConfidenceCalculator class
  - Implement compute_confidence() method: C = 1 - H(p)/H_max
  - Validate weight constraints (Σwᵢ = 1, wᵢ ≥ 0)
  - _Requirements: 1.5, 1.6, 3.8, 5.4_

- [x] 8.1 Write property test for aggregation correctness
  - **Property 4: Aggregation Correctness**
  - **Validates: Requirements 1.5, 3.8**
  - Generate random metric values and weights
  - Verify Q = Σ(wᵢ · mᵢ) and Q ∈ [0,1]
  - _Requirements: 1.5, 3.8_

- [x] 8.2 Write property test for confidence formula correctness
  - **Property 5: Confidence Formula Correctness**
  - **Validates: Requirements 1.6, 5.4**
  - Generate random probability distributions
  - Verify C = 1 - H(p)/H_max
  - _Requirements: 1.6, 5.4_

- [x] 8.3 Write unit tests for aggregation and confidence
  - Test aggregation with various weight configurations
  - Test confidence on uniform distribution (expect low confidence)
  - Test confidence on delta distribution (expect high confidence)
  - _Requirements: 1.5, 1.6_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement file discovery and input processing
  - Create file discovery module with recursive directory traversal
  - Implement file extension filtering (.udl, .dsl, .grammar, .ebnf, .txt)
  - Implement error handling for unreadable files
  - Create input validation module
  - _Requirements: 2.1, 2.3, 9.5_

- [x] 10.1 Write property test for file discovery completeness
  - **Property 6: File Discovery Completeness**
  - **Validates: Requirements 2.1**
  - Generate directory structures with UDL files
  - Verify all files are discovered
  - _Requirements: 2.1_

- [x] 10.2 Write property test for graceful error handling
  - **Property 7: Graceful Error Handling**
  - **Validates: Requirements 2.3**
  - Create directories with unreadable files
  - Verify system continues processing
  - Verify errors are logged
  - _Requirements: 2.3_

- [x] 10.3 Write unit tests for file discovery
  - Test on directory with multiple UDL files
  - Test on nested directory structure
  - Test on empty directory
  - _Requirements: 2.1, 2.5_

- [x] 11. Implement rating computation pipeline
  - Create RatingPipeline class that orchestrates metric computation
  - Implement independent metric computation with error handling
  - Implement result aggregation
  - Create QualityReport dataclass with all required fields
  - Implement computation trace generation
  - _Requirements: 5.1, 5.3, 5.5, 5.6_

- [x] 11.1 Write property test for independent metric computation
  - **Property 18: Independent Metric Computation**
  - **Validates: Requirements 5.1**
  - Verify metrics can be computed independently
  - Verify no side effects between metrics
  - _Requirements: 5.1_

- [x] 11.2 Write property test for result aggregation
  - **Property 8: Result Aggregation**
  - **Validates: Requirements 2.4**
  - Process multiple UDL files
  - Verify all results are included in summary
  - _Requirements: 2.4_

- [x] 11.3 Write unit tests for rating pipeline
  - Test complete pipeline on sample UDL
  - Test error handling when metric fails
  - Test report generation
  - _Requirements: 5.1, 5.3, 5.5_

- [x] 12. Implement report generation and output formatting
  - Create ReportGenerator class
  - Implement JSON output format
  - Implement CSV output format
  - Implement HTML report with visualizations
  - Include all metric scores, formulas, and computation traces
  - _Requirements: 5.5, 5.6, 5.7_

- [x] 12.1 Write unit tests for report generation
  - Test JSON format correctness
  - Test CSV format correctness
  - Test HTML generation
  - Verify all required fields are present
  - _Requirements: 5.5, 5.7_

- [ ] 13. Implement CTM model adapter for UDL rating
  - Create UDLRatingCTM class extending nn.Module
  - Implement token embedding layer
  - Integrate ContinuousThoughtMachine from existing codebase
  - Implement rating head that maps synchronization to quality score
  - Add sigmoid activation to ensure output in [0,1]
  - _Requirements: 4.2, 5.2, 5.3_

- [ ] 13.1 Write property test for embedding dimensionality
  - **Property 14: Embedding Dimensionality**
  - **Validates: Requirements 4.2**
  - Generate random tokens
  - Verify embeddings are in ℝᵈ
  - _Requirements: 4.2_

- [ ] 13.2 Write property test for synchronization extraction
  - **Property 19: Synchronization Extraction**
  - **Validates: Requirements 5.2**
  - Process UDL through CTM
  - Verify S(t) is extracted at all iterations t ∈ [1, T]
  - _Requirements: 5.2_

- [ ] 13.3 Write unit tests for CTM adapter
  - Test forward pass produces correct output shapes
  - Test output is in [0,1] range
  - Test with various sequence lengths
  - _Requirements: 4.2, 5.2_

- [ ] 14. Implement training pipeline
  - Create TrainingPipeline class
  - Implement ground truth computation using mathematical metrics
  - Implement loss function: L = α·L_rating + β·L_confidence
  - Implement training loop with Adam optimizer
  - Implement validation and metric reporting
  - Add checkpoint saving
  - _Requirements: 4.3, 4.4, 4.5, 4.6_

- [ ] 14.1 Write property test for loss function correctness
  - **Property 15: Loss Function Correctness**
  - **Validates: Requirements 4.3**
  - Generate random predictions and targets
  - Verify L = α·L_rating + β·L_confidence
  - _Requirements: 4.3_

- [ ] 14.2 Write property test for ground truth consistency
  - **Property 16: Ground Truth Consistency**
  - **Validates: Requirements 4.5**
  - Generate UDLs
  - Verify training ground truth equals mathematical metric computation
  - _Requirements: 4.5_

- [ ] 14.3 Write unit tests for training pipeline
  - Test training loop runs without errors
  - Test loss decreases over epochs
  - Test checkpoint saving and loading
  - _Requirements: 4.3, 4.4, 4.6_

- [ ] 15. Implement tracking and visualization utilities
  - Add tracking mode to CTM adapter
  - Implement activation recording: a_i(t) for all neurons and iterations
  - Implement synchronization matrix recording: S(t) over time
  - Implement attention weight recording with normalization check
  - Create visualization utilities for activation patterns
  - Create visualization utilities for synchronization evolution
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 15.1 Write property test for activation recording
  - **Property 24: Activation Recording**
  - **Validates: Requirements 7.1**
  - Process UDL with tracking enabled
  - Verify activations recorded for all neurons and iterations
  - _Requirements: 7.1_

- [ ] 15.2 Write property test for synchronization matrix recording
  - **Property 25: Synchronization Matrix Recording**
  - **Validates: Requirements 7.2**
  - Process UDL with tracking enabled
  - Verify S(t) recorded at all time steps
  - _Requirements: 7.2_

- [ ] 15.3 Write property test for attention weight normalization
  - **Property 26: Attention Weight Normalization**
  - **Validates: Requirements 7.3**
  - Record attention weights
  - Verify Σ_j α_ij(t) = 1 for all i, t
  - _Requirements: 7.3_

- [ ] 15.4 Write unit tests for tracking
  - Test tracking mode enables recording
  - Test data export to NumPy/HDF5
  - Test visualization generation
  - _Requirements: 7.1, 7.2, 7.6_

- [ ] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Implement comparison and statistical analysis
  - Create ComparisonEngine class
  - Implement pairwise difference computation: Δ_ij = Q_i - Q_j
  - Implement statistical significance tests (t-test, Wilcoxon)
  - Implement effect size computation (Cohen's d)
  - Implement ranking with confidence intervals
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 17.1 Write property test for consistent rating procedures
  - **Property 29: Consistent Rating Procedures**
  - **Validates: Requirements 8.1**
  - Rate multiple UDLs
  - Verify identical procedures used
  - _Requirements: 8.1_

- [ ] 17.2 Write property test for pairwise difference computation
  - **Property 30: Pairwise Difference Computation**
  - **Validates: Requirements 8.2**
  - Generate UDL pairs
  - Verify Δ_ij = Q_i - Q_j computed correctly
  - _Requirements: 8.2_

- [ ] 17.3 Write property test for statistical significance testing
  - **Property 31: Statistical Significance Testing**
  - **Validates: Requirements 8.3**
  - Compare UDLs
  - Verify p-values are computed and reported
  - _Requirements: 8.3_

- [ ] 17.4 Write property test for effect size computation
  - **Property 32: Effect Size Computation**
  - **Validates: Requirements 8.4**
  - Compare UDLs
  - Verify Cohen's d is computed correctly
  - _Requirements: 8.4_

- [ ] 17.5 Write unit tests for comparison engine
  - Test pairwise comparisons
  - Test ranking generation
  - Test confidence interval computation
  - _Requirements: 8.2, 8.3, 8.4, 8.5_

- [ ] 18. Implement evaluation utilities
  - Create EvaluationSuite class
  - Implement k-fold cross-validation (k ≥ 5)
  - Implement correlation computation (Pearson, Spearman) with confidence intervals
  - Implement calibration error computation: ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n)
  - Implement error distribution analysis with Shapiro-Wilk test
  - Implement bootstrap confidence intervals (B ≥ 1000)
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 18.1 Write property test for correlation reporting
  - **Property 37: Correlation Reporting**
  - **Validates: Requirements 10.2**
  - Run evaluation
  - Verify Pearson and Spearman correlations with 95% CIs
  - _Requirements: 10.2_

- [ ] 18.2 Write property test for calibration error computation
  - **Property 38: Calibration Error Computation**
  - **Validates: Requirements 10.3**
  - Generate predictions with confidences
  - Verify ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n)
  - _Requirements: 10.3_

- [ ] 18.3 Write property test for error distribution analysis
  - **Property 39: Error Distribution Analysis**
  - **Validates: Requirements 10.4**
  - Generate error distributions
  - Verify Shapiro-Wilk test is applied
  - _Requirements: 10.4_

- [ ] 18.4 Write property test for bootstrap confidence intervals
  - **Property 40: Bootstrap Confidence Intervals**
  - **Validates: Requirements 10.5**
  - Compute performance metrics
  - Verify bootstrap CIs with B ≥ 1000
  - _Requirements: 10.5_

- [ ] 18.5 Write unit tests for evaluation suite
  - Test cross-validation execution
  - Test metric computation
  - Test report generation
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 19. Implement CLI interface
  - Create command-line interface using argparse or click
  - Implement 'rate' command for rating UDL files/directories
  - Implement 'train' command for training CTM model
  - Implement 'compare' command for comparing multiple UDLs
  - Implement 'evaluate' command for model evaluation
  - Add configuration file support (YAML)
  - _Requirements: 2.1, 4.6, 8.1, 10.1_

- [ ] 19.1 Write integration tests for CLI
  - Test 'rate' command end-to-end
  - Test 'train' command end-to-end
  - Test 'compare' command end-to-end
  - Test 'evaluate' command end-to-end
  - _Requirements: 2.1, 4.6, 8.1, 10.1_

- [ ] 20. Create mathematical framework document
  - Write LaTeX document with formal definitions
  - Include proofs of metric properties (boundedness, etc.)
  - Include complexity analysis for all algorithms
  - Include worked examples with step-by-step calculations
  - Add literature references
  - Compile to PDF
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 11.4, 11.5_

- [ ] 21. Create example UDLs and documentation
  - Create 10-20 example UDL files with varying quality levels
  - Hand-calculate metric values for each example
  - Create tutorial Jupyter notebooks
  - Write API documentation with Sphinx
  - Create architecture diagrams
  - Write troubleshooting guide
  - _Requirements: 11.1, 11.2, 11.3, 11.6_

- [ ] 21.1 Write validation tests for hand-calculated examples
  - **Property 1: Metric Specification Conformance**
  - **Validates: Requirements 1.3**
  - Load example UDLs with hand-calculated values
  - Verify system produces identical results (within ε = 1e-6)
  - _Requirements: 1.3, 11.2_

- [ ] 22. Performance optimization and benchmarking
  - Implement caching for parsed UDL representations
  - Add multiprocessing for batch file processing
  - Optimize metric computation algorithms
  - Create performance benchmarks
  - Profile and optimize bottlenecks
  - _Requirements: 8.6, 9.2, 10.7_

- [ ] 22.1 Write performance tests
  - Benchmark metric computation time vs UDL size
  - Benchmark CTM inference time vs sequence length
  - Benchmark batch processing throughput
  - Verify complexity bounds (O(n) or O(n log n))
  - _Requirements: 8.6, 9.2, 10.7_

- [ ] 23. Final integration and system testing
  - Run complete end-to-end tests on real UDL examples
  - Verify all 40 correctness properties hold
  - Test error recovery scenarios
  - Test with various UDL formats and sizes
  - Verify mathematical correctness on all examples
  - _Requirements: All_

- [ ] 24. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
