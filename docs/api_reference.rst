API Reference
=============

This section provides detailed API documentation for all modules in the UDL Rating Framework.

Core Modules
------------

UDL Representation
~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.representation
   :members:
   :undoc-members:
   :show-inheritance:

The ``UDLRepresentation`` class provides the formal representation of a UDL as a multi-level structure.

**Mathematical Definition**:

A UDL is represented as a tuple U = (T, G, S, R) where:

- T: Set of tokens (terminal symbols)
- G: Grammar graph G = (V, E) with vertices V (non-terminals) and edges E (production rules)
- S: Semantic mapping S: T → Semantics
- R: Set of constraints/rules

**Example Usage**:

.. code-block:: python

   from udl_rating_framework.core.representation import UDLRepresentation

   udl_content = """
   Expression ::= Term (('+' | '-') Term)*
   Term ::= Factor (('*' | '/') Factor)*
   Factor ::= Number | '(' Expression ')'
   """
   
   udl = UDLRepresentation(udl_content, "calculator.udl")
   
   # Access tokens
   tokens = udl.get_tokens()
   
   # Access grammar graph
   graph = udl.get_grammar_graph()
   
   # Access grammar rules
   rules = udl.get_grammar_rules()

Rating Pipeline
~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

The ``RatingPipeline`` class orchestrates the complete metric computation workflow.

**Example Usage**:

.. code-block:: python

   from udl_rating_framework.core.pipeline import RatingPipeline
   from udl_rating_framework.core.representation import UDLRepresentation

   pipeline = RatingPipeline()
   
   udl = UDLRepresentation(udl_content, "my_language.udl")
   report = pipeline.rate_udl(udl)
   
   print(f"Overall Score: {report.overall_score:.3f}")
   print(f"Confidence: {report.confidence:.3f}")

Aggregation and Confidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.aggregation
   :members:
   :undoc-members:
   :show-inheritance:

The ``MetricAggregator`` combines individual metrics using weighted sum:

**Mathematical Definition**:

.. math::

   Q(U) = \sum_i w_i \cdot m_i(U)

Where:

- :math:`w_i`: Weight for metric i (:math:`\sum w_i = 1, w_i \geq 0`)
- :math:`m_i`: Individual metric function
- :math:`Q`: Overall quality score ∈ [0, 1]

.. automodule:: udl_rating_framework.core.confidence
   :members:
   :undoc-members:
   :show-inheritance:

The ``ConfidenceCalculator`` computes certainty from prediction entropy:

**Mathematical Definition**:

.. math::

   C = 1 - \frac{H(p)}{H_{max}}

Where:

- :math:`H(p) = -\sum_i p_i \log(p_i)`: Shannon entropy of prediction distribution
- :math:`H_{max} = \log(n)`: Maximum entropy for n classes
- :math:`C \in [0, 1]`: Confidence score

Quality Metrics
---------------

Base Metric Class
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.metrics.base
   :members:
   :undoc-members:
   :show-inheritance:

All quality metrics inherit from ``QualityMetric`` and must satisfy:

1. **Boundedness**: :math:`\forall u \in UDL\_Space, 0 \leq f(u) \leq 1`
2. **Determinism**: :math:`f(u_1) = f(u_2)` if :math:`u_1 = u_2`
3. **Computability**: f must terminate in polynomial time

Consistency Metric
~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.metrics.consistency
   :members:
   :undoc-members:
   :show-inheritance:

**Mathematical Definition**:

.. math::

   Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}

**Algorithm**:

1. Build dependency graph from grammar rules
2. Detect cycles using DFS (O(V + E))
3. Identify contradictions using constraint analysis
4. Normalize by total rule count

Completeness Metric
~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.metrics.completeness
   :members:
   :undoc-members:
   :show-inheritance:

**Mathematical Definition**:

.. math::

   Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}

**Algorithm**:

1. Extract defined constructs from grammar
2. Determine required constructs based on language type
3. Compute coverage ratio

Expressiveness Metric
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.metrics.expressiveness
   :members:
   :undoc-members:
   :show-inheritance:

**Mathematical Definition**:

.. math::

   Expressiveness(U) = \frac{Chomsky\_Level + Complexity\_Score}{2}

**Chomsky Levels**:

- Type-3 (Regular): 0.33
- Type-2 (Context-Free): 0.67
- Type-1 (Context-Sensitive): 0.83
- Type-0 (Unrestricted): 1.0

Structural Coherence Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.metrics.structural_coherence
   :members:
   :undoc-members:
   :show-inheritance:

**Mathematical Definition**:

.. math::

   Structural\_Coherence(U) = 1 - \frac{H(G)}{H_{max}}

Where:

- :math:`H(G)`: Shannon entropy of grammar graph structure
- :math:`H_{max}`: Maximum possible entropy (:math:`\log_2|V|`)

Advanced Metrics
~~~~~~~~~~~~~~~~

Semantic Similarity Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.core.metrics.semantic_similarity
   :members:
   :undoc-members:
   :show-inheritance:

Readability Metric
^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.core.metrics.readability
   :members:
   :undoc-members:
   :show-inheritance:

Maintainability Metric
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.core.metrics.maintainability
   :members:
   :undoc-members:
   :show-inheritance:

Cross-Language Compatibility Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.core.metrics.cross_language_compatibility
   :members:
   :undoc-members:
   :show-inheritance:

Evolution Tracking Metric
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.core.metrics.evolution_tracking
   :members:
   :undoc-members:
   :show-inheritance:

Input/Output Modules
--------------------

File Discovery
~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.io.file_discovery
   :members:
   :undoc-members:
   :show-inheritance:

**Supported Extensions**:

- ``.udl`` - UDL files
- ``.dsl`` - Domain Specific Language files
- ``.grammar`` - Grammar definition files
- ``.ebnf`` - Extended Backus-Naur Form files
- ``.txt`` - Plain text grammar files
- ``.g4`` - ANTLR grammar files
- ``.peg`` - PEG grammar files
- ``.y``, ``.yacc`` - Yacc/Bison files
- ``.abnf`` - ABNF grammar files
- ``.rr`` - Railroad diagram format files

Input Validation
~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.io.input_validation
   :members:
   :undoc-members:
   :show-inheritance:

Report Generation
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.io.report_generator
   :members:
   :undoc-members:
   :show-inheritance:

**Supported Output Formats**:

- JSON - Machine-readable structured output
- CSV - Spreadsheet-compatible format
- HTML - Interactive web reports with visualizations
- Markdown - Documentation-friendly format

Model Modules
-------------

CTM Adapter
~~~~~~~~~~~

.. automodule:: udl_rating_framework.models.ctm_adapter
   :members:
   :undoc-members:
   :show-inheritance:

The ``UDLRatingCTM`` class adapts the Continuous Thought Machine architecture for UDL quality prediction.

**Architecture**:

1. Token Embedding: :math:`E: Token \rightarrow \mathbb{R}^d`
2. CTM Core: Processes sequence with T iterations
3. Synchronization: Extracts S(t) at each iteration
4. Rating Head: Maps final S(T) → [0,1]

**Example Usage**:

.. code-block:: python

   from udl_rating_framework.models.ctm_adapter import UDLRatingCTM
   
   model = UDLRatingCTM(
       vocab_size=1000,
       d_model=256,
       d_input=64,
       iterations=20,
       n_synch_out=32
   )
   
   # Forward pass
   ratings, certainties = model(token_ids)

Training Modules
----------------

Training Pipeline
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.training_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Loss Function**:

.. math::

   L = \alpha \cdot L_{rating} + \beta \cdot L_{confidence}

Where:

- :math:`L_{rating}`: MSE(predicted, ground_truth)
- :math:`L_{confidence}`: Calibration loss

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.hyperparameter_optimization
   :members:
   :undoc-members:
   :show-inheritance:

CTM-aware hyperparameter optimization leveraging synchronization, temporal, and memory parameters.

Ensemble Methods
~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.ensemble_methods
   :members:
   :undoc-members:
   :show-inheritance:

CTM ensemble methods with synchronization diversity strategies.

Transfer Learning
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.transfer_learning
   :members:
   :undoc-members:
   :show-inheritance:

CTM transfer learning that transfers temporal dynamics and synchronization patterns.

Active Learning
~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.active_learning
   :members:
   :undoc-members:
   :show-inheritance:

CTM active learning using synchronization entropy and neuron diversity sampling.

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.training.uncertainty_quantification
   :members:
   :undoc-members:
   :show-inheritance:

CTM uncertainty quantification via synchronization matrices and neuron-level model analysis.

Evaluation Modules
------------------

Comparison Engine
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.evaluation.comparison
   :members:
   :undoc-members:
   :show-inheritance:

**Statistical Tests**:

- t-test for parametric comparisons
- Wilcoxon signed-rank test for non-parametric comparisons
- Cohen's d for effect size computation

Evaluation Suite
~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.evaluation.evaluation_suite
   :members:
   :undoc-members:
   :show-inheritance:

**Evaluation Metrics**:

- k-fold cross-validation (k ≥ 5)
- Pearson and Spearman correlations with 95% CIs
- Expected Calibration Error (ECE)
- Bootstrap confidence intervals (B ≥ 1000)

Visualization Modules
---------------------

Activation Visualizer
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.visualization.activation_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Synchronization Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.visualization.synchronization_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Web Visualizer
~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.visualization.web_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Interactive web-based visualizations using D3.js.

WebGL Visualizer
~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.visualization.webgl_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

3D visualization of grammar graphs with WebGL.

Real-time Metrics Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.visualization.realtime_metrics
   :members:
   :undoc-members:
   :show-inheritance:

Real-time metric computation visualization.

Analytics Modules
-----------------

Time Series Analyzer
~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.analytics.time_series_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Time-series analysis of UDL quality evolution.

Portfolio Analyzer
~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.analytics.portfolio_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Comparative analysis across project portfolios.

Trend Predictor
~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.analytics.trend_predictor
   :members:
   :undoc-members:
   :show-inheritance:

Quality trend prediction using historical data.

Improvement Advisor
~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.analytics.improvement_advisor
   :members:
   :undoc-members:
   :show-inheritance:

Automated quality improvement suggestions.

BI Exporter
~~~~~~~~~~~

.. automodule:: udl_rating_framework.analytics.bi_exporter
   :members:
   :undoc-members:
   :show-inheritance:

Export to business intelligence tools.

Integration Modules
-------------------

Git Hooks
~~~~~~~~~

.. automodule:: udl_rating_framework.integration.git_hooks
   :members:
   :undoc-members:
   :show-inheritance:

Git hooks for automatic UDL quality checking.

CI/CD Integration
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.integration.cicd
   :members:
   :undoc-members:
   :show-inheritance:

CI/CD pipeline integration (GitHub Actions, Jenkins).

IDE Plugin
~~~~~~~~~~

.. automodule:: udl_rating_framework.integration.ide_plugin
   :members:
   :undoc-members:
   :show-inheritance:

IDE plugins for real-time quality feedback.

LSP Server
~~~~~~~~~~

.. automodule:: udl_rating_framework.integration.lsp_server
   :members:
   :undoc-members:
   :show-inheritance:

Integration with Language Server Protocol (LSP).

Batch Processor
~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.integration.batch_processor
   :members:
   :undoc-members:
   :show-inheritance:

Batch processing workflows for large codebases.

Performance Modules
-------------------

Caching
~~~~~~~

.. automodule:: udl_rating_framework.core.caching
   :members:
   :undoc-members:
   :show-inheritance:

Caching for parsed UDL representations.

Multiprocessing
~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.multiprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Multiprocessing for batch file processing.

GPU Acceleration
~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.gpu_acceleration
   :members:
   :undoc-members:
   :show-inheritance:

GPU acceleration for CTM inference.

Distributed Computing
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.distributed
   :members:
   :undoc-members:
   :show-inheritance:

Distributed computing support using Ray/Dask.

Streaming Processing
~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.streaming
   :members:
   :undoc-members:
   :show-inheritance:

Streaming processing for large UDL files.

Memory Mapping
~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.memory_mapping
   :members:
   :undoc-members:
   :show-inheritance:

Memory-mapped file processing.

Incremental Computation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.core.incremental
   :members:
   :undoc-members:
   :show-inheritance:

Incremental computation for updated UDLs.

Validation Modules
------------------

Formal Verification
~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.validation.formal_verification
   :members:
   :undoc-members:
   :show-inheritance:

Formal verification of metric properties.

Dataset Benchmark
~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.validation.dataset_benchmark
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark against academic UDL quality datasets.

CLI Interface
-------------

Main CLI
~~~~~~~~

.. automodule:: udl_rating_framework.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

**Available Commands**:

- ``rate`` - Rate UDL files or directories
- ``train`` - Train CTM model on UDL dataset
- ``compare`` - Compare multiple UDLs
- ``evaluate`` - Evaluate trained model
- ``analytics`` - Run analytics on UDL quality data
- ``integrate`` - Set up integrations (git hooks, CI/CD)

Configuration
~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.cli.config
   :members:
   :undoc-members:
   :show-inheritance:

**Configuration File Format** (YAML):

.. code-block:: yaml

   # Metric weights
   weights:
     consistency: 0.3
     completeness: 0.3
     expressiveness: 0.2
     structural_coherence: 0.2
   
   # File discovery settings
   file_discovery:
     extensions: ['.udl', '.dsl', '.grammar', '.ebnf', '.g4', '.peg']
     recursive: true
     ignore_patterns: ['*.tmp', '.*']
   
   # Model settings
   model:
     use_ctm: true
     checkpoint: './models/udl_rating_ctm.pt'
     device: 'auto'  # 'cpu', 'cuda', or 'auto'
   
   # Output settings
   output:
     format: 'json'
     precision: 6
     include_trace: true

Commands
~~~~~~~~

Rate Command
^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.rate
   :members:
   :undoc-members:
   :show-inheritance:

Train Command
^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.train
   :members:
   :undoc-members:
   :show-inheritance:

Compare Command
^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.compare
   :members:
   :undoc-members:
   :show-inheritance:

Evaluate Command
^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

Analytics Command
^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.analytics
   :members:
   :undoc-members:
   :show-inheritance:

Integration Command
^^^^^^^^^^^^^^^^^^^

.. automodule:: udl_rating_framework.cli.integration
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks
----------

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: udl_rating_framework.benchmarks.performance_benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

**Benchmark Categories**:

- Metric computation time vs UDL size
- CTM inference time vs sequence length
- Batch processing throughput
- Memory usage profiling
