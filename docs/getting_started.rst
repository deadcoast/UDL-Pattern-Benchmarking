Getting Started
===============

This guide will help you get up and running with the UDL Rating Framework quickly.

Installation
------------

The UDL Rating Framework requires Python 3.8 or later. Install using pip:

.. code-block:: bash

   pip install udl-rating-framework

For development installation:

.. code-block:: bash

   git clone https://github.com/your-org/udl-rating-framework
   cd udl-rating-framework
   pip install -e .

Basic Usage
-----------

Rating a Single UDL
~~~~~~~~~~~~~~~~~~~~

The simplest way to rate a UDL is using the high-level API:

.. code-block:: python

   from udl_rating_framework.core.representation import UDLRepresentation
   from udl_rating_framework.core.pipeline import RatingPipeline

   # Load UDL from string
   udl_content = """
   Expression ::= Term (('+' | '-') Term)*
   Term ::= Factor (('*' | '/') Factor)*
   Factor ::= Number | '(' Expression ')'
   Number ::= Digit+
   Digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
   """
   
   udl = UDLRepresentation(udl_content, "calculator.udl")
   
   # Create rating pipeline
   pipeline = RatingPipeline()
   
   # Rate the UDL
   report = pipeline.rate_udl(udl)
   
   # Display results
   print(f"Overall Quality Score: {report.overall_score:.3f}")
   print(f"Confidence: {report.confidence:.3f}")
   
   for metric, score in report.metric_scores.items():
       print(f"{metric}: {score:.3f}")

Rating Multiple UDLs
~~~~~~~~~~~~~~~~~~~~

To rate multiple UDL files in a directory:

.. code-block:: python

   from udl_rating_framework.io.file_discovery import FileDiscovery
   from udl_rating_framework.core.pipeline import RatingPipeline

   # Discover UDL files
   discovery = FileDiscovery()
   udl_files = discovery.discover_files("path/to/udl/directory")
   
   # Rate all files
   pipeline = RatingPipeline()
   reports = []
   
   for file_path in udl_files:
       with open(file_path, 'r') as f:
           content = f.read()
       
       udl = UDLRepresentation(content, file_path)
       report = pipeline.rate_udl(udl)
       reports.append(report)
   
   # Generate summary
   avg_score = sum(r.overall_score for r in reports) / len(reports)
   print(f"Average quality score: {avg_score:.3f}")

Understanding the Metrics
-------------------------

The framework evaluates UDLs using four key metrics:

Consistency Metric
~~~~~~~~~~~~~~~~~~

Measures internal coherence by detecting:

* **Cycles**: Circular dependencies in grammar rules
* **Contradictions**: Rules that cannot both be satisfied

.. math::

   Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}

**Example**: A grammar with no cycles or contradictions scores 1.0.

Completeness Metric
~~~~~~~~~~~~~~~~~~~

Measures coverage of required language constructs:

.. math::

   Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}

**Example**: An arithmetic language defining expressions, terms, factors, and numbers scores highly.

Expressiveness Metric
~~~~~~~~~~~~~~~~~~~~~

Measures language power using formal language theory:

.. math::

   Expressiveness(U) = \frac{Chomsky\_Level + Complexity\_Score}{2}

**Chomsky Levels**:
* Type-3 (Regular): 0.33
* Type-2 (Context-Free): 0.67  
* Type-1 (Context-Sensitive): 0.83
* Type-0 (Unrestricted): 1.0

Structural Coherence Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures organizational quality using information theory:

.. math::

   Structural\_Coherence(U) = 1 - \frac{H(G)}{H_{max}}

Where H(G) is the Shannon entropy of the grammar graph structure.

Command Line Interface
----------------------

The framework provides a CLI for common tasks:

Rating UDLs
~~~~~~~~~~~~

.. code-block:: bash

   # Rate a single file
   udl-rate my_language.udl
   
   # Rate all UDLs in a directory
   udl-rate --directory ./udl_files/
   
   # Generate HTML report
   udl-rate --output-format html --output report.html my_language.udl

Comparing UDLs
~~~~~~~~~~~~~~

.. code-block:: bash

   # Compare two UDLs
   udl-compare lang1.udl lang2.udl
   
   # Compare with statistical tests
   udl-compare --statistical-tests lang1.udl lang2.udl

Training CTM Model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train CTM model on UDL dataset
   udl-train --dataset ./training_data/ --epochs 100 --output model.pt

Evaluating Model
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate trained model
   udl-evaluate --model model.pt --test-set ./test_data/

Configuration
-------------

Create a configuration file ``config.yaml``:

.. code-block:: yaml

   # Metric weights
   weights:
     consistency: 0.3
     completeness: 0.3
     expressiveness: 0.2
     structural_coherence: 0.2
   
   # File discovery settings
   file_discovery:
     extensions: ['.udl', '.dsl', '.grammar', '.ebnf']
     recursive: true
     ignore_patterns: ['*.tmp', '.*']
   
   # Output settings
   output:
     format: 'json'
     precision: 6
     include_trace: true

Use the configuration:

.. code-block:: python

   from udl_rating_framework.cli.config import Config
   
   config = Config.from_file("config.yaml")
   pipeline = RatingPipeline(config=config)

Working with Examples
---------------------

The framework includes example UDLs for learning and testing:

.. code-block:: python

   from pathlib import Path
   
   # Load example UDLs
   examples_dir = Path("examples/udl_examples")
   
   # High-quality examples
   calculator = UDLRepresentation.from_file(examples_dir / "simple_calculator.udl")
   json_subset = UDLRepresentation.from_file(examples_dir / "json_subset.udl")
   
   # Low-quality examples  
   broken_grammar = UDLRepresentation.from_file(examples_dir / "broken_grammar.udl")
   
   # Rate and compare
   pipeline = RatingPipeline()
   
   calc_report = pipeline.rate_udl(calculator)
   broken_report = pipeline.rate_udl(broken_grammar)
   
   print(f"Calculator: {calc_report.overall_score:.3f}")
   print(f"Broken: {broken_report.overall_score:.3f}")

Next Steps
----------

* Explore the :doc:`examples` for detailed use cases
* Read the :doc:`api_reference` for complete API documentation
* Check the :doc:`mathematical_framework` for theoretical foundations
* See :doc:`troubleshooting` for common issues and solutions

Advanced Topics
---------------

Custom Metrics
~~~~~~~~~~~~~~

Create custom metrics by extending the base class:

.. code-block:: python

   from udl_rating_framework.core.metrics.base import QualityMetric
   
   class CustomMetric(QualityMetric):
       def compute(self, udl):
           # Your metric computation
           return 0.5
       
       def get_formula(self):
           return "Custom(U) = 0.5"
       
       def get_properties(self):
           return {"bounded": True, "deterministic": True}
   
   # Register the metric
   CustomMetric.register_metric("custom")

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

The framework uses property-based testing for validation:

.. code-block:: python

   from hypothesis import given, strategies as st
   from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
   
   @given(st.text())
   def test_consistency_bounded(udl_text):
       udl = UDLRepresentation(udl_text, "test.udl")
       metric = ConsistencyMetric()
       score = metric.compute(udl)
       assert 0.0 <= score <= 1.0

CTM Integration
~~~~~~~~~~~~~~~

Train neural models to approximate mathematical metrics:

.. code-block:: python

   from udl_rating_framework.models.ctm_adapter import UDLRatingCTM
   from udl_rating_framework.training.training_pipeline import TrainingPipeline
   
   # Create CTM model
   model = UDLRatingCTM(vocab_size=1000, d_model=256)
   
   # Train on UDL dataset
   trainer = TrainingPipeline(model)
   trainer.train(train_dataset, val_dataset, epochs=100)