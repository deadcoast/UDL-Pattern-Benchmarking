UDL Rating Framework Documentation
===================================

The UDL Rating Framework is a mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs) and Domain Specific Languages (DSLs).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   mathematical_framework
   examples
   troubleshooting

Overview
--------

The framework provides objective, reproducible quality assessments based on four key metrics:

* **Consistency**: Internal coherence and absence of contradictions
* **Completeness**: Coverage of required language constructs  
* **Expressiveness**: Language power and complexity
* **Structural Coherence**: Organizational quality and entropy

Key Features
------------

* **Mathematical Foundation**: Every rating is traceable to formal computations
* **Property-Based Testing**: Comprehensive validation using property-based tests
* **CTM Integration**: Optional neural approximation using Continuous Thought Machine
* **Extensible Architecture**: Plugin system for custom metrics
* **Multiple Output Formats**: JSON, CSV, HTML reports with visualizations

Quick Start
-----------

.. code-block:: python

   from udl_rating_framework.core.representation import UDLRepresentation
   from udl_rating_framework.core.pipeline import RatingPipeline

   # Load a UDL
   udl = UDLRepresentation.from_file("my_language.udl")
   
   # Rate it
   pipeline = RatingPipeline()
   report = pipeline.rate_udl(udl)
   
   print(f"Quality Score: {report.overall_score:.3f}")

Installation
------------

.. code-block:: bash

   pip install udl-rating-framework

Or for development:

.. code-block:: bash

   git clone https://github.com/your-org/udl-rating-framework
   cd udl-rating-framework
   pip install -e .

Mathematical Foundation
-----------------------

Each metric is defined by precise mathematical formulas:

**Consistency Metric**:

.. math::

   Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}

**Completeness Metric**:

.. math::

   Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}

**Expressiveness Metric**:

.. math::

   Expressiveness(U) = \frac{Chomsky\_Level + Complexity\_Score}{2}

**Structural Coherence Metric**:

.. math::

   Structural\_Coherence(U) = 1 - \frac{H(G)}{H_{max}}

Where :math:`H(G)` is the Shannon entropy of the grammar graph.

**Overall Quality Score**:

.. math::

   Q(U) = \sum_{i} w_i \cdot m_i(U)

Where :math:`w_i` are metric weights and :math:`m_i` are individual metrics.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`