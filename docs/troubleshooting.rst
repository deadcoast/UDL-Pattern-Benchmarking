Troubleshooting Guide
====================

This guide covers common issues and their solutions when using the UDL Rating Framework.

Installation Issues
-------------------

Python Version Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ImportError or syntax errors when importing the framework.

**Solution**: Ensure you're using Python 3.8 or later:

.. code-block:: bash

   python --version
   # Should show Python 3.8.x or higher

If using an older version, upgrade Python or use a virtual environment:

.. code-block:: bash

   # Create virtual environment with Python 3.8+
   python3.8 -m venv udl_env
   source udl_env/bin/activate  # On Windows: udl_env\Scripts\activate
   pip install udl-rating-framework

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

**Problem**: ModuleNotFoundError for required packages.

**Solution**: Install missing dependencies:

.. code-block:: bash

   # Install core dependencies
   pip install torch networkx numpy scipy hypothesis
   
   # For visualization
   pip install matplotlib seaborn
   
   # For Jupyter notebooks
   pip install jupyter ipykernel

**Problem**: CUDA/GPU related errors with PyTorch.

**Solution**: Install CPU-only PyTorch if GPU is not needed:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

UDL Parsing Issues
------------------

Tokenization Errors
~~~~~~~~~~~~~~~~~~~

**Problem**: UDL fails to parse with unexpected tokens.

**Symptoms**:
.. code-block:: python

   UDLRepresentation(udl_text, "file.udl")
   # Raises parsing error

**Solution**: Check UDL syntax and supported patterns:

.. code-block:: python

   from udl_rating_framework.core.representation import UDLTokenizer
   
   tokenizer = UDLTokenizer()
   tokens = tokenizer.tokenize(udl_text)
   
   # Examine tokens for issues
   for token in tokens[:10]:  # First 10 tokens
       print(f"{token.type}: '{token.text}' at line {token.line}")

**Common Issues**:
* Unsupported characters or symbols
* Malformed string literals (missing quotes)
* Invalid grammar rule syntax

**Fix**: Ensure UDL follows supported patterns:

.. code-block:: text

   # Correct syntax
   Rule ::= 'terminal' | NonTerminal
   
   # Incorrect syntax  
   Rule = terminal  # Missing quotes and wrong operator

Grammar Rule Extraction
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Grammar rules not extracted correctly.

**Solution**: Verify rule format:

.. code-block:: python

   udl = UDLRepresentation(udl_text, "file.udl")
   rules = udl.get_grammar_rules()
   
   print(f"Found {len(rules)} rules:")
   for rule in rules:
       print(f"  {rule.lhs} ::= {' '.join(rule.rhs)}")

**Expected Format**:
* Use ``::=``, ``:=``, or ``=`` for production rules
* Separate alternatives with ``|``
* Quote terminal symbols: ``'terminal'``

Metric Computation Issues
-------------------------

Unexpected Metric Values
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Metrics return unexpected values (e.g., always 0 or 1).

**Debugging Steps**:

1. **Check UDL Structure**:

.. code-block:: python

   udl = UDLRepresentation(udl_text, "file.udl")
   
   print(f"Tokens: {len(udl.get_tokens())}")
   print(f"Rules: {len(udl.get_grammar_rules())}")
   print(f"Graph nodes: {udl.get_grammar_graph().number_of_nodes()}")
   print(f"Graph edges: {udl.get_grammar_graph().number_of_edges()}")

2. **Examine Individual Metrics**:

.. code-block:: python

   from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
   
   metric = ConsistencyMetric()
   
   # Check intermediate values
   rules = udl.get_grammar_rules()
   graph = udl.get_grammar_graph()
   
   cycles = metric.detect_cycles(graph)
   contradictions = metric.find_contradictions(rules)
   
   print(f"Cycles: {len(cycles)}")
   print(f"Contradictions: {len(contradictions)}")
   print(f"Rules: {len(rules)}")

3. **Validate Metric Properties**:

.. code-block:: python

   score = metric.compute(udl)
   
   # Check boundedness
   assert 0.0 <= score <= 1.0, f"Score {score} not in [0,1]"
   
   # Check determinism
   scores = [metric.compute(udl) for _ in range(5)]
   assert all(abs(s - scores[0]) < 1e-10 for s in scores), "Non-deterministic"

Consistency Metric Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Consistency always returns 1.0 even for broken grammars.

**Possible Causes**:
* Grammar graph not built correctly
* Cycle detection not working
* Contradiction detection too strict

**Solution**: Debug step by step:

.. code-block:: python

   from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
   import networkx as nx
   
   metric = ConsistencyMetric()
   graph = udl.get_grammar_graph()
   
   # Check graph structure
   print("Graph nodes:", list(graph.nodes()))
   print("Graph edges:", list(graph.edges()))
   
   # Manual cycle check
   try:
       cycles = list(nx.simple_cycles(graph))
       print(f"NetworkX cycles: {cycles}")
   except Exception as e:
       print(f"Cycle detection error: {e}")
   
   # Check rules for contradictions
   rules = udl.get_grammar_rules()
   for rule in rules:
       print(f"Rule: {rule.lhs} ::= {rule.rhs}")

Completeness Metric Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Completeness metric returns unexpected low values.

**Cause**: Different construct classification between manual calculation and implementation.

**Solution**: Examine construct extraction:

.. code-block:: python

   from udl_rating_framework.core.metrics.completeness import CompletenessMetric
   
   metric = CompletenessMetric()
   
   # Check what constructs are being detected
   defined = metric.extract_defined_constructs(udl)
   required = metric.get_required_constructs("arithmetic")  # or appropriate type
   
   print("Defined constructs:", defined)
   print("Required constructs:", required)
   print("Coverage:", len(defined & required) / len(required))

Performance Issues
------------------

Slow Metric Computation
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Metrics take too long to compute on large UDLs.

**Solutions**:

1. **Profile the computation**:

.. code-block:: python

   import time
   from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
   
   metric = ConsistencyMetric()
   
   start = time.time()
   score = metric.compute(udl)
   end = time.time()
   
   print(f"Computation took {end - start:.3f} seconds")

2. **Check UDL size**:

.. code-block:: python

   print(f"UDL size: {len(udl.source_text)} characters")
   print(f"Tokens: {len(udl.get_tokens())}")
   print(f"Rules: {len(udl.get_grammar_rules())}")

3. **Use caching for repeated computations**:

.. code-block:: python

   from functools import lru_cache
   
   class CachedMetric(ConsistencyMetric):
       @lru_cache(maxsize=128)
       def compute(self, udl_hash):
           return super().compute(udl)

Memory Issues
~~~~~~~~~~~~~

**Problem**: Out of memory errors with large UDL files.

**Solutions**:

1. **Process UDLs in chunks**:

.. code-block:: python

   def process_large_udl(file_path, chunk_size=10000):
       with open(file_path, 'r') as f:
           content = f.read()
       
       if len(content) > chunk_size:
           # Split into smaller sections
           chunks = [content[i:i+chunk_size] 
                    for i in range(0, len(content), chunk_size)]
           
           scores = []
           for chunk in chunks:
               udl = UDLRepresentation(chunk, f"{file_path}_chunk")
               score = pipeline.rate_udl(udl)
               scores.append(score.overall_score)
           
           return sum(scores) / len(scores)
       else:
           udl = UDLRepresentation(content, file_path)
           return pipeline.rate_udl(udl).overall_score

2. **Clear caches periodically**:

.. code-block:: python

   import gc
   
   # After processing many UDLs
   gc.collect()

CTM Model Issues
----------------

Training Failures
~~~~~~~~~~~~~~~~~

**Problem**: CTM model fails to train or converges poorly.

**Solutions**:

1. **Check data quality**:

.. code-block:: python

   # Verify training data
   for udl, target in train_dataset:
       assert 0.0 <= target <= 1.0, f"Invalid target: {target}"
       assert len(udl.get_tokens()) > 0, "Empty UDL"

2. **Adjust hyperparameters**:

.. code-block:: python

   from udl_rating_framework.training.training_pipeline import TrainingPipeline
   
   trainer = TrainingPipeline(
       model,
       learning_rate=1e-4,  # Lower learning rate
       batch_size=16,       # Smaller batch size
       alpha=0.8,          # Adjust loss weights
       beta=0.2
   )

3. **Monitor training progress**:

.. code-block:: python

   # Add logging
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Train with validation
   trainer.train(train_dataset, val_dataset, epochs=100)

CUDA/GPU Issues
~~~~~~~~~~~~~~~

**Problem**: CUDA out of memory or device errors.

**Solutions**:

1. **Use CPU-only mode**:

.. code-block:: python

   import torch
   
   # Force CPU usage
   device = torch.device('cpu')
   model = model.to(device)

2. **Reduce batch size**:

.. code-block:: python

   trainer = TrainingPipeline(model, batch_size=8)  # Smaller batches

3. **Clear GPU cache**:

.. code-block:: python

   import torch
   
   if torch.cuda.is_available():
       torch.cuda.empty_cache()

File I/O Issues
---------------

File Not Found Errors
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Cannot find UDL files or examples.

**Solution**: Check file paths and working directory:

.. code-block:: python

   import os
   from pathlib import Path
   
   # Check current directory
   print("Current directory:", os.getcwd())
   
   # Check if file exists
   file_path = "examples/udl_examples/simple_calculator.udl"
   if not Path(file_path).exists():
       print(f"File not found: {file_path}")
       
       # Find the correct path
       for root, dirs, files in os.walk("."):
           if "simple_calculator.udl" in files:
               print(f"Found at: {os.path.join(root, 'simple_calculator.udl')}")

Permission Errors
~~~~~~~~~~~~~~~~~

**Problem**: Cannot read/write files due to permissions.

**Solution**: Check file permissions and use appropriate paths:

.. code-block:: python

   import os
   import stat
   
   file_path = "my_udl.udl"
   
   # Check permissions
   if os.path.exists(file_path):
       file_stat = os.stat(file_path)
       print(f"File permissions: {stat.filemode(file_stat.st_mode)}")
   
   # Use user directory for output
   output_dir = Path.home() / "udl_reports"
   output_dir.mkdir(exist_ok=True)

Testing and Validation Issues
-----------------------------

Property-Based Test Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Property-based tests fail with counterexamples.

**Analysis Process**:

1. **Examine the counterexample**:

.. code-block:: python

   # If test fails with specific input
   failing_udl_text = "A ::= B\nB ::= A"  # Example cycle
   
   udl = UDLRepresentation(failing_udl_text, "test.udl")
   metric = ConsistencyMetric()
   score = metric.compute(udl)
   
   print(f"Score: {score}")
   print(f"Expected: < 1.0 (due to cycle)")

2. **Check if it's a valid counterexample**:

.. code-block:: python

   # Verify the issue manually
   graph = udl.get_grammar_graph()
   cycles = list(nx.simple_cycles(graph))
   print(f"Cycles found: {cycles}")

3. **Fix the implementation or test**:

.. code-block:: python

   # If implementation is wrong, fix it
   # If test is wrong, adjust the test constraints

Test Environment Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Tests pass locally but fail in CI/CD.

**Common Causes**:
* Different Python versions
* Missing test dependencies
* Platform-specific behavior

**Solutions**:

1. **Pin dependency versions**:

.. code-block:: bash

   pip freeze > requirements-test.txt

2. **Use consistent test environment**:

.. code-block:: bash

   # In CI configuration
   python -m pytest tests/ -v --tb=short

3. **Add platform-specific skips**:

.. code-block:: python

   import sys
   import pytest
   
   @pytest.mark.skipif(sys.platform == "win32", reason="Windows-specific issue")
   def test_unix_only_feature():
       pass

Getting Help
------------

If you encounter issues not covered here:

1. **Check the logs**: Enable debug logging to see detailed information
2. **Create minimal examples**: Isolate the problem with the smallest possible UDL
3. **Check GitHub issues**: Search for similar problems in the project repository
4. **Ask for help**: Create a new issue with:
   - Python version and platform
   - Framework version
   - Minimal code to reproduce the issue
   - Full error traceback
   - Expected vs actual behavior

Debug Logging
~~~~~~~~~~~~~

Enable detailed logging for troubleshooting:

.. code-block:: python

   import logging
   
   # Enable debug logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   # Now run your code
   udl = UDLRepresentation(udl_text, "debug.udl")
   report = pipeline.rate_udl(udl)