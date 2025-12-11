Examples and Tutorials
======================

This section provides comprehensive examples and tutorials for using the UDL Rating Framework.

Example UDL Files
-----------------

The framework includes a collection of example UDL files with varying quality levels to demonstrate different aspects of the rating system.

High Quality Examples
~~~~~~~~~~~~~~~~~~~~~

Simple Calculator Language
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A well-structured arithmetic expression language that demonstrates good grammar design principles.

**File**: ``examples/udl_examples/simple_calculator.udl``

.. code-block:: text

   # Simple Calculator Language
   Expression ::= Term (('+' | '-') Term)*
   Term ::= Factor (('*' | '/') Factor)*
   Factor ::= Number | '(' Expression ')'
   Number ::= Digit+
   Digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'

**Quality Metrics**:
- Consistency: 0.833 (has one cycle: Term→Factor→Expression→Term)
- Completeness: 0.667 (covers most required constructs)
- Expressiveness: 0.430 (context-free grammar with moderate complexity)
- Structural Coherence: 0.624 (well-organized structure)
- **Overall Score: 0.661**

JSON Subset Language
^^^^^^^^^^^^^^^^^^^^

A clean subset of JSON with comprehensive rules for objects, arrays, and values.

**File**: ``examples/udl_examples/json_subset.udl``

.. code-block:: text

   # JSON Subset Language
   Object ::= '{' (Pair (',' Pair)*)? '}'
   Pair ::= String ':' Value
   Array ::= '[' (Value (',' Value)*)? ']'
   Value ::= String | Number | Object | Array | 'true' | 'false' | 'null'
   String ::= '"' Character* '"'
   # ... more rules

**Quality Metrics**:
- Consistency: 0.800 (some cycles in recursive structure)
- Completeness: 1.000 (complete JSON subset)
- Expressiveness: 0.540 (rich recursive structure)
- Structural Coherence: 0.751 (good organization)
- **Overall Score: 0.798**

CSS Subset Language
^^^^^^^^^^^^^^^^^^^

A simplified CSS language with selectors, properties, and values.

**File**: ``examples/udl_examples/css_subset.udl``

**Quality Metrics**:
- Consistency: 1.000 (no cycles or contradictions)
- Completeness: 1.000 (complete for its scope)
- Expressiveness: 0.553 (moderate complexity)
- Structural Coherence: 0.701 (well-structured)
- **Overall Score: 0.851**

Medium Quality Examples
~~~~~~~~~~~~~~~~~~~~~~~

Configuration Language
^^^^^^^^^^^^^^^^^^^^^^^

A configuration DSL with some inconsistencies in assignment operators.

**File**: ``examples/udl_examples/config_language.udl``

**Issues**:
- Uses both ``=`` and ``:`` for property assignment
- Incomplete alphabet definition (only 'a'-'e')

**Quality Metrics**:
- Consistency: 1.000 (no major contradictions detected)
- Completeness: 1.000 (covers required constructs)
- Expressiveness: 0.525 (moderate complexity)
- Structural Coherence: 0.640 (reasonable organization)
- **Overall Score: 0.833**

Template Engine Language
^^^^^^^^^^^^^^^^^^^^^^^^

A template language with some structural issues.

**File**: ``examples/udl_examples/template_engine.udl``

**Issues**:
- Missing closing tags for blocks (endif, endfor, endwhile)
- Limited character set for text content
- No operators in expressions

**Quality Metrics**:
- Consistency: 1.000 (no contradictions)
- Completeness: 0.800 (missing some constructs)
- Expressiveness: 0.568 (limited by structural issues)
- Structural Coherence: 0.739 (decent organization)
- **Overall Score: 0.801**

Low Quality Examples
~~~~~~~~~~~~~~~~~~~~

Broken Grammar
^^^^^^^^^^^^^^

A deliberately broken grammar with cycles and contradictions.

**File**: ``examples/udl_examples/broken_grammar.udl``

**Issues**:
- Cycle: A → B → A
- Contradictory rules for ``Value`` (both 'true' and 'false')
- Missing definitions for referenced symbols

**Quality Metrics**:
- Consistency: 0.833 (1 cycle + 1 contradiction)
- Completeness: 0.571 (many missing constructs)
- Expressiveness: 0.394 (limited by brokenness)
- Structural Coherence: 0.667 (poor organization)
- **Overall Score: 0.634**

Inconsistent Rules
^^^^^^^^^^^^^^^^^^

Multiple contradictions and inconsistencies.

**File**: ``examples/udl_examples/inconsistent_rules.udl``

**Issues**:
- Multiple definitions for ``Boolean`` (true, false, yes, no)
- Three different assignment operators
- Contradictory number definitions

**Quality Metrics**:
- Consistency: 0.611 (multiple contradictions)
- Completeness: 1.000 (defines many constructs, albeit inconsistently)
- Expressiveness: 0.566 (attempts complexity but fails)
- Structural Coherence: 0.684 (mixed organization)
- **Overall Score: 0.733**

Incomplete Specification
^^^^^^^^^^^^^^^^^^^^^^^^

A minimal specification missing essential constructs.

**File**: ``examples/udl_examples/incomplete_spec.udl``

**Issues**:
- Only 3 rules total
- Missing definitions for referenced symbols
- No operators, literals, or control structures

**Quality Metrics**:
- Consistency: 1.000 (no contradictions in minimal rules)
- Completeness: 0.667 (severely incomplete)
- Expressiveness: 0.169 (very limited)
- Structural Coherence: 0.500 (minimal structure)
- **Overall Score: 0.634**

Jupyter Notebooks
-----------------

Getting Started Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``examples/notebooks/01_getting_started.ipynb``

This notebook provides a hands-on introduction to the framework:

- Basic usage with simple examples
- Understanding the four quality metrics
- Computing individual and overall scores
- Comparing good vs. bad UDL examples
- Using the rating pipeline

**Key Topics**:
- Loading and parsing UDL files
- Computing metrics step by step
- Interpreting quality scores
- Using configuration options

Mathematical Verification Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``examples/notebooks/02_mathematical_verification.ipynb``

This notebook demonstrates mathematical verification of the framework:

- Manual step-by-step metric calculations
- Comparing manual calculations with implementation
- Validating metric properties (boundedness, determinism)
- Testing with different UDL types

**Key Topics**:
- Manual consistency calculation with cycle/contradiction detection
- Manual completeness calculation with construct analysis
- Manual structural coherence calculation with Shannon entropy
- Property validation and conformance testing

Using the Examples
------------------

Loading Examples Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from udl_rating_framework.core.representation import UDLRepresentation
   from udl_rating_framework.core.pipeline import RatingPipeline

   # Load an example
   examples_dir = Path("examples/udl_examples")
   
   with open(examples_dir / "simple_calculator.udl", 'r') as f:
       content = f.read()
   
   udl = UDLRepresentation(content, "simple_calculator.udl")
   
   # Rate it
   pipeline = RatingPipeline()
   report = pipeline.rate_udl(udl)
   
   print(f"Quality Score: {report.overall_score:.3f}")

Batch Processing Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from udl_rating_framework.io.file_discovery import FileDiscovery
   
   # Process all examples
   discovery = FileDiscovery()
   udl_files = discovery.discover_files("examples/udl_examples", 
                                       extensions=['.udl'])
   
   results = []
   for file_path in udl_files:
       with open(file_path, 'r') as f:
           content = f.read()
       
       udl = UDLRepresentation(content, file_path)
       report = pipeline.rate_udl(udl)
       
       results.append({
           'file': file_path,
           'score': report.overall_score,
           'metrics': report.metric_scores
       })
   
   # Sort by quality score
   results.sort(key=lambda x: x['score'], reverse=True)
   
   print("UDL Quality Ranking:")
   for i, result in enumerate(results, 1):
       print(f"{i}. {result['file']}: {result['score']:.3f}")

Comparing Examples
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from udl_rating_framework.evaluation.comparison import ComparisonEngine
   
   # Load two examples
   good_udl = UDLRepresentation.from_file("examples/udl_examples/css_subset.udl")
   bad_udl = UDLRepresentation.from_file("examples/udl_examples/broken_grammar.udl")
   
   # Compare them
   comparison = ComparisonEngine()
   result = comparison.compare_udls([good_udl, bad_udl])
   
   print(f"Quality difference: {result.pairwise_differences[0][1]:.3f}")
   print(f"Statistical significance: p = {result.p_values[0][1]:.3f}")
   print(f"Effect size (Cohen's d): {result.effect_sizes[0][1]:.3f}")

Creating Custom Examples
-------------------------

Example Structure
~~~~~~~~~~~~~~~~~

When creating your own example UDLs, follow this structure:

1. **Header comment** describing the language purpose
2. **Grammar rules** using supported syntax (``::=``, ``:=``, or ``=``)
3. **Semantic rules** as comments with ``constraint:`` prefix
4. **Documentation file** with hand-calculated metric values

Example Template
~~~~~~~~~~~~~~~~

.. code-block:: text

   # My Custom Language
   # Description of what this language does
   
   # Grammar Rules
   Program ::= Statement+
   Statement ::= Assignment | Expression
   Assignment ::= Identifier '=' Expression
   Expression ::= Identifier | Number | String
   
   # Terminals
   Identifier ::= Letter (Letter | Digit)*
   Number ::= Digit+
   String ::= '"' Character* '"'
   Letter ::= 'a' | 'b' | 'c' | ... | 'z'
   Digit ::= '0' | '1' | '2' | ... | '9'
   Character ::= Letter | Digit | ' '
   
   # Semantic Rules
   # constraint: Variables must be declared before use
   # constraint: String literals must be properly quoted

Documentation Template
~~~~~~~~~~~~~~~~~~~~~~

Create a corresponding ``.md`` file with hand-calculated values:

.. code-block:: markdown

   # My Custom Language - Hand-Calculated Metrics
   
   ## Grammar Analysis
   - **Rules**: X production rules
   - **Cycles**: Y cycles detected
   - **Contradictions**: Z contradictions found
   
   ## Hand-Calculated Values
   
   ### Consistency: X.XXX
   Formula: 1 - (contradictions + cycles) / (rules + 1)
   Calculation: 1 - (Z + Y) / (X + 1) = X.XXX
   
   ### Completeness: X.XXX
   Required constructs: {...}
   Defined constructs: {...}
   Coverage: |defined| / |required| = X.XXX
   
   ### Expressiveness: X.XXX
   Chomsky level: Type-N → X.XX
   Complexity score: X.XX
   Average: (X.XX + X.XX) / 2 = X.XXX
   
   ### Structural Coherence: X.XXX
   Shannon entropy: H(G) = X.XX
   Max entropy: H_max = X.XX
   Coherence: 1 - H(G)/H_max = X.XXX

Validation Testing
~~~~~~~~~~~~~~~~~~

Add your example to the validation tests:

.. code-block:: python

   # In tests/test_example_validation.py
   expected_values = {
       # ... existing examples
       "my_custom_language.udl": {
           "consistency": 0.XXX,
           "completeness": 0.XXX,
           "expressiveness": 0.XXX,
           "structural_coherence": 0.XXX,
           "overall": 0.XXX
       }
   }

This ensures your example is tested for mathematical conformance with the framework's specifications.

Next Steps
----------

- Try modifying existing examples to see how metrics change
- Create your own UDL examples for your domain
- Use the examples as templates for real language design projects
- Contribute high-quality examples back to the framework