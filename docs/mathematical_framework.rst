Mathematical Framework
======================

This document provides the complete mathematical foundation for the UDL Rating Framework. Every rating computation is traceable to these formal definitions and proven properties.

Overview
--------

The UDL Rating Framework is built on rigorous mathematical foundations from:

- **Formal Language Theory**: Chomsky hierarchy, grammar analysis
- **Graph Theory**: Dependency analysis, cycle detection
- **Information Theory**: Shannon entropy, complexity measures
- **Set Theory**: Construct coverage, completeness analysis

Each quality metric is defined as a mathematical function with proven properties, ensuring objective and reproducible assessments.

UDL Representation
------------------

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

A User Defined Language is formally represented as a tuple:

.. math::

   U = (T, G, S, R)

Where:

- :math:`T`: Set of tokens (terminal symbols)
- :math:`G = (V, E)`: Grammar graph with vertices :math:`V` (non-terminals) and edges :math:`E` (production rules)
- :math:`S: T \rightarrow Semantics`: Semantic mapping function
- :math:`R`: Set of constraints and rules

Token Space
~~~~~~~~~~~

The token space :math:`T` is defined as:

.. math::

   T = \{t | t \in \Sigma^*, type(t) \in TokenTypes\}

Where :math:`\Sigma` is the alphabet and :math:`TokenTypes = \{KEYWORD, IDENTIFIER, OPERATOR, LITERAL, ...\}`.

Grammar Graph
~~~~~~~~~~~~~

The grammar graph :math:`G = (V, E)` represents production rule dependencies:

.. math::

   V &= \{lhs | \exists rule: lhs \rightarrow rhs\} \\
   E &= \{(lhs, symbol) | symbol \in rhs \text{ of rule } lhs \rightarrow rhs\}

Quality Metrics
---------------

Each quality metric is a function :math:`m: UDL\_Space \rightarrow [0,1]` with the following properties:

1. **Boundedness**: :math:`\forall u \in UDL\_Space, 0 \leq m(u) \leq 1`
2. **Determinism**: :math:`m(u_1) = m(u_2)` if :math:`u_1 = u_2`
3. **Computability**: :math:`m` terminates in polynomial time

Consistency Metric
~~~~~~~~~~~~~~~~~~

**Definition**: Measures internal coherence by detecting structural inconsistencies.

.. math::

   Consistency(U) = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}

**Components**:

- :math:`Cycles`: Set of strongly connected components in :math:`G` with :math:`|SCC| > 1`
- :math:`Contradictions`: Set of rule pairs :math:`\{(r_1, r_2) | conflict(r_1, r_2)\}`
- :math:`Rules`: Total number of production rules

**Cycle Detection Algorithm**:

Uses Johnson's algorithm for finding elementary cycles:

.. math::

   \text{Time Complexity: } O((V + E)(C + 1))

Where :math:`C` is the number of elementary cycles.

**Contradiction Detection**:

Two rules :math:`r_1: A \rightarrow \alpha` and :math:`r_2: A \rightarrow \beta` are contradictory if:

.. math::

   \text{same\_lhs}(r_1, r_2) \land \text{mutually\_exclusive}(\alpha, \beta)

**Properties**:
- Bounded: :math:`0 \leq Consistency(U) \leq 1`
- Monotonic in rule quality: fewer issues → higher score
- Discrete: changes with discrete rule modifications

Completeness Metric
~~~~~~~~~~~~~~~~~~~

**Definition**: Measures coverage of required language constructs.

.. math::

   Completeness(U) = \frac{|Defined\_Constructs|}{|Required\_Constructs|}

**Construct Extraction**:

.. math::

   Defined\_Constructs &= \{lhs | \exists rule: lhs \rightarrow rhs\} \cup \{terminal | terminal \in T\} \\
   Required\_Constructs &= RequiredSet(language\_type)

**Language Type Classification**:

- **Arithmetic**: :math:`\{expression, term, factor, number, operators\}`
- **JSON**: :math:`\{object, array, string, number, boolean, null\}`
- **Configuration**: :math:`\{section, property, key, value\}`
- **General**: :math:`\{program, statement, expression, identifier, literal\}`

**Properties**:
- Bounded: :math:`0 \leq Completeness(U) \leq 1`
- Monotonic: more constructs → higher score
- Additive: :math:`Completeness(U_1 \cup U_2) \geq \max(Completeness(U_1), Completeness(U_2))`

Expressiveness Metric
~~~~~~~~~~~~~~~~~~~~~

**Definition**: Measures language power using formal language theory.

.. math::

   Expressiveness(U) = \frac{Chomsky\_Level(U) + Complexity\_Score(U)}{2}

**Chomsky Hierarchy Classification**:

.. math::

   Chomsky\_Level(U) = \begin{cases}
   1.0 & \text{if Type-0 (Unrestricted)} \\
   0.83 & \text{if Type-1 (Context-Sensitive)} \\
   0.67 & \text{if Type-2 (Context-Free)} \\
   0.33 & \text{if Type-3 (Regular)}
   \end{cases}

**Classification Algorithm**:

1. Check for context-sensitive rules: :math:`\alpha A \beta \rightarrow \alpha \gamma \beta` where :math:`|\gamma| \geq 1`
2. Check for context-free rules: :math:`A \rightarrow \gamma` where :math:`A` is non-terminal
3. Check for regular rules: :math:`A \rightarrow aB` or :math:`A \rightarrow a`

**Complexity Score**:

Approximates Kolmogorov complexity using compression ratio:

.. math::

   Complexity\_Score(U) = \frac{|compressed(serialize(U))|}{|serialize(U)|}

**Properties**:
- Bounded: :math:`0 \leq Expressiveness(U) \leq 1`
- Monotonic in language power
- Continuous in complexity measures

Structural Coherence Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: Measures organizational quality using information theory.

.. math::

   Structural\_Coherence(U) = 1 - \frac{H(G)}{H_{max}}

**Shannon Entropy Calculation**:

.. math::

   H(G) = -\sum_{d \in Degrees} p(d) \log_2 p(d)

Where :math:`p(d) = \frac{|nodes\_with\_degree\_d|}{|V|}` is the probability of degree :math:`d`.

**Maximum Entropy**:

.. math::

   H_{max} = \log_2 |V|

This represents maximum disorder (uniform degree distribution).

**Degree Distribution**:

For each node :math:`v \in V`:

.. math::

   degree(v) = |in\_edges(v)| + |out\_edges(v)|

**Properties**:
- Bounded: :math:`0 \leq Structural\_Coherence(U) \leq 1`
- Inversely related to entropy: lower entropy → higher coherence
- Continuous in degree distribution changes

Metric Aggregation
------------------

**Definition**: Combines individual metrics into overall quality score.

.. math::

   Q(U) = \sum_{i=1}^{n} w_i \cdot m_i(U)

**Weight Constraints**:

.. math::

   \sum_{i=1}^{n} w_i = 1 \quad \text{and} \quad w_i \geq 0 \quad \forall i

**Properties**:

1. **Boundedness**: If :math:`m_i \in [0,1]` and :math:`\sum w_i = 1`, then :math:`Q \in [0,1]`
2. **Linearity**: :math:`Q` is linear in each metric
3. **Monotonicity**: If :math:`m_i` increases and :math:`w_i > 0`, then :math:`Q` increases

**Proof of Boundedness**:

.. math::

   Q(U) &= \sum_{i=1}^{n} w_i \cdot m_i(U) \\
   &\leq \sum_{i=1}^{n} w_i \cdot 1 \quad \text{(since } m_i \leq 1\text{)} \\
   &= \sum_{i=1}^{n} w_i = 1

Similarly, :math:`Q(U) \geq \sum_{i=1}^{n} w_i \cdot 0 = 0`.

Confidence Calculation
---------------------

**Definition**: Measures certainty of quality assessment using prediction entropy.

.. math::

   C = 1 - \frac{H(p)}{H_{max}}

**Entropy of Prediction Distribution**:

.. math::

   H(p) = -\sum_{i=1}^{n} p_i \log_2 p_i

Where :math:`p = (p_1, p_2, ..., p_n)` is the probability distribution over quality classes.

**Maximum Entropy**:

.. math::

   H_{max} = \log_2 n

**Properties**:
- Bounded: :math:`0 \leq C \leq 1`
- Maximum confidence (:math:`C = 1`) when :math:`p` is a delta distribution
- Minimum confidence (:math:`C = 0`) when :math:`p` is uniform

Computational Complexity
------------------------

Algorithm Complexities
~~~~~~~~~~~~~~~~~~~~~~

**Consistency Metric**:
- Cycle detection: :math:`O((V + E)(C + 1))` using Johnson's algorithm
- Contradiction detection: :math:`O(R^2)` where :math:`R` is number of rules
- Overall: :math:`O(V + E + R^2)`

**Completeness Metric**:
- Construct extraction: :math:`O(R + T)` where :math:`T` is number of tokens
- Set operations: :math:`O(|constructs|)`
- Overall: :math:`O(R + T)`

**Expressiveness Metric**:
- Chomsky classification: :math:`O(R)`
- Complexity approximation: :math:`O(|serialize(U)|)`
- Overall: :math:`O(R + |U|)`

**Structural Coherence Metric**:
- Degree calculation: :math:`O(V + E)`
- Entropy computation: :math:`O(V)`
- Overall: :math:`O(V + E)`

**Total Framework Complexity**: :math:`O(V + E + R^2 + |U|)`

Space Complexity
~~~~~~~~~~~~~~~~

- Grammar graph storage: :math:`O(V + E)`
- Rule storage: :math:`O(R \cdot avg\_rule\_length)`
- Token storage: :math:`O(T)`
- Overall: :math:`O(V + E + R + T)`

Theoretical Properties
---------------------

Metric Independence
~~~~~~~~~~~~~~~~~~

**Theorem**: The four quality metrics are mathematically independent.

**Proof Sketch**: 
- Consistency depends only on graph structure (cycles, contradictions)
- Completeness depends only on construct coverage
- Expressiveness depends only on grammar type and complexity
- Structural Coherence depends only on degree distribution entropy

Counter-examples can be constructed where any three metrics are fixed while the fourth varies.

Convergence Properties
~~~~~~~~~~~~~~~~~~~~~

**Theorem**: For finite UDLs, all metrics converge to stable values.

**Proof**: All metrics are computed from finite discrete structures (tokens, rules, graph nodes/edges), ensuring termination and deterministic results.

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~

**Consistency Sensitivity**:

.. math::

   \frac{\partial Consistency}{\partial |Cycles|} = -\frac{1}{|Rules| + 1}

**Completeness Sensitivity**:

.. math::

   \frac{\partial Completeness}{\partial |Defined|} = \frac{1}{|Required|}

**Expressiveness Sensitivity**:

.. math::

   \frac{\partial Expressiveness}{\partial Chomsky\_Level} = \frac{1}{2}

**Structural Coherence Sensitivity**:

.. math::

   \frac{\partial Structural\_Coherence}{\partial H(G)} = -\frac{1}{H_{max}}

Validation and Verification
---------------------------

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~

The framework uses property-based testing to verify mathematical properties:

**Property 1: Metric Boundedness**

.. code-block:: python

   @given(udl_generator())
   def test_metric_boundedness(udl):
       for metric in all_metrics:
           score = metric.compute(udl)
           assert 0.0 <= score <= 1.0

**Property 2: Metric Determinism**

.. code-block:: python

   @given(udl_generator())
   def test_metric_determinism(udl):
       for metric in all_metrics:
           scores = [metric.compute(udl) for _ in range(10)]
           assert all(s == scores[0] for s in scores)

**Property 3: Aggregation Correctness**

.. code-block:: python

   @given(metric_values(), weights())
   def test_aggregation_correctness(values, weights):
       result = aggregate(values, weights)
       expected = sum(w * v for w, v in zip(weights, values))
       assert abs(result - expected) < 1e-10

Mathematical Verification
~~~~~~~~~~~~~~~~~~~~~~~~

Each metric implementation is verified against hand-calculated examples:

1. **Load example UDL** with known structure
2. **Manually calculate** metric value using mathematical formula
3. **Compare with implementation** within numerical precision (:math:`\epsilon = 10^{-6}`)
4. **Verify properties** (boundedness, determinism, etc.)

Example verification for consistency metric:

.. math::

   \text{Manual: } 1 - \frac{1 + 1}{5 + 1} = 0.666667 \\
   \text{Implementation: } 0.666667 \\
   \text{Difference: } |0.666667 - 0.666667| < 10^{-6} \quad \checkmark

Literature and References
------------------------

**Formal Language Theory**:
- Hopcroft, J. E., & Ullman, J. D. (1979). *Introduction to Automata Theory, Languages, and Computation*
- Sipser, M. (2012). *Introduction to the Theory of Computation*

**Graph Theory**:
- Johnson, D. B. (1975). Finding all the elementary circuits of a directed graph. *SIAM Journal on Computing*
- Tarjan, R. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*

**Information Theory**:
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*

**Complexity Theory**:
- Li, M., & Vitányi, P. (1997). *An Introduction to Kolmogorov Complexity and Its Applications*
- Chaitin, G. J. (1987). *Algorithmic Information Theory*

**Software Metrics**:
- Fenton, N. E., & Pfleeger, S. L. (1997). *Software Metrics: A Rigorous and Practical Approach*
- IEEE Standard 1061-1998: *Standard for a Software Quality Metrics Methodology*

Conclusion
----------

The UDL Rating Framework provides a mathematically rigorous foundation for evaluating language quality. Every computation is traceable to formal definitions with proven properties, ensuring:

- **Objectivity**: No subjective assessments
- **Reproducibility**: Same input always produces same output  
- **Transparency**: All formulas and algorithms are documented
- **Verifiability**: Property-based testing validates correctness
- **Extensibility**: New metrics can be added with mathematical validation

This mathematical foundation enables confident decision-making in language design and evaluation.