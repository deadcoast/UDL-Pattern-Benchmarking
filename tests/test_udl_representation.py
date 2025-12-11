"""
Property-based and unit tests for UDL representation components.

**Feature: udl-rating-framework, Property 13: Tokenization Consistency**
**Validates: Requirements 4.1**
"""

import pytest
from hypothesis import given, strategies as st, settings
from udl_rating_framework.core.representation import (
    UDLRepresentation, 
    UDLTokenizer, 
    Token, 
    TokenType, 
    GrammarRule,
    Constraint,
    AST
)


class TestTokenizationConsistency:
    """Property-based tests for tokenization consistency."""
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_tokenization_consistency_property(self, udl_text):
        """
        **Feature: udl-rating-framework, Property 13: Tokenization Consistency**
        **Validates: Requirements 4.1**
        
        For any UDL string, tokenization must produce the same token sequence 
        on repeated calls.
        """
        tokenizer = UDLTokenizer()
        
        # Tokenize the same text multiple times
        tokens1 = tokenizer.tokenize(udl_text)
        tokens2 = tokenizer.tokenize(udl_text)
        tokens3 = tokenizer.tokenize(udl_text)
        
        # All tokenizations should be identical
        assert tokens1 == tokens2 == tokens3, (
            f"Tokenization not consistent for text: {repr(udl_text)}\n"
            f"First:  {[(t.text, t.type.value, t.position) for t in tokens1]}\n"
            f"Second: {[(t.text, t.type.value, t.position) for t in tokens2]}\n"
            f"Third:  {[(t.text, t.type.value, t.position) for t in tokens3]}"
        )
        
        # Verify that each token has consistent properties
        for t1, t2, t3 in zip(tokens1, tokens2, tokens3):
            assert t1.text == t2.text == t3.text
            assert t1.type == t2.type == t3.type
            assert t1.position == t2.position == t3.position
            assert t1.line == t2.line == t3.line
            assert t1.column == t2.column == t3.column


class TestUDLRepresentation:
    """Unit tests for UDL representation components."""
    
    def test_token_extraction_sample_udl(self):
        """Test token extraction on sample UDLs."""
        udl_text = """
        # Simple arithmetic grammar
        expr ::= term '+' expr | term
        term ::= factor '*' term | factor
        factor ::= '(' expr ')' | number
        """
        
        udl = UDLRepresentation(udl_text, "test.udl")
        tokens = udl.get_tokens()
        
        # Should have tokens for identifiers, operators, literals, etc.
        assert len(tokens) > 0
        
        # Check for expected token types
        token_types = [t.type for t in tokens]
        assert TokenType.IDENTIFIER in token_types
        assert TokenType.OPERATOR in token_types
        assert TokenType.LITERAL in token_types
        assert TokenType.COMMENT in token_types
        assert TokenType.EOF in token_types
        
        # Check specific tokens
        identifier_tokens = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert any(t.text == 'expr' for t in identifier_tokens)
        assert any(t.text == 'term' for t in identifier_tokens)
        assert any(t.text == 'factor' for t in identifier_tokens)
        
        operator_tokens = [t for t in tokens if t.type == TokenType.OPERATOR]
        assert any(t.text == '::=' for t in operator_tokens)
        assert any(t.text == '|' for t in operator_tokens)
    
    def test_grammar_graph_construction(self):
        """Test grammar graph construction."""
        udl_text = """
        expr ::= term '+' expr | term
        term ::= factor
        """
        
        udl = UDLRepresentation(udl_text, "test.udl")
        graph = udl.get_grammar_graph()
        
        # Should be a NetworkX DiGraph
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Should contain expected nodes
        nodes = list(graph.nodes())
        assert 'expr' in nodes
        assert 'term' in nodes
        assert 'factor' in nodes
        
        # Should have edges representing production rules
        edges = list(graph.edges())
        assert len(edges) > 0
    
    def test_ast_conversion(self):
        """Test AST conversion."""
        udl_text = """
        expr ::= term
        """
        
        udl = UDLRepresentation(udl_text, "test.udl")
        ast = udl.to_ast()
        
        # Should be an AST object
        assert isinstance(ast, AST)
        assert ast.node_type == "UDL"
        assert ast.value == "test.udl"
        
        # Should have children representing the grammar rules
        assert len(ast.children) > 0
    
    def test_empty_udl(self):
        """Test handling of empty UDL."""
        udl = UDLRepresentation("", "empty.udl")
        
        tokens = udl.get_tokens()
        # Should at least have EOF token
        assert len(tokens) >= 1
        assert tokens[-1].type == TokenType.EOF
        
        rules = udl.get_grammar_rules()
        assert len(rules) == 0
        
        graph = udl.get_grammar_graph()
        assert len(graph.nodes()) == 0
    
    def test_comment_only_udl(self):
        """Test UDL with only comments."""
        udl_text = """
        # This is a comment
        # Another comment
        """
        
        udl = UDLRepresentation(udl_text, "comments.udl")
        tokens = udl.get_tokens()
        
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 2
        assert comment_tokens[0].text == "# This is a comment"
        assert comment_tokens[1].text == "# Another comment"
    
    def test_complex_grammar_rules(self):
        """Test extraction of complex grammar rules."""
        udl_text = """
        statement ::= assignment | expression | block
        assignment := identifier '=' expression
        block := '{' statement* '}'
        """
        
        udl = UDLRepresentation(udl_text, "complex.udl")
        rules = udl.get_grammar_rules()
        
        # Should extract all three rules
        assert len(rules) >= 3
        
        # Check rule structure
        rule_names = [rule.lhs for rule in rules]
        assert 'statement' in rule_names
        assert 'assignment' in rule_names
        assert 'block' in rule_names
        
        # Check that RHS contains expected symbols
        statement_rule = next(rule for rule in rules if rule.lhs == 'statement')
        assert 'assignment' in statement_rule.rhs
        assert 'expression' in statement_rule.rhs
        assert 'block' in statement_rule.rhs
        assert '|' in statement_rule.rhs


class TestDataClasses:
    """Tests for data classes used in UDL representation."""
    
    def test_token_creation(self):
        """Test Token dataclass creation."""
        token = Token(
            text="identifier",
            type=TokenType.IDENTIFIER,
            position=10,
            line=2,
            column=5
        )
        
        assert token.text == "identifier"
        assert token.type == TokenType.IDENTIFIER
        assert token.position == 10
        assert token.line == 2
        assert token.column == 5
        
        # Test that Token is hashable (frozen=True)
        token_set = {token}
        assert len(token_set) == 1
    
    def test_grammar_rule_creation(self):
        """Test GrammarRule dataclass creation."""
        constraint = Constraint(
            type="semantic",
            condition="type_check",
            metadata={"rule": "type_safety"}
        )
        
        rule = GrammarRule(
            lhs="expr",
            rhs=["term", "+", "expr"],
            constraints=[constraint],
            metadata={"precedence": 1}
        )
        
        assert rule.lhs == "expr"
        assert rule.rhs == ["term", "+", "expr"]
        assert len(rule.constraints) == 1
        assert rule.constraints[0].type == "semantic"
        assert rule.metadata["precedence"] == 1
    
    def test_constraint_creation(self):
        """Test Constraint dataclass creation."""
        constraint = Constraint(
            type="syntactic",
            condition="balanced_parens"
        )
        
        assert constraint.type == "syntactic"
        assert constraint.condition == "balanced_parens"
        assert constraint.metadata == ()  # Empty tuple for no metadata
        assert constraint.get_metadata_dict() == {}  # Should return empty dict
        
        # Test that Constraint is hashable (frozen=True)
        constraint_set = {constraint}
        assert len(constraint_set) == 1
        
        # Test constraint with metadata
        constraint_with_meta = Constraint(
            type="semantic",
            condition="type_check",
            metadata={"priority": 1, "scope": "local"}
        )
        assert constraint_with_meta.type == "semantic"
        assert constraint_with_meta.condition == "type_check"
        assert constraint_with_meta.get_metadata_dict() == {"priority": 1, "scope": "local"}
        
        # Test that constraint with metadata is also hashable
        constraint_meta_set = {constraint_with_meta}
        assert len(constraint_meta_set) == 1
    
    def test_ast_creation(self):
        """Test AST class creation."""
        child1 = AST("Terminal", "identifier")
        child2 = AST("Terminal", "+")
        parent = AST("Expression", None, [child1, child2])
        
        assert parent.node_type == "Expression"
        assert parent.value is None
        assert len(parent.children) == 2
        assert parent.children[0].value == "identifier"
        assert parent.children[1].value == "+"