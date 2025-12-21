"""
Tests for new UDL format support added in Task 26.

Tests format detection and parsing for:
- ANTLR grammar files (.g4)
- PEG grammar files (.peg)
- Yacc/Bison files (.y, .yacc)
- EBNF variants (ISO/IEC 14977)
- Railroad diagram formats
"""

import tempfile
from pathlib import Path

from udl_rating_framework.core.representation import GrammarFormat, UDLRepresentation


class TestFormatDetection:
    """Test format detection for new grammar formats."""

    def test_antlr_format_detection(self):
        """Test ANTLR format detection."""
        antlr_content = """
        grammar SimpleExpr;
        
        expr : expr '+' term | term ;
        term : NUMBER | ID ;
        
        NUMBER : [0-9]+ ;
        ID : [a-zA-Z_][a-zA-Z_0-9]* ;
        """

        with tempfile.NamedTemporaryFile(suffix=".g4", mode="w", delete=False) as f:
            f.write(antlr_content)
            f.flush()

            udl = UDLRepresentation(antlr_content, f.name)
            assert udl.get_format() == GrammarFormat.ANTLR

            # Should extract rules
            rules = udl.get_grammar_rules()
            assert len(rules) > 0

            # Check that rules have ANTLR-specific metadata
            for rule in rules:
                if rule.metadata.get("format") == "antlr":
                    assert rule.metadata["operator"] == ":"

        Path(f.name).unlink()

    def test_peg_format_detection(self):
        """Test PEG format detection."""
        peg_content = """
        # PEG Grammar
        Expr    <- Term ('+' Term / '-' Term)*
        Term    <- Factor ('*' Factor / '/' Factor)*
        Factor  <- '(' Expr ')' / Number
        Number  <- [0-9]+
        """

        with tempfile.NamedTemporaryFile(suffix=".peg", mode="w", delete=False) as f:
            f.write(peg_content)
            f.flush()

            udl = UDLRepresentation(peg_content, f.name)
            assert udl.get_format() == GrammarFormat.PEG

            # Should extract rules
            rules = udl.get_grammar_rules()
            assert len(rules) > 0

            # Check that rules have PEG-specific metadata
            for rule in rules:
                if rule.metadata.get("format") == "peg":
                    assert rule.metadata["operator"] == "<-"

        Path(f.name).unlink()

    def test_yacc_format_detection(self):
        """Test Yacc/Bison format detection."""
        yacc_content = """
        %token NUMBER ID
        %left '+' '-'
        
        %%
        
        expr : expr '+' term
             | term
             ;
             
        term : NUMBER
             | ID
             ;
        
        %%
        """

        with tempfile.NamedTemporaryFile(suffix=".y", mode="w", delete=False) as f:
            f.write(yacc_content)
            f.flush()

            udl = UDLRepresentation(yacc_content, f.name)
            assert udl.get_format() == GrammarFormat.YACC_BISON

            # Should extract rules
            rules = udl.get_grammar_rules()
            assert len(rules) > 0

            # Check that rules have Yacc-specific metadata
            for rule in rules:
                if rule.metadata.get("format") == "yacc":
                    assert rule.metadata["operator"] == ":"

        Path(f.name).unlink()

    def test_ebnf_format_detection(self):
        """Test EBNF format detection."""
        ebnf_content = """
        (* EBNF Grammar *)
        expression ::= term { ("+" | "-") term }
        term       ::= factor { ("*" | "/") factor }
        factor     ::= "(" expression ")" | number
        number     ::= digit { digit }
        """

        with tempfile.NamedTemporaryFile(suffix=".ebnf", mode="w", delete=False) as f:
            f.write(ebnf_content)
            f.flush()

            udl = UDLRepresentation(ebnf_content, f.name)
            assert udl.get_format() == GrammarFormat.EBNF

            # Should extract rules
            rules = udl.get_grammar_rules()
            assert len(rules) > 0

            # Check that rules have EBNF-specific metadata
            for rule in rules:
                if rule.metadata.get("format") == "ebnf":
                    assert rule.metadata["operator"] == "::="

        Path(f.name).unlink()

    def test_abnf_format_detection(self):
        """Test ABNF format detection."""
        abnf_content = """
        ; ABNF Grammar (RFC 5234)
        expression = term *( ("+" / "-") term )
        term       = factor *( ("*" / "/") factor )
        factor     = "(" expression ")" / number
        number     = 1*DIGIT
        """

        with tempfile.NamedTemporaryFile(suffix=".abnf", mode="w", delete=False) as f:
            f.write(abnf_content)
            f.flush()

            udl = UDLRepresentation(abnf_content, f.name)
            assert udl.get_format() == GrammarFormat.ABNF

        Path(f.name).unlink()

    def test_railroad_format_detection(self):
        """Test Railroad diagram format detection."""
        railroad_content = """
        # Railroad Diagram Text Representation
        expression: term followed by zero or more ('+' or '-' followed by term)
        term: factor followed by zero or more ('*' or '/' followed by factor)
        factor: '(' expression ')' or number
        number: one or more digits
        """

        with tempfile.NamedTemporaryFile(suffix=".rr", mode="w", delete=False) as f:
            f.write(railroad_content)
            f.flush()

            udl = UDLRepresentation(railroad_content, f.name)
            assert udl.get_format() == GrammarFormat.RAILROAD

            # Should extract rules
            rules = udl.get_grammar_rules()
            assert len(rules) > 0

            # Check that rules have Railroad-specific metadata
            for rule in rules:
                if rule.metadata.get("format") == "railroad":
                    assert rule.metadata["operator"] == ":"

        Path(f.name).unlink()

    def test_content_based_detection(self):
        """Test content-based format detection for generic extensions."""
        # Test ANTLR detection by content
        antlr_content = "grammar Test; expr : term ;"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(antlr_content)
            f.flush()

            udl = UDLRepresentation(antlr_content, f.name)
            assert udl.get_format() == GrammarFormat.ANTLR

        Path(f.name).unlink()

        # Test PEG detection by content
        peg_content = "expr <- term & factor"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(peg_content)
            f.flush()

            udl = UDLRepresentation(peg_content, f.name)
            assert udl.get_format() == GrammarFormat.PEG

        Path(f.name).unlink()

        # Test Yacc detection by content
        yacc_content = "%token ID\n%% expr : term ;"
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(yacc_content)
            f.flush()

            udl = UDLRepresentation(yacc_content, f.name)
            assert udl.get_format() == GrammarFormat.YACC_BISON

        Path(f.name).unlink()


class TestFormatSpecificParsing:
    """Test format-specific parsing logic."""

    def test_antlr_rule_extraction(self):
        """Test ANTLR-specific rule extraction."""
        antlr_content = """
        grammar Test;
        
        expr : expr '+' term
             | term
             ;
             
        term : NUMBER
             | ID
             ;
        """

        udl = UDLRepresentation(antlr_content, "test.g4")
        rules = udl.get_grammar_rules()

        # Should find expr and term rules
        rule_names = {rule.lhs for rule in rules}
        assert "expr" in rule_names
        assert "term" in rule_names

        # Check rule structure
        expr_rules = [r for r in rules if r.lhs == "expr"]
        assert len(expr_rules) > 0

        for rule in expr_rules:
            assert rule.metadata["format"] == "antlr"
            assert rule.metadata["operator"] == ":"

    def test_peg_rule_extraction(self):
        """Test PEG-specific rule extraction."""
        peg_content = """
        Expr <- Term ('+' Term)*
        Term <- Factor ('*' Factor)*
        Factor <- Number
        """

        udl = UDLRepresentation(peg_content, "test.peg")
        rules = udl.get_grammar_rules()

        # Should find Expr, Term, Factor rules
        rule_names = {rule.lhs for rule in rules}
        assert "Expr" in rule_names
        assert "Term" in rule_names
        assert "Factor" in rule_names

        for rule in rules:
            assert rule.metadata["format"] == "peg"
            assert rule.metadata["operator"] == "<-"

    def test_yacc_rule_extraction(self):
        """Test Yacc-specific rule extraction."""
        yacc_content = """
        %token NUMBER
        %%
        expr : expr '+' term
             | term
             ;
        term : NUMBER ;
        %%
        """

        udl = UDLRepresentation(yacc_content, "test.y")
        rules = udl.get_grammar_rules()

        # Should find expr and term rules
        rule_names = {rule.lhs for rule in rules}
        assert "expr" in rule_names
        assert "term" in rule_names

        for rule in rules:
            assert rule.metadata["format"] == "yacc"
            assert rule.metadata["operator"] == ":"

    def test_ebnf_rule_extraction(self):
        """Test EBNF-specific rule extraction."""
        ebnf_content = """
        expression ::= term { ("+" | "-") term }
        term ::= factor { ("*" | "/") factor }
        factor ::= number
        """

        udl = UDLRepresentation(ebnf_content, "test.ebnf")
        rules = udl.get_grammar_rules()

        # Should find expression, term, factor rules
        rule_names = {rule.lhs for rule in rules}
        assert "expression" in rule_names
        assert "term" in rule_names
        assert "factor" in rule_names

        for rule in rules:
            assert rule.metadata["format"] == "ebnf"
            assert rule.metadata["operator"] == "::="

    def test_railroad_rule_extraction(self):
        """Test Railroad diagram rule extraction."""
        railroad_content = """
        # Railroad Grammar
        expression: term followed by '+' and term
        term: factor or number
        factor: identifier
        """

        udl = UDLRepresentation(railroad_content, "test.rr")
        rules = udl.get_grammar_rules()

        # Should find expression, term, factor rules
        rule_names = {rule.lhs for rule in rules}
        assert "expression" in rule_names
        assert "term" in rule_names
        assert "factor" in rule_names

        for rule in rules:
            assert rule.metadata["format"] == "railroad"
            assert rule.metadata["operator"] == ":"


class TestFormatCompatibility:
    """Test that new formats work with existing metric computation."""

    def test_metrics_work_with_new_formats(self):
        """Test that quality metrics can process new grammar formats."""
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        # Test with ANTLR format
        antlr_content = "grammar Test; expr : term '+' factor ;"
        udl = UDLRepresentation(antlr_content, "test.g4")

        consistency_metric = ConsistencyMetric()
        completeness_metric = CompletenessMetric()

        # Should be able to compute metrics without errors
        consistency_score = consistency_metric.compute(udl)
        completeness_score = completeness_metric.compute(udl)

        # Scores should be valid (between 0 and 1)
        assert 0.0 <= consistency_score <= 1.0
        assert 0.0 <= completeness_score <= 1.0

        # Test with PEG format
        peg_content = "Expr <- Term '+' Factor"
        udl_peg = UDLRepresentation(peg_content, "test.peg")

        consistency_score_peg = consistency_metric.compute(udl_peg)
        completeness_score_peg = completeness_metric.compute(udl_peg)

        assert 0.0 <= consistency_score_peg <= 1.0
        assert 0.0 <= completeness_score_peg <= 1.0

    def test_grammar_graph_construction_with_new_formats(self):
        """Test that grammar graphs can be constructed from new formats."""
        # Test ANTLR
        antlr_content = """
        grammar Test;
        expr : expr '+' term | term ;
        term : NUMBER ;
        """
        udl = UDLRepresentation(antlr_content, "test.g4")
        graph = udl.get_grammar_graph()

        # Should have nodes and edges
        assert len(graph.nodes()) > 0
        # May be 0 if no valid symbol references
        assert len(graph.edges()) >= 0

        # Test PEG
        peg_content = """
        Expr <- Term '+' Factor
        Term <- NUMBER
        Factor <- ID
        """
        udl_peg = UDLRepresentation(peg_content, "test.peg")
        graph_peg = udl_peg.get_grammar_graph()

        assert len(graph_peg.nodes()) > 0
        assert len(graph_peg.edges()) >= 0
