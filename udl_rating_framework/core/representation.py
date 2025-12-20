"""
UDL Representation module.

Provides formal representation of UDL structure for mathematical analysis.
"""

import re
import networkx as nx
from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    """Token types for UDL parsing."""

    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"
    LITERAL = "LITERAL"
    DELIMITER = "DELIMITER"
    COMMENT = "COMMENT"
    WHITESPACE = "WHITESPACE"
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Token:
    """Represents a single token in UDL."""

    text: str
    type: TokenType
    position: int
    line: int
    column: int


@dataclass(frozen=True)
class Constraint:
    """Represents a constraint in grammar rules."""

    type: str
    condition: str
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self):
        # Convert dict to tuple of tuples for hashability
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", tuple(sorted(self.metadata.items())))

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Convert metadata back to dict for easier access."""
        return dict(self.metadata)


@dataclass
class GrammarRule:
    """Represents a production rule in grammar."""

    lhs: str  # Left-hand side (non-terminal)
    rhs: List[str]  # Right-hand side (sequence of symbols)
    constraints: List[Constraint]
    metadata: Dict[str, Any]


class AST:
    """Simple Abstract Syntax Tree representation."""

    def __init__(self, node_type: str, value: Any = None, children: List["AST"] = None):
        """Initialize an AST node.
        
        Args:
            node_type: The type of this AST node (e.g., 'rule', 'terminal').
            value: The value associated with this node.
            children: List of child AST nodes.
        """
        self.node_type = node_type
        self.value = value
        self.children = children or []

    def __repr__(self):
        if self.children:
            return f"AST({self.node_type}, {self.value}, {self.children})"
        return f"AST({self.node_type}, {self.value})"


class UDLTokenizer:
    """Basic tokenizer for UDL text with support for multiple grammar formats."""

    # Common UDL patterns - enhanced for multiple formats
    TOKEN_PATTERNS = [
        # Special tokens (must come before comments)
        (r"%%", TokenType.DELIMITER),        # Yacc/Bison section separator
        (r"%[a-zA-Z_][a-zA-Z0-9_]*", TokenType.KEYWORD),  # Yacc directives
        (r"@[a-zA-Z_][a-zA-Z0-9_]*", TokenType.KEYWORD),  # ANTLR annotations
        
        # Comments (various formats)
        (r"//.*", TokenType.COMMENT),  # C++ style comments (ANTLR, etc.)
        (r"/\*.*?\*/", TokenType.COMMENT),  # C style block comments
        (r"#.*", TokenType.COMMENT),  # Shell/Python style comments
        (r"%.*", TokenType.COMMENT),  # Yacc/Bison comments (catch-all for other % patterns)$', TokenType.COMMENT),  # Comments
        (r"\n", TokenType.NEWLINE),  # Newlines
        (r"\s+", TokenType.WHITESPACE),  # Whitespace
        
        # Assignment/production operators (various formats)
        (r"<-", TokenType.OPERATOR),   # PEG (must come before individual < and -)
        (r"::=", TokenType.OPERATOR),  # EBNF/BNF
        (r":=", TokenType.OPERATOR),   # Some DSL formats
        (r"->", TokenType.OPERATOR),   # ANTLR
        (r"=", TokenType.OPERATOR),    # Simple assignment
        (r":", TokenType.OPERATOR),    # Yacc/Bison
        
        # Alternation and grouping
        (r"\|", TokenType.OPERATOR),   # Alternation
        (r"/", TokenType.OPERATOR),    # PEG ordered choice
        
        # Brackets and delimiters
        (r"[(){}[\]]", TokenType.DELIMITER),  # Standard brackets
        (r"<>", TokenType.DELIMITER),  # Angle brackets (some formats)
        
        # Repetition and quantifiers
        (r"[*+?]", TokenType.OPERATOR),  # Standard repetition
        (r"\.\.\.", TokenType.OPERATOR), # Ellipsis (some formats)
        (r"~", TokenType.OPERATOR),      # PEG not-predicate
        (r"&", TokenType.OPERATOR),      # PEG and-predicate
        (r"!", TokenType.OPERATOR),      # Negation/not
        
        # String literals (various quote styles)
        (r'"[^"]*"', TokenType.LITERAL),     # Double quotes
        (r"'[^']*'", TokenType.LITERAL),     # Single quotes
        (r"`[^`]*`", TokenType.LITERAL),     # Backticks (some formats)
        

        
        # Identifiers and keywords
        (r"[a-zA-Z_][a-zA-Z0-9_]*", TokenType.IDENTIFIER),  # Standard identifiers
        (r"[0-9]+", TokenType.LITERAL),      # Numbers
        (r"\$[0-9]+", TokenType.IDENTIFIER), # Yacc/Bison positional parameters
        
        # Catch-all for other operators
        (r"[^\s\w]", TokenType.OPERATOR)
    ]

    def __init__(self):
        """Initialize the UDL tokenizer with compiled regex patterns."""
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), token_type)
            for pattern, token_type in self.TOKEN_PATTERNS
        ]

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize UDL text into a sequence of tokens.

        Args:
            text: Source text to tokenize

        Returns:
            List of Token objects
        """
        tokens = []
        lines = text.split("\n")
        position = 0

        for line_num, line in enumerate(lines, 1):
            column = 1
            line_pos = 0

            while line_pos < len(line):
                matched = False

                for pattern, token_type in self.compiled_patterns:
                    match = pattern.match(line, line_pos)
                    if match:
                        token_text = match.group(0)

                        # Skip whitespace tokens for cleaner parsing
                        if token_type != TokenType.WHITESPACE:
                            token = Token(
                                text=token_text,
                                type=token_type,
                                position=position,
                                line=line_num,
                                column=column,
                            )
                            tokens.append(token)

                        line_pos = match.end()
                        column += len(token_text)
                        position += len(token_text)
                        matched = True
                        break

                if not matched:
                    # Unknown character
                    token = Token(
                        text=line[line_pos],
                        type=TokenType.UNKNOWN,
                        position=position,
                        line=line_num,
                        column=column,
                    )
                    tokens.append(token)
                    line_pos += 1
                    column += 1
                    position += 1

            # Add newline token if not at end of file
            if line_num < len(lines):
                tokens.append(
                    Token(
                        text="\n",
                        type=TokenType.NEWLINE,
                        position=position,
                        line=line_num,
                        column=column,
                    )
                )
                position += 1

        # Add EOF token
        tokens.append(
            Token(
                text="",
                type=TokenType.EOF,
                position=position,
                line=len(lines),
                column=1,
            )
        )

        return tokens


class GrammarFormat(Enum):
    """Supported grammar format types."""
    
    GENERIC = "generic"
    ANTLR = "antlr"
    PEG = "peg"
    YACC_BISON = "yacc_bison"
    EBNF = "ebnf"
    BNF = "bnf"
    ABNF = "abnf"
    RAILROAD = "railroad"


class UDLRepresentation:
    """
    Formal representation of a UDL as a multi-level structure.

    Mathematical Definition:
    A UDL is represented as a tuple U = (T, G, S, R) where:
    - T: Set of tokens (terminal symbols)
    - G: Grammar graph G = (V, E) with vertices V (non-terminals) and edges E (production rules)
    - S: Semantic mapping S: T → Semantics
    - R: Set of constraints/rules
    
    Enhanced for Task 26: Support for multiple grammar formats including
    ANTLR (.g4), PEG (.peg), Yacc/Bison (.y, .yacc), EBNF variants, and Railroad diagrams.
    """

    def __init__(self, source_text: str, file_path: str):
        """Parse UDL from source text."""
        self.source_text = source_text
        self.file_path = file_path
        self.tokenizer = UDLTokenizer()
        
        # Detect grammar format based on file extension and content
        self.format = self._detect_format()

        # Parse the UDL
        self._tokens = self.tokenizer.tokenize(source_text)
        self._grammar_rules = self._extract_grammar_rules()
        self._grammar_graph = self._build_grammar_graph()
        self._semantic_map = self._build_semantic_map()
        self._constraints = self._extract_constraints()

    def get_tokens(self) -> List[Token]:
        """Return token sequence T."""
        return self._tokens

    def get_grammar_graph(self) -> nx.DiGraph:
        """Return grammar as directed graph G = (V, E)."""
        return self._grammar_graph

    def get_semantic_map(self) -> Dict[Token, Any]:
        """Return semantic mapping S."""
        return self._semantic_map

    def get_constraints(self) -> Set[Constraint]:
        """Return constraint set R."""
        return self._constraints

    def get_grammar_rules(self) -> List[GrammarRule]:
        """Return extracted grammar rules."""
        return self._grammar_rules
    
    def get_format(self) -> GrammarFormat:
        """Return detected grammar format."""
        return self.format

    def _detect_format(self) -> GrammarFormat:
        """
        Detect grammar format based on file extension and content patterns.
        
        Returns:
            GrammarFormat enum value
        """
        from pathlib import Path
        
        file_path = Path(self.file_path)
        extension = file_path.suffix.lower()
        
        # Format detection based on file extension
        extension_map = {
            '.g4': GrammarFormat.ANTLR,
            '.peg': GrammarFormat.PEG,
            '.y': GrammarFormat.YACC_BISON,
            '.yacc': GrammarFormat.YACC_BISON,
            '.ebnf': GrammarFormat.EBNF,
            '.bnf': GrammarFormat.BNF,
            '.abnf': GrammarFormat.ABNF,
            '.xbnf': GrammarFormat.EBNF,
            '.wsn': GrammarFormat.EBNF,
            '.wirth': GrammarFormat.EBNF,
            '.rr': GrammarFormat.RAILROAD,
            '.railroad': GrammarFormat.RAILROAD,
        }
        
        if extension in extension_map:
            return extension_map[extension]
        
        # Content-based detection for generic extensions
        content_lower = self.source_text.lower()
        
        # ANTLR patterns (very specific to avoid false positives in comments)
        if any(pattern in content_lower for pattern in ['@lexer', '@parser', 'lexer grammar', 'parser grammar']) or \
           (content_lower.startswith('grammar ') or '\ngrammar ' in content_lower):
            return GrammarFormat.ANTLR
        
        # Yacc/Bison patterns
        if any(pattern in self.source_text for pattern in ['%%', '%token', '%type', '%start']):
            return GrammarFormat.YACC_BISON
        
        # EBNF patterns (check before PEG since ::= is more specific to EBNF)
        if '::=' in self.source_text:
            return GrammarFormat.EBNF
        
        # PEG patterns (more specific patterns to avoid false positives)
        if '<-' in self.source_text or ('&' in self.source_text and '~' in self.source_text):
            return GrammarFormat.PEG
        
        # ABNF patterns (RFC 5234)
        if any(pattern in content_lower for pattern in ['abnf', 'rfc', 'vchar', 'alpha']):
            return GrammarFormat.ABNF
        
        # Default to generic
        return GrammarFormat.GENERIC

    def to_ast(self) -> AST:
        """Convert to abstract syntax tree representation."""
        # Simple AST construction from tokens
        root = AST("UDL", self.file_path)

        # Group tokens by lines for structure
        current_line = 1
        line_tokens = []

        for token in self._tokens:
            if token.type == TokenType.EOF:
                break

            if token.line != current_line:
                if line_tokens:
                    line_ast = self._tokens_to_ast(line_tokens)
                    if line_ast:
                        root.children.append(line_ast)
                line_tokens = []
                current_line = token.line

            if token.type not in [TokenType.WHITESPACE, TokenType.NEWLINE]:
                line_tokens.append(token)

        # Handle last line
        if line_tokens:
            line_ast = self._tokens_to_ast(line_tokens)
            if line_ast:
                root.children.append(line_ast)

        return root

    def _extract_grammar_rules(self) -> List[GrammarRule]:
        """Extract grammar rules from tokens based on detected format."""
        if self.format == GrammarFormat.ANTLR:
            return self._extract_antlr_rules()
        elif self.format == GrammarFormat.PEG:
            return self._extract_peg_rules()
        elif self.format == GrammarFormat.YACC_BISON:
            return self._extract_yacc_rules()
        elif self.format in [GrammarFormat.EBNF, GrammarFormat.BNF, GrammarFormat.ABNF]:
            return self._extract_bnf_ebnf_rules()
        elif self.format == GrammarFormat.RAILROAD:
            return self._extract_railroad_rules()
        else:
            return self._extract_generic_rules()
    
    def _extract_generic_rules(self) -> List[GrammarRule]:
        """Extract grammar rules using generic pattern matching."""
        rules = []
        i = 0
        tokens = [
            t
            for t in self._tokens
            if t.type
            not in [TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE]
        ]

        while i < len(tokens):
            if i + 2 < len(tokens) and tokens[i + 1].text in ["::=", ":=", "=", "->", ":"]:
                # Found a production rule
                lhs = tokens[i].text
                operator = tokens[i + 1].text

                # Collect RHS until next rule or EOF
                rhs = []
                i += 2
                while i < len(tokens) and not (
                    i + 1 < len(tokens) and tokens[i + 1].text in ["::=", ":=", "=", "->", ":"]
                ):
                    if tokens[i].type != TokenType.EOF:
                        rhs.append(tokens[i].text)
                    i += 1

                rule = GrammarRule(
                    lhs=lhs, rhs=rhs, constraints=[], 
                    metadata={"operator": operator, "format": "generic"}
                )
                rules.append(rule)
            else:
                i += 1

        return rules
    
    def _extract_antlr_rules(self) -> List[GrammarRule]:
        """Extract ANTLR grammar rules (.g4 format)."""
        rules = []
        tokens = [t for t in self._tokens if t.type not in [TokenType.WHITESPACE, TokenType.NEWLINE]]
        i = 0
        
        while i < len(tokens):
            # Look for rule pattern: identifier : alternatives ;
            if (i + 2 < len(tokens) and 
                tokens[i].type == TokenType.IDENTIFIER and 
                tokens[i + 1].text == ":"):
                
                lhs = tokens[i].text
                i += 2  # Skip identifier and ':'
                
                # Collect alternatives until semicolon
                rhs = []
                while i < len(tokens) and tokens[i].text != ";":
                    if tokens[i].type != TokenType.COMMENT:
                        rhs.append(tokens[i].text)
                    i += 1
                
                rule = GrammarRule(
                    lhs=lhs, rhs=rhs, constraints=[],
                    metadata={"operator": ":", "format": "antlr"}
                )
                rules.append(rule)
            
            i += 1
        
        return rules
    
    def _extract_peg_rules(self) -> List[GrammarRule]:
        """Extract PEG grammar rules (.peg format)."""
        rules = []
        tokens = [t for t in self._tokens if t.type not in [TokenType.WHITESPACE, TokenType.NEWLINE]]
        i = 0
        
        while i < len(tokens):
            # Look for rule pattern: identifier <- expression
            if (i + 1 < len(tokens) and 
                tokens[i].type == TokenType.IDENTIFIER and 
                i + 1 < len(tokens) and tokens[i + 1].text == "<-"):
                
                lhs = tokens[i].text
                i += 2  # Skip identifier and '<-'
                
                # Collect expression until next rule or EOF
                rhs = []
                while i < len(tokens) and not (
                    tokens[i].type == TokenType.IDENTIFIER and 
                    i + 1 < len(tokens) and 
                    tokens[i + 1].text == "<-"
                ):
                    if tokens[i].type not in [TokenType.COMMENT, TokenType.EOF]:
                        rhs.append(tokens[i].text)
                    i += 1
                
                rule = GrammarRule(
                    lhs=lhs, rhs=rhs, constraints=[],
                    metadata={"operator": "<-", "format": "peg"}
                )
                rules.append(rule)
            else:
                i += 1
        
        return rules
    
    def _extract_yacc_rules(self) -> List[GrammarRule]:
        """Extract Yacc/Bison grammar rules (.y, .yacc format)."""
        rules = []
        tokens = [t for t in self._tokens if t.type not in [TokenType.WHITESPACE, TokenType.NEWLINE]]
        i = 0
        
        # Skip header section (until %%)
        while i < len(tokens) and tokens[i].text != "%%":
            i += 1
        if i < len(tokens):
            i += 1  # Skip %%
        
        while i < len(tokens):
            # Look for rule pattern: identifier : alternatives ;
            if (i + 1 < len(tokens) and 
                tokens[i].type == TokenType.IDENTIFIER and 
                i + 1 < len(tokens) and tokens[i + 1].text == ":"):
                
                lhs = tokens[i].text
                i += 2  # Skip identifier and ':'
                
                # Collect alternatives until semicolon
                rhs = []
                while i < len(tokens) and tokens[i].text != ";":
                    if tokens[i].type not in [TokenType.COMMENT, TokenType.EOF]:
                        rhs.append(tokens[i].text)
                    i += 1
                
                rule = GrammarRule(
                    lhs=lhs, rhs=rhs, constraints=[],
                    metadata={"operator": ":", "format": "yacc"}
                )
                rules.append(rule)
                
                # Skip the semicolon
                if i < len(tokens) and tokens[i].text == ";":
                    i += 1
            else:
                i += 1
        
        return rules
    
    def _extract_bnf_ebnf_rules(self) -> List[GrammarRule]:
        """Extract BNF/EBNF grammar rules."""
        rules = []
        tokens = [t for t in self._tokens if t.type not in [TokenType.WHITESPACE, TokenType.NEWLINE]]
        i = 0
        
        # Valid operators for BNF/EBNF rules
        valid_operators = ["::=", ":=", "="]
        
        while i < len(tokens):
            # Look for rule patterns: identifier ::= expression OR identifier := expression OR identifier = expression (ABNF)
            if (i + 1 < len(tokens) and 
                tokens[i].type == TokenType.IDENTIFIER and 
                i + 1 < len(tokens) and tokens[i + 1].text in valid_operators):
                
                lhs = tokens[i].text
                operator = tokens[i + 1].text
                i += 2  # Skip identifier and operator
                
                # Collect expression until next rule or EOF
                rhs = []
                while i < len(tokens) and not (
                    tokens[i].type == TokenType.IDENTIFIER and 
                    i + 1 < len(tokens) and 
                    tokens[i + 1].text in valid_operators
                ):
                    if tokens[i].type not in [TokenType.COMMENT, TokenType.EOF]:
                        rhs.append(tokens[i].text)
                    i += 1
                
                rule = GrammarRule(
                    lhs=lhs, rhs=rhs, constraints=[],
                    metadata={"operator": operator, "format": self.format.value}
                )
                rules.append(rule)
            else:
                i += 1
        
        return rules
    
    def _extract_railroad_rules(self) -> List[GrammarRule]:
        """Extract rules from railroad diagram format (simplified text representation)."""
        # Railroad diagrams are typically visual, but we can parse text descriptions
        rules = []
        lines = self.source_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Look for patterns like "rule_name: description"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs_text = parts[1].strip()
                    
                    # Simple tokenization of RHS
                    rhs = rhs_text.split() if rhs_text else []
                    
                    rule = GrammarRule(
                        lhs=lhs, rhs=rhs, constraints=[],
                        metadata={"operator": ":", "format": "railroad"}
                    )
                    rules.append(rule)
        
        return rules

    def _build_grammar_graph(self) -> nx.DiGraph:
        """Build grammar graph using NetworkX."""
        graph = nx.DiGraph()

        # Add nodes for all symbols
        symbols = set()
        for rule in self._grammar_rules:
            symbols.add(rule.lhs)
            symbols.update(rule.rhs)

        for symbol in symbols:
            graph.add_node(symbol)

        # Add edges for production rules
        for rule in self._grammar_rules:
            for rhs_symbol in rule.rhs:
                if rhs_symbol in symbols:  # Only add edges to known symbols
                    graph.add_edge(rule.lhs, rhs_symbol, rule=rule)

        return graph

    def _build_semantic_map(self) -> Dict[Token, Any]:
        """Build semantic mapping for tokens."""
        semantic_map = {}

        for token in self._tokens:
            if token.type == TokenType.IDENTIFIER:
                semantic_map[token] = {"type": "symbol", "category": "non_terminal"}
            elif token.type == TokenType.LITERAL:
                semantic_map[token] = {"type": "terminal", "value": token.text}
            elif token.type == TokenType.OPERATOR:
                semantic_map[token] = {"type": "operator", "function": token.text}
            elif token.type == TokenType.KEYWORD:
                semantic_map[token] = {"type": "keyword", "meaning": token.text}

        return semantic_map

    def _extract_constraints(self) -> Set[Constraint]:
        """Extract constraints from the UDL."""
        constraints = set()

        # Look for constraint patterns in comments or special syntax
        for token in self._tokens:
            if token.type == TokenType.COMMENT:
                # Simple constraint extraction from comments
                if "constraint:" in token.text.lower():
                    constraint_text = token.text.split("constraint:", 1)[1].strip()
                    constraint = Constraint(
                        type="comment_constraint",
                        condition=constraint_text,
                        metadata={"line": token.line, "source": "comment"},
                    )
                    constraints.add(constraint)

        return constraints

    def _tokens_to_ast(self, tokens: List[Token]) -> Optional[AST]:
        """Convert a sequence of tokens to an AST node."""
        if not tokens:
            return None

        # Simple rule detection
        if len(tokens) >= 3 and tokens[1].text in ["::=", ":=", "="]:
            # Production rule
            lhs = AST("NonTerminal", tokens[0].text)
            rhs_tokens = tokens[2:]
            rhs_children = [
                AST("Symbol", t.text)
                for t in rhs_tokens
                if t.type != TokenType.OPERATOR or t.text == "|"
            ]
            rhs = AST("RHS", None, rhs_children)
            return AST("ProductionRule", None, [lhs, rhs])
        else:
            # Generic statement
            children = [AST("Token", t.text) for t in tokens]
            return AST("Statement", None, children)
