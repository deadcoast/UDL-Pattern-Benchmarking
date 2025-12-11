"""
UDL Representation module.

Provides formal representation of UDL structure for mathematical analysis.
"""

import re
import networkx as nx
from typing import List, Dict, Set, Any, Optional, Union, Tuple
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
            object.__setattr__(self, 'metadata', tuple(sorted(self.metadata.items())))
    
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
    
    def __init__(self, node_type: str, value: Any = None, children: List['AST'] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def __repr__(self):
        if self.children:
            return f"AST({self.node_type}, {self.value}, {self.children})"
        return f"AST({self.node_type}, {self.value})"


class UDLTokenizer:
    """Basic tokenizer for UDL text."""
    
    # Common UDL patterns
    TOKEN_PATTERNS = [
        (r'#.*', TokenType.COMMENT),  # Comments$', TokenType.COMMENT),  # Comments
        (r'\n', TokenType.NEWLINE),  # Newlines
        (r'\s+', TokenType.WHITESPACE),  # Whitespace
        (r'::=|:=|=', TokenType.OPERATOR),  # Assignment operators
        (r'\|', TokenType.OPERATOR),  # Alternation
        (r'[(){}[\]]', TokenType.DELIMITER),  # Brackets
        (r'[*+?]', TokenType.OPERATOR),  # Repetition operators
        (r'"[^"]*"', TokenType.LITERAL),  # String literals
        (r"'[^']*'", TokenType.LITERAL),  # String literals
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),  # Identifiers
        (r'[0-9]+', TokenType.LITERAL),  # Numbers
        (r'[^\s\w]', TokenType.OPERATOR),  # Other operators
    ]
    
    def __init__(self):
        self.compiled_patterns = [(re.compile(pattern, re.MULTILINE), token_type) 
                                  for pattern, token_type in self.TOKEN_PATTERNS]
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize UDL text into a sequence of tokens.
        
        Args:
            text: Source text to tokenize
            
        Returns:
            List of Token objects
        """
        tokens = []
        lines = text.split('\n')
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
                                column=column
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
                        column=column
                    )
                    tokens.append(token)
                    line_pos += 1
                    column += 1
                    position += 1
            
            # Add newline token if not at end of file
            if line_num < len(lines):
                tokens.append(Token(
                    text='\n',
                    type=TokenType.NEWLINE,
                    position=position,
                    line=line_num,
                    column=column
                ))
                position += 1
        
        # Add EOF token
        tokens.append(Token(
            text='',
            type=TokenType.EOF,
            position=position,
            line=len(lines),
            column=1
        ))
        
        return tokens


class UDLRepresentation:
    """
    Formal representation of a UDL as a multi-level structure.
    
    Mathematical Definition:
    A UDL is represented as a tuple U = (T, G, S, R) where:
    - T: Set of tokens (terminal symbols)
    - G: Grammar graph G = (V, E) with vertices V (non-terminals) and edges E (production rules)
    - S: Semantic mapping S: T → Semantics
    - R: Set of constraints/rules
    """
    
    def __init__(self, source_text: str, file_path: str):
        """Parse UDL from source text."""
        self.source_text = source_text
        self.file_path = file_path
        self.tokenizer = UDLTokenizer()
        
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
        """Extract grammar rules from tokens."""
        rules = []
        i = 0
        tokens = [t for t in self._tokens if t.type not in [TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE]]
        
        while i < len(tokens):
            if i + 2 < len(tokens) and tokens[i + 1].text in ['::=', ':=', '=']:
                # Found a production rule
                lhs = tokens[i].text
                operator = tokens[i + 1].text
                
                # Collect RHS until next rule or EOF
                rhs = []
                i += 2
                while i < len(tokens) and not (i + 1 < len(tokens) and tokens[i + 1].text in ['::=', ':=', '=']):
                    if tokens[i].type != TokenType.EOF:
                        rhs.append(tokens[i].text)
                    i += 1
                
                rule = GrammarRule(
                    lhs=lhs,
                    rhs=rhs,
                    constraints=[],
                    metadata={'operator': operator}
                )
                rules.append(rule)
            else:
                i += 1
        
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
                semantic_map[token] = {'type': 'symbol', 'category': 'non_terminal'}
            elif token.type == TokenType.LITERAL:
                semantic_map[token] = {'type': 'terminal', 'value': token.text}
            elif token.type == TokenType.OPERATOR:
                semantic_map[token] = {'type': 'operator', 'function': token.text}
            elif token.type == TokenType.KEYWORD:
                semantic_map[token] = {'type': 'keyword', 'meaning': token.text}
        
        return semantic_map
    
    def _extract_constraints(self) -> Set[Constraint]:
        """Extract constraints from the UDL."""
        constraints = set()
        
        # Look for constraint patterns in comments or special syntax
        for token in self._tokens:
            if token.type == TokenType.COMMENT:
                # Simple constraint extraction from comments
                if 'constraint:' in token.text.lower():
                    constraint_text = token.text.split('constraint:', 1)[1].strip()
                    constraint = Constraint(
                        type='comment_constraint',
                        condition=constraint_text,
                        metadata={'line': token.line, 'source': 'comment'}
                    )
                    constraints.add(constraint)
        
        return constraints
    
    def _tokens_to_ast(self, tokens: List[Token]) -> Optional[AST]:
        """Convert a sequence of tokens to an AST node."""
        if not tokens:
            return None
        
        # Simple rule detection
        if len(tokens) >= 3 and tokens[1].text in ['::=', ':=', '=']:
            # Production rule
            lhs = AST("NonTerminal", tokens[0].text)
            rhs_tokens = tokens[2:]
            rhs_children = [AST("Symbol", t.text) for t in rhs_tokens if t.type != TokenType.OPERATOR or t.text == '|']
            rhs = AST("RHS", None, rhs_children)
            return AST("ProductionRule", None, [lhs, rhs])
        else:
            # Generic statement
            children = [AST("Token", t.text) for t in tokens]
            return AST("Statement", None, children)
