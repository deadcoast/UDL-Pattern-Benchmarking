"""
Semantic Similarity Metric implementation.

Measures semantic similarity between UDL constructs using embeddings.
"""

import numpy as np
from typing import Dict, List, Set, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.representation import UDLRepresentation, Token, TokenType


class SemanticSimilarityMetric(QualityMetric):
    """
    Measures semantic similarity using embeddings and TF-IDF.

    Mathematical Definition:
    SemanticSimilarity(U) = (1/|C|²) * Σᵢⱼ sim(cᵢ, cⱼ)

    Where:
    - C: Set of constructs in the UDL
    - sim(cᵢ, cⱼ): Cosine similarity between embeddings of constructs i and j
    - Higher similarity indicates better semantic coherence

    Algorithm:
    1. Extract semantic constructs from UDL (identifiers, rules, comments)
    2. Generate embeddings using TF-IDF or pre-trained models
    3. Compute pairwise similarities
    4. Aggregate into overall coherence score
    """

    def __init__(self, use_pretrained: bool = False):
        """
        Initialize semantic similarity metric.

        Args:
            use_pretrained: Whether to use pre-trained embeddings (requires additional dependencies)
        """
        self.use_pretrained = use_pretrained
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

    def compute(self, udl: UDLRepresentation) -> float:
        """
        Compute semantic similarity score.

        Args:
            udl: UDLRepresentation instance

        Returns:
            Semantic similarity score in [0, 1]
        """
        # Extract semantic constructs
        constructs = self._extract_semantic_constructs(udl)

        if len(constructs) < 2:
            # Need at least 2 constructs for similarity
            return 1.0 if len(constructs) == 1 else 0.0

        # Generate embeddings
        embeddings = self._generate_embeddings(constructs)

        if embeddings is None or embeddings.shape[0] < 2:
            return 0.0

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)

        # Extract upper triangle (excluding diagonal)
        n = similarity_matrix.shape[0]
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])

        if not similarities:
            return 0.0

        # Compute average similarity
        avg_similarity = np.mean(similarities)

        # Normalize to [0, 1] (cosine similarity is already in [-1, 1])
        normalized_score = (avg_similarity + 1.0) / 2.0

        return max(0.0, min(1.0, normalized_score))

    def _extract_semantic_constructs(self, udl: UDLRepresentation) -> List[str]:
        """
        Extract semantic constructs for similarity analysis.

        Args:
            udl: UDLRepresentation instance

        Returns:
            List of construct texts
        """
        constructs = []
        tokens = udl.get_tokens()
        rules = udl.get_grammar_rules()

        # Extract meaningful identifiers
        identifiers = [
            token.text for token in tokens
            if token.type == TokenType.IDENTIFIER and len(token.text) > 1
        ]
        constructs.extend(identifiers)

        # Extract rule names (LHS of production rules)
        rule_names = [rule.lhs for rule in rules if rule.lhs]
        constructs.extend(rule_names)

        # Extract comments (often contain semantic information)
        comments = [
            token.text.strip('#').strip('//').strip('/*').strip('*/')
            for token in tokens
            if token.type == TokenType.COMMENT and len(token.text.strip()) > 3
        ]
        constructs.extend(comments)

        # Extract string literals (may contain semantic information)
        literals = [
            token.text.strip('"').strip("'")
            for token in tokens
            if token.type == TokenType.LITERAL 
            and len(token.text) > 3
            and not token.text.isdigit()
        ]
        constructs.extend(literals)

        # Remove duplicates and filter out very short constructs
        unique_constructs = list(set(
            construct for construct in constructs
            if len(construct.strip()) > 1
        ))

        return unique_constructs

    def _generate_embeddings(self, constructs: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for constructs.

        Args:
            constructs: List of construct texts

        Returns:
            Embedding matrix or None if generation fails
        """
        if not constructs:
            return None

        try:
            if self.use_pretrained:
                return self._generate_pretrained_embeddings(constructs)
            else:
                return self._generate_tfidf_embeddings(constructs)
        except Exception:
            # Fallback to simple character-based embeddings
            return self._generate_char_embeddings(constructs)

    def _generate_tfidf_embeddings(self, constructs: List[str]) -> Optional[np.ndarray]:
        """
        Generate TF-IDF embeddings.

        Args:
            constructs: List of construct texts

        Returns:
            TF-IDF embedding matrix
        """
        try:
            # Preprocess constructs for better TF-IDF
            processed_constructs = []
            for construct in constructs:
                # Split camelCase and snake_case
                processed = self._split_identifier(construct)
                processed_constructs.append(processed)

            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(processed_constructs)
            return tfidf_matrix.toarray()
        except Exception:
            return None

    def _generate_pretrained_embeddings(self, constructs: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings using pre-trained models (placeholder for future implementation).

        Args:
            constructs: List of construct texts

        Returns:
            Pre-trained embedding matrix
        """
        # Placeholder for pre-trained embeddings (e.g., Word2Vec, BERT)
        # This would require additional dependencies like transformers or gensim
        # For now, fall back to TF-IDF
        return self._generate_tfidf_embeddings(constructs)

    def _generate_char_embeddings(self, constructs: List[str]) -> np.ndarray:
        """
        Generate simple character-based embeddings as fallback.

        Args:
            constructs: List of construct texts

        Returns:
            Character-based embedding matrix
        """
        # Create character frequency vectors
        all_chars = set(''.join(constructs))
        char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        
        embeddings = []
        for construct in constructs:
            embedding = np.zeros(len(char_to_idx))
            for char in construct:
                if char in char_to_idx:
                    embedding[char_to_idx[char]] += 1
            # Normalize
            if np.sum(embedding) > 0:
                embedding = embedding / np.sum(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def _split_identifier(self, identifier: str) -> str:
        """
        Split camelCase and snake_case identifiers into words.

        Args:
            identifier: Identifier to split

        Returns:
            Space-separated words
        """
        import re
        
        # Split camelCase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', identifier)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        
        # Split snake_case and other separators
        s3 = re.sub('[_-]', ' ', s2)
        
        return s3.lower()

    def get_formula(self) -> str:
        """Return LaTeX formula for this metric."""
        return r"SemanticSimilarity(U) = \frac{1}{|C|^2} \sum_{i,j} \text{sim}(c_i, c_j)"

    def get_properties(self) -> Dict[str, bool]:
        """
        Return mathematical properties of the semantic similarity metric.

        Returns:
            Dict with mathematical properties
        """
        return {
            "bounded": True,  # Always produces values in [0, 1]
            "monotonic": False,  # More constructs don't always mean higher similarity
            "additive": False,  # Similarity is not sum of parts
            "continuous": True,  # Small changes in constructs cause small changes in similarity
        }


# Register the metric in the global registry
SemanticSimilarityMetric.register_metric("semantic_similarity")