"""
Confidence calculation module.

Computes certainty of quality assessment using entropy-based measures.
"""


class ConfidenceCalculator:
    """
    Computes confidence from prediction entropy.

    Mathematical Definition:
    C = 1 - H(p) / H_max

    Where:
    - H(p) = -Σ pᵢ log(pᵢ): Shannon entropy of prediction distribution
    - H_max = log(n): Maximum entropy for n classes
    - C ∈ [0, 1]: Confidence score

    Properties:
    1. Boundedness: 0 ≤ C ≤ 1
    2. Monotonicity: Lower entropy → higher confidence
    3. Calibration: C should match empirical accuracy
    """

    def compute_confidence(self, prediction_probs) -> float:
        """
        Compute C = 1 - H(p)/H_max.

        Args:
            prediction_probs: Probability distribution over classes (numpy array or list)

        Returns:
            Confidence score in [0, 1]
        """
        import numpy as np

        # Convert to numpy array if needed
        if not isinstance(prediction_probs, np.ndarray):
            prediction_probs = np.array(prediction_probs)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs = prediction_probs + epsilon

        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs))

        # Compute maximum entropy
        max_entropy = np.log(len(probs))

        # Avoid division by zero
        if max_entropy == 0:
            return 1.0

        # Compute confidence
        confidence = 1.0 - (entropy / max_entropy)

        # Ensure bounded in [0, 1]
        return np.clip(confidence, 0.0, 1.0)
