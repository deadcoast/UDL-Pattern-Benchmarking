"""
Evaluation utilities for UDL rating framework.

This module provides comprehensive evaluation capabilities including:
- K-fold cross-validation
- Correlation analysis with confidence intervals
- Calibration error computation
- Error distribution analysis
- Bootstrap confidence intervals
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr, shapiro
import warnings
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Results from evaluation suite."""

    pearson_correlation: float
    pearson_ci: Tuple[float, float]
    spearman_correlation: float
    spearman_ci: Tuple[float, float]
    calibration_error: float
    shapiro_statistic: float
    shapiro_p_value: float
    bootstrap_metrics: Dict[str, Tuple[float, float]]  # metric -> (lower, upper) CI
    cv_scores: List[float]
    mean_cv_score: float
    std_cv_score: float


class EvaluationSuite:
    """
    Comprehensive evaluation suite for UDL rating framework.

    Provides statistical evaluation methods including:
    - K-fold cross-validation (k ≥ 5)
    - Correlation computation (Pearson, Spearman) with confidence intervals
    - Calibration error computation: ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n)
    - Error distribution analysis with Shapiro-Wilk test
    - Bootstrap confidence intervals (B ≥ 1000)
    """

    def __init__(
        self,
        k_folds: int = 5,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize evaluation suite.

        Args:
            k_folds: Number of folds for cross-validation (≥ 5)
            bootstrap_samples: Number of bootstrap samples (≥ 1000)
            confidence_level: Confidence level for intervals (default 0.95)
        """
        if k_folds < 5:
            raise ValueError("k_folds must be ≥ 5")
        if bootstrap_samples < 1000:
            raise ValueError("bootstrap_samples must be ≥ 1000")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        self.k_folds = k_folds
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def k_fold_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: Callable,
        metric_fn: Callable = None,
    ) -> Tuple[List[float], float, float]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values [n_samples]
            model_fn: Function that returns a trained model given (X_train, y_train)
            metric_fn: Function to compute metric given (y_true, y_pred). Default is MSE.

        Returns:
            Tuple of (cv_scores, mean_score, std_score)
        """
        if metric_fn is None:

            def default_metric(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)

            metric_fn = default_metric

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = model_fn(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_val) if hasattr(model, "predict") else model(X_val)

            # Compute metric
            score = metric_fn(y_val, y_pred)
            cv_scores.append(score)

        return cv_scores, np.mean(cv_scores), np.std(cv_scores)

    def compute_correlations_with_ci(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, Tuple[float, float], float, Tuple[float, float]]:
        """
        Compute Pearson and Spearman correlations with 95% confidence intervals.

        Args:
            y_true: True values [n_samples]
            y_pred: Predicted values [n_samples]

        Returns:
            Tuple of (pearson_r, pearson_ci, spearman_r, spearman_ci)
        """
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(y_true, y_pred)

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

        # Bootstrap confidence intervals for correlations
        n = len(y_true)
        pearson_bootstrap = []
        spearman_bootstrap = []

        np.random.seed(42)
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute correlations
            try:
                p_r, _ = pearsonr(y_true_boot, y_pred_boot)
                s_r, _ = spearmanr(y_true_boot, y_pred_boot)

                if not np.isnan(p_r):
                    pearson_bootstrap.append(p_r)
                if not np.isnan(s_r):
                    spearman_bootstrap.append(s_r)
            except Exception:
                continue

        # Compute confidence intervals
        alpha_half = self.alpha / 2
        pearson_ci = (
            np.percentile(pearson_bootstrap, alpha_half * 100),
            np.percentile(pearson_bootstrap, (1 - alpha_half) * 100),
        )
        spearman_ci = (
            np.percentile(spearman_bootstrap, alpha_half * 100),
            np.percentile(spearman_bootstrap, (1 - alpha_half) * 100),
        )

        return pearson_r, pearson_ci, spearman_r, spearman_ci

    def compute_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ_i |accuracy_i - confidence_i| · (n_i/n)

        Args:
            y_true: True binary labels [n_samples]
            y_pred: Predicted binary labels [n_samples]
            confidences: Confidence scores [n_samples]
            n_bins: Number of bins for calibration

        Returns:
            Expected Calibration Error
        """
        # Convert to binary if needed (for regression, use threshold)
        if len(np.unique(y_true)) > 2:
            # For regression, convert to binary based on median
            threshold = np.median(y_true)
            y_true_bin = (y_true > threshold).astype(int)
            y_pred_bin = (y_pred > threshold).astype(int)
        else:
            y_true_bin = y_true.astype(int)
            y_pred_bin = y_pred.astype(int)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (
                    (y_pred_bin[in_bin] == y_true_bin[in_bin]).mean()
                    if np.any(in_bin)
                    else 0.0
                )

                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def analyze_error_distribution(self, errors: np.ndarray) -> Tuple[float, float]:
        """
        Analyze error distribution using Shapiro-Wilk test for normality.

        Args:
            errors: Array of errors [n_samples]

        Returns:
            Tuple of (shapiro_statistic, p_value)
        """
        # Remove any infinite or NaN values
        clean_errors = errors[np.isfinite(errors)]

        if len(clean_errors) < 3:
            warnings.warn("Too few samples for Shapiro-Wilk test")
            return np.nan, np.nan

        # Shapiro-Wilk test for normality
        statistic, p_value = shapiro(clean_errors)

        return statistic, p_value

    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, Callable] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for performance metrics.

        Args:
            y_true: True values [n_samples]
            y_pred: Predicted values [n_samples]
            metrics: Dict of metric_name -> metric_function. Default includes MAE, RMSE, R²

        Returns:
            Dict mapping metric names to (lower_bound, upper_bound) confidence intervals
        """
        if metrics is None:
            metrics = {
                "MAE": lambda y_t, y_p: np.mean(np.abs(y_t - y_p)),
                "RMSE": lambda y_t, y_p: np.sqrt(np.mean((y_t - y_p) ** 2)),
                "R2": lambda y_t, y_p: 1
                - np.sum((y_t - y_p) ** 2) / np.sum((y_t - np.mean(y_t)) ** 2),
            }

        n = len(y_true)
        bootstrap_results = defaultdict(list)

        np.random.seed(42)
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute metrics
            for metric_name, metric_fn in metrics.items():
                try:
                    value = metric_fn(y_true_boot, y_pred_boot)
                    if np.isfinite(value):
                        bootstrap_results[metric_name].append(value)
                except Exception:
                    continue

        # Compute confidence intervals
        confidence_intervals = {}
        alpha_half = self.alpha / 2

        for metric_name, values in bootstrap_results.items():
            if len(values) > 0:
                lower = np.percentile(values, alpha_half * 100)
                upper = np.percentile(values, (1 - alpha_half) * 100)
                confidence_intervals[metric_name] = (lower, upper)
            else:
                confidence_intervals[metric_name] = (np.nan, np.nan)

        return confidence_intervals

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        model_fn: Optional[Callable] = None,
    ) -> EvaluationResult:
        """
        Run complete evaluation suite.

        Args:
            y_true: True values [n_samples]
            y_pred: Predicted values [n_samples]
            confidences: Confidence scores [n_samples] (optional)
            X: Feature matrix for cross-validation (optional)
            model_fn: Model function for cross-validation (optional)

        Returns:
            EvaluationResult with all computed metrics
        """
        # Correlations with confidence intervals
        pearson_r, pearson_ci, spearman_r, spearman_ci = (
            self.compute_correlations_with_ci(y_true, y_pred)
        )

        # Calibration error (if confidences provided)
        calibration_error = 0.0
        if confidences is not None:
            calibration_error = self.compute_calibration_error(
                y_true, y_pred, confidences
            )

        # Error distribution analysis
        errors = y_true - y_pred
        shapiro_stat, shapiro_p = self.analyze_error_distribution(errors)

        # Bootstrap confidence intervals
        bootstrap_metrics = self.bootstrap_confidence_intervals(y_true, y_pred)

        # Cross-validation (if X and model_fn provided)
        cv_scores, mean_cv, std_cv = [], 0.0, 0.0
        if X is not None and model_fn is not None:
            cv_scores, mean_cv, std_cv = self.k_fold_cross_validation(
                X, y_true, model_fn
            )

        return EvaluationResult(
            pearson_correlation=pearson_r,
            pearson_ci=pearson_ci,
            spearman_correlation=spearman_r,
            spearman_ci=spearman_ci,
            calibration_error=calibration_error,
            shapiro_statistic=shapiro_stat,
            shapiro_p_value=shapiro_p,
            bootstrap_metrics=bootstrap_metrics,
            cv_scores=cv_scores,
            mean_cv_score=mean_cv,
            std_cv_score=std_cv,
        )
