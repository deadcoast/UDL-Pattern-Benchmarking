"""
Quality trend prediction using historical data.

This module provides capabilities to predict future UDL quality trends
using machine learning and statistical forecasting methods.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from udl_rating_framework.core.pipeline import QualityReport


@dataclass
class PredictionResult:
    """Results from trend prediction."""

    predictions: pd.Series
    confidence_intervals: Tuple[pd.Series, pd.Series]
    model_performance: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    prediction_horizon: int
    model_type: str


@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis results."""

    historical_trends: Dict[str, float]  # metric -> trend slope
    # metric -> season -> effect
    seasonal_patterns: Dict[str, Dict[str, float]]
    volatility_metrics: Dict[str, float]  # metric -> volatility
    regime_changes: Dict[str, List[datetime]]  # metric -> change points
    forecast_accuracy: Dict[str, float]  # metric -> accuracy score


class TrendPredictor:
    """
    Predicts UDL quality trends using historical data.

    Provides comprehensive forecasting capabilities including:
    - Multiple forecasting models (linear, polynomial, ensemble)
    - Seasonal pattern detection and modeling
    - Regime change detection
    - Uncertainty quantification
    - Model validation and selection
    """

    def __init__(
        self,
        prediction_horizon: int = 30,
        validation_split: float = 0.2,
        models: List[str] = None,
    ):
        """
        Initialize trend predictor.

        Args:
            prediction_horizon: Number of periods to forecast
            validation_split: Fraction of data for validation
            models: List of models to use ('linear', 'polynomial', 'random_forest')
        """
        self.prediction_horizon = prediction_horizon
        self.validation_split = validation_split
        self.models = models or ["linear", "polynomial", "random_forest"]
        self.fitted_models = {}
        self.scalers = {}

    def predict_quality_trends(
        self, reports: List[QualityReport], udl_file: str, metrics: List[str] = None
    ) -> Dict[str, PredictionResult]:
        """
        Predict quality trends for specified metrics.

        Args:
            reports: Historical quality reports
            udl_file: UDL file to predict trends for
            metrics: List of metrics to predict (default: all available)

        Returns:
            Dict mapping metric names to PredictionResult
        """
        # Filter reports for the specific UDL file
        file_reports = [r for r in reports if r.udl_file == udl_file]

        if len(file_reports) < 10:
            raise ValueError(
                f"Insufficient data: need at least 10 reports, got {len(file_reports)}"
            )

        # Sort by timestamp
        file_reports = sorted(file_reports, key=lambda r: r.timestamp)

        # Determine metrics to predict
        if metrics is None:
            metrics = ["overall_score", "confidence"]
            if file_reports:
                metrics.extend(file_reports[0].metric_scores.keys())

        results = {}

        for metric in metrics:
            try:
                prediction_result = self._predict_single_metric(
                    file_reports, metric)
                results[metric] = prediction_result
            except Exception as e:
                warnings.warn(f"Could not predict metric {metric}: {e}")
                continue

        return results

    def analyze_portfolio_trends(
        self,
        reports: List[QualityReport],
        project_mapping: Optional[Dict[str, str]] = None,
    ) -> TrendAnalysis:
        """
        Analyze trends across entire portfolio.

        Args:
            reports: All quality reports
            project_mapping: Optional mapping from UDL file to project

        Returns:
            TrendAnalysis with portfolio-wide trend insights
        """
        # Group reports by project
        if project_mapping:
            project_reports = {}
            for report in reports:
                project = project_mapping.get(report.udl_file, "unknown")
                if project not in project_reports:
                    project_reports[project] = []
                project_reports[project].append(report)
        else:
            # Use directory structure
            project_reports = {}
            for report in reports:
                project = (
                    report.udl_file.split("/")[0]
                    if "/" in report.udl_file
                    else "default"
                )
                if project not in project_reports:
                    project_reports[project] = []
                project_reports[project].append(report)

        # Analyze trends for each project
        historical_trends = {}
        seasonal_patterns = {}
        volatility_metrics = {}
        regime_changes = {}
        forecast_accuracy = {}

        all_metrics = set()
        for report in reports:
            all_metrics.update(["overall_score", "confidence"])
            all_metrics.update(report.metric_scores.keys())

        for metric in all_metrics:
            try:
                # Collect all data for this metric across projects
                metric_data = []
                for project, project_reports_list in project_reports.items():
                    if len(project_reports_list) >= 5:  # Minimum data requirement
                        project_reports_list = sorted(
                            project_reports_list, key=lambda r: r.timestamp
                        )

                        for report in project_reports_list:
                            if metric == "overall_score":
                                value = report.overall_score
                            elif metric == "confidence":
                                value = report.confidence
                            else:
                                value = report.metric_scores.get(
                                    metric, np.nan)

                            if not np.isnan(value):
                                metric_data.append(
                                    (report.timestamp, value, project))

                if len(metric_data) < 10:
                    continue

                # Convert to DataFrame for analysis
                df = pd.DataFrame(
                    metric_data, columns=["timestamp", "value", "project"]
                )
                df = df.sort_values("timestamp")

                # Compute historical trend
                timestamps_numeric = [
                    (t - df["timestamp"].iloc[0]).total_seconds()
                    for t in df["timestamp"]
                ]
                slope, _, _, p_value, _ = stats.linregress(
                    timestamps_numeric, df["value"]
                )
                historical_trends[metric] = slope

                # Compute volatility
                returns = df["value"].pct_change().dropna()
                volatility_metrics[metric] = returns.std() if len(
                    returns) > 1 else 0.0

                # Detect seasonal patterns (simplified)
                seasonal_patterns[metric] = self._detect_seasonal_patterns(df)

                # Detect regime changes
                regime_changes[metric] = self._detect_regime_changes(df)

                # Estimate forecast accuracy using cross-validation
                forecast_accuracy[metric] = self._estimate_forecast_accuracy(
                    df)

            except Exception as e:
                warnings.warn(
                    f"Could not analyze portfolio trends for metric {metric}: {e}"
                )
                continue

        return TrendAnalysis(
            historical_trends=historical_trends,
            seasonal_patterns=seasonal_patterns,
            volatility_metrics=volatility_metrics,
            regime_changes=regime_changes,
            forecast_accuracy=forecast_accuracy,
        )

    def _predict_single_metric(
        self, reports: List[QualityReport], metric: str
    ) -> PredictionResult:
        """Predict trends for a single metric."""
        # Extract time series data
        timestamps = [r.timestamp for r in reports]

        if metric == "overall_score":
            values = [r.overall_score for r in reports]
        elif metric == "confidence":
            values = [r.confidence for r in reports]
        else:
            values = [r.metric_scores.get(metric, np.nan) for r in reports]

        # Remove NaN values
        valid_data = [(t, v)
                      for t, v in zip(timestamps, values) if not np.isnan(v)]

        if len(valid_data) < 10:
            raise ValueError(f"Insufficient valid data for metric {metric}")

        timestamps, values = zip(*valid_data)
        timestamps = list(timestamps)
        values = list(values)

        # Create time series
        ts = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        ts = ts.sort_index()

        # Prepare features
        X, y = self._prepare_features(ts)

        # Split data for validation
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train and evaluate models
        best_model = None
        best_score = float("inf")
        best_model_type = None
        model_performances = {}

        for model_type in self.models:
            try:
                model, scaler = self._train_model(X_train, y_train, model_type)

                # Validate model
                X_val_scaled = scaler.transform(X_val) if scaler else X_val
                y_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, y_pred)

                model_performances[model_type] = {
                    "mse": mse,
                    "mae": mean_absolute_error(y_val, y_pred),
                    "r2": r2_score(y_val, y_pred),
                }

                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_type = model_type
                    self.fitted_models[metric] = model
                    self.scalers[metric] = scaler

            except Exception as e:
                warnings.warn(f"Could not train {model_type} model: {e}")
                continue

        if best_model is None:
            raise ValueError("No models could be trained successfully")

        # Generate predictions
        last_timestamp = ts.index[-1]
        future_timestamps = [
            last_timestamp + timedelta(days=i + 1)
            for i in range(self.prediction_horizon)
        ]

        # Prepare future features
        future_X = self._prepare_future_features(ts, future_timestamps)

        # Scale features if necessary
        scaler = self.scalers.get(metric)
        if scaler:
            future_X_scaled = scaler.transform(future_X)
        else:
            future_X_scaled = future_X

        # Make predictions
        predictions = best_model.predict(future_X_scaled)

        # Ensure predictions are in valid range [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)

        # Create prediction series
        prediction_series = pd.Series(
            predictions, index=pd.DatetimeIndex(future_timestamps)
        )

        # Generate confidence intervals (simplified approach)
        residuals = y_val - best_model.predict(X_val_scaled)
        residual_std = np.std(residuals)

        margin_of_error = 1.96 * residual_std  # 95% confidence interval
        lower_bound = pd.Series(
            predictions - margin_of_error, index=prediction_series.index
        )
        upper_bound = pd.Series(
            predictions + margin_of_error, index=prediction_series.index
        )

        # Clip confidence intervals
        lower_bound = lower_bound.clip(0.0, 1.0)
        upper_bound = upper_bound.clip(0.0, 1.0)

        # Get feature importance (if available)
        feature_importance = None
        if hasattr(best_model, "feature_importances_"):
            feature_names = self._get_feature_names()
            feature_importance = dict(
                zip(feature_names, best_model.feature_importances_)
            )
        elif hasattr(best_model, "coef_"):
            feature_names = self._get_feature_names()
            feature_importance = dict(
                zip(feature_names, np.abs(best_model.coef_)))

        return PredictionResult(
            predictions=prediction_series,
            confidence_intervals=(lower_bound, upper_bound),
            model_performance=model_performances[best_model_type],
            feature_importance=feature_importance,
            prediction_horizon=self.prediction_horizon,
            model_type=best_model_type,
        )

    def _prepare_features(self, ts: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training."""
        # Convert timestamps to numeric
        timestamps_numeric = np.array(
            [(t - ts.index[0]).total_seconds() / (24 * 3600) for t in ts.index]
        )  # Days since start

        features = []
        targets = []

        # Use sliding window approach
        window_size = min(5, len(ts) // 3)  # Adaptive window size

        for i in range(window_size, len(ts)):
            # Time-based features
            time_features = [
                timestamps_numeric[i],  # Time trend
                timestamps_numeric[i] % 7,  # Day of week effect
                timestamps_numeric[i] % 30,  # Monthly effect
            ]

            # Lag features
            lag_features = ts.iloc[i - window_size: i].values.tolist()

            # Statistical features
            recent_values = ts.iloc[max(0, i - window_size): i]
            stat_features = [
                recent_values.mean(),
                recent_values.std() if len(recent_values) > 1 else 0.0,
                # Last value
                recent_values.iloc[-1] if len(recent_values) > 0 else 0.0,
            ]

            # Combine all features
            feature_vector = time_features + lag_features + stat_features
            features.append(feature_vector)
            targets.append(ts.iloc[i])

        return np.array(features), np.array(targets)

    def _prepare_future_features(
        self, ts: pd.Series, future_timestamps: List[datetime]
    ) -> np.ndarray:
        """Prepare features for future predictions."""
        # Get the last few values for lag features
        window_size = min(5, len(ts) // 3)
        recent_values = ts.iloc[-window_size:].values

        features = []

        for i, future_timestamp in enumerate(future_timestamps):
            # Time-based features
            days_since_start = (future_timestamp - ts.index[0]).total_seconds() / (
                24 * 3600
            )
            time_features = [
                days_since_start,
                days_since_start % 7,
                days_since_start % 30,
            ]

            # For lag features, use recent actual values for first few predictions,
            # then use predicted values
            if i == 0:
                lag_features = recent_values.tolist()
            else:
                # Use a mix of recent actual values and recent predictions
                # This is a simplified approach - more sophisticated methods exist
                lag_features = recent_values.tolist()

            # Statistical features based on recent history
            stat_features = [
                np.mean(recent_values),
                np.std(recent_values) if len(recent_values) > 1 else 0.0,
                recent_values[-1] if len(recent_values) > 0 else 0.0,
            ]

            # Combine all features
            feature_vector = time_features + lag_features + stat_features
            features.append(feature_vector)

        return np.array(features)

    def _train_model(
        self, X: np.ndarray, y: np.ndarray, model_type: str
    ) -> Tuple[Any, Optional[StandardScaler]]:
        """Train a specific model type."""
        scaler = None

        if model_type == "linear":
            model = LinearRegression()
            model.fit(X, y)

        elif model_type == "polynomial":
            # Use polynomial features with regularization
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(X)

            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)

            model = Ridge(alpha=1.0)
            model.fit(X_poly_scaled, y)

            # Wrap model to handle polynomial transformation
            class PolynomialModel:
                """Wrapper for polynomial regression model with feature transformation.

                This class wraps a linear regression model with polynomial feature
                transformation to provide a consistent interface for prediction.
                """

                def __init__(self, model, poly_features):
                    """Initialize the polynomial model wrapper.

                    Args:
                        model: The underlying linear regression model.
                        poly_features: The polynomial feature transformer.
                    """
                    self.model = model
                    self.poly_features = poly_features

                def predict(self, X):
                    """Make predictions using the polynomial model.

                    Args:
                        X: Input features to transform and predict.

                    Returns:
                        Predicted values.
                    """
                    X_poly = self.poly_features.transform(X)
                    return self.model.predict(X_poly)

                @property
                def coef_(self):
                    """Get the model coefficients.

                    Returns:
                        The coefficients of the underlying linear model.
                    """
                    return self.model.coef_

            model = PolynomialModel(model, poly_features)

        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            model.fit(X, y)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model, scaler

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation."""
        return [
            "time_trend",
            "day_of_week",
            "monthly_cycle",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "recent_mean",
            "recent_std",
            "last_value",
        ]

    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect seasonal patterns in the data."""
        # Simplified seasonal pattern detection
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        patterns = {}

        # Day of week effect
        if len(df) > 14:  # Need at least 2 weeks of data
            dow_means = df.groupby("day_of_week")["value"].mean()
            overall_mean = df["value"].mean()
            dow_effect = (dow_means - overall_mean).abs().mean()
            patterns["day_of_week"] = dow_effect

        # Monthly effect
        if len(df) > 60:  # Need at least 2 months of data
            month_means = df.groupby("month")["value"].mean()
            month_effect = (month_means - overall_mean).abs().mean()
            patterns["monthly"] = month_effect

        return patterns

    def _detect_regime_changes(self, df: pd.DataFrame) -> List[datetime]:
        """Detect regime changes in the data."""
        # Simplified regime change detection using rolling variance
        if len(df) < 20:
            return []

        window_size = max(5, len(df) // 10)
        rolling_var = df["value"].rolling(window=window_size).var()

        # Find points where variance changes significantly
        var_changes = rolling_var.diff().abs()
        threshold = var_changes.quantile(0.95)  # Top 5% of changes

        change_points = []
        for idx in var_changes[var_changes > threshold].index:
            if idx < len(df):
                change_points.append(df.iloc[idx]["timestamp"])

        return change_points

    def _estimate_forecast_accuracy(self, df: pd.DataFrame) -> float:
        """Estimate forecast accuracy using time series cross-validation."""
        if len(df) < 20:
            return 0.0

        # Simple time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        # Prepare simple features (just time trend and lag)
        timestamps_numeric = [
            (t - df["timestamp"].iloc[0]).total_seconds() / (24 * 3600)
            for t in df["timestamp"]
        ]

        for train_idx, test_idx in tscv.split(df):
            try:
                # Simple linear model for accuracy estimation
                X_train = np.array(timestamps_numeric)[
                    train_idx].reshape(-1, 1)
                y_train = df["value"].iloc[train_idx].values
                X_test = np.array(timestamps_numeric)[test_idx].reshape(-1, 1)
                y_test = df["value"].iloc[test_idx].values

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Compute R² score
                score = r2_score(y_test, y_pred)
                scores.append(max(0.0, score))  # Ensure non-negative

            except Exception:
                continue

        return np.mean(scores) if scores else 0.0

    def generate_prediction_report(
        self, predictions: Dict[str, PredictionResult], udl_file: str
    ) -> str:
        """Generate comprehensive prediction report."""
        report_lines = [
            "# Quality Trend Prediction Report",
            f"**UDL File:** {udl_file}",
            f"**Prediction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Prediction Horizon:** {self.prediction_horizon} days",
            "",
            "## Executive Summary",
            "",
        ]

        # Overall assessment
        if "overall_score" in predictions:
            overall_pred = predictions["overall_score"]
            current_trend = (
                overall_pred.predictions.iloc[-1] -
                overall_pred.predictions.iloc[0]
            ) / len(overall_pred.predictions)

            if current_trend > 0.001:
                trend_desc = "improving"
            elif current_trend < -0.001:
                trend_desc = "declining"
            else:
                trend_desc = "stable"

            report_lines.extend(
                [
                    f"**Overall Quality Trend:** {trend_desc} ({current_trend:.6f} per day)",
                    f"**Model Used:** {overall_pred.model_type}",
                    f"**Model Performance (R²):** {overall_pred.model_performance.get('r2', 0.0):.3f}",
                    f"**Prediction Confidence:** {1.0 - overall_pred.model_performance.get('mse', 1.0):.3f}",
                    "",
                ]
            )

        # Detailed predictions
        report_lines.extend(["## Detailed Predictions", ""])

        for metric, pred_result in predictions.items():
            report_lines.extend(
                [
                    f"### {metric.replace('_', ' ').title()}",
                    "",
                    f"**Model Type:** {pred_result.model_type}",
                    "**Model Performance:**",
                    f"- R² Score: {pred_result.model_performance.get('r2', 0.0):.3f}",
                    f"- Mean Absolute Error: {pred_result.model_performance.get('mae', 0.0):.6f}",
                    f"- Mean Squared Error: {pred_result.model_performance.get('mse', 0.0):.6f}",
                    "",
                    "**Predicted Values:**",
                    f"- Next 7 days: {pred_result.predictions.iloc[:7].mean():.3f} ± {(pred_result.confidence_intervals[1].iloc[:7] - pred_result.confidence_intervals[0].iloc[:7]).mean() / 2:.3f}",
                    f"- Next 30 days: {pred_result.predictions.mean():.3f} ± {(pred_result.confidence_intervals[1] - pred_result.confidence_intervals[0]).mean() / 2:.3f}",
                    "",
                ]
            )

            if pred_result.feature_importance:
                report_lines.extend(["**Feature Importance:**"])
                sorted_features = sorted(
                    pred_result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                # Top 5 features
                for feature, importance in sorted_features[:5]:
                    report_lines.append(f"- {feature}: {importance:.3f}")
                report_lines.append("")

        # Recommendations
        report_lines.extend(["## Recommendations", ""])

        if "overall_score" in predictions:
            overall_pred = predictions["overall_score"]
            future_mean = overall_pred.predictions.mean()
            current_value = overall_pred.predictions.iloc[
                0
            ]  # Approximate current value

            if future_mean > current_value + 0.05:
                report_lines.append(
                    "✅ Quality is expected to improve significantly. Continue current practices."
                )
            elif future_mean < current_value - 0.05:
                report_lines.append(
                    "⚠️ Quality is expected to decline. Consider implementing corrective measures."
                )
            else:
                report_lines.append(
                    "ℹ️ Quality is expected to remain stable. Monitor for emerging trends."
                )

            # Model reliability assessment
            r2_score = overall_pred.model_performance.get("r2", 0.0)
            if r2_score > 0.7:
                report_lines.append(
                    "🎯 High prediction confidence - model is reliable for decision making."
                )
            elif r2_score > 0.4:
                report_lines.append(
                    "⚠️ Moderate prediction confidence - use predictions as guidance only."
                )
            else:
                report_lines.append(
                    "❌ Low prediction confidence - predictions should be interpreted cautiously."
                )

        return "\n".join(report_lines)
