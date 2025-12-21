"""
Time-series analysis of UDL quality evolution.

This module provides capabilities to analyze how UDL quality metrics
change over time, identifying trends, patterns, and anomalies.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from udl_rating_framework.core.pipeline import QualityReport


@dataclass
class TimeSeriesMetrics:
    """Metrics for time series analysis."""

    trend_slope: float
    trend_p_value: float
    seasonality_strength: float
    volatility: float
    autocorrelation: float
    change_points: List[datetime]
    anomalies: List[Tuple[datetime, float, str]]  # (timestamp, value, reason)


@dataclass
class QualityEvolution:
    """Quality evolution analysis results."""

    metric_name: str
    time_series: pd.Series
    trend_analysis: TimeSeriesMetrics
    forecast: Optional[pd.Series]
    confidence_intervals: Optional[Tuple[pd.Series, pd.Series]]
    summary: str


class TimeSeriesAnalyzer:
    """
    Analyzes UDL quality evolution over time.

    Provides comprehensive time-series analysis including:
    - Trend detection and significance testing
    - Seasonality analysis
    - Change point detection
    - Anomaly detection
    - Quality forecasting
    """

    def __init__(
        self,
        min_observations: int = 10,
        anomaly_threshold: float = 2.0,
        seasonality_periods: List[int] = None,
    ):
        """
        Initialize time series analyzer.

        Args:
            min_observations: Minimum number of observations for analysis
            anomaly_threshold: Z-score threshold for anomaly detection
            seasonality_periods: Periods to check for seasonality (default: [7, 30, 90])
        """
        self.min_observations = min_observations
        self.anomaly_threshold = anomaly_threshold
        self.seasonality_periods = seasonality_periods or [7, 30, 90]

    def analyze_quality_evolution(
        self,
        reports: List[QualityReport],
        udl_file: str,
        metric_name: str = "overall_score",
    ) -> QualityEvolution:
        """
        Analyze quality evolution for a specific UDL file and metric.

        Args:
            reports: List of quality reports over time
            udl_file: UDL file to analyze
            metric_name: Metric to analyze (default: 'overall_score')

        Returns:
            QualityEvolution analysis results
        """
        # Filter reports for the specific UDL file
        file_reports = [r for r in reports if r.udl_file == udl_file]

        if len(file_reports) < self.min_observations:
            raise ValueError(
                f"Insufficient data: need at least {self.min_observations} observations, got {len(file_reports)}"
            )

        # Create time series
        timestamps = [r.timestamp for r in file_reports]

        if metric_name == "overall_score":
            values = [r.overall_score for r in file_reports]
        elif metric_name == "confidence":
            values = [r.confidence for r in file_reports]
        else:
            values = [r.metric_scores.get(metric_name, np.nan)
                      for r in file_reports]

        # Remove NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        timestamps = [timestamps[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]

        if len(values) < self.min_observations:
            raise ValueError(
                "Insufficient valid data after removing NaN values")

        # Create pandas Series
        time_series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        time_series = time_series.sort_index()

        # Perform trend analysis
        trend_analysis = self._analyze_trends(time_series)

        # Generate forecast
        forecast, confidence_intervals = self._forecast_quality(time_series)

        # Generate summary
        summary = self._generate_evolution_summary(time_series, trend_analysis)

        return QualityEvolution(
            metric_name=metric_name,
            time_series=time_series,
            trend_analysis=trend_analysis,
            forecast=forecast,
            confidence_intervals=confidence_intervals,
            summary=summary,
        )

    def analyze_multiple_metrics(
        self, reports: List[QualityReport], udl_file: str, metrics: List[str] = None
    ) -> Dict[str, QualityEvolution]:
        """
        Analyze evolution of multiple metrics for a UDL file.

        Args:
            reports: List of quality reports over time
            udl_file: UDL file to analyze
            metrics: List of metrics to analyze (default: all available)

        Returns:
            Dict mapping metric names to QualityEvolution results
        """
        if metrics is None:
            # Get all available metrics from the first report
            file_reports = [r for r in reports if r.udl_file == udl_file]
            if not file_reports:
                raise ValueError(f"No reports found for UDL file: {udl_file}")

            metrics = ["overall_score", "confidence"] + list(
                file_reports[0].metric_scores.keys()
            )

        results = {}
        for metric in metrics:
            try:
                evolution = self.analyze_quality_evolution(
                    reports, udl_file, metric)
                results[metric] = evolution
            except ValueError as e:
                warnings.warn(f"Could not analyze metric {metric}: {e}")
                continue

        return results

    def _analyze_trends(self, time_series: pd.Series) -> TimeSeriesMetrics:
        """Analyze trends in time series data."""
        # Convert timestamps to numeric for regression
        timestamps_numeric = np.array(
            [(t - time_series.index[0]).total_seconds()
             for t in time_series.index]
        )
        values = time_series.values

        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps_numeric, values
        )

        # Seasonality analysis
        seasonality_strength = self._detect_seasonality(time_series)

        # Volatility (standard deviation of returns)
        returns = time_series.pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0.0

        # Autocorrelation (lag-1)
        autocorr = time_series.autocorr(lag=1) if len(time_series) > 1 else 0.0
        if np.isnan(autocorr):
            autocorr = 0.0

        # Change point detection
        change_points = self._detect_change_points(time_series)

        # Anomaly detection
        anomalies = self._detect_anomalies(time_series)

        return TimeSeriesMetrics(
            trend_slope=slope,
            trend_p_value=p_value,
            seasonality_strength=seasonality_strength,
            volatility=volatility,
            autocorrelation=autocorr,
            change_points=change_points,
            anomalies=anomalies,
        )

    def _detect_seasonality(self, time_series: pd.Series) -> float:
        """Detect seasonality strength in time series."""
        if len(time_series) < max(self.seasonality_periods) * 2:
            return 0.0

        max_strength = 0.0

        for period in self.seasonality_periods:
            if len(time_series) >= period * 2:
                # Compute autocorrelation at seasonal lag
                try:
                    seasonal_autocorr = time_series.autocorr(lag=period)
                    if not np.isnan(seasonal_autocorr):
                        max_strength = max(
                            max_strength, abs(seasonal_autocorr))
                except Exception:
                    continue

        return max_strength

    def _detect_change_points(self, time_series: pd.Series) -> List[datetime]:
        """Detect change points in time series using simple variance-based method."""
        if len(time_series) < 10:
            return []

        values = time_series.values
        change_points = []

        # Simple change point detection using rolling variance
        window_size = max(5, len(values) // 10)

        for i in range(window_size, len(values) - window_size):
            # Variance before and after point
            var_before = np.var(values[i - window_size: i])
            var_after = np.var(values[i: i + window_size])

            # F-test for variance change
            if var_before > 0 and var_after > 0:
                f_stat = max(var_before, var_after) / \
                    min(var_before, var_after)
                # Simple threshold (could be improved with proper statistical test)
                if f_stat > 4.0:  # Significant variance change
                    change_points.append(time_series.index[i])

        return change_points

    def _detect_anomalies(
        self, time_series: pd.Series
    ) -> List[Tuple[datetime, float, str]]:
        """Detect anomalies in time series using statistical methods."""
        if len(time_series) < 5:
            return []

        values = time_series.values
        anomalies = []

        # Z-score based anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val > 0:
            z_scores = np.abs((values - mean_val) / std_val)

            for i, (timestamp, value, z_score) in enumerate(
                zip(time_series.index, values, z_scores)
            ):
                if z_score > self.anomaly_threshold:
                    reason = (
                        f"Z-score: {z_score:.2f} (threshold: {self.anomaly_threshold})"
                    )
                    anomalies.append((timestamp, value, reason))

        # Peak detection for sudden spikes
        peaks, properties = find_peaks(
            values, height=mean_val + 2 * std_val, distance=3
        )
        for peak_idx in peaks:
            timestamp = time_series.index[peak_idx]
            value = values[peak_idx]
            reason = f"Peak detection: value {value:.3f} significantly above trend"
            anomalies.append((timestamp, value, reason))

        # Valley detection for sudden drops
        valleys, properties = find_peaks(
            -values, height=-(mean_val - 2 * std_val), distance=3
        )
        for valley_idx in valleys:
            timestamp = time_series.index[valley_idx]
            value = values[valley_idx]
            reason = f"Valley detection: value {value:.3f} significantly below trend"
            anomalies.append((timestamp, value, reason))

        # Remove duplicates (same timestamp)
        seen_timestamps = set()
        unique_anomalies = []
        for anomaly in anomalies:
            if anomaly[0] not in seen_timestamps:
                unique_anomalies.append(anomaly)
                seen_timestamps.add(anomaly[0])

        return unique_anomalies

    def _forecast_quality(
        self, time_series: pd.Series, forecast_periods: int = 10
    ) -> Tuple[Optional[pd.Series], Optional[Tuple[pd.Series, pd.Series]]]:
        """Generate quality forecast using simple linear extrapolation."""
        if len(time_series) < 5:
            return None, None

        # Convert to numeric timestamps
        timestamps_numeric = np.array(
            [(t - time_series.index[0]).total_seconds()
             for t in time_series.index]
        )
        values = time_series.values

        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps_numeric, values
        )

        # Generate future timestamps
        last_timestamp = time_series.index[-1]
        time_delta = (
            (time_series.index[-1] - time_series.index[-2])
            if len(time_series) > 1
            else timedelta(days=1)
        )

        future_timestamps = [
            last_timestamp + (i + 1) * time_delta for i in range(forecast_periods)
        ]
        future_timestamps_numeric = np.array(
            [(t - time_series.index[0]).total_seconds()
             for t in future_timestamps]
        )

        # Generate forecasts
        forecast_values = slope * future_timestamps_numeric + intercept

        # Ensure forecasts are within valid range [0, 1]
        forecast_values = np.clip(forecast_values, 0.0, 1.0)

        forecast = pd.Series(
            forecast_values, index=pd.DatetimeIndex(future_timestamps))

        # Generate confidence intervals (simple approach using residual standard error)
        residuals = values - (slope * timestamps_numeric + intercept)
        residual_std = np.std(residuals)

        # 95% confidence intervals
        margin_of_error = 1.96 * residual_std
        lower_bound = pd.Series(
            forecast_values - margin_of_error, index=forecast.index)
        upper_bound = pd.Series(
            forecast_values + margin_of_error, index=forecast.index)

        # Clip confidence intervals to valid range
        lower_bound = lower_bound.clip(0.0, 1.0)
        upper_bound = upper_bound.clip(0.0, 1.0)

        confidence_intervals = (lower_bound, upper_bound)

        return forecast, confidence_intervals

    def _generate_evolution_summary(
        self, time_series: pd.Series, trend_analysis: TimeSeriesMetrics
    ) -> str:
        """Generate human-readable summary of quality evolution."""
        summary_parts = []

        # Overall trend
        if trend_analysis.trend_p_value < 0.05:
            if trend_analysis.trend_slope > 0:
                summary_parts.append(
                    f"Quality shows significant improvement over time (slope: {trend_analysis.trend_slope:.6f}, p-value: {trend_analysis.trend_p_value:.4f})"
                )
            else:
                summary_parts.append(
                    f"Quality shows significant decline over time (slope: {trend_analysis.trend_slope:.6f}, p-value: {trend_analysis.trend_p_value:.4f})"
                )
        else:
            summary_parts.append(
                f"No significant trend detected (p-value: {trend_analysis.trend_p_value:.4f})"
            )

        # Volatility
        if trend_analysis.volatility > 0.1:
            summary_parts.append(
                f"High volatility detected (σ: {trend_analysis.volatility:.3f})"
            )
        elif trend_analysis.volatility > 0.05:
            summary_parts.append(
                f"Moderate volatility (σ: {trend_analysis.volatility:.3f})"
            )
        else:
            summary_parts.append(
                f"Low volatility (σ: {trend_analysis.volatility:.3f})")

        # Seasonality
        if trend_analysis.seasonality_strength > 0.3:
            summary_parts.append(
                f"Strong seasonal patterns detected (strength: {trend_analysis.seasonality_strength:.3f})"
            )
        elif trend_analysis.seasonality_strength > 0.1:
            summary_parts.append(
                f"Weak seasonal patterns (strength: {trend_analysis.seasonality_strength:.3f})"
            )

        # Change points
        if trend_analysis.change_points:
            summary_parts.append(
                f"{len(trend_analysis.change_points)} significant change points detected"
            )

        # Anomalies
        if trend_analysis.anomalies:
            summary_parts.append(
                f"{len(trend_analysis.anomalies)} anomalies detected")

        # Autocorrelation
        if abs(trend_analysis.autocorrelation) > 0.5:
            summary_parts.append(
                f"Strong autocorrelation (ρ: {trend_analysis.autocorrelation:.3f})"
            )

        # Current status
        current_value = time_series.iloc[-1]
        mean_value = time_series.mean()

        if current_value > mean_value + time_series.std():
            summary_parts.append(
                f"Current quality ({current_value:.3f}) is above historical average"
            )
        elif current_value < mean_value - time_series.std():
            summary_parts.append(
                f"Current quality ({current_value:.3f}) is below historical average"
            )
        else:
            summary_parts.append(
                f"Current quality ({current_value:.3f}) is near historical average ({mean_value:.3f})"
            )

        return ". ".join(summary_parts) + "."

    def generate_time_series_report(
        self, evolution_results: Dict[str, QualityEvolution], udl_file: str
    ) -> str:
        """Generate comprehensive time series analysis report."""
        report_lines = [
            "# Time Series Analysis Report",
            f"**UDL File:** {udl_file}",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]

        # Overall assessment
        overall_evolution = evolution_results.get("overall_score")
        if overall_evolution:
            report_lines.extend(
                [
                    f"**Overall Quality Trend:** {overall_evolution.summary}",
                    "",
                    f"- **Data Points:** {len(overall_evolution.time_series)}",
                    f"- **Time Range:** {overall_evolution.time_series.index[0].strftime('%Y-%m-%d')} to {overall_evolution.time_series.index[-1].strftime('%Y-%m-%d')}",
                    f"- **Current Score:** {overall_evolution.time_series.iloc[-1]:.3f}",
                    f"- **Average Score:** {overall_evolution.time_series.mean():.3f}",
                    f"- **Score Range:** [{overall_evolution.time_series.min():.3f}, {overall_evolution.time_series.max():.3f}]",
                    "",
                ]
            )

        # Detailed metric analysis
        report_lines.extend(["## Detailed Metric Analysis", ""])

        for metric_name, evolution in evolution_results.items():
            report_lines.extend(
                [
                    f"### {metric_name.replace('_', ' ').title()}",
                    "",
                    evolution.summary,
                    "",
                    "**Statistical Properties:**",
                    f"- Trend Slope: {evolution.trend_analysis.trend_slope:.6f}",
                    f"- Trend Significance: p = {evolution.trend_analysis.trend_p_value:.4f}",
                    f"- Volatility: {evolution.trend_analysis.volatility:.3f}",
                    f"- Autocorrelation: {evolution.trend_analysis.autocorrelation:.3f}",
                    f"- Seasonality Strength: {evolution.trend_analysis.seasonality_strength:.3f}",
                    "",
                ]
            )

            if evolution.trend_analysis.change_points:
                report_lines.extend(["**Change Points:**"])
                for cp in evolution.trend_analysis.change_points:
                    report_lines.append(
                        f"- {cp.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")

            if evolution.trend_analysis.anomalies:
                report_lines.extend(["**Anomalies:**"])
                for timestamp, value, reason in evolution.trend_analysis.anomalies:
                    report_lines.append(
                        f"- {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {value:.3f} ({reason})"
                    )
                report_lines.append("")

        # Recommendations
        report_lines.extend(["## Recommendations", ""])

        if overall_evolution:
            if overall_evolution.trend_analysis.trend_p_value < 0.05:
                if overall_evolution.trend_analysis.trend_slope > 0:
                    report_lines.append(
                        "✅ Quality is improving significantly. Continue current practices."
                    )
                else:
                    report_lines.append(
                        "⚠️ Quality is declining significantly. Review recent changes and consider corrective actions."
                    )
            else:
                report_lines.append(
                    "ℹ️ Quality is stable. Monitor for emerging trends.")

            if overall_evolution.trend_analysis.volatility > 0.1:
                report_lines.append(
                    "⚠️ High volatility detected. Consider implementing more consistent development practices."
                )

            if len(overall_evolution.trend_analysis.anomalies) > 0:
                report_lines.append(
                    "🔍 Anomalies detected. Investigate the causes of unusual quality changes."
                )

        return "\n".join(report_lines)
