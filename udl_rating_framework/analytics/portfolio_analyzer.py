"""
Comparative analysis across project portfolios.

This module provides capabilities to analyze and compare UDL quality
across multiple projects, identifying patterns, best practices, and areas
for improvement at the portfolio level.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from udl_rating_framework.core.pipeline import QualityReport


@dataclass
class ProjectProfile:
    """Profile of a project's UDL quality characteristics."""

    project_name: str
    udl_files: List[str]
    total_reports: int
    avg_overall_score: float
    std_overall_score: float
    avg_confidence: float
    metric_averages: Dict[str, float]
    quality_trend: float  # slope of quality over time
    consistency_score: float  # 1 - coefficient of variation
    maturity_level: str  # 'developing', 'stable', 'mature'
    risk_level: str  # 'low', 'medium', 'high'


@dataclass
class PortfolioComparison:
    """Results of portfolio comparison analysis."""

    project_profiles: Dict[str, ProjectProfile]
    rankings: Dict[str, List[str]]  # metric -> ranked project list
    clusters: Dict[str, List[str]]  # cluster_name -> project list
    correlations: pd.DataFrame
    outliers: List[str]
    recommendations: Dict[str, List[str]]  # project -> recommendations


@dataclass
class BenchmarkAnalysis:
    """Benchmark analysis results."""

    percentiles: Dict[str, Dict[str, float]]  # metric -> percentile -> value
    industry_standards: Dict[str, float]  # metric -> standard value
    performance_gaps: Dict[str, Dict[str, float]]  # project -> metric -> gap
    improvement_potential: Dict[str, float]  # project -> potential score


class PortfolioAnalyzer:
    """
    Analyzes UDL quality across project portfolios.

    Provides comprehensive portfolio-level analysis including:
    - Project profiling and characterization
    - Comparative ranking and benchmarking
    - Cluster analysis to identify similar projects
    - Risk assessment and maturity evaluation
    - Best practice identification
    """

    def __init__(
        self,
        min_reports_per_project: int = 5,
        clustering_method: str = "kmeans",
        n_clusters: int = 3,
    ):
        """
        Initialize portfolio analyzer.

        Args:
            min_reports_per_project: Minimum reports needed for project analysis
            clustering_method: Clustering method ('kmeans', 'hierarchical')
            n_clusters: Number of clusters for analysis
        """
        self.min_reports_per_project = min_reports_per_project
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters

    def analyze_portfolio(
        self,
        reports: List[QualityReport],
        project_mapping: Optional[Dict[str, str]] = None,
    ) -> PortfolioComparison:
        """
        Analyze quality across a portfolio of projects.

        Args:
            reports: List of quality reports from multiple projects
            project_mapping: Optional mapping from UDL file to project name
                           If None, uses directory structure to infer projects

        Returns:
            PortfolioComparison with comprehensive analysis results
        """
        # Group reports by project
        project_reports = self._group_reports_by_project(
            reports, project_mapping)

        # Filter projects with sufficient data
        filtered_projects = {
            project: reports_list
            for project, reports_list in project_reports.items()
            if len(reports_list) >= self.min_reports_per_project
        }

        if len(filtered_projects) < 2:
            raise ValueError(
                f"Need at least 2 projects with {self.min_reports_per_project}+ reports each"
            )

        # Generate project profiles
        project_profiles = {}
        for project, project_reports_list in filtered_projects.items():
            profile = self._create_project_profile(
                project, project_reports_list)
            project_profiles[project] = profile

        # Generate rankings
        rankings = self._generate_rankings(project_profiles)

        # Perform cluster analysis
        clusters = self._perform_cluster_analysis(project_profiles)

        # Compute correlations
        correlations = self._compute_project_correlations(project_profiles)

        # Identify outliers
        outliers = self._identify_outliers(project_profiles)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            project_profiles, clusters, rankings
        )

        return PortfolioComparison(
            project_profiles=project_profiles,
            rankings=rankings,
            clusters=clusters,
            correlations=correlations,
            outliers=outliers,
            recommendations=recommendations,
        )

    def benchmark_against_standards(
        self,
        portfolio_comparison: PortfolioComparison,
        industry_benchmarks: Optional[Dict[str, float]] = None,
    ) -> BenchmarkAnalysis:
        """
        Benchmark portfolio against industry standards.

        Args:
            portfolio_comparison: Results from portfolio analysis
            industry_benchmarks: Optional industry benchmark values

        Returns:
            BenchmarkAnalysis with benchmarking results
        """
        profiles = portfolio_comparison.project_profiles

        # Compute percentiles across portfolio
        percentiles = self._compute_percentiles(profiles)

        # Set industry standards (use portfolio 75th percentile if not provided)
        if industry_benchmarks is None:
            industry_standards = {
                "overall_score": percentiles["overall_score"][75],
                "confidence": percentiles["confidence"][75],
                "consistency_score": percentiles["consistency_score"][75],
            }

            # Add metric-specific standards
            for profile in profiles.values():
                for metric in profile.metric_averages:
                    if metric not in industry_standards:
                        metric_values = [
                            p.metric_averages.get(metric, 0) for p in profiles.values()
                        ]
                        industry_standards[metric] = np.percentile(
                            metric_values, 75)
        else:
            industry_standards = industry_benchmarks.copy()

        # Compute performance gaps
        performance_gaps = {}
        improvement_potential = {}

        for project, profile in profiles.items():
            gaps = {}

            # Overall score gap
            gaps["overall_score"] = (
                industry_standards["overall_score"] - profile.avg_overall_score
            )

            # Confidence gap
            gaps["confidence"] = (
                industry_standards["confidence"] - profile.avg_confidence
            )

            # Consistency gap
            gaps["consistency_score"] = (
                industry_standards["consistency_score"] -
                profile.consistency_score
            )

            # Metric-specific gaps
            for metric, standard in industry_standards.items():
                if metric in profile.metric_averages:
                    gaps[metric] = standard - profile.metric_averages[metric]

            performance_gaps[project] = gaps

            # Compute improvement potential (weighted average of positive gaps)
            positive_gaps = [max(0, gap) for gap in gaps.values()]
            improvement_potential[project] = (
                np.mean(positive_gaps) if positive_gaps else 0.0
            )

        return BenchmarkAnalysis(
            percentiles=percentiles,
            industry_standards=industry_standards,
            performance_gaps=performance_gaps,
            improvement_potential=improvement_potential,
        )

    def _group_reports_by_project(
        self, reports: List[QualityReport], project_mapping: Optional[Dict[str, str]]
    ) -> Dict[str, List[QualityReport]]:
        """Group reports by project."""
        project_reports = {}

        for report in reports:
            if project_mapping and report.udl_file in project_mapping:
                project = project_mapping[report.udl_file]
            else:
                # Infer project from file path (use first directory component)
                path_parts = report.udl_file.split("/")
                project = path_parts[0] if len(path_parts) > 1 else "default"

            if project not in project_reports:
                project_reports[project] = []
            project_reports[project].append(report)

        return project_reports

    def _create_project_profile(
        self, project_name: str, reports: List[QualityReport]
    ) -> ProjectProfile:
        """Create a comprehensive profile for a project."""
        # Basic statistics
        overall_scores = [r.overall_score for r in reports]
        confidences = [r.confidence for r in reports]

        avg_overall_score = np.mean(overall_scores)
        std_overall_score = np.std(overall_scores)
        avg_confidence = np.mean(confidences)

        # Collect all unique UDL files
        udl_files = list(set(r.udl_file for r in reports))

        # Compute metric averages
        all_metrics = set()
        for report in reports:
            all_metrics.update(report.metric_scores.keys())

        metric_averages = {}
        for metric in all_metrics:
            values = [r.metric_scores.get(metric, np.nan) for r in reports]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                metric_averages[metric] = np.mean(valid_values)

        # Compute quality trend (if timestamps are available)
        quality_trend = 0.0
        if len(reports) > 1:
            # Sort by timestamp
            sorted_reports = sorted(reports, key=lambda r: r.timestamp)
            timestamps = [
                (r.timestamp - sorted_reports[0].timestamp).total_seconds()
                for r in sorted_reports
            ]
            scores = [r.overall_score for r in sorted_reports]

            if len(set(timestamps)) > 1:  # Avoid division by zero
                slope, _, _, _, _ = stats.linregress(timestamps, scores)
                quality_trend = slope

        # Compute consistency score (1 - coefficient of variation)
        cv = std_overall_score / avg_overall_score if avg_overall_score > 0 else 1.0
        consistency_score = max(0.0, 1.0 - cv)

        # Determine maturity level
        maturity_level = self._assess_maturity_level(
            avg_overall_score, consistency_score, quality_trend, len(reports)
        )

        # Determine risk level
        risk_level = self._assess_risk_level(
            avg_overall_score,
            std_overall_score,
            quality_trend,
            len([r for r in reports if r.errors]),
        )

        return ProjectProfile(
            project_name=project_name,
            udl_files=udl_files,
            total_reports=len(reports),
            avg_overall_score=avg_overall_score,
            std_overall_score=std_overall_score,
            avg_confidence=avg_confidence,
            metric_averages=metric_averages,
            quality_trend=quality_trend,
            consistency_score=consistency_score,
            maturity_level=maturity_level,
            risk_level=risk_level,
        )

    def _assess_maturity_level(
        self, avg_score: float, consistency: float, trend: float, num_reports: int
    ) -> str:
        """Assess project maturity level."""
        # Scoring system for maturity
        maturity_score = 0

        # High average quality
        if avg_score > 0.8:
            maturity_score += 3
        elif avg_score > 0.6:
            maturity_score += 2
        elif avg_score > 0.4:
            maturity_score += 1

        # High consistency
        if consistency > 0.8:
            maturity_score += 2
        elif consistency > 0.6:
            maturity_score += 1

        # Positive or stable trend
        if trend > 0.001:  # Improving
            maturity_score += 2
        elif trend > -0.001:  # Stable
            maturity_score += 1

        # Sufficient data points
        if num_reports > 20:
            maturity_score += 2
        elif num_reports > 10:
            maturity_score += 1

        # Classify based on score
        if maturity_score >= 7:
            return "mature"
        elif maturity_score >= 4:
            return "stable"
        else:
            return "developing"

    def _assess_risk_level(
        self, avg_score: float, std_score: float, trend: float, error_count: int
    ) -> str:
        """Assess project risk level."""
        risk_score = 0

        # Low average quality increases risk
        if avg_score < 0.4:
            risk_score += 3
        elif avg_score < 0.6:
            risk_score += 2
        elif avg_score < 0.8:
            risk_score += 1

        # High volatility increases risk
        if std_score > 0.2:
            risk_score += 2
        elif std_score > 0.1:
            risk_score += 1

        # Negative trend increases risk
        if trend < -0.001:
            risk_score += 2

        # Errors increase risk
        if error_count > 5:
            risk_score += 2
        elif error_count > 0:
            risk_score += 1

        # Classify based on score
        if risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"

    def _generate_rankings(
        self, profiles: Dict[str, ProjectProfile]
    ) -> Dict[str, List[str]]:
        """Generate rankings for different metrics."""
        rankings = {}

        # Overall score ranking
        rankings["overall_score"] = sorted(
            profiles.keys(), key=lambda p: profiles[p].avg_overall_score, reverse=True
        )

        # Confidence ranking
        rankings["confidence"] = sorted(
            profiles.keys(), key=lambda p: profiles[p].avg_confidence, reverse=True
        )

        # Consistency ranking
        rankings["consistency"] = sorted(
            profiles.keys(), key=lambda p: profiles[p].consistency_score, reverse=True
        )

        # Quality trend ranking
        rankings["improvement"] = sorted(
            profiles.keys(), key=lambda p: profiles[p].quality_trend, reverse=True
        )

        # Individual metric rankings
        all_metrics = set()
        for profile in profiles.values():
            all_metrics.update(profile.metric_averages.keys())

        for metric in all_metrics:
            # Only rank projects that have this metric
            projects_with_metric = [
                p for p in profiles.keys() if metric in profiles[p].metric_averages
            ]

            if len(projects_with_metric) > 1:
                rankings[f"metric_{metric}"] = sorted(
                    projects_with_metric,
                    key=lambda p: profiles[p].metric_averages[metric],
                    reverse=True,
                )

        return rankings

    def _perform_cluster_analysis(
        self, profiles: Dict[str, ProjectProfile]
    ) -> Dict[str, List[str]]:
        """Perform cluster analysis on projects."""
        if len(profiles) < 2:
            return {"cluster_1": list(profiles.keys())}

        # Create feature matrix
        projects = list(profiles.keys())
        features = []

        for project in projects:
            profile = profiles[project]
            feature_vector = [
                profile.avg_overall_score,
                profile.avg_confidence,
                profile.consistency_score,
                profile.quality_trend,
                len(profile.udl_files),
                profile.total_reports,
            ]
            features.append(feature_vector)

        features = np.array(features)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform clustering
        if self.clustering_method == "kmeans":
            n_clusters = min(self.n_clusters, len(projects))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
        else:  # hierarchical
            linkage_matrix = linkage(features_scaled, method="ward")
            cluster_labels = fcluster(
                linkage_matrix, self.n_clusters, criterion="maxclust"
            )
            cluster_labels = cluster_labels - 1  # Convert to 0-based indexing

        # Group projects by cluster
        clusters = {}
        for project, label in zip(projects, cluster_labels):
            cluster_name = f"cluster_{label + 1}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(project)

        return clusters

    def _compute_project_correlations(
        self, profiles: Dict[str, ProjectProfile]
    ) -> pd.DataFrame:
        """Compute correlations between project characteristics."""
        if len(profiles) < 3:
            return pd.DataFrame()

        # Create feature matrix
        projects = list(profiles.keys())
        feature_names = [
            "avg_overall_score",
            "avg_confidence",
            "consistency_score",
            "quality_trend",
            "num_udl_files",
            "total_reports",
        ]

        data = []
        for project in projects:
            profile = profiles[project]
            row = [
                profile.avg_overall_score,
                profile.avg_confidence,
                profile.consistency_score,
                profile.quality_trend,
                len(profile.udl_files),
                profile.total_reports,
            ]
            data.append(row)

        # Create DataFrame and compute correlations
        df = pd.DataFrame(data, index=projects, columns=feature_names)
        correlations = df.corr()

        return correlations

    def _identify_outliers(self, profiles: Dict[str, ProjectProfile]) -> List[str]:
        """Identify outlier projects using statistical methods."""
        if len(profiles) < 3:
            return []

        # Extract key metrics
        overall_scores = [p.avg_overall_score for p in profiles.values()]
        consistency_scores = [p.consistency_score for p in profiles.values()]
        trends = [p.quality_trend for p in profiles.values()]

        outliers = set()
        projects = list(profiles.keys())

        # Z-score based outlier detection
        for values, metric_name in [
            (overall_scores, "overall"),
            (consistency_scores, "consistency"),
            (trends, "trend"),
        ]:
            if np.std(values) > 0:
                z_scores = np.abs(stats.zscore(values))
                outlier_indices = np.where(z_scores > 2.5)[
                    0]  # 2.5 sigma threshold
                for idx in outlier_indices:
                    outliers.add(projects[idx])

        return list(outliers)

    def _generate_recommendations(
        self,
        profiles: Dict[str, ProjectProfile],
        clusters: Dict[str, List[str]],
        rankings: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Generate recommendations for each project."""
        recommendations = {}

        for project, profile in profiles.items():
            project_recommendations = []

            # Quality-based recommendations
            if profile.avg_overall_score < 0.5:
                project_recommendations.append(
                    "Focus on improving overall UDL quality - consider code reviews and quality guidelines"
                )
            elif profile.avg_overall_score < 0.7:
                project_recommendations.append(
                    "Good quality foundation - focus on consistency and best practices"
                )
            else:
                project_recommendations.append(
                    "Excellent quality - consider sharing best practices with other projects"
                )

            # Consistency recommendations
            if profile.consistency_score < 0.6:
                project_recommendations.append(
                    "High variability in quality - implement standardized development processes"
                )

            # Trend recommendations
            if profile.quality_trend < -0.001:
                project_recommendations.append(
                    "Quality is declining - investigate recent changes and implement corrective measures"
                )
            elif profile.quality_trend > 0.001:
                project_recommendations.append(
                    "Quality is improving - continue current practices and document successful approaches"
                )

            # Risk-based recommendations
            if profile.risk_level == "high":
                project_recommendations.append(
                    "High risk project - requires immediate attention and monitoring"
                )
            elif profile.risk_level == "medium":
                project_recommendations.append(
                    "Medium risk - implement additional quality controls"
                )

            # Maturity-based recommendations
            if profile.maturity_level == "developing":
                project_recommendations.append(
                    "Developing project - establish quality baselines and monitoring"
                )
            elif profile.maturity_level == "stable":
                project_recommendations.append(
                    "Stable project - focus on continuous improvement"
                )
            else:
                project_recommendations.append(
                    "Mature project - mentor other projects and share expertise"
                )

            # Benchmarking recommendations
            overall_ranking = rankings["overall_score"].index(project) + 1
            total_projects = len(rankings["overall_score"])

            if overall_ranking > total_projects * 0.75:
                project_recommendations.append(
                    "Below average performance - study top-performing projects for improvement ideas"
                )
            elif overall_ranking <= total_projects * 0.25:
                project_recommendations.append(
                    "Top performer - document and share successful practices"
                )

            # Cluster-based recommendations
            for cluster_name, cluster_projects in clusters.items():
                if project in cluster_projects and len(cluster_projects) > 1:
                    other_projects = [
                        p for p in cluster_projects if p != project]
                    project_recommendations.append(
                        f"Similar to projects: {', '.join(other_projects)} - consider collaboration and knowledge sharing"
                    )
                    break

            recommendations[project] = project_recommendations

        return recommendations

    def _compute_percentiles(
        self, profiles: Dict[str, ProjectProfile]
    ) -> Dict[str, Dict[str, float]]:
        """Compute percentiles for key metrics across the portfolio."""
        percentiles = {}

        # Overall score percentiles
        overall_scores = [p.avg_overall_score for p in profiles.values()]
        percentiles["overall_score"] = {
            25: np.percentile(overall_scores, 25),
            50: np.percentile(overall_scores, 50),
            75: np.percentile(overall_scores, 75),
            90: np.percentile(overall_scores, 90),
        }

        # Confidence percentiles
        confidences = [p.avg_confidence for p in profiles.values()]
        percentiles["confidence"] = {
            25: np.percentile(confidences, 25),
            50: np.percentile(confidences, 50),
            75: np.percentile(confidences, 75),
            90: np.percentile(confidences, 90),
        }

        # Consistency percentiles
        consistency_scores = [p.consistency_score for p in profiles.values()]
        percentiles["consistency_score"] = {
            25: np.percentile(consistency_scores, 25),
            50: np.percentile(consistency_scores, 50),
            75: np.percentile(consistency_scores, 75),
            90: np.percentile(consistency_scores, 90),
        }

        return percentiles

    def generate_portfolio_report(
        self,
        portfolio_comparison: PortfolioComparison,
        benchmark_analysis: Optional[BenchmarkAnalysis] = None,
    ) -> str:
        """Generate comprehensive portfolio analysis report."""
        report_lines = [
            "# Portfolio Analysis Report",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Projects Analyzed:** {len(portfolio_comparison.project_profiles)}",
            "",
            "## Executive Summary",
            "",
        ]

        # Portfolio overview
        profiles = portfolio_comparison.project_profiles
        avg_quality = np.mean([p.avg_overall_score for p in profiles.values()])
        avg_confidence = np.mean([p.avg_confidence for p in profiles.values()])

        report_lines.extend(
            [
                f"**Portfolio Average Quality:** {avg_quality:.3f}",
                f"**Portfolio Average Confidence:** {avg_confidence:.3f}",
                f"**Total UDL Files:** {sum(len(p.udl_files) for p in profiles.values())}",
                f"**Total Reports:** {sum(p.total_reports for p in profiles.values())}",
                "",
            ]
        )

        # Top performers
        top_projects = portfolio_comparison.rankings["overall_score"][:3]
        report_lines.extend(["**Top Performing Projects:**"])
        for i, project in enumerate(top_projects, 1):
            score = profiles[project].avg_overall_score
            report_lines.append(f"{i}. {project} (Score: {score:.3f})")
        report_lines.append("")

        # Project profiles
        report_lines.extend(["## Project Profiles", ""])

        for project, profile in profiles.items():
            report_lines.extend(
                [
                    f"### {project}",
                    "",
                    f"- **Overall Score:** {profile.avg_overall_score:.3f} ± {profile.std_overall_score:.3f}",
                    f"- **Confidence:** {profile.avg_confidence:.3f}",
                    f"- **Consistency:** {profile.consistency_score:.3f}",
                    f"- **Quality Trend:** {profile.quality_trend:.6f}",
                    f"- **Maturity Level:** {profile.maturity_level.title()}",
                    f"- **Risk Level:** {profile.risk_level.title()}",
                    f"- **UDL Files:** {len(profile.udl_files)}",
                    f"- **Total Reports:** {profile.total_reports}",
                    "",
                ]
            )

        # Rankings
        report_lines.extend(["## Rankings", ""])

        for metric, ranking in portfolio_comparison.rankings.items():
            if not metric.startswith("metric_"):
                report_lines.extend(
                    [f"### {metric.replace('_', ' ').title()}", ""])
                for i, project in enumerate(ranking, 1):
                    report_lines.append(f"{i}. {project}")
                report_lines.append("")

        # Clusters
        report_lines.extend(["## Project Clusters", ""])

        for cluster_name, cluster_projects in portfolio_comparison.clusters.items():
            report_lines.extend(
                [f"### {cluster_name.replace('_', ' ').title()}", ""])
            for project in cluster_projects:
                profile = profiles[project]
                report_lines.append(
                    f"- {project} (Quality: {profile.avg_overall_score:.3f}, Maturity: {profile.maturity_level})"
                )
            report_lines.append("")

        # Recommendations
        report_lines.extend(["## Recommendations", ""])

        for project, recommendations in portfolio_comparison.recommendations.items():
            report_lines.extend([f"### {project}", ""])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")

        # Benchmark analysis (if provided)
        if benchmark_analysis:
            report_lines.extend(
                ["## Benchmark Analysis", "", "### Industry Standards Comparison", ""]
            )

            for project, gaps in benchmark_analysis.performance_gaps.items():
                improvement_potential = benchmark_analysis.improvement_potential[
                    project
                ]
                report_lines.extend(
                    [
                        f"**{project}:**",
                        f"- Improvement Potential: {improvement_potential:.3f}",
                        "- Key Gaps:",
                    ]
                )

                for metric, gap in gaps.items():
                    if gap > 0.01:  # Only show significant gaps
                        report_lines.append(
                            f"  - {metric}: {gap:.3f} below standard")

                report_lines.append("")

        return "\n".join(report_lines)
