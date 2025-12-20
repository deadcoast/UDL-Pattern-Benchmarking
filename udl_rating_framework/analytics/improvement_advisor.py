"""
Automated quality improvement suggestions.

This module provides intelligent recommendations for improving UDL quality
based on analysis of quality metrics, patterns, and best practices.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

from udl_rating_framework.core.pipeline import QualityReport


@dataclass
class ImprovementSuggestion:
    """A specific improvement suggestion."""

    category: str  # 'consistency', 'completeness', 'expressiveness', 'structure'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    rationale: str
    expected_impact: float  # Expected improvement in overall score
    effort_level: str  # 'low', 'medium', 'high'
    specific_actions: List[str]
    related_metrics: List[str]


@dataclass
class ImprovementPlan:
    """Comprehensive improvement plan."""

    udl_file: str
    current_score: float
    target_score: float
    suggestions: List[ImprovementSuggestion]
    implementation_order: List[str]  # Suggestion titles in recommended order
    estimated_timeline: str
    success_metrics: List[str]


class ImprovementAdvisor:
    """
    Provides automated quality improvement suggestions.

    Analyzes UDL quality reports and generates specific, actionable
    recommendations for improving quality across all metrics.
    """

    def __init__(self):
        """Initialize improvement advisor."""
        self.best_practices = self._load_best_practices()
        self.common_issues = self._load_common_issues()
        self.improvement_patterns = self._load_improvement_patterns()

    def generate_improvement_plan(
        self,
        reports: List[QualityReport],
        udl_file: str,
        target_score: Optional[float] = None,
    ) -> ImprovementPlan:
        """
        Generate comprehensive improvement plan for a UDL file.

        Args:
            reports: Historical quality reports for the UDL file
            udl_file: UDL file to generate plan for
            target_score: Target quality score (default: current + 0.2)

        Returns:
            ImprovementPlan with specific recommendations
        """
        # Filter reports for the specific UDL file
        file_reports = [r for r in reports if r.udl_file == udl_file]

        if not file_reports:
            raise ValueError(f"No reports found for UDL file: {udl_file}")

        # Get the most recent report
        latest_report = max(file_reports, key=lambda r: r.timestamp)
        current_score = latest_report.overall_score

        # Set target score
        if target_score is None:
            target_score = min(1.0, current_score + 0.2)

        # Analyze current state
        analysis = self._analyze_current_state(latest_report, file_reports)

        # Generate suggestions
        suggestions = self._generate_suggestions(analysis, latest_report)

        # Prioritize and order suggestions
        implementation_order = self._prioritize_suggestions(
            suggestions, current_score, target_score
        )

        # Estimate timeline
        timeline = self._estimate_timeline(suggestions)

        # Define success metrics
        success_metrics = self._define_success_metrics(latest_report, suggestions)

        return ImprovementPlan(
            udl_file=udl_file,
            current_score=current_score,
            target_score=target_score,
            suggestions=suggestions,
            implementation_order=implementation_order,
            estimated_timeline=timeline,
            success_metrics=success_metrics,
        )

    def analyze_portfolio_improvements(
        self,
        reports: List[QualityReport],
        project_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Analyze improvement opportunities across portfolio.

        Args:
            reports: All quality reports
            project_mapping: Optional mapping from UDL file to project

        Returns:
            Dict mapping improvement categories to recommended actions
        """
        # Group reports by project
        if project_mapping:
            project_reports = defaultdict(list)
            for report in reports:
                project = project_mapping.get(report.udl_file, "unknown")
                project_reports[project].append(report)
        else:
            project_reports = defaultdict(list)
            for report in reports:
                project = (
                    report.udl_file.split("/")[0]
                    if "/" in report.udl_file
                    else "default"
                )
                project_reports[project].append(report)

        # Analyze common issues across portfolio
        common_issues = defaultdict(int)
        low_performing_metrics = defaultdict(list)

        for project, project_reports_list in project_reports.items():
            if not project_reports_list:
                continue

            latest_report = max(project_reports_list, key=lambda r: r.timestamp)

            # Identify low-performing metrics
            for metric, score in latest_report.metric_scores.items():
                if score < 0.6:  # Below acceptable threshold
                    low_performing_metrics[metric].append(project)

            # Count common issues
            if latest_report.overall_score < 0.5:
                common_issues["low_overall_quality"] += 1
            if latest_report.confidence < 0.7:
                common_issues["low_confidence"] += 1
            if len(latest_report.errors) > 0:
                common_issues["has_errors"] += 1
            if len(latest_report.warnings) > 2:
                common_issues["many_warnings"] += 1

        # Generate portfolio-wide recommendations
        portfolio_recommendations = {}

        # Quality standardization
        if common_issues["low_overall_quality"] > len(project_reports) * 0.3:
            portfolio_recommendations["Quality Standardization"] = [
                "Establish organization-wide UDL quality standards",
                "Implement mandatory quality reviews for all UDL changes",
                "Create UDL style guides and best practice documentation",
                "Set up automated quality gates in CI/CD pipelines",
            ]

        # Training and knowledge sharing
        if len(low_performing_metrics) > 3:
            portfolio_recommendations["Training & Knowledge Sharing"] = [
                "Conduct UDL quality training sessions",
                "Create internal knowledge base with examples",
                "Establish mentorship program pairing high and low performers",
                "Regular quality review meetings and retrospectives",
            ]

        # Tooling and automation
        if common_issues["has_errors"] > len(project_reports) * 0.2:
            portfolio_recommendations["Tooling & Automation"] = [
                "Implement automated UDL validation tools",
                "Set up continuous quality monitoring",
                "Create IDE plugins for real-time quality feedback",
                "Establish quality dashboards for visibility",
            ]

        # Process improvements
        if common_issues["many_warnings"] > len(project_reports) * 0.4:
            portfolio_recommendations["Process Improvements"] = [
                "Implement quality-first development processes",
                "Establish clear quality criteria for UDL acceptance",
                "Create quality checklists for developers",
                "Regular quality audits and assessments",
            ]

        return portfolio_recommendations

    def _analyze_current_state(
        self, latest_report: QualityReport, historical_reports: List[QualityReport]
    ) -> Dict[str, Any]:
        """Analyze current state of UDL quality."""
        analysis = {
            "current_scores": latest_report.metric_scores.copy(),
            "overall_score": latest_report.overall_score,
            "confidence": latest_report.confidence,
            "errors": latest_report.errors,
            "warnings": latest_report.warnings,
            "trends": {},
            "weak_areas": [],
            "strong_areas": [],
            "volatility": {},
        }

        # Add overall score to metrics for analysis
        analysis["current_scores"]["overall_score"] = latest_report.overall_score
        analysis["current_scores"]["confidence"] = latest_report.confidence

        # Analyze trends if we have historical data
        if len(historical_reports) > 1:
            sorted_reports = sorted(historical_reports, key=lambda r: r.timestamp)

            for metric in analysis["current_scores"]:
                values = []
                for report in sorted_reports:
                    if metric == "overall_score":
                        values.append(report.overall_score)
                    elif metric == "confidence":
                        values.append(report.confidence)
                    else:
                        values.append(report.metric_scores.get(metric, np.nan))

                # Remove NaN values
                valid_values = [v for v in values if not np.isnan(v)]

                if len(valid_values) > 1:
                    # Simple trend analysis
                    trend = (valid_values[-1] - valid_values[0]) / len(valid_values)
                    analysis["trends"][metric] = trend

                    # Volatility (coefficient of variation)
                    if np.mean(valid_values) > 0:
                        cv = np.std(valid_values) / np.mean(valid_values)
                        analysis["volatility"][metric] = cv

        # Identify weak and strong areas
        for metric, score in analysis["current_scores"].items():
            if score < 0.5:
                analysis["weak_areas"].append(metric)
            elif score > 0.8:
                analysis["strong_areas"].append(metric)

        return analysis

    def _generate_suggestions(
        self, analysis: Dict[str, Any], latest_report: QualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate specific improvement suggestions."""
        suggestions = []

        # Consistency improvements
        consistency_score = analysis["current_scores"].get("ConsistencyMetric", 0.0)
        if consistency_score < 0.7:
            suggestions.extend(
                self._generate_consistency_suggestions(consistency_score, latest_report)
            )

        # Completeness improvements
        completeness_score = analysis["current_scores"].get("CompletenessMetric", 0.0)
        if completeness_score < 0.7:
            suggestions.extend(
                self._generate_completeness_suggestions(
                    completeness_score, latest_report
                )
            )

        # Expressiveness improvements
        expressiveness_score = analysis["current_scores"].get(
            "ExpressivenessMetric", 0.0
        )
        if expressiveness_score < 0.7:
            suggestions.extend(
                self._generate_expressiveness_suggestions(
                    expressiveness_score, latest_report
                )
            )

        # Structural coherence improvements
        structure_score = analysis["current_scores"].get(
            "StructuralCoherenceMetric", 0.0
        )
        if structure_score < 0.7:
            suggestions.extend(
                self._generate_structure_suggestions(structure_score, latest_report)
            )

        # Error-based suggestions
        if latest_report.errors:
            suggestions.extend(
                self._generate_error_based_suggestions(latest_report.errors)
            )

        # Warning-based suggestions
        if latest_report.warnings:
            suggestions.extend(
                self._generate_warning_based_suggestions(latest_report.warnings)
            )

        # Trend-based suggestions
        if "trends" in analysis:
            suggestions.extend(
                self._generate_trend_based_suggestions(analysis["trends"])
            )

        # Confidence-based suggestions
        if analysis["confidence"] < 0.7:
            suggestions.extend(
                self._generate_confidence_suggestions(analysis["confidence"])
            )

        return suggestions

    def _generate_consistency_suggestions(
        self, score: float, report: QualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate consistency improvement suggestions."""
        suggestions = []

        if score < 0.3:
            priority = "high"
            impact = 0.3
        elif score < 0.5:
            priority = "medium"
            impact = 0.2
        else:
            priority = "low"
            impact = 0.1

        suggestions.append(
            ImprovementSuggestion(
                category="consistency",
                priority=priority,
                title="Resolve Rule Contradictions",
                description="Eliminate contradictory rules that create logical inconsistencies in the UDL.",
                rationale=f"Current consistency score is {score:.3f}, indicating significant rule conflicts.",
                expected_impact=impact,
                effort_level="medium",
                specific_actions=[
                    "Review all grammar rules for logical contradictions",
                    "Use formal verification tools to detect conflicts",
                    "Refactor conflicting rules into consistent alternatives",
                    "Add unit tests to prevent future contradictions",
                ],
                related_metrics=["ConsistencyMetric", "overall_score"],
            )
        )

        if score < 0.5:
            suggestions.append(
                ImprovementSuggestion(
                    category="consistency",
                    priority="medium",
                    title="Eliminate Circular Dependencies",
                    description="Remove circular references in grammar rules that create infinite loops.",
                    rationale="Circular dependencies reduce consistency and can cause parsing issues.",
                    expected_impact=0.15,
                    effort_level="medium",
                    specific_actions=[
                        "Map all rule dependencies",
                        "Identify and break circular references",
                        "Restructure grammar hierarchy",
                        "Implement dependency validation checks",
                    ],
                    related_metrics=["ConsistencyMetric"],
                )
            )

        return suggestions

    def _generate_completeness_suggestions(
        self, score: float, report: QualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate completeness improvement suggestions."""
        suggestions = []

        if score < 0.3:
            priority = "high"
            impact = 0.4
        elif score < 0.5:
            priority = "medium"
            impact = 0.25
        else:
            priority = "low"
            impact = 0.15

        suggestions.append(
            ImprovementSuggestion(
                category="completeness",
                priority=priority,
                title="Add Missing Language Constructs",
                description="Implement missing essential language constructs to improve completeness.",
                rationale=f"Current completeness score is {score:.3f}, indicating missing required constructs.",
                expected_impact=impact,
                effort_level="high",
                specific_actions=[
                    "Identify missing constructs based on language type",
                    "Prioritize constructs by importance and usage frequency",
                    "Implement missing constructs with proper syntax",
                    "Add comprehensive test cases for new constructs",
                ],
                related_metrics=["CompletenessMetric", "overall_score"],
            )
        )

        if score < 0.6:
            suggestions.append(
                ImprovementSuggestion(
                    category="completeness",
                    priority="medium",
                    title="Enhance Error Handling Constructs",
                    description="Add comprehensive error handling and exception constructs.",
                    rationale="Robust error handling improves language completeness and usability.",
                    expected_impact=0.1,
                    effort_level="medium",
                    specific_actions=[
                        "Define error handling syntax",
                        "Implement exception throwing and catching",
                        "Add error recovery mechanisms",
                        "Create error reporting constructs",
                    ],
                    related_metrics=["CompletenessMetric"],
                )
            )

        return suggestions

    def _generate_expressiveness_suggestions(
        self, score: float, report: QualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate expressiveness improvement suggestions."""
        suggestions = []

        if score < 0.4:
            priority = "medium"
            impact = 0.2
        else:
            priority = "low"
            impact = 0.1

        suggestions.append(
            ImprovementSuggestion(
                category="expressiveness",
                priority=priority,
                title="Enhance Language Power",
                description="Increase the expressiveness of the language by adding more powerful constructs.",
                rationale=f"Current expressiveness score is {score:.3f}, suggesting limited language power.",
                expected_impact=impact,
                effort_level="high",
                specific_actions=[
                    "Move up in Chomsky hierarchy if possible",
                    "Add higher-order constructs (functions, closures)",
                    "Implement advanced control flow structures",
                    "Add metaprogramming capabilities",
                ],
                related_metrics=["ExpressivenessMetric", "overall_score"],
            )
        )

        if score < 0.6:
            suggestions.append(
                ImprovementSuggestion(
                    category="expressiveness",
                    title="Optimize Language Complexity",
                    priority="low",
                    description="Balance language complexity to improve expressiveness without sacrificing usability.",
                    rationale="Optimal complexity enhances expressiveness while maintaining readability.",
                    expected_impact=0.05,
                    effort_level="medium",
                    specific_actions=[
                        "Analyze current complexity metrics",
                        "Simplify overly complex constructs",
                        "Add syntactic sugar for common patterns",
                        "Optimize grammar for better compression",
                    ],
                    related_metrics=["ExpressivenessMetric"],
                )
            )

        return suggestions

    def _generate_structure_suggestions(
        self, score: float, report: QualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate structural coherence improvement suggestions."""
        suggestions = []

        if score < 0.5:
            priority = "medium"
            impact = 0.2
        else:
            priority = "low"
            impact = 0.1

        suggestions.append(
            ImprovementSuggestion(
                category="structure",
                priority=priority,
                title="Improve Grammar Organization",
                description="Reorganize grammar rules to improve structural coherence and reduce entropy.",
                rationale=f"Current structural coherence score is {score:.3f}, indicating poor organization.",
                expected_impact=impact,
                effort_level="medium",
                specific_actions=[
                    "Group related rules together",
                    "Create clear hierarchical structure",
                    "Eliminate redundant rules",
                    "Improve rule naming consistency",
                ],
                related_metrics=["StructuralCoherenceMetric", "overall_score"],
            )
        )

        if score < 0.6:
            suggestions.append(
                ImprovementSuggestion(
                    category="structure",
                    priority="low",
                    title="Enhance Modularity",
                    description="Improve grammar modularity by creating reusable rule components.",
                    rationale="Better modularity improves maintainability and structural coherence.",
                    expected_impact=0.1,
                    effort_level="medium",
                    specific_actions=[
                        "Identify common rule patterns",
                        "Extract reusable rule components",
                        "Create modular grammar sections",
                        "Implement proper rule interfaces",
                    ],
                    related_metrics=["StructuralCoherenceMetric"],
                )
            )

        return suggestions

    def _generate_error_based_suggestions(
        self, errors: List[str]
    ) -> List[ImprovementSuggestion]:
        """Generate suggestions based on reported errors."""
        suggestions = []

        if errors:
            suggestions.append(
                ImprovementSuggestion(
                    category="errors",
                    priority="high",
                    title="Fix Critical Errors",
                    description="Address all critical errors that prevent proper UDL processing.",
                    rationale=f"Found {len(errors)} critical errors that must be resolved.",
                    expected_impact=0.3,
                    effort_level="high",
                    specific_actions=[
                        "Review and fix each reported error",
                        "Test fixes thoroughly",
                        "Implement error prevention measures",
                        "Add validation to prevent similar errors",
                    ],
                    related_metrics=["overall_score", "confidence"],
                )
            )

        return suggestions

    def _generate_warning_based_suggestions(
        self, warnings: List[str]
    ) -> List[ImprovementSuggestion]:
        """Generate suggestions based on reported warnings."""
        suggestions = []

        if len(warnings) > 3:
            suggestions.append(
                ImprovementSuggestion(
                    category="warnings",
                    priority="medium",
                    title="Address Quality Warnings",
                    description="Resolve quality warnings to improve overall UDL quality.",
                    rationale=f"Found {len(warnings)} warnings that indicate potential quality issues.",
                    expected_impact=0.1,
                    effort_level="low",
                    specific_actions=[
                        "Review all warning messages",
                        "Fix issues causing warnings",
                        "Improve code quality practices",
                        "Set up warning monitoring",
                    ],
                    related_metrics=["overall_score"],
                )
            )

        return suggestions

    def _generate_trend_based_suggestions(
        self, trends: Dict[str, float]
    ) -> List[ImprovementSuggestion]:
        """Generate suggestions based on quality trends."""
        suggestions = []

        declining_metrics = [
            metric for metric, trend in trends.items() if trend < -0.01
        ]

        if declining_metrics:
            suggestions.append(
                ImprovementSuggestion(
                    category="trends",
                    priority="high",
                    title="Address Declining Quality Trends",
                    description="Take corrective action to reverse negative quality trends.",
                    rationale=f"Detected declining trends in: {', '.join(declining_metrics)}",
                    expected_impact=0.2,
                    effort_level="medium",
                    specific_actions=[
                        "Investigate causes of quality decline",
                        "Implement quality monitoring alerts",
                        "Review recent changes for quality impact",
                        "Establish quality improvement processes",
                    ],
                    related_metrics=declining_metrics,
                )
            )

        return suggestions

    def _generate_confidence_suggestions(
        self, confidence: float
    ) -> List[ImprovementSuggestion]:
        """Generate suggestions based on confidence scores."""
        suggestions = []

        if confidence < 0.5:
            priority = "high"
            impact = 0.2
        elif confidence < 0.7:
            priority = "medium"
            impact = 0.1
        else:
            return suggestions

        suggestions.append(
            ImprovementSuggestion(
                category="confidence",
                priority=priority,
                title="Improve Prediction Confidence",
                description="Take actions to improve the reliability and confidence of quality assessments.",
                rationale=f"Current confidence score is {confidence:.3f}, indicating uncertain quality assessments.",
                expected_impact=impact,
                effort_level="medium",
                specific_actions=[
                    "Increase data quality and consistency",
                    "Add more comprehensive test cases",
                    "Improve documentation and specifications",
                    "Reduce ambiguity in language definitions",
                ],
                related_metrics=["confidence"],
            )
        )

        return suggestions

    def _prioritize_suggestions(
        self,
        suggestions: List[ImprovementSuggestion],
        current_score: float,
        target_score: float,
    ) -> List[str]:
        """Prioritize suggestions based on impact and effort."""
        # Calculate priority scores
        priority_scores = {}

        for suggestion in suggestions:
            # Base score from priority level
            priority_weight = {"high": 3, "medium": 2, "low": 1}[suggestion.priority]

            # Impact weight
            impact_weight = suggestion.expected_impact * 10

            # Effort weight (inverse - lower effort is better)
            effort_weight = {"low": 3, "medium": 2, "high": 1}[suggestion.effort_level]

            # Combined score
            score = priority_weight * 0.4 + impact_weight * 0.4 + effort_weight * 0.2
            priority_scores[suggestion.title] = score

        # Sort by priority score (descending)
        return sorted(
            priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True
        )

    def _estimate_timeline(self, suggestions: List[ImprovementSuggestion]) -> str:
        """Estimate implementation timeline."""
        effort_days = {"low": 2, "medium": 5, "high": 10}

        total_days = sum(effort_days[s.effort_level] for s in suggestions)

        if total_days <= 7:
            return "1 week"
        elif total_days <= 14:
            return "2 weeks"
        elif total_days <= 30:
            return "1 month"
        elif total_days <= 60:
            return "2 months"
        else:
            return "3+ months"

    def _define_success_metrics(
        self, report: QualityReport, suggestions: List[ImprovementSuggestion]
    ) -> List[str]:
        """Define metrics to track improvement success."""
        metrics = ["overall_score"]  # Always track overall score

        # Add metrics mentioned in suggestions
        for suggestion in suggestions:
            metrics.extend(suggestion.related_metrics)

        # Add current weak metrics
        for metric, score in report.metric_scores.items():
            if score < 0.6:
                metrics.append(metric)

        # Remove duplicates and return
        return list(set(metrics))

    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Load best practices database."""
        return {
            "consistency": [
                "Use consistent naming conventions",
                "Avoid contradictory rules",
                "Implement proper rule hierarchies",
                "Use formal verification tools",
            ],
            "completeness": [
                "Include all essential language constructs",
                "Provide comprehensive error handling",
                "Add proper documentation",
                "Implement complete test coverage",
            ],
            "expressiveness": [
                "Balance power with simplicity",
                "Add higher-order constructs when appropriate",
                "Provide syntactic sugar for common patterns",
                "Consider metaprogramming capabilities",
            ],
            "structure": [
                "Organize rules logically",
                "Use modular design principles",
                "Minimize rule complexity",
                "Implement clear interfaces",
            ],
        }

    def _load_common_issues(self) -> Dict[str, List[str]]:
        """Load common issues database."""
        return {
            "consistency": [
                "Contradictory grammar rules",
                "Circular dependencies",
                "Inconsistent naming",
                "Ambiguous rule definitions",
            ],
            "completeness": [
                "Missing error handling",
                "Incomplete construct coverage",
                "Lack of standard library",
                "Missing documentation",
            ],
            "expressiveness": [
                "Limited language power",
                "Overly restrictive syntax",
                "Missing advanced constructs",
                "Poor abstraction capabilities",
            ],
            "structure": [
                "Poor rule organization",
                "High complexity",
                "Lack of modularity",
                "Unclear hierarchies",
            ],
        }

    def _load_improvement_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load improvement patterns and their typical impacts."""
        return {
            "fix_contradictions": {"consistency": 0.3, "overall": 0.2},
            "add_constructs": {"completeness": 0.4, "overall": 0.25},
            "improve_structure": {"structure": 0.2, "overall": 0.1},
            "enhance_expressiveness": {"expressiveness": 0.3, "overall": 0.15},
            "fix_errors": {"overall": 0.3, "confidence": 0.2},
            "resolve_warnings": {"overall": 0.1, "confidence": 0.1},
        }

    def generate_improvement_report(self, improvement_plan: ImprovementPlan) -> str:
        """Generate comprehensive improvement report."""
        report_lines = [
            "# UDL Quality Improvement Plan",
            f"**UDL File:** {improvement_plan.udl_file}",
            f"**Current Score:** {improvement_plan.current_score:.3f}",
            f"**Target Score:** {improvement_plan.target_score:.3f}",
            f"**Estimated Timeline:** {improvement_plan.estimated_timeline}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]

        # Summary
        total_impact = sum(s.expected_impact for s in improvement_plan.suggestions)
        high_priority = len(
            [s for s in improvement_plan.suggestions if s.priority == "high"]
        )

        report_lines.extend(
            [
                f"This improvement plan contains **{len(improvement_plan.suggestions)} recommendations** with an estimated total impact of **{total_impact:.3f}** points.",
                f"**{high_priority} high-priority** items require immediate attention.",
                f"Following this plan should improve the quality score from **{improvement_plan.current_score:.3f}** to approximately **{min(1.0, improvement_plan.current_score + total_impact):.3f}**.",
                "",
                "## Implementation Roadmap",
                "",
            ]
        )

        # Implementation order
        for i, title in enumerate(improvement_plan.implementation_order, 1):
            suggestion = next(
                s for s in improvement_plan.suggestions if s.title == title
            )
            report_lines.append(
                f"{i}. **{title}** ({suggestion.priority} priority, {suggestion.effort_level} effort)"
            )

        report_lines.extend(["", "## Detailed Recommendations", ""])

        # Group suggestions by category
        by_category = defaultdict(list)
        for suggestion in improvement_plan.suggestions:
            by_category[suggestion.category].append(suggestion)

        for category, category_suggestions in by_category.items():
            report_lines.extend([f"### {category.title()} Improvements", ""])

            for suggestion in category_suggestions:
                report_lines.extend(
                    [
                        f"#### {suggestion.title}",
                        f"**Priority:** {suggestion.priority.title()} | **Effort:** {suggestion.effort_level.title()} | **Expected Impact:** +{suggestion.expected_impact:.3f}",
                        "",
                        f"**Description:** {suggestion.description}",
                        "",
                        f"**Rationale:** {suggestion.rationale}",
                        "",
                        "**Specific Actions:**",
                    ]
                )

                for action in suggestion.specific_actions:
                    report_lines.append(f"- {action}")

                report_lines.extend(
                    [
                        "",
                        f"**Related Metrics:** {', '.join(suggestion.related_metrics)}",
                        "",
                    ]
                )

        # Success metrics
        report_lines.extend(
            [
                "## Success Metrics",
                "",
                "Track the following metrics to measure improvement success:",
                "",
            ]
        )

        for metric in improvement_plan.success_metrics:
            report_lines.append(f"- {metric}")

        report_lines.extend(
            [
                "",
                "## Next Steps",
                "",
                "1. Review and approve this improvement plan",
                "2. Assign resources and set timeline",
                "3. Begin implementation in the recommended order",
                "4. Monitor success metrics regularly",
                "5. Adjust plan based on progress and results",
                "",
                "---",
                "*This improvement plan was generated automatically based on quality analysis. Review recommendations carefully and adapt to your specific context.*",
            ]
        )

        return "\n".join(report_lines)
