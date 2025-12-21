"""
CLI commands for advanced analytics functionality.

Provides command-line interface for time series analysis, portfolio analysis,
trend prediction, improvement advisory, and business intelligence export.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click

from udl_rating_framework.analytics import (
    BusinessIntelligenceExporter,
    ImprovementAdvisor,
    PortfolioAnalyzer,
    TimeSeriesAnalyzer,
    TrendPredictor,
)
from udl_rating_framework.analytics.bi_exporter import BIExportConfig
from udl_rating_framework.core.pipeline import QualityReport
from udl_rating_framework.io.report_generator import ReportGenerator


def load_reports_from_directory(reports_dir: Path) -> List[QualityReport]:
    """Load quality reports from JSON files in directory."""
    reports = []

    for json_file in reports_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Handle both single reports and report collections
            if isinstance(data, dict) and "reports" in data:
                report_data_list = data["reports"]
            elif isinstance(data, list):
                report_data_list = data
            else:
                report_data_list = [data]

            for report_data in report_data_list:
                # Convert timestamp string to datetime
                if isinstance(report_data.get("timestamp"), str):
                    report_data["timestamp"] = datetime.fromisoformat(
                        report_data["timestamp"]
                    )

                # Create QualityReport object
                report = QualityReport(
                    overall_score=report_data.get("overall_score", 0.0),
                    confidence=report_data.get("confidence", 0.0),
                    metric_scores=report_data.get("metric_scores", {}),
                    metric_formulas=report_data.get("metric_formulas", {}),
                    computation_trace=report_data.get("computation_trace", []),
                    error_bounds=report_data.get("error_bounds", {}),
                    timestamp=report_data.get("timestamp", datetime.now()),
                    udl_file=report_data.get("udl_file", ""),
                    errors=report_data.get("errors", []),
                    warnings=report_data.get("warnings", []),
                )
                reports.append(report)

        except Exception as e:
            click.echo(
                f"Warning: Could not load report from {json_file}: {e}", err=True
            )
            continue

    return reports


@click.group()
def analytics():
    """Advanced analytics commands for UDL quality data."""
    pass


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option(
    "--udl-file",
    "-f",
    required=True,
    help="UDL file to analyze (e.g., project1/grammar.udl)",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Specific metrics to analyze (default: all available)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for the analysis report",
)
@click.option(
    "--min-observations", default=10, help="Minimum number of observations required"
)
def timeseries(
    reports_dir: Path,
    udl_file: str,
    metrics: tuple,
    output: Optional[Path],
    min_observations: int,
):
    """Analyze quality evolution over time for a specific UDL file."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Filter reports for the specific UDL file
    file_reports = [r for r in reports if r.udl_file == udl_file]

    if len(file_reports) < min_observations:
        click.echo(
            f"Error: Insufficient data for {udl_file}. "
            f"Found {len(file_reports)} reports, need at least {min_observations}.",
            err=True,
        )
        return

    click.echo(
        f"Analyzing time series for {udl_file} with {len(file_reports)} reports..."
    )

    analyzer = TimeSeriesAnalyzer(min_observations=min_observations)

    try:
        if metrics:
            # Analyze specific metrics
            results = {}
            for metric in metrics:
                try:
                    evolution = analyzer.analyze_quality_evolution(
                        reports, udl_file, metric
                    )
                    results[metric] = evolution
                    click.echo(f"✓ Analyzed {metric}")
                except Exception as e:
                    click.echo(f"✗ Could not analyze {metric}: {e}", err=True)
        else:
            # Analyze all available metrics
            results = analyzer.analyze_multiple_metrics(reports, udl_file)
            click.echo(f"✓ Analyzed {len(results)} metrics")

        if not results:
            click.echo("Error: No metrics could be analyzed.", err=True)
            return

        # Generate report
        report = analyzer.generate_time_series_report(results, udl_file)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo("\n" + "=" * 80)
            click.echo(report)

    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option(
    "--project-mapping",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file mapping UDL files to projects",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for the portfolio analysis report",
)
@click.option(
    "--min-reports", default=5, help="Minimum reports per project for inclusion"
)
@click.option(
    "--clustering-method",
    type=click.Choice(["kmeans", "hierarchical"]),
    default="kmeans",
    help="Clustering method to use",
)
@click.option("--n-clusters", default=3, help="Number of clusters for analysis")
def portfolio(
    reports_dir: Path,
    project_mapping: Optional[Path],
    output: Optional[Path],
    min_reports: int,
    clustering_method: str,
    n_clusters: int,
):
    """Analyze quality across a portfolio of projects."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Load project mapping if provided
    mapping = None
    if project_mapping:
        try:
            with open(project_mapping, "r") as f:
                mapping = json.load(f)
            click.echo(f"Loaded project mapping with {len(mapping)} entries")
        except Exception as e:
            click.echo(
                f"Warning: Could not load project mapping: {e}", err=True)

    click.echo(f"Analyzing portfolio with {len(reports)} reports...")

    analyzer = PortfolioAnalyzer(
        min_reports_per_project=min_reports,
        clustering_method=clustering_method,
        n_clusters=n_clusters,
    )

    try:
        # Perform portfolio analysis
        comparison = analyzer.analyze_portfolio(reports, mapping)
        click.echo(f"✓ Analyzed {len(comparison.project_profiles)} projects")

        # Perform benchmark analysis
        benchmark = analyzer.benchmark_against_standards(comparison)
        click.echo("✓ Completed benchmark analysis")

        # Generate report
        report = analyzer.generate_portfolio_report(comparison, benchmark)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo("\n" + "=" * 80)
            click.echo(report)

        # Display summary
        click.echo(f"\nPortfolio Summary:")
        click.echo(f"- Projects analyzed: {len(comparison.project_profiles)}")
        click.echo(f"- Clusters identified: {len(comparison.clusters)}")
        click.echo(f"- Outlier projects: {len(comparison.outliers)}")

        if comparison.outliers:
            click.echo(f"- Outliers: {', '.join(comparison.outliers)}")

    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option("--udl-file", "-f", required=True, help="UDL file to predict trends for")
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Specific metrics to predict (default: overall_score, confidence)",
)
@click.option("--horizon", default=30, help="Prediction horizon in days")
@click.option(
    "--models",
    multiple=True,
    type=click.Choice(["linear", "polynomial", "random_forest"]),
    help="Models to use for prediction (default: all)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for the prediction report",
)
def predict(
    reports_dir: Path,
    udl_file: str,
    metrics: tuple,
    horizon: int,
    models: tuple,
    output: Optional[Path],
):
    """Predict quality trends for a specific UDL file."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Filter reports for the specific UDL file
    file_reports = [r for r in reports if r.udl_file == udl_file]

    if len(file_reports) < 10:
        click.echo(
            f"Error: Insufficient data for {udl_file}. "
            f"Found {len(file_reports)} reports, need at least 10.",
            err=True,
        )
        return

    click.echo(
        f"Predicting trends for {udl_file} with {len(file_reports)} reports...")

    predictor_models = (
        list(models) if models else ["linear", "polynomial", "random_forest"]
    )
    predictor = TrendPredictor(
        prediction_horizon=horizon, models=predictor_models)

    try:
        # Predict trends
        prediction_metrics = (
            list(metrics) if metrics else ["overall_score", "confidence"]
        )
        predictions = predictor.predict_quality_trends(
            reports, udl_file, prediction_metrics
        )

        click.echo(f"✓ Generated predictions for {len(predictions)} metrics")

        # Generate report
        report = predictor.generate_prediction_report(predictions, udl_file)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo("\n" + "=" * 80)
            click.echo(report)

        # Display summary
        for metric, result in predictions.items():
            current_value = (
                result.predictions.iloc[0] if len(
                    result.predictions) > 0 else 0.0
            )
            future_value = (
                result.predictions.iloc[-1] if len(
                    result.predictions) > 0 else 0.0
            )
            trend = (
                "improving"
                if future_value > current_value
                else "declining"
                if future_value < current_value
                else "stable"
            )

            click.echo(
                f"- {metric}: {trend} (R² = {result.model_performance.get('r2', 0.0):.3f})"
            )

    except Exception as e:
        click.echo(f"Error during prediction: {e}", err=True)


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option(
    "--udl-file", "-f", required=True, help="UDL file to generate improvement plan for"
)
@click.option(
    "--target-score", type=float, help="Target quality score (default: current + 0.2)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for the improvement plan",
)
def improve(
    reports_dir: Path,
    udl_file: str,
    target_score: Optional[float],
    output: Optional[Path],
):
    """Generate improvement plan for a specific UDL file."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Filter reports for the specific UDL file
    file_reports = [r for r in reports if r.udl_file == udl_file]

    if not file_reports:
        click.echo(f"Error: No reports found for {udl_file}.", err=True)
        return

    click.echo(
        f"Generating improvement plan for {udl_file} with {len(file_reports)} reports..."
    )

    advisor = ImprovementAdvisor()

    try:
        # Generate improvement plan
        plan = advisor.generate_improvement_plan(
            file_reports, udl_file, target_score)

        click.echo(
            f"✓ Generated plan with {len(plan.suggestions)} recommendations")

        # Generate report
        report = advisor.generate_improvement_report(plan)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(report)
            click.echo(f"Improvement plan saved to {output}")
        else:
            click.echo("\n" + "=" * 80)
            click.echo(report)

        # Display summary
        click.echo(f"\nImprovement Plan Summary:")
        click.echo(f"- Current score: {plan.current_score:.3f}")
        click.echo(f"- Target score: {plan.target_score:.3f}")
        click.echo(f"- Recommendations: {len(plan.suggestions)}")
        click.echo(f"- Estimated timeline: {plan.estimated_timeline}")

        high_priority = [s for s in plan.suggestions if s.priority == "high"]
        if high_priority:
            click.echo(f"- High priority items: {len(high_priority)}")

    except Exception as e:
        click.echo(f"Error generating improvement plan: {e}", err=True)


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for BI exports",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "json", "excel", "parquet", "xml"]),
    default="csv",
    help="Export format",
)
@click.option(
    "--aggregation",
    type=click.Choice(["detailed", "summary", "executive"]),
    default="detailed",
    help="Aggregation level",
)
@click.option(
    "--time-grouping",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="daily",
    help="Time grouping for summary aggregation",
)
@click.option(
    "--include-trends/--no-trends",
    default=True,
    help="Include trend analysis in export",
)
@click.option(
    "--include-forecasts/--no-forecasts",
    default=False,
    help="Include forecast data in export",
)
@click.option(
    "--project-mapping",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file mapping UDL files to projects",
)
def export(
    reports_dir: Path,
    output_dir: Path,
    format: str,
    aggregation: str,
    time_grouping: str,
    include_trends: bool,
    include_forecasts: bool,
    project_mapping: Optional[Path],
):
    """Export quality data for business intelligence tools."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Load project mapping if provided
    mapping = None
    if project_mapping:
        try:
            with open(project_mapping, "r") as f:
                mapping = json.load(f)
            click.echo(f"Loaded project mapping with {len(mapping)} entries")
        except Exception as e:
            click.echo(
                f"Warning: Could not load project mapping: {e}", err=True)

    click.echo(f"Exporting {len(reports)} reports in {format} format...")

    exporter = BusinessIntelligenceExporter()
    config = BIExportConfig(
        format=format,
        aggregation_level=aggregation,
        time_grouping=time_grouping,
        include_trends=include_trends,
        include_forecasts=include_forecasts,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"udl_quality_{aggregation}.{format}"

        # Export data
        dataset = exporter.export_quality_data(
            reports, config, output_file, mapping)

        click.echo(f"✓ Exported {len(dataset.data)} records to {output_file}")

        # Export metadata
        metadata_file = output_dir / "export_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset.metadata, f, indent=2, default=str)

        click.echo(f"✓ Saved metadata to {metadata_file}")

        # Display summary
        click.echo(f"\nExport Summary:")
        click.echo(f"- Format: {format}")
        click.echo(f"- Aggregation: {aggregation}")
        click.echo(f"- Records: {len(dataset.data)}")
        click.echo(
            f"- Columns: {len(dataset.data.columns) if not dataset.data.empty else 0}"
        )

    except Exception as e:
        click.echo(f"Error during export: {e}", err=True)


@analytics.command()
@click.option(
    "--reports-dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing quality report JSON files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for dashboard packages",
)
@click.option(
    "--platform",
    type=click.Choice(["powerbi", "tableau"]),
    required=True,
    help="BI platform to create package for",
)
@click.option(
    "--project-mapping",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file mapping UDL files to projects",
)
def dashboard(
    reports_dir: Path, output_dir: Path, platform: str, project_mapping: Optional[Path]
):
    """Create dashboard packages for BI platforms."""
    click.echo(f"Loading reports from {reports_dir}...")
    reports = load_reports_from_directory(reports_dir)

    if not reports:
        click.echo(
            "Error: No reports found in the specified directory.", err=True)
        return

    # Load project mapping if provided
    mapping = None
    if project_mapping:
        try:
            with open(project_mapping, "r") as f:
                mapping = json.load(f)
            click.echo(f"Loaded project mapping with {len(mapping)} entries")
        except Exception as e:
            click.echo(
                f"Warning: Could not load project mapping: {e}", err=True)

    click.echo(f"Creating {platform} dashboard package...")

    exporter = BusinessIntelligenceExporter()

    try:
        if platform == "powerbi":
            package_path = exporter.create_powerbi_package(
                reports, output_dir, mapping)
        else:  # tableau
            package_path = exporter.create_tableau_package(
                reports, output_dir, mapping)

        click.echo(f"✓ Created {platform} package at {package_path}")

        # List package contents
        click.echo(f"\nPackage Contents:")
        for item in package_path.iterdir():
            if item.is_file():
                click.echo(f"- {item.name}")

        click.echo(f"\nTo use this package:")
        if platform == "powerbi":
            click.echo("1. Import the CSV files as data sources in Power BI")
            click.echo("2. Use the template and model files as guides")
            click.echo(
                "3. Create visualizations based on the provided structure")
        else:
            click.echo("1. Open Tableau Desktop")
            click.echo("2. Connect to the data source using the TDS file")
            click.echo("3. Open the workbook template and customize as needed")

    except Exception as e:
        click.echo(f"Error creating dashboard package: {e}", err=True)


if __name__ == "__main__":
    analytics()
