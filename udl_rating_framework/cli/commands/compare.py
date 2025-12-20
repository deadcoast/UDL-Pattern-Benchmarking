"""
Compare command for UDL Rating Framework CLI.

Provides functionality to compare multiple UDL files using statistical tests.
"""

import click
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sys

from udl_rating_framework.evaluation.comparison import ComparisonEngine
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.io.report_generator import ReportGenerator

# Import metrics to trigger registration
import udl_rating_framework.core.metrics

logger = logging.getLogger(__name__)


@click.command("compare")
@click.argument(
    "input_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: stdout)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv", "html"]),
    default="json",
    help="Output format",
)
@click.option("--recursive", "-r", is_flag=True, help="Recursively process directories")
@click.option(
    "--extensions",
    "-e",
    default=".udl,.dsl,.grammar,.ebnf,.txt",
    help="Comma-separated list of file extensions to process",
)
@click.option(
    "--significance-level",
    type=float,
    default=0.05,
    help="Significance level for statistical tests",
)
@click.option(
    "--include-effect-sizes", is_flag=True, help="Include effect size computations"
)
@click.option(
    "--ranking", is_flag=True, help="Generate ranking with confidence intervals"
)
@click.option(
    "--consistency-weight",
    type=float,
    default=0.25,
    help="Weight for consistency metric",
)
@click.option(
    "--completeness-weight",
    type=float,
    default=0.25,
    help="Weight for completeness metric",
)
@click.option(
    "--expressiveness-weight",
    type=float,
    default=0.25,
    help="Weight for expressiveness metric",
)
@click.option(
    "--structural-coherence-weight",
    type=float,
    default=0.25,
    help="Weight for structural coherence metric",
)
@click.pass_context
def compare_command(
    ctx: click.Context,
    input_paths: tuple,
    output: Optional[Path],
    format: str,
    recursive: bool,
    extensions: str,
    significance_level: float,
    include_effect_sizes: bool,
    ranking: bool,
    consistency_weight: float,
    completeness_weight: float,
    expressiveness_weight: float,
    structural_coherence_weight: float,
):
    """
    Compare multiple UDL files using statistical tests and effect size analysis.

    INPUT_PATHS can be multiple files or directories. When comparing directories,
    all UDL files in each directory will be compared as groups.

    Examples:

        # Compare two UDL files
        udl-rating compare lang1.udl lang2.udl

        # Compare all files in two directories
        udl-rating compare ./lang1_versions/ ./lang2_versions/ --recursive

        # Generate ranking with confidence intervals
        udl-rating compare *.udl --ranking --output comparison.json

        # Compare with custom significance level
        udl-rating compare lang1.udl lang2.udl --significance-level 0.01
    """
    try:
        # Validate parameters
        if not 0 < significance_level < 1:
            raise click.BadParameter(
                f"significance_level must be between 0 and 1, got {significance_level}"
            )

        weights = [
            consistency_weight,
            completeness_weight,
            expressiveness_weight,
            structural_coherence_weight,
        ]
        if abs(sum(weights) - 1.0) > 1e-6:
            raise click.BadParameter(
                f"Metric weights must sum to 1.0, got {sum(weights)}"
            )

        if any(w < 0 for w in weights):
            raise click.BadParameter("All metric weights must be non-negative")

        # Parse extensions
        ext_list = [ext.strip() for ext in extensions.split(",")]

        # Override with config if available
        config = ctx.obj.get("config", {})
        if "metrics" in config:
            metrics_config = config["metrics"]
            consistency_weight = metrics_config.get(
                "consistency_weight", consistency_weight
            )
            completeness_weight = metrics_config.get(
                "completeness_weight", completeness_weight
            )
            expressiveness_weight = metrics_config.get(
                "expressiveness_weight", expressiveness_weight
            )
            structural_coherence_weight = metrics_config.get(
                "structural_coherence_weight", structural_coherence_weight
            )

        if "output" in config:
            output_config = config["output"]
            if not output:
                output_dir = Path(output_config.get("output_dir", "output"))
                output_dir.mkdir(exist_ok=True)
                output = output_dir / f"comparison_results.{format}"

        logger.info(f"Comparing UDL files from {len(input_paths)} input paths")
        logger.info(f"Significance level: {significance_level}")
        logger.info(f"Include effect sizes: {include_effect_sizes}")
        logger.info(f"Generate ranking: {ranking}")

        # Discover and collect all files
        all_files = []
        file_groups = {}  # Group files by input path for group comparisons

        for i, input_path in enumerate(input_paths):
            if input_path.is_file():
                files = [input_path]
                group_name = input_path.stem
            else:
                file_discovery = FileDiscovery(supported_extensions=set(ext_list))
                discovery_result = file_discovery.discover_files(str(input_path))
                files = discovery_result.discovered_files
                group_name = input_path.name

                # Log any discovery errors
                for error in discovery_result.errors:
                    logger.warning(f"File discovery error: {error}")

            if not files:
                logger.warning(f"No UDL files found in {input_path}")
                continue

            all_files.extend(files)
            file_groups[group_name] = files
            logger.info(f"Group '{group_name}': {len(files)} files")

        if len(all_files) < 2:
            logger.error("Need at least 2 UDL files for comparison")
            sys.exit(1)

        logger.info(f"Total files to compare: {len(all_files)}")

        # Set up metrics and aggregator
        metric_registry = MetricRegistry()
        available_metrics = metric_registry.list_metrics()

        # Map CLI weight names to registry names
        metric_name_mapping = {
            "ConsistencyMetric": "consistency",
            "CompletenessMetric": "completeness",
            "ExpressivenessMetric": "expressiveness",
            "StructuralCoherenceMetric": "structural_coherence",
        }

        weights_dict = {
            metric_name_mapping["ConsistencyMetric"]: consistency_weight,
            metric_name_mapping["CompletenessMetric"]: completeness_weight,
            metric_name_mapping["ExpressivenessMetric"]: expressiveness_weight,
            metric_name_mapping[
                "StructuralCoherenceMetric"
            ]: structural_coherence_weight,
        }

        # Create pipeline with metric names
        metric_names = list(weights_dict.keys())
        pipeline = RatingPipeline(metric_names, weights_dict)

        logger.info("Rating all files...")
        reports = []
        for udl_file in all_files:
            try:
                logger.debug(f"Processing: {udl_file}")

                # Read and parse UDL file
                with open(udl_file, "r", encoding="utf-8") as f:
                    content = f.read()

                from udl_rating_framework.core.representation import UDLRepresentation

                udl_repr = UDLRepresentation(content, str(udl_file))

                # Compute rating
                report = pipeline.compute_rating(udl_repr)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to process {udl_file}: {e}")
                continue

        if len(reports) < 2:
            logger.error("Need at least 2 successfully processed files for comparison")
            sys.exit(1)

        logger.info(f"Successfully rated {len(reports)} files")

        # Create comparison engine
        comparison_engine = ComparisonEngine(alpha=significance_level)

        # Perform comparisons
        logger.info("Performing statistical comparisons...")

        # Perform comparisons using the main compare_udls method
        logger.info("Performing statistical comparisons...")
        comparison_summary = comparison_engine.compare_udls(reports)

        comparison_results = [
            {
                "type": "statistical_comparison",
                "results": {
                    "pairwise_comparisons": comparison_summary.pairwise_results,
                    "rankings": comparison_summary.rankings,
                    "total_comparisons": comparison_summary.total_comparisons,
                    "significant_comparisons": comparison_summary.significant_comparisons,
                    "mean_effect_size": comparison_summary.mean_effect_size,
                    "score_statistics": comparison_summary.score_statistics,
                },
            }
        ]

        # Generate summary statistics
        scores = [r.overall_score for r in reports]
        summary_stats = {
            "total_files": len(reports),
            "mean_score": float(sum(scores) / len(scores)),
            "std_score": float(
                (
                    sum((s - sum(scores) / len(scores)) ** 2 for s in scores)
                    / len(scores)
                )
                ** 0.5
            ),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "score_range": float(max(scores) - min(scores)),
        }

        # Prepare output data
        output_data = {
            "summary": summary_stats,
            "comparisons": comparison_results,
            "individual_reports": [
                {
                    "file": r.udl_file,
                    "score": r.overall_score,
                    "confidence": r.confidence,
                    "metrics": r.metric_scores,
                }
                for r in reports
            ],
            "parameters": {
                "significance_level": significance_level,
                "metric_weights": weights_dict,
                "include_effect_sizes": include_effect_sizes,
                "ranking_generated": ranking,
            },
        }

        # Generate output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            if format == "json":
                with open(output, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
            elif format == "csv":
                # For CSV, flatten the data
                import pandas as pd

                df_data = []
                for report in reports:
                    row = {
                        "file": report.udl_file,
                        "overall_score": report.overall_score,
                        "confidence": report.confidence,
                        **report.metric_scores,
                    }
                    df_data.append(row)
                df = pd.DataFrame(df_data)
                df.to_csv(output, index=False)
            elif format == "html":
                # Generate HTML report
                report_generator = ReportGenerator()
                html_content = report_generator.generate_comparison_report(
                    output_data, format="html"
                )
                with open(output, "w") as f:
                    f.write(html_content)

            logger.info(f"Comparison results saved to: {output}")
        else:
            # Output to stdout
            if format == "json":
                click.echo(json.dumps(output_data, indent=2, default=str))
            else:
                click.echo(str(output_data))

        # Print summary
        logger.info(f"Comparison completed!")
        logger.info(f"Files compared: {len(reports)}")
        logger.info(
            f"Average score: {summary_stats['mean_score']:.4f} ± {summary_stats['std_score']:.4f}"
        )
        logger.info(
            f"Score range: {summary_stats['min_score']:.4f} - {summary_stats['max_score']:.4f}"
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)
