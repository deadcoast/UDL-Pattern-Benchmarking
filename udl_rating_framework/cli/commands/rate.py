"""
Rate command for UDL Rating Framework CLI.

Provides functionality to rate UDL files or directories.
"""

import click
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sys

from udl_rating_framework.core.pipeline import RatingPipeline, QualityReport
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.io.report_generator import ReportGenerator

# Import metrics to trigger registration
import udl_rating_framework.core.metrics

logger = logging.getLogger(__name__)


@click.command("rate")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
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
    "--include-traces", is_flag=True, help="Include computation traces in output"
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
def rate_command(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    format: str,
    recursive: bool,
    extensions: str,
    include_traces: bool,
    consistency_weight: float,
    completeness_weight: float,
    expressiveness_weight: float,
    structural_coherence_weight: float,
):
    """
    Rate UDL files or directories using mathematical quality metrics.

    INPUT_PATH can be a single UDL file or a directory containing UDL files.
    When processing directories, use --recursive to include subdirectories.

    Examples:

        # Rate a single file
        udl-rating rate my_language.udl

        # Rate all UDL files in a directory
        udl-rating rate ./udl_files/ --recursive

        # Rate with custom weights and save to file
        udl-rating rate ./udl_files/ --consistency-weight 0.4 --output results.json

        # Generate HTML report
        udl-rating rate ./udl_files/ --format html --output report.html
    """
    try:
        # Validate weights
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
                output = output_dir / f"rating_results.{format}"
            include_traces = output_config.get("include_traces", include_traces)

        logger.info(f"Rating UDL files in: {input_path}")
        logger.info(f"Output format: {format}")
        logger.info(
            f"Metric weights: consistency={consistency_weight}, "
            f"completeness={completeness_weight}, "
            f"expressiveness={expressiveness_weight}, "
            f"structural_coherence={structural_coherence_weight}"
        )

        # Discover files
        if input_path.is_file():
            udl_files = [input_path]
        else:
            file_discovery = FileDiscovery(supported_extensions=set(ext_list))
            discovery_result = file_discovery.discover_files(str(input_path))
            udl_files = discovery_result.discovered_files

            # Log any discovery errors
            for error in discovery_result.errors:
                logger.warning(f"File discovery error: {error}")

        if not udl_files:
            logger.warning(f"No UDL files found in {input_path}")
            return

        logger.info(f"Found {len(udl_files)} UDL files to process")

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

        # Process files
        reports = []
        for udl_file in udl_files:
            try:
                logger.info(f"Processing: {udl_file}")

                # Read and parse UDL file
                with open(udl_file, "r", encoding="utf-8") as f:
                    content = f.read()

                from udl_rating_framework.core.representation import UDLRepresentation

                udl_repr = UDLRepresentation(content, str(udl_file))

                # Compute rating
                report = pipeline.compute_rating(udl_repr)
                reports.append(report)
                logger.info(
                    f"Completed: {udl_file} (score: {report.overall_score:.4f})"
                )
            except Exception as e:
                logger.error(f"Failed to process {udl_file}: {e}")
                continue

        if not reports:
            logger.error("No files were successfully processed")
            sys.exit(1)

        # Generate output
        report_generator = ReportGenerator()

        if format == "json":
            report_data = report_generator.generate_json_report(reports)
        elif format == "csv":
            report_data = report_generator.generate_csv_report(
                reports, include_trace=include_traces
            )
        elif format == "html":
            report_data = report_generator.generate_html_report(reports)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Output results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(report_data)
            logger.info(f"Results saved to: {output}")
        else:
            # Output to stdout
            click.echo(report_data)

        # Summary statistics
        if len(reports) > 1:
            scores = [r.overall_score for r in reports]
            logger.info(f"Summary: {len(reports)} files processed")
            logger.info(f"Average score: {sum(scores) / len(scores):.4f}")
            logger.info(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

    except Exception as e:
        logger.error(f"Rating failed: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)
