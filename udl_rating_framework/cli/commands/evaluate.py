"""
Evaluate command for UDL Rating Framework CLI.

Provides functionality to evaluate trained CTM models using comprehensive metrics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import torch

# Import metrics to trigger registration
import udl_rating_framework.core.metrics
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.evaluation.evaluation_suite import EvaluationSuite
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary
from udl_rating_framework.training.training_pipeline import UDLDataset

logger = logging.getLogger(__name__)


@click.command("evaluate")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("test_data_path", type=click.Path(exists=True, path_type=Path))
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
@click.option(
    "--k-folds", type=int, default=5, help="Number of folds for cross-validation"
)
@click.option(
    "--bootstrap-samples",
    type=int,
    default=1000,
    help="Number of bootstrap samples for confidence intervals",
)
@click.option(
    "--confidence-level",
    type=float,
    default=0.95,
    help="Confidence level for intervals",
)
@click.option(
    "--calibration-bins",
    type=int,
    default=10,
    help="Number of bins for calibration analysis",
)
@click.option(
    "--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)"
)
@click.option("--batch-size", type=int, default=32, help="Batch size for evaluation")
@click.option(
    "--include-detailed-analysis",
    is_flag=True,
    help="Include detailed error analysis and visualizations",
)
@click.pass_context
def evaluate_command(
    ctx: click.Context,
    model_path: Path,
    test_data_path: Path,
    output: Optional[Path],
    format: str,
    k_folds: int,
    bootstrap_samples: int,
    confidence_level: float,
    calibration_bins: int,
    device: str,
    batch_size: int,
    include_detailed_analysis: bool,
):
    """
    Evaluate a trained CTM model using comprehensive statistical metrics.

    MODEL_PATH should point to a saved model checkpoint (.pt file).
    TEST_DATA_PATH should contain UDL files for evaluation.

    The evaluation includes:
    - K-fold cross-validation
    - Correlation analysis (Pearson, Spearman) with confidence intervals
    - Calibration error computation
    - Error distribution analysis
    - Bootstrap confidence intervals

    Examples:

        # Basic evaluation
        udl-rating evaluate ./checkpoints/final_model.pt ./test_data/

        # Detailed evaluation with custom parameters
        udl-rating evaluate ./model.pt ./test_data/ --k-folds 10 --bootstrap-samples 2000

        # Generate HTML report
        udl-rating evaluate ./model.pt ./test_data/ --format html --output evaluation.html
    """
    try:
        # Validate parameters
        if k_folds < 5:
            raise click.BadParameter(
                f"k_folds must be at least 5, got {k_folds}")

        if bootstrap_samples < 1000:
            raise click.BadParameter(
                f"bootstrap_samples must be at least 1000, got {bootstrap_samples}"
            )

        if not 0 < confidence_level < 1:
            raise click.BadParameter(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        # Override with config if available
        config = ctx.obj.get("config", {})
        if "evaluation" in config:
            eval_config = config["evaluation"]
            k_folds = eval_config.get("k_folds", k_folds)
            bootstrap_samples = eval_config.get(
                "bootstrap_samples", bootstrap_samples)
            confidence_level = eval_config.get(
                "confidence_level", confidence_level)
            calibration_bins = eval_config.get(
                "calibration_bins", calibration_bins)

        if "output" in config:
            output_config = config["output"]
            if not output:
                output_dir = Path(output_config.get("output_dir", "output"))
                output_dir.mkdir(exist_ok=True)
                output = output_dir / f"evaluation_results.{format}"

        # Set up device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Evaluating model: {model_path}")
        logger.info(f"Test data: {test_data_path}")
        logger.info(f"Device: {device}")
        logger.info(
            f"Evaluation parameters: k_folds={k_folds}, bootstrap_samples={bootstrap_samples}"
        )
        logger.info(f"Confidence level: {confidence_level}")

        # Load model
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location=device)

        model_config = checkpoint.get("config", {})
        vocab = checkpoint.get("vocab")

        if vocab is None:
            logger.error("Model checkpoint does not contain vocabulary")
            sys.exit(1)

        # Create model
        model = UDLRatingCTM(
            vocab_size=len(vocab),
            d_model=model_config.get("d_model", 256),
            d_input=model_config.get("d_input", 64),
            iterations=model_config.get("iterations", 20),
            n_synch_out=32,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully (vocab size: {len(vocab)})")

        # Discover test files
        file_discovery = FileDiscovery()
        discovery_result = file_discovery.discover_files(str(test_data_path))
        test_files = discovery_result.discovered_files

        # Log any discovery errors
        for error in discovery_result.errors:
            logger.warning(f"File discovery error: {error}")

        if not test_files:
            logger.error(f"No UDL files found in {test_data_path}")
            sys.exit(1)

        logger.info(f"Found {len(test_files)} test files")

        # Load and parse test files
        logger.info("Loading and parsing test files...")
        test_representations = []
        for test_file in test_files:
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                udl_repr = UDLRepresentation(content, str(test_file))
                test_representations.append(udl_repr)
            except Exception as e:
                logger.warning(f"Failed to parse {test_file}: {e}")
                continue

        if not test_representations:
            logger.error("No test files could be parsed successfully")
            sys.exit(1)

        logger.info(
            f"Successfully parsed {len(test_representations)} test files")

        # Create test dataset
        max_length = model_config.get("max_sequence_length", 512)
        test_dataset = UDLDataset(test_representations, vocab, max_length)

        # Set up ground truth metrics
        metric_registry = MetricRegistry()
        available_metrics = metric_registry.list_metrics()

        # Use equal weights (can be overridden by config)
        weights_dict = {
            "consistency": 0.25,
            "completeness": 0.25,
            "expressiveness": 0.25,
            "structural_coherence": 0.25,
        }

        if "metrics" in config:
            metrics_config = config["metrics"]
            weights_dict.update(
                {
                    "consistency": metrics_config.get("consistency_weight", 0.25),
                    "completeness": metrics_config.get("completeness_weight", 0.25),
                    "expressiveness": metrics_config.get("expressiveness_weight", 0.25),
                    "structural_coherence": metrics_config.get(
                        "structural_coherence_weight", 0.25
                    ),
                }
            )

        aggregator = MetricAggregator(weights_dict)
        metric_names = list(weights_dict.keys())
        metrics = [metric_registry.get_metric(name)() for name in metric_names]

        # Create evaluation suite
        evaluation_suite = EvaluationSuite(
            k_folds=k_folds,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            calibration_bins=calibration_bins,
        )

        # Run evaluation
        logger.info("Running comprehensive evaluation...")

        def model_predict_fn(udl_representations):
            """Prediction function for evaluation."""
            predictions = []
            confidences = []

            with torch.no_grad():
                for udl_repr in udl_representations:
                    # Tokenize
                    tokens = udl_repr.get_tokens()
                    token_ids = [vocab.get_token_id(
                        token.text) for token in tokens]

                    # Pad/truncate
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                    else:
                        token_ids.extend(
                            [vocab.get_token_id("<PAD>")]
                            * (max_length - len(token_ids))
                        )

                    # Convert to tensor
                    input_tensor = torch.tensor([token_ids], device=device)

                    # Forward pass
                    rating, certainty = model(input_tensor)

                    predictions.append(float(rating.cpu().item()))
                    confidences.append(
                        float(torch.softmax(certainty, dim=-1).max().cpu().item())
                    )

            return predictions, confidences

        def ground_truth_fn(udl_representations):
            """Ground truth computation function."""
            ground_truths = []
            for udl_repr in udl_representations:
                # Compute ground truth using mathematical metrics
                metric_values = {}
                for metric in metrics:
                    try:
                        value = metric.compute(udl_repr)
                        metric_values[metric.__class__.__name__] = value
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute {metric.__class__.__name__} for {udl_repr.file_path}: {e}"
                        )
                        metric_values[metric.__class__.__name__] = 0.0

                ground_truth = aggregator.aggregate(metric_values)
                ground_truths.append(ground_truth)

            return ground_truths

        # Run evaluation
        evaluation_result = evaluation_suite.evaluate_model(
            test_representations, model_predict_fn, ground_truth_fn
        )

        # Prepare output data
        output_data = {
            "model_info": {
                "model_path": str(model_path),
                "vocab_size": len(vocab),
                "model_config": model_config,
                "test_files_count": len(test_representations),
            },
            "evaluation_parameters": {
                "k_folds": k_folds,
                "bootstrap_samples": bootstrap_samples,
                "confidence_level": confidence_level,
                "calibration_bins": calibration_bins,
                "metric_weights": weights_dict,
            },
            "results": {
                "pearson_correlation": evaluation_result.pearson_correlation,
                "pearson_ci": evaluation_result.pearson_ci,
                "spearman_correlation": evaluation_result.spearman_correlation,
                "spearman_ci": evaluation_result.spearman_ci,
                "calibration_error": evaluation_result.calibration_error,
                "shapiro_statistic": evaluation_result.shapiro_statistic,
                "shapiro_p_value": evaluation_result.shapiro_p_value,
                "bootstrap_metrics": evaluation_result.bootstrap_metrics,
                "cv_scores": evaluation_result.cv_scores,
                "mean_cv_score": evaluation_result.mean_cv_score,
                "std_cv_score": evaluation_result.std_cv_score,
            },
        }

        # Add detailed analysis if requested
        if include_detailed_analysis:
            logger.info("Generating detailed analysis...")
            # Add more detailed metrics here
            output_data["detailed_analysis"] = {
                "error_distribution_normal": evaluation_result.shapiro_p_value > 0.05,
                "model_well_calibrated": evaluation_result.calibration_error < 0.1,
                "correlation_strength": (
                    "strong"
                    if evaluation_result.pearson_correlation > 0.7
                    else (
                        "moderate"
                        if evaluation_result.pearson_correlation > 0.5
                        else "weak"
                    )
                ),
            }

        # Generate output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            if format == "json":
                with open(output, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
            elif format == "csv":
                # For CSV, flatten the results
                import pandas as pd

                flat_data = {
                    "pearson_correlation": evaluation_result.pearson_correlation,
                    "pearson_ci_lower": evaluation_result.pearson_ci[0],
                    "pearson_ci_upper": evaluation_result.pearson_ci[1],
                    "spearman_correlation": evaluation_result.spearman_correlation,
                    "spearman_ci_lower": evaluation_result.spearman_ci[0],
                    "spearman_ci_upper": evaluation_result.spearman_ci[1],
                    "calibration_error": evaluation_result.calibration_error,
                    "mean_cv_score": evaluation_result.mean_cv_score,
                    "std_cv_score": evaluation_result.std_cv_score,
                    "shapiro_statistic": evaluation_result.shapiro_statistic,
                    "shapiro_p_value": evaluation_result.shapiro_p_value,
                }
                df = pd.DataFrame([flat_data])
                df.to_csv(output, index=False)
            elif format == "html":
                # Generate HTML report
                from udl_rating_framework.io.report_generator import ReportGenerator

                report_generator = ReportGenerator()
                html_content = report_generator.generate_evaluation_report(
                    output_data, format="html"
                )
                with open(output, "w") as f:
                    f.write(html_content)

            logger.info(f"Evaluation results saved to: {output}")
        else:
            # Output to stdout
            if format == "json":
                click.echo(json.dumps(output_data, indent=2, default=str))
            else:
                click.echo(str(output_data))

        # Print summary
        logger.info("Evaluation completed!")
        logger.info(
            f"Pearson correlation: {evaluation_result.pearson_correlation:.4f} "
            f"(95% CI: {evaluation_result.pearson_ci[0]:.4f}-{evaluation_result.pearson_ci[1]:.4f})"
        )
        logger.info(
            f"Spearman correlation: {evaluation_result.spearman_correlation:.4f} "
            f"(95% CI: {evaluation_result.spearman_ci[0]:.4f}-{evaluation_result.spearman_ci[1]:.4f})"
        )
        logger.info(
            f"Calibration error: {evaluation_result.calibration_error:.4f}")
        logger.info(
            f"Cross-validation score: {evaluation_result.mean_cv_score:.4f} ± {evaluation_result.std_cv_score:.4f}"
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)
