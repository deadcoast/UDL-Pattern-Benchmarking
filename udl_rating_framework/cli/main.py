"""
Main CLI interface for UDL Rating Framework.

Provides command-line interface for rating UDL files, training models,
comparing UDLs, and evaluating model performance.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from udl_rating_framework.cli.commands.analytics import analytics
from udl_rating_framework.cli.commands.compare import compare_command
from udl_rating_framework.cli.commands.evaluate import evaluate_command
from udl_rating_framework.cli.commands.integration import integration
from udl_rating_framework.cli.commands.rate import rate_command
from udl_rating_framework.cli.commands.train import train_command
from udl_rating_framework.cli.config import load_config, validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path (YAML format)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, quiet: bool):
    """
    UDL Rating Framework - A mathematically-grounded system for evaluating
    User Defined Languages built on Continuous Thought Machine architecture.

    This tool provides commands for:
    - Rating UDL files and directories
    - Training CTM models
    - Comparing multiple UDLs
    - Evaluating model performance
    - Advanced analytics and reporting
    """
    # Set up logging level
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Load configuration
    ctx.ensure_object(dict)
    if config:
        try:
            ctx.obj["config"] = load_config(config)
            logger.info(f"Loaded configuration from {config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    else:
        ctx.obj["config"] = {}


# Add commands
cli.add_command(rate_command)
cli.add_command(train_command)
cli.add_command(compare_command)
cli.add_command(evaluate_command)
cli.add_command(integration)
cli.add_command(analytics)


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
