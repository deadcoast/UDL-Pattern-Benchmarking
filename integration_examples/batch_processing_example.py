#!/usr/bin/env python3
"""
Example script demonstrating batch processing for large-scale UDL analysis.

This script shows how to efficiently process large numbers of UDL files
with parallel processing, caching, and progress tracking.
"""

import time
from pathlib import Path

from udl_rating_framework.integration.batch_processor import BatchConfig, BatchProcessor


def create_sample_udl_files(output_dir: Path, count: int = 10):
    """Create sample UDL files for demonstration."""
    output_dir.mkdir(exist_ok=True)

    sample_udl_content = """
# Sample UDL Grammar
grammar SampleLanguage {
    
    # Tokens
    token IDENTIFIER = /[a-zA-Z_][a-zA-Z0-9_]*/
    token NUMBER = /[0-9]+/
    token STRING = /"[^"]*"/
    
    # Rules
    rule program = statement*
    rule statement = assignment | expression
    rule assignment = IDENTIFIER "=" expression
    rule expression = term (("+" | "-") term)*
    rule term = factor (("*" | "/") factor)*
    rule factor = NUMBER | STRING | IDENTIFIER | "(" expression ")"
}
"""

    for i in range(count):
        file_path = output_dir / f"sample_{i:03d}.udl"
        # Add some variation to make files different
        content = sample_udl_content + f"\n# File {i}\n"
        if i % 3 == 0:
            content += 'rule extra_rule = "optional" IDENTIFIER\n'
        file_path.write_text(content)

    print(f"Created {count} sample UDL files in {output_dir}")


def main():
    """Demonstrate batch processing."""
    print("UDL Rating Framework - Batch Processing Example")
    print("=" * 60)

    # Create sample files
    sample_dir = Path("sample_udl_files")
    create_sample_udl_files(sample_dir, count=20)

    # Create batch configuration
    config = BatchConfig(
        max_workers=4,
        chunk_size=5,
        timeout_per_file=10.0,
        memory_limit_mb=512,
        enable_caching=True,
        cache_dir=Path("udl_cache"),
        error_handling="continue",
        max_retries=2,
        output_format="json",
        include_detailed_metrics=True,
        generate_summary=True,
    )

    print("Batch Processing Configuration:")
    print(f"  Max workers: {config.max_workers}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Timeout per file: {config.timeout_per_file}s")
    print(f"  Caching enabled: {config.enable_caching}")
    print(f"  Cache directory: {config.cache_dir}")
    print(f"  Output format: {config.output_format}")
    print()

    # Progress callback
    def progress_callback(processed: int, total: int):
        percentage = processed / total * 100 if total > 0 else 0
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")

    config.progress_callback = progress_callback

    # Create batch processor
    processor = BatchProcessor(config)

    # Process directory
    print("Processing UDL files in directory...")
    start_time = time.time()

    result = processor.process_directory(
        sample_dir, patterns=["*.udl"], exclude_patterns=None
    )

    processing_time = time.time() - start_time

    # Display results
    print(f"\nBatch Processing Results:")
    print(f"  Total files: {result.total_files}")
    print(f"  Processed files: {result.processed_files}")
    print(f"  Failed files: {result.failed_files}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Average quality: {result.average_quality:.3f}")
    print(
        f"  Processing rate: {result.summary_stats['processing_rate']:.1f} files/sec")
    print(f"  Success rate: {result.summary_stats['success_rate']:.1%}")

    # Quality distribution
    print(f"\nQuality Distribution:")
    for level, count in result.quality_distribution.items():
        percentage = (
            count / result.processed_files * 100 if result.processed_files > 0 else 0
        )
        print(f"  {level.title()}: {count} files ({percentage:.1f}%)")

    # Show some individual results
    print(f"\nSample File Results:")
    for i, (file_path, file_result) in enumerate(list(result.file_results.items())[:5]):
        if "error" in file_result:
            print(
                f"  ❌ {Path(file_path).name}: Error - {file_result['error']}")
        else:
            score = file_result["overall_score"]
            confidence = file_result["confidence"]
            print(
                f"  ✅ {Path(file_path).name}: {score:.3f} (confidence: {confidence:.3f})"
            )

    if len(result.file_results) > 5:
        print(f"  ... and {len(result.file_results) - 5} more files")

    # Save results in different formats
    print(f"\nSaving results...")

    # JSON format
    json_output = Path("batch_results.json")
    processor.save_results(result, json_output, "json")
    print(f"  JSON report: {json_output}")

    # CSV format
    csv_output = Path("batch_results.csv")
    processor.save_results(result, csv_output, "csv")
    print(f"  CSV report: {csv_output}")

    # HTML format
    html_output = Path("batch_results.html")
    processor.save_results(result, html_output, "html")
    print(f"  HTML report: {html_output}")

    # Demonstrate streaming processing for large datasets
    print(f"\nDemonstrating streaming processing...")

    # Create file list
    file_list = Path("file_list.txt")
    udl_files = list(sample_dir.glob("*.udl"))
    with open(file_list, "w") as f:
        for file_path in udl_files:
            f.write(f"{file_path}\n")

    # Stream processing
    streaming_output = Path("streaming_results.json")
    processed_count = 0

    print("Processing files with streaming output...")
    for file_result in processor.process_files_streaming(udl_files, streaming_output):
        processed_count += 1
        file_name = Path(file_result["file_path"]).name
        print(f"  Streamed result {processed_count}: {file_name}")

    print(f"Streaming results saved to: {streaming_output}")

    # Cache demonstration
    print(f"\nCache Performance:")
    if config.enable_caching and config.cache_dir.exists():
        cache_files = list(config.cache_dir.glob("*.json"))
        print(f"  Cache files created: {len(cache_files)}")
        print(f"  Cache directory: {config.cache_dir}")

        # Process same files again to show cache benefit
        print("  Processing same files again (should be faster due to caching)...")
        start_time = time.time()
        result2 = processor.process_directory(sample_dir, patterns=["*.udl"])
        cached_time = time.time() - start_time

        speedup = processing_time / \
            cached_time if cached_time > 0 else float("inf")
        print(f"  Original time: {processing_time:.2f}s")
        print(f"  Cached time: {cached_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")

    print("\nBatch processing example completed!")
    print("\nNext steps:")
    print("1. Customize BatchConfig for your specific needs")
    print("2. Use appropriate worker count for your system")
    print("3. Enable caching for repeated processing")
    print("4. Choose output format based on your requirements")
    print("5. Use streaming for very large datasets")


if __name__ == "__main__":
    main()
