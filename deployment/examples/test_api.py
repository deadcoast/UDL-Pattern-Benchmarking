#!/usr/bin/env python3
"""
Example script to test the UDL Rating Framework API.

This script demonstrates how to use the deployed API to rate UDL files.
"""

import sys
from pathlib import Path

# Add the client to the path
sys.path.append(str(Path(__file__).parent.parent / "client"))

from python_client import UDLRatingClient


def main():
    """Main function to test the API."""
    print("🚀 Testing UDL Rating Framework API")
    print("=" * 50)

    # Initialize client
    api_url = "http://localhost:8000"
    client = UDLRatingClient(base_url=api_url)

    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        health = client.health_check()
        print(f"   ✅ API Status: {health['status']}")
        print(f"   📊 Model Loaded: {health['model_loaded']}")
        print(f"   ⏱️  Uptime: {health['uptime']:.2f}s")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        print(f"   💡 Make sure the API is running at {api_url}")
        return

    # Test 2: Simple UDL rating
    print("\n2. Testing simple UDL rating...")
    simple_udl = """
    grammar SimpleCalculator {
        expr = term (('+' | '-') term)*
        term = factor (('*' | '/') factor)*
        factor = number | '(' expr ')'
        number = [0-9]+
    }
    """

    try:
        result = client.rate_udl(
            content=simple_udl, filename="simple_calculator.udl", include_trace=True
        )

        print(f"   ✅ Overall Score: {result['overall_score']:.3f}")
        print(f"   🎯 Confidence: {result['confidence']:.3f}")
        print(f"   ⚡ Processing Time: {result['processing_time']:.3f}s")
        print(f"   🔧 Model Used: {result['model_used']}")

        print("   📈 Metric Scores:")
        for metric in result["metrics"]:
            print(f"      • {metric['name']}: {metric['value']:.3f}")

        if result.get("trace"):
            print(f"   🔍 Computation Trace: {len(result['trace'])} steps")

    except Exception as e:
        print(f"   ❌ Simple rating failed: {e}")

    # Test 3: Batch rating
    print("\n3. Testing batch rating...")
    batch_udls = [
        {
            "content": "grammar Test1 { rule = 'hello' }",
            "filename": "test1.udl",
            "use_ctm": False,
            "include_trace": False,
        },
        {
            "content": "grammar Test2 { rule = 'world' | 'universe' }",
            "filename": "test2.udl",
            "use_ctm": False,
            "include_trace": False,
        },
    ]

    try:
        batch_result = client.rate_udl_batch(batch_udls, parallel=True)

        print(
            f"   ✅ Processed: {batch_result['successful']} successful, {batch_result['failed']} failed"
        )
        print(f"   ⏱️  Total Time: {batch_result['total_processing_time']:.3f}s")

        for i, result in enumerate(batch_result["results"]):
            print(f"   📄 UDL {i + 1}: Score {result['overall_score']:.3f}")

    except Exception as e:
        print(f"   ❌ Batch rating failed: {e}")

    # Test 4: Get available metrics
    print("\n4. Testing metrics information...")
    try:
        metrics_info = client.get_available_metrics()
        print(f"   ✅ Available Metrics: {len(metrics_info['metrics'])}")

        for metric in metrics_info["metrics"]:
            print(f"      • {metric['name']}")
            if "properties" in metric:
                props = metric["properties"]
                bounded = "✓" if props.get("bounded", False) else "✗"
                print(f"        Bounded: {bounded}")

    except Exception as e:
        print(f"   ❌ Metrics info failed: {e}")

    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        # Test with invalid UDL
        client.rate_udl(content="invalid grammar syntax {{{", filename="invalid.udl")
        print("   ⚠️  Expected error but got success")
    except Exception as e:
        print(f"   ✅ Error handling works: {type(e).__name__}")

    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("\n💡 Next steps:")
    print("   • Try the web interface (if available)")
    print("   • Test with your own UDL files")
    print("   • Check monitoring dashboards")
    print("   • Scale the deployment as needed")


def create_sample_udl_files():
    """Create sample UDL files for testing."""
    samples_dir = Path(__file__).parent / "sample_udls"
    samples_dir.mkdir(exist_ok=True)

    # Simple calculator grammar
    calculator_udl = """
    grammar Calculator {
        expression = term (('+' | '-') term)*
        term = factor (('*' | '/') factor)*
        factor = number | '(' expression ')'
        number = digit+
        digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
    }
    """

    # JSON subset grammar
    json_udl = """
    grammar JSONSubset {
        value = object | array | string | number | boolean | null
        object = '{' (pair (',' pair)*)? '}'
        pair = string ':' value
        array = '[' (value (',' value)*)? ']'
        string = '"' char* '"'
        char = [^"\\] | '\\' escape
        escape = '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't'
        number = '-'? digit+ ('.' digit+)? (('e' | 'E') ('+' | '-')? digit+)?
        digit = [0-9]
        boolean = 'true' | 'false'
        null = 'null'
    }
    """

    # Simple query language
    query_udl = """
    grammar QueryLanguage {
        query = select_stmt
        select_stmt = 'SELECT' field_list 'FROM' table_name where_clause?
        field_list = field (',' field)*
        field = identifier | '*'
        table_name = identifier
        where_clause = 'WHERE' condition
        condition = field operator value
        operator = '=' | '!=' | '<' | '>' | '<=' | '>='
        value = string | number
        identifier = letter (letter | digit)*
        string = "'" char* "'"
        char = [^']
        number = digit+
        letter = [a-zA-Z]
        digit = [0-9]
    }
    """

    # Write sample files
    (samples_dir / "calculator.udl").write_text(calculator_udl)
    (samples_dir / "json_subset.udl").write_text(json_udl)
    (samples_dir / "query_language.udl").write_text(query_udl)

    print(f"📁 Created sample UDL files in {samples_dir}")
    return samples_dir


def test_file_upload():
    """Test file upload functionality."""
    print("\n6. Testing file upload...")

    # Create sample files
    samples_dir = create_sample_udl_files()
    client = UDLRatingClient(base_url="http://localhost:8000")

    try:
        # Test directory rating
        results = client.rate_directory(samples_dir)

        print(f"   ✅ Rated {len(results)} files from directory")
        for result in results:
            filename = Path(result.get("filename", "unknown")).name
            score = result.get("overall_score", 0)
            print(f"      • {filename}: {score:.3f}")

    except Exception as e:
        print(f"   ❌ Directory rating failed: {e}")


if __name__ == "__main__":
    main()
    test_file_upload()
