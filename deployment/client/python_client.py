"""
Python client for UDL Rating Framework API.

This module provides a convenient Python interface for interacting with
the UDL Rating Framework REST API.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class UDLRatingClient:
    """Client for UDL Rating Framework API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
            api_token: API authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set authentication header
        if api_token:
            self.session.headers.update({
                "Authorization": f"Bearer {api_token}"
            })
    
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def rate_udl(
        self,
        content: str,
        filename: Optional[str] = None,
        use_ctm: bool = False,
        include_trace: bool = False,
    ) -> Dict:
        """
        Rate a UDL from content string.
        
        Args:
            content: UDL content to rate
            filename: Optional filename
            use_ctm: Whether to use CTM model
            include_trace: Whether to include computation trace
        
        Returns:
            Rating response with scores and metadata
        """
        payload = {
            "content": content,
            "filename": filename,
            "use_ctm": use_ctm,
            "include_trace": include_trace,
        }
        
        response = self.session.post(
            f"{self.base_url}/rate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def rate_udl_file(
        self,
        file_path: Union[str, Path],
        use_ctm: bool = False,
        include_trace: bool = False,
    ) -> Dict:
        """
        Rate a UDL from file.
        
        Args:
            file_path: Path to UDL file
            use_ctm: Whether to use CTM model
            include_trace: Whether to include computation trace
        
        Returns:
            Rating response with scores and metadata
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/plain')}
            data = {
                'use_ctm': use_ctm,
                'include_trace': include_trace,
            }
            
            response = self.session.post(
                f"{self.base_url}/rate/file",
                files=files,
                data=data,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        return response.json()
    
    def rate_udl_batch(
        self,
        udls: List[Dict],
        parallel: bool = True,
    ) -> Dict:
        """
        Rate multiple UDLs in batch.
        
        Args:
            udls: List of UDL requests (each with content, filename, etc.)
            parallel: Whether to process in parallel
        
        Returns:
            Batch rating response
        """
        payload = {
            "udls": udls,
            "parallel": parallel,
        }
        
        response = self.session.post(
            f"{self.base_url}/rate/batch",
            json=payload,
            timeout=self.timeout * 5  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()
    
    def get_available_metrics(self) -> Dict:
        """
        Get list of available quality metrics.
        
        Returns:
            Information about available metrics
        """
        response = self.session.get(
            f"{self.base_url}/metrics",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def rate_directory(
        self,
        directory_path: Union[str, Path],
        extensions: List[str] = None,
        use_ctm: bool = False,
        include_trace: bool = False,
        parallel: bool = True,
    ) -> List[Dict]:
        """
        Rate all UDL files in a directory.
        
        Args:
            directory_path: Path to directory containing UDL files
            extensions: File extensions to include (default: ['.udl', '.dsl', '.grammar'])
            use_ctm: Whether to use CTM model
            include_trace: Whether to include computation trace
            parallel: Whether to process in parallel
        
        Returns:
            List of rating results
        """
        if extensions is None:
            extensions = ['.udl', '.dsl', '.grammar', '.ebnf', '.txt']
        
        directory_path = Path(directory_path)
        udl_files = []
        
        # Find all UDL files
        for ext in extensions:
            udl_files.extend(directory_path.glob(f"**/*{ext}"))
        
        if not udl_files:
            return []
        
        # Prepare batch request
        udl_requests = []
        for file_path in udl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                udl_requests.append({
                    "content": content,
                    "filename": str(file_path),
                    "use_ctm": use_ctm,
                    "include_trace": include_trace,
                })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if not udl_requests:
            return []
        
        # Process in batches of 10
        batch_size = 10
        all_results = []
        
        for i in range(0, len(udl_requests), batch_size):
            batch = udl_requests[i:i + batch_size]
            try:
                batch_response = self.rate_udl_batch(batch, parallel=parallel)
                all_results.extend(batch_response['results'])
            except Exception as e:
                print(f"Warning: Batch {i//batch_size + 1} failed: {e}")
        
        return all_results


class UDLRatingError(Exception):
    """Exception raised by UDL Rating Client."""
    pass


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = UDLRatingClient(
        base_url="http://localhost:8000",
        api_token=None,  # Set if authentication is required
    )
    
    # Check health
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"Health check failed: {e}")
        exit(1)
    
    # Example UDL content
    udl_content = """
    grammar SimpleCalculator {
        expr = term (('+' | '-') term)*
        term = factor (('*' | '/') factor)*
        factor = number | '(' expr ')'
        number = [0-9]+
    }
    """
    
    # Rate the UDL
    try:
        result = client.rate_udl(
            content=udl_content,
            filename="simple_calculator.udl",
            include_trace=True
        )
        
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        print(f"Model Used: {result['model_used']}")
        
        print("\nMetric Scores:")
        for metric in result['metrics']:
            print(f"  {metric['name']}: {metric['value']:.3f}")
        
        if result.get('trace'):
            print(f"\nComputation Trace: {len(result['trace'])} steps")
    
    except Exception as e:
        print(f"Rating failed: {e}")
    
    # Get available metrics
    try:
        metrics_info = client.get_available_metrics()
        print(f"\nAvailable Metrics: {len(metrics_info['metrics'])}")
        for metric in metrics_info['metrics']:
            print(f"  - {metric['name']}")
    except Exception as e:
        print(f"Failed to get metrics: {e}")