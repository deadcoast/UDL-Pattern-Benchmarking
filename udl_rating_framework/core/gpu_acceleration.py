"""
GPU acceleration for CTM inference in UDL Rating Framework.

Provides GPU-accelerated processing for CTM models with automatic device
management, memory optimization, and batch processing.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary

logger = logging.getLogger(__name__)


class GPUDeviceManager:
    """
    GPU device management for optimal resource utilization.

    Handles device selection, memory management, and multi-GPU support.
    """

    def __init__(self):
        """Initialize GPU device manager."""
        self.available_devices = []
        self.current_device = None
        self.memory_info = {}

        self._detect_devices()

    def _detect_devices(self) -> None:
        """Detect available GPU devices."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, GPU acceleration disabled")
            return

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA devices")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory

                self.available_devices.append(
                    {
                        "id": i,
                        "name": device_name,
                        "type": "cuda",
                        "memory_total": memory_total,
                        "device": torch.device(f"cuda:{i}"),
                    }
                )

                logger.info(
                    f"  Device {i}: {device_name} ({memory_total / 1024**3:.1f} GB)"
                )

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.available_devices.append(
                {
                    "id": "mps",
                    "name": "Apple Silicon GPU",
                    "type": "mps",
                    "memory_total": None,  # Not available for MPS
                    "device": torch.device("mps"),
                }
            )
            logger.info("Apple Silicon GPU (MPS) available")

        # Fallback to CPU
        if not self.available_devices:
            self.available_devices.append(
                {
                    "id": "cpu",
                    "name": "CPU",
                    "type": "cpu",
                    "memory_total": None,
                    "device": torch.device("cpu"),
                }
            )
            logger.info("No GPU devices found, using CPU")

    def get_best_device(self, memory_required: Optional[int] = None) -> torch.device:
        """
        Get the best available device for processing.

        Args:
            memory_required: Required memory in bytes

        Returns:
            Best available torch device
        """
        if not self.available_devices:
            return torch.device("cpu")

        # Filter devices by memory requirement
        suitable_devices = []
        for device_info in self.available_devices:
            if device_info["type"] == "cpu":
                suitable_devices.append(device_info)
            elif memory_required is None:
                suitable_devices.append(device_info)
            elif (
                device_info["memory_total"]
                and device_info["memory_total"] >= memory_required
            ):
                suitable_devices.append(device_info)

        if not suitable_devices:
            logger.warning("No devices meet memory requirements, using CPU")
            return torch.device("cpu")

        # Select device with most available memory
        if suitable_devices[0]["type"] == "cuda":
            best_device = None
            max_free_memory = 0

            for device_info in suitable_devices:
                if device_info["type"] == "cuda":
                    torch.cuda.set_device(device_info["id"])
                    free_memory = torch.cuda.get_device_properties(
                        device_info["id"]
                    ).total_memory
                    free_memory -= torch.cuda.memory_allocated(
                        device_info["id"])

                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = device_info["device"]

            return best_device or suitable_devices[0]["device"]

        return suitable_devices[0]["device"]

    def get_memory_info(self, device: torch.device) -> Dict[str, int]:
        """
        Get memory information for a device.

        Args:
            device: PyTorch device

        Returns:
            Dictionary with memory information
        """
        if device.type == "cuda":
            torch.cuda.set_device(device)
            return {
                "total": torch.cuda.get_device_properties(device).total_memory,
                "allocated": torch.cuda.memory_allocated(device),
                "cached": torch.cuda.memory_reserved(device),
                "free": torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated(device),
            }
        else:
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

    def clear_cache(self, device: Optional[torch.device] = None) -> None:
        """
        Clear GPU memory cache.

        Args:
            device: Device to clear cache for (all CUDA devices if None)
        """
        if device and device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        elif device is None:
            torch.cuda.empty_cache()


class UDLDataset(Dataset):
    """
    Dataset for UDL representations optimized for GPU processing.
    """

    def __init__(
        self,
        udl_representations: List[UDLRepresentation],
        vocabulary: UDLTokenVocabulary,
        max_length: int = 512,
    ):
        """
        Initialize UDL dataset.

        Args:
            udl_representations: List of UDL representations
            vocabulary: Token vocabulary
            max_length: Maximum sequence length
        """
        self.udl_representations = udl_representations
        self.vocabulary = vocabulary
        self.max_length = max_length

        # Pre-tokenize all UDLs for efficiency
        self.tokenized_udls = []
        for udl in udl_representations:
            tokens = udl.get_tokens()
            token_ids = [vocabulary.get_token_id(
                token.text) for token in tokens]

            # Truncate or pad to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend(
                    [vocabulary.pad_token_id] * (max_length - len(token_ids))
                )

            self.tokenized_udls.append(token_ids)

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.udl_representations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized UDL data
        """
        return {
            "token_ids": torch.tensor(self.tokenized_udls[idx], dtype=torch.long),
            "file_path": self.udl_representations[idx].file_path,
            "udl_idx": idx,
        }


class GPUAcceleratedCTM:
    """
    GPU-accelerated CTM model for UDL rating.

    Provides optimized GPU inference with automatic batching,
    memory management, and multi-GPU support.
    """

    def __init__(
        self,
        model: UDLRatingCTM,
        device_manager: Optional[GPUDeviceManager] = None,
        batch_size: int = 32,
        max_sequence_length: int = 512,
        enable_mixed_precision: bool = True,
        enable_compilation: bool = True,
    ):
        """
        Initialize GPU-accelerated CTM.

        Args:
            model: UDL rating CTM model
            device_manager: GPU device manager
            batch_size: Batch size for inference
            max_sequence_length: Maximum sequence length
            enable_mixed_precision: Whether to use mixed precision
            enable_compilation: Whether to compile model for optimization
        """
        self.model = model
        self.device_manager = device_manager or GPUDeviceManager()
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_compilation = enable_compilation

        # Select best device
        self.device = self.device_manager.get_best_device()
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable optimizations
        if self.enable_compilation and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compilation enabled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        # Setup mixed precision
        self.scaler = None
        if self.enable_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision enabled")

    def predict_batch(
        self,
        udl_representations: List[UDLRepresentation],
        vocabulary: UDLTokenVocabulary,
    ) -> List[Dict[str, float]]:
        """
        Predict quality scores for a batch of UDLs.

        Args:
            udl_representations: List of UDL representations
            vocabulary: Token vocabulary

        Returns:
            List of prediction dictionaries
        """
        # Create dataset and dataloader
        dataset = UDLDataset(udl_representations,
                             vocabulary, self.max_sequence_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for GPU processing
            pin_memory=True if self.device.type == "cuda" else False,
        )

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                token_ids = batch["token_ids"].to(
                    self.device, non_blocking=True)

                # Forward pass with optional mixed precision
                if self.enable_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        ratings, certainties, activations, synch = self.model(
                            token_ids)
                else:
                    ratings, certainties, activations, synch = self.model(
                        token_ids)

                # Convert to CPU and extract results
                ratings = ratings.cpu().numpy()
                certainties = certainties.cpu().numpy()

                # Create prediction dictionaries
                for i in range(len(ratings)):
                    udl_idx = batch["udl_idx"][i].item()
                    file_path = batch["file_path"][i]

                    prediction = {
                        "file_path": file_path,
                        "udl_idx": udl_idx,
                        "overall_score": float(ratings[i, 0]),
                        "confidence": float(
                            certainties[i, 0]
                        ),  # Use first certainty component
                        "raw_certainties": certainties[i].tolist(),
                    }

                    predictions.append(prediction)

        return predictions

    def predict_single(
        self, udl_representation: UDLRepresentation, vocabulary: UDLTokenVocabulary
    ) -> Dict[str, float]:
        """
        Predict quality score for a single UDL.

        Args:
            udl_representation: UDL representation
            vocabulary: Token vocabulary

        Returns:
            Prediction dictionary
        """
        predictions = self.predict_batch([udl_representation], vocabulary)
        return predictions[0]

    def benchmark_inference(
        self,
        udl_representations: List[UDLRepresentation],
        vocabulary: UDLTokenVocabulary,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            udl_representations: List of UDL representations for benchmarking
            vocabulary: Token vocabulary
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        logger.info(
            f"Benchmarking GPU inference with {len(udl_representations)} UDLs")

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.predict_batch(udl_representations, vocabulary)

        # Clear cache and synchronize
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark
        times = []
        memory_usage = []

        for i in range(benchmark_iterations):
            # Record memory before
            if self.device.type == "cuda":
                memory_before = torch.cuda.memory_allocated(self.device)

            start_time = time.perf_counter()
            self.predict_batch(udl_representations, vocabulary)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Record memory after
            if self.device.type == "cuda":
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_usage.append(memory_after - memory_before)

            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(udl_representations) / avg_time

        results = {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "throughput_udls_per_second": throughput,
            "batch_size": self.batch_size,
            "sequence_length": self.max_sequence_length,
            "device": str(self.device),
        }

        if memory_usage:
            results.update(
                {
                    "avg_memory_usage_mb": np.mean(memory_usage) / 1024 / 1024,
                    "max_memory_usage_mb": np.max(memory_usage) / 1024 / 1024,
                }
            )

        logger.info(
            f"Benchmark results: {throughput:.1f} UDLs/sec, {avg_time:.4f}s avg time"
        )

        return results

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage."""
        return self.device_manager.get_memory_info(self.device)

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        self.device_manager.clear_cache(self.device)


class GPUAcceleratedProcessor:
    """
    High-level GPU-accelerated processor for UDL rating.

    Provides easy-to-use interface for GPU-accelerated UDL processing
    with automatic optimization and resource management.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        max_sequence_length: int = 512,
        enable_mixed_precision: bool = True,
    ):
        """
        Initialize GPU-accelerated processor.

        Args:
            model_config: CTM model configuration
            batch_size: Batch size for processing
            max_sequence_length: Maximum sequence length
            enable_mixed_precision: Whether to use mixed precision
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not available. Install PyTorch for GPU acceleration."
            )

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.enable_mixed_precision = enable_mixed_precision

        # Initialize device manager
        self.device_manager = GPUDeviceManager()

        # Create vocabulary (will be initialized when first used)
        self.vocabulary = None

        # Initialize model
        if model_config is None:
            model_config = {
                "vocab_size": 10000,  # Will be updated based on vocabulary
                "d_model": 256,
                "iterations": 20,
                "n_synch_out": 32,
            }

        self.model_config = model_config
        self.gpu_model = None

    def _initialize_model(self, vocabulary: UDLTokenVocabulary) -> None:
        """Initialize the GPU model with vocabulary."""
        if self.gpu_model is not None:
            return

        # Update model config with actual vocabulary size
        self.model_config["vocab_size"] = len(vocabulary)

        # Create CTM model
        model = UDLRatingCTM(**self.model_config)

        # Create GPU-accelerated model
        self.gpu_model = GPUAcceleratedCTM(
            model=model,
            device_manager=self.device_manager,
            batch_size=self.batch_size,
            max_sequence_length=self.max_sequence_length,
            enable_mixed_precision=self.enable_mixed_precision,
        )

        logger.info("GPU-accelerated CTM model initialized")

    def process_udls(
        self,
        udl_representations: List[UDLRepresentation],
        vocabulary: Optional[UDLTokenVocabulary] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process UDL representations using GPU acceleration.

        Args:
            udl_representations: List of UDL representations
            vocabulary: Token vocabulary (created automatically if None)

        Returns:
            List of processing results
        """
        if not udl_representations:
            return []

        # Create vocabulary if not provided
        if vocabulary is None:
            vocabulary = UDLTokenVocabulary()

            # Build vocabulary from UDL representations
            for udl in udl_representations:
                tokens = udl.get_tokens()
                for token in tokens:
                    vocabulary.add_token(token.text)

            logger.info(f"Created vocabulary with {len(vocabulary)} tokens")

        self.vocabulary = vocabulary

        # Initialize model
        self._initialize_model(vocabulary)

        # Process UDLs in batches
        logger.info(
            f"Processing {len(udl_representations)} UDLs with GPU acceleration")

        start_time = time.time()
        predictions = self.gpu_model.predict_batch(
            udl_representations, vocabulary)
        processing_time = time.time() - start_time

        logger.info(
            f"GPU processing completed in {processing_time:.3f}s "
            f"({len(udl_representations) / processing_time:.1f} UDLs/sec)"
        )

        return predictions

    def process_files(
        self, file_paths: List[Path], vocabulary: Optional[UDLTokenVocabulary] = None
    ) -> List[Dict[str, Any]]:
        """
        Process UDL files using GPU acceleration.

        Args:
            file_paths: List of UDL file paths
            vocabulary: Token vocabulary (created automatically if None)

        Returns:
            List of processing results
        """
        # Load UDL representations
        udl_representations = []
        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8")
                udl = UDLRepresentation(content, str(file_path))
                udl_representations.append(udl)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        return self.process_udls(udl_representations, vocabulary)

    def benchmark_performance(
        self,
        test_udls: List[UDLRepresentation],
        vocabulary: Optional[UDLTokenVocabulary] = None,
    ) -> Dict[str, float]:
        """
        Benchmark GPU performance.

        Args:
            test_udls: List of test UDL representations
            vocabulary: Token vocabulary

        Returns:
            Benchmark results
        """
        if vocabulary is None:
            vocabulary = UDLTokenVocabulary()
            for udl in test_udls:
                tokens = udl.get_tokens()
                for token in tokens:
                    vocabulary.add_token(token.text)

        self.vocabulary = vocabulary
        self._initialize_model(vocabulary)

        return self.gpu_model.benchmark_inference(test_udls, vocabulary)

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        return {
            "available_devices": self.device_manager.available_devices,
            "current_device": str(self.device_manager.get_best_device()),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "cuda_version": (
                torch.version.cuda
                if TORCH_AVAILABLE and torch.cuda.is_available()
                else None
            ),
        }

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        if self.gpu_model:
            return self.gpu_model.get_memory_usage()
        return {}

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.gpu_model:
            self.gpu_model.clear_cache()


# Convenience functions
def process_udls_gpu(
    udl_representations: List[UDLRepresentation],
    batch_size: int = 32,
    max_sequence_length: int = 512,
    enable_mixed_precision: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to process UDLs with GPU acceleration.

    Args:
        udl_representations: List of UDL representations
        batch_size: Batch size for processing
        max_sequence_length: Maximum sequence length
        enable_mixed_precision: Whether to use mixed precision
        model_config: CTM model configuration

    Returns:
        List of processing results
    """
    processor = GPUAcceleratedProcessor(
        model_config=model_config,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        enable_mixed_precision=enable_mixed_precision,
    )

    return processor.process_udls(udl_representations)


def process_files_gpu(
    file_paths: List[Path],
    batch_size: int = 32,
    max_sequence_length: int = 512,
    enable_mixed_precision: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to process files with GPU acceleration.

    Args:
        file_paths: List of UDL file paths
        batch_size: Batch size for processing
        max_sequence_length: Maximum sequence length
        enable_mixed_precision: Whether to use mixed precision
        model_config: CTM model configuration

    Returns:
        List of processing results
    """
    processor = GPUAcceleratedProcessor(
        model_config=model_config,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        enable_mixed_precision=enable_mixed_precision,
    )

    return processor.process_files(file_paths)


def benchmark_gpu_performance(
    test_udls: List[UDLRepresentation],
    batch_sizes: List[int] = None,
    sequence_lengths: List[int] = None,
) -> Dict[str, Any]:
    """
    Benchmark GPU performance across different configurations.

    Args:
        test_udls: List of test UDL representations
        batch_sizes: List of batch sizes to test
        sequence_lengths: List of sequence lengths to test

    Returns:
        Comprehensive benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64]

    if sequence_lengths is None:
        sequence_lengths = [128, 256, 512, 1024]

    results = {}

    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            config_name = f"batch_{batch_size}_seq_{seq_len}"

            try:
                processor = GPUAcceleratedProcessor(
                    batch_size=batch_size, max_sequence_length=seq_len
                )

                benchmark_result = processor.benchmark_performance(test_udls)
                results[config_name] = benchmark_result

                logger.info(
                    f"Benchmark {config_name}: {benchmark_result['throughput_udls_per_second']:.1f} UDLs/sec"
                )

            except Exception as e:
                logger.error(f"Benchmark {config_name} failed: {e}")
                results[config_name] = {"error": str(e)}

    return results
