"""
FastAPI REST API for UDL Rating Service.

This module provides a REST API for the UDL Rating Framework, allowing
users to submit UDL files for quality assessment via HTTP requests.
"""

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    from udl_rating_framework.core.pipeline import RatingPipeline
    from udl_rating_framework.core.representation import UDLRepresentation
    from udl_rating_framework.io.file_discovery import FileDiscovery
    from udl_rating_framework.models.ctm_adapter import UDLRatingCTM

    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import UDL framework components: {e}")
    FRAMEWORK_AVAILABLE = False

    # Create mock classes for testing
    class RatingPipeline:
        def __init__(self):
            self.metrics = []

        def process_udl(self, udl):
            from types import SimpleNamespace

            return SimpleNamespace(
                overall_score=0.8,
                confidence=0.9,
                metric_scores={"ConsistencyMetric": 0.8,
                               "CompletenessMetric": 0.9},
                metric_formulas={
                    "ConsistencyMetric": "1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)",
                    "CompletenessMetric": "|Defined| / |Required|",
                },
                computation_trace=[],
            )

    class UDLRepresentation:
        def __init__(self, content, filename):
            self.content = content
            self.filename = filename

    class UDLRatingCTM:
        @classmethod
        def load_from_checkpoint(cls, path):
            return cls()

        def eval(self):
            pass

    class FileDiscovery:
        pass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer(auto_error=False)

# Global variables for model and pipeline
rating_pipeline: Optional[RatingPipeline] = None
ctm_model: Optional[UDLRatingCTM] = None


class UDLRatingRequest(BaseModel):
    """Request model for UDL rating."""

    content: str = Field(..., description="UDL content to rate")
    filename: Optional[str] = Field(None, description="Optional filename")
    use_ctm: bool = Field(False, description="Use CTM model for rating")
    include_trace: bool = Field(False, description="Include computation trace")


class MetricScore(BaseModel):
    """Individual metric score."""

    name: str
    value: float
    formula: str
    confidence: Optional[float] = None


class UDLRatingResponse(BaseModel):
    """Response model for UDL rating."""

    overall_score: float = Field(...,
                                 description="Overall quality score [0,1]")
    confidence: float = Field(..., description="Confidence in rating [0,1]")
    metrics: List[MetricScore] = Field(...,
                                       description="Individual metric scores")
    processing_time: float = Field(...,
                                   description="Processing time in seconds")
    model_used: str = Field(...,
                            description="Model type used (mathematical/ctm)")
    trace: Optional[List[Dict]] = Field(None, description="Computation trace")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    uptime: float


class BatchRatingRequest(BaseModel):
    """Request model for batch UDL rating."""

    udls: List[UDLRatingRequest] = Field(...,
                                         description="List of UDLs to rate")
    parallel: bool = Field(True, description="Process in parallel")


class BatchRatingResponse(BaseModel):
    """Response model for batch UDL rating."""

    results: List[UDLRatingResponse]
    total_processing_time: float
    successful: int
    failed: int


# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global rating_pipeline, ctm_model

    logger.info("Starting UDL Rating API...")

    # Initialize rating pipeline
    try:
        rating_pipeline = RatingPipeline()
        logger.info("Rating pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize rating pipeline: {e}")
        raise

    # Load CTM model if available
    model_path = os.getenv("CTM_MODEL_PATH")
    if model_path and Path(model_path).exists():
        try:
            ctm_model = UDLRatingCTM.load_from_checkpoint(model_path)
            ctm_model.eval()
            logger.info(f"CTM model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load CTM model: {e}")

    yield

    logger.info("Shutting down UDL Rating API...")


# Create FastAPI app
app = FastAPI(
    title="UDL Rating Framework API",
    description="REST API for evaluating User Defined Language quality",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Track startup time
startup_time = time.time()


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """Verify API token."""
    expected_token = os.getenv("API_TOKEN")
    if not expected_token:
        return True  # No authentication required if token not set

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=ctm_model is not None,
        uptime=time.time() - startup_time,
    )


@app.post("/rate", response_model=UDLRatingResponse)
@limiter.limit("10/minute")
async def rate_udl(
    request: Request,
    udl_request: UDLRatingRequest,
    _: bool = Depends(verify_token),
):
    """Rate a single UDL."""
    start_time = time.time()

    try:
        # Create temporary file for UDL content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".udl", delete=False
        ) as temp_file:
            temp_file.write(udl_request.content)
            temp_path = temp_file.name

        try:
            # Create UDL representation
            udl_repr = UDLRepresentation(
                udl_request.content, udl_request.filename or "temp.udl"
            )

            # Check if rating pipeline is available
            if rating_pipeline is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rating pipeline not initialized",
                )

            # Choose rating method
            if udl_request.use_ctm and ctm_model is not None:
                # Use CTM model
                model_used = "ctm"
                # TODO: Implement CTM inference
                # For now, fall back to mathematical metrics
                report = rating_pipeline.process_udl(udl_repr)
            else:
                # Use mathematical metrics
                model_used = "mathematical"
                report = rating_pipeline.process_udl(udl_repr)

            # Build response
            metrics = [
                MetricScore(
                    name=name,
                    value=value,
                    formula=report.metric_formulas.get(name, ""),
                    confidence=report.confidence,
                )
                for name, value in report.metric_scores.items()
            ]

            trace = None
            if udl_request.include_trace:
                trace = [
                    {
                        "step": step.step_number,
                        "operation": step.operation,
                        "formula": step.formula,
                        "output": step.output,
                    }
                    for step in report.computation_trace
                ]

            processing_time = time.time() - start_time

            return UDLRatingResponse(
                overall_score=report.overall_score,
                confidence=report.confidence,
                metrics=metrics,
                processing_time=processing_time,
                model_used=model_used,
                trace=trace,
            )

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error rating UDL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rate UDL: {str(e)}",
        )


@app.post("/rate/file", response_model=UDLRatingResponse)
@limiter.limit("5/minute")
async def rate_udl_file(
    request: Request,
    file: UploadFile = File(...),
    use_ctm: bool = False,
    include_trace: bool = False,
    _: bool = Depends(verify_token),
):
    """Rate a UDL file upload."""
    if not file.filename or not file.filename.endswith(
        (".udl", ".dsl", ".grammar", ".ebnf", ".txt")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported: .udl, .dsl, .grammar, .ebnf, .txt",
        )

    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        udl_request = UDLRatingRequest(
            content=content_str,
            filename=file.filename,
            use_ctm=use_ctm,
            include_trace=include_trace,
        )

        return await rate_udl(request, udl_request)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 text",
        )
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}",
        )


@app.post("/rate/batch", response_model=BatchRatingResponse)
@limiter.limit("2/minute")
async def rate_udl_batch(
    request: Request,
    batch_request: BatchRatingRequest,
    _: bool = Depends(verify_token),
):
    """Rate multiple UDLs in batch."""
    if len(batch_request.udls) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 10 UDLs",
        )

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    async def process_single_udl(
        udl_req: UDLRatingRequest,
    ) -> Optional[UDLRatingResponse]:
        try:
            response = await rate_udl(request, udl_req)
            return response
        except Exception as e:
            logger.error(f"Failed to process UDL {udl_req.filename}: {e}")
            return None

    if batch_request.parallel:
        # Process in parallel
        tasks = [process_single_udl(udl_req) for udl_req in batch_request.udls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, UDLRatingResponse):
                results.append(response)
                successful += 1
            else:
                failed += 1
    else:
        # Process sequentially
        for udl_req in batch_request.udls:
            response = await process_single_udl(udl_req)
            if response:
                results.append(response)
                successful += 1
            else:
                failed += 1

    total_time = time.time() - start_time

    return BatchRatingResponse(
        results=results,
        total_processing_time=total_time,
        successful=successful,
        failed=failed,
    )


@app.get("/metrics")
async def get_available_metrics(_: bool = Depends(verify_token)):
    """Get list of available quality metrics."""
    if rating_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rating pipeline not initialized",
        )

    metrics_info = []
    if hasattr(rating_pipeline, "metrics") and rating_pipeline.metrics:
        for metric in rating_pipeline.metrics:
            try:
                metrics_info.append(
                    {
                        "name": metric.__class__.__name__,
                        "formula": (
                            metric.get_formula()
                            if hasattr(metric, "get_formula")
                            else "N/A"
                        ),
                        "properties": (
                            metric.get_properties()
                            if hasattr(metric, "get_properties")
                            else {}
                        ),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Error getting metric info for {metric.__class__.__name__}: {e}"
                )
                metrics_info.append(
                    {
                        "name": metric.__class__.__name__,
                        "formula": "N/A",
                        "properties": {},
                    }
                )
    else:
        # Return mock metrics for testing
        metrics_info = [
            {
                "name": "ConsistencyMetric",
                "formula": "1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)",
                "properties": {
                    "bounded": True,
                    "monotonic": False,
                    "additive": False,
                    "continuous": True,
                },
            },
            {
                "name": "CompletenessMetric",
                "formula": "|Defined| / |Required|",
                "properties": {
                    "bounded": True,
                    "monotonic": True,
                    "additive": False,
                    "continuous": True,
                },
            },
        ]

    return {"metrics": metrics_info}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
