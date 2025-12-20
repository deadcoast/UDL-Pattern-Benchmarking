"""
Language Server Protocol (LSP) implementation for UDL Rating Framework.

Provides real-time UDL quality feedback in IDEs and editors that support LSP.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import threading
import time

from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """LSP Position."""

    line: int
    character: int


@dataclass
class Range:
    """LSP Range."""

    start: Position
    end: Position


@dataclass
class Diagnostic:
    """LSP Diagnostic."""

    range: Range
    severity: int  # 1=Error, 2=Warning, 3=Information, 4=Hint
    code: Optional[str]
    source: str
    message: str
    related_information: Optional[List[Any]] = None


@dataclass
class TextDocumentItem:
    """LSP TextDocumentItem."""

    uri: str
    language_id: str
    version: int
    text: str


class UDLLanguageServer:
    """
    Language Server Protocol implementation for UDL quality checking.

    Provides:
    - Real-time quality diagnostics
    - Hover information with quality metrics
    - Code actions for quality improvements
    - Document symbols for UDL structure
    """

    def __init__(
        self,
        min_quality_threshold: float = 0.7,
        enable_real_time: bool = True,
        debounce_delay: float = 0.5,
    ):
        """
        Initialize UDL Language Server.

        Args:
            min_quality_threshold: Minimum quality threshold for warnings
            enable_real_time: Enable real-time quality checking
            debounce_delay: Delay before processing changes (seconds)
        """
        self.min_quality_threshold = min_quality_threshold
        self.enable_real_time = enable_real_time
        self.debounce_delay = debounce_delay

        # Initialize rating pipeline with default metrics
        default_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        self.pipeline = RatingPipeline(metric_names=default_metrics)

        # Document management
        self.documents: Dict[str, TextDocumentItem] = {}
        self.document_versions: Dict[str, int] = {}
        self.quality_cache: Dict[str, Dict[str, Any]] = {}

        # Debouncing
        self.debounce_timers: Dict[str, threading.Timer] = {}

        # LSP state
        self.initialized = False
        self.shutdown_requested = False

        # Capabilities
        self.server_capabilities = {
            "textDocumentSync": {
                "openClose": True,
                "change": 2,  # Incremental
                "save": {"includeText": True},
            },
            "diagnosticProvider": {
                "interFileDependencies": False,
                "workspaceDiagnostics": False,
            },
            "hoverProvider": True,
            "codeActionProvider": {"codeActionKinds": ["quickfix", "refactor.rewrite"]},
            "documentSymbolProvider": True,
            "completionProvider": {"triggerCharacters": ["::", "->", "."]},
        }

    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        logger.info("Initializing UDL Language Server")

        # Extract client capabilities
        client_capabilities = params.get("capabilities", {})

        # Configure server based on client capabilities
        text_document_caps = client_capabilities.get("textDocument", {})
        if not text_document_caps.get("publishDiagnostics", {}).get(
            "versionSupport", False
        ):
            # Client doesn't support versioned diagnostics
            self.server_capabilities["diagnosticProvider"] = False

        self.initialized = True

        return {
            "capabilities": self.server_capabilities,
            "serverInfo": {"name": "UDL Rating Language Server", "version": "1.0.0"},
        }

    async def initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification."""
        logger.info("UDL Language Server initialized successfully")

    async def shutdown(self, params: Dict[str, Any]) -> None:
        """Handle shutdown request."""
        logger.info("Shutting down UDL Language Server")
        self.shutdown_requested = True

        # Cancel all debounce tasks
        for task in self.debounce_timers.values():
            if hasattr(task, "cancel"):
                task.cancel()
        self.debounce_timers.clear()

    async def exit(self, params: Dict[str, Any]) -> None:
        """Handle exit notification."""
        logger.info("UDL Language Server exiting")

    async def text_document_did_open(self, params: Dict[str, Any]) -> None:
        """Handle textDocument/didOpen notification."""
        text_document = params["textDocument"]
        uri = text_document["uri"]

        # Store document
        self.documents[uri] = TextDocumentItem(
            uri=uri,
            language_id=text_document["languageId"],
            version=text_document["version"],
            text=text_document["text"],
        )
        self.document_versions[uri] = text_document["version"]

        logger.info(f"Opened document: {uri}")

        # Trigger quality check
        if self.enable_real_time:
            await self._schedule_quality_check(uri)

    async def text_document_did_change(self, params: Dict[str, Any]) -> None:
        """Handle textDocument/didChange notification."""
        text_document = params["textDocument"]
        uri = text_document["uri"]
        version = text_document["version"]

        if uri not in self.documents:
            logger.warning(f"Received change for unknown document: {uri}")
            return

        # Apply changes
        document = self.documents[uri]
        for change in params["contentChanges"]:
            if "range" in change:
                # Incremental change
                self._apply_incremental_change(document, change)
            else:
                # Full document change
                document.text = change["text"]

        document.version = version
        self.document_versions[uri] = version

        # Trigger quality check with debouncing
        if self.enable_real_time:
            await self._schedule_quality_check(uri)

    async def text_document_did_save(self, params: Dict[str, Any]) -> None:
        """Handle textDocument/didSave notification."""
        text_document = params["textDocument"]
        uri = text_document["uri"]

        logger.info(f"Saved document: {uri}")

        # Force quality check on save
        await self._perform_quality_check(uri)

    async def text_document_did_close(self, params: Dict[str, Any]) -> None:
        """Handle textDocument/didClose notification."""
        text_document = params["textDocument"]
        uri = text_document["uri"]

        # Clean up document data
        self.documents.pop(uri, None)
        self.document_versions.pop(uri, None)
        self.quality_cache.pop(uri, None)

        # Cancel debounce task
        if uri in self.debounce_timers:
            task = self.debounce_timers[uri]
            if hasattr(task, "cancel"):
                task.cancel()
            del self.debounce_timers[uri]

        logger.info(f"Closed document: {uri}")

    async def text_document_hover(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle textDocument/hover request."""
        text_document = params["textDocument"]
        position = params["position"]
        uri = text_document["uri"]

        if uri not in self.documents:
            return None

        # Get quality information for hover
        quality_info = self.quality_cache.get(uri)
        if not quality_info:
            return None

        # Create hover content
        content = self._create_hover_content(quality_info)

        return {
            "contents": {"kind": "markdown", "value": content},
            "range": {"start": position, "end": position},
        }

    async def text_document_code_action(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle textDocument/codeAction request."""
        text_document = params["textDocument"]
        range_param = params["range"]
        context = params["context"]
        uri = text_document["uri"]

        if uri not in self.documents:
            return []

        quality_info = self.quality_cache.get(uri)
        if not quality_info:
            return []

        # Generate code actions based on quality issues
        actions = []

        # Add action to show detailed quality report
        actions.append(
            {
                "title": "Show UDL Quality Report",
                "kind": "quickfix",
                "command": {
                    "title": "Show Quality Report",
                    "command": "udl.showQualityReport",
                    "arguments": [uri],
                },
            }
        )

        # Add actions for specific quality improvements
        if quality_info.get("overall_score", 1.0) < self.min_quality_threshold:
            actions.append(
                {
                    "title": "Improve UDL Quality",
                    "kind": "refactor.rewrite",
                    "command": {
                        "title": "Improve Quality",
                        "command": "udl.improveQuality",
                        "arguments": [uri, quality_info],
                    },
                }
            )

        return actions

    async def text_document_document_symbol(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle textDocument/documentSymbol request."""
        text_document = params["textDocument"]
        uri = text_document["uri"]

        if uri not in self.documents:
            return []

        document = self.documents[uri]

        try:
            # Parse UDL to extract symbols
            udl_repr = UDLRepresentation(document.text, uri)
            symbols = self._extract_document_symbols(udl_repr)
            return symbols

        except Exception as e:
            logger.error(f"Error extracting symbols from {uri}: {e}")
            return []

    async def _schedule_quality_check(self, uri: str) -> None:
        """Schedule a quality check with debouncing."""
        # Cancel existing timer
        if uri in self.debounce_timers:
            self.debounce_timers[uri].cancel()

        # Use asyncio.sleep for debouncing instead of threading.Timer
        # This avoids issues with creating tasks from non-async contexts
        async def debounced_check():
            await asyncio.sleep(self.debounce_delay)
            await self._perform_quality_check(uri)

        # Store the task for potential cancellation
        task = asyncio.create_task(debounced_check())
        self.debounce_timers[uri] = task

    async def _perform_quality_check(self, uri: str) -> None:
        """Perform quality check on document."""
        if uri not in self.documents:
            return

        document = self.documents[uri]

        try:
            # Analyze UDL quality
            udl_repr = UDLRepresentation(document.text, uri)
            report = self.pipeline.rate_udl(udl_repr)

            # Cache quality information
            self.quality_cache[uri] = {
                "overall_score": report.overall_score,
                "confidence": report.confidence,
                "metric_scores": report.metric_scores,
                "timestamp": time.time(),
            }

            # Generate diagnostics
            diagnostics = self._create_diagnostics(uri, report)

            # Publish diagnostics
            await self._publish_diagnostics(uri, diagnostics)

        except Exception as e:
            logger.error(f"Error checking quality for {uri}: {e}")

            # Publish error diagnostic
            error_diagnostic = Diagnostic(
                range=Range(Position(0, 0), Position(0, 0)),
                severity=1,  # Error
                code="parse_error",
                source="udl-rating",
                message=f"Failed to analyze UDL: {str(e)}",
            )
            await self._publish_diagnostics(uri, [error_diagnostic])

    def _apply_incremental_change(
        self, document: TextDocumentItem, change: Dict[str, Any]
    ) -> None:
        """Apply incremental change to document."""
        range_info = change["range"]
        text = change["text"]

        # Convert to line/character positions
        lines = document.text.split("\n")
        start_line = range_info["start"]["line"]
        start_char = range_info["start"]["character"]
        end_line = range_info["end"]["line"]
        end_char = range_info["end"]["character"]

        # Apply change
        if start_line == end_line:
            # Single line change
            line = lines[start_line]
            lines[start_line] = line[:start_char] + text + line[end_char:]
        else:
            # Multi-line change
            start_line_text = lines[start_line][:start_char]
            end_line_text = lines[end_line][end_char:]

            new_lines = text.split("\n")
            new_lines[0] = start_line_text + new_lines[0]
            new_lines[-1] = new_lines[-1] + end_line_text

            lines[start_line : end_line + 1] = new_lines

        document.text = "\n".join(lines)

    def _create_diagnostics(self, uri: str, report: Any) -> List[Diagnostic]:
        """Create diagnostics from quality report."""
        diagnostics = []

        # Overall quality diagnostic
        if report.overall_score < self.min_quality_threshold:
            diagnostics.append(
                Diagnostic(
                    range=Range(Position(0, 0), Position(0, 0)),
                    severity=2,  # Warning
                    code="low_quality",
                    source="udl-rating",
                    message=f"UDL quality score {report.overall_score:.3f} is below threshold {self.min_quality_threshold}",
                )
            )

        # Metric-specific diagnostics
        for metric_name, score in report.metric_scores.items():
            if score < 0.5:  # Low metric score
                diagnostics.append(
                    Diagnostic(
                        range=Range(Position(0, 0), Position(0, 0)),
                        severity=3,  # Information
                        code=f"low_{metric_name.lower()}",
                        source="udl-rating",
                        message=f"Low {metric_name} score: {score:.3f}",
                    )
                )

        return diagnostics

    def _create_hover_content(self, quality_info: Dict[str, Any]) -> str:
        """Create hover content from quality information."""
        content = "## UDL Quality Information\n\n"

        overall_score = quality_info.get("overall_score", 0.0)
        confidence = quality_info.get("confidence", 0.0)

        content += f"**Overall Score**: {overall_score:.3f}\n"
        content += f"**Confidence**: {confidence:.3f}\n\n"

        content += "### Metric Scores\n"
        for metric_name, score in quality_info.get("metric_scores", {}).items():
            emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            content += f"- {emoji} **{metric_name}**: {score:.3f}\n"

        return content

    def _extract_document_symbols(
        self, udl_repr: UDLRepresentation
    ) -> List[Dict[str, Any]]:
        """Extract document symbols from UDL representation."""
        symbols = []

        try:
            # Extract grammar rules as symbols
            grammar_graph = udl_repr.get_grammar_graph()

            for node in grammar_graph.nodes():
                symbols.append(
                    {
                        "name": str(node),
                        "kind": 5,  # Class (closest to grammar rule)
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": len(str(node))},
                        },
                        "selectionRange": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": len(str(node))},
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")

        return symbols

    async def _publish_diagnostics(
        self, uri: str, diagnostics: List[Diagnostic]
    ) -> None:
        """Publish diagnostics to client."""
        # Convert diagnostics to LSP format
        lsp_diagnostics = []
        for diag in diagnostics:
            lsp_diagnostics.append(
                {
                    "range": {
                        "start": {
                            "line": diag.range.start.line,
                            "character": diag.range.start.character,
                        },
                        "end": {
                            "line": diag.range.end.line,
                            "character": diag.range.end.character,
                        },
                    },
                    "severity": diag.severity,
                    "code": diag.code,
                    "source": diag.source,
                    "message": diag.message,
                }
            )

        # Send notification (this would be sent to the LSP client)
        notification = {
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "version": self.document_versions.get(uri),
                "diagnostics": lsp_diagnostics,
            },
        }

        logger.debug(
            f"Publishing diagnostics for {uri}: {len(lsp_diagnostics)} diagnostics"
        )


class LSPServer:
    """
    Simple LSP server implementation for UDL Rating Framework.
    """

    def __init__(self, language_server: UDLLanguageServer):
        """Initialize LSP server."""
        self.language_server = language_server
        self.request_id = 0

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle LSP request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                result = await self.language_server.initialize(params)
                return {"id": request_id, "result": result}

            elif method == "initialized":
                await self.language_server.initialized(params)
                return None

            elif method == "shutdown":
                await self.language_server.shutdown(params)
                return {"id": request_id, "result": None}

            elif method == "exit":
                await self.language_server.exit(params)
                return None

            elif method == "textDocument/didOpen":
                await self.language_server.text_document_did_open(params)
                return None

            elif method == "textDocument/didChange":
                await self.language_server.text_document_did_change(params)
                return None

            elif method == "textDocument/didSave":
                await self.language_server.text_document_did_save(params)
                return None

            elif method == "textDocument/didClose":
                await self.language_server.text_document_did_close(params)
                return None

            elif method == "textDocument/hover":
                result = await self.language_server.text_document_hover(params)
                return {"id": request_id, "result": result}

            elif method == "textDocument/codeAction":
                result = await self.language_server.text_document_code_action(params)
                return {"id": request_id, "result": result}

            elif method == "textDocument/documentSymbol":
                result = await self.language_server.text_document_document_symbol(
                    params
                )
                return {"id": request_id, "result": result}

            else:
                logger.warning(f"Unhandled method: {method}")
                return {
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request {method}: {e}")
            return {
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }


def main():
    """CLI entry point for LSP server."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="UDL Language Server")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Minimum quality threshold"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=True,
        help="Enable real-time quality checking",
    )
    parser.add_argument(
        "--debounce", type=float, default=0.5, help="Debounce delay in seconds"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create language server
    language_server = UDLLanguageServer(
        min_quality_threshold=args.threshold,
        enable_real_time=args.real_time,
        debounce_delay=args.debounce,
    )

    lsp_server = LSPServer(language_server)

    logger.info("Starting UDL Language Server")

    # Simple stdio-based LSP server
    async def stdio_server():
        while not language_server.shutdown_requested:
            try:
                # Read request from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line.strip())
                response = await lsp_server.handle_request(request)

                if response:
                    print(json.dumps(response))
                    sys.stdout.flush()

            except Exception as e:
                logger.error(f"Error in stdio server: {e}")
                break

    # Run server
    try:
        asyncio.run(stdio_server())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
