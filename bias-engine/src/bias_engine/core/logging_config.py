"""
Logging configuration for the Bias Engine.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.processors import JSONRenderer

from bias_engine.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Configure structlog
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        processor = JSONRenderer()
    else:
        processor = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [processor],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure specific loggers
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn").propagate = False


def get_correlation_id_processor() -> Any:
    """Create a processor to add correlation IDs to logs."""

    def processor(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        # This would typically extract correlation ID from context
        # For now, we'll add a placeholder
        event_dict["correlation_id"] = getattr(structlog.contextvars, "correlation_id", None)
        return event_dict

    return processor