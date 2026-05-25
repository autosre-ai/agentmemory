"""
Structured Logging Module

Production-ready structured logging for agent memory systems with
support for JSON output, log correlation, and multiple handlers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from logging.handlers import RotatingFileHandler as _RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, TypeVar, Union

T = TypeVar("T")


class LogLevel(IntEnum):
    """Log levels matching Python's logging module."""
    
    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


@dataclass
class LogConfig:
    """Configuration for structured logging."""
    
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # "json" or "console"
    include_timestamp: bool = True
    include_level: bool = True
    include_logger_name: bool = True
    include_correlation_id: bool = True
    include_thread_info: bool = False
    include_process_info: bool = False
    include_stack_info: bool = False
    timestamp_format: str = "iso8601"  # "iso8601" or "unix"
    
    # Output settings
    output_file: Optional[str] = None
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Performance settings
    async_logging: bool = False
    buffer_size: int = 1000


# Thread-local storage for log context
_log_context = threading.local()


class LogContext:
    """
    Thread-local log context for correlation and metadata.
    
    Provides automatic context propagation for logs within a request
    or operation scope.
    
    Example:
        >>> with LogContext.scope(request_id="abc123", user_id="user1"):
        ...     logger.info("Processing request")  # includes request_id and user_id
    """
    
    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get the current log context."""
        return getattr(_log_context, "context", {})
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value."""
        if not hasattr(_log_context, "context"):
            _log_context.context = {}
        _log_context.context[key] = value
    
    @classmethod
    def update(cls, values: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        if not hasattr(_log_context, "context"):
            _log_context.context = {}
        _log_context.context.update(values)
    
    @classmethod
    def remove(cls, key: str) -> None:
        """Remove a context value."""
        if hasattr(_log_context, "context"):
            _log_context.context.pop(key, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        _log_context.context = {}
    
    @classmethod
    def scope(cls, **kwargs: Any) -> "LogContextScope":
        """Create a context scope that auto-cleans on exit."""
        return LogContextScope(kwargs)
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get the current correlation ID."""
        return cls.get().get("correlation_id")
    
    @classmethod
    def set_correlation_id(cls, correlation_id: Optional[str] = None) -> str:
        """Set or generate a correlation ID."""
        cid = correlation_id or uuid.uuid4().hex
        cls.set("correlation_id", cid)
        return cid


class LogContextScope:
    """Context manager for scoped log context."""
    
    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = values
        self._previous: Dict[str, Any] = {}
    
    def __enter__(self) -> "LogContextScope":
        # Save previous values
        current = LogContext.get()
        for key in self.values:
            if key in current:
                self._previous[key] = current[key]
        
        # Set new values
        LogContext.update(self.values)
        return self
    
    def __exit__(self, *args: Any) -> None:
        # Restore previous values
        for key in self.values:
            if key in self._previous:
                LogContext.set(key, self._previous[key])
            else:
                LogContext.remove(key)


class LogFormatter(ABC):
    """Base class for log formatters."""
    
    @abstractmethod
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record."""
        pass


class JSONFormatter(LogFormatter, logging.Formatter):
    """
    Format log records as JSON.
    
    Produces structured logs suitable for log aggregation systems
    like ELK Stack, Loki, or cloud logging services.
    """
    
    def __init__(self, config: Optional[LogConfig] = None) -> None:
        super().__init__()
        self.config = config or LogConfig()
        self._reserved_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "module", "msecs", "message", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName",
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        output: Dict[str, Any] = {}
        
        # Timestamp
        if self.config.include_timestamp:
            if self.config.timestamp_format == "iso8601":
                output["timestamp"] = datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat()
            else:
                output["timestamp"] = record.created
        
        # Level
        if self.config.include_level:
            output["level"] = record.levelname
        
        # Logger name
        if self.config.include_logger_name:
            output["logger"] = record.name
        
        # Message
        output["message"] = record.getMessage()
        
        # Correlation ID from context
        if self.config.include_correlation_id:
            context = LogContext.get()
            if "correlation_id" in context:
                output["correlation_id"] = context["correlation_id"]
        
        # Thread info
        if self.config.include_thread_info:
            output["thread"] = {
                "id": record.thread,
                "name": record.threadName,
            }
        
        # Process info
        if self.config.include_process_info:
            output["process"] = {
                "id": record.process,
                "name": record.processName,
            }
        
        # Source location
        output["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Exception info
        if record.exc_info:
            output["exception"] = self.formatException(record.exc_info)
        
        # Stack info
        if self.config.include_stack_info and record.stack_info:
            output["stack_info"] = record.stack_info
        
        # Add context fields
        context = LogContext.get()
        for key, value in context.items():
            if key not in output:
                output[key] = value
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in self._reserved_attrs and key not in output:
                try:
                    # Try to serialize to ensure it's JSON-safe
                    json.dumps(value)
                    output[key] = value
                except (TypeError, ValueError):
                    output[key] = str(value)
        
        return json.dumps(output)


class ConsoleFormatter(LogFormatter, logging.Formatter):
    """
    Format log records for console output with colors.
    
    Produces human-readable logs with ANSI color codes for
    different log levels.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(
        self,
        config: Optional[LogConfig] = None,
        use_colors: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or LogConfig()
        self.use_colors = use_colors and self._supports_color()
    
    def _supports_color(self) -> bool:
        """Check if terminal supports colors."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        if not hasattr(sys.stdout, "isatty"):
            return False
        return sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record for console."""
        parts = []
        
        # Timestamp
        if self.config.include_timestamp:
            ts = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            parts.append(f"[{ts}]")
        
        # Level with color
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"
        parts.append(level)
        
        # Logger name
        if self.config.include_logger_name:
            parts.append(f"[{record.name}]")
        
        # Correlation ID
        if self.config.include_correlation_id:
            context = LogContext.get()
            if "correlation_id" in context:
                parts.append(f"[{context['correlation_id'][:8]}]")
        
        # Message
        parts.append(record.getMessage())
        
        # Context fields
        context = LogContext.get()
        extra_fields = {
            k: v for k, v in context.items()
            if k != "correlation_id"
        }
        if extra_fields:
            fields_str = " ".join(f"{k}={v}" for k, v in extra_fields.items())
            parts.append(f"| {fields_str}")
        
        result = " ".join(parts)
        
        # Exception
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)
        
        return result


class LogHandler(ABC):
    """Base class for log handlers."""
    
    @abstractmethod
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the handler."""
        pass


class StreamHandler(LogHandler, logging.StreamHandler):
    """Handler that writes to a stream (stdout/stderr)."""
    
    def __init__(
        self,
        stream: Optional[TextIO] = None,
        formatter: Optional[LogFormatter] = None,
    ) -> None:
        logging.StreamHandler.__init__(self, stream)
        if formatter:
            self.setFormatter(formatter)  # type: ignore
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        logging.StreamHandler.emit(self, record)
    
    def close(self) -> None:
        """Close the handler."""
        logging.StreamHandler.close(self)


class FileHandler(LogHandler, logging.FileHandler):
    """Handler that writes to a file."""
    
    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str = "utf-8",
        formatter: Optional[LogFormatter] = None,
    ) -> None:
        # Create parent directories if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        logging.FileHandler.__init__(
            self, filename, mode=mode, encoding=encoding
        )
        if formatter:
            self.setFormatter(formatter)  # type: ignore
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        logging.FileHandler.emit(self, record)
    
    def close(self) -> None:
        """Close the handler."""
        logging.FileHandler.close(self)


class RotatingFileHandler(LogHandler, _RotatingFileHandler):
    """Handler that rotates log files based on size."""
    
    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        encoding: str = "utf-8",
        formatter: Optional[LogFormatter] = None,
    ) -> None:
        # Create parent directories if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        _RotatingFileHandler.__init__(
            self,
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        if formatter:
            self.setFormatter(formatter)  # type: ignore
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        _RotatingFileHandler.emit(self, record)
    
    def close(self) -> None:
        """Close the handler."""
        _RotatingFileHandler.close(self)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that ensures correlation ID is present.
    
    Automatically generates a correlation ID if one is not
    already set in the log context.
    """
    
    def __init__(self, auto_generate: bool = True) -> None:
        super().__init__()
        self.auto_generate = auto_generate
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Ensure correlation ID is set."""
        if self.auto_generate and not LogContext.get_correlation_id():
            LogContext.set_correlation_id()
        
        # Add correlation_id to record for compatibility
        record.correlation_id = LogContext.get_correlation_id() or ""  # type: ignore
        
        return True


class StructuredLogger:
    """
    Main structured logger for agent memory systems.
    
    Provides structured logging with automatic context propagation,
    multiple output formats, and integration with observability tools.
    
    Example:
        >>> logger = StructuredLogger(
        ...     name="agent_memory",
        ...     config=LogConfig(format="json"),
        ... )
        >>> 
        >>> # Basic logging
        >>> logger.info("Memory added", memory_id="abc123", size=1024)
        >>> 
        >>> # With context scope
        >>> with LogContext.scope(request_id="req-001", user="alice"):
        ...     logger.info("Processing request")
        ...     logger.debug("Cache lookup", cache_key="key1")
        >>> 
        >>> # Structured error logging
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     logger.error("Operation failed", exc_info=True, error_code="E001")
    """
    
    def __init__(
        self,
        name: str = "agent_memory_toolkit",
        config: Optional[LogConfig] = None,
        handlers: Optional[List[logging.Handler]] = None,
    ) -> None:
        self.name = name
        self.config = config or LogConfig()
        
        # Get or create logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.config.level)
        
        # Clear existing handlers
        self._logger.handlers = []
        
        # Add correlation ID filter
        self._logger.addFilter(CorrelationIdFilter())
        
        # Setup handlers
        if handlers:
            for handler in handlers:
                self._logger.addHandler(handler)
        else:
            self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default handlers based on config."""
        # Console handler
        if self.config.format == "json":
            formatter = JSONFormatter(self.config)
        else:
            formatter = ConsoleFormatter(self.config)
        
        console_handler = StreamHandler(sys.stdout, formatter)
        console_handler.setLevel(self.config.level)
        self._logger.addHandler(console_handler)
        
        # File handler if configured
        if self.config.output_file:
            file_handler = RotatingFileHandler(
                filename=self.config.output_file,
                max_bytes=self.config.max_file_size_bytes,
                backup_count=self.config.backup_count,
                formatter=JSONFormatter(self.config),  # Always JSON for files
            )
            file_handler.setLevel(self.config.level)
            self._logger.addHandler(file_handler)
    
    def _log(
        self,
        level: int,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Internal logging method that adds extra fields."""
        # Create extra dict with kwargs
        extra = dict(kwargs)
        
        # Add context fields
        context = LogContext.get()
        for key, value in context.items():
            if key not in extra:
                extra[key] = value
        
        # Log with extra fields
        self._logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            extra=extra,
        )
    
    def debug(
        self, msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, msg, *args, **kwargs)
    
    def info(
        self, msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, msg, *args, **kwargs)
    
    def warning(
        self, msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, msg, *args, **kwargs)
    
    def error(
        self, msg: str, *args: Any, exc_info: Any = None, **kwargs: Any
    ) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, msg, *args, exc_info=exc_info, **kwargs)
    
    def critical(
        self, msg: str, *args: Any, exc_info: Any = None, **kwargs: Any
    ) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)
    
    def exception(
        self, msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log an exception with traceback."""
        self._log(LogLevel.ERROR, msg, *args, exc_info=True, **kwargs)
    
    def bind(self, **kwargs: Any) -> "BoundLogger":
        """
        Create a bound logger with preset fields.
        
        Example:
            >>> request_logger = logger.bind(request_id="req-001")
            >>> request_logger.info("Processing")  # includes request_id
        """
        return BoundLogger(self, kwargs)
    
    def with_context(self, **kwargs: Any) -> LogContextScope:
        """
        Create a context scope with preset fields.
        
        Convenience method for LogContext.scope().
        """
        return LogContext.scope(**kwargs)
    
    def get_child(self, suffix: str) -> "StructuredLogger":
        """Get a child logger with the given suffix."""
        return StructuredLogger(
            name=f"{self.name}.{suffix}",
            config=self.config,
        )
    
    def set_level(self, level: Union[LogLevel, int]) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)
    
    def add_handler(self, handler: logging.Handler) -> None:
        """Add a handler to the logger."""
        self._logger.addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove a handler from the logger."""
        self._logger.removeHandler(handler)
    
    def shutdown(self) -> None:
        """Shutdown the logger and close all handlers."""
        for handler in self._logger.handlers:
            handler.close()
        self._logger.handlers = []


class BoundLogger:
    """
    A logger with bound context fields.
    
    Fields are automatically included in every log message.
    """
    
    def __init__(
        self,
        logger: StructuredLogger,
        fields: Dict[str, Any],
    ) -> None:
        self._logger = logger
        self._fields = fields
    
    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge bound fields with kwargs."""
        merged = dict(self._fields)
        merged.update(kwargs)
        return merged
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **self._merge_kwargs(kwargs))
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **self._merge_kwargs(kwargs))
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **self._merge_kwargs(kwargs))
    
    def error(
        self, msg: str, *args: Any, exc_info: Any = None, **kwargs: Any
    ) -> None:
        self._logger.error(
            msg, *args, exc_info=exc_info, **self._merge_kwargs(kwargs)
        )
    
    def critical(
        self, msg: str, *args: Any, exc_info: Any = None, **kwargs: Any
    ) -> None:
        self._logger.critical(
            msg, *args, exc_info=exc_info, **self._merge_kwargs(kwargs)
        )
    
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **self._merge_kwargs(kwargs))
    
    def bind(self, **kwargs: Any) -> "BoundLogger":
        """Create a new bound logger with additional fields."""
        merged = dict(self._fields)
        merged.update(kwargs)
        return BoundLogger(self._logger, merged)


# Module-level convenience function to get a logger
def get_logger(
    name: str = "agent_memory_toolkit",
    config: Optional[LogConfig] = None,
) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    This is the recommended way to get a logger instance.
    
    Example:
        >>> from agent_memory_toolkit.observability.logging import get_logger
        >>> logger = get_logger("my_agent")
        >>> logger.info("Agent started", version="1.0.0")
    """
    return StructuredLogger(name=name, config=config)
