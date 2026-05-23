"""
Background process for memory consolidation.

Provides scheduled consolidation runs with:
- Configurable intervals
- Process management (start/stop/status)
- Progress tracking
- Event callbacks
"""

import logging
import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import json
import os
from pathlib import Path

from .models import ConsolidationConfig, ConsolidationResult
from .consolidator import MemoryConsolidator
from .similarity import MemoryData

logger = logging.getLogger(__name__)


class ConsolidationScheduler:
    """
    Background scheduler for memory consolidation.
    
    Features:
    - Configurable run intervals
    - Graceful start/stop
    - Progress and status tracking
    - Event callbacks
    - State persistence
    """
    
    def __init__(
        self,
        consolidator: Optional[MemoryConsolidator] = None,
        config: Optional[ConsolidationConfig] = None,
        memory_loader: Optional[Callable[[], list[MemoryData]]] = None,
        state_file: Optional[str] = None,
    ):
        """
        Initialize scheduler.
        
        Args:
            consolidator: MemoryConsolidator instance
            config: ConsolidationConfig
            memory_loader: Function to load memories for consolidation
            state_file: Path to persist scheduler state
        """
        self.config = config or ConsolidationConfig()
        self.consolidator = consolidator or MemoryConsolidator(self.config)
        self.memory_loader = memory_loader
        self.state_file = state_file
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_run: Optional[datetime] = None
        self._next_run: Optional[datetime] = None
        self._current_result: Optional[ConsolidationResult] = None
        self._run_history: list[dict[str, Any]] = []
        self._max_history = 100
        
        # Callbacks
        self._on_start: Optional[Callable[[], None]] = None
        self._on_complete: Optional[Callable[[ConsolidationResult], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None
        self._on_progress: Optional[Callable[[str, int, int], None]] = None
        
        # Load state if exists
        self._load_state()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run
    
    @property
    def next_run(self) -> Optional[datetime]:
        return self._next_run
    
    def on_start(self, callback: Callable[[], None]) -> None:
        """Set callback for when consolidation starts."""
        self._on_start = callback
    
    def on_complete(self, callback: Callable[[ConsolidationResult], None]) -> None:
        """Set callback for when consolidation completes."""
        self._on_complete = callback
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for when an error occurs."""
        self._on_error = callback
    
    def on_progress(self, callback: Callable[[str, int, int], None]) -> None:
        """Set callback for progress updates."""
        self._on_progress = callback
    
    def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ConsolidationScheduler",
            daemon=True,
        )
        self._thread.start()
        
        logger.info("Consolidation scheduler started")
    
    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the background scheduler.
        
        Args:
            wait: Wait for current operation to complete
            timeout: Maximum time to wait in seconds
        """
        if not self._running:
            return
        
        logger.info("Stopping consolidation scheduler...")
        self._stop_event.set()
        self._running = False
        
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        
        self._save_state()
        logger.info("Consolidation scheduler stopped")
    
    def run_now(self) -> Optional[ConsolidationResult]:
        """Run consolidation immediately."""
        return self._run_consolidation()
    
    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status."""
        return {
            "is_running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": self._next_run.isoformat() if self._next_run else None,
            "run_interval_hours": self.config.run_interval_hours,
            "current_result": (
                self._current_result.to_dict() if self._current_result else None
            ),
            "total_runs": len(self._run_history),
        }
    
    def get_history(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get run history."""
        end = len(self._run_history) - offset
        start = max(0, end - limit)
        return self._run_history[start:end][::-1]
    
    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                # Check if it's time to run
                now = datetime.utcnow()
                
                if self._should_run(now):
                    self._run_consolidation()
                
                # Calculate next run time
                self._next_run = self._last_run + timedelta(
                    hours=self.config.run_interval_hours
                ) if self._last_run else now
                
                # Wait for next check (check every minute)
                self._stop_event.wait(timeout=60)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                if self._on_error:
                    self._on_error(e)
                # Wait before retrying
                self._stop_event.wait(timeout=300)
    
    def _should_run(self, now: datetime) -> bool:
        """Check if consolidation should run now."""
        if not self._last_run:
            return True
        
        elapsed = now - self._last_run
        return elapsed >= timedelta(hours=self.config.run_interval_hours)
    
    def _run_consolidation(self) -> Optional[ConsolidationResult]:
        """Execute consolidation."""
        if not self.memory_loader:
            logger.error("No memory loader configured")
            return None
        
        try:
            if self._on_start:
                self._on_start()
            
            logger.info("Starting scheduled consolidation")
            
            # Load memories
            memories = self.memory_loader()
            
            # Run consolidation
            result = self.consolidator.consolidate(
                memories,
                progress_callback=self._on_progress,
            )
            
            # Update state
            self._last_run = datetime.utcnow()
            self._current_result = result
            
            # Add to history
            self._run_history.append({
                "run_id": result.run_id,
                "timestamp": result.completed_at.isoformat(),
                "memories_analyzed": result.memories_analyzed,
                "duplicates_removed": result.duplicates_removed,
                "conflicts_resolved": result.conflicts_resolved,
                "success": result.success,
            })
            
            # Trim history
            if len(self._run_history) > self._max_history:
                self._run_history = self._run_history[-self._max_history:]
            
            # Save state
            self._save_state()
            
            if self._on_complete:
                self._on_complete(result)
            
            logger.info(f"Consolidation complete: {result.summary}")
            
            return result
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            if self._on_error:
                self._on_error(e)
            return None
    
    def _save_state(self) -> None:
        """Save scheduler state to file."""
        if not self.state_file:
            return
        
        try:
            state = {
                "last_run": self._last_run.isoformat() if self._last_run else None,
                "run_history": self._run_history,
            }
            
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load scheduler state from file."""
        if not self.state_file or not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            if state.get("last_run"):
                self._last_run = datetime.fromisoformat(state["last_run"])
            
            self._run_history = state.get("run_history", [])
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


class ConsolidationDaemon:
    """
    Daemon process for running consolidation as a service.
    
    Handles:
    - Signal handling (SIGTERM, SIGINT)
    - PID file management
    - Logging to file
    """
    
    def __init__(
        self,
        scheduler: ConsolidationScheduler,
        pid_file: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize daemon.
        
        Args:
            scheduler: ConsolidationScheduler instance
            pid_file: Path to PID file
            log_file: Path to log file
        """
        self.scheduler = scheduler
        self.pid_file = pid_file
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for daemon mode."""
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logging.getLogger().addHandler(handler)
    
    def _write_pid(self) -> None:
        """Write PID to file."""
        if self.pid_file:
            Path(self.pid_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, "w") as f:
                f.write(str(os.getpid()))
    
    def _remove_pid(self) -> None:
        """Remove PID file."""
        if self.pid_file and os.path.exists(self.pid_file):
            os.remove(self.pid_file)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.scheduler.stop()
        self._remove_pid()
    
    def run(self) -> None:
        """Run the daemon."""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Write PID file
        self._write_pid()
        
        try:
            logger.info("Starting consolidation daemon")
            self.scheduler.start()
            
            # Keep main thread alive
            while self.scheduler.is_running:
                time.sleep(1)
                
        finally:
            self._remove_pid()
            logger.info("Consolidation daemon stopped")
    
    @classmethod
    def is_running(cls, pid_file: str) -> bool:
        """Check if daemon is running."""
        if not os.path.exists(pid_file):
            return False
        
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            os.kill(pid, 0)
            return True
            
        except (ValueError, OSError):
            return False
    
    @classmethod
    def stop_daemon(cls, pid_file: str) -> bool:
        """Stop running daemon."""
        if not os.path.exists(pid_file):
            return False
        
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to stop
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            
            return True
            
        except (ValueError, OSError) as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False
