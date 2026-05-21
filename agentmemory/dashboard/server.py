"""
Dashboard HTTP server for serving the analytics dashboard.

Uses Python's built-in HTTP server with minimal dependencies.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .analytics import AnalyticsEngine


@dataclass
class DashboardConfig:
    """Configuration for the dashboard server."""
    host: str = "127.0.0.1"
    port: int = 8080
    db_path: str = "agent_memory.db"
    search_log_path: Optional[str] = None
    auto_open: bool = True
    cors_enabled: bool = True


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    analytics_engine: Optional[AnalyticsEngine] = None
    cors_enabled: bool = True
    static_dir: str = ""
    
    def __init__(self, *args, **kwargs):
        # Set the directory for static files
        if self.static_dir:
            kwargs['directory'] = self.static_dir
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def _send_cors_headers(self):
        """Send CORS headers if enabled."""
        if self.cors_enabled:
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API routes
        if path.startswith('/api/'):
            self._handle_api(path, parsed.query)
            return
        
        # Serve index.html for root
        if path == '/' or path == '':
            self.path = '/index.html'
        
        # Serve static files
        try:
            super().do_GET()
        except Exception as e:
            self.send_error(500, str(e))
    
    def _handle_api(self, path: str, query: str):
        """Handle API requests."""
        params = parse_qs(query)
        
        try:
            if path == '/api/stats':
                data = self._get_all_stats(params)
            elif path == '/api/memories':
                data = self._get_memory_stats(params)
            elif path == '/api/domains':
                data = self._get_domain_distribution()
            elif path == '/api/searches':
                data = self._get_search_trends(params)
            elif path == '/api/storage':
                data = self._get_storage_metrics(params)
            elif path == '/api/branches':
                data = self._get_branch_comparison()
            else:
                self.send_error(404, "API endpoint not found")
                return
            
            self._send_json(data)
            
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
    
    def _get_days_param(self, params: Dict[str, Any]) -> int:
        """Extract days parameter from query string."""
        days_list = params.get('days', ['30'])
        try:
            return int(days_list[0])
        except (ValueError, IndexError):
            return 30
    
    def _get_all_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get all analytics data."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        days = self._get_days_param(params)
        return self.analytics_engine.get_all_analytics(days)
    
    def _get_memory_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        days = self._get_days_param(params)
        return self.analytics_engine.get_memory_stats(days).to_dict()
    
    def _get_domain_distribution(self) -> Dict[str, Any]:
        """Get domain distribution."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        return self.analytics_engine.get_domain_distribution().to_dict()
    
    def _get_search_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get search trends."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        days = self._get_days_param(params)
        return self.analytics_engine.get_search_trends(days).to_dict()
    
    def _get_storage_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get storage metrics."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        days = self._get_days_param(params)
        return self.analytics_engine.get_storage_metrics(days).to_dict()
    
    def _get_branch_comparison(self) -> Dict[str, Any]:
        """Get branch comparison."""
        if not self.analytics_engine:
            return {"error": "Analytics engine not initialized"}
        return self.analytics_engine.get_branch_comparison().to_dict()
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format: str, *args):
        """Override to reduce log noise."""
        pass


class DashboardServer:
    """
    HTTP server for the analytics dashboard.
    
    Example:
        >>> server = DashboardServer(DashboardConfig(db_path="memory.db"))
        >>> server.start()  # Opens browser automatically
        >>> # Press Ctrl+C to stop
        >>> server.stop()
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize the dashboard server.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        
        # Initialize analytics engine
        self.analytics_engine = AnalyticsEngine(
            db_path=config.db_path,
            search_log_path=config.search_log_path,
        )
        
        # Get static files directory
        self.static_dir = str(Path(__file__).parent / "static")
    
    def _create_handler_class(self):
        """Create a handler class with configured attributes."""
        analytics_engine = self.analytics_engine
        cors_enabled = self.config.cors_enabled
        static_dir = self.static_dir
        
        class ConfiguredHandler(DashboardHandler):
            pass
        
        ConfiguredHandler.analytics_engine = analytics_engine
        ConfiguredHandler.cors_enabled = cors_enabled
        ConfiguredHandler.static_dir = static_dir
        
        return ConfiguredHandler
    
    def start(self, blocking: bool = True):
        """
        Start the dashboard server.
        
        Args:
            blocking: If True, block until server is stopped.
                     If False, run in background thread.
        """
        handler_class = self._create_handler_class()
        
        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(
            (self.config.host, self.config.port),
            handler_class
        )
        
        self._running = True
        url = f"http://{self.config.host}:{self.config.port}"
        
        print(f"Dashboard server started at {url}")
        print("Press Ctrl+C to stop")
        
        # Open browser if configured
        if self.config.auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass  # Ignore browser open errors
        
        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down...")
            finally:
                self.stop()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop the dashboard server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    def get_url(self) -> str:
        """Get the server URL."""
        return f"http://{self.config.host}:{self.config.port}"


def run_dashboard(
    db_path: str = "agent_memory.db",
    host: str = "127.0.0.1",
    port: int = 8080,
    auto_open: bool = True,
):
    """
    Run the dashboard server.
    
    Convenience function for starting the dashboard.
    
    Args:
        db_path: Path to the memory database
        host: Host to bind to
        port: Port to bind to
        auto_open: Whether to open browser automatically
    """
    config = DashboardConfig(
        db_path=db_path,
        host=host,
        port=port,
        auto_open=auto_open,
    )
    server = DashboardServer(config)
    server.start(blocking=True)
