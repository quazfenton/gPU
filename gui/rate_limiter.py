"""Rate limiting middleware for GUI application.

This module provides rate limiting functionality to prevent abuse of the GUI
by limiting the number of requests per user/IP within a time window.

Requirements:
    - 12.5: Implement rate limiting to prevent abuse
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import defaultdict
import threading
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_minute: Maximum number of requests allowed per minute (default: 60)
        requests_per_hour: Maximum number of requests allowed per hour (default: 1000)
        enabled: Whether rate limiting is enabled (default: True)
        cleanup_interval: Interval in seconds to clean up old entries (default: 300)
    """
    requests_per_minute: int = field(default=60)
    requests_per_hour: int = field(default=1000)
    enabled: bool = field(default=True)
    cleanup_interval: int = field(default=300)  # 5 minutes


@dataclass
class RequestRecord:
    """Record of requests for a client.
    
    Attributes:
        minute_requests: List of timestamps for requests in the current minute
        hour_requests: List of timestamps for requests in the current hour
    """
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded.
    
    Attributes:
        message: Error message
        retry_after: Number of seconds to wait before retrying
        limit_type: Type of limit exceeded ('minute' or 'hour')
    """
    
    def __init__(self, message: str, retry_after: int, limit_type: str):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class RateLimiter(LoggerMixin):
    """Rate limiter for tracking and enforcing request limits.
    
    This class tracks requests per user/IP and enforces configurable rate limits.
    It supports both per-minute and per-hour limits.
    
    Requirements:
        - 12.5: Implement rate limiting to prevent abuse
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize the rate limiter.
        
        Args:
            config: Rate limit configuration (uses defaults if not provided)
        """
        self.config = config or RateLimitConfig()
        self._records: Dict[str, RequestRecord] = defaultdict(RequestRecord)
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._cleanup_running = False
        
        if self.config.enabled:
            self._start_cleanup_thread()
            self.logger.info(
                f"Rate limiter initialized: {self.config.requests_per_minute} req/min, "
                f"{self.config.requests_per_hour} req/hour"
            )
        else:
            self.logger.info("Rate limiter disabled")
    
    def check_rate_limit(self, client_id: str) -> None:
        """Check if a client has exceeded rate limits.
        
        This method checks both per-minute and per-hour limits. If either limit
        is exceeded, it raises a RateLimitError.
        
        Args:
            client_id: Unique identifier for the client (e.g., username or IP address)
            
        Raises:
            RateLimitError: If rate limit is exceeded
            
        Requirements:
            - 12.5: Implement rate limit tracking per user/IP
            - 12.5: Return rate limit errors when exceeded
        """
        if not self.config.enabled:
            return
        
        current_time = time.time()
        
        with self._lock:
            record = self._records[client_id]
            
            # Clean up old requests
            self._cleanup_old_requests(record, current_time)
            
            # Check per-minute limit
            if len(record.minute_requests) >= self.config.requests_per_minute:
                oldest_request = record.minute_requests[0]
                retry_after = int(60 - (current_time - oldest_request)) + 1
                
                self.logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{len(record.minute_requests)} requests in last minute "
                    f"(limit: {self.config.requests_per_minute})"
                )
                
                raise RateLimitError(
                    f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute. "
                    f"Please try again in {retry_after} seconds.",
                    retry_after=retry_after,
                    limit_type='minute'
                )
            
            # Check per-hour limit
            if len(record.hour_requests) >= self.config.requests_per_hour:
                oldest_request = record.hour_requests[0]
                retry_after = int(3600 - (current_time - oldest_request)) + 1
                
                self.logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{len(record.hour_requests)} requests in last hour "
                    f"(limit: {self.config.requests_per_hour})"
                )
                
                raise RateLimitError(
                    f"Rate limit exceeded: {self.config.requests_per_hour} requests per hour. "
                    f"Please try again in {retry_after} seconds.",
                    retry_after=retry_after,
                    limit_type='hour'
                )
            
            # Record this request
            record.minute_requests.append(current_time)
            record.hour_requests.append(current_time)
    
    def _cleanup_old_requests(self, record: RequestRecord, current_time: float) -> None:
        """Remove requests older than the time windows.
        
        Args:
            record: Request record to clean up
            current_time: Current timestamp
        """
        # Remove requests older than 1 minute
        minute_cutoff = current_time - 60
        record.minute_requests = [
            ts for ts in record.minute_requests if ts > minute_cutoff
        ]
        
        # Remove requests older than 1 hour
        hour_cutoff = current_time - 3600
        record.hour_requests = [
            ts for ts in record.hour_requests if ts > hour_cutoff
        ]
    
    def get_remaining_requests(self, client_id: str) -> Tuple[int, int]:
        """Get the number of remaining requests for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Tuple of (remaining_per_minute, remaining_per_hour)
        """
        if not self.config.enabled:
            return (self.config.requests_per_minute, self.config.requests_per_hour)
        
        current_time = time.time()
        
        with self._lock:
            record = self._records[client_id]
            self._cleanup_old_requests(record, current_time)
            
            remaining_minute = max(0, self.config.requests_per_minute - len(record.minute_requests))
            remaining_hour = max(0, self.config.requests_per_hour - len(record.hour_requests))
            
            return (remaining_minute, remaining_hour)
    
    def reset_client(self, client_id: str) -> None:
        """Reset rate limit tracking for a specific client.
        
        This can be used by administrators to clear rate limits for a client.
        
        Args:
            client_id: Unique identifier for the client
        """
        with self._lock:
            if client_id in self._records:
                del self._records[client_id]
                self.logger.info(f"Rate limit reset for client: {client_id}")
    
    def get_statistics(self) -> dict:
        """Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics including:
            - total_clients: Number of tracked clients
            - enabled: Whether rate limiting is enabled
            - config: Rate limit configuration
        """
        with self._lock:
            return {
                'total_clients': len(self._records),
                'enabled': self.config.enabled,
                'config': {
                    'requests_per_minute': self.config.requests_per_minute,
                    'requests_per_hour': self.config.requests_per_hour
                }
            }
    
    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self.logger.debug("Rate limiter cleanup thread started")
    
    def _stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        self.logger.debug("Rate limiter cleanup thread stopped")
    
    def _cleanup_loop(self) -> None:
        """Background thread that periodically cleans up old records."""
        while self._cleanup_running:
            try:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_all_records()
            except Exception as e:
                self.logger.error(f"Error in rate limiter cleanup thread: {e}")
    
    def _cleanup_all_records(self) -> None:
        """Clean up old records for all clients."""
        current_time = time.time()
        
        with self._lock:
            # Remove clients with no recent requests
            clients_to_remove = []
            
            for client_id, record in self._records.items():
                self._cleanup_old_requests(record, current_time)
                
                # If no requests in the last hour, remove the client
                if not record.hour_requests:
                    clients_to_remove.append(client_id)
            
            for client_id in clients_to_remove:
                del self._records[client_id]
            
            if clients_to_remove:
                self.logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    def shutdown(self) -> None:
        """Shutdown the rate limiter and cleanup resources."""
        self.logger.info("Shutting down rate limiter")
        self._stop_cleanup_thread()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup


def create_rate_limiter_decorator(rate_limiter: RateLimiter):
    """Create a decorator for rate limiting function calls.
    
    This decorator can be used to wrap functions that should be rate limited.
    The client_id is extracted from the function arguments.
    
    Args:
        rate_limiter: RateLimiter instance to use
        
    Returns:
        Decorator function
        
    Example:
        @create_rate_limiter_decorator(rate_limiter)
        def submit_job(client_id: str, template: str, inputs: dict):
            # Function implementation
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract client_id from arguments
            client_id = kwargs.get('client_id')
            if not client_id and args:
                # Try first argument as client_id
                client_id = str(args[0]) if args else 'unknown'
            
            # Check rate limit
            try:
                rate_limiter.check_rate_limit(client_id)
            except RateLimitError as e:
                # Re-raise with additional context
                raise
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
