"""Tests for rate limiting middleware.

Requirements:
    - 12.5: Implement rate limiting to prevent abuse
"""

import pytest
import time
from gui.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitError,
    RequestRecord,
    create_rate_limiter_decorator
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.enabled is True
        assert config.cleanup_interval == 300
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            enabled=False,
            cleanup_interval=60
        )
        
        assert config.requests_per_minute == 10
        assert config.requests_per_hour == 100
        assert config.enabled is False
        assert config.cleanup_interval == 60


class TestRequestRecord:
    """Tests for RequestRecord."""
    
    def test_default_record(self):
        """Test default request record."""
        record = RequestRecord()
        
        assert record.minute_requests == []
        assert record.hour_requests == []
    
    def test_record_with_requests(self):
        """Test request record with timestamps."""
        current_time = time.time()
        record = RequestRecord(
            minute_requests=[current_time - 30, current_time - 10],
            hour_requests=[current_time - 1800, current_time - 900, current_time - 30]
        )
        
        assert len(record.minute_requests) == 2
        assert len(record.hour_requests) == 3


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    def test_initialization_enabled(self):
        """Test rate limiter initialization with rate limiting enabled."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        assert limiter.config.enabled is True
        assert limiter.config.requests_per_minute == 10
        assert limiter.config.requests_per_hour == 100
    
    def test_initialization_disabled(self):
        """Test rate limiter initialization with rate limiting disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        assert limiter.config.enabled is False
    
    def test_check_rate_limit_disabled(self):
        """Test that rate limiting is bypassed when disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        # Should not raise even with many requests
        for _ in range(200):
            limiter.check_rate_limit("client1")
    
    def test_check_rate_limit_within_limits(self):
        """Test that requests within limits are allowed."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Make 5 requests (within limit)
        for _ in range(5):
            limiter.check_rate_limit("client1")
        
        # Should not raise
    
    def test_check_rate_limit_exceeds_minute_limit(self):
        """Test that exceeding per-minute limit raises error."""
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Make 5 requests (at limit)
        for _ in range(5):
            limiter.check_rate_limit("client1")
        
        # 6th request should fail
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_rate_limit("client1")
        
        error = exc_info.value
        assert error.limit_type == 'minute'
        assert error.retry_after > 0
        assert "per minute" in str(error)
    
    def test_check_rate_limit_exceeds_hour_limit(self):
        """Test that exceeding per-hour limit raises error."""
        config = RateLimitConfig(requests_per_minute=100, requests_per_hour=10)
        limiter = RateLimiter(config)
        
        # Make 10 requests (at limit)
        for _ in range(10):
            limiter.check_rate_limit("client1")
        
        # 11th request should fail
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_rate_limit("client1")
        
        error = exc_info.value
        assert error.limit_type == 'hour'
        assert error.retry_after > 0
        assert "per hour" in str(error)
    
    def test_check_rate_limit_different_clients(self):
        """Test that different clients have independent rate limits."""
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Client 1 makes 5 requests
        for _ in range(5):
            limiter.check_rate_limit("client1")
        
        # Client 2 should still be able to make requests
        for _ in range(5):
            limiter.check_rate_limit("client2")
        
        # Both clients should now be at limit
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client1")
        
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client2")
    
    def test_get_remaining_requests(self):
        """Test getting remaining request counts."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Initially, all requests available
        remaining_min, remaining_hour = limiter.get_remaining_requests("client1")
        assert remaining_min == 10
        assert remaining_hour == 100
        
        # Make 3 requests
        for _ in range(3):
            limiter.check_rate_limit("client1")
        
        # Check remaining
        remaining_min, remaining_hour = limiter.get_remaining_requests("client1")
        assert remaining_min == 7
        assert remaining_hour == 97
    
    def test_get_remaining_requests_disabled(self):
        """Test getting remaining requests when rate limiting is disabled."""
        config = RateLimitConfig(enabled=False, requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Should return configured limits even when disabled
        remaining_min, remaining_hour = limiter.get_remaining_requests("client1")
        assert remaining_min == 10
        assert remaining_hour == 100
    
    def test_reset_client(self):
        """Test resetting rate limit for a client."""
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Make 5 requests (at limit)
        for _ in range(5):
            limiter.check_rate_limit("client1")
        
        # Should be at limit
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client1")
        
        # Reset client
        limiter.reset_client("client1")
        
        # Should be able to make requests again
        limiter.check_rate_limit("client1")
    
    def test_get_statistics(self):
        """Test getting rate limiter statistics."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Make requests from different clients
        limiter.check_rate_limit("client1")
        limiter.check_rate_limit("client2")
        limiter.check_rate_limit("client3")
        
        stats = limiter.get_statistics()
        
        assert stats['total_clients'] == 3
        assert stats['enabled'] is True
        assert stats['config']['requests_per_minute'] == 10
        assert stats['config']['requests_per_hour'] == 100
    
    def test_cleanup_old_requests(self):
        """Test that old requests are cleaned up."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Manually add old requests
        current_time = time.time()
        record = limiter._records["client1"]
        
        # Add requests older than 1 minute
        record.minute_requests = [current_time - 120, current_time - 90]
        # Add requests older than 1 hour
        record.hour_requests = [current_time - 7200, current_time - 3700]
        
        # Cleanup
        limiter._cleanup_old_requests(record, current_time)
        
        # Old requests should be removed
        assert len(record.minute_requests) == 0
        assert len(record.hour_requests) == 0
    
    def test_cleanup_keeps_recent_requests(self):
        """Test that recent requests are kept during cleanup."""
        config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Manually add recent requests
        current_time = time.time()
        record = limiter._records["client1"]
        
        # Add recent requests
        record.minute_requests = [current_time - 30, current_time - 10]
        record.hour_requests = [current_time - 1800, current_time - 900, current_time - 30]
        
        # Cleanup
        limiter._cleanup_old_requests(record, current_time)
        
        # Recent requests should be kept
        assert len(record.minute_requests) == 2
        assert len(record.hour_requests) == 3


class TestRateLimiterDecorator:
    """Tests for rate limiter decorator."""
    
    def test_decorator_with_client_id_kwarg(self):
        """Test decorator with client_id as keyword argument."""
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)
        decorator = create_rate_limiter_decorator(limiter)
        
        @decorator
        def test_function(client_id: str, value: int):
            return value * 2
        
        # Should work within limits
        result = test_function(client_id="client1", value=5)
        assert result == 10
        
        # Make more requests
        for _ in range(4):
            test_function(client_id="client1", value=5)
        
        # Should fail on 6th request
        with pytest.raises(RateLimitError):
            test_function(client_id="client1", value=5)
    
    def test_decorator_with_client_id_arg(self):
        """Test decorator with client_id as positional argument."""
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)
        decorator = create_rate_limiter_decorator(limiter)
        
        @decorator
        def test_function(client_id: str, value: int):
            return value * 2
        
        # Should work within limits
        result = test_function("client1", 5)
        assert result == 10
        
        # Make more requests
        for _ in range(4):
            test_function("client1", 5)
        
        # Should fail on 6th request
        with pytest.raises(RateLimitError):
            test_function("client1", 5)
    
    def test_decorator_disabled(self):
        """Test decorator when rate limiting is disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        decorator = create_rate_limiter_decorator(limiter)
        
        @decorator
        def test_function(client_id: str, value: int):
            return value * 2
        
        # Should work even with many requests
        for _ in range(100):
            result = test_function(client_id="client1", value=5)
            assert result == 10


class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""
    
    def test_rate_limit_error_attributes(self):
        """Test that RateLimitError has correct attributes."""
        config = RateLimitConfig(requests_per_minute=2, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Make 2 requests
        limiter.check_rate_limit("client1")
        limiter.check_rate_limit("client1")
        
        # 3rd request should fail
        try:
            limiter.check_rate_limit("client1")
            assert False, "Should have raised RateLimitError"
        except RateLimitError as e:
            assert hasattr(e, 'retry_after')
            assert hasattr(e, 'limit_type')
            assert e.retry_after > 0
            assert e.limit_type in ['minute', 'hour']
            assert len(str(e)) > 0
    
    def test_concurrent_clients_isolation(self):
        """Test that concurrent clients don't interfere with each other."""
        config = RateLimitConfig(requests_per_minute=3, requests_per_hour=100)
        limiter = RateLimiter(config)
        
        # Client 1 makes 3 requests
        for _ in range(3):
            limiter.check_rate_limit("client1")
        
        # Client 2 makes 3 requests
        for _ in range(3):
            limiter.check_rate_limit("client2")
        
        # Client 3 makes 3 requests
        for _ in range(3):
            limiter.check_rate_limit("client3")
        
        # All clients should now be at limit
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client1")
        
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client2")
        
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("client3")
        
        # But a new client should still work
        limiter.check_rate_limit("client4")
