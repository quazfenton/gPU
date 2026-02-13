# Rate Limiting in GUI

The GUI implements rate limiting to prevent abuse and ensure fair resource usage across all clients.

## Configuration

Rate limiting can be configured through environment variables or the GUIConfig:

### Environment Variables

```bash
# Enable/disable rate limiting (default: true)
GUI_ENABLE_RATE_LIMITING=true

# Maximum requests per minute per client (default: 60)
GUI_RATE_LIMIT_PER_MINUTE=60

# Maximum requests per hour per client (default: 1000)
GUI_RATE_LIMIT_PER_HOUR=1000
```

### Configuration File

```env
GUI_ENABLE_RATE_LIMITING=true
GUI_RATE_LIMIT_PER_MINUTE=60
GUI_RATE_LIMIT_PER_HOUR=1000
```

### Programmatic Configuration

```python
from gui.config import GUIConfig

config = GUIConfig(
    enable_rate_limiting=True,
    rate_limit_per_minute=60,
    rate_limit_per_hour=1000
)
```

## How It Works

The rate limiter tracks requests per client (identified by username if authenticated, or IP address otherwise) and enforces two limits:

1. **Per-Minute Limit**: Maximum number of requests allowed within a 60-second sliding window
2. **Per-Hour Limit**: Maximum number of requests allowed within a 3600-second sliding window

When a client exceeds either limit, subsequent requests are rejected with a `RateLimitError` that includes:
- A user-friendly error message
- The number of seconds to wait before retrying (`retry_after`)
- The type of limit that was exceeded (`minute` or `hour`)

## Client Identification

Clients are identified using the following priority:

1. **Authenticated Username**: If authentication is enabled, the username is used
2. **IP Address**: If available from the request, the client's IP address is used
3. **Default Identifier**: A fallback identifier for cases where neither is available

## Rate Limit Tracking

The rate limiter maintains a sliding window of request timestamps for each client:

- Requests older than 1 minute are automatically removed from the per-minute tracking
- Requests older than 1 hour are automatically removed from the per-hour tracking
- Inactive clients (no requests in the last hour) are periodically cleaned up

## Usage in Service Methods

Service methods can check rate limits before processing requests:

```python
from gui.rate_limiter import RateLimitError

def submit_job(self, client_id: str, template: str, inputs: dict):
    """Submit a job with rate limiting."""
    try:
        # Check rate limit
        self.app.check_rate_limit(client_id)
        
        # Process the request
        job_id = self.job_queue.submit(template, inputs)
        return job_id
        
    except RateLimitError as e:
        # Handle rate limit exceeded
        return {
            'error': str(e),
            'retry_after': e.retry_after,
            'limit_type': e.limit_type
        }
```

## Decorator Pattern

For convenience, a decorator is provided to automatically apply rate limiting:

```python
from gui.rate_limiter import create_rate_limiter_decorator

# Create decorator with rate limiter instance
rate_limit = create_rate_limiter_decorator(rate_limiter)

@rate_limit
def my_function(client_id: str, value: int):
    """Function with automatic rate limiting."""
    return value * 2
```

## Monitoring

Get rate limiter statistics:

```python
stats = rate_limiter.get_statistics()
# Returns:
# {
#     'total_clients': 5,
#     'enabled': True,
#     'config': {
#         'requests_per_minute': 60,
#         'requests_per_hour': 1000
#     }
# }
```

Check remaining requests for a client:

```python
remaining_min, remaining_hour = rate_limiter.get_remaining_requests(client_id)
print(f"Remaining: {remaining_min}/min, {remaining_hour}/hour")
```

## Administrative Functions

Reset rate limits for a specific client:

```python
rate_limiter.reset_client(client_id)
```

## Disabling Rate Limiting

To disable rate limiting entirely:

```bash
GUI_ENABLE_RATE_LIMITING=false
```

Or in code:

```python
config = GUIConfig(enable_rate_limiting=False)
```

When disabled, all rate limit checks are bypassed and no tracking is performed.

## Requirements

This implementation satisfies:
- **Requirement 12.5**: Implement rate limiting to prevent abuse
  - Tracks requests per user/IP
  - Implements configurable rate limits
  - Returns rate limit errors when exceeded
