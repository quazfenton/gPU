"""
Webhook Manager for async job notifications.

This module provides webhook notification capabilities for:
- Job completion notifications
- Job failure notifications
- Workflow completion notifications
- Custom event notifications

Supports multiple webhook endpoints with retry logic.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import hashlib
import hmac

import requests

logger = logging.getLogger(__name__)


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    id: str
    url: str
    events: List[str]  # List of event types to subscribe to
    secret: str  # For HMAC signature verification
    active: bool = True
    retry_count: int = 3
    retry_delay: int = 5  # seconds
    timeout: int = 30  # seconds
    headers: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class WebhookEvent:
    """Webhook event payload."""
    event_type: str
    event_id: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebhookManager:
    """
    Webhook notification manager.
    
    Manages webhook endpoints and sends asynchronous notifications
    for job and workflow events.
    
    Example:
        webhook_manager = WebhookManager()
        
        # Register webhook endpoint
        webhook_manager.add_endpoint(
            url='https://example.com/webhook',
            events=['job.completed', 'job.failed'],
            secret='your-webhook-secret'
        )
        
        # Send notification
        webhook_manager.notify_job_complete(
            job_id='job-123',
            result={'status': 'success', 'outputs': {...}}
        )
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize webhook manager.
        
        Args:
            max_workers: Maximum concurrent webhook workers
        """
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.max_workers = max_workers
        self._lock = threading.RLock()
        self._worker_pool: List[threading.Thread] = []
        
        logger.info(f"WebhookManager initialized with {max_workers} workers")
    
    def add_endpoint(
        self,
        url: str,
        events: List[str],
        secret: str,
        endpoint_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Add a webhook endpoint.
        
        Args:
            url: Webhook URL
            events: List of event types to subscribe to
            secret: Secret for HMAC signature
            endpoint_id: Optional endpoint ID (auto-generated if not provided)
            headers: Optional custom headers
            
        Returns:
            Endpoint ID
        """
        import secrets
        
        endpoint_id = endpoint_id or f"webhook-{secrets.token_hex(8)}"
        
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=url,
            events=events,
            secret=secret,
            headers=headers or {}
        )
        
        with self._lock:
            self.endpoints[endpoint_id] = endpoint
        
        logger.info(f"Added webhook endpoint: {endpoint_id} -> {url}")
        return endpoint_id
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """
        Remove a webhook endpoint.
        
        Args:
            endpoint_id: Endpoint ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if endpoint_id in self.endpoints:
                del self.endpoints[endpoint_id]
                logger.info(f"Removed webhook endpoint: {endpoint_id}")
                return True
            return False
    
    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """
        Get webhook endpoint by ID.
        
        Args:
            endpoint_id: Endpoint ID
            
        Returns:
            WebhookEndpoint or None
        """
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all webhook endpoints.
        
        Returns:
            List of endpoint configurations (without secrets)
        """
        endpoints = []
        for endpoint in self.endpoints.values():
            endpoints.append({
                'id': endpoint.id,
                'url': endpoint.url,
                'events': endpoint.events,
                'active': endpoint.active,
                'created_at': endpoint.created_at.isoformat(),
                'last_triggered': endpoint.last_triggered.isoformat() if endpoint.last_triggered else None,
                'success_count': endpoint.success_count,
                'failure_count': endpoint.failure_count
            })
        return endpoints
    
    def send_event(self, event: WebhookEvent) -> Dict[str, int]:
        """
        Send event to all subscribed endpoints.
        
        Args:
            event: WebhookEvent to send
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {'success': 0, 'failure': 0}
        
        with self._lock:
            endpoints_to_notify = [
                ep for ep in self.endpoints.values()
                if ep.active and event.event_type in ep.events
            ]
        
        if not endpoints_to_notify:
            logger.debug(f"No endpoints subscribed to event: {event.event_type}")
            return results
        
        # Send to each endpoint asynchronously
        threads = []
        for endpoint in endpoints_to_notify:
            thread = threading.Thread(
                target=self._send_to_endpoint,
                args=(endpoint, event),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete (with timeout)
        for thread in threads:
            thread.join(timeout=60)
        
        # Count results (will be updated by threads)
        for endpoint in endpoints_to_notify:
            if endpoint.failure_count > 0:
                results['failure'] += 1
            else:
                results['success'] += 1
        
        logger.info(f"Sent event {event.event_type} to {len(endpoints_to_notify)} endpoints")
        return results
    
    def _send_to_endpoint(self, endpoint: WebhookEndpoint, event: WebhookEvent) -> None:
        """
        Send event to a single endpoint with retry logic.
        
        Args:
            endpoint: WebhookEndpoint to send to
            event: WebhookEvent to send
        """
        payload = {
            'event_type': event.event_type,
            'event_id': event.event_id,
            'timestamp': event.timestamp,
            'data': event.data,
            'metadata': event.metadata
        }
        
        json_payload = json.dumps(payload)
        signature = self._generate_signature(json_payload, endpoint.secret)
        
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Signature': signature,
            'X-Webhook-Event': event.event_type,
            'X-Webhook-Timestamp': event.timestamp,
            **endpoint.headers
        }
        
        # Retry logic
        for attempt in range(endpoint.retry_count):
            try:
                response = requests.post(
                    endpoint.url,
                    data=json_payload,
                    headers=headers,
                    timeout=endpoint.timeout
                )
                
                if response.status_code == 200:
                    endpoint.success_count += 1
                    endpoint.last_triggered = datetime.now()
                    logger.debug(f"Webhook sent successfully to {endpoint.url}")
                    return
                else:
                    logger.warning(f"Webhook returned {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                
                if attempt < endpoint.retry_count - 1:
                    time.sleep(endpoint.retry_delay * (attempt + 1))
        
        # All retries failed
        endpoint.failure_count += 1
        logger.error(f"Webhook failed after {endpoint.retry_count} attempts to {endpoint.url}")
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC signature for webhook payload.
        
        Args:
            payload: JSON payload string
            secret: Webhook secret
            
        Returns:
            HMAC-SHA256 signature
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    # Convenience methods for common events
    
    def notify_job_complete(self, job_id: str, result: Dict[str, Any]) -> Dict[str, int]:
        """
        Notify job completion.
        
        Args:
            job_id: Job ID
            result: Job result data
            
        Returns:
            Success/failure counts
        """
        import uuid
        
        event = WebhookEvent(
            event_type='job.completed',
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            data={
                'job_id': job_id,
                'result': result,
                'status': 'completed'
            }
        )
        
        return self.send_event(event)
    
    def notify_job_failed(self, job_id: str, error: str) -> Dict[str, int]:
        """
        Notify job failure.
        
        Args:
            job_id: Job ID
            error: Error message
            
        Returns:
            Success/failure counts
        """
        import uuid
        
        event = WebhookEvent(
            event_type='job.failed',
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            data={
                'job_id': job_id,
                'error': error,
                'status': 'failed'
            }
        )
        
        return self.send_event(event)
    
    def notify_workflow_complete(self, workflow_id: str, results: Dict[str, Any]) -> Dict[str, int]:
        """
        Notify workflow completion.
        
        Args:
            workflow_id: Workflow ID
            results: Workflow results
            
        Returns:
            Success/failure counts
        """
        import uuid
        
        event = WebhookEvent(
            event_type='workflow.completed',
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            data={
                'workflow_id': workflow_id,
                'results': results,
                'status': 'completed'
            }
        )
        
        return self.send_event(event)
    
    def notify_custom_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Notify custom event.
        
        Args:
            event_type: Custom event type
            data: Event data
            
        Returns:
            Success/failure counts
        """
        import uuid
        
        event = WebhookEvent(
            event_type=event_type,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            data=data
        )
        
        return self.send_event(event)


# Module-level instance for easy access
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get or create module-level webhook manager."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


def notify_job_complete(job_id: str, result: Dict[str, Any]) -> Dict[str, int]:
    """Notify job completion via webhooks."""
    return get_webhook_manager().notify_job_complete(job_id, result)


def notify_job_failed(job_id: str, error: str) -> Dict[str, int]:
    """Notify job failure via webhooks."""
    return get_webhook_manager().notify_job_failed(job_id, error)
