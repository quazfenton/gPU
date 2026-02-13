"""Event system for real-time updates via WebSocket.

This module implements the observer pattern for broadcasting events to WebSocket clients.
"""

from typing import Callable, Dict, List


class EventEmitter:
    """Observer pattern for broadcasting events to WebSocket clients.
    
    The EventEmitter allows components to register listeners for specific event types
    and broadcast events to all registered listeners. This is used to implement
    real-time updates via WebSocket connections.
    
    Supported event types:
    - job.status_changed: Emitted when job status changes
    - job.completed: Emitted when job completes
    - job.failed: Emitted when job fails
    - backend.status_changed: Emitted when backend health status changes
    - workflow.step_completed: Emitted when workflow step completes
    
    Example:
        >>> emitter = EventEmitter()
        >>> def on_job_status_changed(data):
        ...     print(f"Job {data['job_id']} status: {data['status']}")
        >>> emitter.on('job.status_changed', on_job_status_changed)
        >>> emitter.emit('job.status_changed', {'job_id': '123', 'status': 'running'})
        Job 123 status: running
    """
    
    def __init__(self):
        """Initialize the EventEmitter with an empty listeners dictionary."""
        self.listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event_type: str, callback: Callable) -> None:
        """Register an event listener.
        
        Args:
            event_type: The type of event to listen for (e.g., 'job.status_changed')
            callback: The function to call when the event is emitted. The callback
                     will receive a single argument containing the event data.
        
        Example:
            >>> emitter = EventEmitter()
            >>> def handler(data):
            ...     print(f"Received: {data}")
            >>> emitter.on('test.event', handler)
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def emit(self, event_type: str, data: dict) -> None:
        """Emit an event to all registered listeners.
        
        Args:
            event_type: The type of event to emit (e.g., 'job.status_changed')
            data: The event data to pass to all listeners
        
        Example:
            >>> emitter = EventEmitter()
            >>> emitter.on('test.event', lambda d: print(d['message']))
            >>> emitter.emit('test.event', {'message': 'Hello'})
            Hello
        """
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    # Log the error but continue notifying other listeners
                    # In production, this should use proper logging
                    print(f"Error in event listener for {event_type}: {e}")
    
    def off(self, event_type: str, callback: Callable) -> None:
        """Unregister an event listener.
        
        Args:
            event_type: The type of event to stop listening for
            callback: The callback function to remove
        
        Example:
            >>> emitter = EventEmitter()
            >>> def handler(data):
            ...     print(data)
            >>> emitter.on('test.event', handler)
            >>> emitter.off('test.event', handler)
        """
        if event_type in self.listeners:
            try:
                self.listeners[event_type].remove(callback)
                # Clean up empty listener lists
                if not self.listeners[event_type]:
                    del self.listeners[event_type]
            except ValueError:
                # Callback not found in listeners list
                pass
