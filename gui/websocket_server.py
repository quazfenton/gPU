"""WebSocket server for real-time updates.

This module implements a FastAPI WebSocket server that broadcasts events
from the EventEmitter to all connected clients.
"""

import asyncio
import json
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from gui.events import EventEmitter


class WebSocketServer:
    """FastAPI WebSocket server for real-time updates.
    
    The WebSocketServer manages WebSocket connections and broadcasts events
    from the EventEmitter to all connected clients. It listens for events
    like job status changes, backend health updates, and workflow progress,
    and pushes these updates to all connected WebSocket clients in real-time.
    
    Example:
        >>> from gui.events import EventEmitter
        >>> emitter = EventEmitter()
        >>> ws_server = WebSocketServer(emitter)
        >>> ws_server.setup_listeners()
        >>> # In FastAPI endpoint:
        >>> @app.websocket("/ws")
        >>> async def websocket_endpoint(websocket: WebSocket):
        ...     await ws_server.connect(websocket)
    """
    
    def __init__(self, event_emitter: EventEmitter):
        """Initialize the WebSocket server.
        
        Args:
            event_emitter: The EventEmitter instance to listen for events
        """
        self.event_emitter = event_emitter
        self.connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to accept
        
        Example:
            >>> @app.websocket("/ws")
            >>> async def websocket_endpoint(websocket: WebSocket):
            ...     await ws_server.connect(websocket)
        """
        await websocket.accept()
        self.connections.append(websocket)
        
        try:
            # Keep the connection alive and handle incoming messages
            while True:
                # Wait for any message from client (ping/pong, etc.)
                await websocket.receive_text()
        except WebSocketDisconnect:
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection.
        
        Args:
            websocket: The WebSocket connection to disconnect
        """
        if websocket in self.connections:
            self.connections.remove(websocket)
    
    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to all connected clients.
        
        Args:
            message: The message dictionary to broadcast. Will be serialized to JSON.
        
        Example:
            >>> await ws_server.broadcast({
            ...     'event_type': 'job.status_changed',
            ...     'data': {'job_id': '123', 'status': 'running'}
            ... })
        """
        # Convert message to JSON
        message_json = json.dumps(message)
        
        # Send to all connected clients
        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                # Mark connection for removal if send fails
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    def setup_listeners(self) -> None:
        """Setup event listeners to broadcast events.
        
        This method registers callbacks with the EventEmitter for all event types
        that should be broadcast to WebSocket clients. The callbacks convert the
        events to WebSocket messages and broadcast them.
        
        Supported event types:
        - job.status_changed: Job status updates
        - job.completed: Job completion notifications
        - job.failed: Job failure notifications
        - backend.status_changed: Backend health status updates
        - workflow.step_completed: Workflow step completion updates
        
        Example:
            >>> ws_server = WebSocketServer(emitter)
            >>> ws_server.setup_listeners()
            >>> # Now events emitted will be broadcast to WebSocket clients
            >>> emitter.emit('job.status_changed', {'job_id': '123', 'status': 'running'})
        """
        # Define event types to broadcast
        event_types = [
            'job.status_changed',
            'job.completed',
            'job.failed',
            'backend.status_changed',
            'workflow.step_completed'
        ]
        
        # Register a listener for each event type
        for event_type in event_types:
            self.event_emitter.on(event_type, self._create_broadcast_callback(event_type))
    
    def _create_broadcast_callback(self, event_type: str):
        """Create a callback function that broadcasts events.
        
        Args:
            event_type: The type of event to broadcast
        
        Returns:
            A callback function that broadcasts the event
        """
        def callback(data: dict):
            """Callback that broadcasts the event to all WebSocket clients."""
            message = {
                'event_type': event_type,
                'data': data
            }
            # Schedule the broadcast as a coroutine
            # Check if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                asyncio.create_task(self.broadcast(message))
            except RuntimeError:
                # No running event loop, skip broadcast
                # This is expected when called from sync context
                pass
        
        return callback
