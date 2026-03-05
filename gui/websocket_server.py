"""WebSocket server for real-time updates.

This module implements a FastAPI WebSocket server that broadcasts events
from the EventEmitter to all connected clients.

SECURITY ENHANCED: WebSocket connections now require JWT authentication.
"""

import asyncio
import json
import logging
from typing import List, Tuple, Optional
from fastapi import WebSocket, WebSocketDisconnect
from gui.events import EventEmitter
from gui.auth import AuthManager, User
from notebook_ml_orchestrator.core.models import Permission

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents an authenticated WebSocket connection."""
    
    def __init__(self, websocket: WebSocket, user: User):
        self.websocket = websocket
        self.user = user
        self.connected_at = asyncio.get_event_loop().time()


class WebSocketServer:
    """FastAPI WebSocket server for real-time updates.

    The WebSocketServer manages WebSocket connections and broadcasts events
    from the EventEmitter to all connected clients. It listens for events
    like job status changes, backend health updates, and workflow progress,
    and pushes these updates to all connected WebSocket clients in real-time.
    
    SECURITY: All connections require valid JWT authentication.
    """

    def __init__(self, event_emitter: EventEmitter, auth_manager: Optional[AuthManager] = None):
        """Initialize the WebSocket server.

        Args:
            event_emitter: The EventEmitter instance to listen for events
            auth_manager: Optional AuthManager for JWT validation (created if not provided)
        """
        self.event_emitter = event_emitter
        self.auth_manager = auth_manager or AuthManager()
        self.connections: List[WebSocketConnection] = []
        self._lock = asyncio.Lock()
        
        logger.info("WebSocketServer initialized with JWT authentication")

    async def connect(self, websocket: WebSocket, token: str) -> Optional[WebSocketConnection]:
        """Accept a WebSocket connection after JWT validation.

        Args:
            websocket: The WebSocket connection to accept
            token: JWT token for authentication

        Returns:
            WebSocketConnection if successful, None if authentication failed

        Example:
            >>> @app.websocket("/ws")
            >>> async def websocket_endpoint(websocket: WebSocket):
            ...     token = await websocket.receive_text()
            ...     conn = await ws_server.connect(websocket, token)
            ...     if not conn:
            ...         return  # Authentication failed
        """
        # SECURITY: Validate JWT token
        try:
            user = self.auth_manager.verify_token(token)
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            try:
                await websocket.close(code=4001, reason=f"Invalid token: {str(e)}")
            except Exception:
                pass
            return None
        
        # SECURITY: Verify user has permission to view events
        if not user.has_permission(Permission.VIEW_BACKEND_STATUS):
            logger.warning(f"User {user.username} lacks permission for WebSocket access")
            try:
                await websocket.close(code=4003, reason="Unauthorized: insufficient permissions")
            except Exception:
                pass
            return None
        
        # Accept authenticated connection
        await websocket.accept()
        connection = WebSocketConnection(websocket, user)
        
        async with self._lock:
            self.connections.append(connection)
        
        logger.info(f"WebSocket connected for user {user.username}")
        
        # Send welcome message
        await self._send_to_connection(
            connection,
            {
                'event_type': 'system.connected',
                'data': {
                    'user': user.username,
                    'role': user.role.value,
                    'message': 'Connected to real-time updates'
                }
            }
        )
        
        return connection

    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection.

        Args:
            websocket: The WebSocket connection to disconnect
        """
        async with self._lock:
            for conn in self.connections:
                if conn.websocket == websocket:
                    self.connections.remove(conn)
                    logger.info(f"WebSocket disconnected for user {conn.user.username}")
                    break

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
        async with self._lock:
            disconnected = []
            
            for connection in self.connections:
                if not await self._send_to_connection(connection, message):
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.connections.remove(conn)
                logger.info(f"Removed disconnected WebSocket for user {conn.user.username}")

    async def broadcast_to_user(self, username: str, message: dict) -> None:
        """Broadcast a message to a specific user's connections.

        Args:
            username: Username to send message to
            message: The message dictionary to broadcast
        """
        async with self._lock:
            for connection in self.connections:
                if connection.user.username == username:
                    await self._send_to_connection(connection, message)

    async def _send_to_connection(
        self,
        connection: WebSocketConnection,
        message: dict
    ) -> bool:
        """Send message to a single connection.

        Args:
            connection: The connection to send to
            message: The message to send

        Returns:
            True if sent successfully, False if connection failed
        """
        try:
            message_json = json.dumps(message)
            await connection.websocket.send_text(message_json)
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {connection.user.username}: {e}")
            return False

    def setup_listeners(self) -> None:
        """Set up event listeners to broadcast events to WebSocket clients."""
        
        def on_event(event_data: dict):
            """Broadcast event to all connected clients."""
            asyncio.create_task(self.broadcast(event_data))
        
        # Listen for all events
        self.event_emitter.on('*', on_event)
        logger.info("WebSocket event listeners configured")
    
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
