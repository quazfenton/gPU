"""Unit tests for the WebSocketServer class."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocket, WebSocketDisconnect
from gui.websocket_server import WebSocketServer
from gui.events import EventEmitter


class TestWebSocketServer:
    """Test suite for WebSocketServer class."""
    
    def test_init(self):
        """Test WebSocketServer initialization."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        assert ws_server.event_emitter is emitter
        assert ws_server.connections == []
    
    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self):
        """Test that connect() accepts a WebSocket connection."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())
        
        # Connect should accept the websocket and add it to connections
        await ws_server.connect(websocket)
        
        websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_adds_to_connections(self):
        """Test that connect() adds the WebSocket to the connections list."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())
        
        # Connect
        await ws_server.connect(websocket)
        
        # Connection should be removed after disconnect
        assert websocket not in ws_server.connections
    
    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self):
        """Test that disconnect() removes a WebSocket from connections."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        
        # Manually add to connections
        ws_server.connections.append(websocket)
        
        # Disconnect
        await ws_server.disconnect(websocket)
        
        assert websocket not in ws_server.connections
    
    @pytest.mark.asyncio
    async def test_disconnect_handles_nonexistent_connection(self):
        """Test that disconnect() handles removing a connection that doesn't exist."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket that was never connected
        websocket = AsyncMock(spec=WebSocket)
        
        # Should not raise an exception
        await ws_server.disconnect(websocket)
    
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_connections(self):
        """Test that broadcast() sends messages to all connected clients."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSockets
        websocket1 = AsyncMock(spec=WebSocket)
        websocket2 = AsyncMock(spec=WebSocket)
        websocket3 = AsyncMock(spec=WebSocket)
        
        # Add to connections
        ws_server.connections = [websocket1, websocket2, websocket3]
        
        # Broadcast a message
        message = {'event_type': 'job.status_changed', 'data': {'job_id': '123'}}
        await ws_server.broadcast(message)
        
        # All connections should receive the message
        expected_json = json.dumps(message)
        websocket1.send_text.assert_called_once_with(expected_json)
        websocket2.send_text.assert_called_once_with(expected_json)
        websocket3.send_text.assert_called_once_with(expected_json)
    
    @pytest.mark.asyncio
    async def test_broadcast_serializes_to_json(self):
        """Test that broadcast() serializes messages to JSON."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        
        # Broadcast a message with nested data
        message = {
            'event_type': 'job.completed',
            'data': {
                'job_id': '123',
                'results': {'output': 'success'},
                'list': [1, 2, 3]
            }
        }
        await ws_server.broadcast(message)
        
        # Should send JSON string
        expected_json = json.dumps(message)
        websocket.send_text.assert_called_once_with(expected_json)
    
    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self):
        """Test that broadcast() removes connections that fail to send."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSockets - one that fails
        websocket1 = AsyncMock(spec=WebSocket)
        websocket2 = AsyncMock(spec=WebSocket)
        websocket2.send_text = AsyncMock(side_effect=Exception("Connection lost"))
        websocket3 = AsyncMock(spec=WebSocket)
        
        ws_server.connections = [websocket1, websocket2, websocket3]
        
        # Broadcast a message
        message = {'event_type': 'test', 'data': {}}
        await ws_server.broadcast(message)
        
        # Failed connection should be removed
        assert websocket1 in ws_server.connections
        assert websocket2 not in ws_server.connections
        assert websocket3 in ws_server.connections
    
    @pytest.mark.asyncio
    async def test_broadcast_with_no_connections(self):
        """Test that broadcast() handles having no connections."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # No connections
        assert ws_server.connections == []
        
        # Should not raise an exception
        message = {'event_type': 'test', 'data': {}}
        await ws_server.broadcast(message)
    
    def test_setup_listeners_registers_all_event_types(self):
        """Test that setup_listeners() registers listeners for all event types."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        ws_server.setup_listeners()
        
        # Check that listeners are registered for all event types
        expected_event_types = [
            'job.status_changed',
            'job.completed',
            'job.failed',
            'backend.status_changed',
            'workflow.step_completed'
        ]
        
        for event_type in expected_event_types:
            assert event_type in emitter.listeners
            assert len(emitter.listeners[event_type]) > 0
    
    @pytest.mark.asyncio
    async def test_setup_listeners_broadcasts_events(self):
        """Test that events are broadcast to WebSocket clients after setup_listeners()."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        
        # Setup listeners
        ws_server.setup_listeners()
        
        # Emit an event
        event_data = {'job_id': '123', 'status': 'running'}
        emitter.emit('job.status_changed', event_data)
        
        # Give asyncio time to process the task
        await asyncio.sleep(0.1)
        
        # WebSocket should have received the broadcast
        assert websocket.send_text.called
        
        # Check the message content
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'job.status_changed'
        assert message['data'] == event_data
    
    @pytest.mark.asyncio
    async def test_job_status_changed_event_broadcast(self):
        """Test that job.status_changed events are broadcast correctly."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        event_data = {'job_id': '123', 'status': 'completed'}
        emitter.emit('job.status_changed', event_data)
        
        await asyncio.sleep(0.1)
        
        assert websocket.send_text.called
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'job.status_changed'
        assert message['data']['job_id'] == '123'
        assert message['data']['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_job_completed_event_broadcast(self):
        """Test that job.completed events are broadcast correctly."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        event_data = {'job_id': '456', 'results': {'output': 'success'}}
        emitter.emit('job.completed', event_data)
        
        await asyncio.sleep(0.1)
        
        assert websocket.send_text.called
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'job.completed'
        assert message['data']['job_id'] == '456'
    
    @pytest.mark.asyncio
    async def test_job_failed_event_broadcast(self):
        """Test that job.failed events are broadcast correctly."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        event_data = {'job_id': '789', 'error': 'Backend unavailable'}
        emitter.emit('job.failed', event_data)
        
        await asyncio.sleep(0.1)
        
        assert websocket.send_text.called
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'job.failed'
        assert message['data']['error'] == 'Backend unavailable'
    
    @pytest.mark.asyncio
    async def test_backend_status_changed_event_broadcast(self):
        """Test that backend.status_changed events are broadcast correctly."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        event_data = {'backend_name': 'aws', 'status': 'healthy'}
        emitter.emit('backend.status_changed', event_data)
        
        await asyncio.sleep(0.1)
        
        assert websocket.send_text.called
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'backend.status_changed'
        assert message['data']['backend_name'] == 'aws'
    
    @pytest.mark.asyncio
    async def test_workflow_step_completed_event_broadcast(self):
        """Test that workflow.step_completed events are broadcast correctly."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        event_data = {'workflow_id': 'wf-123', 'step_id': 'step1', 'status': 'completed'}
        emitter.emit('workflow.step_completed', event_data)
        
        await asyncio.sleep(0.1)
        
        assert websocket.send_text.called
        call_args = websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['event_type'] == 'workflow.step_completed'
        assert message['data']['workflow_id'] == 'wf-123'
    
    @pytest.mark.asyncio
    async def test_multiple_connections_receive_broadcasts(self):
        """Test that multiple WebSocket connections all receive broadcasts."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        # Create multiple mock WebSockets
        websocket1 = AsyncMock(spec=WebSocket)
        websocket2 = AsyncMock(spec=WebSocket)
        websocket3 = AsyncMock(spec=WebSocket)
        
        ws_server.connections = [websocket1, websocket2, websocket3]
        ws_server.setup_listeners()
        
        # Emit an event
        event_data = {'job_id': '999', 'status': 'running'}
        emitter.emit('job.status_changed', event_data)
        
        await asyncio.sleep(0.1)
        
        # All connections should receive the broadcast
        assert websocket1.send_text.called
        assert websocket2.send_text.called
        assert websocket3.send_text.called
    
    @pytest.mark.asyncio
    async def test_multiple_event_types_broadcast(self):
        """Test that different event types are broadcast independently."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        websocket = AsyncMock(spec=WebSocket)
        ws_server.connections = [websocket]
        ws_server.setup_listeners()
        
        # Emit different event types
        emitter.emit('job.status_changed', {'job_id': '1', 'status': 'running'})
        emitter.emit('backend.status_changed', {'backend_name': 'aws', 'status': 'healthy'})
        
        await asyncio.sleep(0.1)
        
        # Should have been called twice
        assert websocket.send_text.call_count == 2
        
        # Check both messages were sent
        calls = websocket.send_text.call_args_list
        messages = [json.loads(call[0][0]) for call in calls]
        
        event_types = [msg['event_type'] for msg in messages]
        assert 'job.status_changed' in event_types
        assert 'backend.status_changed' in event_types
    
    @pytest.mark.asyncio
    async def test_create_broadcast_callback(self):
        """Test that _create_broadcast_callback creates a valid callback."""
        emitter = EventEmitter()
        ws_server = WebSocketServer(emitter)
        
        callback = ws_server._create_broadcast_callback('test.event')
        
        # Callback should be callable
        assert callable(callback)
        
        # Callback should accept data parameter
        # This will schedule a broadcast task (requires event loop)
        callback({'test': 'data'})
        
        # Give asyncio time to process the task
        await asyncio.sleep(0.1)
