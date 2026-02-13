"""Unit tests for the EventEmitter class."""

import pytest
from gui.events import EventEmitter


class TestEventEmitter:
    """Test suite for EventEmitter class."""
    
    def test_on_registers_listener(self):
        """Test that on() registers a listener for an event type."""
        emitter = EventEmitter()
        callback = lambda data: None
        
        emitter.on('test.event', callback)
        
        assert 'test.event' in emitter.listeners
        assert callback in emitter.listeners['test.event']
    
    def test_on_supports_multiple_listeners(self):
        """Test that multiple listeners can be registered for the same event type."""
        emitter = EventEmitter()
        callback1 = lambda data: None
        callback2 = lambda data: None
        
        emitter.on('test.event', callback1)
        emitter.on('test.event', callback2)
        
        assert len(emitter.listeners['test.event']) == 2
        assert callback1 in emitter.listeners['test.event']
        assert callback2 in emitter.listeners['test.event']
    
    def test_emit_calls_registered_listeners(self):
        """Test that emit() calls all registered listeners with event data."""
        emitter = EventEmitter()
        results = []
        
        def callback1(data):
            results.append(('callback1', data))
        
        def callback2(data):
            results.append(('callback2', data))
        
        emitter.on('test.event', callback1)
        emitter.on('test.event', callback2)
        
        event_data = {'message': 'Hello'}
        emitter.emit('test.event', event_data)
        
        assert len(results) == 2
        assert ('callback1', event_data) in results
        assert ('callback2', event_data) in results
    
    def test_emit_with_no_listeners(self):
        """Test that emit() does nothing when no listeners are registered."""
        emitter = EventEmitter()
        
        # Should not raise an exception
        emitter.emit('nonexistent.event', {'data': 'test'})
    
    def test_emit_continues_on_listener_error(self):
        """Test that emit() continues calling other listeners if one raises an error."""
        emitter = EventEmitter()
        results = []
        
        def failing_callback(data):
            raise ValueError("Test error")
        
        def success_callback(data):
            results.append(data)
        
        emitter.on('test.event', failing_callback)
        emitter.on('test.event', success_callback)
        
        event_data = {'message': 'Hello'}
        emitter.emit('test.event', event_data)
        
        # Success callback should still be called despite failing callback
        assert event_data in results
    
    def test_off_removes_listener(self):
        """Test that off() removes a registered listener."""
        emitter = EventEmitter()
        callback = lambda data: None
        
        emitter.on('test.event', callback)
        emitter.off('test.event', callback)
        
        assert 'test.event' not in emitter.listeners or callback not in emitter.listeners.get('test.event', [])
    
    def test_off_removes_event_type_when_empty(self):
        """Test that off() removes the event type when no listeners remain."""
        emitter = EventEmitter()
        callback = lambda data: None
        
        emitter.on('test.event', callback)
        emitter.off('test.event', callback)
        
        assert 'test.event' not in emitter.listeners
    
    def test_off_with_nonexistent_callback(self):
        """Test that off() handles removing a callback that was never registered."""
        emitter = EventEmitter()
        callback = lambda data: None
        
        # Should not raise an exception
        emitter.off('test.event', callback)
    
    def test_off_with_nonexistent_event_type(self):
        """Test that off() handles removing from a nonexistent event type."""
        emitter = EventEmitter()
        callback = lambda data: None
        
        # Should not raise an exception
        emitter.off('nonexistent.event', callback)
    
    def test_off_removes_only_specified_callback(self):
        """Test that off() only removes the specified callback, not all callbacks."""
        emitter = EventEmitter()
        callback1 = lambda data: None
        callback2 = lambda data: None
        
        emitter.on('test.event', callback1)
        emitter.on('test.event', callback2)
        emitter.off('test.event', callback1)
        
        assert 'test.event' in emitter.listeners
        assert callback1 not in emitter.listeners['test.event']
        assert callback2 in emitter.listeners['test.event']
    
    def test_job_status_changed_event(self):
        """Test job.status_changed event type."""
        emitter = EventEmitter()
        received_data = []
        
        def on_job_status_changed(data):
            received_data.append(data)
        
        emitter.on('job.status_changed', on_job_status_changed)
        
        event_data = {'job_id': '123', 'status': 'running'}
        emitter.emit('job.status_changed', event_data)
        
        assert len(received_data) == 1
        assert received_data[0] == event_data
    
    def test_job_completed_event(self):
        """Test job.completed event type."""
        emitter = EventEmitter()
        received_data = []
        
        def on_job_completed(data):
            received_data.append(data)
        
        emitter.on('job.completed', on_job_completed)
        
        event_data = {'job_id': '123', 'results': {'output': 'success'}}
        emitter.emit('job.completed', event_data)
        
        assert len(received_data) == 1
        assert received_data[0] == event_data
    
    def test_job_failed_event(self):
        """Test job.failed event type."""
        emitter = EventEmitter()
        received_data = []
        
        def on_job_failed(data):
            received_data.append(data)
        
        emitter.on('job.failed', on_job_failed)
        
        event_data = {'job_id': '123', 'error': 'Backend unavailable'}
        emitter.emit('job.failed', event_data)
        
        assert len(received_data) == 1
        assert received_data[0] == event_data
    
    def test_backend_status_changed_event(self):
        """Test backend.status_changed event type."""
        emitter = EventEmitter()
        received_data = []
        
        def on_backend_status_changed(data):
            received_data.append(data)
        
        emitter.on('backend.status_changed', on_backend_status_changed)
        
        event_data = {'backend_name': 'aws', 'status': 'healthy'}
        emitter.emit('backend.status_changed', event_data)
        
        assert len(received_data) == 1
        assert received_data[0] == event_data
    
    def test_workflow_step_completed_event(self):
        """Test workflow.step_completed event type."""
        emitter = EventEmitter()
        received_data = []
        
        def on_workflow_step_completed(data):
            received_data.append(data)
        
        emitter.on('workflow.step_completed', on_workflow_step_completed)
        
        event_data = {'workflow_id': 'wf-123', 'step_id': 'step1', 'status': 'completed'}
        emitter.emit('workflow.step_completed', event_data)
        
        assert len(received_data) == 1
        assert received_data[0] == event_data
    
    def test_multiple_event_types(self):
        """Test that different event types are handled independently."""
        emitter = EventEmitter()
        job_events = []
        backend_events = []
        
        def on_job_event(data):
            job_events.append(data)
        
        def on_backend_event(data):
            backend_events.append(data)
        
        emitter.on('job.status_changed', on_job_event)
        emitter.on('backend.status_changed', on_backend_event)
        
        job_data = {'job_id': '123', 'status': 'running'}
        backend_data = {'backend_name': 'aws', 'status': 'healthy'}
        
        emitter.emit('job.status_changed', job_data)
        emitter.emit('backend.status_changed', backend_data)
        
        assert len(job_events) == 1
        assert len(backend_events) == 1
        assert job_events[0] == job_data
        assert backend_events[0] == backend_data
    
    def test_listener_receives_correct_data(self):
        """Test that listeners receive the exact data passed to emit()."""
        emitter = EventEmitter()
        received_data = None
        
        def callback(data):
            nonlocal received_data
            received_data = data
        
        emitter.on('test.event', callback)
        
        event_data = {
            'job_id': '123',
            'status': 'running',
            'nested': {'key': 'value'},
            'list': [1, 2, 3]
        }
        emitter.emit('test.event', event_data)
        
        assert received_data == event_data
        assert received_data['nested'] == {'key': 'value'}
        assert received_data['list'] == [1, 2, 3]
