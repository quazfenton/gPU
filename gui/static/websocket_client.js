/**
 * WebSocket Client for Real-Time Updates
 * 
 * Provides real-time communication with the Notebook ML Orchestrator backend
 * for job status updates, backend health monitoring, and workflow progress.
 * 
 * Usage:
 *   const ws = new WebSocketClient('ws://localhost:7861');
 *   ws.connect();
 *   ws.subscribe('job.status_changed', (data) => console.log(data));
 *   ws.disconnect();
 */

class WebSocketClient {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            reconnectInterval: 3000,
            maxReconnectAttempts: 10,
            heartbeatInterval: 30000,
            ...options
        };
        
        this.ws = null;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        this.subscriptions = new Map();
        this.messageQueue = [];
        this.isConnected = false;
        this.eventHandlers = new Map();
        
        // Bind methods
        this._onOpen = this._onOpen.bind(this);
        this._onMessage = this._onMessage.bind(this);
        this._onClose = this._onClose.bind(this);
        this._onError = this._onError.bind(this);
    }
    
    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('[WebSocket] Already connected');
            return;
        }
        
        try {
            this.ws = new WebSocket(this.url);
            this.ws.addEventListener('open', this._onOpen);
            this.ws.addEventListener('message', this._onMessage);
            this.ws.addEventListener('close', this._onClose);
            this.ws.addEventListener('error', this._onError);
        } catch (error) {
            console.error('[WebSocket] Connection error:', error);
            this._scheduleReconnect();
        }
    }
    
    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        this._clearReconnect();
        this._clearHeartbeat();
        
        if (this.ws) {
            this.ws.removeEventListener('open', this._onOpen);
            this.ws.removeEventListener('message', this._onMessage);
            this.ws.removeEventListener('close', this._onClose);
            this.ws.removeEventListener('error', this._onError);
            this.ws.close();
            this.ws = null;
        }
        
        this.isConnected = false;
    }
    
    /**
     * Subscribe to an event type
     * @param {string} eventType - Event type to subscribe to
     * @param {function} callback - Callback function
     */
    subscribe(eventType, callback) {
        if (!this.subscriptions.has(eventType)) {
            this.subscriptions.set(eventType, []);
        }
        this.subscriptions.get(eventType).push(callback);
        
        // Send subscription message to server
        if (this.isConnected) {
            this._send({
                type: 'subscribe',
                eventType: eventType
            });
        }
    }
    
    /**
     * Unsubscribe from an event type
     * @param {string} eventType - Event type to unsubscribe from
     * @param {function} callback - Callback function to remove
     */
    unsubscribe(eventType, callback) {
        if (this.subscriptions.has(eventType)) {
            const callbacks = this.subscriptions.get(eventType);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
            
            if (callbacks.length === 0) {
                this.subscriptions.delete(eventType);
                
                // Send unsubscription message to server
                if (this.isConnected) {
                    this._send({
                        type: 'unsubscribe',
                        eventType: eventType
                    });
                }
            }
        }
    }
    
    /**
     * Send a message to the server
     * @param {object} data - Data to send
     */
    send(data) {
        if (this.isConnected) {
            this._send(data);
        } else {
            // Queue message for later
            this.messageQueue.push(data);
        }
    }
    
    /**
     * Register event handler for connection events
     * @param {string} event - Event type ('connected', 'disconnected', 'error')
     * @param {function} handler - Event handler
     */
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    /**
     * Remove event handler
     * @param {string} event - Event type
     * @param {function} handler - Handler to remove
     */
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    /**
     * Internal: Send message through WebSocket
     * @private
     */
    _send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    /**
     * Internal: Handle connection open
     * @private
     */
    _onOpen() {
        console.log('[WebSocket] Connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Start heartbeat
        this._startHeartbeat();
        
        // Send subscriptions
        for (const [eventType, callbacks] of this.subscriptions) {
            if (callbacks.length > 0) {
                this._send({
                    type: 'subscribe',
                    eventType: eventType
                });
            }
        }
        
        // Send queued messages
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this._send(message);
        }
        
        // Trigger connected event
        this._triggerEvent('connected', { url: this.url });
    }
    
    /**
     * Internal: Handle incoming messages
     * @private
     */
    _onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Handle server messages
            if (data.type === 'heartbeat') {
                this._resetHeartbeat();
                return;
            }
            
            if (data.type === 'event' && data.eventType) {
                this._handleEvent(data.eventType, data.payload);
            }
        } catch (error) {
            console.error('[WebSocket] Message parse error:', error);
        }
    }
    
    /**
     * Internal: Handle event from server
     * @private
     */
    _handleEvent(eventType, payload) {
        if (this.subscriptions.has(eventType)) {
            const callbacks = this.subscriptions.get(eventType);
            callbacks.forEach(callback => {
                try {
                    callback(payload);
                } catch (error) {
                    console.error(`[WebSocket] Event handler error for ${eventType}:`, error);
                }
            });
        }
    }
    
    /**
     * Internal: Handle connection close
     * @private
     */
    _onClose(event) {
        console.log('[WebSocket] Disconnected:', event.code, event.reason);
        this.isConnected = false;
        this._clearHeartbeat();
        
        // Trigger disconnected event
        this._triggerEvent('disconnected', { 
            code: event.code, 
            reason: event.reason 
        });
        
        // Attempt reconnect
        if (event.code !== 1000) { // Normal closure
            this._scheduleReconnect();
        }
    }
    
    /**
     * Internal: Handle connection error
     * @private
     */
    _onError(error) {
        console.error('[WebSocket] Error:', error);
        
        // Trigger error event
        this._triggerEvent('error', { error });
    }
    
    /**
     * Internal: Schedule reconnection
     * @private
     */
    _scheduleReconnect() {
        if (this.reconnectTimer) {
            // Reconnect already scheduled, prevent multiple timers from running
            return;
        }
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('[WebSocket] Max reconnection attempts reached');
            return;
        }
        
        const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts);
        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null; // Clear the timer ID now that it's firing
            this.reconnectAttempts++;
            this.connect();
        }, delay);
    }
    
    /**
     * Internal: Clear reconnection timer
     * @private
     */
    _clearReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }
    
    /**
     * Internal: Start heartbeat
     * @private
     */
    _startHeartbeat() {
        this._clearHeartbeat();
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this._send({ type: 'heartbeat' });
            }
        }, this.options.heartbeatInterval);
    }
    
    /**
     * Internal: Reset heartbeat timer
     * @private
     */
    _resetHeartbeat() {
        this._startHeartbeat();
    }
    
    /**
     * Internal: Clear heartbeat timer
     * @private
     */
    _clearHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    /**
     * Internal: Trigger event handlers
     * @private
     */
    _triggerEvent(event, data) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            handlers.forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`[WebSocket] Event handler error for ${event}:`, error);
                }
            });
        }
    }
}

/**
 * WebSocket Manager for Gradio Integration
 * 
 * Manages WebSocket connections for Gradio applications with automatic
 * reconnection and event handling.
 */
class GradioWebSocketManager {
    constructor(baseUrl, options = {}) {
        this.baseUrl = baseUrl;
        this.options = options;
        this.client = null;
        this.initialized = false;
    }
    
    /**
     * Initialize WebSocket connection
     */
    initialize() {
        if (this.initialized) {
            return;
        }
        
        const wsUrl = this.baseUrl.replace('http', 'ws');
        this.client = new WebSocketClient(wsUrl, this.options);
        
        // Set up default event handlers
        this.client.on('connected', () => {
            console.log('[GradioWebSocket] Connected to server');
            this._updateConnectionStatus('connected');
        });
        
        this.client.on('disconnected', () => {
            console.log('[GradioWebSocket] Disconnected from server');
            this._updateConnectionStatus('disconnected');
        });
        
        this.client.on('error', (error) => {
            console.error('[GradioWebSocket] Error:', error);
            this._updateConnectionStatus('error');
        });
        
        this.client.connect();
        this.initialized = true;
    }
    
    /**
     * Subscribe to job status updates
     * @param {function} callback - Callback for job status changes
     */
    onJobStatusChange(callback) {
        if (!this.client) return;
        this.client.subscribe('job.status_changed', callback);
    }
    
    /**
     * Subscribe to backend health updates
     * @param {function} callback - Callback for backend health changes
     */
    onBackendHealthChange(callback) {
        if (!this.client) return;
        this.client.subscribe('backend.status_changed', callback);
    }
    
    /**
     * Subscribe to workflow progress updates
     * @param {function} callback - Callback for workflow progress
     */
    onWorkflowProgress(callback) {
        if (!this.client) return;
        this.client.subscribe('workflow.step_completed', callback);
    }
    
    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.client) {
            this.client.disconnect();
            this.initialized = false;
        }
    }
    
    /**
     * Update connection status indicator in UI
     * @private
     */
    _updateConnectionStatus(status) {
        const indicator = document.getElementById('websocket-status');
        if (indicator) {
            indicator.className = `status-indicator status-${status}`;
            indicator.title = `WebSocket: ${status}`;
        }
    }
}

// Export for use in Gradio components
window.WebSocketClient = WebSocketClient;
window.GradioWebSocketManager = GradioWebSocketManager;

// Auto-initialize if running in Gradio context
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        // Check if we're in a Gradio app
        const gradioApp = document.querySelector('.gradio-container');
        if (gradioApp) {
            // Get WebSocket URL from Gradio config or use default
            const wsPort = 7861;
            const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const wsUrl = `${wsProtocol}://${window.location.hostname}:${wsPort}`;
            const wsManager = new GradioWebSocketManager(wsUrl);
            wsManager.initialize();
            
            // Store in window for access from Gradio components
            window.gradioWebSocket = wsManager;
        }
    });
}
