#!/bin/bash
# =============================================================================
# Notebook ML Orchestrator - Docker Entrypoint Script
# =============================================================================
# Handles initialization, migrations, and application startup
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# =============================================================================
# Environment Validation
# =============================================================================
log "Validating environment configuration..."

# Check required environment variables
if [ -z "$MASTER_KEY" ]; then
    warn "MASTER_KEY not set. Using default (NOT SECURE FOR PRODUCTION)"
    export MASTER_KEY="default-dev-key-do-not-use-in-production-32bytes"
fi

if [ -z "$JWT_SECRET" ]; then
    warn "JWT_SECRET not set. Using default (NOT SECURE FOR PRODUCTION)"
    export JWT_SECRET="default-jwt-secret-do-not-use-in-production-32b"
fi

if [ -z "$CREDENTIAL_SALT" ]; then
    warn "CREDENTIAL_SALT not set. Generating random salt..."
    export CREDENTIAL_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")
fi

# Validate key lengths
if [ ${#MASTER_KEY} -lt 32 ]; then
    error "MASTER_KEY must be at least 32 characters long"
fi

if [ ${#JWT_SECRET} -lt 32 ]; then
    error "JWT_SECRET must be at least 32 characters long"
fi

log "Environment validation passed"

# =============================================================================
# Directory Setup
# =============================================================================
log "Setting up directories..."

# Create required directories
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/data

# Set permissions
chmod 755 /app/uploads
chmod 755 /app/logs
chmod 755 /app/data

log "Directories ready"

# =============================================================================
# Security Configuration
# =============================================================================
log "Configuring security settings..."

# Generate secure random salt if not provided
if [ -z "$CREDENTIAL_SALT" ]; then
    export CREDENTIAL_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")
    log "Generated new CREDENTIAL_SALT"
fi

# Check if running in production
if [ "$APP_ENV" = "production" ]; then
    log "Running in PRODUCTION mode"
    
    # Verify production requirements
    if [ "$MASTER_KEY" = "default-dev-key-do-not-use-in-production-32bytes" ]; then
        error "MASTER_KEY must be set in production"
    fi
    
    if [ "$JWT_SECRET" = "default-jwt-secret-do-not-use-in-production-32b" ]; then
        error "JWT_SECRET must be set in production"
    fi
    
    # Enable security features
    export GUI_ENABLE_AUTH=${GUI_ENABLE_AUTH:-true}
    export GUI_ENABLE_RATE_LIMITING=${GUI_ENABLE_RATE_LIMITING:-true}
else
    log "Running in DEVELOPMENT mode"
    export DEBUG=${DEBUG:-true}
fi

log "Security configuration complete"

# =============================================================================
# Database Initialization
# =============================================================================
log "Initializing database..."

# Run database migrations if needed
python -c "
from notebook_ml_orchestrator.core.database import DatabaseManager
import os
import sys

db_path = os.environ.get('ORCHESTRATOR_DB_PATH', '/app/data/orchestrator.db')
try:
    db = DatabaseManager(db_path)
    db.initialize()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}', file=sys.stderr)
    sys.exit(1)
" || error "Database initialization failed"
log "Database ready"

# =============================================================================
    raise
" || warn "Database initialization skipped"
log "Database ready"

# =============================================================================
# Health Check Server
# =============================================================================
# Start a simple health check in background
if [ "$HEALTH_CHECK_ENABLED" != "false" ]; then
    log "Starting health check endpoint..."
    python -c "
import http.server
import socketserver
import os
import sys

PORT = int(os.environ.get('HEALTH_CHECK_PORT', 8080))

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\": \"healthy\"}')
        else:
            self.send_response(404)
            self.end_headers()

with socketserver.TCPServer(('', PORT), HealthHandler) as httpd:
    httpd.serve_forever()
" &
    log "Health check available at port ${HEALTH_CHECK_PORT:-8080}"
fi

# =============================================================================
# Application Startup
# =============================================================================
log "Starting Notebook ML Orchestrator..."

# Print configuration summary
echo ""
echo "=============================================="
echo "  Notebook ML Orchestrator"
echo "=============================================="
echo "  Host: ${GUI_HOST:-0.0.0.0}"
echo "  Port: ${GUI_PORT:-7860}"
echo "  WebSocket Port: ${GUI_WEBSOCKET_PORT:-7861}"
echo "  Environment: ${APP_ENV:-development}"
echo "  Debug: ${DEBUG:-false}"
echo "  Auth Enabled: ${GUI_ENABLE_AUTH:-false}"
echo "  Rate Limiting: ${GUI_ENABLE_RATE_LIMITING:-true}"
echo "=============================================="
echo ""

# Execute the main command
exec "$@"
