# Notebook ML Orchestrator - GUI Interface

Web-based graphical interface for the Notebook ML Orchestrator built with Gradio.

## Features

- **Job Submission**: Submit ML jobs through a web form with template selection and dynamic input fields
- **Job Monitoring**: Monitor job status and view results in real-time with WebSocket updates
- **Workflow Builder**: Create multi-step workflows visually with DAG editor
- **Template Management**: Browse and explore available templates with metadata
- **Backend Status**: Monitor backend health and performance metrics

## Quick Start

### Basic Usage

Launch the GUI with default settings:

```bash
python -m gui.main
```

Access the GUI at: http://0.0.0.0:7860

### Custom Configuration

Launch with custom host and port:

```bash
python -m gui.main --host 127.0.0.1 --port 8080
```

Launch with configuration file:

```bash
python -m gui.main --config /path/to/config.env
```

### Authentication

Enable authentication:

```bash
python -m gui.main --enable-auth
```

Default users (when using SimpleAuthProvider):
- Username: `admin`, Password: `admin`, Role: ADMIN
- Username: `user`, Password: `user`, Role: USER
- Username: `viewer`, Password: `viewer`, Role: VIEWER

### Public Sharing

Create a public Gradio link:

```bash
python -m gui.main --share
```

## Configuration

### Registering Backends

The GUI requires backends to be registered before jobs can be executed. Backends are not auto-discovered and must be configured based on your deployment.

To register backends, you can create a startup script or modify `gui/main.py` to register your backends. Example:

```python
from notebook_ml_orchestrator.core.backends.modal_backend import ModalBackend
from notebook_ml_orchestrator.core.backends.kaggle_backend import KaggleBackend

# In initialize_orchestrator_components function:
# Register Modal backend
modal_backend = ModalBackend(
    backend_id="modal-1",
    api_key=os.getenv("MODAL_API_KEY")
)
backend_router.register_backend(modal_backend)

# Register Kaggle backend
kaggle_backend = KaggleBackend(
    backend_id="kaggle-1",
    username=os.getenv("KAGGLE_USERNAME"),
    key=os.getenv("KAGGLE_KEY")
)
backend_router.register_backend(kaggle_backend)
```

Without registered backends, you can still:
- Browse templates
- View template metadata
- Monitor jobs (if any exist in the database)
- Build workflows

But you cannot submit new jobs until at least one backend is registered.

### Environment Variables

The GUI can be configured using environment variables:

- `GUI_HOST`: Host address (default: "0.0.0.0")
- `GUI_PORT`: Port number (default: 7860)
- `GUI_WEBSOCKET_PORT`: WebSocket port (default: 7861)
- `GUI_ENABLE_AUTH`: Enable authentication ("true"/"false", default: "false")
- `GUI_AUTH_PROVIDER`: Authentication provider name
- `GUI_ENABLE_WEBSOCKET`: Enable WebSocket ("true"/"false", default: "true")
- `GUI_THEME`: Gradio theme name (default: "default")
- `GUI_PAGE_SIZE`: Items per page (default: 50)
- `GUI_AUTO_REFRESH_INTERVAL`: Auto-refresh interval in seconds (default: 5)
- `GUI_SESSION_TIMEOUT`: Session timeout in seconds (default: 3600)

### Configuration File

Create a `.env` file with configuration:

```env
GUI_HOST=127.0.0.1
GUI_PORT=8080
GUI_ENABLE_AUTH=true
GUI_THEME=soft
```

Then launch with:

```bash
python -m gui.main --config .env
```

## Command-Line Options

```
usage: main.py [-h] [--config CONFIG] [--host HOST] [--port PORT]
               [--websocket-port WEBSOCKET_PORT] [--no-websocket]
               [--enable-auth] [--auth-provider AUTH_PROVIDER]
               [--theme {default,soft,monochrome}] [--db-path DB_PATH]
               [--share] [--debug]

Options:
  --config CONFIG       Path to configuration file (.env format)
  --host HOST          Host address to bind (overrides config)
  --port PORT          Port number to bind (overrides config)
  --websocket-port WEBSOCKET_PORT
                       WebSocket port number (overrides config)
  --no-websocket       Disable WebSocket for real-time updates
  --enable-auth        Enable authentication
  --auth-provider AUTH_PROVIDER
                       Authentication provider name
  --theme {default,soft,monochrome}
                       Gradio theme to use
  --db-path DB_PATH    Path to SQLite database (default: orchestrator.db)
  --share              Create a public Gradio link
  --debug              Enable debug mode with verbose logging
```

## Architecture

The GUI follows a layered architecture:

- **Presentation Layer**: Gradio UI components and event handlers
- **Service Layer**: Business logic for job management, workflow orchestration, and backend monitoring
- **Integration Layer**: Adapters for existing orchestrator components
- **Data Layer**: SQLite database and in-memory state management

### Components

- `gui/app.py`: Main Gradio application
- `gui/main.py`: Entry point script with CLI arguments
- `gui/config.py`: Configuration management
- `gui/auth.py`: Authentication and authorization
- `gui/events.py`: Event emitter for real-time updates
- `gui/websocket_server.py`: WebSocket server for broadcasting events
- `gui/services/`: Service layer components
- `gui/components/`: UI tab components

## Integration with Orchestrator

The GUI integrates seamlessly with existing orchestrator components:

- Uses the same `JobQueue` as CLI submissions
- Retrieves data from the same SQLite database
- Uses the same `WorkflowEngine` for workflow execution
- Uses the same `TemplateRegistry` for template discovery
- Uses the same `BackendRouter` for backend health monitoring

This ensures data consistency between CLI and GUI interfaces.

## Real-Time Updates

The GUI supports real-time updates via WebSocket connections:

- Job status changes are broadcast to all connected clients
- Backend health status updates are pushed in real-time
- Workflow execution progress is displayed live

WebSocket can be disabled with `--no-websocket` if not needed.

## Development

### Running Tests

```bash
pytest gui/tests/
```

### Debug Mode

Enable debug mode for verbose logging:

```bash
python -m gui.main --debug
```

## Requirements

- Python 3.8+
- Gradio
- FastAPI (for WebSocket support)
- SQLite
- All dependencies from `notebook_ml_orchestrator`

## License

Same as the main Notebook ML Orchestrator project.
