# Notebook ML Orchestrator

A comprehensive ML orchestration platform that leverages free notebook platforms (Colab, Kaggle, Modal free tier, HF Spaces) to provide a unified GUI with template library, Zapier-style workflow automation, persistent job queuing, multi-backend routing, and batch processing capabilities for ML pipelines.

## Features

- **Persistent Job Queue**: SQLite-based job queue that survives runtime disconnects
- **Multi-Backend Routing**: Intelligent routing across local GPU, Modal, HuggingFace, Kaggle, and Colab
- **Template System**: Plugin-based architecture for ML services (audio, vision, language, multimodal)
- **Workflow Automation**: DAG-based workflows with conditional logic and data passing
- **Batch Processing**: Efficient processing of multiple jobs with parallel execution
- **Error Handling**: Comprehensive retry mechanisms and failure recovery
- **Real-time Monitoring**: Job status tracking and progress reporting

## Project Status

This project is currently in development. **Task 1** (Set up project structure and core interfaces) has been completed, which includes:

✅ **Completed:**
- Python package structure with proper modules
- Core abstract base classes and interfaces for templates, backends, and jobs
- SQLite database schema and connection management
- Logging configuration and error handling
- Testing framework with pytest and Hypothesis
- Job queue management with persistence and retry logic
- Basic backend routing and workflow engine structure
- Batch processing foundation

🚧 **In Progress:**
- Full backend routing implementation (Task 4)
- Complete workflow execution engine (Task 6)
- Template implementations (Tasks 8-10)
- GUI interface (Task 11)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd notebook-ml-orchestrator

# Install dependencies
pip install -r requirements-orchestrator.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Using the CLI

```bash
# Create a job
notebook-orchestrator create --job-id "test-job-1" --user-id "user1" --template "mock-template" --inputs '{"param": "value"}'

# List jobs
notebook-orchestrator list

# Check job status
notebook-orchestrator status test-job-1

# View statistics
notebook-orchestrator stats

# Clean up old jobs
notebook-orchestrator cleanup --days 30
```

### Using the Python API

```python
from notebook_ml_orchestrator.core.job_queue import JobQueueManager
from notebook_ml_orchestrator.core.interfaces import Job
from notebook_ml_orchestrator.core.models import JobStatus

# Initialize job queue
job_queue = JobQueueManager("orchestrator.db")

# Create and submit a job
job = Job(
    id="example-job",
    user_id="user123",
    template_name="example-template",
    inputs={"param1": "value1", "param2": 42}
)

job_id = job_queue.submit_job(job)
print(f"Job {job_id} submitted")

# Check job status
job = job_queue.get_job(job_id)
print(f"Job status: {job.status}")

# Get queue statistics
stats = job_queue.get_queue_statistics()
print(f"Total jobs: {stats['total_jobs']}")
```

## Architecture

The system follows a layered microservices architecture:

```
Frontend Layer (GUI/API) → Orchestration Layer (Job Queue, Router, Workflow) → Execution Layer (Backends) → Storage Layer (Database, Files)
```

### Core Components

- **JobQueueManager**: Persistent job queuing with SQLite
- **MultiBackendRouter**: Intelligent backend selection and load balancing
- **WorkflowEngine**: DAG-based workflow execution
- **BatchProcessor**: Efficient batch job processing
- **MLTemplate**: Plugin-based template system for ML services
- **Backend**: Abstract interface for compute resources

## Testing

The project uses pytest with Hypothesis for property-based testing:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m property      # Property-based tests

# Run with coverage
pytest --cov=notebook_ml_orchestrator --cov-report=html
```

## Configuration

Configuration is managed through environment variables or the `OrchestratorConfig` class:

```python
from notebook_ml_orchestrator.config import get_config

config = get_config()
print(f"Database path: {config.database.path}")
print(f"Log level: {config.logging.level}")
```

### Environment Variables

- `ORCHESTRATOR_DB_PATH`: Database file path
- `ORCHESTRATOR_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ORCHESTRATOR_LOG_FILE`: Log file path
- `ORCHESTRATOR_MAX_RETRIES`: Maximum job retry attempts
- `ORCHESTRATOR_MAX_PARALLEL_ITEMS`: Maximum parallel batch items

## Development

### Project Structure

```
notebook_ml_orchestrator/
├── core/                   # Core orchestration components
│   ├── interfaces.py       # Abstract base classes
│   ├── models.py          # Data models and enums
│   ├── database.py        # SQLite database management
│   ├── job_queue.py       # Job queue implementation
│   ├── backend_router.py  # Multi-backend routing
│   ├── workflow_engine.py # Workflow execution
│   ├── batch_processor.py # Batch processing
│   ├── exceptions.py      # Custom exceptions
│   └── logging_config.py  # Logging configuration
├── templates/             # ML template implementations (future)
├── backends/              # Backend implementations (future)
├── gui/                   # Gradio GUI interface (future)
├── tests/                 # Test suite
├── cli.py                 # Command-line interface
└── config.py              # Configuration management
```

### Adding New Components

1. **Templates**: Inherit from `MLTemplate` and implement required methods
2. **Backends**: Inherit from `Backend` and implement execution logic
3. **Tests**: Add unit tests and property-based tests for new components

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing
- **Hypothesis** for property-based testing

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project structure and interfaces
- [x] Database and job queue
- [x] Basic routing and workflow engines
- [x] Testing framework

### Phase 2: Backend Integration (Tasks 4-5)
- [ ] Complete multi-backend routing
- [ ] Backend implementations (Local, Modal, HuggingFace)
- [ ] Health monitoring and failover

### Phase 3: Template System (Tasks 8-10)
- [ ] Audio processing templates
- [ ] Vision processing templates  
- [ ] Language processing templates
- [ ] Multimodal pipelines

### Phase 4: User Interface (Task 11)
- [ ] Gradio web interface
- [ ] Workflow builder
- [ ] Real-time monitoring dashboard

### Phase 5: Advanced Features (Tasks 12-17)
- [ ] Security and data protection
- [ ] Performance optimization
- [ ] Creative AI templates
- [ ] Deployment automation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of existing backend infrastructure
- Integrates with popular ML platforms and services
- Inspired by workflow orchestration tools like Airflow and Prefect