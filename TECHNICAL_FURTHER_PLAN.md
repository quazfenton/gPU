# Technical Further Plan - Notebook ML Orchestrator

## Executive Summary

This document provides a comprehensive technical plan for completing the Notebook ML Orchestrator based on thorough review of the existing codebase, specifications, and implementation status. The project is a comprehensive ML orchestration platform that leverages free notebook platforms (Colab, Kaggle, Modal, HF Spaces) with a unified GUI.

**Current Status (Updated March 2, 2026):**

### ✅ Completed Components (~75% Overall)

**Backend Implementations (100%, Updated March 2, 2026):**
- ✅ ModalBackend - Full implementation with GPU support, cost estimation, retry logic
- ✅ HuggingFaceBackend - Inference API + Spaces integration
- ✅ KaggleBackend - Kernel execution with quota monitoring
- ✅ ColabBackend - OAuth + Google Drive integration
**Template Library (100%):**
- ✅ 29 templates discovered and working across 8 categories
- ✅ Audio (6), Vision (11), Text (4), ML (4), Training (2), Video (3), 3D (1), Test (1)
- ✅ Template registry with auto-discovery and validation
- ✅ Fixed 3 syntax errors in templates (llm_chat, sound_generator, training_data_augmentor)

**Core Infrastructure (100%):**
- ✅ Job queue with SQLite persistence
- ✅ Workflow engine with DAG execution
- ✅ Batch processor
- ✅ Backend router with LoadBalancer, CostOptimizer, HealthMonitor

**GUI Interface (90%):**
- ✅ Gradio app with 7 tabs
- ✅ Service layer (JobService, TemplateService, WorkflowService, BackendMonitorService)
- ✅ Authentication middleware with session management
- ✅ Rate limiting (60 req/min, 1000 req/hour)
- ✅ WebSocket server for real-time updates
- ⚠️ WebSocket client integration pending
- ⚠️ Visual DAG editor needs JavaScript integration

**Test Coverage (80%):**
- ✅ Backend tests (Modal, HF, Kaggle, Colab)
- ✅ Core tests (job queue, database, workflow, template registry)
- ✅ Template tests with property-based testing
- ⚠️ Security tests missing
- ⚠️ GUI component tests missing

### ⚠️ Partial Implementations (~60%)

**Security Features:**
- ✅ Basic authentication (SimpleAuthProvider)
- ✅ Session management
- ✅ Rate limiting
- ✅ Input validation
- ❌ Credential encryption (AES-256-GCM) - MISSING
- ❌ JWT/OAuth support - MISSING
- ❌ Secrets manager integration - MISSING
- ❌ XSS prevention - MISSING
- ❌ Security logging - MISSING

### ❌ Missing Components (~20%)

**Deployment:**
- ❌ Dockerfile
- ❌ docker-compose.yml
- ❌ Kubernetes manifests
- ❌ CI/CD pipeline

---

## 1. Architecture Review

### 1.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Gradio GUI (gui/)                                   │   │
│  │  - Job Submission (V2 - JSON input)                  │   │
│  │  - Job Monitoring (V2 - Real data)                   │   │
│  │  - Template Management (V2)                          │   │
│  │  - Backend Status                                    │   │
│  │  - Backend Registration                              │   │
│  │  - File Manager                                      │   │
│  │  - Workflow Builder                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CLI (runna.py)                                      │   │
│  │  - Kaggle operations                                 │   │
│  │  - Modal deployment                                  │   │
│  │  - AWS Lambda packaging                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Job Queue    │  │ Backend      │  │ Workflow     │     │
│  │ (SQLite)     │  │ Router       │  │ Engine       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Template     │  │ Batch        │                        │
│  │ Registry     │  │ Processor    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Modal        │  │ Kaggle       │  │ HuggingFace  │     │
│  │ (Partial)    │  │ (CLI only)   │  │ (Missing)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                          │
│  │ Colab        │                                          │
│  │ (Missing)    │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Files Structure

```
gPu/
├── gui/                          # GUI Interface (Gradio)
│   ├── app.py                    # Main Gradio application
│   ├── main.py                   # Entry point with CLI
│   ├── config.py                 # Configuration management
│   ├── events.py                 # Event emitter for real-time updates
│   ├── websocket_server.py       # WebSocket server
│   ├── auth.py                   # Authentication middleware
│   ├── rate_limiter.py           # Rate limiting
│   ├── validation.py             # Input validation
│   ├── error_handling.py         # Error handling utilities
│   ├── health.py                 # Health checker
│   ├── services/                 # Service layer
│   │   ├── job_service.py        # Job submission/monitoring
│   │   ├── template_service.py   # Template metadata
│   │   ├── workflow_service.py   # Workflow execution
│   │   └── backend_monitor_service.py
│   └── components/               # UI components
│       ├── job_submission_tab_v2.py
│       ├── job_monitoring_tab_v2.py
│       ├── template_management_tab_v2.py
│       ├── backend_status_tab.py
│       ├── backend_registration_tab.py
│       ├── file_manager_tab.py
│       └── workflow_builder_tab.py
├── notebook_ml_orchestrator/     # Core orchestrator (needs verification)
│   └── core/
│       ├── interfaces.py         # Abstract base classes
│       ├── models.py             # Data models
│       ├── database.py           # SQLite management
│       ├── job_queue.py          # Job queue implementation
│       ├── backend_router.py     # Multi-backend routing
│       ├── workflow_engine.py    # DAG workflow execution
│       └── template_registry.py  # Template discovery
├── runna.py                      # Main Kaggle CLI tool
├── app_library.py                # Modal app library manager
├── modal_deploy.py               # Modal deployment template
├── deploy_model.py               # Model deployment helper
├── doctor.py                     # Environment diagnostics
├── apps/                         # Modal app templates
│   ├── library.json
│   ├── image_classifier.py
│   ├── text_generator.py
│   └── ...
├── specs/                        # Specification documents
│   ├── gui-interface/
│   ├── enhanced-backend-support/
│   ├── security-enhancements/
│   └── template-library-expansion/
└── docs/                         # Documentation
```

---

## 2. Critical Gaps Analysis

### 2.1 Backend Implementations (HIGH PRIORITY)

**Current State:**
- `MultiBackendRouter` exists but has no concrete backend implementations
- Modal deployment exists via CLI (`runna.py deploy-modal`) but not integrated with router
- Kaggle operations work via CLI but not as backend
- No HuggingFace or Colab backend implementations

**Impact:**
- Jobs submitted via GUI cannot be executed (no registered backends)
- Automatic routing not functional
- Health monitoring not implemented
- Failover not available

**Required Implementations:**

#### 2.1.1 ModalBackend
```python
# File: notebook_ml_orchestrator/core/backends/modal_backend.py
class ModalBackend(Backend):
    - execute_job(job, template) -> JobResult
    - check_health() -> HealthStatus
    - supports_template(template_name) -> bool
    - estimate_cost(resource_estimate) -> float
    - Authentication via MODAL_TOKEN_ID, MODAL_TOKEN_SECRET
    - GPU support: T4, A10G, A100
    - Timeout handling
    - Volume mounting for model caching
```

#### 2.1.2 HuggingFaceBackend
```python
# File: notebook_ml_orchestrator/core/backends/huggingface_backend.py
class HuggingFaceBackend(Backend):
    - execute_job(job, template) -> JobResult
    - check_health() -> HealthStatus
    - supports_template(template_name) -> bool
    - estimate_cost(resource_estimate) -> float (always 0 for free tier)
    - Authentication via HF_TOKEN
    - Inference API integration
    - Spaces API for custom models
    - Gradio client for Space interaction
```

#### 2.1.3 KaggleBackend
```python
# File: notebook_ml_orchestrator/core/backends/kaggle_backend.py
class KaggleBackend(Backend):
    - execute_job(job, template) -> JobResult
    - check_health() -> HealthStatus
    - supports_template(template_name) -> bool
    - estimate_cost(resource_estimate) -> float (always 0 for free tier)
    - Authentication via KAGGLE_USERNAME, KAGGLE_KEY
    - Kernel creation and execution
    - Output file retrieval
    - Quota monitoring (30 hours GPU/week)
```

#### 2.1.4 ColabBackend
```python
# File: notebook_ml_orchestrator/core/backends/colab_backend.py
class ColabBackend(Backend):
    - execute_job(job, template) -> JobResult
    - check_health() -> HealthStatus
    - supports_template(template_name) -> bool
    - estimate_cost(resource_estimate) -> float (always 0 for free tier)
    - Authentication via Google OAuth
    - Notebook creation in Google Drive
    - Cell execution via Colab API
    - Result retrieval
```

### 2.2 Template Library (HIGH PRIORITY)

**Current State:**
- Template registry exists with discovery mechanism
- 14 templates discovered at startup (per GUI_IMPLEMENTATION_STATUS.md)
- Template metadata structure defined
- No actual template implementations in codebase

**Required Template Categories:**

#### 2.2.1 Audio Templates
```
templates/audio/
├── speech_recognition.py    # Audio → Text (Whisper, etc.)
├── audio_generation.py      # Text → Audio (TTS)
├── music_processing.py      # Audio analysis/generation
└── sound_classification.py  # Audio classification
```

#### 2.2.2 Vision Templates
```
templates/vision/
├── image_classification.py  # Image → Labels
├── object_detection.py      # Image → Bounding boxes
├── image_segmentation.py    # Image → Masks
├── image_generation.py      # Text → Image (Stable Diffusion)
└── video_processing.py      # Video → Analysis
```

#### 2.2.3 Language Templates
```
templates/language/
├── text_generation.py       # Text completion (GPT-2, etc.)
├── sentiment_analysis.py    # Text → Sentiment
├── named_entity_recognition.py  # Text → Entities
├── translation.py           # Text → Translated text
└── summarization.py         # Text → Summary
```

#### 2.2.4 Multimodal Templates
```
templates/multimodal/
├── image_captioning.py      # Image → Description
├── visual_qa.py             # Image + Question → Answer
├── text_to_image.py         # Text → Generated image
└── document_understanding.py # Document → Structured data
```

### 2.3 Security Enhancements (MEDIUM PRIORITY)

**Current State:**
- Basic authentication in `gui/auth.py` (SimpleAuthProvider)
- Rate limiting in `gui/rate_limiter.py`
- Input validation in `gui/validation.py`
- Error handling in `gui/error_handling.py`

**Missing Security Features:**

#### 2.3.1 Credential Encryption
```python
# File: notebook_ml_orchestrator/core/security/credential_store.py
class CredentialStore:
    - encrypt_credential(key: str, value: str) -> str
    - decrypt_credential(key: str, encrypted_value: str) -> str
    - store_credential(service: str, key: str, value: str)
    - get_credential(service: str, key: str) -> str
    - Support for:
      * AES-256-GCM encryption
      * Key derivation via PBKDF2
      * Integration with HashiCorp Vault
      * AWS Secrets Manager
      * Azure Key Vault
```

#### 2.3.2 Enhanced Authentication
```python
# File: notebook_ml_orchestrator/core/security/auth_manager.py
class AuthenticationManager:
    - Support multiple auth methods:
      * API keys
      * JWT tokens
      * OAuth 2.0
    - Session management with secure cookies
    - Password hashing (bcrypt/Argon2)
    - Account lockout after failed attempts
    - API key rotation
```

#### 2.3.3 SQL Injection Prevention
```python
# Already partially implemented via parameterized queries
# Need to verify all database operations use:
# - Parameterized queries (✓ in job_queue_old.py)
# - ORM or query builder
# - Input validation for identifiers
```

#### 2.3.4 XSS Prevention (GUI)
```python
# File: gui/security/content_sanitizer.py
class ContentSanitizer:
    - escape_html(content: str) -> str
    - sanitize_html(content: str, allowlist: List[str]) -> str
    - set_security_headers(response) -> response
    - Content-Security-Policy headers
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
```

### 2.4 Workflow Engine (MEDIUM PRIORITY)

**Current State:**
- `WorkflowEngine` class exists
- DAG-based execution model
- Basic step execution

**Missing Features:**
- Visual DAG editor (GUI component exists but mock)
- Type validation for step connections
- Data passing between steps
- Conditional execution
- Parallel step execution
- Error handling and recovery
- Workflow persistence

---

## 3. Implementation Roadmap (Updated)

### Phase 1: Backend Integration ✅ COMPLETE

**Status:** All backend implementations are complete and tested.

**Completed:**
- ✅ ModalBackend with full GPU support and cost estimation
- ✅ HuggingFaceBackend with Inference API and Spaces
- ✅ KaggleBackend with kernel execution and quota monitoring
- ✅ ColabBackend with OAuth and Drive integration
- ✅ BackendRouter with LoadBalancer, CostOptimizer, HealthMonitor
- ✅ All backend tests passing

### Phase 2: Template Library ✅ COMPLETE

**Status:** Template library is complete with 29 templates.

**Completed:**
- ✅ Template base class with validation
- ✅ Template registry with auto-discovery
- ✅ 29 templates across 8 categories
- ✅ Template-backend support mapping
- ✅ All template tests passing
- ✅ Fixed 3 syntax errors in templates

### Phase 3: Security Hardening (NEW PRIORITY 1)

**Duration:** 2 weeks

**Goal:** Production-ready security with encryption, JWT, and secrets management.

#### Week 1: Credential Management & Authentication
- [ ] 3.1 Create security package structure
  - `notebook_ml_orchestrator/security/__init__.py`
  - `notebook_ml_orchestrator/security/credential_store.py`
  - `notebook_ml_orchestrator/security/auth_manager.py`
  - `notebook_ml_orchestrator/security/security_logger.py`

- [ ] 3.2 Implement CredentialStore with AES-256-GCM
  - Key derivation using PBKDF2
  - Encrypt/decrypt credential methods
  - Integration with environment variables
  - Support for master key from env/config

- [ ] 3.3 Integrate secrets managers
  - HashiCorp Vault support
  - AWS Secrets Manager support
  - Azure Key Vault support
  - Fallback to encrypted file storage

- [ ] 3.4 Enhance AuthenticationManager
  - JWT token generation and validation
  - OAuth 2.0 flow support
  - API key rotation
  - Password hashing with bcrypt

#### Week 2: Security Logging & XSS Prevention
- [ ] 3.5 Implement SecurityLogger
  - Authentication attempt logging
  - Authorization failure logging
  - Rate limit violation logging
  - Input validation failure logging
  - SIEM integration support

- [ ] 3.6 Add XSS prevention in GUI
  - HTML escaping utility
  - Content sanitizer with allowlist
  - Content-Security-Policy headers
  - X-Content-Type-Options header
  - X-Frame-Options header

- [ ] 3.7 SQL injection prevention audit
  - Verify all queries use parameterization
  - Add input validation for identifiers
  - Audit ORM usage

- [ ] 3.8 Security testing
  - Unit tests for credential encryption
  - Tests for JWT authentication
  - XSS prevention tests
  - SQL injection tests

### Phase 4: Deployment Automation (NEW PRIORITY 2)

**Duration:** 1-2 weeks

**Goal:** One-command deployment with Docker and Kubernetes.

#### Week 3: Docker & Local Development
- [ ] 4.1 Create Dockerfile
  - Multi-stage build for small image
  - Non-root user for security
  - Health check endpoint
  - Proper layer caching

- [ ] 4.2 Create docker-compose.yml
  - Orchestrator service
  - PostgreSQL service (optional)
  - Volume mounts for persistence
  - Environment variable configuration

- [ ] 4.3 Create .dockerignore
- [ ] 4.4 Create deployment documentation
  - Docker quickstart guide
  - Environment variable reference
  - Troubleshooting guide

#### Week 4: Kubernetes & CI/CD
- [ ] 4.5 Create Kubernetes manifests
  - Deployment with 3 replicas
  - Service with LoadBalancer
  - ConfigMap for configuration
  - Secret for credentials
  - Ingress for external access

- [ ] 4.6 Create Helm chart
  - Values.yaml for configuration
  - Templates for all resources
  - Health check probes

- [ ] 4.7 Set up GitHub Actions CI/CD
  - Test workflow on PR
  - Build and push Docker image
  - Deploy to Kubernetes on merge

### Phase 5: GUI Polish (NEW PRIORITY 3)

**Duration:** 1 week

**Goal:** Fully functional real-time GUI.

#### Week 5: Real-time Updates & Workflow Builder
- [ ] 5.1 WebSocket client integration
  - Add JavaScript to JobMonitoringTab
  - Add JavaScript to BackendStatusTab
  - Handle reconnection logic

- [ ] 5.2 Real-time job status updates
  - Update job list on status change
  - Show notifications for completed jobs
  - Update job details panel

- [ ] 5.3 Visual workflow builder
  - Integrate Mermaid.js for DAG visualization
  - Add drag-and-drop step addition
  - Implement connection validation
  - Add workflow export/import

- [ ] 5.4 File download functionality
  - Implement output file downloads
  - Add file preview for common formats

### Phase 6: Testing & Documentation (NEW PRIORITY 4)

**Duration:** 1 week

**Goal:** Comprehensive test coverage and documentation.

#### Week 6: Testing & Docs
- [ ] 6.1 GUI component tests
  - Test each tab rendering
  - Test event handlers
  - Test service layer integration

- [ ] 6.2 End-to-end integration tests
  - Full job submission flow
  - Workflow execution flow
  - Backend failover flow

- [ ] 6.3 API documentation
  - Generate API reference with Sphinx
  - Add usage examples
  - Document all endpoints

- [ ] 6.4 User documentation
  - Template usage guide
  - Security best practices
  - Production deployment guide
  - Troubleshooting guide

### Phase 3: Security Hardening (Weeks 7-8)

**Goal:** Production-ready security

#### Week 7: Credential Management & Auth
- [ ] 7.1 Implement `CredentialStore` with AES-256-GCM
- [ ] 7.2 Add key derivation via PBKDF2
- [ ] 7.3 Integrate with environment variables
- [ ] 7.4 Add support for secrets managers:
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault
- [ ] 7.5 Enhance `AuthenticationManager`
  - JWT token support
  - OAuth 2.0 flow
  - API key rotation
- [ ] 7.6 Implement secure session management
  - Cryptographically random session IDs
  - Secure cookie attributes
  - Session timeout enforcement

#### Week 8: Input Validation & XSS Prevention
- [ ] 8.1 Enhance input validation
  - String length limits
  - Path traversal prevention
  - SQL injection prevention
  - File type allowlist
- [ ] 8.2 Implement content sanitization
  - HTML escaping
  - HTML sanitizer with allowlist
  - CSP headers
- [ ] 8.3 Add security logging
  - Authentication attempts
  - Authorization failures
  - Rate limit violations
  - Input validation failures
- [ ] 8.4 Security testing
  - SQL injection tests
  - XSS tests
  - Rate limiting tests
  - Credential encryption tests

### Phase 4: Workflow Engine Completion (Weeks 9-10)

**Goal:** Full workflow orchestration

#### Week 9: Workflow Core
- [ ] 9.1 Enhance `WorkflowEngine`
  - Type validation for connections
  - Data passing between steps
  - Parallel execution support
- [ ] 9.2 Implement conditional execution
  - If-then-else logic
  - Branching based on outputs
- [ ] 9.3 Add error handling
  - Step-level retry
  - Workflow-level recovery
  - Compensation transactions
- [ ] 9.4 Workflow persistence
  - Save/load workflow definitions
  - Execution state persistence

#### Week 10: Visual Workflow Builder
- [ ] 10.1 Complete GUI workflow builder component
  - Visual DAG editor (using Mermaid.js or similar)
  - Drag-and-drop step addition
  - Connection validation
- [ ] 10.2 Step configuration UI
  - Dynamic field generation
  - Type-aware inputs
- [ ] 10.3 Workflow execution monitoring
  - Real-time progress display
  - Per-step status indicators
  - Result visualization
- [ ] 10.4 Workflow templates
  - Pre-built workflow examples
  - Import/export workflows

### Phase 5: Polish & Production Readiness (Weeks 11-12)

**Goal:** Production deployment

#### Week 11: Performance & Scalability
- [ ] 11.1 Database optimization
  - Index optimization
  - Query performance tuning
  - Connection pooling
- [ ] 11.2 Caching layer
  - Template metadata caching
  - Backend status caching
  - Result caching
- [ ] 11.3 Rate limiting enhancement
  - Distributed rate limiting
  - Per-endpoint limits
  - Burst handling
- [ ] 11.4 Load testing
  - Concurrent user simulation
  - Backend stress testing
  - Performance benchmarks

#### Week 12: Documentation & Deployment
- [ ] 12.1 Complete documentation
  - API reference
  - User guide
  - Deployment guide
  - Security best practices
- [ ] 12.2 Deployment automation
  - Docker containerization
  - Kubernetes manifests
  - CI/CD pipeline
- [ ] 12.3 Monitoring & alerting
  - Prometheus metrics
  - Grafana dashboards
  - Alert rules
- [ ] 12.4 Final testing
  - End-to-end tests
  - Security audit
  - Performance validation

---

## 4. Technical Debt & Refactoring

### 4.1 Code Quality Issues

**Issue 1: Inconsistent Error Handling**
- Some components use exceptions, others return error dicts
- Need standardized error handling across all services

**Fix:**
```python
# Create unified exception hierarchy
class OrchestratorError(Exception):
    base_class

class ValidationError(OrchestratorError):
    pass

class BackendError(OrchestratorError):
    pass

class AuthenticationError(OrchestratorError):
    pass
```

**Issue 2: Mixed Sync/Async Patterns**
- Some code uses async, some sync
- WebSocket is async, but job queue is sync

**Fix:**
- Decide on async-first or sync-first approach
- Refactor job queue to support async operations
- Use `asyncio.to_thread()` for blocking operations

**Issue 3: Tight Coupling**
- GUI components directly access job queue
- Should go through service layer only

**Fix:**
- Enforce layer boundaries
- Add interface checks
- Dependency injection for all components

### 4.2 Database Schema Evolution

**Current Schema:**
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    template TEXT NOT NULL,
    payload TEXT NOT NULL,
    route TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT,
    retries INTEGER DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    workflow_id TEXT,
    parent_job_id TEXT
)
```

**Missing Fields:**
- `user_id` - For multi-user support
- `backend_id` - Which backend executed the job
- `priority` - Job priority
- `cost` - Actual execution cost
- `started_at` - When execution started
- `completed_at` - When execution completed

**Migration Plan:**
```python
# File: notebook_ml_orchestrator/core/migrations/001_add_user_tracking.py
def migrate_001_add_user_tracking(conn):
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT DEFAULT 'default_user'")
    cursor.execute("ALTER TABLE jobs ADD COLUMN backend_id TEXT")
    cursor.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0")
    cursor.execute("ALTER TABLE jobs ADD COLUMN cost REAL DEFAULT 0.0")
    cursor.execute("ALTER TABLE jobs ADD COLUMN started_at REAL")
    cursor.execute("ALTER TABLE jobs ADD COLUMN completed_at REAL")
    conn.commit()
```

### 4.3 Testing Gaps

**Current Test Coverage:**
- Job queue: Partial (job_queue_old.py has tests)
- GUI components: Minimal
- Backend implementations: None (not implemented yet)
- Templates: None (not implemented yet)
- Security: None

**Required Test Suites:**
```
tests/
├── unit/
│   ├── core/
│   │   ├── test_job_queue.py
│   │   ├── test_backend_router.py
│   │   ├── test_workflow_engine.py
│   │   └── test_template_registry.py
│   ├── backends/
│   │   ├── test_modal_backend.py
│   │   ├── test_huggingface_backend.py
│   │   ├── test_kaggle_backend.py
│   │   └── test_colab_backend.py
│   ├── templates/
│   │   ├── test_audio_templates.py
│   │   ├── test_vision_templates.py
│   │   ├── test_language_templates.py
│   │   └── test_multimodal_templates.py
│   └── security/
│       ├── test_credential_store.py
│       ├── test_auth_manager.py
│       └── test_input_validator.py
├── integration/
│   ├── test_job_submission_flow.py
│   ├── test_workflow_execution.py
│   ├── test_backend_failover.py
│   └── test_gui_integration.py
└── property/
    ├── test_job_queue_properties.py
    ├── test_backend_properties.py
    └── test_routing_properties.py
```

---

## 5. Configuration Management

### 5.1 Environment Variables

**Required Environment Variables:**
```bash
# Database
ORCHESTRATOR_DB_PATH=orchestrator.db

# Logging
ORCHESTRATOR_LOG_LEVEL=INFO
ORCHESTRATOR_LOG_FILE=orchestrator.log

# Backend Credentials
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
HF_TOKEN=your_huggingface_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REFRESH_TOKEN=your_google_refresh_token

# GUI Configuration
GUI_HOST=0.0.0.0
GUI_PORT=7860
GUI_WEBSOCKET_PORT=7861
GUI_ENABLE_AUTH=false
GUI_ENABLE_RATE_LIMITING=true
GUI_RATE_LIMIT_PER_MINUTE=60
GUI_RATE_LIMIT_PER_HOUR=1000
GUI_UPLOAD_DIR=uploads

# Security
MASTER_ENCRYPTION_KEY=your_32_byte_key
SESSION_TIMEOUT=3600
```

### 5.2 Configuration File Format

**Recommended: YAML Configuration**
```yaml
# config.yaml
database:
  path: orchestrator.db

logging:
  level: INFO
  file: orchestrator.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

backends:
  modal:
    enabled: true
    default_gpu: "A10G"
    timeout: 300
    max_concurrent_jobs: 10

  huggingface:
    enabled: true
    default_space_hardware: "cpu-basic"
    max_concurrent_jobs: 5

  kaggle:
    enabled: true
    max_concurrent_kernels: 1
    quota_warning_threshold: 0.8  # Warn at 80% quota usage

  colab:
    enabled: false
    max_concurrent_runtimes: 1

routing:
  strategy: "cost-optimized"  # cost-optimized, round-robin, least-loaded
  prefer_free_tier: true
  health_check_interval: 300  # seconds
  max_retries: 3
  retry_backoff_base: 2

security:
  enable_auth: false
  session_timeout: 3600
  enable_rate_limiting: true
  rate_limit_per_minute: 60
  rate_limit_per_hour: 1000
  credential_encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2"

gui:
  host: "0.0.0.0"
  port: 7860
  websocket_port: 7861
  theme: "default"
  page_size: 50
  auto_refresh_interval: 5
```

---

## 6. Deployment Guide

### 6.1 Local Development

```bash
# Clone repository
git clone <repository-url>
cd gPu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-orchestrator.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run tests
pytest

# Start GUI
python -m gui.main --debug

# Or use CLI
python runna.py doctor
```

### 6.2 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-orchestrator.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-orchestrator.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 orchestrator
USER orchestrator

# Expose ports
EXPOSE 7860 7861

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')"

# Run application
CMD ["python", "-m", "gui.main", "--host", "0.0.0.0", "--port", "7860"]
```

```bash
# Build and run
docker build -t notebook-ml-orchestrator .
docker run -p 7860:7860 -p 7861:7861 \
    -e MODAL_TOKEN_ID=$MODAL_TOKEN_ID \
    -e MODAL_TOKEN_SECRET=$MODAL_TOKEN_SECRET \
    -e HF_TOKEN=$HF_TOKEN \
    -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
    -e KAGGLE_KEY=$KAGGLE_KEY \
    notebook-ml-orchestrator
```

### 6.3 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: notebook-ml-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: notebook-ml-orchestrator:latest
        ports:
        - containerPort: 7860
        - containerPort: 7861
        env:
        - name: MODAL_TOKEN_ID
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: modal-token-id
        - name: MODAL_TOKEN_SECRET
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: modal-token-secret
        # ... other secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 7860
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 7860
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
spec:
  selector:
    app: orchestrator
  ports:
  - name: http
    port: 80
    targetPort: 7860
  - name: websocket
    port: 7861
    targetPort: 7861
  type: LoadBalancer
```

---

## 7. Monitoring & Observability

### 7.1 Metrics to Collect

**Job Metrics:**
- Jobs submitted (counter)
- Jobs completed (counter)
- Jobs failed (counter)
- Job queue length (gauge)
- Job execution time (histogram)
- Job cost (histogram)

**Backend Metrics:**
- Backend health status (gauge)
- Backend queue length (gauge)
- Backend success rate (gauge)
- Backend response time (histogram)
- Backend cost total (counter)

**System Metrics:**
- Active WebSocket connections (gauge)
- Authentication attempts (counter)
- Rate limit violations (counter)
- Database query time (histogram)
- Memory usage (gauge)
- CPU usage (gauge)

### 7.2 Prometheus Integration

```python
# File: notebook_ml_orchestrator/core/metrics.py
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Job metrics
jobs_submitted = Counter('orchestrator_jobs_submitted_total', 'Total jobs submitted')
jobs_completed = Counter('orchestrator_jobs_completed_total', 'Total jobs completed')
jobs_failed = Counter('orchestrator_jobs_failed_total', 'Total jobs failed')
job_queue_length = Gauge('orchestrator_job_queue_length', 'Current job queue length')
job_execution_time = Histogram('orchestrator_job_execution_time_seconds', 'Job execution time')

# Backend metrics
backend_health = Gauge('orchestrator_backend_health', 'Backend health status', ['backend'])
backend_success_rate = Gauge('orchestrator_backend_success_rate', 'Backend success rate', ['backend'])

# Start metrics server
def start_metrics_server(port: int = 9090):
    start_http_server(port)
```

### 7.3 Logging Strategy

**Log Levels:**
- DEBUG: Detailed debugging information
- INFO: General operational events
- WARNING: Warning conditions (e.g., backend degraded)
- ERROR: Error conditions (e.g., job failed)
- CRITICAL: Critical failures (e.g., database unavailable)

**Structured Logging:**
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "job_submitted",
    job_id=job_id,
    template=template_name,
    backend=backend_id,
    user_id=user_id,
    timestamp=datetime.now().isoformat()
)
```

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backend API changes | Medium | High | Abstract API calls, add version checks |
| Free tier quota limits | High | Medium | Implement quota monitoring, alerting |
| Database corruption | Low | High | Regular backups, WAL mode |
| Security vulnerabilities | Medium | High | Regular security audits, dependency scanning |
| Performance degradation | Medium | Medium | Load testing, caching, optimization |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Credential leakage | Medium | Critical | Encryption, secrets management, audit logs |
| Service downtime | Low | High | Health monitoring, failover, backups |
| Data loss | Low | High | Regular backups, transaction logging |
| Cost overruns | Medium | Medium | Cost tracking, budget alerts |

---

## 9. Success Criteria

### 9.1 Functional Criteria

- [ ] Jobs can be submitted via GUI and executed on at least one backend
- [ ] Job status can be monitored in real-time
- [ ] At least 10 templates available across all categories
- [ ] Workflows can be created and executed
- [ ] Backend health is monitored and displayed
- [ ] Authentication and authorization work correctly
- [ ] Rate limiting prevents abuse

### 9.2 Non-Functional Criteria

- [ ] GUI response time < 2 seconds for all operations
- [ ] Job submission latency < 500ms
- [ ] System supports 100+ concurrent users
- [ ] 99.9% uptime (excluding backend downtime)
- [ ] Zero critical security vulnerabilities
- [ ] Test coverage > 80%

### 9.3 User Experience Criteria

- [ ] New users can submit first job within 5 minutes
- [ ] Template documentation is clear and complete
- [ ] Error messages are helpful and actionable
- [ ] Real-time updates work reliably
- [ ] File uploads work seamlessly

---

## 10. Next Immediate Actions

### Week 1 Sprint Plan

**Day 1-2: Setup & Planning**
- [ ] Review all specification documents
- [ ] Set up development environment
- [ ] Create GitHub project board with all tasks
- [ ] Set up CI/CD pipeline skeleton

**Day 3-5: Modal Backend Implementation**
- [ ] Create `ModalBackend` class
- [ ] Implement basic job execution
- [ ] Add health check
- [ ] Write unit tests
- [ ] Integration test with job queue

**Day 6-7: Testing & Documentation**
- [ ] Test Modal backend end-to-end
- [ ] Document Modal backend configuration
- [ ] Update GUI backend registration to support Modal
- [ ] Create example jobs for testing

---

## Appendix A: Glossary

- **Backend**: A compute platform that executes ML jobs (Modal, HuggingFace, Kaggle, Colab)
- **Job**: A unit of ML work submitted for execution
- **Template**: A reusable ML service component
- **Workflow**: A DAG of jobs executed in sequence/parallel
- **GUI**: Gradio-based web interface
- **CLI**: Command-line interface (`runna.py`, `notebook-orchestrator`)
- **Router**: Component that routes jobs to backends
- **Health Monitor**: Component that tracks backend availability

## Appendix B: References

- [Modal Documentation](https://modal.com/docs)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Google Colab Documentation](https://colab.research.google.com/)
- [Gradio Documentation](https://gradio.app/docs)
- [OWASP Security Guidelines](https://owasp.org/www-project-top-ten/)

## Appendix C: Contact & Support

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/notebook-ml-orchestrator/notebook-ml-orchestrator/issues)
- Documentation: [Read the docs](https://notebook-ml-orchestrator.readthedocs.io/)

---

**Document Version:** 1.0  
**Last Updated:** March 2, 2026  
**Author:** Notebook ML Orchestrator Team
