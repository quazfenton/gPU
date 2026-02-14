"""
Backend Registration Tab component for GUI interface.

This module provides the UI component for registering and managing compute backends.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import os

from gui.services.backend_monitor_service import BackendMonitorService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin
from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter
from notebook_ml_orchestrator.core.models import BackendType, BackendCapabilities


class BackendRegistrationTab(LoggerMixin):
    """UI component for registering and managing backends."""
    
    def __init__(self, backend_router: MultiBackendRouter, backend_monitor: BackendMonitorService):
        """Initialize backend registration tab."""
        self.backend_router = backend_router
        self.backend_monitor = backend_monitor
        self.logger.info("BackendRegistrationTab initialized")
        
        # Auto-register backends from environment variables
        self._auto_register_backends()
    
    def _save_to_env(self, key: str, value: str):
        """Save a key-value pair to the .env file."""
        try:
            env_path = ".env"
            
            # Update current process environment immediately
            os.environ[key] = value
            
            # Create .env if it doesn't exist
            if not os.path.exists(env_path):
                with open(env_path, "w") as f:
                    f.write(f"{key}={value}\n")
                return
                
            # Read existing lines
            with open(env_path, "r") as f:
                lines = f.readlines()
                
            # Update or add key
            key_found = False
            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    key_found = True
                else:
                    new_lines.append(line)
                    
            if not key_found:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines.append("\n")
                new_lines.append(f"{key}={value}\n")
                
            # Write back
            with open(env_path, "w") as f:
                f.writelines(new_lines)
                
            self.logger.info(f"Saved {key} to .env")
            
        except Exception as e:
            self.logger.error(f"Failed to save to .env: {e}")

    def _auto_register_backends(self):
        """Auto-register backends if credentials exist in environment."""
        try:
            # Check for Modal
            modal_token_id = os.getenv("MODAL_TOKEN_ID")
            modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")
            if modal_token_id and modal_token_secret:
                try:
                    backend = self._create_modal_backend("modal-auto", modal_token_id, modal_token_secret)
                    if "modal-auto" not in self.backend_router.backends:
                        self.backend_router.register_backend(backend)
                        self.logger.info("Auto-registered Modal backend")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-register Modal: {e}")

            # Check for Kaggle
            kaggle_user = os.getenv("KAGGLE_USERNAME")
            kaggle_key = os.getenv("KAGGLE_KEY")
            if kaggle_user and kaggle_key:
                try:
                    backend = self._create_kaggle_backend("kaggle-auto", kaggle_user, kaggle_key)
                    if "kaggle-auto" not in self.backend_router.backends:
                        self.backend_router.register_backend(backend)
                        self.logger.info("Auto-registered Kaggle backend")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-register Kaggle: {e}")

            # Check for HuggingFace
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    backend = self._create_hf_backend("hf-auto", hf_token)
                    if "hf-auto" not in self.backend_router.backends:
                        self.backend_router.register_backend(backend)
                        self.logger.info("Auto-registered HuggingFace backend")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-register HuggingFace: {e}")
                    
            # Check for Colab
            colab_client_id = os.getenv("GOOGLE_CLIENT_ID")
            colab_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
            colab_refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
            if colab_client_id and colab_client_secret and colab_refresh_token:
                try:
                    backend = self._create_colab_backend("colab-auto", colab_client_id, colab_client_secret, colab_refresh_token)
                    if "colab-auto" not in self.backend_router.backends:
                        self.backend_router.register_backend(backend)
                        self.logger.info("Auto-registered Colab backend")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-register Colab: {e}")

        except Exception as e:
            self.logger.error(f"Error during backend auto-registration: {e}")
    
    def render(self) -> gr.Blocks:
        """Render the backend registration tab."""
        with gr.Blocks() as tab:
            gr.Markdown("## Backend Registration")
            gr.Markdown("Register compute backends for job execution.")
            
            # Current backends section
            gr.Markdown("### Registered Backends")
            
            registered_backends_list = gr.Textbox(
                label="Currently Registered Backends",
                value=self._get_registered_backends_text(),
                lines=5,
                interactive=False
            )
            
            refresh_backends_btn = gr.Button("🔄 Refresh List", variant="secondary")
            
            gr.Markdown("---")
            
            # Backend registration form
            gr.Markdown("### Register New Backend")
            
            with gr.Row():
                backend_type = gr.Dropdown(
                    label="Backend Type",
                    choices=["Mock (Testing)", "Local", "Modal", "Kaggle", "HuggingFace"],
                    value="Mock (Testing)",
                    interactive=True,
                    info="Select the type of backend to register"
                )
            
            with gr.Row():
                backend_id = gr.Textbox(
                    label="Backend ID",
                    placeholder="e.g., mock-backend-1, modal-gpu-1",
                    interactive=True,
                    info="Unique identifier for this backend"
                )
            
            # Configuration fields (shown/hidden based on backend type)
            with gr.Column(visible=False) as modal_config:
                gr.Markdown("**Modal Configuration**")
                modal_token_id = gr.Textbox(
                    label="Modal Token ID",
                    placeholder="Enter your Modal Token ID",
                    interactive=True
                )
                modal_token_secret = gr.Textbox(
                    label="Modal Token Secret",
                    placeholder="Enter your Modal Token Secret",
                    type="password",
                    interactive=True
                )
            
            with gr.Column(visible=False) as kaggle_config:
                gr.Markdown("**Kaggle Configuration**")
                kaggle_username = gr.Textbox(
                    label="Kaggle Username",
                    placeholder="Your Kaggle username",
                    interactive=True
                )
                kaggle_key = gr.Textbox(
                    label="Kaggle API Key",
                    placeholder="Your Kaggle API key",
                    type="password",
                    interactive=True
                )
            
            with gr.Column(visible=False) as hf_config:
                gr.Markdown("**HuggingFace Configuration**")
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="Your HuggingFace API token",
                    type="password",
                    interactive=True
                )
            
            with gr.Column(visible=False) as colab_config:
                gr.Markdown("**Google Colab Configuration**")
                gr.Markdown("Configure Google OAuth credentials for Colab/Drive access")
                colab_client_id = gr.Textbox(
                    label="Client ID",
                    placeholder="Google OAuth Client ID",
                    interactive=True
                )
                colab_client_secret = gr.Textbox(
                    label="Client Secret",
                    placeholder="Google OAuth Client Secret",
                    type="password",
                    interactive=True
                )
                colab_refresh_token = gr.Textbox(
                    label="Refresh Token",
                    placeholder="Google OAuth Refresh Token",
                    type="password",
                    interactive=True
                )
            
            with gr.Column(visible=True) as local_config:
                gr.Markdown("**Local Backend Configuration**")
                gr.Markdown("*Executes jobs on this machine.*")
                local_max_jobs = gr.Slider(
                    label="Max Concurrent Jobs",
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True
                )
            
            # Register button
            register_btn = gr.Button("Register Backend", variant="primary")
            
            # Status message
            registration_status = gr.Markdown(value="", visible=False)
            
            # Event handlers
            def update_config_visibility(backend_type_val):
                """Show/hide configuration fields based on backend type."""
                show_modal = backend_type_val == "Modal"
                show_kaggle = backend_type_val == "Kaggle"
                show_hf = backend_type_val == "HuggingFace"
                show_colab = backend_type_val == "Google Colab"
                show_local = backend_type_val == "Local Machine"
                
                return (
                    gr.Column(visible=show_modal),
                    gr.Column(visible=show_kaggle),
                    gr.Column(visible=show_hf),
                    gr.Column(visible=show_colab),
                    gr.Column(visible=show_local)
                )
            
            backend_type.change(
                fn=update_config_visibility,
                inputs=[backend_type],
                outputs=[modal_config, kaggle_config, hf_config, colab_config, local_config]
            )
            
            register_btn.click(
                fn=self.register_backend,
                inputs=[
                    backend_type,
                    backend_id,
                    modal_token_id,
                    modal_token_secret,
                    kaggle_username,
                    kaggle_key,
                    hf_token,
                    colab_client_id,
                    colab_client_secret,
                    colab_refresh_token,
                    local_max_jobs
                ],
                outputs=[registration_status, registered_backends_list]
            )
            
            refresh_backends_btn.click(
                fn=lambda: self._get_registered_backends_text(),
                outputs=[registered_backends_list]
            )
        
        return tab
    
    def _get_registered_backends_text(self) -> str:
        """Get text representation of registered backends."""
        try:
            backends = self.backend_router.backends
            
            if not backends:
                return "No backends registered yet.\n\nRegister a backend below to start executing jobs."
            
            lines = []
            for backend_id, backend in backends.items():
                backend_type = backend.type.value if hasattr(backend, 'type') else 'Unknown'
                lines.append(f"• {backend_id} ({backend_type})")
            
            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Failed to get registered backends: {e}")
            return "Error loading backends"
    
    def register_backend(
        self,
        backend_type: str,
        backend_id: str,
        modal_token_id: str,
        modal_token_secret: str,
        kaggle_username: str,
        kaggle_key: str,
        hf_token: str,
        colab_client_id: str,
        colab_client_secret: str,
        colab_refresh_token: str,
        local_max_jobs: int
    ) -> Tuple[gr.Markdown, str]:
        """Register a new backend."""
        try:
            # Validate backend ID
            if not backend_id or not backend_id.strip():
                error_msg = "❌ **Error:** Backend ID is required"
                return (
                    gr.Markdown(value=error_msg, visible=True),
                    self._get_registered_backends_text()
                )
            
            backend_id = backend_id.strip()
            
            # Check if backend already exists
            if backend_id in self.backend_router.backends:
                error_msg = f"❌ **Error:** Backend '{backend_id}' is already registered"
                return (
                    gr.Markdown(value=error_msg, visible=True),
                    self._get_registered_backends_text()
                )
            
            # Register based on type
            if backend_type == "Local Machine":
                backend = self._create_local_backend(backend_id, local_max_jobs)
            elif backend_type == "Modal":
                if not modal_token_id or not modal_token_secret:
                    error_msg = "❌ **Error:** Modal Token ID and Secret are required"
                    return (
                        gr.Markdown(value=error_msg, visible=True),
                        self._get_registered_backends_text()
                    )
                backend = self._create_modal_backend(backend_id, modal_token_id, modal_token_secret)
                self._save_to_env("MODAL_TOKEN_ID", modal_token_id)
                self._save_to_env("MODAL_TOKEN_SECRET", modal_token_secret)
            elif backend_type == "Kaggle":
                if not kaggle_username or not kaggle_key:
                    error_msg = "❌ **Error:** Kaggle username and API key are required"
                    return (
                        gr.Markdown(value=error_msg, visible=True),
                        self._get_registered_backends_text()
                    )
                backend = self._create_kaggle_backend(backend_id, kaggle_username, kaggle_key)
                self._save_to_env("KAGGLE_USERNAME", kaggle_username)
                self._save_to_env("KAGGLE_KEY", kaggle_key)
            elif backend_type == "HuggingFace":
                if not hf_token:
                    error_msg = "❌ **Error:** HuggingFace token is required"
                    return (
                        gr.Markdown(value=error_msg, visible=True),
                        self._get_registered_backends_text()
                    )
                backend = self._create_hf_backend(backend_id, hf_token)
                self._save_to_env("HF_TOKEN", hf_token)
            elif backend_type == "Google Colab":
                if not colab_client_id or not colab_client_secret or not colab_refresh_token:
                    error_msg = "❌ **Error:** Google OAuth credentials are required"
                    return (
                        gr.Markdown(value=error_msg, visible=True),
                        self._get_registered_backends_text()
                    )
                backend = self._create_colab_backend(backend_id, colab_client_id, colab_client_secret, colab_refresh_token)
                self._save_to_env("GOOGLE_CLIENT_ID", colab_client_id)
                self._save_to_env("GOOGLE_CLIENT_SECRET", colab_client_secret)
                self._save_to_env("GOOGLE_REFRESH_TOKEN", colab_refresh_token)
            else:
                error_msg = f"❌ **Error:** Backend type '{backend_type}' is not yet implemented"
                return (
                    gr.Markdown(value=error_msg, visible=True),
                    self._get_registered_backends_text()
                )
            
            # Register the backend
            self.backend_router.register_backend(backend)
            
            success_msg = f"""
✅ **Backend Registered Successfully!**

**Backend ID:** `{backend_id}`  
**Type:** {backend_type}  
**Status:** Active

You can now submit jobs to this backend.
"""
            
            return (
                gr.Markdown(value=success_msg, visible=True),
                self._get_registered_backends_text()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register backend: {e}", exc_info=True)
            error_msg = f"❌ **Error:** {str(e)}"
            return (
                gr.Markdown(value=error_msg, visible=True),
                self._get_registered_backends_text()
            )
    
    def _create_mock_backend(self, backend_id: str, supports_gpu: bool):
        """Create a mock backend for testing."""
        from notebook_ml_orchestrator.core.interfaces import Backend, Job
        from notebook_ml_orchestrator.core.models import BackendType, BackendCapabilities, HealthStatus
        
        class MockBackend(Backend):
            """Mock backend for testing."""
            
            def __init__(self, backend_id: str, supports_gpu: bool):
                self.id = backend_id
                self.type = BackendType.LOCAL_GPU if supports_gpu else BackendType.LOCAL_CPU
                self.capabilities = BackendCapabilities(
                    supports_gpu=supports_gpu,
                    max_job_duration_minutes=60,
                    supported_templates=["*"]  # Supports all templates
                )
            
            def execute_job(self, job: Job, template) -> dict:
                """Mock job execution."""
                return {
                    "status": "completed",
                    "message": "Mock execution successful",
                    "outputs": {"result": "mock_result"}
                }
            
            def check_health(self) -> HealthStatus:
                """Mock health check."""
                return HealthStatus.HEALTHY
            
            def estimate_cost(self, resource_estimate) -> float:
                """Mock cost estimation."""
                return 0.0  # Free for mock
            
            def supports_template(self, template_name: str) -> bool:
                """Mock template support."""
                return True  # Supports all templates
            
            def get_queue_length(self) -> int:
                """Mock queue length."""
                return 0
        
        return MockBackend(backend_id, supports_gpu)
    
    def _create_local_backend(self, backend_id: str, max_jobs: int):
        """Create a local backend."""
        try:
            from notebook_ml_orchestrator.core.backends.local_backend import LocalBackend
            return LocalBackend(
                backend_id=backend_id,
                config={
                    'options': {
                        'max_concurrent_jobs': max_jobs
                    }
                }
            )
        except ImportError:
            raise Exception("Local backend not available.")
    
    def _create_modal_backend(self, backend_id: str, token_id: str, token_secret: str):
        """Create a Modal backend."""
        try:
            from notebook_ml_orchestrator.core.backends.modal_backend import ModalBackend
            return ModalBackend(
                backend_id=backend_id, 
                config={
                    'credentials': {
                        'token_id': token_id,
                        'token_secret': token_secret
                    }
                }
            )
        except ImportError:
            raise Exception("Modal backend not available. Install modal package.")
    
    def _create_kaggle_backend(self, backend_id: str, username: str, key: str):
        """Create a Kaggle backend."""
        try:
            from notebook_ml_orchestrator.core.backends.kaggle_backend import KaggleBackend
            return KaggleBackend(
                backend_id=backend_id, 
                config={
                    'credentials': {
                        'username': username,
                        'key': key
                    }
                }
            )
        except ImportError:
            raise Exception("Kaggle backend not available. Install kaggle package.")
    
    def _create_hf_backend(self, backend_id: str, token: str):
        """Create a HuggingFace backend."""
        try:
            from notebook_ml_orchestrator.core.backends.huggingface_backend import HuggingFaceBackend
            return HuggingFaceBackend(
                backend_id=backend_id, 
                config={
                    'credentials': {
                        'token': token
                    }
                }
            )
        except ImportError:
            raise Exception("HuggingFace backend not available. Install huggingface_hub package.")

    def _create_colab_backend(self, backend_id: str, client_id: str, client_secret: str, refresh_token: str):
        """Create a Google Colab backend."""
        try:
            from notebook_ml_orchestrator.core.backends.colab_backend import ColabBackend
            return ColabBackend(
                backend_id=backend_id,
                config={
                    'credentials': {
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'refresh_token': refresh_token
                    }
                }
            )
        except ImportError:
            raise Exception("Colab backend not available. Install google-api-python-client package.")
