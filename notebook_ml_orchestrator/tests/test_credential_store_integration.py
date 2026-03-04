"""
Test CredentialStore integration with Modal backend.

This test suite verifies that:
1. Credentials are retrieved from CredentialStore when available
2. Fallback to config works when CredentialStore is not available
3. Credentials are cleared from memory after use
4. Security logging works correctly
5. Error handling is proper
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from notebook_ml_orchestrator.core.backends.modal_backend import ModalBackend
from notebook_ml_orchestrator.core.models import BackendType
from notebook_ml_orchestrator.core.exceptions import BackendAuthenticationError, BackendConnectionError
from notebook_ml_orchestrator.security.credential_store import CredentialStore
from notebook_ml_orchestrator.security.security_logger import SecurityLogger


class TestCredentialStoreIntegration:
    """Test CredentialStore integration with Modal backend."""

    def test_init_with_credential_store(self):
        """Test Modal backend initialization with CredentialStore."""
        credential_store = Mock(spec=CredentialStore)
        security_logger = Mock(spec=SecurityLogger)
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store,
            security_logger=security_logger
        )
        
        assert backend.credential_store is credential_store
        assert backend.security_logger is security_logger
        assert backend._credentials is None
        assert backend._credentials_loaded_at is None

    def test_init_without_credential_store(self):
        """Test Modal backend initialization without CredentialStore (backward compatibility)."""
        backend = ModalBackend(
            backend_id="modal-test",
            config={"credentials": {"token_id": "test", "token_secret": "test"}}
        )
        
        assert backend.credential_store is None
        assert backend.security_logger is None

    @patch.dict(os.environ, {}, clear=False)
    def test_get_credentials_from_credential_store(self):
        """Test retrieving credentials from CredentialStore."""
        # Mock CredentialStore
        credential_store = Mock(spec=CredentialStore)
        credential_store.get_credential.side_effect = lambda service, key: f"{service}_{key}_value"
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store
        )
        
        credentials = backend._get_credentials()
        
        assert credentials == {
            "token_id": "modal_token_id_value",
            "token_secret": "modal_token_secret_value"
        }
        assert credential_store.get_credential.call_count == 2
        assert backend._credentials_loaded_at is not None

    @patch.dict(os.environ, {}, clear=False)
    def test_get_credentials_fallback_to_config(self):
        """Test fallback to config when CredentialStore is not available."""
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "config_token_id",
                    "token_secret": "config_token_secret"
                }
            }
        )
        
        credentials = backend._get_credentials()
        
        assert credentials == {
            "token_id": "config_token_id",
            "token_secret": "config_token_secret"
        }

    @patch.dict(os.environ, {}, clear=False)
    def test_get_credentials_caching(self):
        """Test that credentials are cached for 5 minutes."""
        credential_store = Mock(spec=CredentialStore)
        credential_store.get_credential.side_effect = lambda service, key: f"{service}_{key}_value"
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store
        )
        
        # First call - should retrieve from CredentialStore
        credentials1 = backend._get_credentials()
        call_count_1 = credential_store.get_credential.call_count
        
        # Second call - should use cached credentials
        credentials2 = backend._get_credentials()
        call_count_2 = credential_store.get_credential.call_count
        
        assert call_count_1 == 2  # Called twice (token_id and token_secret)
        assert call_count_2 == 2  # Still 2 (cached)
        assert credentials1 == credentials2

    @patch.dict(os.environ, {}, clear=False)
    def test_get_credentials_cache_expiration(self):
        """Test that credentials expire after 5 minutes."""
        credential_store = Mock(spec=CredentialStore)
        credential_store.get_credential.side_effect = lambda service, key: f"{service}_{key}_value"
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store
        )
        
        # Get credentials
        backend._get_credentials()
        call_count_1 = credential_store.get_credential.call_count
        
        # Manually expire cache
        backend._credentials_loaded_at = datetime.now() - timedelta(minutes=6)
        
        # Get credentials again - should refresh
        backend._get_credentials()
        call_count_2 = credential_store.get_credential.call_count
        
        assert call_count_2 > call_count_1  # Called again

    def test_get_credentials_missing_raises_error(self):
        """Test that missing credentials raise BackendAuthenticationError."""
        backend = ModalBackend(
            backend_id="modal-test",
            config={}  # No credentials
        )
        
        with pytest.raises(BackendAuthenticationError) as exc_info:
            backend._get_credentials()
        
        assert "Modal credentials not configured" in str(exc_info.value)

    def test_clear_credentials(self):
        """Test that credentials are cleared from memory."""
        backend = ModalBackend(backend_id="modal-test")
        backend._credentials = {"token_id": "test", "token_secret": "test"}
        backend._credentials_loaded_at = datetime.now()
        
        backend._clear_credentials()
        
        assert backend._credentials is None
        assert backend._credentials_loaded_at is None

    @patch('modal.App')
    def test_authenticate_success(self, mock_modal_app):
        """Test successful authentication."""
        # Mock Modal App creation
        mock_modal_app.return_value = Mock()
        
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "valid_token_id",
                    "token_secret": "valid_token_secret"
                }
            }
        )
        
        backend._authenticate()
        
        assert backend._authenticated is True
        assert os.environ.get('MODAL_TOKEN_ID') == "valid_token_id"
        assert os.environ.get('MODAL_TOKEN_SECRET') == "valid_token_secret"
        
        # Verify credentials were cleared
        assert backend._credentials is None

    @patch('modal.App')
    def test_authenticate_invalid_credentials(self, mock_modal_app):
        """Test authentication with invalid credentials."""
        # Mock Modal App to raise authentication error
        mock_modal_app.side_effect = Exception("Unauthorized: Invalid credentials")
        
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "invalid_token_id",
                    "token_secret": "invalid_token_secret"
                }
            }
        )
        
        with pytest.raises(BackendAuthenticationError) as exc_info:
            backend._authenticate()
        
        assert "Invalid credentials" in str(exc_info.value)
        assert backend._authenticated is False

    def test_authenticate_sdk_not_installed(self):
        """Test authentication when Modal SDK is not installed."""
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "test",
                    "token_secret": "test"
                }
            }
        )
        
        with patch.dict('sys.modules', {'modal': None}):
            with pytest.raises(BackendConnectionError) as exc_info:
                backend._authenticate()
            
            assert "Modal SDK not installed" in str(exc_info.value)

    @patch('modal.App')
    def test_authenticate_logs_success(self, mock_modal_app):
        """Test that successful authentication is logged."""
        mock_modal_app.return_value = Mock()
        
        security_logger = Mock(spec=SecurityLogger)
        
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "test",
                    "token_secret": "test"
                }
            },
            security_logger=security_logger
        )
        
        backend._authenticate()
        
        # Verify security logging
        assert security_logger.log_auth_success.called
        call_args = security_logger.log_auth_success.call_args
        assert call_args[1]['username'] == 'modal_backend'
        assert call_args[1]['details']['backend_id'] == 'modal-test'

    @patch('modal.App')
    def test_authenticate_logs_failure(self, mock_modal_app):
        """Test that authentication failure is logged."""
        mock_modal_app.side_effect = Exception("Unauthorized")
        
        security_logger = Mock(spec=SecurityLogger)
        
        backend = ModalBackend(
            backend_id="modal-test",
            config={
                "credentials": {
                    "token_id": "invalid",
                    "token_secret": "invalid"
                }
            },
            security_logger=security_logger
        )
        
        with pytest.raises(BackendAuthenticationError):
            backend._authenticate()
        
        # Verify security logging
        assert security_logger.log_auth_failure.called


class TestSecurityLogging:
    """Test security logging integration."""

    def test_credential_access_logged(self):
        """Test that credential access is logged."""
        credential_store = Mock(spec=CredentialStore)
        credential_store.get_credential.return_value = "test_value"
        security_logger = Mock(spec=SecurityLogger)
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store,
            security_logger=security_logger
        )
        
        backend._get_credentials()
        
        # Verify credential access was logged
        assert security_logger.log_credential_access.called

    def test_credential_access_failure_logged(self):
        """Test that credential access failure is logged."""
        credential_store = Mock(spec=CredentialStore)
        credential_store.get_credential.side_effect = Exception("Access denied")
        security_logger = Mock(spec=SecurityLogger)
        
        backend = ModalBackend(
            backend_id="modal-test",
            credential_store=credential_store,
            security_logger=security_logger
        )
        
        # Should fallback to config without raising
        backend._get_credentials()
        
        # Verify failure was logged
        assert security_logger.log_credential_access.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
