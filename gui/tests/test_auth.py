"""Unit tests for authentication and authorization module.

Tests cover:
- Authentication providers
- Session management with timeout
- Role-based access control
- Authentication middleware
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from gui.auth import (
    AuthenticationMiddleware,
    AuthenticationProvider,
    Permission,
    Role,
    Session,
    SessionManager,
    SimpleAuthProvider,
    User,
)


class TestUser:
    """Test User class and permission checking."""
    
    def test_user_creation(self):
        """Test creating a user with role."""
        user = User(username="testuser", role=Role.USER)
        assert user.username == "testuser"
        assert user.role == Role.USER
        assert user.user_id is not None
        assert isinstance(user.metadata, dict)
    
    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions."""
        admin = User(username="admin", role=Role.ADMIN)
        
        # Admin should have all permissions
        assert admin.has_permission(Permission.SUBMIT_JOB)
        assert admin.has_permission(Permission.VIEW_ALL_JOBS)
        assert admin.has_permission(Permission.VIEW_OWN_JOBS)
        assert admin.has_permission(Permission.EXECUTE_WORKFLOW)
        assert admin.has_permission(Permission.MANAGE_BACKENDS)
        assert admin.has_permission(Permission.VIEW_BACKEND_STATUS)
    
    def test_user_has_limited_permissions(self):
        """Test that user role has limited permissions."""
        user = User(username="user", role=Role.USER)
        
    