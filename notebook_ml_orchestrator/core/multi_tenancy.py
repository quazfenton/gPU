"""
Multi-tenancy Support for Notebook ML Orchestrator.

This module provides multi-tenancy capabilities for:
- Tenant isolation
- Resource quotas
- User management
- Team collaboration
- Billing and cost tracking

Supports both single-tenant and multi-tenant deployments.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TenantRole(Enum):
    """Tenant user roles."""
    OWNER = "owner"  # Full control
    ADMIN = "admin"  # Administrative control
    MEMBER = "member"  # Standard member
    VIEWER = "viewer"  # Read-only
    GUEST = "guest"  # Limited access


class ResourceQuota(Enum):
    """Resource quota types."""
    MAX_JOBS_PER_DAY = "max_jobs_per_day"
    MAX_CONCURRENT_JOBS = "max_concurrent_jobs"
    MAX_WORKFLOWS = "max_workflows"
    MAX_STORAGE_GB = "max_storage_gb"
    MAX_GPU_HOURS = "max_gpu_hours"
    MAX_BUDGET_USD = "max_budget_usd"


@dataclass
class Tenant:
    """Tenant organization."""
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    plan: str = "free"  # free, pro, enterprise
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource quotas
    quotas: Dict[ResourceQuota, Any] = field(default_factory=dict)
    
    # Usage tracking
    usage: Dict[str, Any] = field(default_factory=dict)
    
    # Billing
    billing_email: Optional[str] = None
    payment_method: Optional[str] = None


@dataclass
class TenantUser:
    """Tenant user membership."""
    id: str
    tenant_id: str
    user_id: str  # Reference to user in auth system
    role: TenantRole = TenantRole.MEMBER
    joined_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Resource usage record."""
    id: str
    tenant_id: str
    user_id: str
    resource_type: str
    amount: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TenantManager:
    """
    Multi-tenancy manager.
    
    Manages tenants, users, quotas, and resource tracking.
    
    Example:
        tenant_manager = TenantManager()
        
        # Create tenant
        tenant = tenant_manager.create_tenant("Acme Corp", plan="pro")
        
        # Add user to tenant
        tenant_manager.add_user_to_tenant(tenant.id, "user-123", TenantRole.MEMBER)
        
        # Check quota
        if tenant_manager.check_quota(tenant.id, "jobs_per_day"):
            # Allow job submission
            pass
    """
    
    def __init__(self):
        """Initialize tenant manager."""
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_users: Dict[str, Dict[str, TenantUser]] = {}  # tenant_id -> user_id -> TenantUser
        self.usage_records: Dict[str, List[ResourceUsage]] = {}  # tenant_id -> list of usage records
        self._default_quotas = self._get_default_quotas()
        
        logger.info("TenantManager initialized")
    
    def _get_default_quotas(self) -> Dict[str, Dict[ResourceQuota, Any]]:
        """Get default quotas for each plan."""
        return {
            'free': {
                ResourceQuota.MAX_JOBS_PER_DAY: 10,
                ResourceQuota.MAX_CONCURRENT_JOBS: 2,
                ResourceQuota.MAX_WORKFLOWS: 5,
                ResourceQuota.MAX_STORAGE_GB: 1,
                ResourceQuota.MAX_GPU_HOURS: 5,
                ResourceQuota.MAX_BUDGET_USD: 0,
            },
            'pro': {
                ResourceQuota.MAX_JOBS_PER_DAY: 100,
                ResourceQuota.MAX_CONCURRENT_JOBS: 10,
                ResourceQuota.MAX_WORKFLOWS: 50,
                ResourceQuota.MAX_STORAGE_GB: 50,
                ResourceQuota.MAX_GPU_HOURS: 100,
                ResourceQuota.MAX_BUDGET_USD: 100,
            },
            'enterprise': {
                ResourceQuota.MAX_JOBS_PER_DAY: 1000,
                ResourceQuota.MAX_CONCURRENT_JOBS: 100,
                ResourceQuota.MAX_WORKFLOWS: 500,
                ResourceQuota.MAX_STORAGE_GB: 1000,
                ResourceQuota.MAX_GPU_HOURS: 10000,
                ResourceQuota.MAX_BUDGET_USD: 10000,
            }
        }
    
    def create_tenant(
        self,
        name: str,
        plan: str = "free",
        billing_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            plan: Plan type (free, pro, enterprise)
            billing_email: Billing email address
            metadata: Additional metadata
            
        Returns:
            Created tenant
        """
        tenant_id = f"tenant-{secrets.token_hex(8)}"
        
        # Get default quotas for plan
        quotas = self._default_quotas.get(plan, self._default_quotas['free']).copy()
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            plan=plan,
            billing_email=billing_email,
            metadata=metadata or {},
            quotas=quotas
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_users[tenant_id] = {}
        self.usage_records[tenant_id] = []
        
        logger.info(f"Created tenant: {name} ({tenant_id}), plan: {plan}")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def list_tenants(self, active_only: bool = True) -> List[Tenant]:
        """List all tenants."""
        tenants = list(self.tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.active]
        return tenants
    
    def update_tenant(self, tenant_id: str, **kwargs) -> bool:
        """
        Update tenant properties.
        
        Args:
            tenant_id: Tenant ID
            **kwargs: Properties to update
            
        Returns:
            True if updated
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        logger.info(f"Updated tenant: {tenant_id}")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Soft delete a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if deleted
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        tenant.active = False
        logger.info(f"Deleted tenant: {tenant_id}")
        return True
    
    def add_user_to_tenant(
        self,
        tenant_id: str,
        user_id: str,
        role: TenantRole = TenantRole.MEMBER,
        permissions: Optional[Set[str]] = None
    ) -> Optional[TenantUser]:
        """
        Add user to tenant.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            role: User role
            permissions: Additional permissions
            
        Returns:
            TenantUser or None if already exists
        """
        if tenant_id not in self.tenant_users:
            logger.error(f"Tenant {tenant_id} not found")
            return None
        
        if user_id in self.tenant_users[tenant_id]:
            logger.warning(f"User {user_id} already in tenant {tenant_id}")
            return self.tenant_users[tenant_id][user_id]
        
        tenant_user = TenantUser(
            id=f"tu-{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            permissions=permissions or self._get_default_permissions(role)
        )
        
        self.tenant_users[tenant_id][user_id] = tenant_user
        
        logger.info(f"Added user {user_id} to tenant {tenant_id} as {role.value}")
        return tenant_user
    
    def remove_user_from_tenant(self, tenant_id: str, user_id: str) -> bool:
        """Remove user from tenant."""
        if tenant_id in self.tenant_users and user_id in self.tenant_users[tenant_id]:
            del self.tenant_users[tenant_id][user_id]
            logger.info(f"Removed user {user_id} from tenant {tenant_id}")
            return True
        return False
    
    def get_user_tenants(self, user_id: str) -> List[Tenant]:
        """Get all tenants a user belongs to."""
        tenant_ids = [
            tenant_id for tenant_id, users in self.tenant_users.items()
            if user_id in users
        ]
        return [self.tenants[tid] for tid in tenant_ids if tid in self.tenants]
    
    def check_user_permission(self, tenant_id: str, user_id: str, permission: str) -> bool:
        """Check if user has permission in tenant."""
        if tenant_id not in self.tenant_users:
            return False
        
        tenant_user = self.tenant_users[tenant_id].get(user_id)
        if not tenant_user or not tenant_user.active:
            return False
        
        return permission in tenant_user.permissions
    
    def _get_default_permissions(self, role: TenantRole) -> Set[str]:
        """Get default permissions for role."""
        permissions = {
            TenantRole.OWNER: {
                'jobs.submit', 'jobs.view', 'jobs.cancel',
                'workflows.create', 'workflows.view', 'workflows.delete',
                'templates.view', 'templates.create',
                'billing.view', 'billing.manage',
                'users.manage', 'settings.manage'
            },
            TenantRole.ADMIN: {
                'jobs.submit', 'jobs.view', 'jobs.cancel',
                'workflows.create', 'workflows.view', 'workflows.delete',
                'templates.view', 'templates.create',
                'billing.view',
                'users.manage'
            },
            TenantRole.MEMBER: {
                'jobs.submit', 'jobs.view',
                'workflows.create', 'workflows.view',
                'templates.view'
            },
            TenantRole.VIEWER: {
                'jobs.view',
                'workflows.view',
                'templates.view'
            },
            TenantRole.GUEST: {
                'jobs.view'
            }
        }
        return permissions.get(role, set())
    
    def check_quota(self, tenant_id: str, quota_type: ResourceQuota, current_usage: float = 0) -> bool:
        """
        Check if tenant has quota remaining.
        
        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
            current_usage: Current usage amount
            
        Returns:
            True if within quota
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        quota_limit = tenant.quotas.get(quota_type)
        if quota_limit is None:
            return True  # No quota set
        
        # Get current usage
        usage = self.get_usage(tenant_id, quota_type.value)
        
        return (usage + current_usage) <= quota_limit
    
    def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        resource_type: str,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record resource usage."""
        if tenant_id not in self.usage_records:
            self.usage_records[tenant_id] = []
        
        usage = ResourceUsage(
            id=f"usage-{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            amount=amount,
            metadata=metadata or {}
        )
        
        self.usage_records[tenant_id].append(usage)
        
        # Update tenant usage summary
        if tenant_id in self.tenants:
            if resource_type not in self.tenants[tenant_id].usage:
                self.tenants[tenant_id].usage[resource_type] = 0
            self.tenants[tenant_id].usage[resource_type] += amount
    
    def get_usage(self, tenant_id: str, resource_type: str, days: int = 30) -> float:
        """Get usage for resource type over last N days."""
        if tenant_id not in self.usage_records:
            return 0.0
        
        cutoff = datetime.now() - timedelta(days=days)
        total = sum(
            record.amount for record in self.usage_records[tenant_id]
            if record.resource_type == resource_type and record.timestamp > cutoff
        )
        
        return total
    
    def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant dashboard with usage and quota information."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}
        
        dashboard = {
            'tenant': {
                'id': tenant.id,
                'name': tenant.name,
                'plan': tenant.plan,
                'active': tenant.active
            },
            'usage': {},
            'quotas': {},
            'quota_utilization': {}
        }
        
        # Get usage for each quota type
        for quota_type in ResourceQuota:
            usage_key = quota_type.value
            usage = self.get_usage(tenant_id, usage_key)
            limit = tenant.quotas.get(quota_type)
            
            dashboard['usage'][usage_key] = usage
            dashboard['quotas'][usage_key] = limit
            
            if limit:
                dashboard['quota_utilization'][usage_key] = (usage / limit) * 100
            else:
                dashboard['quota_utilization'][usage_key] = 0
        
        return dashboard


# Module-level instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get or create module-level tenant manager."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
