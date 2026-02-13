"""
Template service for GUI interface.

This module provides business logic for template discovery, metadata retrieval,
and search functionality through the GUI interface.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import threading

from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class TemplateService(LoggerMixin):
    """Service for template discovery and metadata."""
    
    def __init__(
        self,
        template_registry: TemplateRegistry,
        cache_ttl_seconds: int = 300  # 5 minutes default cache TTL
    ):
        """
        Initialize template service.
        
        Args:
            template_registry: Template registry instance
            cache_ttl_seconds: Time-to-live for metadata cache in seconds
        """
        self.template_registry = template_registry
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Metadata cache: {template_name: (metadata_dict, timestamp)}
        self._metadata_cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        self._cache_lock = threading.RLock()
        
        self.logger.info(
            f"TemplateService initialized with cache TTL: {cache_ttl_seconds}s"
        )
    
    def get_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve templates optionally filtered by category.
        
        Args:
            category: Optional category to filter by (e.g., "audio", "vision", "language")
            
        Returns:
            List of template dictionaries with summary information:
            - name: Template name
            - category: Template category
            - description: Template description
            - version: Template version
            - gpu_required: Whether GPU is required
            - memory_mb: Memory requirement in MB
            - timeout_sec: Timeout in seconds
        """
        # Get templates from registry
        templates = self.template_registry.list_templates(category=category)
        
        # Convert to summary dictionaries
        result = []
        for template in templates:
            result.append({
                'name': template.name,
                'category': template.category,
                'description': template.description,
                'version': template.version,
                'gpu_required': template.gpu_required,
                'memory_mb': template.memory_mb,
                'timeout_sec': template.timeout_sec
            })
        
        self.logger.debug(
            f"Retrieved {len(result)} templates"
            f"{f' for category {category}' if category else ''}"
        )
        
        return result
    
    def get_template_metadata(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve template metadata with caching.
        
        This method implements metadata caching to reduce database queries.
        Cached metadata is stored with a timestamp and expires after cache_ttl_seconds.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary containing complete template metadata including:
            - name: Template name
            - category: Template category
            - description: Template description
            - version: Template version
            - inputs: List of input field definitions
            - outputs: List of output field definitions
            - gpu_required: Whether GPU is required
            - gpu_type: GPU type if required
            - memory_mb: Memory requirement
            - timeout_sec: Timeout in seconds
            - pip_packages: Required pip packages
            - supported_backends: List of backend IDs that support this template
            
            Returns None if template not found.
        """
        with self._cache_lock:
            # Check cache first
            if template_name in self._metadata_cache:
                cached_metadata, cached_time = self._metadata_cache[template_name]
                
                # Check if cache is still valid
                age = (datetime.now() - cached_time).total_seconds()
                if age < self.cache_ttl_seconds:
                    self.logger.debug(
                        f"Cache hit for template '{template_name}' "
                        f"(age: {age:.1f}s)"
                    )
                    return cached_metadata
                else:
                    self.logger.debug(
                        f"Cache expired for template '{template_name}' "
                        f"(age: {age:.1f}s, TTL: {self.cache_ttl_seconds}s)"
                    )
            
            # Cache miss or expired - fetch from registry
            metadata = self.template_registry.get_template_metadata(template_name)
            
            if metadata is None:
                self.logger.warning(f"Template '{template_name}' not found")
                return None
            
            # Add supported backends to metadata
            supported_backends = self.template_registry.get_supported_backends(template_name)
            metadata['supported_backends'] = supported_backends
            
            # Store in cache
            self._metadata_cache[template_name] = (metadata, datetime.now())
            
            self.logger.debug(f"Cached metadata for template '{template_name}'")
            
            return metadata
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """
        Search templates by name, category, or capability.
        
        Performs case-insensitive search across:
        - Template name
        - Template category
        - Template description
        - GPU type (if applicable)
        
        Args:
            query: Search query string
            
        Returns:
            List of matching template dictionaries with summary information
        """
        if not query:
            # Empty query returns all templates
            return self.get_templates()
        
        query_lower = query.lower().strip()
        
        # Get all templates
        all_templates = self.template_registry.list_templates()
        
        # Search across multiple fields
        matching_templates = []
        for template in all_templates:
            # Check name
            if query_lower in template.name.lower():
                matching_templates.append(template)
                continue
            
            # Check category
            if query_lower in template.category.lower():
                matching_templates.append(template)
                continue
            
            # Check description
            if query_lower in template.description.lower():
                matching_templates.append(template)
                continue
            
            # Check GPU type if applicable
            if template.gpu_required and hasattr(template, 'gpu_type'):
                if query_lower in template.gpu_type.lower():
                    matching_templates.append(template)
                    continue
        
        # Convert to summary dictionaries
        result = []
        for template in matching_templates:
            result.append({
                'name': template.name,
                'category': template.category,
                'description': template.description,
                'version': template.version,
                'gpu_required': template.gpu_required,
                'memory_mb': template.memory_mb,
                'timeout_sec': template.timeout_sec
            })
        
        self.logger.debug(
            f"Search query '{query}' returned {len(result)} templates"
        )
        
        return result
    
    def clear_cache(self, template_name: Optional[str] = None) -> None:
        """
        Clear metadata cache.
        
        Args:
            template_name: Optional template name to clear specific entry.
                          If None, clears entire cache.
        """
        with self._cache_lock:
            if template_name is None:
                cache_size = len(self._metadata_cache)
                self._metadata_cache.clear()
                self.logger.info(f"Cleared entire metadata cache ({cache_size} entries)")
            elif template_name in self._metadata_cache:
                del self._metadata_cache[template_name]
                self.logger.info(f"Cleared cache for template '{template_name}'")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics:
            - total_entries: Number of cached entries
            - cache_ttl_seconds: Cache TTL setting
            - cached_templates: List of cached template names with ages
        """
        with self._cache_lock:
            now = datetime.now()
            cached_templates = []
            
            for template_name, (_, cached_time) in self._metadata_cache.items():
                age = (now - cached_time).total_seconds()
                cached_templates.append({
                    'template_name': template_name,
                    'age_seconds': age,
                    'expired': age >= self.cache_ttl_seconds
                })
            
            return {
                'total_entries': len(self._metadata_cache),
                'cache_ttl_seconds': self.cache_ttl_seconds,
                'cached_templates': cached_templates
            }
