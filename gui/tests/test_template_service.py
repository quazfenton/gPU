"""Tests for TemplateService."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import time

from gui.services.template_service import TemplateService
from templates.base import Template, InputField, OutputField


class MockTemplate(Template):
    """Mock template for testing."""
    
    def __init__(self, name="test_template", category="test", description="Test template"):
        super().__init__()
        self.name = name
        self.category = category
        self.description = description
        self.version = "1.0.0"
        self.gpu_required = False
        self.memory_mb = 1024
        self.timeout_sec = 300
        self.inputs = [
            InputField(name="input1", type="string", description="Test input", required=True)
        ]
        self.outputs = [
            OutputField(name="output1", type="string", description="Test output")
        ]
    
    def run(self, **kwargs):
        """Mock run method (required abstract method)."""
        return {"output1": "test_result"}
    
    def execute(self, inputs):
        """Mock execute method."""
        return {"output1": "test_result"}


class TestTemplateService:
    """Test suite for TemplateService."""
    
    @pytest.fixture
    def mock_template_registry(self):
        """Create mock template registry."""
        return Mock()
    
    @pytest.fixture
    def template_service(self, mock_template_registry):
        """Create TemplateService instance with mocks."""
        return TemplateService(mock_template_registry, cache_ttl_seconds=2)
    
    @pytest.fixture
    def sample_templates(self):
        """Create sample templates for testing."""
        return [
            MockTemplate(name="audio_gen", category="audio", description="Generate audio"),
            MockTemplate(name="image_gen", category="vision", description="Generate images"),
            MockTemplate(name="text_gen", category="language", description="Generate text"),
        ]
    
    def test_get_templates_all(self, template_service, mock_template_registry, sample_templates):
        """Test retrieving all templates without category filter."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.get_templates()
        
        # Verify
        assert len(result) == 3
        assert result[0]['name'] == "audio_gen"
        assert result[0]['category'] == "audio"
        assert result[0]['description'] == "Generate audio"
        assert result[0]['gpu_required'] is False
        assert result[0]['memory_mb'] == 1024
        assert result[0]['timeout_sec'] == 300
        
        mock_template_registry.list_templates.assert_called_once_with(category=None)
    
    def test_get_templates_by_category(self, template_service, mock_template_registry, sample_templates):
        """Test retrieving templates filtered by category."""
        # Setup
        audio_templates = [t for t in sample_templates if t.category == "audio"]
        mock_template_registry.list_templates.return_value = audio_templates
        
        # Execute
        result = template_service.get_templates(category="audio")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "audio_gen"
        assert result[0]['category'] == "audio"
        
        mock_template_registry.list_templates.assert_called_once_with(category="audio")
    
    def test_get_templates_empty(self, template_service, mock_template_registry):
        """Test retrieving templates when none exist."""
        # Setup
        mock_template_registry.list_templates.return_value = []
        
        # Execute
        result = template_service.get_templates()
        
        # Verify
        assert len(result) == 0
    
    def test_get_template_metadata_success(self, template_service, mock_template_registry):
        """Test retrieving template metadata successfully."""
        # Setup
        metadata = {
            'name': 'test_template',
            'category': 'test',
            'description': 'Test template',
            'version': '1.0.0',
            'inputs': [{'name': 'input1', 'type': 'string'}],
            'outputs': [{'name': 'output1', 'type': 'string'}],
            'gpu_required': False,
            'memory_mb': 1024,
            'timeout_sec': 300
        }
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = ["modal", "huggingface"]
        
        # Execute
        result = template_service.get_template_metadata("test_template")
        
        # Verify
        assert result is not None
        assert result['name'] == 'test_template'
        assert result['supported_backends'] == ["modal", "huggingface"]
        
        mock_template_registry.get_template_metadata.assert_called_once_with("test_template")
        mock_template_registry.get_supported_backends.assert_called_once_with("test_template")
    
    def test_get_template_metadata_not_found(self, template_service, mock_template_registry):
        """Test retrieving metadata for non-existent template."""
        # Setup
        mock_template_registry.get_template_metadata.return_value = None
        
        # Execute
        result = template_service.get_template_metadata("nonexistent")
        
        # Verify
        assert result is None
    
    def test_get_template_metadata_caching(self, template_service, mock_template_registry):
        """Test that template metadata is cached."""
        # Setup
        metadata = {
            'name': 'test_template',
            'category': 'test',
            'description': 'Test template'
        }
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = ["modal"]
        
        # Execute - first call
        result1 = template_service.get_template_metadata("test_template")
        
        # Execute - second call (should use cache)
        result2 = template_service.get_template_metadata("test_template")
        
        # Verify
        assert result1 == result2
        # Registry should only be called once (second call uses cache)
        assert mock_template_registry.get_template_metadata.call_count == 1
        assert mock_template_registry.get_supported_backends.call_count == 1
    
    def test_get_template_metadata_cache_expiration(self, template_service, mock_template_registry):
        """Test that cache expires after TTL."""
        # Setup
        metadata = {
            'name': 'test_template',
            'category': 'test',
            'description': 'Test template'
        }
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = ["modal"]
        
        # Execute - first call
        result1 = template_service.get_template_metadata("test_template")
        
        # Wait for cache to expire (TTL is 2 seconds in fixture)
        time.sleep(2.1)
        
        # Execute - second call (cache should be expired)
        result2 = template_service.get_template_metadata("test_template")
        
        # Verify
        assert result1 == result2
        # Registry should be called twice (cache expired)
        assert mock_template_registry.get_template_metadata.call_count == 2
        assert mock_template_registry.get_supported_backends.call_count == 2
    
    def test_search_templates_by_name(self, template_service, mock_template_registry, sample_templates):
        """Test searching templates by name."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("audio")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "audio_gen"
    
    def test_search_templates_by_category(self, template_service, mock_template_registry, sample_templates):
        """Test searching templates by category."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("vision")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "image_gen"
        assert result[0]['category'] == "vision"
    
    def test_search_templates_by_description(self, template_service, mock_template_registry, sample_templates):
        """Test searching templates by description."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("Generate text")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "text_gen"
    
    def test_search_templates_case_insensitive(self, template_service, mock_template_registry, sample_templates):
        """Test that search is case-insensitive."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("AUDIO")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "audio_gen"
    
    def test_search_templates_empty_query(self, template_service, mock_template_registry, sample_templates):
        """Test that empty query returns all templates."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("")
        
        # Verify
        assert len(result) == 3
    
    def test_search_templates_no_matches(self, template_service, mock_template_registry, sample_templates):
        """Test searching with no matches."""
        # Setup
        mock_template_registry.list_templates.return_value = sample_templates
        
        # Execute
        result = template_service.search_templates("nonexistent")
        
        # Verify
        assert len(result) == 0
    
    def test_search_templates_by_gpu_type(self, template_service, mock_template_registry):
        """Test searching templates by GPU type."""
        # Setup
        gpu_template = MockTemplate(name="gpu_template", category="test", description="GPU template")
        gpu_template.gpu_required = True
        gpu_template.gpu_type = "A100"
        
        mock_template_registry.list_templates.return_value = [gpu_template]
        
        # Execute
        result = template_service.search_templates("A100")
        
        # Verify
        assert len(result) == 1
        assert result[0]['name'] == "gpu_template"
    
    def test_clear_cache_all(self, template_service, mock_template_registry):
        """Test clearing entire cache."""
        # Setup - populate cache
        metadata = {'name': 'test1', 'category': 'test'}
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = []
        
        template_service.get_template_metadata("test1")
        template_service.get_template_metadata("test2")
        
        # Verify cache has entries
        stats_before = template_service.get_cache_stats()
        assert stats_before['total_entries'] == 2
        
        # Execute
        template_service.clear_cache()
        
        # Verify cache is empty
        stats_after = template_service.get_cache_stats()
        assert stats_after['total_entries'] == 0
    
    def test_clear_cache_specific(self, template_service, mock_template_registry):
        """Test clearing specific cache entry."""
        # Setup - populate cache
        metadata = {'name': 'test', 'category': 'test'}
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = []
        
        template_service.get_template_metadata("test1")
        template_service.get_template_metadata("test2")
        
        # Verify cache has 2 entries
        stats_before = template_service.get_cache_stats()
        assert stats_before['total_entries'] == 2
        
        # Execute - clear only test1
        template_service.clear_cache("test1")
        
        # Verify only test1 is removed
        stats_after = template_service.get_cache_stats()
        assert stats_after['total_entries'] == 1
        
        # Verify test2 is still cached
        template_names = [t['template_name'] for t in stats_after['cached_templates']]
        assert "test2" in template_names
        assert "test1" not in template_names
    
    def test_get_cache_stats(self, template_service, mock_template_registry):
        """Test getting cache statistics."""
        # Setup - populate cache
        metadata = {'name': 'test', 'category': 'test'}
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = []
        
        template_service.get_template_metadata("test1")
        template_service.get_template_metadata("test2")
        
        # Execute
        stats = template_service.get_cache_stats()
        
        # Verify
        assert stats['total_entries'] == 2
        assert stats['cache_ttl_seconds'] == 2
        assert len(stats['cached_templates']) == 2
        
        # Check template details
        template_names = [t['template_name'] for t in stats['cached_templates']]
        assert "test1" in template_names
        assert "test2" in template_names
        
        # Check age and expiration
        for template_info in stats['cached_templates']:
            assert 'age_seconds' in template_info
            assert 'expired' in template_info
            assert template_info['age_seconds'] >= 0
            assert template_info['expired'] is False  # Should not be expired yet
    
    def test_get_cache_stats_with_expired_entries(self, template_service, mock_template_registry):
        """Test cache stats showing expired entries."""
        # Setup - populate cache
        metadata = {'name': 'test', 'category': 'test'}
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = []
        
        template_service.get_template_metadata("test1")
        
        # Wait for cache to expire
        time.sleep(2.1)
        
        # Execute
        stats = template_service.get_cache_stats()
        
        # Verify
        assert stats['total_entries'] == 1
        assert stats['cached_templates'][0]['expired'] is True
        assert stats['cached_templates'][0]['age_seconds'] > 2.0
    
    def test_concurrent_cache_access(self, template_service, mock_template_registry):
        """Test thread-safe cache access."""
        import threading
        
        # Setup
        metadata = {'name': 'test', 'category': 'test'}
        mock_template_registry.get_template_metadata.return_value = metadata
        mock_template_registry.get_supported_backends.return_value = []
        
        # Execute - multiple threads accessing cache
        def access_cache():
            for i in range(10):
                template_service.get_template_metadata(f"test{i % 3}")
        
        threads = [threading.Thread(target=access_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify - no exceptions and cache is consistent
        stats = template_service.get_cache_stats()
        assert stats['total_entries'] <= 3  # Should have at most 3 unique templates
    
    def test_search_templates_multiple_matches(self, template_service, mock_template_registry):
        """Test searching templates with multiple matches."""
        # Setup
        templates = [
            MockTemplate(name="gen_audio", category="audio", description="Generate audio"),
            MockTemplate(name="gen_image", category="vision", description="Generate images"),
            MockTemplate(name="gen_text", category="language", description="Generate text"),
        ]
        mock_template_registry.list_templates.return_value = templates
        
        # Execute - search for "gen" which matches all names
        result = template_service.search_templates("gen")
        
        # Verify
        assert len(result) == 3
        names = [t['name'] for t in result]
        assert "gen_audio" in names
        assert "gen_image" in names
        assert "gen_text" in names
    
    def test_get_templates_preserves_all_fields(self, template_service, mock_template_registry):
        """Test that get_templates preserves all required fields."""
        # Setup
        template = MockTemplate(name="test", category="test", description="Test")
        template.gpu_required = True
        template.memory_mb = 2048
        template.timeout_sec = 600
        
        mock_template_registry.list_templates.return_value = [template]
        
        # Execute
        result = template_service.get_templates()
        
        # Verify all fields are present
        assert len(result) == 1
        t = result[0]
        assert t['name'] == "test"
        assert t['category'] == "test"
        assert t['description'] == "Test"
        assert t['version'] == "1.0.0"
        assert t['gpu_required'] is True
        assert t['memory_mb'] == 2048
        assert t['timeout_sec'] == 600
