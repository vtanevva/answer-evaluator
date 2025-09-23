"""
Test embedding cache functionality without API calls
"""

import os
import tempfile
import pytest

from services.embedding_storage import EmbeddingStorage


class TestEmbeddingCache:
    """Test embedding cache save/load functionality"""

    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file_path = temp_file.name
        yield temp_file_path
        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    @pytest.fixture
    def embedding_storage(self, temp_cache_file):
        """Create an embedding storage instance with temp file"""
        return EmbeddingStorage(temp_cache_file)

    @pytest.fixture
    def test_data(self):
        """Create comprehensive test data"""
        return {
            'metadata': {
                'created_at': '2025-01-01T00:00:00Z',
                'openai_model': 'test-model',
                'total_questions': 2,
                'total_embeddings': 4
            },
            'embeddings': {
                1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            },
        }
