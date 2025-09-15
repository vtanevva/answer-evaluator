"""
Test cache validation functionality
"""

import os
import tempfile
import pytest

from services.embedding_storage import EmbeddingStorage


class TestCacheValidation:
    """Test embedding cache validation functionality"""

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
        """Create test data for embedding storage"""
        return {
            'embeddings': {1: [[0.1, 0.2]], 2: [[0.3, 0.4]]},
            'keywords': {1: [{'test'}], 2: [{'example'}]},
            'metadata': {
                1: {
                    'question_text': 'Test?',
                    'key_points_count': 1,
                    'key_points_texts': ['Test point']
                },
                2: {
                    'question_text': 'Example?',
                    'key_points_count': 1,
                    'key_points_texts': ['Example point']
                }
            }
        }

    def test_cache_validation_with_same_metadata(self, embedding_storage, test_data):
        """Test that cache loads successfully with same metadata"""
        # Save cache
        success = embedding_storage.cache_embeddings(
            test_data['embeddings'],
            test_data['keywords'],
            test_data['metadata']
        )
        assert success

        # Load with same metadata (should succeed)
        result = embedding_storage.load_cached_embeddings(test_data['metadata'])
        assert result is not None
        
        loaded_embeddings, loaded_keywords = result
        assert loaded_embeddings == test_data['embeddings']

    def test_cache_validation_with_modified_metadata(self, embedding_storage, test_data):
        """Test that cache validation detects metadata changes"""
        # Save cache
        success = embedding_storage.cache_embeddings(
            test_data['embeddings'],
            test_data['keywords'],
            test_data['metadata']
        )
        assert success

        # Modify metadata
        modified_metadata = test_data['metadata'].copy()
        modified_metadata[1] = modified_metadata[1].copy()
        modified_metadata[1]['question_text'] = 'Modified Test?'
        
        # Load with modified metadata (should fail validation)
        result = embedding_storage.load_cached_embeddings(modified_metadata)
        assert result is None, "Cache validation should have detected the change"

    def test_cache_cleanup(self, embedding_storage, test_data):
        """Test cache cleanup functionality"""
        # Save cache
        success = embedding_storage.cache_embeddings(
            test_data['embeddings'],
            test_data['keywords'],
            test_data['metadata']
        )
        assert success

        # Verify cache exists
        cache_info = embedding_storage.get_cache_info()
        assert cache_info is not None

        # Clear cache
        cleanup_success = embedding_storage.clear_cache()
        assert cleanup_success

        # Verify cache is cleared
        cache_info_after = embedding_storage.get_cache_info()
        assert cache_info_after is None
