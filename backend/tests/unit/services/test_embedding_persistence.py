"""
Test embedding persistence functionality
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from services.question_service import QuestionService
from services.evaluation_service import EvaluationService
from services.embedding_storage import EmbeddingStorage
from core.config import settings


class TestEmbeddingPersistence:
    """Test embedding persistence functionality"""

    @pytest.fixture
    def question_service(self):
        """Create a question service with test data"""
        service = QuestionService()
        # Mock the load_questions_bank method to avoid file dependencies
        service._questions_bank = {
            1: {
                'question_id': 1,
                'question_text': 'Test question 1',
                'key_points': [{'text': 'Test point 1'}]
            },
            2: {
                'question_id': 2,
                'question_text': 'Test question 2', 
                'key_points': [{'text': 'Test point 2'}]
            }
        }
        return service

    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file path"""
        fd, temp_file_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)  # Close the file descriptor
        os.unlink(temp_file_path)  # Remove the file so it doesn't exist initially
        yield temp_file_path
        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        return MagicMock()

    def test_embedding_cache_creation(self, question_service, temp_cache_file, mock_openai_client):
        """Test that embeddings cache is created when precompute_embeddings is called"""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)
            
        # Create evaluation service with custom embedding storage
        evaluation_service = EvaluationService(question_service, mock_openai_client)
        evaluation_service._embedding_storage = EmbeddingStorage(temp_cache_file)
        
        # Mock the embedding service to avoid API calls
        with patch.object(evaluation_service._embedding_service, 'get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            # Cache should not exist before
            assert not os.path.exists(temp_cache_file)
            
            # Precompute embeddings
            evaluation_service.precompute_embeddings()
            
            # Cache should exist after
            assert os.path.exists(temp_cache_file)
            
            # Verify the embedding method was called
            assert mock_get_embedding.call_count == 2  # One for each key point

    def test_embedding_cache_loading(self, question_service, temp_cache_file, mock_openai_client):
        """Test loading embeddings from cache"""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)
            
        # Create evaluation service with custom embedding storage
        evaluation_service = EvaluationService(question_service, mock_openai_client)
        evaluation_service._embedding_storage = EmbeddingStorage(temp_cache_file)
        
        # First, create a cache
        with patch.object(evaluation_service._embedding_service, 'get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            evaluation_service.precompute_embeddings()
        
        # Now test loading from cache
        original_precompute_setting = settings.evaluation.precompute_embeddings
        try:
            settings.evaluation.precompute_embeddings = False
            
            evaluation_service_2 = EvaluationService(question_service, mock_openai_client)
            evaluation_service_2._embedding_storage = EmbeddingStorage(temp_cache_file)
            cache_info = evaluation_service_2._embedding_storage.get_cache_info()
            
            assert cache_info is not None
            assert cache_info.get('total_questions', 0) > 0
            
        finally:
            settings.evaluation.precompute_embeddings = original_precompute_setting

    def test_cache_info_retrieval(self, question_service, temp_cache_file, mock_openai_client):
        """Test retrieving cache information"""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)
            
        # Create evaluation service with custom embedding storage
        evaluation_service = EvaluationService(question_service, mock_openai_client)
        evaluation_service._embedding_storage = EmbeddingStorage(temp_cache_file)
        
        # Create cache with mock data
        with patch.object(evaluation_service._embedding_service, 'get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            evaluation_service.precompute_embeddings()
        
        cache_info = evaluation_service._embedding_storage.get_cache_info()
        
        assert cache_info is not None
        assert 'total_questions' in cache_info
        assert 'total_embeddings' in cache_info
        assert 'created_at' in cache_info
        assert cache_info['total_questions'] > 0
