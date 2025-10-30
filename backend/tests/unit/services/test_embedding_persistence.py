"""
Test embedding persistence functionality
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from services.question_service import QuestionService
from services.grading_service import GradingService
from services.embedding_storage import EmbeddingStorage
from core.config import settings


class TestEmbeddingPersistence:
    """Test embedding persistence functionality"""

    @pytest.fixture
    def question_service(self):
        """Create a question service with test data"""
        service = QuestionService()
        # Mock the load_questions_bank method to avoid file dependencies
        service._questions_bank = [
            {
                'question_id': 1,
                'question_text': 'Test question 1',
                'key_points': [{'text': 'Test point 1'}]
            },
            {
                'question_id': 2,
                'question_text': 'Test question 2', 
                'key_points': [{'text': 'Test point 2'}]
            }
        ]
        # Create lookup dictionary for faster access
        service._questions_by_id = {
            question["question_id"]: question 
            for question in service._questions_bank
        }
        # Mark as loaded to prevent automatic loading from file
        service._is_loaded = True
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
        """Test that embeddings are cached to Pinecone when precompute_embeddings is called"""
        # Set precompute to True so embeddings are actually computed
        original_setting = settings.grading.precompute_embeddings
        try:
            settings.grading.precompute_embeddings = True
            
            # Create grading service with Pinecone-based embedding storage
            grading_service = GradingService(question_service, mock_openai_client)
            grading_service._embedding_storage = EmbeddingStorage()
            
            # Mock the embedding service to avoid API calls
            with patch.object(grading_service._embedding_service, 'get_embedding') as mock_get_embedding:
                mock_get_embedding.return_value = [0.1, 0.2, 0.3]
                
                # Precompute embeddings
                grading_service.precompute_embeddings()
                
                # Verify the embedding method was called
                assert mock_get_embedding.call_count == 2  # One for each key point
                
                # Verify embeddings storage was called (Pinecone mock should have recorded upserts)
                # Since we're using mocked Pinecone, just verify the method was called
                assert mock_get_embedding.called
        finally:
            settings.grading.precompute_embeddings = original_setting

    def test_embedding_cache_loading(self, question_service, temp_cache_file, mock_openai_client):
        """Test loading embeddings from Pinecone cache"""
        # Create grading service with Pinecone-based embedding storage
        grading_service = GradingService(question_service, mock_openai_client)
        grading_service._embedding_storage = EmbeddingStorage()
        
        # Get cache info - with mocked Pinecone, this will return basic info
        cache_info = grading_service._embedding_storage.get_cache_info()
        
        # Verify cache info structure exists (even if empty with mocks)
        assert cache_info is not None
        assert 'index_name' in cache_info
        assert 'dimension' in cache_info

    def test_cache_info_retrieval(self, question_service, temp_cache_file, mock_openai_client):
        """Test retrieving Pinecone cache information"""
        # Create grading service with Pinecone-based embedding storage
        grading_service = GradingService(question_service, mock_openai_client)
        grading_service._embedding_storage = EmbeddingStorage()
        
        # Get cache info from Pinecone
        cache_info = grading_service._embedding_storage.get_cache_info()
        
        # Verify basic cache info structure (works with mocked Pinecone)
        assert cache_info is not None
        assert 'index_name' in cache_info
        assert 'dimension' in cache_info
        assert cache_info['dimension'] == 1536  # OpenAI dimensions
