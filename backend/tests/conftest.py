"""
Shared pytest fixtures and configuration
"""

import pytest
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path so imports work correctly
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_questions():
    """Sample questions data for testing"""
    return {
        1: {
            'question_text': 'What is inflation and how does it affect the economy?',
            'key_points': [
                {'text': 'Inflation is a general increase in prices'},
                {'text': 'It reduces purchasing power of money'}
            ]
        },
        2: {
            'question_text': 'Explain the relationship between supply and demand',
            'key_points': [
                {'text': 'Supply and demand determine market prices'},
                {'text': 'Higher demand typically leads to higher prices'}
            ]
        }
    }

@pytest.fixture
def sample_embeddings():
    """Sample embeddings data for testing"""
    return {
        1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        2: [[0.7, 0.8, 0.9], [0.1, 0.3, 0.5]]
    }

@pytest.fixture
def sample_keywords():
    """Sample keywords data for testing"""
    return {
        1: [{"inflation", "prices", "increase"}, {"purchasing", "power", "money"}],
        2: [{"supply", "demand", "market"}, {"prices", "economics", "relationship"}]
    }

