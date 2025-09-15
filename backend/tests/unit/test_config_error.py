"""
Test configuration error handling
"""

import os
import tempfile
import pytest
from pathlib import Path

from core.config import load_settings_from_yaml, get_configuration_summary


class TestConfigurationErrorHandling:
    """Test configuration error handling functionality"""

    def test_missing_configuration_file(self):
        """Test loading configuration from nonexistent file"""
        settings = load_settings_from_yaml("nonexistent.yaml")
        assert settings.server.port is not None
        assert isinstance(settings.server.port, int)

    def test_empty_configuration_file(self):
        """Test loading configuration from empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name
        
        try:
            settings = load_settings_from_yaml(temp_file_path)
            assert settings.server.port is not None
            assert isinstance(settings.server.port, int)
        finally:
            os.unlink(temp_file_path)

    def test_invalid_yaml_syntax(self):
        """Test loading configuration with invalid YAML syntax"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("server:\n  port: [invalid yaml syntax")
            temp_file_path = temp_file.name
        
        try:
            settings = load_settings_from_yaml(temp_file_path)
            assert settings.server.port is not None
            assert isinstance(settings.server.port, int)
        finally:
            os.unlink(temp_file_path)

    def test_invalid_configuration_values(self):
        """Test loading configuration with invalid values"""
        config_content = """
server:
  port: 99999  # Invalid port
evaluation:
  similarity_thresholds:
    high_similarity: 1.5  # Invalid threshold
    mid_similarity: 0.9
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(config_content)
            temp_file_path = temp_file.name
        
        try:
            settings = load_settings_from_yaml(temp_file_path)
            assert settings.server.port is not None
            assert isinstance(settings.server.port, int)
            
            # Test that configuration summary can be generated
            summary = get_configuration_summary(settings)
            assert summary is not None
            assert isinstance(summary, str)
        finally:
            os.unlink(temp_file_path)
