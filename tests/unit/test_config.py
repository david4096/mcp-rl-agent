"""Tests for configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from mcp_rl_agent.config.manager import ConfigurationManager
from mcp_rl_agent.config.models import SystemConfig, LLMProvider, OperatorType


class TestConfigurationManager:
    """Test configuration manager functionality."""

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "mcp_servers": [
                {
                    "id": "test_server",
                    "name": "Test Server",
                    "transport": "stdio",
                    "command": ["python", "-m", "test"]
                }
            ],
            "llm": {
                "provider": "mock",
                "model_name": "test_model"
            },
            "operator": {
                "type": "mock"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            manager = ConfigurationManager()
            manager.load_config(config_path)
            config = manager.get_config()

            assert isinstance(config, SystemConfig)
            assert len(config.mcp_servers) == 1
            assert config.mcp_servers[0].id == "test_server"
            assert config.llm.provider == LLMProvider.MOCK
            assert config.operator.type == OperatorType.MOCK

        finally:
            Path(config_path).unlink()

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in configuration."""
        import os
        os.environ["TEST_MODEL"] = "expanded_model"

        config_data = {
            "mcp_servers": [],
            "llm": {
                "provider": "mock",
                "model_name": "${TEST_MODEL}"
            },
            "operator": {
                "type": "mock"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            manager = ConfigurationManager()
            manager.load_config(config_path)
            config = manager.get_config()

            assert config.llm.model_name == "expanded_model"

        finally:
            Path(config_path).unlink()
            if "TEST_MODEL" in os.environ:
                del os.environ["TEST_MODEL"]

    def test_default_config_creation(self):
        """Test creation of default configuration."""
        manager = ConfigurationManager()
        config = manager.get_config()

        assert isinstance(config, SystemConfig)
        assert len(config.mcp_servers) >= 1
        assert config.llm.provider == LLMProvider.MOCK

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid transport
        with pytest.raises(ValueError):
            SystemConfig(
                mcp_servers=[{
                    "id": "test",
                    "name": "Test",
                    "transport": "invalid_transport"
                }],
                llm={"provider": "mock", "model_name": "test"},
                operator={"type": "mock"}
            )

    def test_unique_server_ids(self):
        """Test that server IDs must be unique."""
        with pytest.raises(ValueError, match="MCP server IDs must be unique"):
            SystemConfig(
                mcp_servers=[
                    {
                        "id": "duplicate",
                        "name": "Server 1",
                        "transport": "stdio",
                        "command": ["test"]
                    },
                    {
                        "id": "duplicate",
                        "name": "Server 2",
                        "transport": "stdio",
                        "command": ["test"]
                    }
                ],
                llm={"provider": "mock", "model_name": "test"},
                operator={"type": "mock"}
            )