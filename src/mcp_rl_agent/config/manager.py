"""Configuration manager implementation."""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..interfaces import ConfigurationInterface
from .models import SystemConfig, MCPServerConfig, LLMConfig, PPOConfig, EnvironmentConfig


class ConfigurationManager(ConfigurationInterface):
    """Configuration manager using Pydantic models and YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self._config: Optional[SystemConfig] = None
        self._config_path = config_path

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            import json
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        # Expand environment variables
        raw_config = self._expand_env_vars(raw_config)

        # Validate and parse with Pydantic
        self._config = SystemConfig(**raw_config)

        return raw_config

    def get_config(self) -> SystemConfig:
        """Get the loaded configuration."""
        if self._config is None:
            if self._config_path:
                self.load_config(self._config_path)
            else:
                self._load_default_config()

        return self._config

    def get_mcp_servers(self) -> List[Dict[str, Any]]:
        """Get MCP server configurations."""
        config = self.get_config()
        return [server.dict() for server in config.mcp_servers]

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        config = self.get_config()
        return config.llm.dict()

    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration."""
        config = self.get_config()
        return config.rl.dict()

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        config = self.get_config()
        return config.environment.dict()

    def get_operator_config(self) -> Dict[str, Any]:
        """Get operator configuration."""
        config = self.get_config()
        return config.operator.dict()

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        config = self.get_config()
        return config.logging.dict()

    def _load_default_config(self) -> None:
        """Load default configuration."""
        # Look for config files in standard locations
        possible_paths = [
            "config.yaml",
            "config.yml",
            "configs/default.yaml",
            os.path.expanduser("~/.config/mcp-rl-agent/config.yaml"),
            "/etc/mcp-rl-agent/config.yaml"
        ]

        for path in possible_paths:
            if Path(path).exists():
                self.load_config(path)
                return

        # If no config file found, create minimal default
        self._create_default_config()

    def _create_default_config(self) -> None:
        """Create a minimal default configuration."""
        default_config = {
            "mcp_servers": [
                {
                    "id": "mock_server",
                    "name": "Mock MCP Server",
                    "transport": "stdio",
                    "command": ["python", "-m", "mcp_rl_agent.mcp.mock_server"]
                }
            ],
            "llm": {
                "provider": "mock",
                "model_name": "mock_model"
            },
            "operator": {
                "type": "mock"
            }
        }

        self._config = SystemConfig(**default_config)

    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            return os.path.expandvars(data)
        else:
            return data

    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded to save")

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config.dict(exclude_none=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if self._config is None:
            self.get_config()

        # Deep merge updates into current config
        config_dict = self._config.dict()
        merged = self._deep_merge(config_dict, updates)

        # Validate merged configuration
        self._config = SystemConfig(**merged)

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result