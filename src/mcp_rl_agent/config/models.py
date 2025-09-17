"""Pydantic models for configuration."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class MCPTransport(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    HTTP = "http"


class LLMProvider(str, Enum):
    """LLM provider types."""
    HUGGINGFACE = "huggingface"
    CLAUDE = "claude"
    MOCK = "mock"


class OperatorType(str, Enum):
    """Operator interface types."""
    CONSOLE = "console"
    WEBSOCKET = "websocket"
    MOCK = "mock"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""
    id: str = Field(..., description="Unique identifier for this server")
    name: str = Field(..., description="Human-readable name")
    transport: MCPTransport = Field(..., description="Transport method")

    # Transport-specific configuration
    command: Optional[List[str]] = Field(None, description="Command for stdio transport")
    args: Optional[List[str]] = Field(None, description="Arguments for stdio command")
    url: Optional[str] = Field(None, description="URL for websocket/http transport")

    # Connection settings
    timeout: float = Field(30.0, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('command')
    def validate_stdio_command(cls, v, values):
        if values.get('transport') == MCPTransport.STDIO and not v:
            raise ValueError("command is required for stdio transport")
        return v

    @validator('url')
    def validate_url_transports(cls, v, values):
        transport = values.get('transport')
        if transport in [MCPTransport.WEBSOCKET, MCPTransport.HTTP] and not v:
            raise ValueError(f"url is required for {transport} transport")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    provider: LLMProvider = Field(..., description="LLM provider type")
    model_name: str = Field(..., description="Model name/identifier")

    # API configuration
    api_key: Optional[str] = Field(None, description="API key for provider")
    api_url: Optional[str] = Field(None, description="Custom API URL")

    # Generation parameters
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")

    # Model-specific settings
    device: Optional[str] = Field(None, description="Device for local models")
    dtype: Optional[str] = Field(None, description="Data type for local models")
    cache_dir: Optional[str] = Field(None, description="Cache directory for models")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PPOConfig(BaseModel):
    """Configuration for PPO algorithm."""
    learning_rate: float = Field(3e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    clip_range: float = Field(0.2, description="PPO clip range")

    # Training parameters
    n_steps: int = Field(2048, description="Number of steps per rollout")
    batch_size: int = Field(64, description="Batch size for training")
    n_epochs: int = Field(10, description="Number of epochs per update")

    # Network architecture
    hidden_size: int = Field(256, description="Hidden layer size")
    n_layers: int = Field(2, description="Number of hidden layers")
    activation: str = Field("relu", description="Activation function")

    # Entropy regularization
    ent_coef: float = Field(0.01, description="Entropy coefficient")
    vf_coef: float = Field(0.5, description="Value function coefficient")
    max_grad_norm: float = Field(0.5, description="Maximum gradient norm")

    # Checkpointing
    save_freq: int = Field(1000, description="Save checkpoint every N updates")
    eval_freq: int = Field(100, description="Evaluate every N updates")


class EnvironmentConfig(BaseModel):
    """Configuration for the RL environment."""
    max_episode_length: int = Field(100, description="Maximum steps per episode")
    reward_scale: float = Field(1.0, description="Reward scaling factor")

    # State representation
    context_window: int = Field(10, description="Number of messages in context")
    embedding_dim: int = Field(512, description="Embedding dimension")

    # Reward function parameters
    success_reward: float = Field(1.0, description="Reward for successful actions")
    failure_penalty: float = Field(-0.1, description="Penalty for failed actions")
    step_penalty: float = Field(-0.01, description="Penalty per step")
    repetition_penalty: float = Field(-0.1, description="Penalty for repeated actions")

    # Action space configuration
    max_actions_per_turn: int = Field(5, description="Maximum actions per turn")
    action_timeout: float = Field(30.0, description="Timeout for action execution")


class OperatorConfig(BaseModel):
    """Configuration for operator interface."""
    type: OperatorType = Field(..., description="Operator interface type")

    # Console configuration
    prompt: str = Field("Human: ", description="Console prompt")

    # WebSocket configuration
    host: str = Field("localhost", description="WebSocket host")
    port: int = Field(8080, description="WebSocket port")

    # Mock configuration
    responses: List[str] = Field(default_factory=list, description="Mock responses")
    feedback_scores: List[float] = Field(default_factory=list, description="Mock feedback scores")

    # General settings
    timeout: float = Field(60.0, description="Message timeout")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field("INFO", description="Log level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")

    # File logging
    log_file: Optional[str] = Field(None, description="Log file path")
    max_file_size: str = Field("10MB", description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup log files")

    # Structured logging
    structured: bool = Field(True, description="Enable structured logging")
    json_logs: bool = Field(False, description="Output logs in JSON format")


class SystemConfig(BaseModel):
    """Root configuration model."""
    mcp_servers: List[MCPServerConfig] = Field(..., description="MCP server configurations")
    llm: LLMConfig = Field(..., description="LLM configuration")
    rl: PPOConfig = Field(default_factory=PPOConfig, description="RL configuration")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration")
    operator: OperatorConfig = Field(..., description="Operator interface configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    # Global settings
    seed: Optional[int] = Field(None, description="Random seed")
    debug: bool = Field(False, description="Enable debug mode")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('mcp_servers')
    def validate_unique_server_ids(cls, v):
        ids = [server.id for server in v]
        if len(ids) != len(set(ids)):
            raise ValueError("MCP server IDs must be unique")
        return v