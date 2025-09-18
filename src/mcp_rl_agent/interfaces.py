"""Core abstract interfaces for the MCP RL Agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class MCPTool:
    """Represents an MCP tool/action that can be executed."""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_id: str


@dataclass
class MCPAction:
    """Represents an action to be taken using an MCP tool."""
    tool: MCPTool
    arguments: Dict[str, Any]
    action_id: str


@dataclass
class MCPResult:
    """Result of executing an MCP action."""
    action_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ActionHistoryItem:
    """Single item in the action history chain."""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    step: int
    embedding: Optional[np.ndarray] = None


class MessageType(Enum):
    """Types of messages in the conversation."""
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class Message:
    """A message in a conversation."""
    type: MessageType
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """A conversation between agent and operator."""
    id: str
    messages: List[Message]
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Enhanced state representation for the RL agent with separate components."""
    conversation: Conversation
    available_actions: List[MCPTool]

    # Current user query (most recent human message)
    current_query: Optional[str] = None
    current_query_embedding: Optional[np.ndarray] = None

    # Action history chain
    action_history: List[ActionHistoryItem] = None
    action_history_embeddings: Optional[np.ndarray] = None  # Matrix of action embeddings

    # Context embedding (for backward compatibility)
    context_embedding: Optional[np.ndarray] = None

    # Step info
    step: int = 0

    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.action_history is None:
            self.action_history = []


@dataclass
class AgentObservation:
    """Observation returned by the environment."""
    state: AgentState
    reward: float
    done: bool
    info: Dict[str, Any]


class MCPClientInterface(ABC):
    """Abstract interface for MCP client connections."""

    @abstractmethod
    async def connect(self, server_config: Dict[str, Any]) -> bool:
        """Connect to an MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the MCP server."""
        pass

    @abstractmethod
    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute an action using the MCP server."""
        pass

    @property
    @abstractmethod
    def server_id(self) -> str:
        """Get the server identifier."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        pass


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response given conversation history."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass


class OperatorInterface(ABC):
    """Abstract interface for operator interactions."""

    @abstractmethod
    async def send_message(self, message: str) -> None:
        """Send a message to the operator."""
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[str]:
        """Receive a message from the operator."""
        pass

    @abstractmethod
    async def get_feedback(self, action: MCPAction, result: MCPResult) -> float:
        """Get feedback score for an action-result pair."""
        pass


class RLAgentInterface(ABC):
    """Abstract interface for the RL agent."""

    @abstractmethod
    def select_action(self, state: AgentState) -> int:
        """Select an action given the current state."""
        pass

    @abstractmethod
    def update(
        self,
        states: List[AgentState],
        actions: List[int],
        rewards: List[float],
        next_states: List[AgentState],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Update the agent with experience data."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        pass


class EnvironmentInterface(ABC):
    """Abstract interface for the RL environment."""

    @abstractmethod
    async def reset(self) -> AgentObservation:
        """Reset the environment and return initial observation."""
        pass

    @abstractmethod
    async def step(self, action_idx: int) -> AgentObservation:
        """Take a step in the environment."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Get the size of the action space."""
        pass

    @property
    @abstractmethod
    def observation_space_size(self) -> int:
        """Get the size of the observation space."""
        pass


class ConfigurationInterface(ABC):
    """Abstract interface for configuration management."""

    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        pass

    @abstractmethod
    def get_mcp_servers(self) -> List[Dict[str, Any]]:
        """Get MCP server configurations."""
        pass

    @abstractmethod
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        pass

    @abstractmethod
    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration."""
        pass

    @abstractmethod
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        pass