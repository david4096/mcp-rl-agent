"""MCP Reinforcement Learning Agent.

A reinforcement learning agent framework that uses Model Context Protocol (MCP)
servers to discover and execute actions. The agent learns chains of actions through
PPO (Proximal Policy Optimization) to interact with operators (humans or other agents)
via natural language.
"""

__version__ = "0.1.0"
__author__ = "MCP RL Agent Team"

from .main import MCPRLAgentSystem

__all__ = [
    "MCPRLAgentSystem",
]