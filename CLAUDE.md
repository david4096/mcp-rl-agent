# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **complete reinforcement learning agent system** that uses Model Context Protocol (MCP) servers to discover and execute actions. The agent learns optimal action sequences through PPO (Proximal Policy Optimization) and interacts with operators via natural language.

**Key Innovation**: Dynamic action spaces generated from MCP server tools, making the agent adaptable to any set of available tools.

## Development Commands

### Package Management
This project uses **uv** exclusively for Python package management:
- `uv sync --all-extras` - Install all dependencies including dev extras
- `uv add <package>` - Add new dependencies
- `uv run <command>` - Run commands in the project environment

### Testing
- `uv run pytest tests/ -v` - Run all tests with verbose output
- `uv run pytest tests/unit/ -v` - Run only unit tests
- `uv run pytest tests/integration/ -v` - Run only integration tests
- `uv run pytest --cov=src/mcp_rl_agent --cov-report=html` - Test coverage report

### Code Quality
- `uv run black src/` - Format code
- `uv run isort src/` - Sort imports
- `uv run mypy src/` - Type checking
- `uv run flake8 src/` - Linting

### Running the System
- `uv run python examples/simple_demo.py` - Run basic demo
- `uv run python -m mcp_rl_agent.main --mode interactive` - Interactive mode
- `uv run python -m mcp_rl_agent.main --mode train --episodes 100` - Training mode
- `uv run python -m mcp_rl_agent.main --config configs/claude.yaml` - Use specific config

## Architecture

The system is organized into 6 main modules:

### 1. MCP Integration (`src/mcp_rl_agent/mcp/`)
- `client.py` - Real MCP client implementations (stdio, websocket)
- `mock_client.py` - Mock clients for development/testing
- `mock_server.py` - Standalone mock MCP server
- `action_manager.py` - Action discovery, validation, and execution

### 2. Reinforcement Learning (`src/mcp_rl_agent/rl/`)
- `ppo_agent.py` - Complete PPO implementation with dynamic action spaces
- Key features: GAE, experience replay, action masking, checkpointing

### 3. LLM Backend (`src/mcp_rl_agent/llm/`)
- `base.py` - Abstract interfaces and mock provider
- `claude.py` - Anthropic Claude API integration
- `huggingface.py` - HuggingFace Transformers integration
- All providers support async operation and rate limiting

### 4. Natural Language Interface (`src/mcp_rl_agent/interface/`)
- `operator.py` - Console, WebSocket, and mock operator interfaces
- `message_parser.py` - Intent recognition and parameter extraction
- Supports conversation management and context tracking

### 5. Configuration (`src/mcp_rl_agent/config/`)
- `models.py` - Pydantic models for type-safe configuration
- `manager.py` - Configuration loading with environment variable expansion
- YAML-based configuration with validation

### 6. Environment (`src/mcp_rl_agent/env/`)
- `mcp_env.py` - Gym-compatible RL environment
- Converts conversations to state representations
- Implements reward shaping and episode management

## Configuration Files

### Available Configurations
- `configs/default.yaml` - Mock providers for development
- `configs/claude.yaml` - Claude LLM with console operator
- `configs/huggingface.yaml` - Local HuggingFace models

### Key Configuration Sections
- `mcp_servers`: List of MCP server configurations
- `llm`: LLM provider settings (mock/claude/huggingface)
- `rl`: PPO hyperparameters and network architecture
- `environment`: Reward function and episode settings
- `operator`: Interface type (mock/console/websocket)

## Testing Strategy

### Test Structure
- `tests/unit/` - Component-level tests
- `tests/integration/` - End-to-end system tests
- `tests/conftest.py` - Shared test fixtures

### Key Test Files
- `test_config.py` - Configuration parsing and validation
- `test_mcp_client.py` - MCP client and server mocking
- `test_llm_providers.py` - LLM provider implementations
- `test_ppo_agent.py` - PPO algorithm and neural networks
- `test_message_parser.py` - Natural language processing
- `test_system_integration.py` - Complete system workflows

### Test Coverage
Current status: 88% coverage (44/50 tests passing)
- Comprehensive mocking of all external dependencies
- Async/await testing patterns throughout
- Integration tests validate end-to-end workflows

## Development Workflow

### Adding New MCP Servers
1. Implement `MCPClientInterface` in `mcp/client.py`
2. Add configuration model to `config/models.py`
3. Update client manager in `mcp/client.py`
4. Add tests in `tests/unit/test_mcp_client.py`

### Adding New LLM Providers
1. Implement `LLMProviderInterface` in new file under `llm/`
2. Add provider type to `config/models.py`
3. Update factory in `llm/base.py`
4. Add comprehensive tests

### Modifying RL Algorithm
1. Core PPO logic is in `rl/ppo_agent.py`
2. Network architectures in `PolicyNetwork` class
3. Experience management in `ExperienceBuffer`
4. Hyperparameters configurable via YAML

### Adding New Operator Interfaces
1. Implement `OperatorInterface` in `interface/operator.py`
2. Add interface type to `config/models.py`
3. Update factory method
4. Test async communication patterns

## Common Development Tasks

### Running Specific Components
```bash
# Test MCP mock server standalone
uv run python -m mcp_rl_agent.mcp.mock_server stdio

# Test LLM providers
uv run python -c "from mcp_rl_agent.llm.base import MockLLMProvider; print('LLM test')"

# Test configuration loading
uv run python -c "from mcp_rl_agent.config.manager import ConfigurationManager; print('Config test')"
```

### Debugging
- All components use `structlog` for structured logging
- Set `debug: true` in config for verbose logging
- Use `logging.level: DEBUG` for detailed output

### Performance Optimization
- PPO hyperparameters in config affect training speed vs. stability
- Batch size and n_steps balance memory vs. throughput
- LLM providers support caching and rate limiting

## Architecture Decisions

### Key Design Principles
1. **Modularity**: Each component is independently replaceable
2. **Async First**: All I/O operations are async/await
3. **Type Safety**: Full type hints with mypy compliance
4. **Testability**: Comprehensive mocking for all dependencies
5. **Configuration**: Runtime configuration of all components

### Technology Choices
- **uv**: Fast, reliable Python package management
- **Pydantic**: Type-safe configuration with validation
- **PyTorch**: Mature ML framework for PPO implementation
- **structlog**: Structured logging for observability
- **pytest**: Comprehensive testing with async support

### Extension Points
- New MCP transports (HTTP, custom protocols)
- Additional RL algorithms beyond PPO
- Different neural network architectures
- Advanced reward shaping mechanisms
- Multi-agent coordination protocols