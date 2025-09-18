# MCP Gym Atari Server Setup Guide

This guide explains how to connect and train the MCP RL Agent with your MCP Gym Atari Learning Environment server.

## Quick Start

Your MCP server is running on:
```
http://0.0.0.0:8000/mcp
```

### 1. Test Connection

First, verify the connection works:

```bash
# Test basic connectivity
python scripts/test_mcp_gym_connection.py

# Test specific URL if different
python scripts/test_mcp_gym_connection.py --url http://localhost:8000/mcp
```

### 2. Train the Agent

```bash
# Start training with the pre-configured setup
python examples/mcp_gym_example.py

# Or use the main interface
python -m mcp_rl_agent.main --config config/mcp_gym_atari_config.yaml --mode train --episodes 500
```

### 3. Interactive Mode

```bash
# Interactive session to test game environments
python examples/mcp_gym_example.py --test-only

# Full interactive mode
python -m mcp_rl_agent.main --config config/mcp_gym_atari_config.yaml --mode interactive
```

## Configuration Details

The configuration file `config/mcp_gym_atari_config.yaml` is set up for:

- **HTTP Transport**: Connects to your server via HTTP
- **Enhanced PPO**: Uses the advanced architecture with attention
- **Atari Rewards**: Automatically detects and applies game-specific rewards
- **Robust Connection**: Includes retry logic and health monitoring

### Key Settings

```yaml
mcp_servers:
  - id: "mcp_gym_atari"
    name: "MCP Gym - Atari Learning Environment"
    transport: "http"
    base_url: "http://0.0.0.0:8000/mcp"
    timeout: 60
    retry_attempts: 3
```

### Server Requirements

Your MCP server should support:

1. **MCP Protocol**: JSON-RPC 2.0 over HTTP
2. **Tools Discovery**: `tools/list` method
3. **Tool Execution**: `tools/call` method
4. **Atari Games**: Various Gymnasium Atari environments

Expected tool response format for games:
```json
{
  "content": [
    {
      "type": "text",
      "text": "Game result with observation, reward, done, info"
    }
  ]
}
```

## Troubleshooting

### Connection Issues

1. **Server Not Responding**:
   ```bash
   # Check if server is running
   curl http://0.0.0.0:8000/mcp

   # Test with different URL
   python scripts/test_mcp_gym_connection.py --url http://127.0.0.1:8000/mcp
   ```

2. **Timeout Errors**:
   - Increase timeout in config: `timeout: 120`
   - Check server performance
   - Verify network connectivity

3. **Tool Discovery Fails**:
   ```bash
   # Test tool discovery manually
   curl -X POST http://0.0.0.0:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
   ```

### Training Issues

1. **Low Rewards**:
   - Check game-specific reward configuration
   - Enable reward debugging: `debug_settings.log_reward_calculations: true`

2. **Connection Drops**:
   - Enable health checks: `health_check.enabled: true`
   - Increase retry attempts: `retry_attempts: 5`

3. **Slow Training**:
   - Reduce batch size: `batch_size: 64`
   - Lower network complexity: `hidden_size: 256`

### Debug Mode

Enable detailed logging:

```yaml
logging:
  level: "DEBUG"
  debug_settings:
    log_mcp_requests: true
    log_reward_calculations: true
    log_action_selection: true
```

## Expected Game Tools

Your MCP Gym server likely provides tools like:

- `create_environment` - Initialize a specific Atari game
- `reset_environment` - Reset the game state
- `step_environment` - Take an action in the game
- `get_observation` - Get current game state
- `list_games` - List available Atari games
- `render_environment` - Get visual representation

## Performance Optimization

### For Single Game Training:
```yaml
rl:
  batch_size: 128
  learning_rate: 0.0003
  hidden_size: 512

environment:
  max_episode_length: 2000
  atari_rewards:
    universal:
      score_reward_scale: 0.01
```

### For Multi-Game Training:
```yaml
rl:
  batch_size: 256
  entropy_coef: 0.03
  max_action_history: 15

environment:
  max_episode_length: 1500
```

## Integration with Enhanced Architecture

The system automatically:

1. **Detects Game Environment**: Identifies Atari games from tool names/descriptions
2. **Applies Appropriate Rewards**: Uses game-specific reward functions
3. **Tracks Action Chains**: Learns effective action sequences
4. **Separates Processing**: Query understanding vs. action history

## Monitoring Training

Watch for these metrics:

```
Episode 100 complete: Reward=1250.50, Avg=890.25, Steps=847
Game-specific reward: 12.3, Universal reward: 8.9
Action efficiency: 0.78, Attention focus: [0.1, 0.3, 0.2, 0.4]
```

## Next Steps

1. **Start Simple**: Test connection and basic training
2. **Monitor Performance**: Watch episode rewards and training logs
3. **Tune Rewards**: Adjust game-specific reward parameters
4. **Scale Up**: Increase episode count and save checkpoints
5. **Evaluate**: Test on multiple games to verify learning

For advanced configuration and troubleshooting, see:
- [README_ATARI.md](README_ATARI.md) - Detailed Atari integration guide
- [Enhanced PPO Architecture](docs/enhanced_ppo_architecture.md) - Technical details