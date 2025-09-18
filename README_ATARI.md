# Atari Games Integration Guide

This guide explains how to use the MCP RL Agent with Atari learning environments hosted via MCP servers.

## Overview

The MCP RL Agent now includes comprehensive support for Atari games through:

- **Universal Reward Functions**: Work across all Atari games with common mechanics
- **Game-Specific Rewards**: Optimized reward functions for individual games
- **Adaptive Reward Learning**: Automatically learns optimal reward weights
- **Enhanced PPO Architecture**: Separated query and action history processing
- **Flexible Configuration**: Easy setup for different game environments

## Quick Start

### 1. Basic Breakout Training

```bash
# Train on Atari Breakout
python examples/atari_training_example.py --config config/atari_breakout_config.yaml
```

### 2. Multi-Game Training

```bash
# Train on multiple Atari games
python examples/atari_training_example.py --config config/multi_game_config.yaml
```

### 3. Evaluation Only

```bash
# Evaluate a trained model
python examples/atari_training_example.py --config config/atari_pong_config.yaml --eval-only --eval-episodes 20
```

## Configuration

### Game-Specific Configurations

Pre-configured setups are available for:

- **Breakout** (`config/atari_breakout_config.yaml`)
  - Optimized for brick destruction and ball control
  - Rewards paddle hits and efficient brick clearing

- **Pong** (`config/atari_pong_config.yaml`)
  - Focused on rally length and scoring
  - Rewards ball returns and strategic positioning

- **Pac-Man** (`config/atari_pacman_config.yaml`)
  - Complex maze navigation rewards
  - Bonuses for dots, power pellets, and ghost avoidance

- **Multi-Game** (`config/multi_game_config.yaml`)
  - Trains across multiple games simultaneously
  - Balances rewards for different game mechanics

### Reward System Configuration

```yaml
atari_rewards:
  # Universal rewards work across all games
  universal:
    score_reward_scale: 0.01     # Reward per point scored
    life_penalty: -1.0           # Penalty for losing a life
    level_completion_bonus: 10.0 # Bonus for completing levels
    efficiency_bonus_scale: 0.5  # Bonus for efficient actions
    exploration_bonus: 0.1       # Bonus for exploring new areas

  # Game-specific rewards for individual optimization
  game_specific:
    breakout:
      brick_destroyed_reward: 0.1
      paddle_hit_bonus: 0.05
      wall_clear_bonus: 20.0

    pong:
      point_scored_reward: 5.0
      rally_length_bonus: 0.01
      paddle_hit_bonus: 0.1

  # Adaptive learning automatically tunes weights
  adaptive:
    enabled: true
    learning_rate: 0.001
    adaptation_window: 100
```

## MCP Server Requirements

Your MCP server should provide tools that return game state information:

```python
# Example MCP server response for Atari games
{
    "observation": [...],  # Game screen/state
    "reward": 100,         # Score change
    "done": False,         # Episode ended
    "info": {              # Additional game info
        "lives": 3,
        "level": 1,
        "score": 1250
    }
}
```

## Architecture Features

### Enhanced PPO with Separated Embeddings

The system uses an enhanced PPO architecture that separates:

- **Query Processing**: Understands user intentions and goals
- **Action History**: Tracks and learns from action sequences
- **Attention Mechanism**: Focuses on important past actions

```python
# State representation
state = AgentState(
    current_query="Play Breakout and maximize score",
    current_query_embedding=[...],      # [256] dimensions
    action_history=[...],               # List of past actions
    action_history_embeddings=[...],    # [10, 128] matrix
    available_actions=[...]             # Available game actions
)
```

### Reward Function Hierarchy

1. **Universal Rewards**: Applied to all games
2. **Game-Specific Rewards**: Applied to specific games
3. **Adaptive Weights**: Learned automatically during training
4. **Action Chain Bonuses**: Rewards for effective action sequences

## Training Tips

### For Single Games

- Use game-specific configurations for best performance
- Tune `score_reward_scale` based on game scoring mechanics
- Adjust `efficiency_bonus_scale` for games requiring quick actions

### For Multi-Game Training

- Use larger networks (`hidden_size: 512+`)
- Higher entropy coefficients for exploration
- Longer action history tracking (`max_action_history: 15+`)
- Balanced reward scales across different games

### Reward Tuning

1. **Start with Universal Rewards**: Get basic game interaction working
2. **Add Game-Specific Rewards**: Optimize for game mechanics
3. **Enable Adaptive Learning**: Let the system find optimal weights
4. **Monitor Training Logs**: Watch for reward balance issues

## Performance Monitoring

### Key Metrics to Watch

```python
# Training logs will show:
{
    "episode_reward": 1250.5,
    "game_score": 1200,           # Actual game score
    "universal_reward": 12.0,     # From universal function
    "game_specific_reward": 8.5,  # From game-specific function
    "adaptive_weights": {...},    # Current adaptive weights
    "action_efficiency": 0.85,    # Action chain effectiveness
}
```

### Evaluation Metrics

- **Score Performance**: Average game scores
- **Sample Efficiency**: Learning speed (episodes to threshold)
- **Action Diversity**: Variety of actions used
- **Chain Effectiveness**: Success of action sequences

## Troubleshooting

### Common Issues

1. **Low Scores**: Check `score_reward_scale` and game-specific rewards
2. **Repetitive Behavior**: Increase `repetition_penalty` and entropy
3. **Slow Learning**: Enable adaptive rewards and check network size
4. **Unstable Training**: Lower learning rate and increase batch size

### Debug Mode

Add to your configuration:

```yaml
logging:
  level: "DEBUG"

environment:
  debug_rewards: true    # Log detailed reward calculations
  debug_actions: true    # Log action selection process
```

## Advanced Usage

### Custom Reward Functions

Create your own reward function by extending the base class:

```python
from mcp_rl_agent.env.atari_rewards import AtariRewardFunction

class CustomGameReward(AtariRewardFunction):
    def calculate_reward(self, action, result, prev_state, current_state, action_history):
        # Your custom reward logic
        return reward_value
```

### Multi-Server Setups

Configure multiple MCP servers for different games:

```yaml
mcp_servers:
  - id: "atari_breakout"
    # ... breakout server config
  - id: "atari_pong"
    # ... pong server config
```

The environment will automatically detect game types and apply appropriate rewards.

## Integration with Enhanced PPO

The Atari reward system works seamlessly with the enhanced PPO architecture:

1. **Action History Tracking**: Rewards consider previous action chains
2. **Query Understanding**: Game goals influence reward calculations
3. **Attention Mechanism**: Focuses on successful action patterns
4. **Separated Processing**: Clean separation of game state and user intent

This creates a powerful system capable of learning complex game strategies while maintaining the flexibility to work across different Atari environments.