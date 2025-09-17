"""PPO agent implementation with dynamic action spaces."""

import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import structlog

from ..interfaces import RLAgentInterface, AgentState

logger = structlog.get_logger(__name__)


class PolicyNetwork(nn.Module):
    """Policy network for PPO agent with dynamic action space support."""

    def __init__(
        self,
        input_dim: int,
        max_actions: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_actions = max_actions

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Shared feature extraction layers
        layers = []
        current_dim = input_dim

        for i in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                self.activation,
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Policy head (outputs logits for actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            self.activation,
            nn.Linear(hidden_size // 2, max_actions)
        )

        # Value head (outputs state value)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            self.activation,
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        # Shared feature extraction
        features = self.shared_layers(x)

        # Policy logits
        logits = self.policy_head(features)

        # Apply action mask if provided
        if action_mask is not None:
            # Set logits of unavailable actions to very negative values
            masked_logits = logits + (action_mask - 1) * 1e8
        else:
            masked_logits = logits

        # Value
        value = self.value_head(features).squeeze(-1)

        return masked_logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action distribution and value."""
        logits, value = self.forward(x, action_mask)

        # Create action distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if action is None:
            # Sample action
            action = torch.multinomial(probs, 1).squeeze(-1)

        # Get log probability of taken action
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Calculate entropy for exploration bonus
        entropy = -(probs * log_probs).sum(-1)

        return action, action_log_prob, entropy, value


class ExperienceBuffer:
    """Buffer for storing and managing experience data."""

    def __init__(self, buffer_size: int, state_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim

        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.action_masks = np.zeros((buffer_size, 1), dtype=np.float32)  # Dynamic size

        self.ptr = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None
    ):
        """Store experience in the buffer."""
        idx = self.ptr % self.buffer_size

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done

        if action_mask is not None:
            # Handle dynamic action mask size
            if self.action_masks.shape[1] < len(action_mask):
                # Expand action_masks array
                new_masks = np.zeros((self.buffer_size, len(action_mask)), dtype=np.float32)
                new_masks[:, :self.action_masks.shape[1]] = self.action_masks
                self.action_masks = new_masks
            self.action_masks[idx, :len(action_mask)] = action_mask

        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch of experiences."""
        indices = np.arange(min(self.size, self.ptr))

        return {
            "states": torch.FloatTensor(self.states[indices]),
            "actions": torch.LongTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "values": torch.FloatTensor(self.values[indices]),
            "log_probs": torch.FloatTensor(self.log_probs[indices]),
            "dones": torch.BoolTensor(self.dones[indices]),
            "action_masks": torch.FloatTensor(self.action_masks[indices])
        }

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent(RLAgentInterface):
    """PPO agent with dynamic action space support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)

        # Training parameters
        self.n_steps = config.get("n_steps", 2048)
        self.batch_size = config.get("batch_size", 64)
        self.n_epochs = config.get("n_epochs", 10)

        # Network architecture
        self.hidden_size = config.get("hidden_size", 256)
        self.n_layers = config.get("n_layers", 2)
        self.activation = config.get("activation", "relu")

        # Coefficients
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Checkpointing
        self.save_freq = config.get("save_freq", 1000)
        self.eval_freq = config.get("eval_freq", 100)

        # State dimensions (will be updated based on environment)
        self.state_dim = config.get("state_dim", 512)
        self.max_actions = config.get("max_actions", 50)  # Maximum possible actions

        # Initialize network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyNetwork(
            input_dim=self.state_dim,
            max_actions=self.max_actions,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            activation=self.activation
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Experience buffer
        self.buffer = ExperienceBuffer(self.n_steps, self.state_dim)

        # Training state
        self.update_count = 0
        self.episode_count = 0
        self.total_steps = 0

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        logger.info("Initialized PPO agent", device=str(self.device))

    def select_action(self, state: AgentState) -> int:
        """Select an action given the current state."""
        # Convert state to tensor
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        action_mask = self._get_action_mask(state)

        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                state_tensor, action_mask.unsqueeze(0) if action_mask is not None else None
            )

        action_int = action.item()

        # Store experience (will be completed when we get the reward)
        self._current_state = state_tensor.squeeze(0)
        self._current_action = action_int
        self._current_value = value.item()
        self._current_log_prob = log_prob.item()
        self._current_action_mask = action_mask

        return action_int

    def update(
        self,
        states: List[AgentState],
        actions: List[int],
        rewards: List[float],
        next_states: List[AgentState],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Update the agent with experience data."""
        # Store experiences in buffer
        for i in range(len(states)):
            state_tensor = self._state_to_tensor(states[i])
            action_mask = self._get_action_mask(states[i])

            # Get value for the state
            with torch.no_grad():
                _, _, _, value = self.network.get_action_and_value(
                    state_tensor.unsqueeze(0),
                    action_mask.unsqueeze(0) if action_mask is not None else None
                )

            self.buffer.store(
                state=state_tensor.numpy(),
                action=actions[i],
                reward=rewards[i],
                value=value.item(),
                log_prob=0.0,  # Will be recalculated
                done=dones[i],
                action_mask=action_mask.numpy() if action_mask is not None else None
            )

        self.total_steps += len(states)

        # Update if buffer is full or episode ended
        if self.buffer.size >= self.n_steps or any(dones):
            return self._update_policy()

        return {}

    def _update_policy(self) -> Dict[str, float]:
        """Update the policy using PPO."""
        if self.buffer.size == 0:
            return {}

        # Get batch
        batch = self.buffer.get_batch()
        batch_size = batch["states"].shape[0]

        if batch_size == 0:
            return {}

        # Move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)

        # Calculate advantages using GAE
        advantages, returns = self._calculate_gae(
            batch["rewards"], batch["values"], batch["dones"]
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old log probabilities
        with torch.no_grad():
            old_logits, _ = self.network(batch["states"], batch["action_masks"])
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        # Training loop
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_loss_sum = 0

        indices = torch.arange(batch_size)

        for epoch in range(self.n_epochs):
            # Shuffle indices for mini-batch training
            perm = torch.randperm(batch_size)
            indices = indices[perm]

            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                mb_indices = indices[start:end]

                # Mini-batch
                mb_states = batch["states"][mb_indices]
                mb_actions = batch["actions"][mb_indices]
                mb_masks = batch["action_masks"][mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass
                logits, values = self.network(mb_states, mb_masks)
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

                # Calculate ratio
                ratio = torch.exp(action_log_probs - mb_old_log_probs)

                # Policy loss (PPO clip)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                # Entropy loss
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(-1).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate losses
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_loss_sum += entropy.item()

        # Clear buffer
        self.buffer.clear()
        self.update_count += 1

        # Prepare metrics
        n_updates = self.n_epochs * (batch_size // self.batch_size)
        metrics = {
            "total_loss": total_loss / n_updates,
            "policy_loss": policy_loss_sum / n_updates,
            "value_loss": value_loss_sum / n_updates,
            "entropy": entropy_loss_sum / n_updates,
            "update_count": self.update_count,
            "total_steps": self.total_steps
        }

        # Save checkpoint if needed
        if self.update_count % self.save_freq == 0:
            self._save_checkpoint()

        logger.info("Policy updated", **metrics)
        return metrics

    def _calculate_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Generalized Advantage Estimation."""
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for i in reversed(range(batch_size)):
            if i == batch_size - 1:
                next_value = 0  # Terminal state
                next_non_terminal = 0
            else:
                next_value = values[i + 1]
                next_non_terminal = 1 - dones[i]

            delta = rewards[i] + self.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[i] = gae
            returns[i] = gae + values[i]

        return advantages, returns

    def _state_to_tensor(self, state: AgentState) -> torch.Tensor:
        """Convert AgentState to tensor."""
        if state.context_embedding is not None:
            embedding = state.context_embedding
        else:
            # Fallback to zero embedding
            embedding = np.zeros(self.state_dim, dtype=np.float32)

        # Ensure correct dimension
        if len(embedding) != self.state_dim:
            if len(embedding) > self.state_dim:
                embedding = embedding[:self.state_dim]
            else:
                padded = np.zeros(self.state_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded

        return torch.FloatTensor(embedding)

    def _get_action_mask(self, state: AgentState) -> Optional[torch.Tensor]:
        """Get action mask for valid actions."""
        if not state.available_actions:
            return None

        mask = torch.ones(self.max_actions, dtype=torch.float32)
        # Set mask to 0 for unavailable actions
        if len(state.available_actions) < self.max_actions:
            mask[len(state.available_actions):] = 0

        return mask

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "total_steps": self.total_steps,
            "config": self.config
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved", path=path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint["update_count"]
        self.total_steps = checkpoint["total_steps"]
        logger.info("Checkpoint loaded", path=path)

    def _save_checkpoint(self) -> None:
        """Save checkpoint automatically."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"ppo_agent_{self.update_count}.pt")
        self.save_checkpoint(path)