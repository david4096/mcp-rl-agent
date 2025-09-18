"""Enhanced PPO agent that uses separated query and action history embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import structlog

from .ppo_agent import ExperienceBuffer, PPOAgent
from ..interfaces import AgentState

logger = structlog.get_logger(__name__)


class EnhancedPolicyNetwork(nn.Module):
    """Enhanced policy network that processes query and action history separately."""

    def __init__(
        self,
        query_embedding_dim: int,
        action_embedding_dim: int,
        max_action_history: int,
        max_actions: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        activation: str = "relu"
    ):
        super().__init__()
        self.query_embedding_dim = query_embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.max_action_history = max_action_history
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

        # Query processing branch
        self.query_processor = nn.Sequential(
            nn.Linear(query_embedding_dim, hidden_size // 2),
            self.activation,
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1)
        )

        # Action history processing branch (using attention)
        self.action_history_processor = nn.Sequential(
            nn.Linear(action_embedding_dim, hidden_size // 4),
            self.activation,
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.1)
        )

        # Attention mechanism for action history
        self.action_attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Combined processing
        combined_input_dim = (hidden_size // 2) + (hidden_size // 4)  # query + attention output

        # Shared layers for combined features
        shared_layers = []
        current_dim = combined_input_dim

        for i in range(n_layers):
            shared_layers.extend([
                nn.Linear(current_dim, hidden_size),
                self.activation,
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_size

        self.shared_layers = nn.Sequential(*shared_layers)

        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            self.activation,
            nn.Linear(hidden_size // 2, max_actions)
        )

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

    def forward(
        self,
        query_embeddings: torch.Tensor,
        action_history_embeddings: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with separated query and action history inputs."""
        batch_size = query_embeddings.shape[0]

        # Process query embeddings
        query_features = self.query_processor(query_embeddings)  # (batch, hidden_size//2)

        # Process action history with attention
        # action_history_embeddings: (batch, max_history, action_embedding_dim)
        action_features = self.action_history_processor(action_history_embeddings)  # (batch, max_history, hidden_size//4)

        # Apply attention to action history
        attended_actions, attention_weights = self.action_attention(
            action_features, action_features, action_features
        )  # (batch, max_history, hidden_size//4)

        # Pool the attended action features (mean pooling)
        action_summary = attended_actions.mean(dim=1)  # (batch, hidden_size//4)

        # Combine query and action features
        combined_features = torch.cat([query_features, action_summary], dim=1)  # (batch, combined_input_dim)

        # Process through shared layers
        shared_output = self.shared_layers(combined_features)

        # Generate policy logits and value
        logits = self.policy_head(shared_output)
        value = self.value_head(shared_output).squeeze(-1)

        # Apply action mask if provided
        if action_mask is not None:
            masked_logits = logits + (action_mask - 1) * 1e8
        else:
            masked_logits = logits

        return masked_logits, value

    def get_action_and_value(
        self,
        query_embeddings: torch.Tensor,
        action_history_embeddings: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action distribution and value with enhanced inputs."""
        logits, value = self.forward(query_embeddings, action_history_embeddings, action_mask)

        # Create action distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if action is None:
            # Sample action
            action = torch.multinomial(probs, 1).squeeze(-1)

        # Get log probability of taken action
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Calculate entropy
        entropy = -(probs * log_probs).sum(-1)

        return action, action_log_prob, entropy, value


class EnhancedPPOAgent(PPOAgent):
    """Enhanced PPO agent that processes query and action history separately."""

    def __init__(self, config: Dict[str, Any]):
        # Extract enhanced dimensions from config
        self.query_embedding_dim = config.get("query_embedding_dim", 256)
        self.action_embedding_dim = config.get("action_embedding_dim", 128)
        self.max_action_history = config.get("max_action_history", 10)

        # Call parent constructor
        super().__init__(config)

        # Override network with enhanced version
        self.network = EnhancedPolicyNetwork(
            query_embedding_dim=self.query_embedding_dim,
            action_embedding_dim=self.action_embedding_dim,
            max_action_history=self.max_action_history,
            max_actions=self.max_actions,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            activation=self.activation
        ).to(self.device)

        # Update optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Enhanced experience buffer
        self.buffer = EnhancedExperienceBuffer(
            buffer_size=self.n_steps,
            query_embedding_dim=self.query_embedding_dim,
            action_embedding_dim=self.action_embedding_dim,
            max_action_history=self.max_action_history
        )

        logger.info("Enhanced PPO agent initialized",
                   query_dim=self.query_embedding_dim,
                   action_dim=self.action_embedding_dim,
                   max_history=self.max_action_history)

    def select_action(self, state: AgentState) -> int:
        """Select action using enhanced state representation."""
        # Extract enhanced components
        query_tensor, action_history_tensor = self._extract_enhanced_tensors(state)
        action_mask = self._get_action_mask(state)

        query_tensor = query_tensor.unsqueeze(0)  # Add batch dimension
        action_history_tensor = action_history_tensor.unsqueeze(0)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                query_tensor, action_history_tensor, action_mask
            )

        action_int = action.item()

        # Store for experience buffer
        self._current_query = query_tensor.squeeze(0)
        self._current_action_history = action_history_tensor.squeeze(0)
        self._current_action = action_int
        self._current_value = value.item()
        self._current_log_prob = log_prob.item()
        self._current_action_mask = action_mask.squeeze(0) if action_mask is not None else None

        return action_int

    def _extract_enhanced_tensors(self, state: AgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract query and action history tensors from agent state."""
        # Query embedding
        if state.current_query_embedding is not None:
            query_tensor = torch.FloatTensor(state.current_query_embedding)
        else:
            query_tensor = torch.zeros(self.query_embedding_dim, dtype=torch.float32)

        # Ensure correct query dimension
        if query_tensor.shape[0] != self.query_embedding_dim:
            if query_tensor.shape[0] > self.query_embedding_dim:
                query_tensor = query_tensor[:self.query_embedding_dim]
            else:
                padded = torch.zeros(self.query_embedding_dim, dtype=torch.float32)
                padded[:query_tensor.shape[0]] = query_tensor
                query_tensor = padded

        # Action history embeddings
        if state.action_history_embeddings is not None:
            action_history_tensor = torch.FloatTensor(state.action_history_embeddings)
        else:
            action_history_tensor = torch.zeros(
                (self.max_action_history, self.action_embedding_dim),
                dtype=torch.float32
            )

        # Ensure correct action history dimensions
        if action_history_tensor.shape != (self.max_action_history, self.action_embedding_dim):
            # Create correctly sized tensor
            correct_tensor = torch.zeros(
                (self.max_action_history, self.action_embedding_dim),
                dtype=torch.float32
            )
            # Copy as much as possible
            min_history = min(action_history_tensor.shape[0], self.max_action_history)
            min_embed = min(action_history_tensor.shape[1], self.action_embedding_dim)
            correct_tensor[:min_history, :min_embed] = action_history_tensor[:min_history, :min_embed]
            action_history_tensor = correct_tensor

        return query_tensor, action_history_tensor

    def _update_policy(self) -> Dict[str, float]:
        """Enhanced policy update with separated embeddings."""
        if self.buffer.size == 0:
            return {}

        # Get enhanced batch
        batch = self.buffer.get_batch()
        batch_size = batch["query_embeddings"].shape[0]

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
            old_logits, _ = self.network(
                batch["query_embeddings"],
                batch["action_history_embeddings"],
                batch["action_masks"]
            )
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        # Training loop
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_loss_sum = 0

        indices = torch.arange(batch_size)

        for epoch in range(self.n_epochs):
            # Shuffle for mini-batch training
            perm = torch.randperm(batch_size)
            indices = indices[perm]

            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                mb_indices = indices[start:end]

                # Mini-batch data
                mb_query = batch["query_embeddings"][mb_indices]
                mb_action_history = batch["action_history_embeddings"][mb_indices]
                mb_actions = batch["actions"][mb_indices]
                mb_masks = batch["action_masks"][mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass
                logits, values = self.network(mb_query, mb_action_history, mb_masks)
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

                # PPO update
                ratio = torch.exp(action_log_probs - mb_old_log_probs)

                # Policy loss
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

        # Clear buffer and update counters
        self.buffer.clear()
        self.update_count += 1

        # Metrics
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

        logger.info("Enhanced policy updated", **metrics)
        return metrics


class EnhancedExperienceBuffer:
    """Enhanced experience buffer that stores query and action history separately."""

    def __init__(
        self,
        buffer_size: int,
        query_embedding_dim: int,
        action_embedding_dim: int,
        max_action_history: int
    ):
        self.buffer_size = buffer_size
        self.query_embedding_dim = query_embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.max_action_history = max_action_history

        # Initialize buffers
        self.query_embeddings = np.zeros((buffer_size, query_embedding_dim), dtype=np.float32)
        self.action_history_embeddings = np.zeros(
            (buffer_size, max_action_history, action_embedding_dim), dtype=np.float32
        )
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.action_masks = np.zeros((buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(
        self,
        query_embedding: np.ndarray,
        action_history_embedding: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None
    ):
        """Store enhanced experience in buffer."""
        idx = self.ptr % self.buffer_size

        self.query_embeddings[idx] = query_embedding
        self.action_history_embeddings[idx] = action_history_embedding
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done

        if action_mask is not None:
            # Handle dynamic action mask size
            if self.action_masks.shape[1] < len(action_mask):
                new_masks = np.zeros((self.buffer_size, len(action_mask)), dtype=np.float32)
                new_masks[:, :self.action_masks.shape[1]] = self.action_masks
                self.action_masks = new_masks
            self.action_masks[idx, :len(action_mask)] = action_mask

        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get batch of enhanced experiences."""
        indices = np.arange(min(self.size, self.ptr))

        return {
            "query_embeddings": torch.FloatTensor(self.query_embeddings[indices]),
            "action_history_embeddings": torch.FloatTensor(self.action_history_embeddings[indices]),
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