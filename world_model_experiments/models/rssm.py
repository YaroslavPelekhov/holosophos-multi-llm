
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) for generative world modeling.
    Simplified Dreamer-like architecture with deterministic and stochastic components.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        stoch_discrete: int = 32,
        hidden_dim: int = 400,
        activation: str = 'elu'
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete

        # Use discrete or continuous stochastic states
        self.discrete = stoch_discrete > 0
        self.num_categorical = stoch_dim  # Number of categorical variables
        self.num_categories = stoch_discrete  # Categories per variable
        if self.discrete:
            self.stoch_dim = stoch_dim * stoch_discrete  # Total latent dimension

        # Activation function
        if activation == 'elu':
            self.act = F.elu
        elif activation == 'relu':
            self.act = F.relu
        else:
            self.act = F.tanh

        # Encoder network (observation to features)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Deterministic state network (GRU-like)
        self.deter_net = nn.GRUCell(hidden_dim + action_dim, deter_dim)
        
        # Feature network for imagination (maps state to features)
        self.state_to_feature = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Prior network (predict stochastic state from deterministic state)
        if self.discrete:
            prior_output_dim = 2 * self.num_categorical * self.num_categories
        else:
            prior_output_dim = 2 * self.stoch_dim
        
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, prior_output_dim)
        )

        # Posterior network (infer stochastic state from encoder features + deterministic state)
        if self.discrete:
            post_output_dim = 2 * self.num_categorical * self.num_categories
        else:
            post_output_dim = 2 * self.stoch_dim
        
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, post_output_dim)
        )

        # Decoder network (state to observation reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Reward predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Discount predictor (for termination)
        self.discount_pred = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to feature vector."""
        return self.encoder(obs)

    def get_stoch_state(self, prior_params: torch.Tensor) -> torch.Tensor:
        """Sample stochastic state from prior/posterior parameters."""
        if self.discrete:
            # prior_params shape: [batch_size, 2 * self.num_categorical * self.num_categories]
            # Split into logits for categorical distribution
            logits = prior_params.view(-1, 2 * self.num_categorical, self.num_categories)
            logits = logits[:, :self.num_categorical, :]  # Take first half for logits
            dist = torch.distributions.OneHotCategorical(logits=logits)
            stoch_state = dist.sample()
            stoch_state = stoch_state.view(-1, self.num_categorical * self.num_categories)
        else:
            mean, std = torch.chunk(prior_params, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = torch.distributions.Normal(mean, std)
            stoch_state = dist.rsample()

        return stoch_state

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RSSM.

        Returns:
            deter_state: Deterministic state
            stoch_state: Stochastic state
            obs_recon: Reconstructed observation
            reward_pred: Predicted reward
            discount_pred: Predicted discount
        """
        # Encode observation
        encoded = self.encode_obs(obs)

        # Combine with previous stochastic state and action for deterministic update
        deter_input = torch.cat([encoded, action], dim=-1)
        deter_state = self.deter_net(deter_input, prev_deter)

        # Posterior (inference)
        post_input = torch.cat([encoded, deter_state], dim=-1)
        post_params = self.post_net(post_input)
        stoch_state = self.get_stoch_state(post_params)

        # Decode observation
        state = torch.cat([deter_state, stoch_state], dim=-1)
        obs_recon = self.decoder(state)
        reward_pred = self.reward_pred(state)
        discount_pred = torch.sigmoid(self.discount_pred(state))

        return deter_state, stoch_state, obs_recon, reward_pred, discount_pred

    def imagine_rollout(
        self,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        actions: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform imagination rollout in latent space.

        Args:
            initial_state: Tuple of (deter_state, stoch_state)
            actions: Sequence of actions of shape (horizon, batch, action_dim)
            horizon: Rollout horizon

        Returns:
            Dictionary containing imagined states and predictions
        """
        deter_state, stoch_state = initial_state
        batch_size = deter_state.shape[0]

        # Initialize storage
        all_deter = torch.zeros(horizon, batch_size, self.deter_dim)
        all_stoch = torch.zeros(horizon, batch_size, self.stoch_dim)
        all_rewards = torch.zeros(horizon, batch_size, 1)
        all_discounts = torch.zeros(horizon, batch_size, 1)

        # Rollout in latent space
        for t in range(horizon):
            # Prior network (predict next stochastic state from deterministic state)
            prior_params = self.prior_net(deter_state)
            stoch_state = self.get_stoch_state(prior_params)

            # Combine state with action for next deterministic state
            state = torch.cat([deter_state, stoch_state], dim=-1)
            features = self.state_to_feature(state)
            deter_input = torch.cat([features, actions[t]], dim=-1)
            deter_state = self.deter_net(deter_input, deter_state)

            # Predict reward and discount
            state = torch.cat([deter_state, stoch_state], dim=-1)
            reward = self.reward_pred(state)
            discount = torch.sigmoid(self.discount_pred(state))

            # Store
            all_deter[t] = deter_state
            all_stoch[t] = stoch_state
            all_rewards[t] = reward
            all_discounts[t] = discount

        return {
            'deter_states': all_deter,
            'stoch_states': all_stoch,
            'rewards': all_rewards,
            'discounts': all_discounts
        }


class WorldModelTrainer:
    """Trainer for the RSSM world model."""

    def __init__(
        self,
        model: RSSM,
        learning_rate: float = 1e-3,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
        reconstruction_scale: float = 1.0,
        reward_scale: float = 1.0
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.reconstruction_scale = reconstruction_scale
        self.reward_scale = reward_scale

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute world model loss."""
        batch_size, seq_len = obs.shape[:2]

        # Initialize states
        deter_state = torch.zeros(batch_size, self.model.deter_dim)
        stoch_state = torch.zeros(batch_size, self.model.stoch_dim)

        total_loss = 0
        losses = {}

        for t in range(seq_len - 1):
            # Forward pass
            deter_state, stoch_state, obs_recon, reward_pred, discount_pred = self.model(
                obs[:, t], actions[:, t], deter_state, stoch_state
            )

            # Reconstruction loss (observation)
            recon_loss = F.mse_loss(obs_recon, obs[:, t+1])
            total_loss += self.reconstruction_scale * recon_loss

            # Reward prediction loss
            reward_loss = F.mse_loss(reward_pred, rewards[:, t:t+1].squeeze(-1))
            total_loss += self.reward_scale * reward_loss

            # Discount prediction loss
            discount_loss = F.binary_cross_entropy(discount_pred, 1 - dones[:, t:t+1].squeeze(-1))
            total_loss += discount_loss

            # Store losses
            if 'recon' not in losses:
                losses['recon'] = recon_loss.item()
                losses['reward'] = reward_loss.item()
                losses['discount'] = discount_loss.item()
            else:
                losses['recon'] += recon_loss.item()
                losses['reward'] += reward_loss.item()
                losses['discount'] += discount_loss.item()

        # Average losses
        losses = {k: v / (seq_len - 1) for k, v in losses.items()}
        losses['total'] = total_loss.item() / (seq_len - 1)

        return losses, total_loss

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        losses, total_loss = self.compute_loss(obs, actions, rewards, dones)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()

        return losses
