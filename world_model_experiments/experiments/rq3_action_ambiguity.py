
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns

from models.rssm import RSSM, WorldModelTrainer
from environments.gridworld import KeyDoorGridWorld
from environments.cartpole import ContinuousCartPole, DiscreteActionWrapper
from utils.experiment_utils import set_seed, collect_experience, prepare_data_for_training, save_results

class ActionPredictor(nn.Module):
    """Neural network to predict actions from latent states."""

    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, latent_state):
        return self.network(latent_state)

class RQ3Experiment:
    """Experiment for RQ3: Latent action ambiguity."""

    def __init__(self, seed=42):
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Experiment parameters
        self.num_episodes = 50
        self.max_steps = 100
        self.seq_len = 20
        self.batch_size = 32
        self.num_epochs = 100

    def run_action_reconstruction_experiment(self, env_name='gridworld'):
        """Run action reconstruction experiment."""
        print(f"\n=== Running {env_name} Action Reconstruction Experiment ===")

        # Create environment
        if env_name == 'gridworld':
            env = KeyDoorGridWorld(grid_size=10, max_steps=self.max_steps)
            discrete_actions = True
        elif env_name == 'cartpole_continuous':
            env = ContinuousCartPole()
            discrete_actions = False
        elif env_name == 'cartpole_discrete':
            env = DiscreteActionWrapper(ContinuousCartPole(), num_actions=5)
            discrete_actions = True
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        # Collect experience with random policy
        print("Collecting experience...")
        obs, actions, rewards, dones = collect_experience(
            env, 'random', num_episodes=self.num_episodes, max_steps=self.max_steps
        )

        # Prepare data for training
        obs_seq, act_seq, rew_seq, done_seq = prepare_data_for_training(
            obs, actions, rewards, dones, seq_len=self.seq_len
        )

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_seq).to(self.device)
        act_tensor = torch.FloatTensor(act_seq).to(self.device)
        rew_tensor = torch.FloatTensor(rew_seq).to(self.device)
        done_tensor = torch.FloatTensor(done_seq).to(self.device)

        # Create world model
        obs_dim = obs_seq.shape[-1]
        action_dim = act_seq.shape[-1]
        model = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            deter_dim=128,
            stoch_dim=16,
            stoch_discrete=32 if discrete_actions else 0,
            hidden_dim=256
        ).to(self.device)

        trainer = WorldModelTrainer(model)

        # Train world model
        print("Training world model...")
        for epoch in range(self.num_epochs):
            indices = np.random.permutation(len(obs_tensor))
            epoch_loss = 0

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]

                batch_obs = obs_tensor[batch_idx]
                batch_act = act_tensor[batch_idx]
                batch_rew = rew_tensor[batch_idx]
                batch_done = done_tensor[batch_idx]

                losses = trainer.train_step(batch_obs, batch_act, batch_rew, batch_done)
                epoch_loss += losses['total']

            avg_loss = epoch_loss / (len(indices) / self.batch_size)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        # Train action predictor
        print("\nTraining action predictor...")

        # Extract latent states from world model
        with torch.no_grad():
            latent_states = []
            true_actions = []

            for i in range(0, len(obs_tensor), self.batch_size):
                batch_obs = obs_tensor[i:i + self.batch_size]
                batch_act = act_tensor[i:i + self.batch_size]

                batch_size, seq_len, _ = batch_obs.shape

                # Initialize states
                deter_state = torch.zeros(batch_size, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(batch_size, model.stoch_dim).to(self.device)

                batch_latents = []
                batch_actions = []

                for t in range(seq_len - 1):
                    obs = batch_obs[:, t]
                    action = batch_act[:, t]

                    # Get latent state
                    deter_state, stoch_state, _, _, _ = model(obs, action, deter_state, stoch_state)
                    latent_state = torch.cat([deter_state, stoch_state], dim=-1)

                    batch_latents.append(latent_state)
                    batch_actions.append(batch_act[:, t+1])  # Predict next action

                latent_states.append(torch.cat(batch_latents, dim=0))
                true_actions.append(torch.cat(batch_actions, dim=0))

        latent_states = torch.cat(latent_states, dim=0)
        true_actions = torch.cat(true_actions, dim=0)

        # Create and train action predictor
        action_predictor = ActionPredictor(
            latent_dim=model.deter_dim + model.stoch_dim,
            action_dim=action_dim,
            hidden_dim=128
        ).to(self.device)

        optimizer = torch.optim.Adam(action_predictor.parameters(), lr=1e-3)
        criterion = nn.MSELoss() if not discrete_actions else nn.CrossEntropyLoss()

        action_pred_losses = []
        num_batches = len(latent_states) // self.batch_size

        for epoch in range(50):  # Train action predictor for fewer epochs
            indices = np.random.permutation(len(latent_states))
            epoch_loss = 0

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]

                batch_latent = latent_states[batch_idx]
                batch_action = true_actions[batch_idx]

                optimizer.zero_grad()
                pred_action = action_predictor(batch_latent)

                if discrete_actions:
                    # For discrete actions, use cross-entropy
                    loss = criterion(pred_action, batch_action.argmax(dim=-1))
                else:
                    # For continuous actions, use MSE
                    loss = criterion(pred_action, batch_action)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            action_pred_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Action Predictor Epoch {epoch + 1}/50, Loss: {avg_loss:.4f}")

        # Test action prediction accuracy
        print("\nTesting action prediction...")

        # Collect test data
        test_obs, test_actions, test_rewards, test_dones = collect_experience(
            env, 'random', num_episodes=20, max_steps=20
        )

        test_obs_tensor = torch.FloatTensor(test_obs[:, :20]).to(self.device)
        test_act_tensor = torch.FloatTensor(test_actions[:, :20]).to(self.device)

        action_pred_errors = []
        action_accuracies = []
        policy_divergences = []

        for episode in range(test_obs.shape[0]):
            # Initialize states
            deter_state = torch.zeros(1, model.deter_dim).to(self.device)
            stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

            episode_pred_errors = []
            episode_correct = 0
            episode_steps = 0

            for step in range(19):
                obs = test_obs_tensor[episode, step:step+1]
                action = test_act_tensor[episode, step:step+1]
                true_next_action = test_act_tensor[episode, step+1:step+2]

                # Get latent state
                deter_state, stoch_state, _, _, _ = model(obs, action, deter_state, stoch_state)
                latent_state = torch.cat([deter_state, stoch_state], dim=-1)

                # Predict action
                pred_action = action_predictor(latent_state)

                # Calculate error/accuracy
                if discrete_actions:
                    # For discrete actions, check if prediction matches
                    pred_class = pred_action.argmax(dim=-1)
                    true_class = true_next_action.argmax(dim=-1)
                    error = (pred_class != true_class).float().mean().item()
                    episode_pred_errors.append(error)

                    if pred_class.item() == true_class.item():
                        episode_correct += 1
                    episode_steps += 1
                else:
                    # For continuous actions, calculate MSE
                    error = torch.mean((pred_action - true_next_action)**2).item()
                    episode_pred_errors.append(error)

            # Calculate metrics for this episode
            if discrete_actions and episode_steps > 0:
                accuracy = episode_correct / episode_steps
                action_accuracies.append(accuracy)
            else:
                action_accuracies.append(0.0)

            avg_error = np.mean(episode_pred_errors) if episode_pred_errors else 1.0
            action_pred_errors.append(avg_error)

            # Calculate policy divergence (variance in predicted actions)
            if len(episode_pred_errors) > 1:
                divergence = np.var(episode_pred_errors)
                policy_divergences.append(divergence)

        results = {
            'action_pred_error_mean': np.mean(action_pred_errors),
            'action_pred_error_std': np.std(action_pred_errors),
            'action_accuracy_mean': np.mean(action_accuracies) if action_accuracies else 0.0,
            'action_accuracy_std': np.std(action_accuracies) if action_accuracies else 0.0,
            'policy_divergence_mean': np.mean(policy_divergences) if policy_divergences else 0.0,
            'policy_divergence_std': np.std(policy_divergences) if policy_divergences else 0.0,
            'world_model_loss': avg_loss,
            'action_pred_losses': action_pred_losses,
            'discrete_actions': discrete_actions
        }

        print(f"  Action Prediction Error: {results['action_pred_error_mean']:.4f} ± {results['action_pred_error_std']:.4f}")
        if discrete_actions:
            print(f"  Action Accuracy: {results['action_accuracy_mean']:.3f} ± {results['action_accuracy_std']:.3f}")
        print(f"  Policy Divergence: {results['policy_divergence_mean']:.4f} ± {results['policy_divergence_std']:.4f}")

        return results

    def run(self):
        """Run complete RQ3 experiment."""
        print("\n" + "="*60)
        print("RQ3: Latent action ambiguity experiment")
        print("="*60)

        # Run experiments for different environments/action spaces
        gridworld_results = self.run_action_reconstruction_experiment('gridworld')
        cartpole_continuous_results = self.run_action_reconstruction_experiment('cartpole_continuous')
        cartpole_discrete_results = self.run_action_reconstruction_experiment('cartpole_discrete')

        # Create visualizations
        self.create_visualizations(gridworld_results, cartpole_continuous_results, cartpole_discrete_results)

        # Save results
        results = {
            'gridworld': gridworld_results,
            'cartpole_continuous': cartpole_continuous_results,
            'cartpole_discrete': cartpole_discrete_results
        }

        save_results(results, "world_model_experiments/results/rq3_action_ambiguity/results.json")

        return results

    def create_visualizations(self, gridworld_results, cartpole_continuous_results, cartpole_discrete_results):
        """Create visualizations for RQ3 results."""
        os.makedirs("world_model_experiments/results/rq3_action_ambiguity/plots", exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Action Prediction Error Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart for action prediction error
        environments = ['GridWorld
(Discrete)', 'CartPole
(Continuous)', 'CartPole
(Discrete)']
        errors = [
            gridworld_results['action_pred_error_mean'],
            cartpole_continuous_results['action_pred_error_mean'],
            cartpole_discrete_results['action_pred_error_mean']
        ]
        error_stds = [
            gridworld_results['action_pred_error_std'],
            cartpole_continuous_results['action_pred_error_std'],
            cartpole_discrete_results['action_pred_error_std']
        ]

        x_pos = np.arange(len(environments))
        bars = ax1.bar(x_pos, errors, yerr=error_stds, capsize=5, 
                      color=['blue', 'orange', 'green'], alpha=0.7)

        ax1.set_xlabel('Environment and Action Space')
        ax1.set_ylabel('Action Prediction Error (MSE/Cross-Entropy)')
        ax1.set_title('Action Prediction Error by Environment')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(environments)

        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{error:.3f}', ha='center', va='bottom')

        # Plot 2: Action Accuracy (for discrete actions)
        ax2.bar(['GridWorld', 'CartPole Discrete'], 
                [gridworld_results['action_accuracy_mean'], cartpole_discrete_results['action_accuracy_mean']],
                yerr=[gridworld_results['action_accuracy_std'], cartpole_discrete_results['action_accuracy_std']],
                capsize=5, color=['blue', 'green'], alpha=0.7)

        ax2.set_xlabel('Environment')
        ax2.set_ylabel('Action Prediction Accuracy')
        ax2.set_title('Action Prediction Accuracy (Discrete Actions)')
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq3_action_ambiguity/plots/action_prediction_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Policy Divergence
        fig, ax = plt.subplots(figsize=(10, 6))

        environments = ['GridWorld
(Discrete)', 'CartPole
(Continuous)', 'CartPole
(Discrete)']
        divergences = [
            gridworld_results['policy_divergence_mean'],
            cartpole_continuous_results['policy_divergence_mean'],
            cartpole_discrete_results['policy_divergence_mean']
        ]
        div_stds = [
            gridworld_results['policy_divergence_std'],
            cartpole_continuous_results['policy_divergence_std'],
            cartpole_discrete_results['policy_divergence_std']
        ]

        x_pos = np.arange(len(environments))
        bars = ax.bar(x_pos, divergences, yerr=div_stds, capsize=5, 
                     color=['blue', 'orange', 'green'], alpha=0.7)

        ax.set_xlabel('Environment and Action Space')
        ax.set_ylabel('Policy Divergence (Variance)')
        ax.set_title('Policy Divergence by Environment')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(environments)

        # Add value labels on bars
        for bar, div in zip(bars, divergences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{div:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq3_action_ambiguity/plots/policy_divergence.png", dpi=150)
        plt.close()

        # Plot 4: Action Predictor Training Curves
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.plot(gridworld_results['action_pred_losses'], label='GridWorld', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('GridWorld Action Predictor Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(cartpole_continuous_results['action_pred_losses'], label='CartPole Continuous', 
                linewidth=2, color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('CartPole Continuous Action Predictor Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(cartpole_discrete_results['action_pred_losses'], label='CartPole Discrete', 
                linewidth=2, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('CartPole Discrete Action Predictor Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq3_action_ambiguity/plots/training_curves.png", dpi=150)
        plt.close()

        print("\nVisualizations saved to world_model_experiments/results/rq3_action_ambiguity/plots/")

if __name__ == "__main__":
    experiment = RQ3Experiment(seed=42)
    results = experiment.run()
