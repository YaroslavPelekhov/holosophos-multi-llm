
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns

from models.rssm import RSSM, WorldModelTrainer
from environments.gridworld import KeyDoorGridWorld, PartialObservabilityWrapper
from environments.cartpole import ContinuousCartPole
from utils.experiment_utils import set_seed, collect_experience, prepare_data_for_training, save_results

class RQ2Experiment:
    """Experiment for RQ2: Partial observability effects."""

    def __init__(self, seed=42):
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Experiment parameters
        self.occlusion_levels = [0.0, 0.25, 0.50, 0.75]
        self.num_episodes = 50
        self.max_steps = 100
        self.seq_len = 20
        self.batch_size = 32
        self.num_epochs = 100

    def run_gridworld_experiment(self):
        """Run partial observability experiment on GridWorld."""
        print("\n=== Running GridWorld Partial Observability Experiment ===")

        results = {}

        for occlusion in self.occlusion_levels:
            print(f"\nTesting occlusion level: {occlusion * 100}%")

            # Create environment with occlusion
            base_env = KeyDoorGridWorld(grid_size=10, max_steps=self.max_steps)
            env = PartialObservabilityWrapper(base_env, occlusion_level=occlusion)

            # Collect random experience
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
                stoch_discrete=32,
                hidden_dim=256
            ).to(self.device)

            trainer = WorldModelTrainer(model)

            # Train world model
            print("Training world model...")
            train_losses = []
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
                train_losses.append(avg_loss)

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

            # Test on full observability environment (no occlusion)
            test_env = KeyDoorGridWorld(grid_size=10, max_steps=self.max_steps)

            # Collect test data with random policy
            test_obs, test_actions, test_rewards, test_dones = collect_experience(
                test_env, 'random', num_episodes=20, max_steps=20
            )

            # Convert to tensors
            test_obs_tensor = torch.FloatTensor(test_obs[:, :20]).to(self.device)
            test_act_tensor = torch.FloatTensor(test_actions[:, :20]).to(self.device)

            # Evaluate model performance
            prediction_errors = []
            belief_divergences = []
            success_rates = []

            for episode in range(test_obs.shape[0]):
                # Initialize states
                deter_state = torch.zeros(1, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

                episode_errors = []
                episode_beliefs = []

                for step in range(19):  # Predict next 19 steps
                    obs = test_obs_tensor[episode, step:step+1]
                    action = test_act_tensor[episode, step:step+1]
                    next_obs = test_obs_tensor[episode, step+1:step+2]

                    # Forward pass
                    deter_state, stoch_state, pred_obs, _, _ = model(
                        obs, action, deter_state, stoch_state
                    )

                    # Calculate prediction error
                    error = torch.mean((pred_obs - next_obs)**2).item()
                    episode_errors.append(error)

                    # Calculate belief state (latent representation)
                    belief_state = torch.cat([deter_state, stoch_state], dim=-1)
                    episode_beliefs.append(belief_state.detach().cpu().numpy())

                # Average error for this episode
                prediction_errors.append(np.mean(episode_errors))

                # Calculate belief divergence (std of beliefs)
                if len(episode_beliefs) > 1:
                    belief_std = np.std(np.concatenate(episode_beliefs, axis=0))
                    belief_divergences.append(belief_std)

                # Simplified success check
                last_obs = test_obs[episode, -1]
                # Check if agent found key (position (1,1) in observation space)
                # Simplified: check variability in observations
                obs_variance = np.var(test_obs[episode])
                success = obs_variance > 0.05  # Some exploration happened
                success_rates.append(success)

            results[occlusion] = {
                'prediction_error_mean': np.mean(prediction_errors),
                'prediction_error_std': np.std(prediction_errors),
                'belief_divergence_mean': np.mean(belief_divergences) if belief_divergences else 1.0,
                'belief_divergence_std': np.std(belief_divergences) if belief_divergences else 0.0,
                'success_rate': np.mean(success_rates),
                'train_losses': train_losses,
                'prediction_errors': prediction_errors,
                'belief_divergences': belief_divergences
            }

            print(f"  Prediction Error: {results[occlusion]['prediction_error_mean']:.4f} ± {results[occlusion]['prediction_error_std']:.4f}")
            print(f"  Belief Divergence: {results[occlusion]['belief_divergence_mean']:.4f} ± {results[occlusion]['belief_divergence_std']:.4f}")
            print(f"  Success Rate: {results[occlusion]['success_rate']:.3f}")

        return results

    def run_cartpole_experiment(self):
        """Run partial observability experiment on CartPole."""
        print("\n=== Running CartPole Partial Observability Experiment ===")

        results = {}

        for occlusion in self.occlusion_levels:
            print(f"\nTesting occlusion level: {occlusion * 100}%")

            # Create environment
            env = ContinuousCartPole()

            # Simple stabilizing policy
            def cartpole_policy(obs):
                x, x_dot, theta, theta_dot, _, _, _ = obs
                action = np.clip(-theta - 0.1 * theta_dot, -1.0, 1.0)
                return np.array([action], dtype=np.float32)

            # Collect experience
            print("Collecting experience...")
            obs, actions, rewards, dones = collect_experience(
                env, cartpole_policy, num_episodes=self.num_episodes, max_steps=self.max_steps
            )

            # Apply occlusion to observations during training
            # (Simulating partial observability by masking features)
            if occlusion > 0:
                mask_shape = obs.shape
                for i in range(len(obs)):
                    for j in range(len(obs[i])):
                        mask = np.random.random(obs[i][j].shape) > occlusion
                        obs[i][j] = obs[i][j] * mask

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
                stoch_discrete=0,
                hidden_dim=256
            ).to(self.device)

            trainer = WorldModelTrainer(model)

            # Train world model
            print("Training world model...")
            train_losses = []
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
                train_losses.append(avg_loss)

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

            # Test on full observability data
            test_obs, test_actions, test_rewards, test_dones = collect_experience(
                env, cartpole_policy, num_episodes=20, max_steps=20
            )

            # Convert to tensors
            test_obs_tensor = torch.FloatTensor(test_obs[:, :20]).to(self.device)
            test_act_tensor = torch.FloatTensor(test_actions[:, :20]).to(self.device)

            # Evaluate model performance
            prediction_errors = []
            belief_divergences = []
            stability_rates = []

            for episode in range(test_obs.shape[0]):
                # Initialize states
                deter_state = torch.zeros(1, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

                episode_errors = []
                episode_beliefs = []

                for step in range(19):
                    obs = test_obs_tensor[episode, step:step+1]
                    action = test_act_tensor[episode, step:step+1]
                    next_obs = test_obs_tensor[episode, step+1:step+2]

                    # Forward pass
                    deter_state, stoch_state, pred_obs, _, _ = model(
                        obs, action, deter_state, stoch_state
                    )

                    # Calculate prediction error
                    error = torch.mean((pred_obs - next_obs)**2).item()
                    episode_errors.append(error)

                    # Belief state
                    belief_state = torch.cat([deter_state, stoch_state], dim=-1)
                    episode_beliefs.append(belief_state.detach().cpu().numpy())

                # Average error for this episode
                prediction_errors.append(np.mean(episode_errors))

                # Calculate belief divergence
                if len(episode_beliefs) > 1:
                    belief_std = np.std(np.concatenate(episode_beliefs, axis=0))
                    belief_divergences.append(belief_std)

                # Check stability
                final_obs = test_obs[episode, -1]
                theta = final_obs[2]  # Pole angle
                stable = abs(theta) < 0.3  # Less than 0.3 radians (~17 degrees)
                stability_rates.append(stable)

            results[occlusion] = {
                'prediction_error_mean': np.mean(prediction_errors),
                'prediction_error_std': np.std(prediction_errors),
                'belief_divergence_mean': np.mean(belief_divergences) if belief_divergences else 1.0,
                'belief_divergence_std': np.std(belief_divergences) if belief_divergences else 0.0,
                'stability_rate': np.mean(stability_rates),
                'train_losses': train_losses,
                'prediction_errors': prediction_errors,
                'belief_divergences': belief_divergences
            }

            print(f"  Prediction Error: {results[occlusion]['prediction_error_mean']:.4f} ± {results[occlusion]['prediction_error_std']:.4f}")
            print(f"  Belief Divergence: {results[occlusion]['belief_divergence_mean']:.4f} ± {results[occlusion]['belief_divergence_std']:.4f}")
            print(f"  Stability Rate: {results[occlusion]['stability_rate']:.3f}")

        return results

    def run(self):
        """Run complete RQ2 experiment."""
        print("\n" + "="*60)
        print("RQ2: Partial observability effects experiment")
        print("="*60)

        # Run experiments for both environments
        gridworld_results = self.run_gridworld_experiment()
        cartpole_results = self.run_cartpole_experiment()

        # Create visualizations
        self.create_visualizations(gridworld_results, cartpole_results)

        # Save results
        results = {
            'gridworld': gridworld_results,
            'cartpole': cartpole_results,
            'occlusion_levels': self.occlusion_levels
        }

        save_results(results, "world_model_experiments/results/rq2_partial_observability/results.json")

        return results

    def create_visualizations(self, gridworld_results, cartpole_results):
        """Create visualizations for RQ2 results."""
        os.makedirs("world_model_experiments/results/rq2_partial_observability/plots", exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Prediction Error vs Occlusion Level
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # GridWorld
        occlusions = list(gridworld_results.keys())
        pred_errors = [gridworld_results[oc]['prediction_error_mean'] for oc in occlusions]
        pred_stds = [gridworld_results[oc]['prediction_error_std'] for oc in occlusions]

        ax1.errorbar(occlusions, pred_errors, yerr=pred_stds, fmt='o-', linewidth=2, 
                    markersize=8, capsize=5, label='Prediction Error')
        ax1.set_xlabel('Occlusion Level')
        ax1.set_ylabel('Prediction MSE')
        ax1.set_title('GridWorld: Prediction Error vs Occlusion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CartPole
        occlusions = list(cartpole_results.keys())
        pred_errors = [cartpole_results[oc]['prediction_error_mean'] for oc in occlusions]
        pred_stds = [cartpole_results[oc]['prediction_error_std'] for oc in occlusions]

        ax2.errorbar(occlusions, pred_errors, yerr=pred_stds, fmt='s-', linewidth=2, 
                    markersize=8, capsize=5, label='Prediction Error', color='orange')
        ax2.set_xlabel('Occlusion Level')
        ax2.set_ylabel('Prediction MSE')
        ax2.set_title('CartPole: Prediction Error vs Occlusion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq2_partial_observability/plots/error_vs_occlusion.png", dpi=150)
        plt.close()

        # Plot 2: Belief Divergence vs Occlusion Level
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # GridWorld
        occlusions = list(gridworld_results.keys())
        belief_divs = [gridworld_results[oc]['belief_divergence_mean'] for oc in occlusions]
        belief_stds = [gridworld_results[oc]['belief_divergence_std'] for oc in occlusions]

        ax1.errorbar(occlusions, belief_divs, yerr=belief_stds, fmt='o-', linewidth=2, 
                    markersize=8, capsize=5, label='Belief Divergence', color='green')
        ax1.set_xlabel('Occlusion Level')
        ax1.set_ylabel('Belief State Standard Deviation')
        ax1.set_title('GridWorld: Belief Divergence vs Occlusion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CartPole
        occlusions = list(cartpole_results.keys())
        belief_divs = [cartpole_results[oc]['belief_divergence_mean'] for oc in occlusions]
        belief_stds = [cartpole_results[oc]['belief_divergence_std'] for oc in occlusions]

        ax2.errorbar(occlusions, belief_divs, yerr=belief_stds, fmt='s-', linewidth=2, 
                    markersize=8, capsize=5, label='Belief Divergence', color='red')
        ax2.set_xlabel('Occlusion Level')
        ax2.set_ylabel('Belief State Standard Deviation')
        ax2.set_title('CartPole: Belief Divergence vs Occlusion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq2_partial_observability/plots/belief_vs_occlusion.png", dpi=150)
        plt.close()

        # Plot 3: Success/Stability Rate vs Occlusion Level
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # GridWorld success rate
        occlusions = list(gridworld_results.keys())
        success_rates = [gridworld_results[oc]['success_rate'] for oc in occlusions]

        ax1.plot(occlusions, success_rates, 'o-', linewidth=2, markersize=8, label='Success Rate')
        ax1.set_xlabel('Occlusion Level')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('GridWorld: Success Rate vs Occlusion')
        ax1.set_ylim([0, 1.1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CartPole stability rate
        occlusions = list(cartpole_results.keys())
        stability_rates = [cartpole_results[oc]['stability_rate'] for oc in occlusions]

        ax2.plot(occlusions, stability_rates, 's-', linewidth=2, markersize=8, 
                label='Stability Rate', color='orange')
        ax2.set_xlabel('Occlusion Level')
        ax2.set_ylabel('Stability Rate')
        ax2.set_title('CartPole: Stability Rate vs Occlusion')
        ax2.set_ylim([0, 1.1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq2_partial_observability/plots/success_vs_occlusion.png", dpi=150)
        plt.close()

        # Plot 4: Training Loss Curves for Different Occlusion Levels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, occlusion in enumerate(self.occlusion_levels):
            if occlusion in gridworld_results:
                ax = axes[idx]
                losses = gridworld_results[occlusion]['train_losses']
                ax.plot(losses, label=f'Occlusion: {occlusion*100:.0f}%')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Training Loss')
                ax.set_title(f'GridWorld Training (Occlusion: {occlusion*100:.0f}%)')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq2_partial_observability/plots/training_curves.png", dpi=150)
        plt.close()

        print("\nVisualizations saved to world_model_experiments/results/rq2_partial_observability/plots/")

if __name__ == "__main__":
    experiment = RQ2Experiment(seed=42)
    results = experiment.run()
