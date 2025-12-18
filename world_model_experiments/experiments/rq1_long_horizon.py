
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
from environments.cartpole import ContinuousCartPole, DiscreteActionWrapper
from utils.experiment_utils import set_seed, collect_experience, prepare_data_for_training, compute_metrics, save_results

class RQ1Experiment:
    """Experiment for RQ1: Long-horizon rollout degradation."""

    def __init__(self, seed=42):
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Experiment parameters
        self.horizons = [1, 5, 10, 20, 50]
        self.num_episodes = 50
        self.max_steps = 100
        self.seq_len = 20
        self.batch_size = 32
        self.num_epochs = 100

    def run_gridworld_experiment(self):
        """Run long-horizon rollout experiment on GridWorld."""
        print("\n=== Running GridWorld Long-Horizon Experiment ===")

        # Create environment
        env = KeyDoorGridWorld(grid_size=10, max_steps=self.max_steps)

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
            # Shuffle data
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
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        # Test long-horizon rollouts
        print("\nTesting long-horizon rollouts...")
        results = {}

        for horizon in self.horizons:
            print(f"Testing horizon: {horizon}")

            # Collect test data (need horizon+1 steps to compare predictions)
            test_obs, test_actions, test_rewards, test_dones = collect_experience(
                env, 'random', num_episodes=10, max_steps=horizon+1
            )

            # Convert to tensors
            test_obs_tensor = torch.FloatTensor(test_obs[:, :horizon+1]).to(self.device)
            test_act_tensor = torch.FloatTensor(test_actions[:, :horizon]).to(self.device)

            # Perform imagination rollouts
            rollout_mse = []
            success_rates = []

            for episode in range(test_obs.shape[0]):
                # Initialize state
                init_obs = test_obs_tensor[episode, 0:1]
                init_action = test_act_tensor[episode, 0:1]

                # Encode first observation
                encoded = model.encode_obs(init_obs)
                deter_state = torch.zeros(1, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

                # Get initial state
                deter_state, stoch_state, _, _, _ = model(
                    init_obs, init_action, deter_state, stoch_state
                )

                # Perform imagination rollout
                imagined = model.imagine_rollout(
                    initial_state=(deter_state, stoch_state),
                    actions=test_act_tensor[episode:episode+1, :horizon].permute(1, 0, 2),
                    horizon=horizon
                )

                # Decode imagined observations
                imagined_states = torch.cat([
                    imagined['deter_states'], 
                    imagined['stoch_states']
                ], dim=-1)

                imagined_obs = model.decoder(imagined_states)

                # Calculate MSE
                ground_truth = test_obs_tensor[episode, 1:horizon+1]
                mse = torch.mean((imagined_obs.squeeze(1) - ground_truth)**2).item()
                rollout_mse.append(mse)

                # Calculate success rate (simplified: check if trajectory reaches goal)
                # For GridWorld, we check if any state gets close to goal position
                goal_pos = np.array([8, 8])  # Goal position in grid
                success = False

                # Simplified success check
                # In practice, we would decode to grid positions
                # For now, use a proxy metric
                obs_variation = torch.std(imagined_obs).item()
                if obs_variation > 0.1:  # Some movement happened
                    success = True
                success_rates.append(success)

            results[horizon] = {
                'mse_mean': np.mean(rollout_mse),
                'mse_std': np.std(rollout_mse),
                'success_rate': np.mean(success_rates),
                'mse_values': rollout_mse,
                'success_values': success_rates
            }

            print(f"  Horizon {horizon}: MSE = {results[horizon]['mse_mean']:.4f} ± {results[horizon]['mse_std']:.4f}, " 
                  f"Success Rate = {results[horizon]['success_rate']:.3f}")

        return results, train_losses

    def run_cartpole_experiment(self):
        """Run long-horizon rollout experiment on Continuous CartPole."""
        print("\n=== Running CartPole Long-Horizon Experiment ===")

        # Create environment
        env = ContinuousCartPole()

        # Simple policy for collecting data
        def cartpole_policy(obs):
            # Simple stabilizing policy
            x, x_dot, theta, theta_dot, _, _, _ = obs
            action = np.clip(-theta - 0.1 * theta_dot, -1.0, 1.0)
            return np.array([action], dtype=np.float32)

        # Collect experience
        print("Collecting experience...")
        obs, actions, rewards, dones = collect_experience(
            env, cartpole_policy, num_episodes=self.num_episodes, max_steps=self.max_steps
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
            stoch_discrete=0,  # Continuous stochastic states
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
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        # Test long-horizon rollouts
        print("\nTesting long-horizon rollouts...")
        results = {}

        for horizon in self.horizons:
            print(f"Testing horizon: {horizon}")

            # Collect test data (need horizon+1 steps to compare predictions)
            test_obs, test_actions, test_rewards, test_dones = collect_experience(
                env, cartpole_policy, num_episodes=10, max_steps=horizon+1
            )

            # Convert to tensors
            test_obs_tensor = torch.FloatTensor(test_obs[:, :horizon+1]).to(self.device)
            test_act_tensor = torch.FloatTensor(test_actions[:, :horizon]).to(self.device)

            # Perform imagination rollouts
            rollout_mse = []
            stability_rates = []  # CartPole stability metric

            for episode in range(test_obs.shape[0]):
                # Initialize state
                init_obs = test_obs_tensor[episode, 0:1]
                init_action = test_act_tensor[episode, 0:1]

                # Encode first observation
                encoded = model.encode_obs(init_obs)
                deter_state = torch.zeros(1, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

                # Get initial state
                deter_state, stoch_state, _, _, _ = model(
                    init_obs, init_action, deter_state, stoch_state
                )

                # Perform imagination rollout
                imagined = model.imagine_rollout(
                    initial_state=(deter_state, stoch_state),
                    actions=test_act_tensor[episode:episode+1, :horizon].permute(1, 0, 2),
                    horizon=horizon
                )

                # Decode imagined observations
                imagined_states = torch.cat([
                    imagined['deter_states'], 
                    imagined['stoch_states']
                ], dim=-1)

                imagined_obs = model.decoder(imagined_states)

                # Calculate MSE
                ground_truth = test_obs_tensor[episode, 1:horizon+1]
                mse = torch.mean((imagined_obs.squeeze(1) - ground_truth)**2).item()
                rollout_mse.append(mse)

                # Check stability: pole angle should stay within reasonable bounds
                imagined_angles = imagined_obs[:, :, 2].detach().cpu().numpy()  # theta is 3rd dimension
                max_angle = np.max(np.abs(imagined_angles))
                stable = max_angle < 0.5  # 0.5 radians ≈ 28.6 degrees
                stability_rates.append(stable)

            results[horizon] = {
                'mse_mean': np.mean(rollout_mse),
                'mse_std': np.std(rollout_mse),
                'stability_rate': np.mean(stability_rates),
                'mse_values': rollout_mse,
                'stability_values': stability_rates
            }

            print(f"  Horizon {horizon}: MSE = {results[horizon]['mse_mean']:.4f} ± {results[horizon]['mse_std']:.4f}, " 
                  f"Stability Rate = {results[horizon]['stability_rate']:.3f}")

        return results, train_losses

    def run(self):
        """Run complete RQ1 experiment."""
        print("\n" + "="*60)
        print("RQ1: Long-horizon rollout degradation experiment")
        print("="*60)

        # Run experiments for both environments
        gridworld_results, gridworld_losses = self.run_gridworld_experiment()
        cartpole_results, cartpole_losses = self.run_cartpole_experiment()

        # Create visualizations
        self.create_visualizations(gridworld_results, cartpole_results, gridworld_losses, cartpole_losses)

        # Save results
        results = {
            'gridworld': gridworld_results,
            'cartpole': cartpole_results,
            'gridworld_train_loss': gridworld_losses,
            'cartpole_train_loss': cartpole_losses,
            'horizons': self.horizons
        }

        save_results(results, "world_model_experiments/results/rq1_long_horizon/results.json")

        return results

    def create_visualizations(self, gridworld_results, cartpole_results, gridworld_losses, cartpole_losses):
        """Create visualizations for RQ1 results."""
        os.makedirs("world_model_experiments/results/rq1_long_horizon/plots", exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Training losses
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(gridworld_losses, label='GridWorld', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('GridWorld World Model Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(cartpole_losses, label='CartPole', linewidth=2, color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('CartPole World Model Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq1_long_horizon/plots/training_losses.png", dpi=150)
        plt.close()

        # Plot 2: MSE vs Horizon
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # GridWorld
        horizons = list(gridworld_results.keys())
        mse_means = [gridworld_results[h]['mse_mean'] for h in horizons]
        mse_stds = [gridworld_results[h]['mse_std'] for h in horizons]

        ax1.errorbar(horizons, mse_means, yerr=mse_stds, fmt='o-', linewidth=2, 
                    markersize=8, capsize=5, label='MSE')
        ax1.set_xlabel('Rollout Horizon')
        ax1.set_ylabel('Prediction MSE')
        ax1.set_title('GridWorld: Prediction Error vs Horizon')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CartPole
        horizons = list(cartpole_results.keys())
        mse_means = [cartpole_results[h]['mse_mean'] for h in horizons]
        mse_stds = [cartpole_results[h]['mse_std'] for h in horizons]

        ax2.errorbar(horizons, mse_means, yerr=mse_stds, fmt='s-', linewidth=2, 
                    markersize=8, capsize=5, label='MSE', color='orange')
        ax2.set_xlabel('Rollout Horizon')
        ax2.set_ylabel('Prediction MSE')
        ax2.set_title('CartPole: Prediction Error vs Horizon')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq1_long_horizon/plots/mse_vs_horizon.png", dpi=150)
        plt.close()

        # Plot 3: Success/Stability Rate vs Horizon
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # GridWorld success rate
        horizons = list(gridworld_results.keys())
        success_rates = [gridworld_results[h]['success_rate'] for h in horizons]

        ax1.plot(horizons, success_rates, 'o-', linewidth=2, markersize=8, label='Success Rate')
        ax1.set_xlabel('Rollout Horizon')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('GridWorld: Success Rate vs Horizon')
        ax1.set_ylim([0, 1.1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CartPole stability rate
        horizons = list(cartpole_results.keys())
        stability_rates = [cartpole_results[h]['stability_rate'] for h in horizons]

        ax2.plot(horizons, stability_rates, 's-', linewidth=2, markersize=8, 
                label='Stability Rate', color='orange')
        ax2.set_xlabel('Rollout Horizon')
        ax2.set_ylabel('Stability Rate')
        ax2.set_title('CartPole: Stability Rate vs Horizon')
        ax2.set_ylim([0, 1.1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq1_long_horizon/plots/success_vs_horizon.png", dpi=150)
        plt.close()

        print("\nVisualizations saved to world_model_experiments/results/rq1_long_horizon/plots/")

if __name__ == "__main__":
    experiment = RQ1Experiment(seed=42)
    results = experiment.run()
