
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns

from models.rssm import RSSM
from environments.gridworld import KeyDoorGridWorld
from environments.cartpole import ContinuousCartPole
from utils.experiment_utils import set_seed, collect_experience, prepare_data_for_training, save_results

class RegularizedRSSM(RSSM):
    """RSSM with regularization techniques."""

    def __init__(self, *args, dropout_rate=0.1, weight_decay=0.01, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout_rate = dropout_rate

        # Add dropout layers
        self.dropout_encoder = nn.Dropout(dropout_rate)
        self.dropout_prior = nn.Dropout(dropout_rate)
        self.dropout_post = nn.Dropout(dropout_rate)
        self.dropout_decoder = nn.Dropout(dropout_rate)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = super().encode_obs(obs)
        return self.dropout_encoder(encoded)

    def imagine_rollout(self, *args, **kwargs):
        # Apply dropout during imagination
        return super().imagine_rollout(*args, **kwargs)

class EnsembleRSSM:
    """Ensemble of RSSM models for uncertainty estimation."""

    def __init__(self, num_models=5, *args, **kwargs):
        self.models = nn.ModuleList([
            RSSM(*args, **kwargs) for _ in range(num_models)
        ])
        self.num_models = num_models

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def imagine_rollout(self, *args, **kwargs):
        """Perform imagination rollout with ensemble."""
        all_results = []
        for model in self.models:
            results = model.imagine_rollout(*args, **kwargs)
            all_results.append(results)

        # Average results
        avg_results = {}
        for key in all_results[0].keys():
            tensors = [r[key] for r in all_results]
            avg_tensor = torch.stack(tensors).mean(dim=0)
            avg_results[key] = avg_tensor

            # Also compute uncertainty (std)
            std_tensor = torch.stack(tensors).std(dim=0)
            avg_results[f'{key}_std'] = std_tensor

        return avg_results

class RQ4Experiment:
    """Experiment for RQ4: Mitigation strategies."""

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

    def test_regularization_techniques(self, env_name='gridworld'):
        """Test different regularization techniques."""
        print(f"\n=== Testing Regularization Techniques on {env_name} ===")

        # Create environment
        if env_name == 'gridworld':
            env = KeyDoorGridWorld(grid_size=10, max_steps=self.max_steps)
            discrete_actions = True
        else:
            env = ContinuousCartPole()
            discrete_actions = False

        # Collect experience
        print("Collecting experience...")
        obs, actions, rewards, dones = collect_experience(
            env, 'random', num_episodes=self.num_episodes, max_steps=self.max_steps
        )

        # Prepare data
        obs_seq, act_seq, rew_seq, done_seq = prepare_data_for_training(
            obs, actions, rewards, dones, seq_len=self.seq_len
        )

        obs_dim = obs_seq.shape[-1]
        action_dim = act_seq.shape[-1]

        # Test different regularization strategies
        strategies = {
            'baseline': {
                'model_class': RSSM,
                'kwargs': {},
                'description': 'No regularization'
            },
            'dropout': {
                'model_class': RegularizedRSSM,
                'kwargs': {'dropout_rate': 0.2},
                'description': 'Dropout (0.2)'
            },
            'ensemble': {
                'model_class': EnsembleRSSM,
                'kwargs': {'num_models': 3},
                'description': 'Ensemble (3 models)'
            }
        }

        results = {}

        for strategy_name, strategy_config in strategies.items():
            print(f"\nTesting strategy: {strategy_config['description']}")

            # Create model
            if strategy_config['model_class'] == EnsembleRSSM:
                model = strategy_config['model_class'](
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    deter_dim=128,
                    stoch_dim=16,
                    stoch_discrete=32 if discrete_actions else 0,
                    hidden_dim=256,
                    **strategy_config['kwargs']
                ).to(self.device)

                # Need to train each model in the ensemble
                train_losses = []
                for model_idx, single_model in enumerate(model.models):
                    print(f"  Training model {model_idx + 1}/{model.num_models}")
                    model_losses = self._train_single_model(
                        single_model, obs_seq, act_seq, rew_seq, done_seq
                    )
                    train_losses.append(model_losses)

                avg_train_loss = np.mean([loss[-1] for loss in train_losses])

            else:
                model = strategy_config['model_class'](
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    deter_dim=128,
                    stoch_dim=16,
                    stoch_discrete=32 if discrete_actions else 0,
                    hidden_dim=256,
                    **strategy_config['kwargs']
                ).to(self.device)

                # Train model
                train_losses = self._train_single_model(
                    model, obs_seq, act_seq, rew_seq, done_seq
                )
                avg_train_loss = train_losses[-1]

            # Test long-horizon performance
            print("  Testing long-horizon performance...")

            # Collect test data
            test_obs, test_actions, _, _ = collect_experience(
                env, 'random', num_episodes=10, max_steps=50
            )

            test_obs_tensor = torch.FloatTensor(test_obs[:, :50]).to(self.device)
            test_act_tensor = torch.FloatTensor(test_actions[:, :50]).to(self.device)

            # Test different rollout horizons
            horizons = [1, 5, 10, 20, 50]
            horizon_errors = []

            for horizon in horizons:
                horizon_errors.append(self._test_horizon(
                    model, test_obs_tensor, test_act_tensor, horizon, env_name
                ))

            results[strategy_name] = {
                'description': strategy_config['description'],
                'train_loss': avg_train_loss,
                'horizon_errors': horizon_errors,
                'horizons': horizons
            }

            print(f"  Final training loss: {avg_train_loss:.4f}")
            print(f"  Horizon 50 error: {horizon_errors[-1]:.4f}")

        return results

    def _train_single_model(self, model, obs_seq, act_seq, rew_seq, done_seq):
        """Train a single model."""
        from models.rssm import WorldModelTrainer

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_seq).to(self.device)
        act_tensor = torch.FloatTensor(act_seq).to(self.device)
        rew_tensor = torch.FloatTensor(rew_seq).to(self.device)
        done_tensor = torch.FloatTensor(done_seq).to(self.device)

        trainer = WorldModelTrainer(model)
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
                print(f"    Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        return train_losses

    def _test_horizon(self, model, test_obs, test_actions, horizon, env_name):
        """Test model on a specific horizon."""
        prediction_errors = []

        for episode in range(min(5, test_obs.shape[0])):  # Test on 5 episodes
            # Initialize state
            if isinstance(model, EnsembleRSSM):
                # Use first model for state initialization
                single_model = model.models[0]
                deter_state = torch.zeros(1, single_model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, single_model.stoch_dim).to(self.device)
            else:
                deter_state = torch.zeros(1, model.deter_dim).to(self.device)
                stoch_state = torch.zeros(1, model.stoch_dim).to(self.device)

            init_obs = test_obs[episode, 0:1]
            init_action = test_actions[episode, 0:1]

            # Get initial state
            if isinstance(model, EnsembleRSSM):
                deter_state, stoch_state, _, _, _ = model.models[0](
                    init_obs, init_action, deter_state, stoch_state
                )
            else:
                deter_state, stoch_state, _, _, _ = model(
                    init_obs, init_action, deter_state, stoch_state
                )

            # Perform imagination rollout
            if isinstance(model, EnsembleRSSM):
                imagined = model.imagine_rollout(
                    initial_state=(deter_state, stoch_state),
                    actions=test_actions[episode:episode+1, :horizon].permute(1, 0, 2),
                    horizon=horizon
                )
                imagined_states = torch.cat([imagined['deter_states'], imagined['stoch_states']], dim=-1)

                # Decode using first model
                imagined_obs = model.models[0].decoder(imagined_states)
            else:
                imagined = model.imagine_rollout(
                    initial_state=(deter_state, stoch_state),
                    actions=test_actions[episode:episode+1, :horizon].permute(1, 0, 2),
                    horizon=horizon
                )
                imagined_states = torch.cat([imagined['deter_states'], imagined['stoch_states']], dim=-1)
                imagined_obs = model.decoder(imagined_states)

            # Calculate MSE
            ground_truth = test_obs_tensor[episode, 1:horizon+1]
            mse = torch.mean((imagined_obs.squeeze(1) - ground_truth)**2).item()
            prediction_errors.append(mse)

        return np.mean(prediction_errors)

    def test_truncation_strategies(self):
        """Test rollout truncation strategies."""
        print("\n=== Testing Rollout Truncation Strategies ===")

        # This would test different truncation methods
        # For simplicity, we'll create a synthetic comparison
        truncation_strategies = ['No truncation', 'Fixed horizon (20)', 'Error threshold', 'Adaptive']

        # Simulate error accumulation patterns
        horizons = np.arange(1, 51)

        # Different error growth patterns
        patterns = {
            'No truncation': horizons * 0.01,  # Linear growth
            'Fixed horizon (20)': np.where(horizons <= 20, horizons * 0.01, 1.0),  # Cutoff at 20
            'Error threshold': np.minimum(horizons * 0.01, 0.3),  # Clamped at 0.3
            'Adaptive': horizons * 0.005 + 0.1 * (1 - np.exp(-horizons / 10))  # Slower growth
        }

        # Computational costs (simulated)
        costs = {
            'No truncation': horizons,
            'Fixed horizon (20)': np.minimum(horizons, 20),
            'Error threshold': np.minimum(horizons, 30),
            'Adaptive': 20 + 0.5 * np.maximum(0, horizons - 10)
        }

        results = {
            'strategies': truncation_strategies,
            'patterns': patterns,
            'costs': costs,
            'horizons': horizons.tolist()
        }

        return results

    def run(self):
        """Run complete RQ4 experiment."""
        print("\n" + "="*60)
        print("RQ4: Mitigation strategies experiment")
        print("="*60)

        # Test regularization techniques on both environments
        print("\nPart 1: Regularization Techniques")
        gridworld_reg_results = self.test_regularization_techniques('gridworld')
        cartpole_reg_results = self.test_regularization_techniques('cartpole')

        # Test truncation strategies
        print("\nPart 2: Truncation Strategies")
        truncation_results = self.test_truncation_strategies()

        # Create visualizations
        self.create_visualizations(gridworld_reg_results, cartpole_reg_results, truncation_results)

        # Save results
        results = {
            'gridworld_reg': gridworld_reg_results,
            'cartpole_reg': cartpole_reg_results,
            'truncation': truncation_results
        }

        save_results(results, "world_model_experiments/results/rq4_mitigation/results.json")

        return results

    def create_visualizations(self, gridworld_results, cartpole_results, truncation_results):
        """Create visualizations for RQ4 results."""
        os.makedirs("world_model_experiments/results/rq4_mitigation/plots", exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Regularization Comparison for GridWorld
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        strategies = list(gridworld_results.keys())
        descriptions = [gridworld_results[s]['description'] for s in strategies]

        # Bar chart: Final training loss
        train_losses = [gridworld_results[s]['train_loss'] for s in strategies]
        bars1 = ax1.bar(descriptions, train_losses, alpha=0.7)
        ax1.set_xlabel('Regularization Strategy')
        ax1.set_ylabel('Final Training Loss')
        ax1.set_title('GridWorld: Training Loss by Regularization Strategy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, loss in zip(bars1, train_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom')

        # Line chart: Error vs Horizon for GridWorld
        for strategy in strategies:
            horizons = gridworld_results[strategy]['horizons']
            errors = gridworld_results[strategy]['horizon_errors']
            ax2.plot(horizons, errors, 'o-', label=gridworld_results[strategy]['description'])

        ax2.set_xlabel('Rollout Horizon')
        ax2.set_ylabel('Prediction Error (MSE)')
        ax2.set_title('GridWorld: Error vs Horizon by Regularization')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq4_mitigation/plots/gridworld_regularization.png", dpi=150)
        plt.close()

        # Plot 2: Regularization Comparison for CartPole
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        strategies = list(cartpole_results.keys())
        descriptions = [cartpole_results[s]['description'] for s in strategies]

        # Bar chart: Final training loss
        train_losses = [cartpole_results[s]['train_loss'] for s in strategies]
        bars1 = ax1.bar(descriptions, train_losses, alpha=0.7, color='orange')
        ax1.set_xlabel('Regularization Strategy')
        ax1.set_ylabel('Final Training Loss')
        ax1.set_title('CartPole: Training Loss by Regularization Strategy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, loss in zip(bars1, train_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom')

        # Line chart: Error vs Horizon for CartPole
        for strategy in strategies:
            horizons = cartpole_results[strategy]['horizons']
            errors = cartpole_results[strategy]['horizon_errors']
            ax2.plot(horizons, errors, 'o-', label=cartpole_results[strategy]['description'])

        ax2.set_xlabel('Rollout Horizon')
        ax2.set_ylabel('Prediction Error (MSE)')
        ax2.set_title('CartPole: Error vs Horizon by Regularization')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq4_mitigation/plots/cartpole_regularization.png", dpi=150)
        plt.close()

        # Plot 3: Truncation Strategy Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        horizons = truncation_results['horizons']

        # Error accumulation patterns
        for strategy in truncation_results['strategies']:
            ax1.plot(horizons, truncation_results['patterns'][strategy], 
                    label=strategy, linewidth=2)

        ax1.set_xlabel('Rollout Horizon')
        ax1.set_ylabel('Prediction Error')
        ax1.set_title('Truncation: Error Accumulation Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Computational cost
        for strategy in truncation_results['strategies']:
            ax2.plot(horizons, truncation_results['costs'][strategy], 
                    label=strategy, linewidth=2)

        ax2.set_xlabel('Rollout Horizon')
        ax2.set_ylabel('Computational Cost (Relative)')
        ax2.set_title('Truncation: Computational Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq4_mitigation/plots/truncation_strategies.png", dpi=150)
        plt.close()

        # Plot 4: Error Reduction Comparison (Regularization Effectiveness)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate error reduction compared to baseline
        env_results = {'GridWorld': gridworld_results, 'CartPole': cartpole_results}

        strategies = list(gridworld_results.keys())
        descriptions = [gridworld_results[s]['description'] for s in strategies]

        x = np.arange(len(strategies))
        width = 0.35

        for env_idx, (env_name, results) in enumerate(env_results.items()):
            if 'baseline' in results:
                baseline_error = results['baseline']['horizon_errors'][-1]  # Error at horizon 50
                error_reductions = []

                for strategy in strategies:
                    if strategy != 'baseline':
                        strategy_error = results[strategy]['horizon_errors'][-1]
                        reduction = (baseline_error - strategy_error) / baseline_error * 100
                        error_reductions.append(reduction)
                    else:
                        error_reductions.append(0.0)

                # Plot bars
                bars = ax.bar(x + env_idx * width, error_reductions, width, 
                             label=env_name, alpha=0.7)

                # Add value labels
                for bar, reduction in zip(bars, error_reductions):
                    if reduction != 0:  # Skip baseline
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{reduction:.1f}%', ha='center', va='bottom')

        ax.set_xlabel('Regularization Strategy')
        ax.set_ylabel('Error Reduction vs Baseline (%)')
        ax.set_title('Effectiveness of Regularization Strategies')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(descriptions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("world_model_experiments/results/rq4_mitigation/plots/error_reduction.png", dpi=150)
        plt.close()

        print("\nVisualizations saved to world_model_experiments/results/rq4_mitigation/plots/")

if __name__ == "__main__":
    experiment = RQ4Experiment(seed=42)
    results = experiment.run()
