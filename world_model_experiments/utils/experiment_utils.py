
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any
import pickle
import json
import os

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect_experience(env, policy, num_episodes: int = 10, max_steps: int = 100):
    """Collect experience using a policy."""
    observations = []
    actions = []
    rewards = []
    dones = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_obs = [obs]
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        for step in range(max_steps):
            # Get action from policy
            if policy == 'random':
                action = env.action_space.sample()
            else:
                # Assume policy is a callable function
                action = policy(obs)

            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            episode_obs.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            obs = next_obs

            if done:
                break

        observations.append(episode_obs[:-1])  # Exclude last observation
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        dones.append(episode_dones)

    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    # Ensure actions have correct shape
    if actions.ndim == 2:
        # Already 2D array, ensure it's the right shape for neural network
        actions = actions.reshape(actions.shape[0], actions.shape[1], 1)
    
    return observations, actions, rewards, dones

def prepare_data_for_training(observations, actions, rewards, dones, seq_len: int = 20):
    """Prepare collected data for training."""
    batch_size, num_steps = observations.shape[:2]

    # Create sequences
    num_sequences = batch_size * (num_steps - seq_len + 1)
    obs_seq = np.zeros((num_sequences, seq_len, observations.shape[-1]), dtype=np.float32)
    
    # Handle actions dimension
    if actions.ndim == 2:
        # Actions are scalars (discrete)
        actions = actions.reshape(actions.shape[0], actions.shape[1], 1)
    
    act_seq = np.zeros((num_sequences, seq_len, actions.shape[-1]), dtype=np.float32)
    rew_seq = np.zeros((num_sequences, seq_len, 1), dtype=np.float32)
    done_seq = np.zeros((num_sequences, seq_len, 1), dtype=np.float32)

    idx = 0
    for b in range(batch_size):
        for t in range(num_steps - seq_len + 1):
            obs_seq[idx] = observations[b, t:t+seq_len]
            act_seq[idx] = actions[b, t:t+seq_len]
            rew_seq[idx, :, 0] = rewards[b, t:t+seq_len]
            done_seq[idx, :, 0] = dones[b, t:t+seq_len]
            idx += 1

    return obs_seq, act_seq, rew_seq, done_seq

def compute_metrics(predictions: Dict[str, np.ndarray], 
                   ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute various metrics comparing predictions to ground truth."""
    metrics = {}

    # Mean Squared Error for state predictions
    if 'states' in predictions and 'states' in ground_truth:
        mse = np.mean((predictions['states'] - ground_truth['states'])**2)
        metrics['state_mse'] = float(mse)

    # Reward prediction error
    if 'rewards' in predictions and 'rewards' in ground_truth:
        reward_error = np.mean((predictions['rewards'] - ground_truth['rewards'])**2)
        metrics['reward_mse'] = float(reward_error)

    # Success rate (for binary success metrics)
    if 'success' in predictions and 'success' in ground_truth:
        success_rate = np.mean(predictions['success'] == ground_truth['success'])
        metrics['success_rate'] = float(success_rate)

    # Correlation coefficients
    if 'states' in predictions and 'states' in ground_truth:
        pred_flat = predictions['states'].flatten()
        gt_flat = ground_truth['states'].flatten()
        if len(pred_flat) > 1:
            correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
            metrics['state_correlation'] = float(correlation)

    return metrics

def save_results(results: Dict[str, Any], filename: str):
    """Save experiment results to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(obj.dtype) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    results_json = convert_for_json(results)

    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Also save as pickle for easier reloading
    with open(filename.replace('.json', '.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {filename}")

def load_results(filename: str) -> Dict[str, Any]:
    """Load experiment results from file."""
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def create_experiment_logger(experiment_name: str):
    """Create a logger for experiment tracking."""
    import datetime
    import logging

    log_dir = f"world_model_experiments/results/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_dir
