
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ContinuousCartPole(gym.Env):
    """
    Modified CartPole environment with continuous action space.

    Changes from standard CartPole:
    1. Continuous action space (force ∈ [-10, 10])
    2. Longer pole for more challenging dynamics
    3. Additional state variables for richer dynamics
    4. Modified reward function
    """

    def __init__(self, render_mode=None):
        super().__init__()

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.3  # Increased from 0.1 for more challenging dynamics
        self.total_mass = self.masscart + self.masspole
        self.length = 1.5  # Increased from 0.5 for longer horizon effects
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Thresholds
        self.x_threshold = 4.8  # Increased from 2.4
        self.theta_threshold_radians = 45 * np.pi / 180  # Increased from 12 degrees to 45 degrees

        # Action space: continuous force
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State space:
        # [x, x_dot, theta, theta_dot, theta_ddot, force_history_1, force_history_2]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # State history for richer dynamics
        self.force_history = [0.0, 0.0]

        # Step counter
        self.steps = 0
        self.max_steps = 500

        self.render_mode = render_mode

        # Initialize state
        self.reset()

    def step(self, action):
        """Take a step in the environment."""
        self.steps += 1
        action = np.clip(action, -1.0, 1.0)[0]

        # Apply force
        force = self.force_mag * action

        # Update force history
        self.force_history.append(force)
        self.force_history.pop(0)

        # Get current state
        state = self.state
        x, x_dot, theta, theta_dot = state

        # Calculate acceleration
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Calculate theta acceleration for state
        theta_ddot = thetaacc

        # Form observation with additional state variables
        obs = np.array([
            x,
            x_dot,
            theta,
            theta_dot,
            theta_ddot,
            self.force_history[0],
            self.force_history[1]
        ], dtype=np.float32)

        # Check termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )

        # Reward calculation
        reward = 1.0  # Base reward for staying alive

        # Bonus for being near center
        x_reward = 1.0 - abs(x) / self.x_threshold
        theta_reward = 1.0 - abs(theta) / self.theta_threshold_radians

        # Penalty for large actions (energy efficiency)
        action_penalty = 0.01 * abs(action)

        # Total reward
        reward = reward + 0.3 * x_reward + 0.3 * theta_reward - action_penalty

        if terminated:
            # Penalty for failure
            if self.steps < self.max_steps:
                reward = -10.0  # Early termination penalty
            else:
                reward = 100.0  # Success bonus for surviving full episode

        truncated = self.steps >= self.max_steps

        info = {
            'x': x,
            'theta': theta,
            'steps': self.steps,
            'force': force,
            'theta_acc': theta_ddot
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed)

        # Random initialization
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        self.force_history = [0.0, 0.0]

        # Initial observation
        obs = np.array([
            self.state[0],  # x
            self.state[1],  # x_dot
            self.state[2],  # theta
            self.state[3],  # theta_dot
            0.0,           # theta_ddot (initial)
            0.0,           # force_history_1
            0.0            # force_history_2
        ], dtype=np.float32)

        info = {'reset': True}

        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # Simple text rendering
            x, x_dot, theta, theta_dot = self.state
            print(f"Step: {self.steps}, x: {x:.3f}, theta: {theta:.3f}, Force History: {self.force_history}")

    def close(self):
        pass


class DiscreteActionWrapper(gym.Wrapper):
    """Wrapper to convert continuous action space to discrete for comparison experiments."""

    def __init__(self, env, num_actions=5):
        super().__init__(env)
        self.num_actions = num_actions
        self.action_space = spaces.Discrete(num_actions)

        # Create discrete action mapping
        self.action_mapping = np.linspace(-1.0, 1.0, num_actions)

    def step(self, action):
        # Convert discrete action to continuous
        continuous_action = np.array([self.action_mapping[action]], dtype=np.float32)
        return self.env.step(continuous_action)
