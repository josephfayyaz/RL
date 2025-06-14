"""Implementation of the Hopper environment supporting
domain randomization optimization.
"""

from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from cma import CMAEvolutionStrategy


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        self.domain = domain
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # [torso, thigh, leg, foot]

        # Unified Domain Randomization (UDR)
        if domain == 'sudr':
            # Mass scaling ranges for each body part (torso, thigh, leg, foot)
            self.udr_mass_ranges = {
                0: (0.5, 1.5),  # torso
                1: (0.5, 1.5),  # thigh
                2: (0.5, 1.5),  # leg
                3: (0.5, 1.5),  # foot
            }

        elif domain == 'source':
            # Imprecise torso mass for source domain
            self.sim.model.body_mass[1] *= 0.7

    def set_random_parameters(self):
        """Set random masses for UDR."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample new masses for the hopper using UDR ranges."""
        if self.domain != 'sudr':
            return self.original_masses.copy()

        randomized_masses = self.original_masses.copy()
        for i in range(len(randomized_masses)):
            low, high = self.udr_mass_ranges[i]
            scale = np.random.uniform(low, high)
            randomized_masses[i] *= scale
        return randomized_masses

    def get_parameters(self):
        """Get current link masses."""
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, task):
        """Set the link masses to new values."""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Simulate one environment step."""
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        """Return current observation (qpos[1:] + qvel)."""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the model and apply domain randomization."""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.domain in ['sudr']:  # Apply UDR
            self.set_random_parameters()

        return self._get_obs()

    def viewer_setup(self):
        """Configure camera settings for rendering."""
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_mujoco_state(self, state):
        """Set MuJoCo simulator state from a vector."""
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        """Set simulator state."""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Return full MuJoCo state object."""
        return self.sim.get_state()

    # DROID methods (optional)
    def random_search_optimization(self, real_actions, real_rewards, n_trials=100):
        """Optimize dynamics using random search."""
        best_params = None
        best_cost = float('inf')
        for _ in range(n_trials):
            solution = self.sample_parameters()
            cost = self.evaluate_solution(solution, real_actions, real_rewards)
            if cost < best_cost:
                best_cost = cost
                best_params = solution
        return best_params

    def evaluate_solution(self, solution, real_actions, real_rewards):
        """Compute cost between simulated and real rewards."""
        self.set_parameters(solution)
        simulated_rewards = self.simulate_task_with_actions(real_actions)
        min_length = min(len(simulated_rewards), len(real_rewards))
        simulated_rewards = simulated_rewards[:min_length]
        real_rewards = real_rewards[:min_length]
        return np.sum((simulated_rewards - real_rewards) ** 2)

    def simulate_task_with_actions(self, actions):
        """Run the simulation with a set of actions."""
        rewards = []
        obs = self.reset()
        for a in actions:
            ep_reward = 0
            for action in a:
                obs, reward, done, _ = self.step(action)
                ep_reward += reward
                if done:
                    break
        return np.array(rewards)

    def collect_real_data(self, human, num_episodes=10):
        """Collect rollout data from a human policy."""
        actions = []
        rewards = []
        for _ in range(num_episodes):
            obs = self.reset()
            done = False
            episode_actions = []
            episode_rewards = 0
            while not done:
                action, _ = human.predict(obs)
                obs, reward, done, _ = self.step(action)
                episode_actions.append(action)
                episode_rewards += reward
            actions.append(episode_actions)
            rewards.append(episode_rewards)
        return actions, rewards


"""
    Register Gym environments for different domains
"""
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-sudr-v0",  # UDR
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "sudr"}
)


gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "target"}
)
