"""
Custom Hopper environment with domain randomization (UDR/CDR),
used for sim-to-real transfer and robustness training in MuJoCo.
"""

from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from cma import CMAEvolutionStrategy  # Used for CMA-ES optimization (optional)
from stable_baselines3.common.callbacks import BaseCallback

# -------------------- Custom Hopper Environment -------------------- #
class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, total_timesteps: int = None):
        """
        Initialize Hopper with optional domain randomization.
        """
        self.domain = domain
        self.total_timesteps = total_timesteps or 1
        self.elapsed = 0  # used for Curriculum Domain Randomization (CDR)

        MujocoEnv.__init__(self, 4)  # 4 is frame skip
        utils.EzPickle.__init__(self)

        # Save original body masses
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # torso, thigh, leg, foot

        # Unified Domain Randomization (UDR)
        if domain == 'udr':
            self.sim.model.body_mass[1] *= 0.7
            self.udr_mass_ranges = {
                1: (0.5, 1.5),  # thigh
                2: (0.5, 1.5),  # leg
                3: (0.5, 1.5),  # foot
            }

        # Source domain: fixed torso mass mismatch
        elif domain == 'source':
            self.sim.model.body_mass[1] *= 0.7

    # ------------------ Domain Randomization Logic ------------------ #
    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self, level: float = 1.0):
        if self.domain == 'udr':
            # Random scale factors for each body segment
            randomized_masses = self.original_masses.copy()
            for i in self.udr_mass_ranges:
                low, high = self.udr_mass_ranges[i]
                scale = np.random.uniform(low, high)
                randomized_masses[i - 1] *= scale
            return randomized_masses

        elif self.domain == 'cdr':
            variation = 0.3
            low = self.original_masses * (1.0 - variation * level)
            high = self.original_masses
            return np.random.uniform(low=low, high=high)

        else:
            return self.original_masses.copy()

    def get_parameters(self):
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, task):
        self.sim.model.body_mass[1:] = task

    # ------------------ Environment Step ------------------ #
    def step(self, a):
        if self.domain == "cdr":
            self.elapsed += 1

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        obs = self._get_obs()

        # Termination conditions
        done = (
            not np.isfinite(obs).all() or
            np.any(np.abs(self.sim.data.qvel) > 23.0) or
            height < 0.7 or height > 2.0 or
            abs(ang) > 1.1
        )

        # ------------------ Reward Components ------------------ #
        forward_vel = (posafter - posbefore) / self.dt
        backward_penalty = 2.0 * max(0.0, -forward_vel)
        forward_reward = max(0.0, forward_vel)
        speed_term = -0.1 * (forward_vel - 2.5) ** 2

        # Encourage vertical jumps if velocity is forward
        h = max(0.0, height - 1.1)
        jump_bonus = (4.0 if forward_vel > 0 else 2.0) * h ** 2

        alive_bonus = 2.0
        tilt_penalty = 2.0 * max(0.0, abs(ang) - 0.5) ** 2
        control_cost = 5e-4 * np.sum(a ** 2)
        vel_penalty = 1e-2 * forward_vel * abs(ang)

        # ------------------ Final Reward ------------------ #
        reward = (
            0.9 * forward_reward +
            speed_term +
            jump_bonus +
            alive_bonus -
            backward_penalty -
            tilt_penalty -
            control_cost -
            vel_penalty
        )

        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],  # ignore absolute x position
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        # Reset pose with small noise
        qpos = self.init_qpos + self.np_random.uniform(-.005, .005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-.005, .005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.domain == 'udr':
            self.set_parameters(self.sample_parameters())
        elif self.domain == 'cdr':
            level = min(1.0, self.elapsed / self.total_timesteps)
            self.set_parameters(self.sample_parameters(level))
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # ------------------ MuJoCo State Control ------------------ #
    def set_mujoco_state(self, state):
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()

    # ------------------ Random Search for Sim2Real ------------------ #
    def random_search_optimization(self, real_actions, real_rewards, n_trials=100):
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
        self.set_parameters(solution)
        simulated_rewards = self.simulate_task_with_actions(real_actions)
        min_len = min(len(simulated_rewards), len(real_rewards))
        return np.sum((simulated_rewards[:min_len] - real_rewards[:min_len]) ** 2)

    def simulate_task_with_actions(self, actions):
        rewards = []
        obs = self.reset()
        for episode in actions:
            ep_reward = 0
            for action in episode:
                obs, reward, done, _ = self.step(action)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
        return np.array(rewards)

    def collect_real_data(self, human, num_episodes=10):
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

# -------------------- Gym Registration -------------------- #
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-udr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "udr"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)

gym.envs.register(
    id="CustomHopper-cdr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "cdr"}
)

# -------------------- Entropy Scheduling Callback -------------------- #
class EntropyScheduler(BaseCallback):
    def __init__(self, start_coef, end_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start = start_coef
        self.end = end_coef
        self.total = total_timesteps

    def _on_step(self) -> bool:
        # Linearly interpolate entropy coefficient
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True
