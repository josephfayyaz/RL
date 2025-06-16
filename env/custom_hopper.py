"""Implementation of the Hopper environment supporting
domain randomization optimization.
"""

from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from cma import CMAEvolutionStrategy
from stable_baselines3.common.callbacks import BaseCallback

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None ,total_timesteps: int = None):
        self.domain = domain
        self.total_timesteps = total_timesteps or 1
        self.elapsed = 0
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        # Unified Domain Randomization (UDR)
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # [torso, thigh, leg, foot]

        if domain == 'udr':
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

    def sample_parameters(self, level: float = 1.0):
        """Sample new masses for the hopper using UDR ranges."""
        if self.domain == 'udr':

            randomized_masses = self.original_masses.copy()
            for i in range(len(randomized_masses)):
                low, high = self.udr_mass_ranges[i]
                scale = np.random.uniform(low, high)
                randomized_masses[i] *= scale
            return randomized_masses

        elif self.domain == 'cdr':
            variation = 0.3
            low  = self.original_masses * (1.0 - variation * level)
            high = self.original_masses
            return np.random.uniform(low=low, high=high)

        else:
            return self.original_masses.copy()

    def get_parameters(self):
        """Get current link masses."""
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, task):
        """Set the link masses to new values."""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Simulate one environment step."""

        if self.domain == "cdr":
            self.elapsed += 1
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        obs    = self._get_obs()
        height = self.sim.data.qpos[1]
        angle  = self.sim.data.qpos[2]

        # 2) Termination checks (milder thresholds)
        nan_fall        = not np.isfinite(obs).all()
        max_velocity    = np.any(np.abs(self.sim.data.qvel) > 23.0)
        fallen_too_low  = height < 0.7     # allow a bit more crouch
        jumped_too_high = height > 2.0     # same upper bound
        tilt_too_far    = abs(angle) > 1.1 # ~69°, more forgiving

        done = (
            nan_fall
            or max_velocity
            or fallen_too_low
            or jumped_too_high
            or tilt_too_far
        )

                # 1) Forward reward (linear)
        forward_vel    = (posafter - posbefore) / self.dt
        #forward_reward = forward_vel   #chaneged in episode 18000
        backward_penalty = 2.0 * max(0.0, -forward_vel)
        forward_reward = max(0.0, forward_vel)

        #clipped_forward  = np.clip(forward_reward, 0.0, 3.0) # 2) Clip & speed‐center around 2.5 m/s   #REMOVED IN EPISODE 18000
        speed_term = -0.1 * (forward_vel - 2.5)**2
        h = max(0.0, height - 1.1)
        if forward_vel > 0:
            jump_height_bonus = 4.0 * h**2
        else:
            jump_height_bonus = 2 * h**2



        # 3) Survival
        alive_bonus = 2.0

        # 4) Tilt penalty (quadratic beyond 0.5 rad)
        tilt_penalty = 2.0 * max(0.0, abs(angle) - 0.5)**2

        # 5) Control cost (energy)
        control_cost = 5e-4 * np.sum(a**2)

        # 6) Jump‐height bonus (quadratic above 1.1 m)
        #h = max(0.0, height - 1.1) ##CHANGED IN EPISODE 18000 MOVED TO IF ABOVE
        #jump_height_bonus = 2 * h**2

        # 7) Flight bonus (actually in the air)

        # 8) Dive penalty (forward dive when tilted)
        forward_vel_penalty = 1e-2 * forward_vel * abs(angle)

        # Combine
        reward = (
            forward_reward *0.9 # 0.5 factor to reduce reward magnitudEe
           +speed_term
            - backward_penalty
          + alive_bonus
          - tilt_penalty
          - control_cost
          + jump_height_bonus

          - forward_vel_penalty
        )
        return obs, reward, done, {}

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

        if self.domain in ['udr']:  # Apply UDR
            self.set_parameters(self.sample_parameters())
        if self.domain in ['cdr']:
            level = min(1.0, self.elapsed / self.total_timesteps)
            self.set_parameters(self.sample_parameters(level))
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
    id="CustomHopper-udr-v0",  # UDR
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "udr"}
)


gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "target"}
)
gym.envs.register(
    id="CustomHopper-cdr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=5000,
    kwargs={"domain": "cdr"}
)
class EntropyScheduler(BaseCallback):
    def __init__(self, start_coef, end_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start = start_coef
        self.end   = end_coef
        self.total = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True