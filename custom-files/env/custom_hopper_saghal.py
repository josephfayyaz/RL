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
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # Default link masses

        try:
            if self.domain == 'sadr':
                self.ranges = {1: 0, 2: 0, 3: 0}
            else:
                self.ranges = {1: 0.3, 2: 0.2, 3: 0.5}

            # ADR
            self.adr_params = {
                'ranges': self.ranges,
                'performance_thresholds': {'low': 50, 'high': 200},
                'update_step': 0.01,
                'evaluation_window': 100
            }
        except:
            pass

        self.performance_buffer = []
        self.current_episode_reward = 0

        if self.domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

    # ADR
    def update_adr_ranges(self):
        if len(self.performance_buffer) >= self.adr_params['evaluation_window']:
            average_performance = np.mean(self.performance_buffer)
            print(average_performance)
            for i in range(1, 4):
                if average_performance > self.adr_params['performance_thresholds']['high']:
                    self.ranges[i] += self.adr_params['update_step']
                elif average_performance < self.adr_params['performance_thresholds']['low']:
                    if self.ranges[i] >= self.adr_params['update_step']:
                        self.ranges[i] -= self.adr_params['update_step']
            print("changing ranges")
            print(self.ranges)

            self.performance_buffer = []

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""

        random_masses = self.original_masses.copy()
        random_masses[0] -= 1.0
        random_masses[1] += np.random.uniform(-self.ranges[1], self.ranges[1])  # Randomize thigh mass
        random_masses[2] += np.random.uniform(-self.ranges[2], self.ranges[2])  # Randomize leg mass
        random_masses[3] += np.random.uniform(-self.ranges[3], self.ranges[3])  # Randomize foot mass
        # [3.53429174 3.92699082 2.71433605 5.0893801 ]

        return random_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
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

        try:
            self.current_episode_reward += reward  # Accumulate reward
            if self.domain == 'sadr':
                if done:
                    self.performance_buffer.append(self.current_episode_reward)  # Add episode reward to buffer
                    self.current_episode_reward = 0  # Reset cumulative reward for the next episode
                    self.update_adr_ranges()  # Update ADR ranges based on cumulative reward
        except:
            pass

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        if self.domain == 'sudr' or self.domain == 'sadr':  # Uniform domain randomization or AutoDr
            self.set_random_parameters()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()

    # DROID
    def random_search_optimization(self, real_actions, real_rewards, n_trials=100):
        """Optimize parameters using random search"""
        best_params = None
        best_cost = float('inf')

        for _ in range(n_trials):
            solution = self.sample_parameters()
            cost = self.evaluate_solution(solution, real_actions, real_rewards)
            if cost < best_cost:
                best_cost = cost
                best_params = solution

        return best_params

    # DROID
    def evaluate_solution(self, solution, real_actions, real_rewards):
        """Evaluate a solution by comparing simulated and real rewards"""
        self.set_parameters(solution)
        simulated_rewards = self.simulate_task_with_actions(real_actions)

        # Ensure arrays have the same length
        min_length = min(len(simulated_rewards), len(real_rewards))
        simulated_rewards = simulated_rewards[:min_length]
        real_rewards = real_rewards[:min_length]

        cost = np.sum((simulated_rewards - real_rewards) ** 2)
        return cost

    # DROID
    def simulate_task_with_actions(self, actions):
        """Simulate the task using provided actions and collect rewards"""
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

    # DROID
    def collect_real_data(self, human, num_episodes=10):
        """Collect actions and rewards from the target environment"""
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
    Registered environments
"""
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

# UDR
gym.envs.register(
    id="CustomHopper-sudr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "sudr"}
)

# ADR
gym.envs.register(
    id="CustomHopper-sadr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "sadr"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)