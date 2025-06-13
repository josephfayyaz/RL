"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        # if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
        #     self.sim.model.body_mass[1] -= 1.0

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] -= 1.0

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self, distance=0):
        """Sample masses according to a domain randomization distribution.
            The 'distance' parameter sets the minimum and maximum range for the uniform distribution,
            according to a proportion of the original mass. For example a distance = 0.2, means each part is sampled from a distribution having
            ranges +- 20% from the original mass.
        """

        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        #         at the start of each training episode.

        #raise NotImplementedError()
        #print("SAMPLING INSIDE THE ENVIRONMENT")

        for i in range(3):
            #original_masses doesn't take world (body_mass[0]) into consideration
            if distance < 1 :
                self.sim.model.body_mass[i+2] = np.random.uniform(self.original_masses[i+1] * (1-distance), self.original_masses[i+1] * (1+distance))
            else: #for greater or equal than 100%, clip lower bound to 0.00001 (cannot be zero)
                self.sim.model.body_mass[i+2] = np.random.uniform(0.00001, self.original_masses[i+1] * (1+distance))
        return

    def sample_parameters_autoDR(self, bounds, param_id=None, fixed_bound=None):
        """Sample all the masses according to their current distribution, except for one mass which is fixed and passed according to
        the autoDR algorithm (if x<0.5 -> lower, else higher bound)
        """
        # always sample all the masses according to current bounds
        for i in range(len(bounds)):
            self.sim.model.body_mass[i+2] = np.random.uniform(bounds[i][0], bounds[i][1])

        #additionally, if autoDR is called, fix a parameter instead of the previous value
        if param_id is not None: #this gets executed only in the case of autoDR bound fixing, otherwise it works like
                                #normal DR with the current bounds
            self.sim.model.body_mass[param_id+2] = fixed_bound #+2 because 0 is world and 1 is torso

        return


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    #ADDED
    def get_original_parameters(self) :
        """Get original masses of the hopper"""
        return self.original_masses

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
        return self._get_obs()


    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        # self.viewer.cam.distance = self.model.stat.extent * 0.75
        # self.viewer.cam.lookat[2] = 1.15
        # self.viewer.cam.elevation = -20
        torso_id = self.model.body_name2id('torso')  # ✅ get correct ID by name
        self.viewer.cam.trackbodyid = torso_id
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



"""
    Registered environments
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
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=5000,
        kwargs={"domain": "target"}
)

