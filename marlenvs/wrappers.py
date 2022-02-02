import gym

class NormalizeObsWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.scale = 1.0/(env.observation_space.high - env.observation_space.low)
        self.bias = - env.observation_space.low

    def observation(self, obs):
        obs = (obs + self.bias)*self.scale
        return obs

class NormalizeActWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.scale = (env.action_space.high - env.action_space.low)
        self.bias =  self.action_space.low

    def action(self, act):
        act = act*self.scale + self.bias
        return act

class NormalizeRewWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.scale = 1.0/(2*env.n_agents)

    def reward(self, rew):
        # modify rew
        return rew*self.scale
