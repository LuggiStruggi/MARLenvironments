import gym

class NormalizeObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, high: float = 1.0, low: float = 0.0):
        super().__init__(env)
        self.scale = (high - low)/(env.observation_space.high - env.observation_space.low)
        self.bias = low - env.observation_space.low*self.scale

    def observation(self, obs):
        return obs*self.scale + self.bias

class NormalizeActWrapper(gym.ActionWrapper):

    def __init__(self, env, high: float = 1.0, low: float = 0.0):
        super().__init__(env)
        self.scale = (env.action_space.high - env.action_space.low)/(high - low)
        self.bias =  self.action_space.low - low*self.scale

    def action(self, act):
        return act*self.scale + self.bias

class NormalizeRewWrapper(gym.RewardWrapper):

	def __init__(self, env, high: float = 1.0, low: float = 0.0):
		super().__init__(env)
		self.scale = (high - low)/(env.reward_space.high[0] - env.reward_space.low[0])
		self.bias = low - env.reward_space.low[0]*self.scale

	def reward(self, rew):
		return rew *self.scale + self.bias
