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

	"""
	random policy zero calculates average reward of random policy and sets it such that 0 is at that value.
	"""
	def __init__(self, env, high: float = 1.0, low: float = 0.0, random_policy_zero: bool = False):
		super().__init__(env)
		self.scale = (high - low)/(env.reward_space.high[0] - env.reward_space.low[0])
		self.bias = low - env.reward_space.low[0]*self.scale
		if random_policy_zero:
			print("Calculating random policy reward over 1000 episodes with 100 steps each.")
			self.bias -= get_avg_random_reward(env) * self.scale
			print("Random policy average reward calculated and set as 0.\n")

	def reward(self, rew):
		return rew *self.scale + self.bias


def get_avg_random_reward(env):
	avg_reward = 0
	for episode in range(1000):
		env.reset()
		for steps in range(100):
			_, rew, _, _ = env.step(env.action_space.sample())
			avg_reward += rew
	return avg_reward/100000
