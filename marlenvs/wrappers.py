import gym

class Mapper:

	def __init__(self, low_in, high_in, low_out, high_out):
		
		self.scale = (high_out - low_out)/(high_in - low_in)
		self.low_out = low_out
		self.low_in = low_in

	def map(self, value):

		return self.low_out + self.scale * (value - self.low_in)


class NormalizeObsWrapper(gym.ObservationWrapper):

	def __init__(self, env, high: float = 1.0, low: float = 0.0):
		super().__init__(env)
		self.mapper = Mapper(low_in = env.observation_space.low, high_in = env.observation_space.high, low_out = low, high_out = high)
	
	def observation(self, obs):
		return self.mapper.map(obs)

class NormalizeActWrapper(gym.ActionWrapper):

	def __init__(self, env, high: float = 1.0, low: float = 0.0):
		super().__init__(env)
		self.mapper = Mapper(low_in = low, high_in = high, low_out = env.action_space.low, high_out = env.action_space.high)

	def action(self, act):
		return self.mapper.map(act)

class NormalizeRewWrapper(gym.RewardWrapper):

	"""
	random policy zero calculates average reward of random policy and sets it such that 0 is at that value.
	"""
	def __init__(self, env, high: float = 1.0, low: float = 0.0, random_policy_zero: bool = False):
		super().__init__(env)
		self.mapper = Mapper(low_in = env.reward_space.low[0], high_in = env.reward_space.high[0], low_out = low, high_out = high)

	def reward(self, rew):
		return self.mapper.map(rew)


def get_avg_random_reward(env):
	avg_reward = 0
	for episode in range(1000):
		env.reset()
		for steps in range(100):
			_, rew, _, _ = env.step(env.action_space.sample())
			avg_reward += rew
	return avg_reward/100000
