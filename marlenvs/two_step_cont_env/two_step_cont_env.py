import gym
from gym import logger, spaces
import numpy as np
import matplotlib.pyplot as plt


class TwoStepContEnv(gym.Env):
	"""
	Description:
		Continous version of two step environment wit variable amount of agents.
	Source:
		Own idea based on discrete environment which appeared in the QMIX paper.
	Observation:
		Type: Box(n_agents, 1)
		Num    Observation
		gives 1 for being first agent, 0 else	

	Actions:
		Type: Box(n_agents, 1)
		Num    Action         		  Min  Max
		 i	   Action agent i		   0    1

	Reward:
		The agents get a reward only in the final state based on this formula:
			reward avg(all_final_actions)**2 * agent1_first_action + (1 - agent1_first_action) * self.intersection
				
	Episode Termination:
		 After two rounds
	"""

	metadata = {"render.modes": ["human"]}

	def __init__(self, n_agents: int  = 2, intersection: int = 0.5):
		
		self.game_state = 0
		self.intersection = intersection
		self.n_agents = n_agents


		self.action_space = spaces.Box(
			low = 0, high = 1, shape = (self.n_agents, 1)
		)
		
		self.observation_space = spaces.Box(
			low = 0, high = 1, shape = (self.n_agents, 1)
		)

		self.reward_space = spaces.Box(
			low = 0, high = 1, shape = (1,)
		)
		
		# actions
		self.weight = None
		self.actions = None
		
		# render stuff
		self.fig = None
		self.ax = None
		self.ln1 = None
		self.ln2 = None
		self.ln3 = None

	def step(self, act):
		
		if self.game_state == 0:
			self.weight = 1.0 - act[0][0]
			rew = 0.0
			done = 0

		elif self.game_state == 1:
			self.actions = np.squeeze(act)
			rew = self._reward(np.mean(self.actions))
			done = 1

		else:
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		obs = np.array([[2]] + [[2]]*(self.n_agents - 1))
		self.game_state += 1
		
		return obs, rew, done, {}

	def reset(self):
		self.game_state = 0
		self.weight = None
		self.actions = None

		obs = np.array([[1]] + [[0]]*(self.n_agents - 1))
		return obs

	def _reward(self, t):
		return t**2 * self.weight + (1 - self.weight) * self.intersection

	def render(self, mode="human"):
		
		resolution = 1000
		t = [v/resolution for v in range(resolution)]
		w_0 = [self.intersection for v in t]
		w_1 = [v**2 * 1 for v in t]
	
		if self.fig is None:
			plt.ion()
			self.fig, self.ax = plt.subplots(1)
			self.ax.fill_between(t, w_0, w_1, facecolor='blue', alpha=0.5, zorder=1)
			self.ax.set_xlabel('average action value')
			self.ax.set_ylabel('reward')

	
		if self.game_state > 0:
			
			if self.ln1 is not None:
				self.ln1.remove()
				self.ln1 = None
			if self.ln2 is not None:
				self.ln2.remove()
				self.ln2 = None
			if self.ln3 is not None:
				self.ln3.remove()
				self.ln3 = None
	
			f = [self._reward(v) for v in t]
			self.ln1, = self.ax.plot(t, f, color='red', zorder=2)
		
		if self.game_state > 1:
			r = [self._reward(v) for v in self.actions]
			avg = self._reward(np.mean(self.actions))
			self.ln2 = self.ax.scatter(self.actions, r, color='red', zorder=3)
			self.ln3 = self.ax.scatter(np.mean(self.actions), avg, marker='*', color='yellow', zorder=4)	

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


	def get_obs_dim(self):
		return 1

	def get_act_dim(self):
		return 1

if __name__ == '__main__':
	
	import time
	n_agents = 3
	env = TwoStepContEnv(n_agents = n_agents, intersection = 0.6)

	while True:	
		env.reset()
		env.render()
		done = False
		while not done:
			_, _, done, _ = env.step(np.random.uniform(size=(n_agents, 1)))
			time.sleep(0.3)
			env.render()
