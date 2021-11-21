import gym
from gym import logger
import torch

class TwoStepEnv(gym.Env):

	metadata = {"render.modes": ["human"]}

	def __init__(self, linear=False):
		self.game_state = 0
		self.linear = linear

	def step(self, act):
		
		# observation so each agent knows which one it is
		obs = [0, 1]

		if self.game_state == 0:
			if act[0] == 0:
				self.game_state = 1
			elif act[0] == 1:
				self.game_state = 2
			done = [0, 0]
			rew = [0]
		
		elif self.game_state == 1:
			done = [1, 1]
			rew = [7]
			self.game_state = 3 + 2*int(act[0]) + int(act[1])

		elif self.game_state == 2:
			done = [1, 1]
			if act[0] == 0 and act[1] == 0:
				rew = [0]
			elif act[0] == 1 and act[1] == 1:
				rew = [2 if self.linear else 8]
			elif act[0] == 0 and act[1] == 1:
				rew = [1]
			elif act[0] == 1 and act[1] == 0:
				rew = [1]
			
			self.game_state = 7 + 2*int(act[0]) + int(act[1])

		elif self.game_state >= 3:
			rew = [0]
			done = [1, 1]
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		return torch.Tensor(obs), torch.Tensor(rew), torch.Tensor(done), {}

	def reset(self):
		self.game_state = 0
		self.final_rew = 0

	def render(self, mode="human"):
		l1 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 3 else "█"
		l2 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 4 else "█"
		l3 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 5 else "█"
		l4 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 6 else "█"
		r1 = "0" if self.game_state == 0 or self.game_state == 2 or self.game_state == 7 else "█"
		r2 = "1" if self.game_state == 0 or self.game_state == 2 or self.game_state == 8 else "█"
		r3 = "1" if self.game_state == 0 or self.game_state == 2 or self.game_state == 9 else "█"
		r4 = ("2" if self.linear else "8") if self.game_state == 0 or self.game_state == 2 or self.game_state == 10 else "█"
		print(f"|{l1}|{l2}|   |{r1}|{r2}|\n|{l3}|{l4}|   |{r3}|{r4}|\n")
