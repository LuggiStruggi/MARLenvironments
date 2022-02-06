import gym
from gym import logger
import numpy as np

class TwoStepEnv(gym.Env):
	"""
	Description:
		Two Agents play a tabular game with a shared reward. In the first round the first agent can decide to
		either play one of the tabular games 2A or 2B in the next round.
	Source:
		The environment appeared in the QMIX paper.
	Observation:
		Type: MultiDiscrete(2)
		Num    Observation					0			  1				2
		0	   Agent 1 label			  Agent 1		Agent 2			?
		1	   Agent 2 label			  Agent 1		Agent 2			?
	Actions:
		Type: Discrete(2)
		Num    Action
		0	   Action 0
		1	   Action 1
	Reward:
		  
		  (0/0) | (0/1)    (1/0) | (1/1)
				/				 \
		  2A 0 1				2B 0 1
		   0|7|7|				 0|0|1|
		   1|7|7|				 1|1|8|

				
	Starting State:
		 Both agents recieve their respective label in the beginning in the second round only 2 for unknown.

	Episode Termination:
		 After two rounds
	"""

	metadata = {"render.modes": ["human"]}

	def __init__(self):
		
		self.game_state = 0

	def step(self, act):
		
		# unknown observation
		obs = [[2], [2]]

		if self.game_state == 0:
			# next game state
			if act[0] == 0:
				self.game_state = 1
			elif act[0] == 1:
				self.game_state = 2
			
			done = 0
			rew = 0
		
		elif self.game_state == 1:
			done = 1
			rew = 7
			
			# next game state
			self.game_state = 3 + 2*int(act[0]) + int(act[1])

		elif self.game_state == 2:
			done = 1
			if act[0] == 0 and act[1] == 0:
				rew = 0
			elif act[0] == 1 and act[1] == 1:
				rew = 8
			elif act[0] == 0 and act[1] == 1:
				rew = 1
			elif act[0] == 1 and act[1] == 0:
				rew = 1
			
			# next game state
			self.game_state = 7 + 2*int(act[0]) + int(act[1])
		
		# game is over yet step is called
		elif self.game_state >= 3:
			done = 1
			rew = 0
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		return np.array(obs), rew, done, {}

	def reset(self):
		self.game_state = 0
		obs = [[0], [1]]
		return np.array(obs)

	def render(self, mode="human"):
		l1 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 3 else "█"
		l2 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 4 else "█"
		l3 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 5 else "█"
		l4 = "7" if self.game_state == 0 or self.game_state == 1 or self.game_state == 6 else "█"
		r1 = "0" if self.game_state == 0 or self.game_state == 2 or self.game_state == 7 else "█"
		r2 = "1" if self.game_state == 0 or self.game_state == 2 or self.game_state == 8 else "█"
		r3 = "1" if self.game_state == 0 or self.game_state == 2 or self.game_state == 9 else "█"
		r4 = "8" if self.game_state == 0 or self.game_state == 2 or self.game_state == 10 else "█"
		print(f"|{l1}|{l2}|   |{r1}|{r2}|\n|{l3}|{l4}|	 |{r3}|{r4}|\n")
