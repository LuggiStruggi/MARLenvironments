import gym
from gym import logger
import numpy as np
from gym.envs.classic_control import rendering
from gym import spaces

class SwitchEnv(gym.Env):
	"""
	Description:
		Two Agents in a 2D-grid partially observable environment. The task is for both agents to reach the switch at the opposite side of the map.
		In the middle theres a corridor, to narrow for both agents to pass.

	Source:
		The environment appeared in the paper Value-Decomposition Networks FOr Cooperative Multi-Agent Learning. (https://arxiv.org/pdf/1706.05296.pdf)

	Observation:
		Type: np.array of shape (2, view, view) consisting of the partial observations for the two agents:
		Num		Obs
		0		Free space
		1		Wall
		2		Agent 0
		3 		Agent 1
		4		Switch for Agent 0
		5		Switch for Agent 1

	Actions:
		Type: np.array of shape (2). For each entry:
		Num    Action
		0	   Move forward
		1	   Turn left
		2      Turn right

	Reward:
		If an agent reaches its switch for the first time the reward is 1.

	Starting State:
		 Both agents start on opposing sides of the map with their respective observation.

	Episode Termination:
		 After both agents have found their respective switch.
	"""

	metadata = {"render.modes": ["human"]}

	def __init__(self, height: int = 7, width: int = 30, view: int = 5, flatten_obs: bool = False):
		self.pixel_map = self._create_corridor_map(height, width, view - 1)
		if view % 2 == 0:
			raise ValueError("View must be odd integer value such that agent can be centered.")
		self.view = view
		self.width = width
		self.height = height
		self.flatten = flatten_obs
		self.agents_pos = [[view-1+height//2, view-1, 1], [view-1+height//2, width+(view-1)-1, 3]]
		self.switch_pos = [[view-1+height//2, 11*width//12+(view-1)-1], [view-1+height//2, width//12+view]]
		self.switch_state = [0, 0]
		self.received_reward = False
		self.done = 0
		self.viewer = None

		obs_space = spaces.Box(low=0, high=5, shape=(self.view+1 if self.view % 2 == 0 else self.view, self.view))
		self.observation_space = spaces.Tuple((obs_space, obs_space))

		self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))

	def step(self, act: np.array):
		
		if self.done:
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		# act
		self._act_agent(0, act[0])				
		self._act_agent(1, act[1])

		# get observations
		obs = np.stack((self._get_observation(0), self._get_observation(1)))

		# both agents reach goal simultaneously
		if sum(self.switch_state) == 2 and not self.received_reward:
			rew = 2
			self.done = 1
		# both agents reached goal
		elif sum(self.switch_state) == 2:
			rew = 1
			self.done = 1
		# one agent reached goal
		elif sum(self.switch_state) == 1 and not self.received_reward:
			rew = 1
			self.done = 0
			self.received_reward = True
		# no agent reached goal
		else:
			rew = 0
			self.done = 0

		if self.flatten:
			obs = obs.reshape((2, self.view**2))

		return obs, rew, self.done, {}
	
	def reset(self) -> np.array:

		self.pixel_map[self.agents_pos[0][0], self.agents_pos[0][1]] = 0
		self.pixel_map[self.agents_pos[1][0], self.agents_pos[1][1]] = 0
		self.pixel_map[self.switch_pos[0][0], self.switch_pos[0][1]] = 4
		self.pixel_map[self.switch_pos[1][0], self.switch_pos[1][1]] = 5

		self.agents_pos = [[self.view-1+self.height//2, self.view-1, 1], [self.view-1+self.height//2, self.width+(self.view-1)-1, 3]]
		self.switch_pos = [[self.view-1+self.height//2, 11*self.width//12+(self.view-1)-1], [self.view-1+self.height//2, self.width//12+self.view]]
		
		self.pixel_map[self.agents_pos[0][0], self.agents_pos[0][1]] = 2
		self.pixel_map[self.agents_pos[1][0], self.agents_pos[1][1]] = 3

		self.switch_state = [0, 0]
		obs = np.stack((self._get_observation(0), self._get_observation(1)))

		if self.flatten:
			obs = obs.reshape((2, self.view**2))

		self.done = 0
		
		return obs

	def render(self, mode: str = "human"):
		
		if self.viewer == None:
			self.viewer = rendering.Viewer((100*self.pixel_map.shape[1])//4, (100*self.pixel_map.shape[0])//4)
			self.viewer.set_bounds(0, self.pixel_map.shape[1], 0, self.pixel_map.shape[0])
			for i in range(self.pixel_map.shape[0]):
				for j in range(self.pixel_map.shape[1]):
					if self.pixel_map[i][j] == 1:
						wall = rendering.FilledPolygon([(j, i), (j, i+1), (j+1, i+1), (j+1, i)])
						self.viewer.add_geom(wall)

		for i in range(self.pixel_map.shape[0]):
			for j in range(self.pixel_map.shape[1]):
				
				if self.pixel_map[i][j] != 0:
					x = j
					y = self.pixel_map.shape[0] - i - 1

				if self.pixel_map[i][j] == 2:
					agent, view = self._get_rendered_agent(0, x, y, self.agents_pos[0][2])
					self.viewer.add_onetime(agent)
					self.viewer.add_onetime(view)

				elif self.pixel_map[i][j] == 3:
					agent, view = self._get_rendered_agent(1, x, y, self.agents_pos[1][2])
					self.viewer.add_onetime(agent)
					self.viewer.add_onetime(view)

				elif self.pixel_map[i][j] == 4:
					switch = self._get_rendered_switch(0, x, y)
					self.viewer.add_onetime(switch)

				elif self.pixel_map[i][j] == 5:
					switch = self._get_rendered_switch(1, x, y)
					self.viewer.add_onetime(switch)

		self.viewer.render(return_rgb_array=mode == "rgb_array")


	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None	


	def _get_observation(self, agent_id: int) -> np.array:
		
		y = self.agents_pos[agent_id][0]
		x = self.agents_pos[agent_id][1]
		rot = self.agents_pos[agent_id][2]

		if rot == 0:
			obs = self.pixel_map[y-self.view+1:y+1, x-self.view//2:x+self.view//2+1]
		elif rot == 1:
			obs = self.pixel_map[y-self.view//2:y+self.view//2+1, x:x+self.view]
		elif rot == 2:
			obs = self.pixel_map[y:y+self.view, x-self.view//2:x+self.view//2+1]
		elif rot == 3:
			obs = self.pixel_map[y-self.view//2:y+self.view//2+1, x-self.view+1:x+1]
		obs = np.rot90(m=obs, k=rot)
		return obs


	def _rotate_agent(self, agent_id: int, right: bool):
		
		self.agents_pos[agent_id][2] += (1 if right else -1)
		self.agents_pos[agent_id][2] %= 4


	def _move_agent(self, agent_id: int):
		
		y = self.agents_pos[agent_id][0]
		x = self.agents_pos[agent_id][1]
		rot = self.agents_pos[agent_id][2]
		
		if rot == 0 and self.pixel_map[y-1, x] not in [1, 2, 3]: 
			self.agents_pos[agent_id][0] -= 1
			self.pixel_map[y, x] = 0
			self.pixel_map[y-1, x] = agent_id + 2
		elif rot == 1 and self.pixel_map[y, x+1] not in [1, 2, 3]:
			self.agents_pos[agent_id][1] += 1
			self.pixel_map[y, x] = 0
			self.pixel_map[y, x+1] = agent_id + 2
		elif rot == 2 and self.pixel_map[y+1, x] not in [1, 2, 3]:
			self.agents_pos[agent_id][0] += 1
			self.pixel_map[y, x] = 0
			self.pixel_map[y+1, x] = agent_id + 2
		elif rot == 3 and self.pixel_map[y, x-1] not in [1, 2, 3]:
			self.agents_pos[agent_id][1] -= 1
			self.pixel_map[y, x] = 0
			self.pixel_map[y, x-1] = agent_id + 2
	
		for switch_id in range(2):
			visible = True
			for agent_id in range(2):	
				if self.agents_pos[agent_id][0] == self.switch_pos[switch_id][0] and self.agents_pos[agent_id][1] == self.switch_pos[switch_id][1]:
					visible = False
					if agent_id == switch_id:
						self.switch_state[switch_id] = 1
			if visible:
				self.pixel_map[self.switch_pos[switch_id][0], self.switch_pos[switch_id][1]] = switch_id + 4

	
	def _act_agent(self, agent_id: int, act: int):
		
		if act == 0:
			self._move_agent(agent_id)
		elif act == 1 or act == 2:
			self._rotate_agent(agent_id, right=act-1)


	def _get_rendered_agent(self, agent_id: int, x: int, y: int, rot: int) -> rendering.Geom:
		
		if rot == 0:
			agent = rendering.FilledPolygon([(x + 1/2, y + 5/6), (x + 4/5, y + 1/6), (x + 1/5, y + 1/6)])
			view = rendering.PolyLine([(x-self.view//2, y+self.view), (x+self.view//2+1, y+self.view), (x+self.view//2+1, y), (x-self.view//2, y)],close=True)
		elif rot == 1:
			agent = rendering.FilledPolygon([(x + 5/6, y + 1/2), (x + 1/6, y + 4/5), (x + 1/6, y + 1/5)])
			view = rendering.PolyLine([(x, y+self.view//2+1), (x+self.view, y+self.view//2+1), (x+self.view, y-self.view//2), (x, y-self.view//2)],close=True)
		elif rot == 2:
			agent = rendering.FilledPolygon([(x + 1/2, y + 1/6), (x + 4/5, y + 5/6), (x + 1/5, y + 5/6)])
			view = rendering.PolyLine([(x-self.view//2, y-self.view+1), (x+self.view//2+1, y-self.view+1), (x+self.view//2+1, y+1), (x-self.view//2, y+1)],close=True)
		elif rot == 3:
			agent = rendering.FilledPolygon([(x + 1/6, y + 1/2), (x + 5/6, y + 4/5), (x + 5/6, y + 1/5)])
			view = rendering.PolyLine([(x+1, y+self.view//2+1), (x-self.view+1, y+self.view//2+1), (x-self.view+1, y-self.view//2), (x+1, y-self.view//2)],close=True)
		agent.set_color(*((1, 0, 0) if agent_id else (0, 0, 1)))
		view.set_color(*((1, 0, 0) if agent_id else (0, 0, 1)))
		
		return agent, view


	def _get_rendered_switch(self, switch_id: int, x: int, y: int) -> rendering.Geom:
		switch = rendering.FilledPolygon([(x + 1/2, y + 1/4), (x + 3/4, y + 1/2), (x + 1/2, y + 3/4), (x + 1/4, y + 1/2)])
		switch.set_color(*((1, 0, 0) if switch_id else (0, 0, 1)))
		return switch

	def _create_corridor_map(self, height: int, width: int, padding: int) -> np.array:
		pixel_map = []
		room_width = (width + 1)//4
		corridor_length = width - 2*room_width
		corridor_position = height//2

		for h in range(height):
			pixel_map.append([])
			for w in range(width):
				if w >= room_width and w < room_width + corridor_length and h != corridor_position:
					tile = 1
				else:
					tile = 0
				pixel_map[h].append(tile)

		return np.pad(np.array(pixel_map), padding, 'constant', constant_values=(1))
