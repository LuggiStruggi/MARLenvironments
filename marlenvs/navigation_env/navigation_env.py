import gym
from gym import logger, spaces
import numpy as np
import pygame
from pygame import gfxdraw

class NavigationEnv(gym.Env):
	"""
	Description:
		N Agents cooperatively try to cover a set of N landmarks while avoiding colliding with other agents.
	Source:
		The environment appeared in the MADDPG paper (https://arxiv.org/pdf/1706.02275.pdf).
	Observation:
		Type: Box(n_agents, n_agents*4)
        Num    						 		    Observation               		  Min                             Max
        [i < n_agents, 2*j    <  n_agents*2]    Relative xPos agent to agent      -sqrt(2)*n_agents*world_size    sqrt(2)*n_agents*world_size
        [i < n_agents, 2*j+1  <  n_agents*2]    Relative yPos agent to agent      -sqrt(2)*n_agents*world_size    sqrt(2)*n_agents*world_size
        [i < n_agents, n_agents*2 <=    2*j]    Relative xPos agent to landmark   -sqrt(2)*n_agents*world_size    sqrt(2)*n_agents*world_size
        [i < n_agents, n_agents*2 <=  2*j+1]    Relative xPos agent to landmark   -sqrt(2)*n_agents*world_size    sqrt(2)*n_agents*world_size

	Actions:
		Type: Box(n_agents, 2)
        Num    						 		 	Observation               		  Min                             Max
		[i < n_agents, 0]						Angle of movement				  -pi							  pi
		[i < n_agents, 1]						Distance of movement			  0.0							  1.0

	Reward:
		 The agents are punished for uncovered landmarks and for colliding. The trade off between both can be adjusted with the parameter tau:
		 reward = - (tau * n_uncovered + (1-tau) * n_colliding)
 
				
	Starting State:
		 Agents start at random position in (2.5*N)x(2.5*N) units big environment one agents diameter is 1 unit.

	Episode Termination:
		 Either after max_steps or after keeping all landmarks covered for hold_steps.
	"""

	metadata = {"render.modes": ["human"]}

	def __init__(self, n_agents, world_size: float = 2.5, tau: float = 0.5, max_steps: int = np.inf,
                 hold_steps: int = 10, sparse: bool = False, init_agent_pos: np.array = None, init_landmark_pos: np.array = None, action_type = "translation"):
		
		self.border = world_size*n_agents
		self.n_agents = n_agents
		self.tau = tau
		self.sparse = sparse
		self.action_type = action_type

		self.init_agent_pos = None if init_agent_pos is None else init_agent_pos
		self.init_landmark_pos = None if init_landmark_pos is None else init_landmark_pos
		
		self.low_state = np.array(
        	[[-self.border for j in range(n_agents*4 - 2)] for i in range(n_agents)], dtype=np.float32
        )

		self.high_state = np.array(
        	[[self.border for j in range(n_agents*4 - 2)] for i in range(n_agents)], dtype=np.float32
        )
		
		if self.action_type == "translation":
			self.action_space = spaces.Box(
				low=np.array([[-1.0, -1.0]]*n_agents, dtype=np.float32), high=np.array([[1.0, 1.0]]*n_agents, dtype=np.float32), dtype=np.float32
        	)
		elif self.action_type == "rotation_translation":
			self.action_space = spaces.Box(
				low=np.array([[-np.pi, 0.0]]*n_agents, dtype=np.float32), high=np.array([[np.pi, 1.0]]*n_agents, dtype=np.float32), dtype=np.float32
        	)
		elif self.action_type == "velocity":
			self.action_space = spaces.Box(
				low=np.array([[-0.3, -0.3]]*n_agents, dtype=np.float32), high=np.array([[0.3, 0.3]]*n_agents, dtype=np.float32), dtype=np.float32
        	)
		elif self.action_type == "rotation_velocity":
			self.action_space = spaces.Box(
				low=np.array([[-np.pi, 0.0]]*n_agents, dtype=np.float32), high=np.array([[np.pi, 0.3]]*n_agents, dtype=np.float32), dtype=np.float32
        	)

		
		self.observation_space = spaces.Box(
			low=self.low_state, high=self.high_state, dtype=np.float32
        )

		
		# min reward for all situations -> all landmarks in one corner all agents in the corner diagonally across
		# -> maximum distance from landmarks + maximal overlapping
		self.agents = np.full(shape=(n_agents, 2), fill_value=self.border - 0.5)
		self.agents_vel = np.zeros((n_agents, 2))

		self.landmarks = np.full(shape=(n_agents, 2), fill_value=0.25)
		_, dist = self._get_observations()
		min_rew, _ = self._calculate_reward(dist)

		self.reward_space = spaces.Box(
			low=min_rew, high=0, shape=(1,)
		)
	
		if self.init_agent_pos is None:
			self.agents = np.random.rand(self.n_agents, 2)*(self.border - 1) + 0.5
		else:
			if self.init_agent_pos.shape != (self.n_agents, 2):
				raise ValueError(f"Agent position shape incorrect. Should be: {(self.n_agents, 2)}. Is {self.init_agent_pos.shape}")
			self.agents = self.init_agent_pos.copy()

		if self.init_landmark_pos is None:
			self.landmarks = np.random.rand(self.n_agents, 2)*(self.border - 0.5) + 0.25
		else:
			if self.init_landmark_pos.shape != (self.n_agents, 2):
				raise ValueError(f"Landmark position shape incorrect. Should be: {(self.n_agents, 2)}. Is {self.init_landmark_pos.shape}")
			self.landmarks = self.init_landmark_pos.copy()

		self.done = False
		self.step_counter = 0
		self.max_steps = max_steps
		
		self.hold_step_counter = 0
		self.hold_steps = hold_steps

		self.surface = None
		self.screen = None

	def step(self, act: np.array):
		
		if self.done:
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		self._check_action(act)

		if self.action_type == "translation":
			self.agents_vel = act
		elif self.action_type == "rotation_translation":
			self.agents_vel = [[r*np.cos(phi), r*np.sin(phi)] for phi, r in act]
		elif self.action_type == "velocity":
			self.agents_vel += act
			self.agents_vel.clip(min=-1.0, max=1.0)
		elif self.action_type == "rotation_velocity":
			self.agents_vel += [[r*np.cos(phi), r*np.sin(phi)] for phi, r in act]
			self.agents_vel.clip(min=-1.0, max=1.0)

		self.agents += self.agents_vel
		self.agents = np.clip(self.agents, 0.5, self.border-0.5)
		obs, dist = self._get_observations()
		
		rew, n_uncovered = self._calculate_reward(dist)

		self.step_counter += 1

		if n_uncovered == 0:
			self.hold_step_counter += 1
		
		else:
			self.hold_step_counter = 0

		if self.step_counter >= self.max_steps or self.hold_step_counter >= self.hold_steps:
			self.done = True
		
		return obs, rew, self.done, {}


	def reset(self) -> np.array:
		
		if self.init_agent_pos is None:
			self.agents = np.random.rand(self.n_agents, 2)*(self.border - 1) + 0.5
		else:
			self.agents = self.init_agent_pos.copy()

		if self.init_landmark_pos is None:
			self.landmarks = np.random.rand(self.n_agents, 2)*(self.border - 0.5) + 0.25
		else:
			self.landmarks = self.init_landmark_pos.copy()

		self.done = False
		self.step_counter = 0
		self.hold_step_counter = 0
		obs, _ = self._get_observations()
		
		return obs


	def _calculate_reward(self, dist):
	
		# number of agent collistions (all distances between two agents smaller than agent diameter. Divided by two due to every distance twice in matrix) 
		n_collisions = (np.count_nonzero(dist[:, :self.n_agents - 1] < 1))//2
		
		# number of uncovered landmarks
		n_uncovered = np.count_nonzero(np.all(dist[:, self.n_agents - 1:] > 0.25, axis=0))
		
		# sum of distances of closest agent for each landmark
		min_dist = np.sum(np.min(dist[:, self.n_agents - 1:], axis=0))
		
		if self.sparse:
			rew = - (self.tau * n_uncovered + (1-self.tau) * n_collisions)
	
		else:
			rew = - (self.tau * min_dist + (1-self.tau) * n_collisions)

		return rew, n_uncovered
	

	def _check_action(self, action):
		
		if action.shape != (self.n_agents, 2):
			raise ValueError(f"Action has wrong shape. Expected ({self.n_agents}, 2), got {action.shape}.")
				

	def _get_observations(self) -> np.array:
		# n_agents x (n_agents - 1 + n_landmarks)*2
		observations = np.zeros((self.n_agents, (2*self.n_agents - 1)*2))
		# n_agents x (n_agents - 1 + n_landmarks)
		dist = np.zeros((self.n_agents, 2*self.n_agents - 1))

		for i in range(self.n_agents):
			for j in range(self.n_agents):
				
				# agent-landmark x, y distances
				index = (self.n_agents-1)*2 + 2*j
				observations[i, index:index + 2] = self.agents[i] - self.landmarks[j]
				
				# agent-landmark direct distances
				dist[i, self.n_agents-1 + j] = np.linalg.norm(self.agents[i] - self.landmarks[j], ord=2)
				
				# agent-agent not with itself
				if i == j:
					continue
				
				k = j if j < i else j - 1

				# agent-agent x, y distances
				observations[i, 2*k:2*k+2] = self.agents[i] - self.agents[j]
				
				# agent-landmark direct distances
				dist[i, k] = np.linalg.norm(self.agents[i] - self.agents[j], ord=2)			
		
		return observations, dist

	# Render environment
	def render(self, mode="human"):
		
		if self.surface == None:
			length = int(self._world_to_render(self.border))
			self.screen = pygame.display.set_mode((length, length))
			self.surface = pygame.Surface((length, length))
		
		self.surface.fill((255, 255, 255))

		for landmark in self.landmarks:
			gfxdraw.filled_circle(self.surface, self._world_to_render(landmark[0]), self._world_to_render(landmark[1]), self._world_to_render(0.25), (255, 0, 0))

		for agent in self.agents:
			gfxdraw.filled_circle(self.surface, self._world_to_render(agent[0]), self._world_to_render(agent[1]), self._world_to_render(0.5), (0, 255, 0))
		
		self.screen.blit(self.surface, (0, 0))
		pygame.display.flip()
	
	def _world_to_render(self, coordinates):
		if isinstance(coordinates, int) or isinstance(coordinates, float):
			return int((100*coordinates)//4)
		elif isinstance(coordinates, list):
			return [int((c*100)//4) for c in coordinates]
		elif isinstance(coordinates, tuple):
			return (int((c*100)//4) for c in coordinates)

	def close(self):
		if self.screen:
			self.surface = None
			self.screen = None
			pygame.quit()

	def get_obs_dim(self):
		return self.n_agents*4 - 2

	def get_act_dim(self):
		return 2



if __name__ == '__main__':
	
	from marlenvs.wrappers import NormalizeActWrapper, NormalizeObsWrapper
	import time

	env = NavigationEnv(n_agents = 3, max_steps = 100, tau=1.0, world_size=5)
	env = NormalizeActWrapper(env)
	env = NormalizeObsWrapper(env)
	env.reset()
	env.render()
	while True:
		obs, rew, _, _ =  env.step(np.array([[0, 0], [0, 0], [0, 0]]))
		env.render()
		time.sleep(0.03)
