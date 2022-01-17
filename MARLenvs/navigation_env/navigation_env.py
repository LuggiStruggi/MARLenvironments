import gym
from gym import logger, spaces
import numpy as np
from gym.envs.classic_control import rendering

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
                 hold_steps: int = 10, sparse: bool = False):
		
		self.border = world_size*n_agents
		self.n_agents = n_agents
		self.agents = np.random.rand(n_agents, 2)*(self.border - 1) + 0.5
		self.landmarks = np.random.rand(n_agents, 2)*(self.border - 0.5) + 0.25
		self.tau = tau
		self.sparse = sparse
		
		self.low_state = np.array(
        	[[-np.sqrt(2)*self.border for j in range(n_agents*4)] for i in range(n_agents)], dtype=np.float32
        )

		self.high_state = np.array(
        	[[np.sqrt(2)*self.border for j in range(n_agents*4)] for i in range(n_agents)], dtype=np.float32
        )

		self.action_space = spaces.Box(
			low=np.array([[-np.pi, -1.0]]*n_agents), high=np.array([[np.pi, 1.0]]*n_agents), dtype=np.float32
        )
		
		self.observation_space = spaces.Box(
			low=self.low_state, high=self.high_state, dtype=np.float32
        )

		self.done = False
		self.step_counter = 0
		self.max_steps = max_steps
		
		self.hold_step_counter = 0
		self.hold_steps = hold_steps

		self.viewer = None


	def step(self, act: np.array):
		
		if self.done:
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		self._check_action(act)
		trans = [[r*np.cos(phi), r*np.sin(phi)] for phi, r in act]
		self.agents += trans
		self.agents = np.clip(self.agents, 0.5, self.border-0.5)
		obs, dist = self._get_observations()
		
		# number of agent collistions
		n_collisions = (np.count_nonzero(dist[:, :self.n_agents] < 1) - self.n_agents)//2
		
		# number of uncovered landmarks
		n_uncovered = np.count_nonzero(np.all(dist[:, self.n_agents:] > 0.25, axis=0))
		
		# distance of closest agent for each landmark
		min_dist = np.sum(np.min(dist[:, self.n_agents:], axis=0))

		
		if self.sparse:
			rew = - (self.tau * n_uncovered + (1-self.tau) * n_collisions)
	
		else:
			rew = - (self.tau * min_dist + (1-self.tau) * n_collisions)
		

		self.step_counter += 1

		if n_uncovered == 0:
			self.hold_step_counter += 1
		
		else:
			self.hold_step_counter = 0

		if self.step_counter >= self.max_steps or self.hold_step_counter >= self.hold_steps:
			self.done = True

		return obs, rew, self.done, {}


	def reset(self, reset_positions: bool = True) -> np.array:
		
		if reset_positions:	
			self.agents = np.random.rand(self.n_agents, 2)*(self.border - 1) + 0.5
			self.landmarks = np.random.rand(self.n_agents, 2)*(self.border - 0.5) + 0.25

		self.done = False
		self.step_counter = 0
		self.hold_step_counter = 0
		obs, _ = self._get_observations()
		
		return obs


	def _check_action(self, action):
		
		if action.shape != (self.n_agents, 2):
			raise ValueError(f"Action has wrong shape. Expected ({self.n_agents}, 2), got {action.shape}.")
		


	def _get_observations(self) -> np.array:
		# n_agents x (n_agents + n_landmarks)
		observations = np.zeros((self.n_agents, 4*self.n_agents))
		dist = np.zeros((self.n_agents, 2*self.n_agents))
		for i in range(len(self.agents)):
			for j in range(len(self.agents)):	
				observations[i, 2*self.n_agents + 2*j : 2*self.n_agents + 2*j + 2] = self.agents[i] - self.landmarks[j]
				dist[i, self.n_agents + j] = np.linalg.norm(self.agents[i] - self.landmarks[j])
				if i == j:
					continue
				observations[i, 2*j:2*j+2] = self.agents[i] - self.agents[j]
				dist[i, j] = np.linalg.norm(self.agents[i] - self.agents[j])
				dist[j, i] = dist[i, j]
		return observations, dist

	# Render environment

	def render(self, mode="human"):
		
		if self.viewer == None:
			length = int(self._world_to_render(self.border))
			self.viewer = rendering.Viewer(width=length, height=length)
			self.viewer.set_bounds(0, self.border, 0, self.border)
			for i in range(len(self.landmarks)):
				landmark = rendering.make_circle(radius=0.25)
				landmark.set_color(1, 0, 0)
				landmark.add_attr(rendering.Transform(translation=(self.landmarks[i][0], self.landmarks[i][1])))
				self.viewer.add_geom(landmark)

		for i in range(len(self.agents)):
			agent = rendering.make_circle(radius=0.5)
			agent.set_color(0, 1, 0)	
			agent.add_attr(rendering.Transform(translation=(self.agents[i][0], self.agents[i][1])))
			self.viewer.add_onetime(agent)
		
		self.viewer.render(return_rgb_array=mode == "rgb_array")

	
	def _world_to_render(self, coordinates):
		if isinstance(coordinates, int) or isinstance(coordinates, float):
			return 100*coordinates//4
		elif isinstance(coordinates, list):
			return [c*100//4 for c in coordinates]
		elif isinstance(coordinates, tuple):
			return (c*100//4 for c in coordinates)

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None	



if __name__ == "__main__":
	n_agents = 3
	env = NavigationEnv(n_agents)
	env.render()
	while True:
		obs, r, done, _ = env.step(env.action_space.sample())
		#env.step(np.array([[6.28, 0.2]]*n_agents), normalize=True)
		env.render()