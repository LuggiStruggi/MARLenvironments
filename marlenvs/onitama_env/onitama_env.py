import numpy as np
import gym
from gym import spaces, logger
import random

class Card:
	
	def __init__(self, name, move, color):
		self.name = name
		if move.shape != (5, 5):
			raise ValueError(f"Shape of movement card must be (5, 5) but was {move.shape} for card {name}")
		self.move = move
		self.color = color
		self.actions = []
		for i in range(5):
			for j in range(5):
				if self.move[i][j] == 1:
					self.actions.append((i-2, j-2))	

CARDS = [
	Card(name="Tiger", move=np.array([[0, 0, 1, 0, 0], [0]*5, [0]*5, [0, 0, 1, 0, 0], [0]*5]), color="blue"),
	Card(name="Dragon", move=np.array([[0]*5, [1, 0, 0, 0, 1], [0]*5, [0, 1, 0, 1, 0], [0]*5]), color="red"),
	Card(name="Frog", move=np.array([[0]*5, [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0]*5]), color="red"),
	Card(name="Rabbit", move=np.array([[0]*5, [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0]*5]), color="blue"),
	Card(name="Crab", move=np.array([[0]*5, [0, 0, 1, 0, 0], [1, 0, 0, 0, 1], [0]*5, [0]*5]), color="blue"),
	Card(name="Elephant", move=np.array([[0]*5, [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0]*5, [0]*5]), color="red"),
	Card(name="Goose", move=np.array([[0]*5, [0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0]*5]), color="blue"),
	Card(name="Rooster", move=np.array([[0]*5, [0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0]*5]), color="red"),
	Card(name="Monkey", move=np.array([[0]*5, [0, 1, 0, 1, 0], [0]*5, [0, 1, 0, 1, 0], [0]*5]), color="blue"),
	Card(name="Mantis", move=np.array([[0]*5, [0, 1, 0, 1, 0], [0]*5, [0, 0, 1, 0, 0], [0]*5]), color="red"),
	Card(name="Horse", move=np.array([[0]*5, [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0]*5]), color="red"),
	Card(name="Ox", move=np.array([[0]*5, [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0]*5]), color="blue"),
	Card(name="Crane", move=np.array([[0]*5, [0, 0, 1, 0, 0], [0]*5, [0, 1, 0, 1, 0], [0]*5]), color="blue"),
	Card(name="Boar", move=np.array([[0]*5, [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0]*5, [0]*5]), color="red"),
	Card(name="Eel", move=np.array([[0]*5, [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0]*5]), color="blue"),
	Card(name="Cobra", move=np.array([[0]*5, [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0]*5]), color="red"),
]


class OnitamaEnv(gym.Env):
	"""
	Description:
		Chess like game where two agents play against each other. Each agent has one master- and 4 student-pawns. The movements are defined by movement cards.
		The Goal is to capture the enemy master or reach the middle field of the oppisite role with ones master. (https://en.wikipedia.org/wiki/Onitama)

	Source:
		Game was created by Shimpei Sato in 2014.

	Observation:
		Type: np.array of shape (6, 5, 5) consisting of the game board (5x5) and the 5 movement cards (5x5) out of the perspective of the current player:
		Num		Obs
		0		Free space
		1		own student-pawn (board) / possible movement tile (move-card)
		2		own master-pawn
		-1 		other student pawn
		-2		other master pawn

	Actions:
		Type: int act_id which encodes any possible even non-legal action. Meaning moving from any field to any other field using any of the two movement cards:
		Num    Action
		0	   n in range (0, 25*25*2-1)

	Reward:
		If the own master-pawn reaches the goal field or other master-pawn is captured reward is 1 else 0.

	Starting State:
		 Agent with the same color as the idle-card starts.

	Episode Termination:
		 After any agent has won.
	"""


	
	def __init__(self, cards_red=None, cards_blue=None, cards_idle=None):
		
		self.assign_cards(cards_red, cards_blue, cards_idle)
		self.state = np.array([[-1, -1, -2, -1, -1], [0]*5, [0]*5, [0]*5, [1, 1, 2, 1, 1]])
		self.observation_space = spaces.Box(-2, 2, (5, 5))
		self.action_space = spaces.Discrete(25 * 25 * 2)
		self.turn = self.cards_idle.color
		self.legal_actions = self.get_legal_actions(self.turn)
		self.done = False
	
	def reset(self, cards_red=None, cards_blue=None, cards_idle=None):

		self.assign_cards(cards_red, cards_blue, cards_idle)
		self.state = np.array([[-1, -1, -2, -1, -1], [0]*5, [0]*5, [0]*5, [1, 1, 2, 1, 1]])
		self.turn = self.cards_idle.color
		self.legal_actions = self.get_legal_actions(self.turn)
		self.done = False

	def step(self, act):

		if self.done:
			logger.warn(
				"You are calling 'step()' even though this "
				"environment has already returned done = True. You "
				"should always call 'reset()' once you receive 'done = "
				"True' -- any further steps are undefined behavior."
			)

		# act
		self.move_legally(act)		

		# obs
		if self.turn == 'blue':
			obs = np.stack((self.state, *(c.move for c in self.cards_blue), *(np.flip(c.move) for c in self.cards_red), self.cards_idle.move))
		elif self.turn == 'red':
			obs = np.stack((np.flip(-1*self.state), *(c.move for c in self.cards_red), *(np.flip(c.move) for c in self.cards_blue), self.cards_idle.move))

	
		# reward	
		rew = 0
		if self.get_winner():
			rew = 1
			self.done = True
		
		return obs, rew, self.done, {}


	def assign_cards(self, cards_red=None, cards_blue=None, cards_idle=None):
		
		if cards_red is None:
			self.cards_red = random.sample(CARDS, 2)
		
		else:
			if len(cards_red) != 2:
				raise ValueError("Length of red cards must be 2.")
			
			self.cards_red = []
			for c in cards_red:
				if c is None:
					self.cards_red = random.sample([c for c in CARDS if c not in self.cards_red], 2)
				elif isinstance(c, str):
					new = [x for x in CARDS if x.name == c]
					if new == []:
						raise ValueError(f"Card with name {c} doesn't exist.")
					self.cards_red += new
				elif isinstance(c, Card):
					self.cards_red.append(c)
				else:
					raise ValueError("Card must be either None, String or Card.")

		if cards_blue is None:
			self.cards_blue = random.sample([c for c in CARDS if c not in self.cards_red], 2)
		else:
			if len(cards_blue) != 2:
				raise ValueError("Length of red cards must be 2.")
			self.cards_blue = []
			for c in cards_blue:
				if c is None:
					self.cards_blue = random.sample([c for c in CARDS if c not in self.cards_red and c not in cards.blue], 2)
				elif isinstance(c, str):
					new = [x for x in CARDS if x.name == c]
					if new == []:
						raise ValueError(f"Card with name {c} doesn't exist.")
					self.cards_blue += new
				elif isinstance(c, Card):
					self.cards_blue.append(c)
				else:
					raise ValueError("Card must be either None, String or Card.")

		if cards_idle is None:
			self.cards_idle = random.choice([c for c in CARDS if c not in self.cards_red and c not in self.cards_blue])
		elif isinstance(cards_idle, str):
			self.cards_idle = [x for x in CARDS if x.name == cards_idle][0]
		elif isinstance(cards_idle, Card):
			self.cards_idle = c
		else:
			raise ValueError("Card must be either None, String or Card.")

		# check if cards with duplicate movesets are at play
		moves = [frozenset(x.actions) for x in self.cards_blue + self.cards_red + [self.cards_idle]]
		if len(moves) != len(set(moves)):
			logger.warn("Playing game with a duplicate card. In the real game this is not possible.")


	# action decoding	
	def _id_to_act(self, act_id):
		card_id = 0
		if act_id >= 25*25:
			act_id -= 25*25
			card_id = 1
		start_pos, end_pos = [divmod(x, 5) for x in divmod(act_id, 25)]
		return start_pos, end_pos, card_id

	# action encoding
	def _act_to_id(self, start_pos, end_pos, card_id):
		return 25*(start_pos[0] * 5 + start_pos[1]) + end_pos[0] * 5 + end_pos[1] + card_id * 25 * 25


	def get_legal_actions(self, color):
		actions = set()
		for i in range(5):
			for j in range(5):
				if self.state[i, j] == 0:
					continue
				
				if self.state[i, j] > 0 and color == 'blue':
					for cid, card in enumerate(self.cards_blue):
						for di, dj in card.actions:
							if  0 <= (i + di) < 5 and 0 <= (j + dj) < 5 and self.state[i + di, j + dj] <= 0:
								actions.add(self._act_to_id((i, j), (i + di, j + dj), cid))
						
				elif self.state[i, j] < 0 and color == 'red':
					for cid, card in enumerate(self.cards_red):
						for di, dj in card.actions:
							if  0 <= (i - di) < 5 and 0 <= (j - dj) < 5 and self.state[i - di, j - dj] >= 0:
								actions.add(self._act_to_id((i, j), (i - di, j - dj), cid))
		
		return np.array(list(actions))


	def move_legally(self, act_id):

		# check if action is legal and if not choose closest legal action by id	
		legal_act = self.legal_actions
	
		# no legal move
		if legal_act.size == 0:
			self.turn = "red" if self.turn == "blue" else "blue"
			self.legal_actions = self.get_legal_actions(self.turn)
			return 1250 # 25*25*2 (new id for pass-action)

		if act_id not in legal_act:
			act_id = legal_act[np.abs(legal_act - act_id).argmin()]
		pos_start, pos_end, card_idx = self._id_to_act(act_id)
		self.state[pos_start], self.state[pos_end] = 0, self.state[pos_start]
		if self.turn == "blue":
			self.cards_blue[card_idx], self.cards_idle = self.cards_idle, self.cards_blue[card_idx]
			self.turn = "red"
		elif self.turn == "red":
			self.cards_red[card_idx], self.cards_idle = self.cards_idle, self.cards_red[card_idx]
			self.turn = "blue"
		
		# get new legal actions
		self.legal_actions = self.get_legal_actions(self.turn)
		return act_id


	def get_winner(self):

		if self.state[0, 2] == 2 or not any(-2 in row for row in self.state):
			return 'blue'

		elif self.state[4, 2] == -2 or not any(2 in row for row in self.state):
			return 'red'

		else:
			return None
	

	# function to get next state based on id
	def get_next_state(self, act_id, turn):

		board = np.copy(self.state)
		blue = np.copy(self.cards_blue)
		red = np.copy(self.cards_red)
		idle = np.copy(self.cards_idle)

		pos_start, pos_end, card_idx = self._id_to_act(act_id)
		board[pos_start], board[pos_end] = 0, self.state[pos_start]
		
		if turn == "blue":
			blue[card_idx], idle = self.cards_idle, self.cards_blue[card_idx]
			obs = np.stack((board, *(c.move for c in blue), *(np.flip(c.move) for c in red), idle.move))
		
		elif turn == "red":
			red[card_idx], idle = self.cards_idle, self.cards_red[card_idx]
			obs = np.stack((np.flip(-1*board), *(c.move for c in red), *(np.flip(c.move) for c in blue), idle.move))

		return obs
	

	def get_all_possible_next_states(self):
		
		act = self.get_legal_actions(self.turn)
		obs = np.zeros((act.size, 6, 5, 5))
		
		for i, act_id in enumerate(act):
			obs[i, :, :, :] = self.get_next_state(act_id, self.turn)
		
		return act, obs
		

	def __str__(self):

		visual = {-2: "R", -1: "r", 0: ".", 1: "b", 2: "B", 3: "o", 4: "x", 5: " "}
		mid = np.zeros((5, 5))
		mid[2, 2] = 4
		board = np.ones((5,10))*5
		board[:,::2] = self.state
		red = np.concatenate((np.ones((5, 2))*5, np.flip(self.cards_red[0].move*3) + mid, np.ones((5, 3))*5, np.flip(self.cards_red[1].move*3) + mid), axis=1)
		blue = np.concatenate((np.ones((5, 2))*5, self.cards_blue[0].move*3 + mid, np.ones((5, 3))*5, self.cards_blue[1].move*3 + mid), axis=1)
		game = np.concatenate((np.ones((5, 4))*5, board, np.ones((5, 2))*5, self.cards_idle.move*3 + mid), axis=1) 
		out = "\n".join("".join(visual[int(field)] for field in row) for row in red) + "\n"
		out += "\n"
		out += "\n".join("".join(visual[int(field)] for field in row) for row in game) + "\n"
		out += "\n"
		out += "\n".join("".join(visual[int(field)] for field in row) for row in blue) + "\n"
		return out
	

if __name__ == '__main__':
	env = OnitamaEnv()
	print(env)
	for i in range(100):
		print("-"*30)
		print(f"MOVE {i+1} by {env.turn}:")
		obs, rew, done, _ = env.step(random.randint(0, 25*25*2))
		print(env)
		input("")
