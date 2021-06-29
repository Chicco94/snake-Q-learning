import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

	def __init__(self):
		self.n_games = 0
		self.epsilon = 0 # randomness
		self.gamma = 0 # discount rate
		self.memory = deque(maxlen=MAX_MEMORY) # automatically remove the oldes memory
		self.mode = None # TODO
		self.trainer = None # TODO

	def get_state(self, game):
		'''get the state of the game:
		    - head of the snake
		'''
		# head of the snake
		head = game.snake[0]

		# danger, check if hits the boundary
		point_l = Point(head.x - 20, head.y)
		point_r = Point(head.x + 20, head.y)
		point_u = Point(head.x, head.y - 20)
		point_d = Point(head.x, head.y + 20)

		# current direction
		dir_l = game.direction == Direction.LEFT
		dir_r = game.direction == Direction.RIGHT
		dir_u = game.direction == Direction.UP
		dir_d = game.direction == Direction.DOWN

		# create the state
		state = [
			# Danger straight
			(dir_l and game.is_collision(point_l)) or
			(dir_r and game.is_collision(point_r)) or
			(dir_u and game.is_collision(point_u)) or
			(dir_d and game.is_collision(point_d)),

			# Danger right
			(dir_u and game.is_collision(point_r)) or
			(dir_d and game.is_collision(point_l)) or
			(dir_l and game.is_collision(point_u)) or
			(dir_r and game.is_collision(point_d)),

			# Danger left
			(dir_d and game.is_collision(point_r)) or
			(dir_u and game.is_collision(point_l)) or
			(dir_r and game.is_collision(point_u)) or
			(dir_l and game.is_collision(point_d)),

			# move direction
			dir_l,
			dir_r,
			dir_u,
			dir_d,

			# food location
			game.food.x < game.head.x, # food left
			game.food.x > game.head.x, # food right
			game.food.y < game.head.y, # food up
			game.food.y > game.head.y  # food down
		]
		

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) # popleft if max_mem is reach

	def train_long_memory(self):
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE)
		else:
			mini_sample = self.memory

		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones)

	def train_short_memory(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)

	def get_action(self, state):
		# random moves: tradeoff exploration /exploitation
		self.epsilon = 80 - self.n_games
		final_move = [0,0,0]
		if random.randint(0,200) < self.epsilon:
			move = random.randint(0,2)
			final_move[move] = 1
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model.predict(state0) # prediction is still a raw value like [5.0, 2.3, 1.8]
			move = torch.argmax(prediction).item() # get index of max value [5.0, 2.3, 1.8] -> 0
			final_move[move] = 1
		
		return final_move

def train():
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	record = 0
	agent = Agent()
	game = SnakeGameAI()
	while True:
		# 1. get the old state
		state_old = agent.get_state(game)

		# 2. get move based on current state
		final_move = agent.get_action(state_old)

		# 3. perform move based on finale_move and get new state
		reward, done, score = game.play_step(final_move)
		state_new = agent.get_state(game)

		# 4. train the short memory of the agent
		agent.train_short_memory(state_old, final_move, reward, state_new, done)

		# 5. store in the memory
		agent.remember(state_old, final_move, reward, state_new, done)

		if done:
			# train the long memory
			game.reset()
			agent.n_games += 1
			agent.train_long_memory()
			# set new record
			if score > record:
				record = score
				# if new record, save the model

			# plot the results
			print('Game: {} \tScore: {}\tRecord: {}'.format(agent.n_games,score,record))
			# TODO: plot



if __name__=="__main__":
	train()