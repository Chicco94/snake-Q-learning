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
		# TODO: model, trainer

	def get_state(self, game):
		pass

	def remember(self, state, action, reward, next_state, done):
		pass

	def train_long_memory(self):
		pass

	def train_short_memory(self, state, action, reward, next_state, done):
		pass

	def get_action(self, state):
		pass

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