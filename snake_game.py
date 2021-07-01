import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
	RIGHT = 1
	LEFT = 2
	UP = 3
	DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 60

class SnakeGameAI:

	def __init__(self, w=640, h=480, game_n=1):
		self.w = w
		self.h = h
		# init display
		self.display = pygame.display.set_mode((self.w, self.h))
		pygame.display.set_caption('Snake')
		self.clock = pygame.time.Clock()
		self.game_n = game_n

		self.reset()


	def reset(self):
		'''Reset the game (snake, food and score)'''
		self.direction = Direction.RIGHT

		self.head = Point(self.w/2, self.h/2)
		self.snake = [self.head,
					  Point(self.head.x-BLOCK_SIZE, self.head.y),
					  Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

		self.score = 0
		self.game_n += 1
		self.food = None
		self._place_food()
		self.frame_iteration = 0

	def _place_food(self):
		'''Place food inside the screen randomly. If the food lands already in the snake it replace it'''
		x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
		y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
		self.food = Point(x, y)
		if self.food in self.snake:
			self._place_food()

	def snake_vision(self):
		''' return an array with -1 where there is a border or the snake, 1 where there is food 0 aywhere'''
		# vision_corners
		left_upper_corner = Point(self.snake[0].x - 2*BLOCK_SIZE, self.snake[0].y - 2*BLOCK_SIZE)

		vision_grid = [0 for _ in range(25)]
		for row in range (5):
			point_x = left_upper_corner.x+row*BLOCK_SIZE
			for col in range(5):
				point_y = left_upper_corner.y+col*BLOCK_SIZE
				point = Point(point_x,point_y)
				index = row*5+col
				# if I collide with myself or a wall
				if self.is_collision(point):
					vision_grid[index] = -1
				# if there is an apple I keep it in mind
				if self.food == self.head:
					vision_grid[index] = 1
		return vision_grid

	def play_step(self,action):
		'''Play a step based on a action, '''
		self.frame_iteration += 1
		# 1. collect user input
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		
		# 2. move
		self._move(action) # update the head
		self.snake.insert(0, self.head)
		
		# 3. check if game over and calculate the reward
		reward = 0
		game_over = False
		# if the snake do nothing then we stop him ()
		if self.is_collision() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			reward -= 10 # the snake has died
			return reward, game_over, self.score

		# 4. place new food or just move
		if self.head == self.food:
			self.score += 1
			reward += 10 # snake eat food
			self._place_food()
		else:
			self.snake.pop()
		
		# 5. update ui and clock
		self._update_ui()
		self.clock.tick(SPEED)
		# 6. return game over and score
		return reward, game_over, self.score

	def is_collision(self, point:Point=None)->bool:
		if point is None:
			point = self.head
		# hits boundary
		if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
			return True
		# hits itself
		if point in self.snake[1:]:
			return True

		return False

	def _update_ui(self):
		self.display.fill(BLACK)

		# draw snake
		for pt in self.snake:
			pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
			pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
		
		# snake vision
		pygame.draw.rect(self.display, WHITE, pygame.Rect(self.snake[0].x - 2*BLOCK_SIZE, self.snake[0].y - 2*BLOCK_SIZE, 5*BLOCK_SIZE, 5*BLOCK_SIZE), 3)

		# draw food
		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

		# draw score
		text = font.render("Score: {} Game: {}".format(str(self.score),str(self.game_n)), True, WHITE)
		
		# update screen
		self.display.blit(text, [0, 0])
		pygame.display.flip()

	def _move(self, action):
		''' |1,0,0| -> straight
			|0,1,0| -> right turn
			|0,0,1| -> left turn
		'''
		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clock_wise.index(self.direction)

		if np.array_equal(action,[1,0,0]):
			# no change
			new_dir = clock_wise[idx] 
		elif np.array_equal(action,[0,1,0]):
			# right turn right -> down -> left -> up
			next_idx = (idx + 1) % 4
			new_dir = clock_wise[next_idx]
		elif np.array_equal(action,[0,0,1]):
			 # left turn right -> up -> left -> down
			next_idx = (idx - 1) % 4
			new_dir = clock_wise[next_idx]

		self.direction = new_dir

		x = self.head.x
		y = self.head.y
		if self.direction == Direction.RIGHT:
			x += BLOCK_SIZE
		elif self.direction == Direction.LEFT:
			x -= BLOCK_SIZE
		elif self.direction == Direction.DOWN:
			y += BLOCK_SIZE
		elif self.direction == Direction.UP:
			y -= BLOCK_SIZE

		self.head = Point(x, y)
