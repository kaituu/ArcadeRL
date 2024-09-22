import math
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('Roboto-Regular.ttf', 25)


class Action(Enum):
    RIGHT = 1
    LEFT = 2
    FIRE = 3
    IDLE = 4


Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# asteroid colors
RED = (178, 34, 34)

# spaceship colors
GRAY1 = (60, 60, 60)
BLUE = (44, 117, 255)

# bullet colors
GOLD = (181, 166, 66)
GRAY2 = (55, 57, 58)

BLOCK_SIZE = 20
SPEED = 4000


class ArcadeGameRL:

    def __init__(self, width=480, height=640):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Arcade')
        self.clock = pygame.time.Clock()
        self.reset()

    # Resets the game state to start a new game
    def reset(self):
        self.action = Action.RIGHT
        self.head = Point(self.width / 2, self.height - BLOCK_SIZE)
        self.score = 0
        self.asteroid = None
        self._place_asteroid()
        self.bullets = []
        self.frame_iteration = 0

    # Main game loop to process each game step
    def play_step(self, action):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move / fire
        self._action(action)
        self._move_bullets()

        # 3. check if game over
        self.reward = 0
        game_over = False
        if not self._move_asteroid():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. score limit : 50
        if self.score > 50:
            game_over = True
            reward = 20
            return reward, game_over, self.score

        # 5. update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return self.reward, game_over, self.score

    # Handles player action based on the neural network output
    def _action(self, action):
        # [right, left, fire, idle]
        new_action = Action.IDLE

        if np.array_equal(action, [1, 0, 0, 0]):
            new_action = Action.RIGHT
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_action = Action.LEFT
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_action = Action.FIRE

        self.action = new_action

        x = self.head.x
        y = self.head.y

        if self.action == Action.RIGHT:
            x += BLOCK_SIZE
            if x >= 480 - BLOCK_SIZE:
                x = 480 - BLOCK_SIZE
                self.action = Action.IDLE
        elif self.action == Action.LEFT:
            x -= BLOCK_SIZE
            if x <= 0:
                x = 0
                self.action = Action.IDLE
        elif self.action == Action.FIRE:
            self._fire_bullet()
            self.action = Action.IDLE

        self.head = Point(x, y)

    # Places the asteroid randomly at the top of the screen
    def _place_asteroid(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = 0
        self.asteroid = Point(x, y)

    # Fires a bullet and calculates the reward based on accuracy
    def _fire_bullet(self):
        x = self.head.x
        y = self.head.y - 1 * BLOCK_SIZE
        if x == self.asteroid.x:
            self.reward = max(1, int(2 * math.sqrt(self.score)))
        else:
            self.reward = -1
        self.bullets.append(Point(x, y))

    # Moves bullets upwards and handles collision detection
    def _move_bullets(self):
        count = 0
        for i in range(len(self.bullets)):
            y = self.bullets[i - count].y - BLOCK_SIZE
            if y < BLOCK_SIZE:
                self.bullets.pop(i - count)
                count += 1
            else:
                self.bullets[i - count] = Point(self.bullets[i - count].x, y)
                if self.is_collision(i - count):
                    self.bullets.pop(i - count)
                    count += 1

    # Asteroid moves downwards; returns False if it goes off screen
    def _move_asteroid(self):
        x = self.asteroid.x
        y = self.asteroid.y

        y += BLOCK_SIZE / 2
        if y > self.height:
            return False
        self.asteroid = Point(x, y)
        return True

    # Checks for collision between bullets and the asteroid
    def is_collision(self, index):
        if self.bullets[index].x == self.asteroid.x and self.bullets[index].y <= self.asteroid.y:
            self.score += 1
            self._place_asteroid()
            return True
        return False

    # Updates the UI for the game (spaceship, bullets, asteroid, score)
    def _update_ui(self):
        self.display.fill(BLACK)

        # display spaceship
        pygame.draw.rect(self.display, GRAY1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE, pygame.Rect(self.head.x + 4, self.head.y + 4, 12, 12))

        # display bullets
        for bullet in self.bullets:
            pygame.draw.rect(self.display, GRAY2, pygame.Rect(bullet.x, bullet.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GOLD, pygame.Rect(bullet.x + 4, bullet.y + 4, 12, 12))

        # display asteroid
        pygame.draw.rect(self.display, RED, pygame.Rect(self.asteroid.x, self.asteroid.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
