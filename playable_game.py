import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('Roboto-Regular.ttf', 25)

class Action(Enum):
    RIGHT = 1
    LEFT = 2
    IDLE = 3
    FIRE = 4


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
SPEED = 10


class ArcadeGame:

    def __init__(self, width=480, height=640):
        self.width = width
        self.height = height

        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Arcade')
        self.clock = pygame.time.Clock()

        # init game state
        self.action = Action.IDLE

        self.head = Point(self.width / 2, self.height - BLOCK_SIZE)

        self.score = 0
        self.asteroid = None
        self._place_asteroid()

        self.bullets = []

    # Method to handle key inputs and actions (movement, firing)
    def _action(self, action):
        x = self.head.x
        y = self.head.y
        if action == Action.RIGHT:
            x += BLOCK_SIZE
            if x >= 480 - BLOCK_SIZE:
                x = 480 - BLOCK_SIZE
                self.action = Action.IDLE
            self.head = Point(x, y)
        elif action == Action.LEFT:
            x -= BLOCK_SIZE
            if x <= 0:
                x = 0
                self.action = Action.IDLE
            self.head = Point(x, y)
        elif action == Action.FIRE:
            self._fire_bullet()
            self.action = Action.IDLE

    # Places the asteroid in a random position at the top
    def _place_asteroid(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = 0
        self.asteroid = Point(x, y)

    # Fires a bullet from the spaceship
    def _fire_bullet(self):
        x = self.head.x
        y = self.head.y - 1 * BLOCK_SIZE
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
                if self._is_collision(i - count):
                    self.bullets.pop(i - count)
                    count += 1

    # Moves the asteroid down the screen
    def _move_asteroid(self):
        x = self.asteroid.x
        y = self.asteroid.y

        y += BLOCK_SIZE / 2
        if y > self.height:
            return False
        self.asteroid = Point(x, y)
        return True

    # Checks for collision between bullets and the asteroid
    def _is_collision(self, index):
        if self.bullets[index].x == self.asteroid.x and self.bullets[index].y <= self.asteroid.y:
            self.score += 1
            self._place_asteroid()
            return True
        return False

    # Updates the UI (spaceship, bullets, asteroid, score)
    def _update_ui(self):
        self.display.fill(BLACK)

        # display spaceship
        pygame.draw.rect(self.display, GRAY1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE, pygame.Rect(self.head.x + 4, self.head.y + 4, 12, 12))

        # display bullets
        for pt in self.bullets:
            pygame.draw.rect(self.display, GRAY2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GOLD, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # display asteroid
        pygame.draw.rect(self.display, RED, pygame.Rect(self.asteroid.x, self.asteroid.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # Main game loop - collects input, moves objects, updates game state
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.action = Action.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.action = Action.RIGHT
                elif event.key == pygame.K_SPACE:
                    self.action = Action.FIRE

        # 2. move / fire
        self._action(self.action)
        self._move_bullets()

        # 3. check if game over
        game_over = False
        if not self._move_asteroid():
            game_over = True
            return game_over, self.score

        # 5. update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return game_over, self.score


if __name__ == '__main__':

    game = ArcadeGame()

    # game loop
    while True:

        game_over, score = game.play_step()

        if game_over == True:
            break

    print('Score', score)

    pygame.quit()
