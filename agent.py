import torch
import random
import numpy as np
from collections import deque
from game import ArcadeGameRL, Action, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:


    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness factor (exploration vs exploitation)
        self.gamma = 0.9  # discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # replay memory for experience replay
        self.model = Linear_QNet(7, 128, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Get the current state of the game environment
    def get_state(self, game):

        head = game.head  # Position of the spaceship
        asteroid = game.asteroid  # Position of the asteroid

        # Boolean checks for the current action
        action_l = game.action == Action.LEFT
        action_r = game.action == Action.RIGHT
        action_f = game.action == Action.FIRE
        action_i = game.action == Action.IDLE

        # The state is a combination of current action and relative position of the asteroid to the spaceship
        state = [
            # Current action type: left, right, fire, idle
            action_l,
            action_r,
            action_f,
            action_i,

            # Relative asteroid location: is it to the left, right, or directly ahead of the spaceship
            asteroid.x < head.x,
            asteroid.x > head.x,
            asteroid.x == head.x
        ]

        return np.array(state, dtype=int)

    # Store the experience (state, action, reward, next state, done) into memory for replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Train the model using a batch of experiences from the memory replay buffer
    def train_long_memory(self):
        # If the memory contains more experiences than the batch size, sample a random batch for training
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Train the model using a single step of experience from the current game
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Choose an action based on the current state
    def get_action(self, state):
        self.epsilon = 400 - self.n_games  # Decrease epsilon as games increase (less exploration over time)

        final_move = [0, 0, 0, 0]

        # Random action (exploration) with probability proportional to epsilon
        if random.randint(0, 600) < self.epsilon:
            move = random.randint(0, 3)  # Choose a random action (left, right, fire, idle)
            final_move[move] = 1
        else:
            # Model-based action (exploitation): use the neural network to predict the best action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Get the model's prediction (Q-values) for each action
            move = torch.argmax(prediction).item()  # Choose the action with the highest Q-value
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_last_20_scores = []
    plot_mean_last_20_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = ArcadeGameRL()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_last_20_scores.append(score)

            # If more than 20 games, remove the oldest score
            if len(plot_last_20_scores) > 20:
                plot_last_20_scores.pop(0)

            # Calculate mean of the last 20 scores
            if len(plot_last_20_scores) > 0:  # Ensure list is not empty
                plot_mean_last_20 = sum(plot_last_20_scores) / len(plot_last_20_scores)
            else:
                plot_mean_last_20 = 0  # Handle case when there are no games yet

            # Save the mean of the last 20 scores to the list
            plot_mean_last_20_scores.append(plot_mean_last_20)
            plot(plot_scores, plot_mean_scores, plot_mean_last_20_scores)


if __name__ == '__main__':
    train()
