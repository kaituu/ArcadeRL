import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    # Neural network with a single hidden layer
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # first linear layer (input to hidden)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # second linear layer (hidden to output)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Forward pass for the neural network
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # Save the model's parameters (weights) to a file
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    # Trainer class for training the Q-network
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.model = model  # The Q-network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # Training step for updating the Q-network with a single experience
    def train_step(self, state, action, reward, next_state, done):
        # Convert state, next_state, action, reward to tensors for processing
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If the input is a single experience (1D array), convert it to a batch format (2D)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

            # Predicted Q-values
        pred = self.model(state)

        # Clone the predicted Q-values
        target = pred.clone()

        # Update target Q-values for each sample in the batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            # If the game is not over, add the discounted future reward
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the target Q-value for the specific action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # 4: Update the model's weights
        self.optimizer.step()
