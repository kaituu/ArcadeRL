**Asteroid Shooter with Reinforcement Learning**

Welcome to **ArcadeRL**, an arcade-style asteroid shooting game powered by reinforcement learning! This project combines the classic mechanics of arcade space shooters with a reinforcement learning (RL) agent that learns to navigate and destroy asteroids. The AI agent is trained using Deep Q-Learning.

---

## **Features**

- 🛸 Control a spaceship that can move left, right, and fire bullets.
- ☄️ Asteroids spawn randomly and descend; destroy them to score points.
- 🧠 The RL agent learns to play the game through Q-learning.
- 📈 Displays score and tracks high scores across games.
- 🎮 Manual and RL-controlled modes available.

---

## **Gameplay**

In **ArcadeRL**, the goal is to shoot down as many asteroids as possible. The player (or the RL agent) can:

- Move left or right
- Fire bullets to destroy asteroids
- Avoid being overwhelmed by the falling asteroids

### **Controls**

- **Left Arrow**: Move left
- **Right Arrow**: Move right
- **Spacebar**: Fire bullets

---

## **Reinforcement Learning**

The core of the project is the agent, trained using Deep Q-Learning. Here’s how it works:

- **State**: The agent’s state is represented by the position of the spaceship, the asteroid’s position, and the current action (move left, right, fire, or idle).
- **Action**: The agent chooses one of four possible actions: move left, move right, fire a bullet, or stay idle.
- **Reward**: The agent is rewarded for hitting asteroids and penalized when missing or letting an asteroid fall too far.
- **Q-Learning**: The agent uses a neural network to approximate Q-values, learning from past actions and outcomes to improve its performance over time.
