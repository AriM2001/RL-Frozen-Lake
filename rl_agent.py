"""
rl_agent.py

Group: Jose Fuentes, Ari Montes, Daniel Peralta, Perry Wang

This module contains the RLAgent class, which implements table-based reinforcement learning
using both Q-learning and SARSA algorithms for the FrozenLake environment.
"""

import numpy as np
import random

class RLAgent:
    # self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01
    def __init__(self, state_space_size, action_space_size, learning_rate=0.5, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the RLAgent with the given hyperparameters.
        
        Args:
            state_space_size (int): Number of states in the environment.
            action_space_size (int): Number of actions available per state.
            learning_rate (float): Learning rate (alpha) for Q-value updates.
            discount_factor (float): Discount factor (gamma) for future rewards.
            exploration_rate (float): Initial epsilon for exploration in the epsilon-greedy policy.
            exploration_decay (float): Decay factor for the exploration rate per episode.
            min_exploration_rate (float): Minimum exploration rate (epsilon).
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        
        # Initialize Q-table as a 2D NumPy array filled with zeros.
        self.Q = np.zeros((state_space_size, action_space_size))
    
    def choose_action(self, state, method='q_learning'):
        """
        Choose an action using an epsilon-greedy policy.
        
        Args:
            state (int): The current state.
            method (str): The learning method being used ('q_learning' or 'sarsa'). 
                          This parameter is reserved for potential method-specific tweaks.
        
        Returns:
            int: The chosen action.
        """
        #print(self.Q)
        '''

        '''
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action.
            # '-1' because it will get an error for out of bounds
            # Exploration
            return random.randint(0, self.action_space_size - 1)
        else:
            # print(self.Q[state])
            # Choose the best action according to the Q-table.
            # Exploitation
            return int(np.argmax(self.Q[state]))   
    
    def learn_q_learning(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning algorithm.
        
        Q(s, a) ← Q(s, a) + α * (reward + γ * max_a' Q(next_state, a') - Q(s, a))
        
        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state after taking the action.
            done (bool): True if the episode has ended.
        """
        current_qvalue = self.Q[state][action]
        future_estimate = np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (reward + self.gamma * future_estimate - current_qvalue)

        #raise NotImplementedError("This method is not implemented yet.")
    
    def learn_sarsa(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table using the SARSA algorithm.
        
        Q(s, a) ← Q(s, a) + α * (reward + γ * Q(next_state, next_action) - Q(s, a))
        
        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state after taking the action.
            next_action (int): The next action chosen by the agent.
            done (bool): True if the episode has ended.
        """
        current_qvalue = self.Q[state][action]
        future_estimate = self.Q[next_state][next_action]
        self.Q[state, action] += self.lr * (reward + self.gamma * future_estimate - current_qvalue)

    def update_exploration_rate(self):
        """
        Decay the exploration rate (epsilon) after each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.exploration_decay)