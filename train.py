#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 23:57:52 2018

@author: vikram
"""

# Core imports
import gym
import numpy as np

# Import the agent
from agents.agent import DDPG

# Set hyperparameters here
NUM_EPISODES = 5000          # Number of episodes to train for
MAX_STEPS = 700              # Maximum number of steps to run per episode
ACTOR_LR = 1e-4              # Actor network learning rate
CRITIC_LR = 1e-3             # Critic network learning rate
MU = 0.0                     # Ornstein-uhlenbeck noise parameter
THETA = 0.15                 # Ornstein-uhlenbeck noise parameter
SIGMA = 0.2                  # Ornstein-uhlenbeck noise parameter
BUFFER_SIZE = 1000000        # Max size of the replay buffer
BATCH_SIZE = 128             # Number of samples to pick from replay buffer
GAMMA = 0.99                 # Discount factor
TAU = 0.001                  # Soft update to target network factor

# Create the environment
env = gym.make('BipedalWalker-v2')

# Create the agent
agent = DDPG(env, ACTOR_LR, CRITIC_LR, MU, THETA, SIGMA, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU)

# Reset the environment
S = env.reset()
rewards = []

# Train the DDPG agent in the environment
for episode in range(1, NUM_EPISODES+1):
    state = agent.reset_episode() # Start a new episode
    
    while True:
        env.render()
        
        # Perform the action given by the actor network + noise 
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Perform a learning step on the agent
        agent.step(action, reward, next_state, done)
        
        # Move to next state
        state = next_state
        
        if done:
            rewards.append(agent.score)
            
            print("\rEpisode = {:4d}, steps run = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                    episode, agent.count, agent.score, agent.best_score))
            break

# Save the weights of the model
filepath = 'saved_weights/actor.h5'
agent.actor_target.model.save(filepath)