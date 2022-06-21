# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:07:09 2020

@author: mourad
"""


'''*******************************
First-visit Monte Carlo prediction w/o exploring-Starts 
*******************************'''


import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


gamma = 0.8 # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numActions = 4
numIterations = 10000


# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

def generateEpisode():
    initState = random.choice(states[1:-1])  # any cells but the two termination states
    episode = []
    episode.append([list(initState), 0]) # at t=0 R=0 
    
    while list(initState) not in terminationStates:
        action = random.choice(actions)  #here the policy is random
        finalState = np.array(initState)+np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState):
            finalState = initState
        episode.append([list(finalState), rewardSize])
        initState = finalState
    return episode
    
def play_game():
    episode= generateEpisode()
    # calculate the returns by working backwards from the terminal state
    G = 0
    states_and_returns = []
    last = True
    for s, r in episode[::-1]:
        if last:      #ignore the last one, because it G=0 for the last one by def in sutton&barto
            last = False
        else:
            states_and_returns.append((s, G))
        G = r + gamma*G
    states_and_returns= states_and_returns[::-1]  # we want it to be in order of state visited
    return states_and_returns

for it in tqdm(range(numIterations)):
  states_and_returns = play_game()
  seen_states = []
  for s, G in states_and_returns:
    if s not in seen_states:
      returns[s[0],s[1]].append(G)
      newValue= np.mean(returns[s[0],s[1]])
      deltas[s[0], s[1]].append(np.abs(V[s[0],s[1]]-newValue))
      V[s[0],s[1]] = newValue
      seen_states.append(s)
       
# try gamma = 0.6 and gamma = 1
plt.figure(figsize=(16,9))
all_series = [list(x)[:50] for x in deltas.values()]
for series in all_series:
    plt.plot(series)
    
