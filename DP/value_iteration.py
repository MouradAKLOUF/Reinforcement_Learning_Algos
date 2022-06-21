# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:59:47 2020

@author: mourad
"""

"""**********************
Value Iteration 
**********************"""  

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

gamma = 1 # discounting rate
rewardSize = -1
gridSize = 4
numActions = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

def move(InitState, action):
    if InitState in terminationStates:
        return InitState, 0
    reward = rewardSize
    finalState = np.array(InitState) + np.array(action)
    if -1 in finalState or 4 in finalState: 
        finalState = InitState   
    return finalState, reward

deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function 
V = np.zeros((gridSize, gridSize))
for s in states:
    V [s[0], s[1]]= np.random.random()

small_enough = 1e-4  
it=0
 # policy evaluation

while True:
    it+=1
    bigestChange = 0
    for s in states:
      before=V[s[0], s[1]]
      best_value = float('-inf')
      new_a = None
      v=0
      for a in actions:
        finalState, reward = move(s, a)
        v = (1/numActions)* (reward + gamma * V[finalState[0], finalState[1]])
        if v > best_value:
          best_value = v
          indx= actions.index(a)
          new_a = indx   
      deltas[s[0], s[1]].append(float(np.abs(best_value-V[s[0], s[1]])))
      bigestChange = max(bigestChange, float(np.abs(best_value - V[s[0], s[1]])))
      V[s[0], s[1]] = best_value
    if bigestChange <= small_enough:
      break
  # policy improvement 

Policy = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
for s in states:
    new_a = None
    biggestValue = float('-inf')
    for a in actions:
        finalState, reward = move(s, a)
        v = (1/numActions)* (reward + gamma * V[finalState[0], finalState[1]])
        if v > biggestValue:
            biggestValue = v
            Policy[s[0], s[1]] = actions.index(a)


plt.figure(figsize=(16,9))
cells = [list(x)[:] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)