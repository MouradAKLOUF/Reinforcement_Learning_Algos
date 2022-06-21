# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:58:58 2020

@author: mourad aklouf

"""




"""**********************
TD(0) or one-step Prediction
**********************"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")   #theme for ploting
import random

gamma = 0.1 # discounting factor
alpha = 0.1 # (0,1]  stepSize

StepReward = -1
gridSize = 4
numActions = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]  # the two corners
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

numIterations = 10000

# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

for it in tqdm(range(numIterations)):
    state = random.choice(states[1:-1])
    while True:
        action = random.choice(actions)
        if list(state) in terminationStates:
            break
        nextState = np.array(state)+np.array(action)
        reward= StepReward
        # you can not crosse the wall, 
        if -1 in list(nextState) or gridSize in list(nextState):
            nextState = list(state)   
        # update Value function
        before =  V[state[0], state[1]]
        V[state[0], state[1]] += alpha*(reward + gamma*V[nextState[0], nextState[1]] - V[state[0], state[1]])
        # compute Value function evolution
        deltas[state[0], state[1]].append(float(np.abs(before-V[state[0], state[1]])))
        
        state = nextState
        
plt.figure(figsize=(16,9))
cells = [list(x)[:100] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)
 