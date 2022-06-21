# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:52:57 2020

@author: mourad
"""

"""**********************
Policy evaluation
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
numIterations = 1000


def move(InitState, action):
    if InitState in terminationStates:
        return InitState, 0
    reward = rewardSize
    finalState = np.array(InitState) + np.array(action)
    if -1 in finalState or 4 in finalState: 
        finalState = InitState   
    return finalState, reward

V = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function

for it in tqdm(range(numIterations)):
    copyV = np.copy(V)
    deltaState = []
    for state in states:
        weightedRewards = 0
        for action in actions:
            finalState, reward = move(state, action)
            weightedRewards += (1/numActions)*(reward+(gamma*V[finalState[0], finalState[1]]))
        deltas[state[0], state[1]].append(float(np.abs(copyV[state[0], state[1]]-weightedRewards)))
        copyV[state[0], state[1]] = weightedRewards
        
    V = copyV 
    if it in [0,1,2,9, 99, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(V)
        print("")
        
plt.figure(figsize=(16,9))
cells = [list(x)[:500] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)
    
########################################################
#   this converge faster, when using the same matrix V 
#########################################################

for it in tqdm(range(numIterations)):
    deltaState = []
    for state in states:
        weightedRewards = 0
        before=V[state[0], state[1]]
        for action in actions:
            finalState, reward = move(state, action)
            weightedRewards += (1/numActions)*(reward+(gamma*V[finalState[0], finalState[1]]))
        deltas[state[0], state[1]].append(float(np.abs(before-weightedRewards)))
        V[state[0], state[1]] = weightedRewards
        
    if it in [0,1,2,9, 99, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(V)
        print("")
        
plt.figure(figsize=(16,9))
cells = [list(x)[:500] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)