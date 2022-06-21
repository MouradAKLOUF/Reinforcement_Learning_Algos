# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:41:44 2020

@author: mourad
"""

"""**********************
Policy Iteration 
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

V = np.zeros((gridSize, gridSize))
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function

Policy = Policy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
for s in states:
    action = random.choice(actions)  
    Policy[s[0], s[1]] = actions.index(action)
    V [s[0], s[1]]= np.random.random()


def move(InitState, action):
    if InitState in terminationStates:
        return InitState, 0
    reward = rewardSize
    finalState = np.array(InitState) + np.array(action)
    if -1 in finalState or 4 in finalState: 
        finalState = InitState   
    return finalState, reward

initPolicy=Policy
small_enough = 1e-4   
it=0
while True:
  # policy evaluation
  it+=1
  while True:
    bigestChange = 0
    for s in states:
      before=V[s[0], s[1]]
      a = Policy[s[0], s[1]]
      action=actions[a]
      finalState, reward = move(s, action)
      V[s[0], s[1]] = (1/numActions)*(reward + gamma * V[finalState[0], finalState[1]])
      deltas[s[0], s[1]].append(float(np.abs(before-V[s[0], s[1]])))
      bigestChange = max(bigestChange, float(np.abs(before - V[s[0], s[1]])))
    if bigestChange <= small_enough:
      break
  # policy improvement 
  policyConverged = True
  for s in states:
      old_a = Policy[s[0], s[1]]
      new_a = None
      biggestValue = float('-inf')
      for a in actions:
        finalState, reward = move(s, a)
        v = (1/numActions)* (reward + gamma * V[finalState[0], finalState[1]])
        if v > biggestValue:
          biggestValue = v
          indx= actions.index(a)
          new_a = indx
      Policy[s[0], s[1]] = new_a
      if new_a != old_a:
        policyConverged = False
  if policyConverged:
    break

plt.figure(figsize=(16,9))
cells = [list(x)[:] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)
    
    
    
    
"""**********************
Policy Iteration : other implimentation
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



V = np.zeros((gridSize, gridSize))
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function

Policy = np.zeros((gridSize, gridSize, numActions))
for s in states:
    randomList=[random.random() for i in range(4)] 
    proba= randomList/np.sum(randomList)
    Policy[s[0], s[1], :] = proba
    V [s[0], s[1]]= np.random.random()

def move(InitState, action):
    if InitState in terminationStates:
        return InitState, 0
    reward = rewardSize
    finalState = np.array(InitState) + np.array(action)
    if -1 in finalState or 4 in finalState: 
        finalState = InitState   
    return finalState, reward

initPolicy=Policy
small_enough = 1e-4   
it=0
while True:
  # policy evaluation
  it+=1
  while True:
    bigestChange = 0
    for s in states:
        v=0
        for a in actions:
            action_indx = actions.index(a)
            finalState, reward = move(s, a)
            v += Policy[s[0], s[1],action_indx]*(reward + gamma * V[finalState[0], finalState[1]]) # u can add (1/numActions)*
        deltas[s[0], s[1]].append( float(np.abs(V[s[0], s[1]]-v)) )
        bigestChange = max(bigestChange, float(np.abs(V[s[0], s[1]]-v)))
        V[s[0], s[1]]=v
    if bigestChange <= small_enough:
      break
  # policy improvement 
  policyConverged = True
  for s in states:
      old_a = Policy[s[0], s[1]]
      new_a = None
      biggestValue = float('-inf')
      for a in actions:
        finalState, reward = move(s, a)
        v =  (reward + gamma * V[finalState[0], finalState[1]]) # u can add (1/numActions)*
        if v > biggestValue:
          biggestValue = v
          indx= actions.index(a)
          new_a = indx
      Policy[s[0], s[1],:] = 0
      Policy[s[0], s[1],new_a] = 1
      policyConverged = (Policy[s[0], s[1],:] == old_a).all()
  if policyConverged:
    break

plt.figure(figsize=(16,9))
cells = [list(x)[:] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)