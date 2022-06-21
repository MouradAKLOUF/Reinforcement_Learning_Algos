# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:08:24 2020

@author: mourad
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")   #theme for ploting
import random

StepReward = -1
gridSize = 4
numActions = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]  # the two corners
Actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
States = [[i, j] for i in range(gridSize) for j in range(gridSize)]

Q = np.zeros((gridSize, gridSize,numActions))
deltas = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
counts_sa= np.ones((gridSize, gridSize,numActions)) #for an adaptif learning rate

eps=0.1
gamma = 1
alpha = 0.1
numIterations=100000
n=12
x=1 


for it in tqdm(range(numIterations)):  
    if it % 100 == 0:
      #x += 1e-2
      x =1   
      
    t = 0
    T = np.inf
    
    state = random.choice(States[1:-1])
    # epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - (eps/x) ):
      action_indx = np.argmax(Q[state[0],state[1],:])
      action = Actions[action_indx]
    else:
      action = random.choice(Actions)
      action_indx = Actions.index(action)

    actions = [action_indx]
    states = [state]
    rewards = [0]
    
    while True:
        if t < T and (list(state) not in  terminationStates):
            nextState = np.array(state)+np.array(action)
            reward= StepReward
            # you can not crosse the wall,
            if -1 in list(nextState) or gridSize in list(nextState):
                nextState = list(state)
            
            if list(nextState)  in  terminationStates:
                T = t + 1
                
            states.append(nextState)
            rewards.append(reward)

            # epsilon-soft to ensure all states are visited 
            p = np.random.random()
            if p < (1 - (eps/x) ):
                nextAction_indx = np.argmax(Q[nextState[0],nextState[1],:])
                nextAction = Actions[nextAction_indx]
            else:
                nextAction = random.choice(Actions)
                nextAction_indx = Actions.index(nextAction)
            
            actions.append(nextAction_indx)
                
            # next state becomes current state
            state = nextState
            action = nextAction
            action_indx = nextAction_indx

        # state tau being updated
        tau = t - n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + n + 1, T + 1)):
                G += np.power(gamma, i - tau - 1) * rewards[i]
            if tau +n < T:
                G += np.power(gamma, n) * Q[states[tau+n][0],states[tau+n][1],actions[tau+n]]
                
            Q[states[tau][0],states[tau][1],actions[tau]]  += alpha * (
                        G - Q[states[tau][0],states[tau][1],actions[tau]] )

        if tau == T - 1:
            break

        t += 1
        
        
        
        
        
        
plt.figure(figsize=(16,9))
cells = [list(x)[:500] for x in deltas.values()]
for cell in cells:
    plt.plot(cell)

#### let's just see some cells
plt.figure(figsize=(16,9))
plt.plot(cells[50])
plt.plot(cells[13])
plt.plot(cells[6])
plt.plot(cells[9])
plt.plot(cells[15])
    
finalPolicy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function
finalV = np.zeros((gridSize, gridSize))
for s in States:
  action_indx = np.argmax(Q[s[0],s[1],:])
  finalPolicy[s[0], s[1]] = Actions[action_indx]
  finalV[s[0], s[1]] = np.max(Q[s[0],s[1],:])

print(finalV)
