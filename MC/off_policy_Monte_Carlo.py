# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:42:29 2020

@author: mourad
"""

'''*******************************
off policy Monte Carlo control 
*******************************'''

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


rewardSize = -1
gridSize = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numActions = 4
numIterations = 10000


eps=0.5
gamma = 0.6
numIterations=15000

Q = np.zeros((gridSize, gridSize,numActions))
C = np.zeros((gridSize, gridSize,numActions))
deltas = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
states_actions = [[i, j, k] for i in range(gridSize) for j in range(gridSize)for k in range(numActions)]
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

targetPolicy = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
behaviorPolicy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
behaviorProba = np.zeros((gridSize, gridSize,numActions))
for s in states:
    action = random.choice(actions)  #here the policy is random
    targetPolicy[s[0], s[1]] = actions.index(action)
    
initPolicy=targetPolicy
t=1
epis_changes=[] 

for it in tqdm(range(numIterations)):
  if it % 100 == 0:
      t +=  1e-2
  for s in states:
    action = random.choice(actions)  #here the policy is random
    p = np.random.random()
    if (p < 1 - (eps/t)):
        behaviorPolicy[s[0], s[1]] = targetPolicy[s[0], s[1]]
        behaviorProba[s[0],s[1],: ] = (eps/numActions)
        behaviorProba[s[0],s[1],targetPolicy[s[0], s[1]] ] = 1 - ((eps/t))+ ((eps/t)/numActions)
    else:
        behaviorPolicy[s[0], s[1]] = actions.index(action)
        behaviorProba[s[0],s[1],:] = ((eps/t)/numActions)
  W = 1      
  states_actions_returns = play_game_3(behaviorPolicy,float('Inf'))
  bigestChangeInEpisode = 0
  for s, a, G in states_actions_returns:
      before = Q[s[0],s[1],a]
      C[s[0],s[1],a]  += W
      Q[s[0],s[1],a] = Q[s[0],s[1],a]+ (W / C[s[0],s[1],a])*(G-Q[s[0],s[1],a])
      deltas[s[0], s[1],a].append(float(np.abs(before-Q[s[0],s[1],a])))
      bigestChangeInEpisode = max(bigestChangeInEpisode, np.abs(Q[s[0],s[1],a]))
      #targetPolicy[s[0],s[1]] = np.argmax(Q[s[0],s[1],:])
      if a !=  targetPolicy[s[0], s[1]]:
          break
      W = W * 1/behaviorProba[s[0],s[1],a ]
              
  epis_changes.append(bigestChangeInEpisode) 


 
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
    
#### ploting changes
plt.figure(figsize=(16,9))
plt.plot(epis_changes[:])

finalPolicy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function
finalV = np.zeros((gridSize, gridSize))
for s in states:
  action_indx = np.argmax(Q[s[0],s[1],:])
  finalPolicy[s[0], s[1]] =action_indx         #actions[action_indx]
  finalV[s[0], s[1]] = np.max(Q[s[0],s[1],:])