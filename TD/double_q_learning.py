# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:30:47 2020

@author: mourad
"""

"""**********************
double Q Learning
**********************"""  
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
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


Q1 = np.zeros((gridSize, gridSize,numActions))
Q2 = np.zeros((gridSize, gridSize,numActions))

deltas = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
counts_sa= np.ones((gridSize, gridSize,numActions))
epis_changes=[] 


eps=0.025
gamma = 1
alpha = 0.1
numIterations=50000
t=1

for it in tqdm(range(numIterations)):
  if it % 100 == 0:
      #t += 1e-2
      t =1
  state = random.choice(states[1:-1])
  
  avrgQ= Q1[state[0],state[1],:]+ Q2[state[0],state[1],:] 
  action_indx = np.argmax(avrgQ)
  action = actions[action_indx]
  
  bigestChangeInEpisode=0
  while True:
    # epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - (eps/t)):
        action = action
        action_indx = action_indx
    else:
        action = random.choice(actions)
        action_indx = actions.index(action)

    if list(state) in terminationStates:
        break
    nextState = np.array(state)+np.array(action)
    reward= StepReward
    # you can not crosse the wall,
    if -1 in list(nextState) or gridSize in list(nextState):
        nextState = list(state)
    
    #you can use the avrg to get next action or just follow sutton&barto algo
    #avrgQ= Q1[state[0],state[1],:]+ Q2[state[0],state[1],:]     
    #nextAction_indx = np.argmax(avrgQ)
    #nextAction = actions[nextAction_indx]
        
    # we will update Q(s,a) AS we experience the episode
    
    #ALPHA= alpha/counts_sa[state[0],state[1],action_indx]
    ALPHA= alpha
    x = np.random.random()
    if x <= 0.5:
        nextAction_indx = np.argmax(Q1[nextState[0],nextState[1],:])
        nextAction = actions[nextAction_indx]
        
        before = Q1[state[0],state[1],action_indx]
        Q1[state[0],state[1],action_indx] = Q1[state[0],state[1],action_indx] + ALPHA*(reward + gamma*Q2[nextState[0],nextState[1],nextAction_indx] - Q1[state[0],state[1],action_indx])
        deltas[state[0], state[1],action_indx].append(float(np.abs(before-Q1[state[0],state[1],action_indx])))
        bigestChangeInEpisode= max(bigestChangeInEpisode, float(np.abs(before-Q1[state[0],state[1],action_indx])) )    
    else:
        nextAction_indx = np.argmax(Q2[nextState[0],nextState[1],:])
        nextAction = actions[nextAction_indx]
        
        before = Q2[state[0],state[1],action_indx]
        Q2[state[0],state[1],action_indx] = Q2[state[0],state[1],action_indx] + ALPHA*(reward + gamma*Q1[nextState[0],nextState[1],nextAction_indx] - Q2[state[0],state[1],action_indx])
        deltas[state[0], state[1],action_indx].append(float(np.abs(before-Q2[state[0],state[1],action_indx])))
        bigestChangeInEpisode= max(bigestChangeInEpisode, float(np.abs(before-Q2[state[0],state[1],action_indx])) )    
    
    counts_sa[state[0],state[1],action_indx] += 0.05
    #changes.append(float(np.abs(before-Q[state[0],state[1],action_indx])))
    
    # next state becomes current state
    state = nextState
    action = nextAction
    action_indx = nextAction_indx

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
plt.plot(epis_changes[:10000])

Q=(Q1+Q2)/2
finalPolicy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}     # plotting evolution of V_function
finalV = np.zeros((gridSize, gridSize))
for s in states:
  action_indx = np.argmax(Q[s[0],s[1],:])
  finalPolicy[s[0], s[1]] = actions[action_indx]
  finalV[s[0], s[1]] = np.max(Q[s[0],s[1],:])  
  