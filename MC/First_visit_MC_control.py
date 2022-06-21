# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:28:26 2020

@author: mourad
"""
'''*******************************
First-visit Monte Carlo control with or w/o exploring-Starts 
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
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


Q = np.zeros((gridSize, gridSize,numActions))
deltas = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
returns = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
states_actions = [[i, j, k] for i in range(gridSize) for j in range(gridSize)for k in range(numActions)]

Policy  = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
for s in states:
    action = random.choice(actions)  #here the policy is random
    Policy[s[0], s[1]] = actions.index(action)  

initPolicy=Policy

eps=0.5
gamma = 1
numIterations=10000
t=1

def generateEpisode_2(Policy,s=[],a=[]):
    if s==[] and a==[]:
        initState = random.choice(states[1:-1])  # any cells but the two termination states
        action = random.choice(actions)  #here the policy is random
        action_indx = actions.index(action)
    else:
        initState = s
        action_indx = a
        action = actions[a]  

    episode = []
    episode.append([list(initState), action_indx, 0]) # at t=0 R=0 
    robot_is_stuck= False
    count=0
    while list(initState) not in terminationStates and robot_is_stuck== False and len(episode)<100 :
        finalState = np.array(initState)+np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState):
            finalState = initState
        nextAction_indx = Policy[finalState[0], finalState[1]]
        nextAction = actions[nextAction_indx]
        episode.append([list(finalState), nextAction_indx, rewardSize])
        if initState[0]==finalState[0] and initState[1]==finalState[1] :
           count+=1 
        initState = finalState
        action= nextAction
        if count==3:
           robot_is_stuck= True 
    return episode
    
def play_game_2(Policy,t,s=[],a=[]):
    episode= generateEpisode_2(Policy,s=[],a=[])
    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    last = True
    for s,a,r in episode[::-1]:
        if last:      #ignore the last one, because it G=0 for the last one by def in sutton&barto
            last = False
        else:
            states_actions_returns.append((s, a , G))
        G = r + (gamma/t)*G
    states_actions_returns= states_actions_returns[::-1]  # we want it to be in order of state visited
    return states_actions_returns

exploring_starts=False
t=1
epis_changes=[] 
for it in tqdm(range(numIterations)):
  if it % 100 == 0:
      t +=  1e-2
  if exploring_starts==True:
      init_sa = states_actions[it % 64] 
      init_s=init_sa[0:2]
      init_a=init_sa[2]
  else:
      init_s=[]
      init_a=[]   
      
  states_actions_returns = play_game_2(Policy,t,init_s,init_a)
  bigestChangeInEpisode = 0
  seen_states_actions = []
  for s, a, G in states_actions_returns:
      sa=[s[0],s[1],a]
      #  first-visit MC policy evaluation
      if sa not in seen_states_actions:
          before = Q[s[0],s[1],a]
          returns[s[0],s[1],a].append(G)
          before=Q[s[0],s[1],a]
          Q[s[0],s[1],a] = np.mean(returns[s[0],s[1],a])
          deltas[s[0], s[1],a].append(float(np.abs(before-Q[s[0],s[1],a])))
          bigestChangeInEpisode = max(bigestChangeInEpisode, np.abs(Q[s[0],s[1],a]))
          seen_states_actions.append(sa)
   
  epis_changes.append(bigestChangeInEpisode) 
  # update policy
  for s in states:
      Policy[s[0],s[1]] = np.argmax(Q[s[0],s[1],:])


    
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


