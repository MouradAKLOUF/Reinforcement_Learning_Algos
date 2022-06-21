# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:09:59 2020

@author: mourad
"""

"""**********************
SARSA
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

Q = np.zeros((gridSize, gridSize,numActions))
deltas = {(i, j, k):list() for i in range(gridSize) for j in range(gridSize) for k in range(numActions)}     # plotting evolution of Q_function
counts_sa= np.ones((gridSize, gridSize,numActions)) #for an adaptif learning rate
epis_changes=[] 

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
      
  state = random.choice(states[1:-1])
  # epsilon-soft to ensure all states are visited
  p = np.random.random()
  if p < (1 - (eps/x) ):
    action_indx = np.argmax(Q[state[0],state[1],:])
    action = actions[action_indx]
  else:
    action = random.choice(actions)
    action_indx = actions.index(action)
    
  t=0
  eps_rewards=[0]
  eps_states=[list(state)]
  eps_actions=[action_indx]
  
  while True:
    if list(state) in terminationStates:
        break
    nextState = np.array(state)+np.array(action)
    reward= StepReward
    # you can not crosse the wall,
    if -1 in list(nextState) or gridSize in list(nextState):
        nextState = list(state)
    # epsilon-soft to ensure all states are visited 
    p = np.random.random()
    if p < (1 - (eps/x) ):
        nextAction_indx = np.argmax(Q[nextState[0],nextState[1],:])
        nextAction = actions[nextAction_indx]
    else:
        nextAction = random.choice(actions)
        nextAction_indx = actions.index(nextAction)
    # we will update Q(s,a) AS we experience the episode
    
    t=t+1
    eps_rewards.append(reward)
    eps_states.append(list(nextState))
    eps_actions.append(nextAction_indx)
    
    tau=t-n
    if tau>=0:
        
        G=0
        for i in range(tau+1, tau+n+1):
            G=G+(gamma**(i-tau-1))* eps_rewards[i]
            
        G=G+(gamma**n)*Q[eps_states[tau+n][0],eps_states[tau+n][1],eps_actions[tau+n]]    
        before = Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]]
        #ALPHA= alpha/counts_sa[state[0],state[1],action_indx]
        ALPHA= alpha
        Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]] += ALPHA*(G - Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]])
        counts_sa[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]] += 0.25
        deltas[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]].append(float(np.abs(before-Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]])))
    
    # next state becomes current state
    state = nextState
    action = nextAction
    action_indx = nextAction_indx
  
  T=t
  while True:
    if tau == T - 1:
        break
    t=t+1
    tau=t-n
    if tau>=0:
        G=0
        for i in range(tau+1, len(eps_rewards)):
             G=G+(gamma**(i-tau-1))* eps_rewards[i]
             
        before = Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]]
        #ALPHA= alpha/counts_sa[state[0],state[1],action_indx]
        ALPHA= alpha
        Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]] +=  ALPHA*(G - Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]])
        counts_sa[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]] += 0.25
        deltas[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]].append(float(np.abs(before-Q[eps_states[tau][0],eps_states[tau][1],eps_actions[tau]])))
    



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
for s in states:
  action_indx = np.argmax(Q[s[0],s[1],:])
  finalPolicy[s[0], s[1]] = actions[action_indx]
  finalV[s[0], s[1]] = np.max(Q[s[0],s[1],:])

print(finalV)