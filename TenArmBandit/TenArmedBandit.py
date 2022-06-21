# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:48:37 2020

@author: mourad
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_bandits = 10
mean_rewards = np.random.normal(size=num_bandits)
sample_size = 100

possible_rewards = []
x_axe = []

for a in range(num_bandits):
    vect = []
    for i in range(sample_size):
        vect.append(np.random.normal(loc=mean_rewards[a]))
    possible_rewards= np.concatenate((possible_rewards, vect), axis=0)
    x_axe= np.concatenate((x_axe, a*np.ones(sample_size)), axis=0)
                             
plt.figure(figsize=(15,8))
ax = sns.violinplot(x=x_axe+1, y=possible_rewards)

""" %%%%%%%%%%%%%%% """

num_bandits = 10
mean_rewards = np.random.normal(num_bandits)
num_iterations = 100
num_experiments = 200
reward_averages_eps = []

for eps in [0.1, 0.01, 0]:
    reward_histories = np.zeros(num_iterations)  
    
    for exp_number in range(num_experiments):
        mean_rewards = np.random.normal(size=num_bandits)
        #####################
        N = np.zeros(num_bandits, dtype=np.int32)
        reward_estimates_q = np.zeros(num_bandits)
        history = np.zeros(num_iterations)
        for itrn in range(num_iterations):
            if np.random.uniform() > eps:
                chosen_bandit = np.argmax(reward_estimates_q) #greedy
            else:
                chosen_bandit = np.random.randint(num_bandits) #exploring
                
            reward = np.random.normal(loc=mean_rewards[chosen_bandit], scale=1)      
            N[chosen_bandit] +=1
            reward_estimates_q[chosen_bandit] = reward_estimates_q[chosen_bandit]+ (1/N[chosen_bandit]) * (reward - reward_estimates_q[chosen_bandit]) 
            history[itrn] = reward
        #####################
        reward_histories = np.add(history,reward_histories )
        
    reward_averages_eps.append(reward_histories/num_experiments)

plt.figure(figsize=(16,9))
for i, eps in enumerate([0.1, 0.01, 0]):
    plt.plot(reward_averages_eps[i], label=str(eps))
plt.legend()

""" %%%%%%%%%%%%%%% """


