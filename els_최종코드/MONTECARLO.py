# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:11:17 2018

@author: 한승표
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

n_trials = 1000
n_steps = 300
T= 6
r = 0.022
q1 = 0
q2 = 0
v1 =  0.127
v2 = 0.238
rho = 0.164
dt = 0.5 /n_steps;
Sum = 0
K0 = [237.15,3088.18]
F = 10000
c = [0.90, 0.90, 0.85, 0.85, 0.80,0.8]
ret = [0.035, 0.07, 0.105, 0.14, 0.175, 0.21]
KI = [0.6,0.6]

#
#tion between prices of the two assets
#r = 0.02; # Interest rate
#K0 = [237.15, 3088.18]; # Reference price of each asset
#F = 10000; # Face value (이론가액 9057.7원)
#T = 3; # Maturation of ctract
#c = [0.035, 0.07, 0.105, 0.14, 0.175, 0.21]; # Rate of return on each early redemption date
#K = [0.9, 0.9, 0.85, 0.85, 0.80, 0.80]; # Exercise price on each early redemption date
#KI = 0.6; # Knock-In barrier level
#Nx = 100;
#Ny=100;
#p = 0.061


#%%
def monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI):
    #process 설정
    randn_matrix_1 = np.random.normal(size=(n_trials, n_steps*T)) #epsilon1
    randn_matrix_2 = np.random.normal(size=(n_trials, n_steps*T)) #epsilon2
    randn_matrix_S1 = randn_matrix_1
    randn_matrix_S2 =  rho * randn_matrix_S1 + np.sqrt(1 - rho ** 2) * randn_matrix_2
    s1_matrix = np.zeros((n_trials, n_steps*T))
    s1_matrix[:,0] = K0[0]
    s2_matrix = np.zeros((n_trials, n_steps*T))
    s2_matrix[:,0] = K0[1] 
    for j in range((n_steps*T)-1):
        s1_matrix[:,j+1] = s1_matrix[:,j] * np.exp((r-q1-0.5*v1**2)*dt + v1*np.sqrt(dt)*randn_matrix_S1[:,j])
        s2_matrix[:,j+1] = s2_matrix[:,j] * np.exp((r-q2-0.5*v2**2)*dt + v2*np.sqrt(dt)*randn_matrix_S2[:,j])
    
    #조기상환 조건
    exp1 = np.ones((n_trials,T)) * K0[0]
    exp2 = np.ones((n_trials,T)) * K0[1]
    exp_c = c
    sit = np.zeros(7)
    f= np.zeros(n_trials)
    #Payoff 구조
    for i in range(len(exp_c)):
        exp1[:,i] = exp1[:,i] * exp_c[i]
        exp2[:,i] = exp2[:,i] * exp_c[i]
        
    for i in range(n_trials):
        #0.5년 때 조기상환
        if s1_matrix[i,n_steps-1] >= exp1[i,0] and s2_matrix[i,n_steps-1] >= exp2[i,0] :
            f[i,] = F * (1+ret[0]) * np.exp(-r*0.5)
            sit[0] = sit[0] +1

        #1년 때 조기상환        
        elif s1_matrix[i,2*n_steps-1] >= exp1[i,1] and s2_matrix[i,2*n_steps-1] >= exp2[i,1] :
            f[i,] = F * (1+ret[1]) * np.exp(-r*1)
            sit[1] = sit[1] +1

        #1.5년 때 조기상환    
        elif s1_matrix[i,3*n_steps-1] >= exp1[i,2] and s2_matrix[i,3*n_steps-1] >= exp2[i,2] :
            f[i,] = F * (1+ret[2]) * np.exp(-r*1.5)
            sit[2] = sit[2] +1

        #2년 때 조기상환
        elif s1_matrix[i,4*n_steps-1] >= exp1[i,3] and s2_matrix[i,4*n_steps-1] >= exp2[i,3] :
            f[i,] = F * (1+ret[3]) * np.exp(-r*2)
            sit[3] = sit[3] +1

        #2.5년 때 조기상환
        elif s1_matrix[i,5*n_steps-1] >= exp1[i,4] and s2_matrix[i,5*n_steps-1] >= exp2[i,4] :
            f[i,] = F * (1+ret[4]) * np.exp(-r*2.5)
            sit[4] = sit[4] +1

        #만기 때 Payoff
        elif  s1_matrix[i,-1] >= exp1[i,5] and s2_matrix[i,-1] >= exp2[i,5]:
                f[i,] = F * (1+ret[5]) * np.exp(-r*3)
                sit[5] = sit[5] +1

        elif  s1_matrix[i,-1] <exp1[i,5] or s2_matrix[i,-1] < exp2[i,5]:
            if (len([x for x in s1_matrix.flatten() < K0[0]*KI[0]])==0) and (len([y for y in s2_matrix.flatten() < K0[1]*KI[1]])==0):
                f[i,] = F * (1+ret[5]) * np.exp(-r*3)
                sit[5] = sit[5] +1
                
            else:
                f[i,] = F * np.minimum(s2_matrix[i,-1]/s2_matrix[i,0],s1_matrix[i,-1]/s1_matrix[i,0]) * np.exp(-r*3)
                sit[6] = sit[6] + 1
             
    Sum = f.sum()
    probability = np.ones(len(sit))
    for i in range(7):
        probability[i] = sit[i]/ n_trials
    values = Sum/n_trials
    err = np.std(f)/np.sqrt(n_trials)
    return values, err ,probability
#%%
values, err ,probability = monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)
print('price is %4f' %(monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)[0]).round(3))
print('err is %4f' %(monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)[1]).round(3))
print('probability is ',(monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)[2]).round(3))

#%%
#pro_KI = 0
#price = 0
#simul  = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
#for i in range(10):
#    n_trials = simul[i]
#    price = price + monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)[0]
#    pro_KI =pro_KI+ monte_els(K0,v1,v2,r,q1,q2,rho,dt,T,n_trials,n_steps,c,ret,KI)[2][-1]
#
#pro_KI_avg = pro_KI/10
#price = price/ 10
#print('pro_KI_avg is ',pro_KI_avg.round(3))
#print('price_KI_avg is ',price)
