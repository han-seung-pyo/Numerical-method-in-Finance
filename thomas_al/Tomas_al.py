# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:02:29 2018

@author: 한승표
"""
import numpy as np
import pandas as pd

#Thomas algorism

#matrix = np.matrix(np.diag(diagonal)+np.diag(l_diagonal,k=-1)+np.diag(u_diagonal,k=1))
alpha = 0.5
diagonal = np.ones(10) * (1+alpha)
l_diagonal = np.ones(10) * (-alpha/2)
u_diagonal = np.ones(10) * (-alpha/2)
d = np.array([0,0,0,0,0,0,0,0,0,1]) * (alpha/2)


def cal(diagonal,l_diagonal,u_diagonal,d):
    a1 =[diagonal[0]]
    d1 =[d[0]]
    x = []
    for i in range(len(diagonal)-1):
        a_prime = diagonal[i+1] - (l_diagonal[i+1] * u_diagonal[i]/a1[i])
        d_prime = d[i+1] - (l_diagonal[i+1] * d1[i]/a1[i])
        a1.append(a_prime)
        d1.append(d_prime)
    x.append(d1[-1]/a1[-1])
    for i in range(9,0,-1):
        x.append((-u_diagonal[i]/a1[i])*x[9-i])
    x.reverse()
    x = [x]
    x= pd.DataFrame(x, columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']).round(3)
    return x
x = cal(diagonal, l_diagonal, u_diagonal, d)


solution = pd.DataFrame()
for alpha in np.arange(0.5,5.1,0.1):
    diagonal = np.ones(10) * (1+alpha)
    l_diagonal = np.ones(10) * (-alpha/2)
    u_diagonal = np.ones(10) * (-alpha/2)
    d = np.array([0,0,0,0,0,0,0,0,0,1]) * (alpha/2)
    solution = solution.append(cal(diagonal, l_diagonal, u_diagonal,d))
solution

#%%
def TDMAsolver(l_diagonal,diagonal,u_diagonal,d):
    a1 =[diagonal[0]]
    d1 =[d[0]]
    x = []
    n = len(d)
    #prime 인덱스는 2부터/
    for i in range(n-1):
        a1.append(diagonal[i+1] - (l_diagonal[i] * u_diagonal[i]/a1[i]))
        d1.append(d[i+1] - (l_diagonal[i] * d[i]/a1[i]))
    
    x.append(d1[-1]/a1[-1])
    
    for i in range(n-2,0,-1):
        x.append(((d1[i]-u_diagonal[i])/a1[i])*x[n-2-i])
    x.reverse()
    return x

diagonal = np.ones(10) * (1+alpha)
l_diagonal = np.ones(10) * (-alpha/2)
u_diagonal = np.ones(10) * (-alpha/2)
d = np.array([0,0,0,0,0,0,0,0,0,1]) * (alpha/2)

l_diagonal = l_diagonal[1:]
u_diagonal = u_diagonal[:-1]

sol2= TDMAsolver(l_diagonal,diagonal,u_diagonal,d)
#%%
def TDMAsolver2(a, b, c, d):
    a = list(a)
    b = list(b)
    c = list(c)
    d = list(d)
    
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

sol3 = TDMAsolver2(l_diagonal,diagonal,u_diagonal,d)
#%%
def TDMAsolver3(a,b,c,d):
    lower = list(a)
    diag = list(b)
    upper = list(c)
    f = list(d)
    n = len(f)
    sol = np.ones(n)
    for i in range(n-1):
        diag[i+1] = diag[i+1] - upper[i] * lower[i] / diag[i]
        f[i+1] = f[i+1] - f[i]/lower[i]/diag[i]
        
    sol[-1] = f[-1]/diag[-1]
    
    for i in np.arange(n-1,0,-1):
        sol[i-1] = (f[i-1] - upper[i-1] * sol[i])/diag[i-1]
    return sol

sol4 = TDMAsolver3(l_diagonal,diagonal,u_diagonal,d)