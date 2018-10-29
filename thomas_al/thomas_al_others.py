# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:16:47 2018

@author: 한승표
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
from scipy import stats
import pandas as pd

def TDMASolve(a,b,c,f):
    n = len(f)
    v = np.zeros(n)
    y=v
    w = a[0]
    y[0] =f[0]/w
    for i in range(2,n+1): #i = 2,3,4,5,,,,,,n
        v[i-2] = c[i-2]/w
        w = a[i-1] - b[i-1]*v[i-2]
        y[i-1] = (f[i-1] - b[i-1]*y[i-2])/w
    for i in np.arange(n-1,0,-1): #n-1,n-2,,,,,1
        y[i-1] = y[i-1] - v[i-1]*y[i]
    return y

#%%
def plz(lower,diag,upper,f):
    n = len(f)
    sol = np.ones(n)
    for i in range(n-1):
        diag[i+1] = diag[i+1] - upper[i] * lower[i] / diag[i]
        f[i+1] = f[i+1] - f[i]/lower[i]/diag[i]
        
    sol[-1] = f[-1]/diag[-1]
    
    for i in np.arange(n-1,0,-1):
        sol[i-1] = (f[i-1] - upper[i-1] * sol[i])/diag[i-1]
    return sol
def plz2(sub,diag,sup,f):
    n = len(f)
    v = np.zeros(n)
    y = v 
    w = diag[0]
    y[0] = f[0]/w
    for i in range (1,n):
        v[i-1] = sup[i-1]/w
        w = diag[i]-sub[i]*v[i-1]  
        y[i] = (f[i]-sub[i]*y[i-1])/w
    for j in np.arange(n-2,-1,-1):
        y[j] = y[j]-v[j]*y[j+1]
    return y

def plz3(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
#%%
def plz4(diagonal,l_diagonal,u_diagonal,d):
    a1 =[diagonal[0]]
    d1 =[d[0]]
    x = []
    for i in range(len(diagonal)-1):
        a_prime = diagonal[i+1] - (l_diagonal[i+1] * u_diagonal[i]/a1[i])
        d_prime = d[i+1] - (l_diagonal[i+1] * d1[i]/a1[i])
        a1.append(a_prime)
        d1.append(d_prime)
    
    x.append(d1[-1]/a1[-1])
    
    for i in range(len(d)-1,0,-1):
        x.append((-u_diagonal[i]/a1[i])*x[len(d)-1-i])
    x.reverse()
#    x = [x]
#    x= pd.DataFrame(x, columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']).round(3)
    return x    
#%%
#a = lower, b = diag, c = upper
n = 10; a = 5*np.ones(n); b = np.ones(n); c = 7*np.ones(n);
f = [5,6,7,8,9,5,6,7,8,9]
sol = plz(a,b,c,f)
sol2 = plz2(a,b,c,f)
sol3 = plz3(a,b,c,f)
sol4 = plz4(a,b,c,f)

