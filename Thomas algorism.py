# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:09:48 2018

@author: 한승표
날짜: 2018.09.03
과목: 수치해석학
수치해석학 과목의 목표는 ELS 상품의 PRICING 및 델타, 감마 헷지
목표를 달성하기 위한 과정으로 알고리즘을 익혀야 한다.
그 중 하나인 system of linear equations을 풀 수 있도록 접근. \
matrix A를 U(upper)와 L(lower)의 곱으로 나타내어,
Ax = U*L*x = U*y = d  (L*x = y)
이 중에서 특별한 matrix인 tridiagonal matrix를 통해 Thomas algorithm 을  만들고자 한다.
수식은 lecture note2참고 (p13)
a'_n = (a_n - b_n * (c_n-1/a'_n-1))
d'_n = (d_n - b_n * (d'_n-1/a'_n-1))

Convergence 확인 하는 방법
|a_ii| > sum|(a_iJ)| for all i 이면 strictly diagonally dominant한다.
따라서 tridiagonal matrix가 strictly diagonally dominant하다면 이는 Thomas algorism을 이용하여 항상 풀 수 있다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import scipy as si

alpha =  0.5 ;
matrix =np.matrix(np.eye(10)*(1+alpha) + np.eye(10, k = -1) *(-alpha/2) + np.eye(10, k = 1) *(-alpha/2))
d= np.zeros(10)
d[0] , d[-1]=0, 1
d = d*(alpha/2)
np.linalg.solve(matrix, d)

def thomas_al(alpha, size,d1,dn):
    matrix =np.matrix(np.eye(size)*(1+alpha) + np.eye(size, k = -1) *(-alpha/2) + np.eye(size, k = 1) *(-alpha/2))
    d= np.zeros(size)
    d[0] , d[-1]= d1, dn
    d = d*(alpha/2)
    sol = np.linalg.solve(matrix, d).round(3)
    return sol

sol = thomas_al(0.1,10,0,1)


for i in np.arange(0.5,5,0.1):
    x = thomas_al(i,10,0,1).round(2)
    print(x)
print('-'*50)
print('-'*50)
'''-> 해석: alpha값이 증가할수록, x_ndl 증가하며 0이 아닌 값을 같는 x가 많아진다.
왜그럴까?
'''
for i in np.arange(0.5,5,0.1):
    y = thomas_al(i,10,0,5).round(2)
    print(y)
print('-'*50)
print('-'*50)

#해석
