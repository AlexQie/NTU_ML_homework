#-*- coding: utf8 -*-
#Run by python3.4
#Created by Qie Chunguang on June 14th, 2016
#This codes is to solve homework-2, Question 16-20 of
#Machine Learning Foundations of NTU on Coursera

import numpy as np
from numpy.random import uniform

def generate_x(n):
    '''Generate a n-d vector by a uniform distribution'''
    return uniform(-1, 1, n)


def generate_y(x):
    '''Generate y by f(x)=s(x)+noise where s(x)=sign(x)
    and noise flips the result with 20% probility'''
    y = np.array([np.sign((uniform(0, 1) - 0.2) * i)
                  for i in x])
    return y

def compute_Ein(x, y, theta, s):
    out_y = s * np.sign(x - theta)
    return np.sum(out_y == y)

def dicision_stump(x, y):
    thetas = (np.sort(x) + np.roll(np.sort(x), -1)) / 2
    thetas[-1] = 1
    #print(x, y, thetas)
    min_Ein = x.shape[0]
    theta = None
    sign = None
    for t in thetas:
        for s in [-1, 1]:
            Ein = compute_Ein(x, y, t, s)
            #print(Ein)
            if Ein < min_Ein:
                min_Ein = Ein
                theta = t
                sign = s
    return theta, sign

def q_17():
    '''Question 17 he problem set'''
    i = 5000
    error_sum = 0
    d = 20
    for _ in range(i):
        print('iter', _)
        x = generate_x(d)
        y = generate_y(x)
        theta, s = dicision_stump(x, y)
        error_sum += compute_Ein(x, y, theta, s) / d
    print('Average Ein of {0} iterations is {1}'.format(i, error_sum/i))


def test():
    x = generate_x(10)
    y = generate_y(x)
    theta, sign = dicision_stump(x, y)
    print(x, y)
    print(compute_Ein(x, y, theta, sign))

if __name__ == '__main__':
    q_17()