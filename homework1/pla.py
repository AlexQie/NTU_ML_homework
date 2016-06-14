'''
Created by Qie Chunguang on May 5th, 2016
---------------- Run by Python 3.5  ----------------
-----------------  PLA Algorithm  ------------------
'''

import numpy as np
from random import shuffle, randrange, seed
import time

def read_data(file_name):
    arr = np.loadtxt(file_name)
    ones = np.ones(np.size(arr, axis=0)).T
    ones = np.reshape(ones, (arr.shape[0], 1))
    x = np.append(arr[:, 0:4], ones, axis=1)
    return x, arr[:, 4]

def sign(i):
    if i <= 0: return -1
    else: return 1

def update(x, y, w):
    global count
    if sign(np.sum(x * w)) == y: return w, 0
    else:
        count += 1
        return w + y * x, 1

def test(x, y, w):
    for i in range(np.size(x, axis=0)):
        #print(sign(np.sum(x[i] * w)), y[i])
        assert(sign(np.sum(x[i] * w)) == y[i])

def PLA(x, y):
    w = np.zeros(x.shape[1])
    while True:
        old_w = w
        seq = list(range(x.shape[0]))
        shuffle(seq)
        for i in seq:
            w, updated = update(x[i, :], y[i], w)
        if np.array_equal(w, old_w):
            break
    test(x[0:x.shape[0], :], y[0:x.shape[0]], w)
    return w

def PLA_p(x, y, times):
    '''PLA only update times of iterations'''
    w = np.zeros(x.shape[1])
    t = 0
    while True:
        seq = list(range(x.shape[0]))
        shuffle(seq)
        for i in seq:
            w, updated = update(x[i, :], y[i], w)
            t += updated
            if t >= times:
                return w

def count_error(x, y, w):
    '''count error for w on x and y,
    return times of errors'''
    return find_mistakes(x, y, w).shape[0]

def find_mistakes(x, y, w):
    target = np.sum(x * w, axis=1)
    for i in range(target.shape[0]):
        target[i] = sign(target[i])

    return np.where((target != y) == True)[0]

def pocket_PLA(x, y, times):
    '''pocket PLA'''
    seed(time.time())
    cur_time = 0

    w = np.zeros(x.shape[1])
    min_error = count_error(x, y, w)
    opt_w = w

    while cur_time < times:
        mistakes = find_mistakes(x, y, w)
        idx = randrange(0, mistakes.shape[0])
        i = mistakes[idx]
        w, updated = update(x[i, :], y[i], w)
        if updated is 1:
            cur_time += updated
            new_errors = count_error(x, y, w)
            if new_errors < min_error:
                min_error = new_errors
                #print(new_errors)
                opt_w = w
            #print(min_error)

    return opt_w

if __name__ == '__main__':
    global count
    x, y = read_data('./data1.dat')
    count = 0
    w = PLA(x, y)
    print(count_error(x, y, w))

    '''pocket PLA'''
    x2_train, y2_train = read_data('./data2_train.dat')
    x2_test, y2_test = read_data('./data2_test.dat')

    w3 = pocket_PLA(x2_train, y2_train, 50)
    print(w3)
    errors = count_error(x2_test, y2_test, w3)
    print(errors / x2_test.shape[0])

    all_errors = 0
    for _ in range(2000):
        w3 = pocket_PLA(x2_train, y2_train, 50)
        errors = count_error(x2_test, y2_test, w3)
        all_errors += errors
    print('After 2000 iterations: {0}'.format(all_errors / x2_test.shape[0] / 2000))

    all_errors = 0
    for _ in range(2000):
        w3 = pocket_PLA(x2_train, y2_train, 100)
        errors = count_error(x2_test, y2_test, w3)
        all_errors += errors
    print('After 2000 iterations: {0}'.format(all_errors / x2_test.shape[0] / 2000))

    all_errors = 0
    for _ in range(2000):
        w3 = PLA_p(x2_train, y2_train, 50)
        errors = count_error(x2_test, y2_test, w3)
        all_errors += errors
    print('After 2000 iterations: {0}'.format(all_errors / x2_test.shape[0] / 2000))