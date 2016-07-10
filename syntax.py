import random
import numpy as np
from syntax_class import SyntaxClass
from syntax_test import X
n = 5
R = 2
N = n*R
my_arr = []
other_arr = [[] for i in range(N)]

def set():
    global my_arr
    my_arr = np.random.permutation(np.arange(n))
    print my_arr

def set_else():
    global other_arr, xarr
    xarr = my_arr
    list_index = np.repeat(np.arange(n), R)
    for i in range(N):
        other_arr[i] = my_arr[list_index[i]]

def modify():
    global other_arr, my_arr, xarr
    my_arr += 1
    for i in range(N):
        other_arr[i] = other_arr[i] + 2

def modify2():
    global other_arr, my_arr, xarr
    my_arr = np.ones(n)
    other_arr = my_arr
    for i in range(n):
        other_arr[i] = other_arr[i] + 2
    print my_arr, other_arr
    my_arr = np.ones(n)
    other_arr = my_arr
    other_arr += 2 # of interest
    print my_arr, other_arr

def reveal():
    set()
    set_else()
    modify()
    print my_arr, other_arr, xarr
    modify2()

def foo():
    maxa = 0
    for i in range(3):
        b = random.randint(0, 5)
        b = 0
        if b > maxa:
            maxa = b
            a = b+1
    print maxa, a
    print maxa, # a comma puts the next on the same line, interestingly
    print a

if __name__ == '__main__':
    global X
    reveal()
    print X
    a = SyntaxClass(3)
    print X
    X = 4
    # print a.obj
    # print a.obj.a
    a.say()
    print X
    # a.obj.hello()
