#This program is the implementation of example program of perceptron algorithm from Zurada's Introduction to Artificial Neural networks

import numpy as np
X = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1],[-1,1,0.5,-1],])
print "Inputs", X
d = np.array([-1,-1,1])
print "Teacher values", d
w= ([1,-1,0,0.5])
print "initial values of weights", w
c = 0.1
for n in range(1,7):# Number of iterations  = 7
    for i, x in enumerate(X):
        net = np.dot(X[i],w)
        if net > 0:
            out = 1
        else:
            out = -1
        r = c*(d[i] - out)
        delta_w = r*x
        w = delta_w+w
        print n, i, w
print "Final sets of weights",w
final_out = []
print "Testing"
for i,x in enumerate(X):
    net = np.dot(X[i],w)
    if net>0:
        out = 1
    else:
        out = -1
    final_out = final_out+[out]
print "final output", final_out
print "d", d
