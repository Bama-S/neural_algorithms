#This program is the implementation of example program of perceptron algorithm from Zurada's Introduction to Artificial Neural networks

import numpy as np
#/*----------------Function for perceptron algorithm --------------*/
def perceptron(c,X,d,w,iter):
    for n in range(1,iter):# Number of iterations  = 7
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
    return w
#/*---------------------Function for testing the perceptron-----------*/

def test_perceptron(final_out,X,w):
    for i,x in enumerate(X):
        net = np.dot(X[i],w)
        if net>0:
            out = 1
        else:
            out = -1
        final_out = final_out+[out]
    return final_out

#*---------------Training---------------------------------*/
X = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1],[-1,1,0.5,-1],])
print "Inputs", X
d = np.array([-1,-1,1])
print "Teacher values", d
w= ([1,-1,0,0.5])
print "initial values of weights", w
c = 0.1
iterations = 7
print "Training"
print "----------"
final_weight = perceptron(c,X,d,w,iterations)
print "Final sets of weights: ", final_weight
#*-----------------Testing-------------------------------*/
final_out = []
print "Testing"
print "--------"
final_output = test_perceptron(final_out,X,final_weight)
print "Final output: ", final_output
print "Original Teacher values", d
