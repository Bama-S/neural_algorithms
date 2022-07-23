#This program is the implementation of example program of perceptron algorithm from Zurada's Introduction to Artificial Neural networks

import numpy as np
#/*----------------Function for perceptron algorithm --------------*/
def perceptron(c,X,d,w,iter):
    for n in range(1,iter):# Number of iterations
        for i, x in enumerate(X):
            print("i", i)
            print("x ", x)
            print("w ", w)
            net = np.dot(X[i],w)
            print(net)
            print("net shape ", net.shape)
            if net > 0:
                out = 1
            else:
                out = -1
            print("out ", out)
            r = c*(d[i] - out)
            print ("r ", r)
            delta_w = r*x
            delta_w = delta_w.reshape(2,1)
            print("delta_w ", delta_w)
            print(delta_w.shape)
            print("----------")
            print(w.shape)
            w = delta_w+w
            print ("weight ", w)
    return w
#/*---------------------Function for testing the perceptron-----------*/

def test_perceptron(final_out,X,w):
    for i,x in enumerate(X):
        net = np.dot(X[i],w)
        print("net ", net)
        if net>0:
            out = 1
        else:
            out = -1
        final_out = final_out+[out]
    return final_out

#*---------------Training---------------------------------*/
X = np.array([[5,1],[7,3],[3,2],[5,4],[0,0],[-1,-3],[-2,3],[-3,0],])
new_input = np.array([[4,2],[0,5],[36/13,0]])#feed new inputs and see
print ("Inputs", X)
d = np.array([1,1,1,1,-1,-1,-1,-1])
print ("Teacher values", d)
#w= ([1,1])
w = np.array([[0.1],[0.3],])
print(w.shape)
print ("initial values of weights", w)
c = 1
iterations = 5
print ("Training")
print ("----------")
final_weight = perceptron(c,X,d,w,iterations)
print ("Final sets of weights: ", final_weight)
#*-----------------Testing-------------------------------*/
final_out = []
print ("Testing")
print ("--------")
final_output = test_perceptron(final_out,new_input,final_weight)
print ("Final output: ", final_output)
#print ("Original Teacher values", d)
