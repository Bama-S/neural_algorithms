import numpy as np
import three_c_perceptron

def test_three_c():
    global w
    w = np.array([[5,3,5],[0,-1,2],[-9,1,0],])
    three_c_perceptron.multiple_iterations()==w

def check_three_c():
    y_new = np.array([[11,3,1],[2,-6,1],[-6,6,1],])
    three_c_perceptron.check_new(w) == np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1],])
