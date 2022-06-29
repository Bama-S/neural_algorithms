import numpy as np

X = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1]])
print X
d = np.array([-1,-1,1])
w = np.array([1,-1,0,0.5])
print d
net = np.dot(X[0],w)
if net>0:
    o=1
else:
    o=-1
print o
