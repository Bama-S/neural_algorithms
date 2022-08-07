#implementation of softmax
from math import exp

#find exponents
x = [50,10,1,0.1]
y = [0,0,0,0]
soft_max_result = [0,0,0,0]

#find exponents
for i in range(len(x)):
    y[i] = exp(x[i])
    print(x[i],":",y[i])

#normalize
total_y = sum(y)
print("denominator ", total_y)
for i in range(len(x)):
    soft_max_result[i] = y[i]/total_y
print("softmax result")
print(soft_max_result)
