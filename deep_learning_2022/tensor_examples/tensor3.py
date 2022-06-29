#tensor demo
#1D
import numpy as np
x = np.array([2,5,7])
print(x)
print("Dimension ", x.ndim)
print("Shape ", x.shape)

y = np.array([[2,5,7],[3,6,8]])
print(y)
print("Dimension ", y.ndim)
print("Shape ", y.shape)

z = np.array([[[1,3,5,6],[2,4,6,7],[1,2,3,4]],[[4,3,2,1],[7,9,11,12],[8,10,12,9]]])
print(z)
print("Dimension ", z.ndim)
print("Shape ", z.shape)
