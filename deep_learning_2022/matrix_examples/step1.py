import numpy as np
#step1 of p class
def p_class():
    y = np.array([[10,2,-1],[2,-5,-1],[-5,5,-1],])
    print("inputs")
    print(y)
    w = np.array([[1,-2,0],[0,-1,2],[1,3,-1],])
    print("initial weights")
    print(w)
    d = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1],])
    print("teacher values")
    print(d)
    r = np.zeros((3,3))
    out = np.zeros((3,3))
    #-------------training
    #step1
    net0 = np.dot(w,y[0])
    print(net0)
    out[0]=net0
    out[0][out[0]>0] = 1
    out[0][out[0]<=0] = -1
    print(out[0])
    r[0]=(1/2)*(d[0]-out[0])
    print("r = d - out")
    print(r[0])
    rt = r[0].reshape(3,1)
    print(rt)
    print(rt.shape)
    yt = y[0].reshape(1,3)
    print(yt)
    print(yt.shape)
    del_w = np.matmul(rt,yt)
    print(del_w)
    w = w+del_w
    print("weight")
    print(w)
    return w
