import numpy as np
#step1 of p class
def multiple_iterations():
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

    #-------------training
    #step1
    n = len(y)
    num_iterations = 4
    for iter in range(0,num_iterations):
        for i in range(0,n):
            out = np.zeros((3,3))
            net0 = np.dot(w,y[i])
            out[i]=net0
            out[i][out[i]>0] = 1
            out[i][out[i]<=0] = -1
            r[i]=(1/2)*(d[i]-out[i])
            rt = r[i].reshape(3,1)
            yt = y[i].reshape(1,3)
            del_w = np.matmul(rt,yt)
            w = w+del_w
    return (w)
### Testing
w = multiple_iterations()
def check_new(w):
    def activate(net):
        out = net
        out[out>0]=1
        out[out<=0]=-1
        return out

    y_new = np.array([[11,3,1],[2,-6,1],[-6,6,1],])
    net1 = np.dot(w,y_new[0])
    net2 = np.dot(w,y_new[1])
    net3 = np.dot(w,y_new[2])
    out = np.zeros((3,3))
    out[0] = activate(net1)
    out[1] = activate(net2)
    out[2] = activate(net3)
    print("output for new inputs \n ", y_new)
    print("-----------")
    print(out)
    return (out )
check_new(w)
