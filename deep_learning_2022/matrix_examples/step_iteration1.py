import numpy as np
#step1 of p class
def multiple_iterations:

    def activate(net):
        out = net
        out[out>0]=1
        out[out<=0]=-1
        return out

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
        print("Iteration ", iter+1)
        print("--------------------")
        for i in range(0,n):
            out = np.zeros((3,3))
            print(y[i])
            net0 = np.dot(w,y[i])
            print("net")
            print(net0)
            print("calculation of output ")
            print(out[i])
            out[i]=net0
            out[i][out[i]>0] = 1
            out[i][out[i]<=0] = -1
            print(out[i])
            print(out[i])
            r[i]=(1/2)*(d[i]-out[i])
            print("r = d - out")
            print(r[i])
            rt = r[i].reshape(3,1)
            print(rt)
            print(rt.shape)
            yt = y[i].reshape(1,3)
            print(yt)
            print(yt.shape)
            del_w = np.matmul(rt,yt)
            print(del_w)
            w = w+del_w
            print("weight")
            print(w)
    return (w)
### Testing
y_new = np.array([[11,3,1],[2,-6,1],[-6,6,1],])
net1 = np.dot(w,y_new[0])
net2 = np.dot(w,y_new[1])
net3 = np.dot(w,y_new[2])
out1 = activate(net1)
out2 = activate(net2)
out3 = activate(net3)
print("output for ", y_new)
print(out1,out2,out3)
