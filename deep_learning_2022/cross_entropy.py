#determination of cross entropy
from matplotlib import pyplot
from math import log2 

def cross_entropy(p,q):
    entropy = 0
    for i in range(len(p)):
        entropy = entropy + (p[i]*log2(q[i]))
    entropy = -entropy
    return entropy

p = [0.10, 0.40, 0.50]
q = [0.8,0.15,0.05]
ce_pq = cross_entropy(p,q)
print('H(P,Q): %.3f bits' % ce_pq)

ce_qp = cross_entropy(q,p)
print('H(Q,P): %.3f bits' % ce_qp)

print("----------------------------")
p = [0.10, 0.40, 0.50]
q = [0.10,0.40,0.50]
ce_pq = cross_entropy(p,q)
print("Cross entropy with equal values")
print('H(P,Q): %.3f bits' % ce_pq)
ce_qp = cross_entropy(q,p)
print('H(Q,P): %.3f bits' % ce_qp)
print("----------------------------")
p = [0.01, 0.40, 0.50]
q = [0.99,0.40,0.50]
ce_pq = cross_entropy(p,q)
print("Cross entropy with 0.1 difference in q")
print('H(P,Q): %.3f bits' % ce_pq)
ce_qp = cross_entropy(q,p)
print('H(Q,P): %.3f bits' % ce_qp)
print("----------------------------")
p = [0.10, 0.40, 0.50]
q = [0.8,0.15,0.05]
ce_pq = cross_entropy(p,p)
print("Cross entropy with same probability")
print('H(P,Q): %.3f bits' % ce_pq)
ce_qp = cross_entropy(q,q)
print('H(Q,P): %.3f bits' % ce_qp)


