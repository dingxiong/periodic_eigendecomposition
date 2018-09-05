from py_ped import pyPED
from personalFunctions import *

case = 1

if case == 1:
    """
    test power iteration for single J
    """
    ped = pyPED()
    n = 4
    J = rand(n, n)
    Q = rand(n, n)
    q, r, d, c = ped.PowerIter(J, Q, True, 1000, 1e-15, True, 100)
    print eig(J.T)[0]
    print r.T
    print d.T

if case == 2:
    """
    test power iteration for multiple J.
    Q can be square or rectangle
    """
    ped = pyPED()
    n = 8
    m = 3
    m2 = 4
    J = rand(m*n, n)
    Q = rand(m2, n)
    
    q, r, d, c = ped.PowerIter(J, Q, True, 1000, 1e-15, True, 20)
    JJ = eye(n)
    rr = eye(m2)
    for i in range(m):
        JJ = dot(JJ, J[i*n:(i+1)*n, :].T)  # be cautious of the order
        rr = dot(rr, r[i*m2:(i+1)*m2, :].T)  # use dot for multiplication
    print sort(eig(JJ)[0])
    print diag(rr)
    print d.T

if case == 3:
    """
    test my QR
    """
    ped = pyPED()
    A = rand(4, 10)
    q, r = ped.QR(A)
    print q
    print r.T

if case == 4:
    """
    test the eigenvalue function
    """
    ped = pyPED()
    n = 8
    m = 3
    m2 = 8
    J = rand(m*n, n)
    Q = rand(m2, n)
    
    e = ped.PowerEigE(J, Q, 1000, 1e-15, True, 20)
    JJ = eye(n)
    for i in range(m):
        JJ = dot(JJ, J[i*n:(i+1)*n, :].T)  # be cautious of the order
    e2 = eig(JJ)[0]
    idx = argsort(abs(e2))
    e2 = e2[idx]
    print log(abs(e2))
    print e
