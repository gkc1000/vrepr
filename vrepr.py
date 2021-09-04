#!/usr/bin/env python
import numpy as np
import scipy.optimize

def genrho(n, theta):
    occ = np.diag([n, 1.-n])
    u = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    return np.dot(u.T, np.dot(occ, u))

def tb(n):
    h = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if abs(i-j)==1: h[i,j]=1.
    #h[0,-1]=1.
    #h[-1,0]=1.
    return h
    
def unitdm(u):
    h =tb(4)
    h[0,1] += u[0]
    h[1,0] += u[0]
    h[1,1] += u[1]
    #h[2:,2:] = h[0:2,0:2]
    h[2,3] += u[0]
    h[3,2] += u[0]
    h[2,2] += u[1]
    
    e,v = np.linalg.eigh(h)

    #print (h)
    #print (e)
    #print(v)
    #dm = np.outer(v[:,0], v[:,0]) + np.outer(v[:,1], v[:,1])
    dm = np.outer(v[:,0], v[:,0]) + np.outer(v[:,1], v[:,1])
    #print (dm)
    return dm[:2,:2]


def gap(u):
    h =tb(4)
    h[0,1] += u[0]
    h[1,0] += u[0]
    h[1,1] += u[1]
    #h[2:,2:] = h[0:2,0:2]
    h[2,3] += u[0]
    h[3,2] += u[0]
    h[2,2] += u[1]
    
    e,v = np.linalg.eigh(h)
    return e[2]-e[1]


    

def fit(rho):
    def error(u):
        val = np.linalg.norm(rho-unitdm(u))
        return val
    u0 = np.array([0.1,0.3])
    res = scipy.optimize.minimize(error, u0, options={"disp":True, "gtol":1.e-3})
    return res.x, res.fun


def test():
    col = []
    gap_col = []
    for n in np.linspace(0.0, 1.0, 31):
        col_tmp = []
        gap_tmp = []
        for angle in np.linspace(0, np.pi, 31):
            rho0 = genrho(n, np.pi*angle)
            print("target\n", rho0)
            print(np.linalg.eigvalsh(rho0))
            ufit, fun = fit(rho0)
            print("fit error", fun)
            print("fitted\n", unitdm(ufit))
            print("gap\n",gap(ufit))
            col_tmp.append(fun)
            gap_tmp.append(gap(ufit))

        col.append(col_tmp)
        gap_col.append(gap_tmp)
    col = np.array(col)
    np.save("col.npy", col)
    gap_col = np.array(gap_col)
    np.save("gap_col.npy", gap_col)
    print (col)

if __name__ == '__main__':
    test()
