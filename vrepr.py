#!/usr/bin/env python
import numpy as np
import scipy.optimize

def genrho(n, theta):
    occ = np.diag([n, 1.-n])
    u = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    return np.dot(u.T, np.dot(occ, u))

def tb(n, pbc=False):
    h = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if abs(i-j)==1: h[i,j]=1.
    #pbc=True
    if pbc:
        h[0,-1]=1.
        h[-1,0]=1.
    return h
    
def unitdm4(u, full=False):
    h =tb(4)
    h[0,1] += u[0]
    h[1,0] += u[0]
    h[1,1] += u[1]
    #h[2:,2:] = h[0:2,0:2]
    #h[2,3] += u[0]
    #h[3,2] += u[0]
    #h[2,2] += u[1]
    
    e,v = np.linalg.eigh(h)

    #print (h)
    #print (e)
    #print(v)
    #dm = np.outer(v[:,0], v[:,0]) + np.outer(v[:,1], v[:,1])
    dm = np.outer(v[:,0], v[:,0]) + np.outer(v[:,1], v[:,1])
    #print (dm)
    if full:
        return dm
    else:
        return dm[:2,:2]

def unitdm_pbc(u, l):
    h =tb(l,pbc=True)
    for c in range(l//2):
        h[2*c,2*c+1] += u[0]
        h[2*c+1,2*c] += u[0]
        h[2*c+1,2*c+1] += u[1]

    e,v = np.linalg.eigh(h)
    dm = np.zeros([l,l])
    for i in range(l//2):
        dm += np.outer(v[:,i], v[:,i])
    return dm[:2,:2]
    
def gap4(u):
    h =tb(4)
    h[0,1] += u[0]
    h[1,0] += u[0]
    h[1,1] += u[1]
    h[2,3] += u[0]
    h[3,2] += u[0]
    h[2,2] += u[1]
    
    e,v = np.linalg.eigh(h)
    return e[2]-e[1]

def gap_pbc(u, l):
    h =tb(l,pbc=True)
    for c in range(l//2):
        h[2*c,2*c+1] += u[0]
        h[2*c+1,2*c] += u[0]
        h[2*c+1,2*c+1] += u[1]

    e,v = np.linalg.eigh(h)
    print(e)
    return e[l//2]-e[l//2-1]
    
def fit(rho, dmfn=None):
    def error(u):
        val = np.linalg.norm(rho-dmfn(u))
        return val
    u0 = np.array([0.1, 0.3])
    res = scipy.optimize.minimize(error, u0, method='BFGS', options={"disp":True, "gtol":1.e-4})
    #res = scipy.optimize.minimize(error, u0, method='L-BFGS-B', options={"disp":False, "gtol":1.e-4})
    #return res.x, res.fun
    #res = scipy.optimize.brute(error, (slice(-1, 1, 0.05), slice(-4, 4, 0.25)),
    #                           full_output=True, finish=scipy.optimize.fmin)
    return res.x, res.fun#[0], res[1]
    #return res[0], res[1]

def test():
    col = []
    gap_col = []
    u0_col = []
    u1_col = []


    ##### This code is to plot the surface of u for a given rho0 ########
    n = 0.05
    angle = 0.3*np.pi
    rho0 = genrho(n, np.pi*angle)
    print(rho0)
    print("rho0 eigs", np.linalg.eigvalsh(rho0))
    ufit, fun = fit(rho0, unitdm4)
    print(gap4(ufit))
    print(ufit, fun)

    print("dm eigs", np.linalg.eigvalsh(unitdm4(ufit)))
    
    for u0 in np.linspace(-3.,3.,30):
        col_tmp = []
        u0_tmp = []
        u1_tmp = []

        for u1 in np.linspace (-3., 3., 30):
            val = np.linalg.norm(rho0-unitdm4([u0,u1]))
            print(rho0)
            print(unitdm4([u0,u1]))
            print("dm eigs", np.linalg.eigvalsh(unitdm4([u0,u1])))
            print("gap", gap4([u0,u1]))
            print("params, val", u0, u1, val)
            col_tmp.append(val)
            u0_tmp.append(u0)
            u1_tmp.append(u1)

        col.append(col_tmp)
        u0_col.append(u0_tmp)
        u1_col.append(u1_tmp)

    ##### This code tests the 6 site pbc model    
    # l = 6
    # for n in np.linspace(0.0, 1.0, 31):
    #     col_tmp = []
    #     gap_tmp = []
    #     u0_tmp = []
    #     u1_tmp = []
    #     for angle in np.linspace(0, np.pi, 31):
    #         rho0 = genrho(n, np.pi*angle)
    #         print("target\n", rho0)
    #         print(np.linalg.eigvalsh(rho0))
    #         ufit, fun = fit(rho0,l)
    #         print("fit error", fun)
    #         print("fitted\n", unitdm_pbc(ufit,l))
    #         print("gap\n",gap_pbc(ufit,l))
    #         col_tmp.append(fun)
    #         gap_tmp.append(gap_pbc(ufit,l))
    #         u0_tmp.append(ufit[0])
    #         u1_tmp.append(ufit[1])

    #     col.append(col_tmp)
    #     gap_col.append(gap_tmp)
    #     u0_col.append(u0_tmp)
    #     u1_col.append(u1_tmp)

    col = np.array(col)
    np.save("col.npy", col)
    gap_col = np.array(gap_col)
    np.save("gap_col.npy", gap_col)
    u0_col = np.array(u0_col)
    np.save("u0_col.npy", u0_col)
    u1_col = np.array(u1_col)
    np.save("u1_col.npy", u1_col)
    print (col)

if __name__ == '__main__':
    test()
