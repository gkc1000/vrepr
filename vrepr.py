#!/usr/bin/env python

import numpy as np
import scipy
from scipy import linalg as la
from scipy import optimize as opt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def gen_rho(n, theta):
    """
    2x2 density matrix for two sites.
    """
    assert 0 <= n <= 1
    occ = np.array([n, 1.0 - n])
    u = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(u * occ, u.T)


def tb(L, pbc=False):
    """
    1D tight-binding Hamiltonian for lattice of length L
    """
    h = np.zeros((L, L))
    for i in range(L-1):
        h[i, i+1] = h[i+1, i] = 1.0

    if pbc:
        h[0, -1] = 1.0
        h[-1, 0] = 1.0
    return h


def add_u_pbc(h, u):
    L = h.shape[0]
    hu = h.copy()
    for c in range(L//2):
        hu[2*c, 2*c+1] += u[0]
        hu[2*c+1, 2*c] += u[0]
        hu[2*c+1, 2*c+1] += u[1]
    return hu
    
def dm_pbc(u, L, occ=None):
    h=tb(L,pbc=True)
    h = add_u_pbc(h, u)

    e,v = np.linalg.eigh(h)
    dm = np.zeros([L,L])
    if occ is None:
        occ = range(L//2)
    for i in occ:
        dm += np.outer(v[:,i], v[:,i])
    return dm

    
def gap_pbc(u, L):
    """
    half-filled gap, assuming pbc tb, chain L, and correlation potential u
    """
    h =tb(L,pbc=True)
    for c in range(L//2):
        h[2*c,2*c+1] += u[0]
        h[2*c+1,2*c] += u[0]
        h[2*c+1,2*c+1] += u[1]

    e,v = np.linalg.eigh(h)
    return e[L//2]-e[L//2-1]

def get_embedding_h(u,L,occ=None):
    """
    embedding Hamiltonian for first two sites
    """
    h_pbc=tb(L,pbc=True)
    h_pbc = add_u_pbc(h_pbc, u)
    dm=dm_pbc(u,L,occ)
    u,s,v=np.linalg.svd(dm[:2,2:], full_matrices=False)

    basis = np.zeros([L, 4])
    basis[:2,:2]=np.eye(2)
    basis[2:,2:]=v.T
    return np.dot(basis.T, np.dot(h_pbc, basis))
    
def fit(rho,L,occ=None,method="bfgs"):
    """
    global fit
    """
    def error(u,occ=occ):
        dm = dm_pbc(u, L, occ)
        val = np.linalg.norm(rho-dm[:2,:2])
        return val
    u0 = np.array([0.1, 0.3])
    if method == "bfgs":
        res = scipy.optimize.minimize(error, u0, method='bfgs', options={"disp":True, "gtol":1.e-6})
        return res.x, res.fun#[0], res[1]
    elif method =="brute":

        res = scipy.optimize.brute(error, (slice(-3, 3, 0.05), slice(-4, 4, 0.25)),
                                   full_output=True, finish=scipy.optimize.fmin_bfgs)
        return res[0], res[1]
    else:
        raise NotImplementedError

def fit_local(rho,L,occ=None,method="bfgs"):
    """
    local u fit
    """
    def error(u,occ=occ):
        h_emb=get_embedding_h(u,L,occ)
        dm=np.zeros_like(h_emb)
        e,v=np.linalg.eigh(h_emb)
        if occ is None:
            occ = [0,1]
        for i in occ:
            dm+=np.outer(v[:,i],v[:,i])
        val = np.linalg.norm(rho-dm[:2,:2])
        return val
    u0 = [0, 0]
    if method == "bfgs":
        res = scipy.optimize.minimize(error, u0, method='BFGS', options={"disp":True, "gtol":1.e-6})
        return res.x, res.fun#[0], res[1]
    elif method == "brute":

        res = scipy.optimize.brute(error, (slice(-3, 3, 0.05), slice(-4, 4, 0.25)),
                                   full_output=True, finish=scipy.optimize.fmin_bfgs)
        return res[0], res[1]
    else:
        raise NotImplementedError

def gap_local(u,L):
    hemb=get_embedding_h(u,L)
    e,v=np.linalg.eigh(hemb)
    return e[2]-e[1]

def generate_data(occ,occ_loc):
    import csv
    l = 6
    occstr = "".join(str(o) for o in occ)
    occlocstr = "".join(str(o) for o in occ_loc)
    with open("data"+occstr+occlocstr, "w", newline='') as datafile:
        writer = csv.writer(datafile)
        writer.writerow(["n", "angle", "fun", "fun_local", "gap", "gap_loc", "small_gap", "small_gap_loc", "large_gap", "large_gap_loc", "big_diff"])
        for n in np.linspace(0.0, 1.0, 31):
            for angle in np.linspace(0, np.pi, 31):
                rho0 = gen_rho(n, np.pi*angle)
                ufit, fun = fit(rho0,l,occ)
                ufit_local, fun_local = fit_local(rho0,l,occ_loc)
                gap = gap_pbc(ufit,l)
                gap_loc = gap_local(ufit,l)
                small_gap = int(abs(gap)<1.e-3)
                small_gap_loc = int(abs(gap_loc)<1.e-3)
                large_gap = int(abs(gap)>5)
                large_gap_loc = int(abs(gap_loc)>5)
                big_diff = int(abs(fun - fun_local)>1.e-4)

                writer.writerow([n, angle, fun, fun_local, gap, gap_loc, small_gap, small_gap_loc, large_gap, large_gap_loc, big_diff])
            
            
            
#             print("global, local", fun,fun_local)
#             print(ufit)
#             print(ufit_local)
#             print("gap\n",gap_pbc(ufit,l), gap_loc)
            
#             dm_p = dm_pbc(ufit, l)
#             print(dm_p)

#             h_emb=get_embedding_h(ufit_local,l)
#             dm_l=np.zeros_like(h_emb)
#             e,v=np.linalg.eigh(h_emb)
#             dm_l = np.outer(v[:,0],v[:,0])+np.outer(v[:,1],v[:,1])
#             print(dm_l)
            
#             if(gap<1.e-3):
#                 print("SMALL GAP===================")
#             if (abs(fun-fun_local)>1.e-5):
#                 print("WARNING********************")
            
#             #print("fit error", fun)
#             print("fitted\n", unitdm_pbc(ufit,l))
#             print("target\n", rho0)

#             col_tmp.append(fun)
#             gap_tmp.append(gap_pbc(ufit,l))
#             u0_tmp.append(ufit[0])
#             u1_tmp.append(ufit[1])

#         col.append(col_tmp)
#         gap_col.append(gap_tmp)
#         u0_col.append(u0_tmp)
#         u1_col.append(u1_tmp)



            


# def test_embedding():
#     # this script demonstrates that embedding non-aufbau leads to an excited state
#     # configuration of the local problem
#     L = 6
#     h_pbc=tb(L, pbc=True)
#     occ=[0,1,3]
#     print(dm_pbc([0,0],L,occ))
#     h_embed=get_embedding_h([0,0], L,occ)
#     print(h_embed)
#     e,v=np.linalg.eigh(h_embed)

#     print("------------")
#     print(np.outer(v[:,0],v[:,0])+np.outer(v[:,1],v[:,1]))
#     print(np.outer(v[:,0],v[:,0])+np.outer(v[:,2],v[:,2]))
#     print(np.outer(v[:,0],v[:,0])+np.outer(v[:,3],v[:,2]))


            
    
# def test_equality():
#     col = []
#     gap_col = []
#     u0_col = []
#     u1_col = []

#     # check local vs global
#     n = 0.05
#     angle = 0.3*np.pi
#     rho0 = gen_rho(n, np.pi*angle)
    
#     l = 6
#     for n in np.linspace(0.0, 1.0, 31):
#         col_tmp = []
#         gap_tmp = []
#         u0_tmp = []
#         u1_tmp = []
#         for angle in np.linspace(0, np.pi, 31):
#             rho0 = gen_rho(n, np.pi*angle)

#             print(np.linalg.eigvalsh(rho0))
#             ufit, fun = fit(rho0,l)
#             ufit_local, fun_local = fit_local(rho0,l)
#             print("global, local", fun,fun_local)
#             print(ufit)
#             print(ufit_local)
#             gap = gap_pbc(ufit,l)
#             gap_loc = gap_local(ufit,l)
#             print("gap\n",gap_pbc(ufit,l), gap_loc)
            
#             dm_p = dm_pbc(ufit, l)
#             print(dm_p)

#             h_emb=get_embedding_h(ufit_local,l)
#             dm_l=np.zeros_like(h_emb)
#             e,v=np.linalg.eigh(h_emb)
#             dm_l = np.outer(v[:,0],v[:,0])+np.outer(v[:,1],v[:,1])
#             print(dm_l)
            
#             if(gap<1.e-3):
#                 print("SMALL GAP===================")
#             if (abs(fun-fun_local)>1.e-5):
#                 print("WARNING********************")
            
#             #print("fit error", fun)
#             print("fitted\n", unitdm_pbc(ufit,l))
#             print("target\n", rho0)

#             col_tmp.append(fun)
#             gap_tmp.append(gap_pbc(ufit,l))
#             u0_tmp.append(ufit[0])
#             u1_tmp.append(ufit[1])

#         col.append(col_tmp)
#         gap_col.append(gap_tmp)
#         u0_col.append(u0_tmp)
#         u1_col.append(u1_tmp)


    
# def test():
#     col = []
#     gap_col = []
#     u0_col = []
#     u1_col = []

# #    ##### This code is to plot the surface of u for a given rho0 ########
# #    n = 0.05
# #    angle = 0.3*np.pi
# #    rho0 = gen_rho(n, np.pi*angle)
# #    print(rho0)
# #    print("rho0 eigs", np.linalg.eigvalsh(rho0))
# #    ufit, fun = fit(rho0, unitdm4)
# #    print(gap4(ufit))
# #    print(ufit, fun)
# #
# #    print("dm eigs", np.linalg.eigvalsh(unitdm4(ufit)))
# #    
# #    for u0 in np.linspace(-3.,3.,30):
# #        col_tmp = []
# #        u0_tmp = []
# #        u1_tmp = []
# #
# #        for u1 in np.linspace (-3., 3., 30):
# #            val = np.linalg.norm(rho0-unitdm4([u0,u1]))
# #            print(rho0)
# #            print(unitdm4([u0,u1]))
# #            print("dm eigs", np.linalg.eigvalsh(unitdm4([u0,u1])))
# #            print("gap", gap4([u0,u1]))
# #            print("params, val", u0, u1, val)
# #            col_tmp.append(val)
# #            u0_tmp.append(u0)
# #            u1_tmp.append(u1)
# #
# #        col.append(col_tmp)
# #        u0_col.append(u0_tmp)
# #        u1_col.append(u1_tmp)

#     #### This code tests the 6 site pbc model    
#     l = 6
#     for n in np.linspace(0.0, 1.0, 31):
#         col_tmp = []
#         gap_tmp = []
#         u0_tmp = []
#         u1_tmp = []
#         for angle in np.linspace(0, np.pi, 31):
#             rho0 = gen_rho(n, np.pi*angle)
#             print("target\n", rho0)
#             print(np.linalg.eigvalsh(rho0))
#             ufit, fun = fit(rho0,l)
#             print("fit error", fun)
#             print("fitted\n", unitdm_pbc(ufit,l))
#             print("gap\n",gap_pbc(ufit,l))
#             col_tmp.append(fun)
#             gap_tmp.append(gap_pbc(ufit,l))
#             u0_tmp.append(ufit[0])
#             u1_tmp.append(ufit[1])

#         col.append(col_tmp)
#         gap_col.append(gap_tmp)
#         u0_col.append(u0_tmp)
#         u1_col.append(u1_tmp)

#     col = np.array(col)
#     np.save("col.npy", col)
#     gap_col = np.array(gap_col)
#     np.save("gap_col.npy", gap_col)
#     u0_col = np.array(u0_col)
#     np.save("u0_col.npy", u0_col)
#     u1_col = np.array(u1_col)
#     np.save("u1_col.npy", u1_col)
#     print (col)

# if __name__ == '__main__':
#     test_gen_rho()
#     test_tb()
#     exit()

#     print (gen_rho(n, theta))
#     #test()
#     test_equality()
if __name__=='__main__':
    generate_data([0,1,2],[0,1])
