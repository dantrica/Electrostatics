#-*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:02:47 2018

@author: Silvi
version for python 2.7 working on
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nq = input('Ingrese el número de cargas n = ')
rp = np.array(input('Ingrese el vector al punto p, coordenadas (x,y,z) y en [m] = '))


def e_field(rp, rq, q, n, ke=9e9):
    ri = np.zeros((n, 3))
    mag_ri = np.zeros(n)
    uni_ri = np.zeros((n, 3))
    vec_E = np.zeros((n, 3))
    vec_Ep = np.zeros(3)
    mag_Ep = 0.0    
    if n < 1:
        print('1 es el mínimo número de cargas para producir campo eléctrico')
    else:
        ri = rp - rq
        i = 0
        while i < n:
            mag_ri[i] = np.sqrt(sum(ri[i]**2))
            uni_ri[i] = ri[i]/mag_ri[i]
            vec_E[i] = (ke*q[i]/mag_ri[i]**2)*uni_ri[i]
            i += 1
        vec_Ep = np.array([sum(vec_E[:,0]), sum(vec_E[:,1]), sum(vec_E[:,2])])
        mag_Ep = np.sqrt(sum(vec_Ep**2))
    return ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep

def figure(nf=1):
    # Create figure
    fig = plt.figure(nf)
    ax = fig.gca(projection='3d')
    return ax
  
def plot_reference(rp, rq):
    ax = figure()
    # Draw coordinate system
    mag_rp = np.sqrt(rp[0]**2+rp[1]**2+rp[2]**2)
    arrows = np.array([[0, 0, 0, 2*mag_rp, 0, 0], [0, 0, 0, 0, 2*mag_rp, 0], [0, 0, 0, 0, 0, 2*mag_rp]])
    X, Y, Z, U, V, W = zip(*arrows)
    ax.quiver3D(X, Y, Z, U, V, W, pivot='tail', length = 1.0, color='k', arrow_length_ratio=0.1) #length = 2*mag_rp
    ax.set_xlim([-rq.max()-rp.max(), rq.max()+rp.max()])
    ax.set_ylim([-rq.max()-rp.max(), rq.max()+rp.max()])
    ax.set_zlim([-rq.max()-rp.max(), rq.max()+rp.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Draw one rp and the point
    ax.scatter3D(rp[0], rp[1], rp[2], c='k', s=50)
    ax.quiver3D([0], [0], [0], rp[0], rp[1], rp[2], pivot='tail', length = 1.0, arrow_length_ratio=0.1, color='g')
    return

def plot_charges(rp, rq, q, n):
    ax = figure()
    t = 0
    while t < n:
        if q[t]>0:
            co = 'r' # Positive electric chargue
        else:
            co = 'b' # Negative electric charge
        # Draw the electric charges
        ax.scatter3D(rq[t][0], rq[t][1], rq[t][2], c=co, s=100)
        t += 1
    return

def plot_vectors(rq, ri, n):
    ax = figure()
    j = 0
    while j < n:
        # Draw rq vectors
        ax.quiver3D([0], [0], [0], rq[j][0], rq[j][1], rq[j][2], pivot='tail', length = 1.0, arrow_length_ratio=0.1) #length = np.sqrt(rq[i][0]**2+rq[i][1]**2+rq[i][2]**2)
        # Draw ri vectors
        ax.quiver3D(rq[j][0], rq[j][1], rq[j][2], ri[j][0], ri[j][1], ri[j][2], pivot='tail', length = 1.0, arrow_length_ratio=0.1, color='m') #length = mag_ri[i]
        j += 1
    return
  
def plot_vector(rq, ri, n):
    ax = figure()
    w = int(n/6)
    # Draw drq vector
    ax.quiver3D([0], [0], [0], rq[w][0], rq[w][1], rq[w][2], pivot='tail', length = 1.0, arrow_length_ratio=0.1) #length = np.sqrt(rq[i][0]**2+rq[i][1]**2+rq[i][2]**2)
    # Draw ri vector
    ax.quiver3D(rq[w][0], rq[w][1], rq[w][2], ri[w][0], ri[w][1], ri[w][2], pivot='tail', length = 1.0, arrow_length_ratio=0.1, color='m') #length = mag_ri[i]
    return

def plot_E(vec_Ep):
    ax = figure()
    # Draw one vector electric field
    ax.quiver3D(rp[0], rp[1], rp[2], vec_Ep[0], vec_Ep[1], vec_Ep[2], pivot='tail', length = 1, color='c', arrow_length_ratio=0.3, normalize=True)
    # The result of the electric field has been written in the title
    ax.set_title('$E_p = $'+str(vec_Ep[0])+' $i$ +'+str(vec_Ep[1])+' $j$ +'+str(vec_Ep[2])+' $k$'+str( )+' $[V/m]$')
    return


def punctual_charges(rp, n):
    l = 0
    q = np.zeros(n)
    rq = np.zeros((n, 3))
    while l < n:
        q[l] = input('Ingrese el valor de la carga numero '+str(l+1)+' en Coulomb = ')
        rq[l] = input('Ingrese el vector posicion (x,y,z) de la carga numero '+str(l+1)+' = ')
        l += 1
    ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = e_field(rp=rp, rq=rq, q=q, n=nq)
    plot_reference(rp, rq)
    plot_charges(rp, rq, q, n)
    plot_vectors(rq, ri, n)
    plot_E(vec_Ep)
    plt.show()
    return q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep
  

def ring(rp, R, ld, n=100):
    s = 0
    p = 2*np.pi*R
    qo = ld*p/n
    rq = np.zeros((n, 3))
    q = np.zeros(n)
    while s < n:
        phi = s*2*np.pi/n
        q[s] = qo
        rq[s] = R*np.array([np.cos(phi), np.sin(phi), 0])
        s += 1
    ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = e_field(rp=rp, rq=rq, q=q, n=n)
    plot_reference(rp, rq)
    plot_charges(rp, rq, q, n)
    plot_vector(rq, ri, n)
    plot_E(vec_Ep)
    plt.show()
    return q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep
  
def line(rp, L, ld, n=100):
    v = 0
    dL = L/(n-1)
    qo = ld*L/n
    rq = np.zeros((n, 3))
    q = np.zeros(n)
    while v < n:
        q[v] = qo
        rq[v] = np.array([v*dL-L/2, 0, 0])
        v += 1
    ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = e_field(rp=rp, rq=rq, q=q, n=n)
    plot_reference(rp, rq)
    plot_charges(rp, rq, q, n)
    plot_vector(rq, ri, n)
    plot_E(vec_Ep)
    plt.show()
    return q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep

q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = line(rp=rp, L=2.0, ld=1e-6)
#q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = ring(rp=rp, R=1.0, ld=1e-6)
#q, rq, ri, mag_ri, uni_ri, vec_E, vec_Ep, mag_Ep = punctual_charges(rp=rp, n=nq)