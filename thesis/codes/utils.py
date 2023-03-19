import numpy as np
import scipy as sp
from scipy import integrate
import math
#from numba import jit
#import numexpr as ne
from tqdm import tqdm
import networkx as nx
import os
#const
pi=np.pi

def thetas2orerparameter(thetas):
    '''
    Input: thetas(ndarray)
    Output: x,y-value of order parameter(list)
    '''
    rcos,rsin=np.mean(np.cos(thetas)),np.mean(np.sin(thetas))
    return [rcos,rsin]

def thetas2np_op(thetas):
    z=thetas2orerparameter(thetas)
    radius=np.sqrt(z[0]**2+z[1]**2)
    return radius

def kuramoto(thetas,omegas,K):
    #rcos,rsin=thetas2orerparameter(thetas)
    cos,sin=np.cos(thetas),np.sin(thetas)
    rcos,rsin=np.mean(cos),np.mean(sin)
    v=omegas+K*(rsin*cos-rcos*sin)
    return v

def kuramoto_graph(thetas,omegas,W,size,K):
    cos,sin=np.cos(thetas),np.sin(thetas)
    v=omegas+(K/size)*(cos*np.dot(W,sin)-sin*np.dot(W,cos))
    return v

def kuramoto_sparse(thetas,omegas,W,size,K):
    cos,sin=np.cos(thetas),np.sin(thetas)
    v=omegas+(K/size)*(cos*W.dot(sin)-sin*W.dot(cos))
    return v
    
def kuramoto_sparse2(thetas,omegas,W,size,K,a):
    cos,sin=np.cos(thetas),np.sin(thetas)
    cos2,sin2=np.cos(2*thetas),np.sin(2*thetas)
    v=omegas+(K/size)*(cos*W.dot(sin)-sin*W.dot(cos))+(a*K/size)*(cos2*W.dot(sin2)-sin2*W.dot(cos2))
    return v

def kuramoto_ne(thetas,omegas,K):
    c,s=ne.evaluate('cos(thetas)'),ne.evaluate('sin(thetas)')
    rc,rs=np.mean(c),np.mean(s)
    v=ne.evaluate('omegas+K*(rs*c-rc*s)')
    return v

def lorentz_rvs(size,gamma=1,seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    xs=np.random.rand(size)
    omegas=gamma*np.tan(pi*(xs-0.5))
    return omegas
    
def lorentz_n_rvs(size,n,gamma=1,seed=None):
    def f(x):
        return n*np.sin(pi/(2*n))*gamma**(2*n-1)/(pi*(x**(2*n)+gamma**(2*n)))
    y_max=f(0)
    if seed is not None:
        np.random.seed(seed=seed)
    counter=0
    omegas=[]
    while counter<size:
        x=20*np.random.rand()-10
        y=y_max*np.random.rand()
        if y<f(x):
            omegas.append(x)
            counter+=1
    return np.array(omegas)

def gauss_n_rvs(size,n,gamma=1,seed=None):
    def f(x):
        return n*gamma*np.exp(-(gamma*x)**(2*n))/math.gamma(1/(2*n))
    y_max=f(0)
    if seed is not None:
        np.random.seed(seed=seed)
    counter=0
    omegas=[]
    while counter<size:
        x=20*np.random.rand()-10
        y=y_max*np.random.rand()
        if y<f(x):
            omegas.append(x)
            counter+=1
    return np.array(omegas)
    
def lorentz_inf_rvs(size,seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    xs=2*np.random.rand(size)-1
    return xs

def rungekutta(function,xs,args,t_max,dt):
    repeat=int(t_max/dt)
    rs=[]
    for _ in range(repeat):
        k1=function(xs,*args)
        k2=function(xs+0.5*dt*k1,*args)
        k3=function(xs+0.5*dt*k2,*args)
        k4=function(xs+dt*k3,*args)
        xs+=(k1+2*k2+2*k3+k4)*dt/6
        rs.append(thetas2np_op(xs))
    return [np.array(rs),xs]
    
def heun(function,xs,args,t_max,dt):
    repeat=int(t_max/dt)
    rs=[]
    for _ in range(repeat):
        k1=function(xs,*args)
        k2=function(xs+dt*k1,*args)
        xs+=(k1+k2)*dt*0.5
        rs.append(thetas2np_op(xs))
    return [np.array(rs),xs]

def gauss_n_Orderparameter(K,n,gamma=1.0):
    def gauss_n(x):
        return n*gamma*np.exp(-(gamma*x)**(2*n))/math.gamma(1/(2*n))
    def hisekibun(theta,x):
        return np.cos(theta)**2*gauss_n(K*x*np.sin(theta))
    def f(x):
        integral=integrate.quad(lambda y:hisekibun(y,x),-pi/2,pi/2)[0]
        return K*integral-1
    Kc=2/pi/gauss_n(0)
    if K<Kc:
        return 0
    else:
        r_min,r_max=0,1
        eps=1
        counter=0
        while np.abs(eps)>10**(-6) and counter<100:
            counter+=1
            f1,eps=f(r_min),f((r_min+r_max)*0.5)
            if f1*eps<0:
                r_max=(r_min+r_max)*0.5
            else:
                r_min=(r_min+r_max)*0.5
        return (r_min+r_max)*0.5


def gauss_inf_Orderparameter(K,n,gamma=1.0):
    def gauss_inf(x):
        return 0.5/gamma if np.abs(x)<gamma else 0
    def hisekibun(theta,x):
        return np.cos(theta)**2*gauss_inf(K*x*np.sin(theta))
    def f(x):
        integral=integrate.quad(lambda y:hisekibun(y,x),-pi/2,pi/2)[0]
        return K*integral-1
    Kc=2/pi/gauss_inf(0)
    if K<Kc:
        return 0
    else:
        r_min,r_max=0,1
        eps=1
        counter=0
        while np.abs(eps)>10**(-6) and counter<100:
            counter+=1
            f1,eps=f(r_min),f((r_min+r_max)*0.5)
            if f1*eps<0:
                r_max=(r_min+r_max)*0.5
            else:
                r_min=(r_min+r_max)*0.5
        return (r_min+r_max)*0.5