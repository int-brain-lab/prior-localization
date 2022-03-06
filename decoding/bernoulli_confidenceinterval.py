#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 21:53:54 2022

@author: bensonb
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
# import 


def PNsuccess_gmuN(Nsuccess,mu,N):
    return math.comb(N,Nsuccess) * (mu**Nsuccess) * ((1-mu)**(N-Nsuccess))

def PWilson(Nsuccess,mu,N):
    p_hat = Nsuccess/N
    sig_N = np.sqrt((mu*(1-mu))/N)
    
    z = (mu - p_hat)/sig_N
    return scipy.stats.norm.pdf(z)

def Bernoulli_ci(Nsuccess,N,ci_lower=0.025,ci_upper=0.975):
    '''
    
    three digits of accuracy
    ----------
    Nsuccess : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    ci : tuple
        (lower bound, upper bound)

    '''
    
    mus = np.linspace(0,1,1001)
    Ps = np.array([PNsuccess_gmuN(Nsuccess, mu, N) for mu in mus])
    ci = (scipy.stats.percentileofscore(np.cumsum(Ps)/np.sum(Ps), 
                                        ci_lower, kind='mean')/100.0,
          scipy.stats.percentileofscore(np.cumsum(Ps)/np.sum(Ps), 
                                        ci_upper, kind='mean')/100.0)
    return ci

N = 200
Nsuccess = 20
mus = np.linspace(0,1,1001)
dmu = mus[1] - mus[0]
Ps = [PNsuccess_gmuN(Nsuccess, mu, N) for mu in mus]
Ps = np.array(Ps)/(np.sum(Ps)*dmu)
Ps_Wilson = [PWilson(Nsuccess, mu, N) for mu in mus]
Ps_Wilson = np.array(Ps_Wilson)/(np.sum(Ps_Wilson)*dmu)

plt.plot(mus,np.cumsum(Ps)*dmu)
ci = (scipy.stats.percentileofscore(np.cumsum(Ps)*dmu, 
                                    0.975, kind='mean')/100.0,
      scipy.stats.percentileofscore(np.cumsum(Ps)*dmu, 
                                    0.025, kind='mean')/100.0)

plt.plot(mus,np.cumsum(Ps_Wilson)*dmu)
ci_W = (scipy.stats.percentileofscore(np.cumsum(Ps_Wilson)*dmu, 
                                      0.975, kind='mean')/100.0,
      scipy.stats.percentileofscore(np.cumsum(Ps_Wilson)*dmu, 
                                    0.025, kind='mean')/100.0)
plt.show()
