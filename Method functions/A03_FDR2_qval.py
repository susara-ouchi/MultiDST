import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate

#loading p 
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

# Function to calculate adjusted q_values
def q_adj_func(p_values, pi0=None, m=None, verbose=True):
    """
    Estimates q-values from p-values

    Args
    =====
    p_values: list
        List of p-values
    m: int, optional
        Number of tests. If not specified m = len(p_values)
    verbose: bool, optional
        Print verbose messages? (default False)
    pi0: float or None, optional
        If None, it's estimated as suggested in Storey and Tibshirani, 2003.
        For most GWAS this is not necessary, since pi0 is extremely likely to be 1.
    """
    #assert(all(0 <= p <= 1 for p in p_values)), "p-values should be between 0 and 1"

    if m is None:
        m = float(len(p_values))
    else:
        m *= 1.0

    if len(p_values) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(p_values > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(0 <= pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = sorted(range(len(p_values)), key=lambda k: p_values[k])
    p_values = [p_values[i] for i in p_ordered]
    q_adj = [0] * len(p_values)
    q_adj[-1] = min(pi0 * p_values[-1], 1.0)
    for i in range(len(p_values)-2, -1, -1):
        q_adj[i] = min(pi0 * m * p_values[i] /(i+1), q_adj[i + 1])

    # Re-ordering the q-values
    q_adj_temp = q_adj.copy()
    q_adj = [0] * len(p_values)
    for i, idx in enumerate(p_ordered):
        q_adj[idx] = q_adj_temp[i]

    return q_adj

#Define function for Q value Procedure 
def q_value(p_values, alpha=0.05, weights = True):
    m = len(p_values)
    if weights == True:
        p_values = weighted_p

        adj_p = q_adj_func(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        # Q value correction
        adj_p = q_adj_func(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    return adj_p, sig_index

p_values = [0.0005279804659690256, 0.05107595122255753, 0.005380747546894805, 0.008293070676726721, 0.015261930084251897, 0.09399292181095295, 0.04916062506442831, 0.08455877419751781, 0.026622720150619863, 0.060671184302609794, 0.014792473316734833, 0.029279038132892888, 0.039948575984906864, 0.05455860141093238, 0.06495646577203158, 0.01393407242591071, 0.06592036470024257, 0.03370049417508525, 0.08285377432610773, 0.055087308119778314]
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]
q
sig_q
