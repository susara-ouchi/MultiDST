import numpy as np
from itertools import combinations

def fwe_minp(pvalues, distr, combine='tippett'):
    # Choose the combining function
    funcs = {'fisher': fisher, 'liptak': liptak, 'tippett': tippett}
    if combine not in funcs:
        raise ValueError(combine + " is not a valid combining function.")
    
    p = len(pvalues)
    
    if p < 2:
        raise ValueError("Nothing to combine!")
        
    if distr.shape[1] != p:
        raise ValueError("Different number of p-values and null distributions")
    
    combn_func = funcs[combine]
    
    # Order the p-values
    p_ord = np.sort(pvalues)
    perm_pvalues = np.apply_along_axis(pvalue_distr, 1, distr, alternative='greater')
    perm_pvalues_ord = perm_pvalues[:, np.argsort(pvalues)]
    
    # Step down tree of combined hypotheses
    p_ris = np.full(p, np.nan)
    combined_stats = np.apply_along_axis(combn_func, 1, perm_pvalues_ord)
    obs_stat = combn_func(p_ord)
    p_ris[0] = t2p(obs_stat, combined_stats, alternative='greater')
    
    if p > 2:
        for j in range(1, p - 1):
            obs_stat = combn_func(p_ord[j:p])
            combined_stats = np.apply_along_axis(combn_func, 1, perm_pvalues_ord[:, j:p])
            p_ris[j] = max(t2p(obs_stat, combined_stats, alternative='greater'), p_ris[j - 1])
    
    p_ris[p - 1] = max(p_ord[-1], p_ris[p - 2])
    
    p_ris[np.argsort(pvalues)] = p_ris
    return p_ris

def fisher(pvalues):
    return -2 * np.sum(np.log(pvalues))

def liptak(pvalues):
    return np.sum(-np.log(pvalues))

def tippett(pvalues):
    return np.max(-np.log(pvalues))

def pvalue_distr(distr, alternative='greater'):
    if alternative == 'greater':
        return np.mean(distr > 0)
    elif alternative == 'less':
        return np.mean(distr < 0)
    elif alternative == 'two-sided':
        return np.mean(np.abs(distr) > np.abs(0))
    else:
        raise ValueError("Invalid alternative")
    
def t2p(obs_stat, combined_stats, alternative='greater'):
    if alternative == 'greater':
        return np.mean(combined_stats >= obs_stat)
    elif alternative == 'less':
        return np.mean(combined_stats <= obs_stat)
    elif alternative == 'two-sided':
        return np.mean(np.abs(combined_stats) >= np.abs(obs_stat))
    else:
        raise ValueError("Invalid alternative")


pvalues = np.array([0.02, 0.05, 0.1, 0.005])
np.random.seed(123)
distr = np.random.normal(loc=0, scale=1, size=(100, 4))

# Apply the function
adjusted_pvalues = fwe_minp(pvalues, distr, combine='tippett')

# Print the adjusted p-values
print(adjusted_pvalues)
