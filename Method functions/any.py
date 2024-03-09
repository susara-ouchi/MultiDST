import numpy as np
from scipy.stats import norm

def SGoF(u, alpha=0.05, gamma=0.05):
    def sgof(u, alpha=0.05, gamma=0.05):
        v = np.array(u <= gamma, dtype=int)
        n = len(v)

        SGoF = max(min(np.floor(n * (np.mean(v) - gamma) - n * np.sqrt(np.mean(v) * (1 - np.mean(v)) / n) * norm.ppf(1 - alpha) + 1), sum(np.array(np.arange(1, n+1) <= n * (np.mean(v) - gamma) - n * np.sqrt(np.mean(v) * (1 - np.mean(v)) / n) * norm.ppf(1 - alpha), dtype=int)) + 1), 0)

        su = np.sort(u)
        jj = np.where(u == 1)[0]
        if len(jj) != 0:
            pi0 = 1
        else:
            pi0 = min((-1 / n) * np.sum(np.log(1 - u)), 1)

        if SGoF == 0:
            FDR_S = 0
        else:
            FDR_S = round((pi0 * su[int(SGoF)]) / (np.mean(u <= su[int(SGoF)])), 4)

        Nu1 = np.maximum(n * (np.arange(1, n+1)/n - su) - np.sqrt(n * np.arange(1, n+1)/n * (1 - np.arange(1, n+1)/n)) * norm.ppf(1 - su/n) + 1, 0)

        Nu2 = np.zeros(n)
        for i in range(n):
            Nu2[i] = max(np.where(np.array(np.arange(1, n+1) <= np.ceil(n * np.mean(u <= su[i]))), 0))

        Nu = np.minimum(Nu1, Nu2)

        Nmax = max(Nu[~np.isnan(Nu)])

        a_p = np.ones(n)
        for i in range(Nmax.astype(int)):
            a_p[i] = min(su[np.array(np.arange(1, n+1) <= Nu[i], dtype=int)], na_rm=True)

        return {'Rejections': SGoF, 'FDR': min(FDR_S, 1), 'Adjusted_pvalues': a_p}

    u = np.array(u)
    res = sgof(u, alpha, gamma)
    res['data'] = np.sort(u)
    res['alpha'] = alpha
    res['gamma'] = gamma
    return res


# Example data
u = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

# Applying SGoF method
result = SGoF(u)

# Displaying the result
print("Rejections:", result['Rejections'])
print("FDR:", result['FDR'])
print("Adjusted p-values:", result['Adjusted_pvalues'])
