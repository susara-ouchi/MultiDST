import numpy as np

def sdminp(res_ob, res_perm):
    m = res_perm.shape[0]  # initialization
    B = res_perm.shape[1]  # initialization

    # Step 0: calculate the raw-pvalue
    print("Step 0: calculate the raw-pvalue")
    rawp = np.sum(res_perm >= res_ob, axis=1) / B
    order_rawp = np.argsort(rawp)
    order_rawp_trans = np.argsort(np.arange(1, m + 1))

    rawp = rawp[order_rawp]
    Tm = res_perm[order_rawp]  # initialization
    Pm = np.zeros((m, B))  # initialization
    Qm = np.zeros((m + 1, B))  # initialization
    Pm[:, B - 1] = 1
    Qm[-1] = 1

    # Step 1: calculate the T matrix
    print("Step 1-1: calculate the T matrix")
    orderTm = np.flip(np.argsort(Tm, axis=1), axis=1)
    trans_orderTm = np.argsort(np.arange(1, orderTm.shape[1] + 1))

    for i in range(m):
        Tm[i, :] = Tm[i, orderTm[i, :]]

    print("Step 1-2: calculate the P matrix")
    for j in range(B - 2, -1, -1):
        Pm[:, j] = Pm[:, j + 1]
        Pm[Tm[:, j] != Tm[:, j + 1], j] = j / B

    for i in range(m):
        Pm[i, :] = Pm[i, trans_orderTm]

    # Step 2: calculate the Q matrix
    print("Step 2: calculate the Q matrix")
    for i in range(m - 1, -1, -1):
        Qm[i, :] = np.minimum(Qm[i + 1, :], Pm[i, :])

    # Step 3
    print("Step 3")
    ep = np.sum(Qm[1:m + 1] <= rawp[:, None], axis=1) / B

    # Step 6
    print("Step 6")
    for i in range(1, m):
        ep[i] = max(ep[i - 1:i + 1])

    # Sort result to original order
    print("Sort result to original order")
    ep = ep[np.argsort(order_rawp_trans)]
    return ep


def find_rank(target, ruler):
    m = len(ruler)
    res = np.zeros((m, 1))
    cur_rank = 1

    for i in range(m):
        while cur_rank <= m and target[cur_rank - 1] >= ruler[i]:
            cur_rank += 1
        res[i] = cur_rank - 1

    return res


def sdminp_fdr(res_ob, res_perm, tao0=0.2):
    m = res_perm.shape[0]  # initialization
    B = res_perm.shape[1]  # initialization

    # Step 0: calculate the ob-tao
    print("Step 0: calculate the ob-tao")
    tao = res_ob
    W0 = np.sum(res_ob <= tao0)

    order_tao = np.argsort(tao)[::-1]
    order_tao_trans = np.argsort(np.arange(1, m + 1))

    tao = tao[order_tao]

    R = np.zeros((m, 1))  # initialization
    Rm = np.zeros((m, B))  # initialization
    qval = np.zeros((m, 1))  # initialization
    adj_pval = np.zeros((m, 1))  # initialization

    # Step 1: calculate the R matrix
    print("Step 1-1: calculate the R matrix")
    res_obS = np.sort(res_ob)[::-1]
    res_permS = np.sort(res_perm, axis=0)[::-1]

    R = find_rank(res_obS, tao)

    for j in range(B):
        Rm[:, j] = find_rank(res_permS[:, j], tao)

    print("Step 1-2: calculate the W0 vector")
    W0m = np.sum(res_perm <= tao0, axis=1)

    # Step 2: calculate meanR, meanI meanW
    print("Step 2: calculate meanR, meanI meanW")
    meanR = np.mean(Rm, axis=1)
    meanI = np.mean(Rm > 0, axis=1)
    meanW0 = np.mean(W0)

    # Step 3: calculate pFDR and FDR
    print("Step 3: calculate pFDR and FDR")
    pFDR = W0 * meanR / (meanW0 * ((R + 1) + abs(R - 1)) * meanI / 2)
    FDR = W0 * meanR / (meanW0 * ((R + 1) + abs(R - 1)) / 2)

    # Step 4: enforcing monotonicity
    print("Step 4 enforcing monotonicity")
    qval[-1] = pFDR[-1]
    adj_pval[-1] = FDR[-1]

    for i in range(m - 2, -1, -1):
        qval[i] = min(qval[i + 1], pFDR[i])
        adj_pval[i] = min(adj_pval[i + 1], FDR[i])

    # Sort result to original order
    print("Sort result to original order")
    qval = qval[order_tao_trans]
    adj_pval = adj_pval[order_tao_trans]
    return np.hstack((adj_pval, qval))


def rawp_cal(res_ob, res_perm):
    rawp = np.sum(res_perm >= res_ob[:, None], axis=1) / res_perm.shape[1]
    return rawp


# Example usage
ob = np.random.normal(2, 2, 10)
perm = np.random.normal(2, 2, (10, 10))

sdminp(ob, perm)
sdminp_fdr(ob,perm)

import numpy as np



# Generate hypothetical initial p-values for 100 genes (between 0 and 1)
initial_p_values = np.random.uniform(0, 1, 100)

# Generate permuted data for 100 genes (let's assume 100 permutations for simplicity)
num_permutations = 100
permuted_data = np.random.uniform(0, 1, (100, num_permutations))

# Calculate adjusted p-values using Step Down minP method
adjusted_p_values = sdminp(initial_p_values, permuted_data)

# Print the adjusted p-values
print("Adjusted p-values:", adjusted_p_values)



#####################################################

import numpy as np

# Step 1: Data Preparation
# Assume gene expression data for control and treatment groups are stored in arrays control_data and treatment_data

# Step 2: Initial Analysis
# Perform a statistical test (e.g., t-test) for each gene to compare expression levels between groups
initial_p_values = perform_statistical_test(control_data, treatment_data)

# Step 3: Permutation Procedure
num_permutations = 1000  # Number of permutations
permuted_p_values = []
for i in range(num_permutations):
    # Generate permuted dataset by shuffling sample labels
    permuted_control_data = shuffle_labels(control_data)
    permuted_treatment_data = shuffle_labels(treatment_data)
    
    # Perform statistical test on permuted dataset
    permuted_p_values.append(perform_statistical_test(permuted_control_data, permuted_treatment_data))

# Step 4: Multiple Testing Correction
adjusted_p_values = sdminp(initial_p_values, np.array(permuted_p_values))

# Step 5: Interpretation
# Identify significant genes based on adjusted p-values
significant_genes = identify_significant_genes(adjusted_p_values)
