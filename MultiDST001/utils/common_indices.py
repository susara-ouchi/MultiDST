from collections import Counter

from collections import Counter
from scipy.optimize import minimize


from collections import Counter

def common_indices(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q):
    # Combine all indices into a single list
    all_indices = sig_bonf_p + sig_holm_p + sig_sgof_p + sig_bh_p + sig_by_p + sig_q
    p_values_ind = [i for i in range(len(p_values))]

    # Count occurrences of each index across all lists
    index_counts = Counter(all_indices)

    # Initialize dictionaries to store indices
    index_dict = {i: [] for i in range(len(p_values))}
    l0, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], []

    # Iterate over all indices and assign them to the corresponding lists
    for index, count in index_counts.items():
        index_dict[index] = [(i, p_values_ind[i]) for i in range(len(p_values_ind)) if p_values_ind[i] == index]
        if count == 6:
            l6.extend(index_dict[index])
        elif count == 5:
            l5.extend(index_dict[index])
        elif count == 4:
            l4.extend(index_dict[index])
        elif count == 3:
            l3.extend(index_dict[index])
        elif count == 2:
            l2.extend(index_dict[index])
        elif count == 1:
            l1.extend(index_dict[index])

    # Find indices not present in any of the lists
    for i in range(len(p_values)):
        if i not in all_indices:
            l0.append((i, p_values_ind[i]))

    l0_list = [item[0] for item in l0]
    l1_list = [item[0] for item in l1]
    l2_list = [item[0] for item in l2]
    l3_list = [item[0] for item in l3]
    l4_list = [item[0] for item in l4]
    l5_list = [item[0] for item in l5]
    l6_list = [item[0] for item in l6]

    return l0_list, l1_list, l2_list, l3_list, l4_list, l5_list, l6_list, index_dict
