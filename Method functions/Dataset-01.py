from A01_weighting import weighted_p_list
from A02_FWER1_bonferroni import bonferroni
from A02_FWER3_holm import holm
from A02_FWER5_sgof import sgof_test
from A03_FDR1_bh import bh_method
from A03_FDR2_qval import q_value
from A03_FDR3_BY import BY_method

from visualization import draw_histogram

import pandas as pd

# Opening the file in read mode
 
with open('MultiDST/MultiDST - Real Dataset/experimental_pvalues.txt', 'r') as file:
    # Read the entire content of the file
    content = file.read()

# Split the content into lines
lines = content.splitlines()

# Split each line into columns (assuming a space delimiter)
table_data = [line.split() for line in lines]
colnames = lines[0].split()

# Create a DataFrame using pandas
df = pd.DataFrame(table_data[1:], columns=colnames)

P_MPRA = df['p.value.MPRA']
P_STARR = df['p.value.STARR']

########################### For the full dataset #################################

p_values = P_MPRA.values.astype(float)
draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 01 - Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

############################# Shortlisting p values - Remove Bonferroni ########################

p_values2 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_bonf_p]
p_values = p_values2

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))


############################# Iter 3 - Shortlisting p values ########################

p_values3 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_q]
p_values = p_values3

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

############################# Iter 4 - Shortlisting p values ########################

p_values4 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_sgof_p]
p_values = p_values4

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))


############################# Iter 5 - Shortlisting p values ########################

p_values5 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_sgof_p]
p_values = p_values5

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))


# Now that all the methods have shortlisted the p-values, take a look at the uncorrected ones
[p_values5[i] for i in sig_index]

#As we can see, most of the values are close to the cut off of 0.05, hence the methods have done their bit