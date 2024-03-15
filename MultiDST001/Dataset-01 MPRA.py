from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import sig_index_plot
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 
from utils.visualization import fire_hist

import pandas as pd
import numpy as np

from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.common_indices import common_indices

from simulation_functions import simulation_01
from simulation_functions import confmat
from simulation_functions import seq_test
from simulation_functions import DSTmulti_testing
from functions import multi_DST

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

### Weighted MPRA
random_ind = df.loc[df['Type'] == "Random"].index
rand_ind = random_ind
CRE_ind = df.loc[df['Type'] == "CRE"].index

CRE_ind = CRE_ind.tolist()
random_ind = random_ind.tolist()

CRE_ind_og = CRE_ind
rand_ind_og = rand_ind

P_MPRA = df['p.value.MPRA']
P_STARR = df['p.value.STARR']

###################################### For the full dataset #################################

import matplotlib.pyplot as plt
p_valuesMPRA = P_MPRA.values.astype(float)
p_valuesSTARR = P_STARR.values.astype(float)

p_values = p_valuesMPRA
p_values = p_valuesSTARR

# Initialize an empty dictionary with empty lists for each column
df_sigp_dict = {
"Uncorrected": [],
"Bonferroni":[],
"Holm": [],
"SGoF": [],
"BH": [],
"BY": [],
"Q-value":[],
"pi0 estimate": []
}
df_sigp = pd.DataFrame(df_sigp_dict)
og_p_values = p_values
rejections = []
###################### Try 01 - Applying the methods #################################

#for i in range(num_iter):
results = multi_DST(p_values, alpha=0.05, weights=False)
print(results)   

sig_uncorrected = results["Uncorrected"]
sig_bonf_p = results["Bonferroni"]
sig_holm_p = results["Holm"]
sig_sgof_p = results["SGoF"]
sig_bh_p = results["BH"]
sig_by_p = results["BY"]
sig_q = results["Q-value"]
pi0_est = results["pi0 estimate"]

# Cut off constraint
total = len(p_values)
cutoff = (1-pi0_est[0])*total
print(cutoff)

p_sig_dict = {
"Uncorrected": len(sig_uncorrected),
"Bonferroni": len(sig_bonf_p),
"Holm": len(sig_holm_p),
"SGoF": len(sig_sgof_p),
"BH": len(sig_bh_p),
"BY": len(sig_by_p),
"Q-value": len(sig_q),
"pi0 estimate": pi0_est,
"cutoff": cutoff,
"Total":total
}
p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

# Concatenate the p_values_df DataFrame with your existing DataFrame
df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
df_sigp

CRE_ind = CRE_ind.tolist()
random_ind = random_ind.tolist()

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'Random']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q, random_ind]

# CRE_p = [(i,p_values[i]) for (i,val) in enumerate(CRE_ind)]
# Rand_p = [(i,p_values[i]) for (i,val) in enumerate(random_ind)]

# CRE_ind = list(map(lambda x: x[0], CRE_p))
# rand_ind =  list(map(lambda x: x[0], Rand_p))

fire_hist(p_values, CRE_ind, rand_ind, title="Histogram of CRE and Random - STARR ",col1 = 'skyblue', col2 = 'purple',left='CRE',right='Random')


plot_heatmap(methods, sig_indices, title=f"Significant index plot for STARR p-values")

sig_uncCRE = [p for p in sig_uncorrected if p in CRE_ind]
sig_uncRand = [p for p in sig_uncorrected if p in rand_ind]

sig_bonfCRE = [p for p in sig_bonf_p if p in CRE_ind]
sig_bonfRand = [p for p in sig_bonf_p if p in rand_ind]

sig_holmCRE = [p for p in sig_holm_p if p in CRE_ind]
sig_holmRand = [p for p in sig_holm_p if p in rand_ind]

sig_SGoFCRE = [p for p in sig_sgof_p if p in CRE_ind]
sig_SGoFRand = [p for p in sig_sgof_p if p in rand_ind]

sig_BHCRE = [p for p in sig_bh_p if p in CRE_ind]
sig_BHRand = [p for p in sig_bh_p if p in rand_ind]

sig_BYCRE = [p for p in sig_by_p if p in CRE_ind]
sig_BYRand = [p for p in sig_by_p if p in rand_ind]

sig_QCRE = [p for p in sig_q if p in CRE_ind]
sig_QRand = [p for p in sig_q if p in rand_ind]

p_sig_dict = {
"Uncorrected": len(sig_uncCRE),
"Bonferroni": len(sig_bonfCRE),
"Holm": len(sig_holmCRE),
"SGoF": len(sig_SGoFCRE),
"BH": len(sig_BHCRE),
"BY": len(sig_BYCRE),
"Q-value": len(sig_QCRE),
"pi0 estimate": pi0_est,
"cutoff": cutoff,
"Total":len(CRE_ind)
}
p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

# Concatenate the p_values_df DataFrame with your existing DataFrame
df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
df_sigp

p_sig_dict = {
"Uncorrected": len(sig_uncRand),
"Bonferroni": len(sig_bonfRand),
"Holm": len(sig_holmRand),
"SGoF": len(sig_SGoFRand),
"BH": len(sig_BHRand),
"BY": len(sig_BYRand),
"Q-value": len(sig_QRand),
"pi0 estimate": pi0_est,
"cutoff": cutoff,
"Total":len(rand_ind)
}
p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

# Concatenate the p_values_df DataFrame with your existing DataFrame
df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
df_sigp

len(p_values)

############################# Step 02 - Hybrid procedure #############################################

num_iter = 3
for i in range(num_iter):
      results = multi_DST(p_values, alpha=0.05, weights=False)
      print(results)   

      sig_uncorrected = results["Uncorrected"]
      sig_bonf_p = results["Bonferroni"]
      sig_holm_p = results["Holm"]
      sig_sgof_p = results["SGoF"]
      sig_bh_p = results["BH"]
      sig_by_p = results["BY"]
      sig_q = results["Q-value"]
      pi0_est = results["pi0 estimate"]

      # Cut off constraint
      total = len(p_values)
      cutoff = (1-pi0_est[0])*total
      print(cutoff)

      p_sig_dict = {
      "Uncorrected": len(sig_uncorrected),
      "Bonferroni": len(sig_bonf_p),
      "Holm": len(sig_holm_p),
      "SGoF": len(sig_sgof_p),
      "BH": len(sig_bh_p),
      "BY": len(sig_by_p),
      "Q-value": len(sig_q),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":total
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'Random']
      sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q, random_ind]

      fire_hist(og_p_values, CRE_ind, rand_ind, title=f"Histogram of CRE and Random - STARR - iteration {i} ",col1 = 'skyblue', col2 = 'purple',left='CRE',right='Random')
      plot_heatmap(methods, sig_indices, title=f"Significant index plot - STARR")

      len(p_values)
      len(CRE_ind)
      len(rand_ind)

      sig_uncCRE = [p for p in sig_uncorrected if p in CRE_ind]
      sig_uncRand = [p for p in sig_uncorrected if p in rand_ind]

      sig_bonfCRE = [p for p in sig_bonf_p if p in CRE_ind]
      sig_bonfRand = [p for p in sig_bonf_p if p in rand_ind]

      sig_holmCRE = [p for p in sig_holm_p if p in CRE_ind]
      sig_holmRand = [p for p in sig_holm_p if p in rand_ind]

      sig_SGoFCRE = [p for p in sig_sgof_p if p in CRE_ind]
      sig_SGoFRand = [p for p in sig_sgof_p if p in rand_ind]

      sig_BHCRE = [p for p in sig_bh_p if p in CRE_ind]
      sig_BHRand = [p for p in sig_bh_p if p in rand_ind]

      sig_BYCRE = [p for p in sig_by_p if p in CRE_ind]
      sig_BYRand = [p for p in sig_by_p if p in rand_ind]

      sig_QCRE = [p for p in sig_q if p in CRE_ind]
      sig_QRand = [p for p in sig_q if p in rand_ind]

      p_sig_dict = {
      "Uncorrected": len(sig_uncCRE),
      "Bonferroni": len(sig_bonfCRE),
      "Holm": len(sig_holmCRE),
      "SGoF": len(sig_SGoFCRE),
      "BH": len(sig_BHCRE),
      "BY": len(sig_BYCRE),
      "Q-value": len(sig_QCRE),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(CRE_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      p_sig_dict = {
      "Uncorrected": len(sig_uncRand),
      "Bonferroni": len(sig_bonfRand),
      "Holm": len(sig_holmRand),
      "SGoF": len(sig_SGoFRand),
      "BH": len(sig_BHRand),
      "BY": len(sig_BYRand),
      "Q-value": len(sig_QRand),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(rand_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      min_list = sig_q
      p_values = rejections = rejections + min_list

      p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
      p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
      CRE_ind2  = [i for i, val in enumerate(CRE_ind_og) if i not in rejections]
      rand_ind2 = [val for i, val in enumerate(rand_ind_og) if i not in rejections]

      p_values = p_values2
      CRE_ind = CRE_ind2 
      rand_ind = rand_ind2


df_sigp.to_csv('MultiDST/Simulation Test results/MPRA1.csv', index=False)
















len(p_values)
####################### Weighting approach #####################################
p_values = p_valuesSTARR
len(p_values)


# Initialize an empty dictionary with empty lists for each column
df_sigp_dict = {
"Uncorrected": [],
"Bonferroni":[],
"Holm": [],
"SGoF": [],
"BH": [],
"BY": [],
"Q-value":[],
"pi0 estimate": []
}
df_sigp = pd.DataFrame(df_sigp_dict)
og_p_values = p_values
rejections = []


k1_list = [0.005]
k2_list = [1.05, 1.2]

for minw in k1_list:
      for maxw in k2_list:
        # Picture it before everything :)
        # fire_hist(p_values, fire_index, nonfire_index,title=fr"Histogram of p-values (Unweighted)",col1 = 'skyblue', col2 = 'greenyellow')
        if minw=="--" and maxw=="--":
            p_values = og_p_values
        elif minw=="--" or maxw=="--":
            continue
        else:
          p_values = og_p_values
          l0,l1,l2,l3,l4,l5,l6,l7 = common_indices(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q)
          weighted_p = weighted_p_list(p_values,weights="multi",l0=l0,l1=l1,l2=l2,l3=l3,l4=l4,l5=l5,l6=l6, max_weight = maxw, min_weight=minw)
          p_values = weighted_p[1]

      results = multi_DST(p_values, alpha=0.05, weights=False)
      print(results)   

      sig_uncorrected = results["Uncorrected"]
      sig_bonf_p = results["Bonferroni"]
      sig_holm_p = results["Holm"]
      sig_sgof_p = results["SGoF"]
      sig_bh_p = results["BH"]
      sig_by_p = results["BY"]
      sig_q = results["Q-value"]
      pi0_est = results["pi0 estimate"]

      # Cut off constraint
      total = len(p_values)
      cutoff = (1-pi0_est[0])*total
      print(cutoff)

      p_sig_dict = {
      "Uncorrected": len(sig_uncorrected),
      "Bonferroni": len(sig_bonf_p),
      "Holm": len(sig_holm_p),
      "SGoF": len(sig_sgof_p),
      "BH": len(sig_bh_p),
      "BY": len(sig_by_p),
      "Q-value": len(sig_q),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":total
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'Random']
      sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q, random_ind]

      fire_hist(p_values, CRE_ind, rand_ind, title=fr"Histogram under $k_1$={minw} & $k_2$={maxw}",col1 = 'skyblue', col2 = 'purple',left='CRE',right='Random')
      plot_heatmap(methods, sig_indices, title=fr"Significant index plot under $k_1$={minw} & $k_2$={maxw}")

      len(p_values)
      len(CRE_ind)
      len(rand_ind)

      sig_uncCRE = [p for p in sig_uncorrected if p in CRE_ind]
      sig_uncRand = [p for p in sig_uncorrected if p in rand_ind]

      sig_bonfCRE = [p for p in sig_bonf_p if p in CRE_ind]
      sig_bonfRand = [p for p in sig_bonf_p if p in rand_ind]

      sig_holmCRE = [p for p in sig_holm_p if p in CRE_ind]
      sig_holmRand = [p for p in sig_holm_p if p in rand_ind]

      sig_SGoFCRE = [p for p in sig_sgof_p if p in CRE_ind]
      sig_SGoFRand = [p for p in sig_sgof_p if p in rand_ind]

      sig_BHCRE = [p for p in sig_bh_p if p in CRE_ind]
      sig_BHRand = [p for p in sig_bh_p if p in rand_ind]

      sig_BYCRE = [p for p in sig_by_p if p in CRE_ind]
      sig_BYRand = [p for p in sig_by_p if p in rand_ind]

      sig_QCRE = [p for p in sig_q if p in CRE_ind]
      sig_QRand = [p for p in sig_q if p in rand_ind]

      p_sig_dict = {
      "K1":minw,
      "K2":maxw,
      "Uncorrected": len(sig_uncCRE),
      "Bonferroni": len(sig_bonfCRE),
      "Holm": len(sig_holmCRE),
      "SGoF": len(sig_SGoFCRE),
      "BH": len(sig_BHCRE),
      "BY": len(sig_BYCRE),
      "Q-value": len(sig_QCRE),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(CRE_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      p_sig_dict = {
      "K1":minw,
      "K2":maxw,
      "Uncorrected": len(sig_uncRand),
      "Bonferroni": len(sig_bonfRand),
      "Holm": len(sig_holmRand),
      "SGoF": len(sig_SGoFRand),
      "BH": len(sig_BHRand),
      "BY": len(sig_BYRand),
      "Q-value": len(sig_QRand),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(rand_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

min_list = sig_q
p_values = rejections = rejections + min_list

p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
CRE_ind2  = [i for i, val in enumerate(CRE_ind_og) if i not in rejections]
rand_ind2 = [val for i, val in enumerate(rand_ind_og) if i not in rejections]

p_values = p_values2
CRE_ind = CRE_ind2 
rand_ind = rand_ind2


####################### Part 3 - IHW Approach ###########################################


og_p_values = p_valuesSTARR
p_values = og_p_values  
len(p_values)

num_iter = 3
for i in range(num_iter):
      results = multi_DST(p_values, alpha=0.05, weights=False)
      print(results)   

      sig_uncorrected = results["Uncorrected"]
      sig_bonf_p = results["Bonferroni"]
      sig_holm_p = results["Holm"]
      sig_sgof_p = results["SGoF"]
      sig_bh_p = results["BH"]
      sig_by_p = results["BY"]
      sig_q = results["Q-value"]
      pi0_est = results["pi0 estimate"]

      # Cut off constraint
      total = len(p_values)
      cutoff = (1-pi0_est[0])*total
      print(cutoff)

      p_sig_dict = {
      "Uncorrected": len(sig_uncorrected),
      "Bonferroni": len(sig_bonf_p),
      "Holm": len(sig_holm_p),
      "SGoF": len(sig_sgof_p),
      "BH": len(sig_bh_p),
      "BY": len(sig_by_p),
      "Q-value": len(sig_q),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":total
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'Random']
      sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q, random_ind]

      fire_hist(p_values, CRE_ind, rand_ind, title=f"Histogram of CRE and Random - STARR - iteration {i} ",col1 = 'skyblue', col2 = 'purple',left='CRE',right='Random')
      plot_heatmap(methods, sig_indices, title=f"Significant index plot - STARR")

      len(p_values)
      len(CRE_ind)
      len(rand_ind)

      sig_uncCRE = [p for p in sig_uncorrected if p in CRE_ind]
      sig_uncRand = [p for p in sig_uncorrected if p in rand_ind]

      sig_bonfCRE = [p for p in sig_bonf_p if p in CRE_ind]
      sig_bonfRand = [p for p in sig_bonf_p if p in rand_ind]

      sig_holmCRE = [p for p in sig_holm_p if p in CRE_ind]
      sig_holmRand = [p for p in sig_holm_p if p in rand_ind]

      sig_SGoFCRE = [p for p in sig_sgof_p if p in CRE_ind]
      sig_SGoFRand = [p for p in sig_sgof_p if p in rand_ind]

      sig_BHCRE = [p for p in sig_bh_p if p in CRE_ind]
      sig_BHRand = [p for p in sig_bh_p if p in rand_ind]

      sig_BYCRE = [p for p in sig_by_p if p in CRE_ind]
      sig_BYRand = [p for p in sig_by_p if p in rand_ind]

      sig_QCRE = [p for p in sig_q if p in CRE_ind]
      sig_QRand = [p for p in sig_q if p in rand_ind]

      p_sig_dict = {
      "Uncorrected": len(sig_uncCRE),
      "Bonferroni": len(sig_bonfCRE),
      "Holm": len(sig_holmCRE),
      "SGoF": len(sig_SGoFCRE),
      "BH": len(sig_BHCRE),
      "BY": len(sig_BYCRE),
      "Q-value": len(sig_QCRE),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(CRE_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      p_sig_dict = {
      "Uncorrected": len(sig_uncRand),
      "Bonferroni": len(sig_bonfRand),
      "Holm": len(sig_holmRand),
      "SGoF": len(sig_SGoFRand),
      "BH": len(sig_BHRand),
      "BY": len(sig_BYRand),
      "Q-value": len(sig_QRand),
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":len(rand_ind)
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      min_list = sig_q
      p_values = rejections = rejections + min_list

      p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
      p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
      CRE_ind2  = [i for i, val in enumerate(CRE_ind_og) if i not in rejections]
      rand_ind2 = [val for i, val in enumerate(rand_ind_og) if i not in rejections]

      p_values = p_values2
      CRE_ind = CRE_ind2 
      rand_ind = rand_ind2


df_sigp.to_csv('MultiDST/Simulation Test results/MPRA1.csv', index=False)


