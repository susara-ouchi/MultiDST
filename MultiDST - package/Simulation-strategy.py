import pandas as pd
import numpy as np

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

from simulation_functions import simulation_01
from simulation_functions import confmat
from simulation_functions import seq_test
from simulation_functions import DSTmulti_testing


np.random.seed(42)
# 01
#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,2000,8000,effect=1.5,n0=15,n1=15,threshold=0.05,show_plot=False,s0=1.0,s1=1.0)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
og_p_values = p_values
fire_index_og, nonfire_index_og = fire_index, nonfire_index
p_value_fire = [p_values[i] for i in fire_index]
p_value_nonfire = [p_values[i] for i in nonfire_index]

# Observing histogram
fire_hist(p_values, fire_index, nonfire_index)

sig_fire = [p for p in significant_p if p in fire_index]
sig_nonfire = [p for p in significant_p if p in nonfire_index]
len_p_fire = len(fire_index)
len_p_nonfire = len(nonfire_index)
res = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
power,pi0 = res[0],res[1]

rejections = []
seq_test(rejections, og_p_values, p_value_fire, p_value_nonfire, threshold =0.05)

# Dataframe to store rejections 

# Initialize an empty dictionary with empty lists for each column
df_sigp_dict = {
    "Firing":[],
    "Uncorrected": [],
    "Bonferroni":[],
    "Holm": [],
    "SGoF": [],
    "BH": [],
    "BY": [],
    "Q-value":[],
    "TP":[],
    "TN":[],
    "FP":[],
    "FN":[],
    "Power":[],
    "pi0": [],
    "pi0 estimate": [],
    "cutoff": [],
    "Total": []
}
df_sigp = pd.DataFrame(df_sigp_dict)

###################### Try 01 - Applying the methods #################################
k = 1
num_iter = 10

#for i in range(num_iter):
results = DSTmulti_testing(p_values, alpha=0.05, weights=False)
print(results)   

#fire_hist(p_values, fire_index, nonfire_index, title=f"Histogram of signals - {i+1}")
res = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
TP,TN,FP,FN,power,pi0 = res
total = TP+TN+FP+FN

sig_uncorrected = results["Uncorrected"]
sig_bonf_p = results["Bonferroni"]
sig_holm_p = results["Holm"]
sig_sgof_p = results["SGoF"]
sig_bh_p = results["BH"]
sig_by_p = results["BY"]
sig_q = results["Q-value"]
pi0_est = results["pi0 estimate"]

# Cut off constraint
cutoff = (1-pi0_est[0])*total/k
print(cutoff)

p_sig_dict = {
"Firing": len_p_fire,
"Uncorrected": len(sig_uncorrected),
"Bonferroni": len(sig_bonf_p),
"Holm": len(sig_holm_p),
"SGoF": len(sig_sgof_p),
"BH": len(sig_bh_p),
"BY": len(sig_by_p),
"Q-value": len(sig_q),
"TP":TP,
"TN":TN,
"FP":FP,
"FN":FN,
"Power":power,
"pi0": pi0,
"pi0 estimate": pi0_est,
"cutoff": cutoff,
"Total":total
}
p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

# Concatenate the p_values_df DataFrame with your existing DataFrame
df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
df_sigp

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'True  Signals']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q,fire_index]
#plot_heatmap(methods, sig_indices, title=f"Significant index plot for iteration - {i+1}")

# Create a dictionary to map each list to its name
indexed_sig_indices = dict(zip(methods, sig_indices))
min_name = min(indexed_sig_indices, key=lambda x: len(indexed_sig_indices[x]))
print(f"Min elements are found in {min_name}")
low_ind = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0]
valid_indices1 = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0] 
valid_indices = [min(valid_indices1) if len(valid_indices1)>0 else min(low_ind)]
min_index = valid_indices[0]
min_list = sig_indices[min_index]
min_list = sig_sgof_p

# Create a sublist containing the values corresponding to the first 7 keys
min_list
rejections = rejections + min_list
len(rejections)

p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
fire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_fire]
nonfire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_nonfire]
len(nonfire_index2)
len(p_values2)
significant_p = min_list
sig_fire = [p for p in significant_p if p in fire_index2]
sig_nonfire = [p for p in significant_p if p in nonfire_index2]
len_p_fire = len(fire_index2)
len_p_nonfire = len(nonfire_index2)
confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)

p_values = p_values2
fire_index = fire_index2
nonfire_index = nonfire_index2
len(p_values)
df_sigp
pi0_est
df_sigp


df_sigp.to_csv('MultiDST/Simulation Test results/setting4.csv', index=False)





df_sigp_dict = {
    "Firing":[],
    "Uncorrected": [],
    "Bonferroni":[],
    "Holm": [],
    "SGoF": [],
    "BH": [],
    "BY": [],
    "Q-value":[],
    "TP":[],
    "TN":[],
    "FP":[],
    "FN":[],
    "Power":[],
    "pi0": [],
    "pi0 estimate": [],
    "cutoff": [],
    "Total": []
}
df_sigp = pd.DataFrame(df_sigp_dict)


def sequential_sim(p_values, df_sigp, method = "q"):
      import pandas as pd
      import numpy as np

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

      from simulation_functions import simulation_01
      from simulation_functions import confmat
      from simulation_functions import seq_test
      from simulation_functions import DSTmulti_testing

      # Initialize an empty dictionary with empty lists for each column


      #for i in range(num_iter):
      results = DSTmulti_testing(p_values, alpha=0.05, weights=False)
      print(results)   

      sig_uncorrected = results["Uncorrected"]
      sig_bonf_p = results["Bonferroni"]
      sig_holm_p = results["Holm"]
      sig_sgof_p = results["SGoF"]
      sig_bh_p = results["BH"]
      sig_by_p = results["BY"]
      sig_q = results["Q-value"]
      pi0_est = results["pi0 estimate"]

      if method == "bonf":
            method = sig_bonf_p
      elif method == "holm":
            method = sig_holm_p
      elif method == "sgof":
            method = sig_sgof_p
      elif method == "bh":
            method = sig_bh_p
      elif method == "by":
            method = sig_by_p
      elif method == "q":
            method = sig_q
      else:
            print("Give one of [bong,holm,sgof,bh,by,q]")

      # Cut off constraint
      cutoff = (1-pi0_est[0])*total/k
      print(cutoff)

      # Create a dictionary to map each list to its name
      indexed_sig_indices = dict(zip(methods, sig_indices))
      min_name = min(indexed_sig_indices, key=lambda x: len(indexed_sig_indices[x]))
      print(f"Min elements are found in {min_name}")
      low_ind = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0]
      valid_indices1 = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0] 
      valid_indices = [min(valid_indices1) if len(valid_indices1)>0 else min(low_ind)]
      min_index = valid_indices[0]
      min_list = sig_indices[min_index]
      min_list = method

      print("Works!")

      # Create a sublist containing the values corresponding to the first 7 keys
      min_list
      rejections = rejections + min_list
      len(rejections)

      p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
      p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
      fire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_fire]
      nonfire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_nonfire]
      len(nonfire_index2)
      len(p_values2)
      significant_p = min_list
      sig_fire = [p for p in significant_p if p in fire_index2]
      sig_nonfire = [p for p in significant_p if p in nonfire_index2]
      len_p_fire = len(fire_index2)
      len_p_nonfire = len(nonfire_index2)
      confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)

      p_values = p_values2
      fire_index = fire_index2
      nonfire_index = nonfire_index2
      len(p_values)
      df_sigp
      pi0_est

      #fire_hist(p_values, fire_index, nonfire_index, title=f"Histogram of signals - {i+1}")
      res = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
      TP,TN,FP,FN,power,pi0 = res
      total = TP+TN+FP+FN

      p_sig_dict = {
      "Firing": len_p_fire,
      "Uncorrected": len(sig_uncorrected),
      "Bonferroni": len(sig_bonf_p),
      "Holm": len(sig_holm_p),
      "SGoF": len(sig_sgof_p),
      "BH": len(sig_bh_p),
      "BY": len(sig_by_p),
      "Q-value": len(sig_q),
      "TP":TP,
      "TN":TN,
      "FP":FP,
      "FN":FN,
      "Power":power,
      "pi0": pi0,
      "pi0 estimate": pi0_est,
      "cutoff": cutoff,
      "Total":total
      }
      p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

      # Concatenate the p_values_df DataFrame with your existing DataFrame
      df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
      df_sigp

      methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'True  Signals']
      sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q,fire_index]
      #plot_heatmap(methods, sig_indices, title=f"Significant index plot for iteration - {i+1}"

      return df_sigp

sequential_sim(p_values,df_sigp, method = "q")




df_sigp.to_csv('MultiDST/Simulation Test results/setting0.csv', index=False)


