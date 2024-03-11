import numpy as np
import matplotlib.pyplot as plt

###################################### Simulation distribution loading ###########################################
from A01_sim_data import simulation_01
from A02_FWER5_sgof import sgof_test

p_values = [0.00001,0, 0.03,0.1]

sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]