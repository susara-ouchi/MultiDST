import pandas as pd
from A01_sim_data import simulation_01

sim1 = simulation_01(42,9500,500,effect=0.5,n0=30,n1=30,threshold=0.05,show_plot=False,s0=1,s1=1)

p_values = sim1[0]
significant_p = sim1[1]
fire_index = sim1[2]
nonfire_index = sim1[3]

index_p = [i+1 for i,p in enumerate(p_values)]
p_cls = ['fire' if num in fire_index else 'non-fire' for num,p in enumerate(p_values)]

# Create Dataframe
p_val_df = pd.DataFrame({'index':index_p, 'p_values':p_values, 'class':p_cls})