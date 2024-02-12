import pandas as pd
from A01_sim_data import p_values, significant_p, fire_index, nonfire_index

p_values
significant_p
fire_index
nonfire_index

index_p = [i+1 for i,p in enumerate(p_values)]
p_cls = ['fire' if num in fire_index else 'non-fire' for num,p in enumerate(p_values)]

# Create Dataframe
p_val_df = pd.DataFrame({'index':index_p, 'p_values':p_values, 'class':p_cls})