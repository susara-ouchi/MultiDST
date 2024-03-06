import csv
import pandas as pd

# Opening each dataset
control_1 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_1.tsv', sep='\t')
control_2 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_2.tsv', sep='\t')
control_3 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_3.tsv', sep='\t')
control_4 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_4.tsv', sep='\t')
control_5 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_4.tsv', sep='\t')
test_1 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/test_1.tsv', sep='\t')
test_2 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/test_2.tsv', sep='\t')
test_3 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/test_3.tsv', sep='\t')
test_4 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/test_4.tsv', sep='\t')
test_5 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/test_5.tsv', sep='\t')

control_1['5455178010_A.BEAD_STDERR'].mean()
data = control_1['VALUE']

import numpy as np

data = np.array(data)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler to the data and transform the data
standardized_data = scaler.fit_transform(data)

# Person 01
C1_p = control_1['5455178010_A.Detection Pval']
T1_p = test_1['5455178010_B.Detection Pval']

# Person 02
C2_p = control_2['5455178010_E.Detection Pval']
T2_p = test_2['5455178010_F.Detection Pval']

# Person 03
C3_p = control_3['5455178010_I.Detection Pval']
T3_p = test_3['5455178010_J.Detection Pval']

# Person 04
C4_p = control_4['5522887032_E.Detection Pval']
T4_p = test_4['5522887032_F.Detection Pval']

# Person 05
C5_p = control_5['5522887032_E.Detection Pval'].mean()
T5_p = test_5['5522887032_J.Detection Pval'].mean()


