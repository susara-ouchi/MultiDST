import csv
import pandas as pd

# Opening each dataset
control_1 = pd.read_csv('Real-dataset-2-paired/control_1.tsv', sep='\t')
control_2 = pd.read_csv('Real-dataset-2-paired/control_2.tsv', sep='\t')
control_3 = pd.read_csv('Real-dataset-2-paired/control_3.tsv', sep='\t')
control_4 = pd.read_csv('Real-dataset-2-paired/control_4.tsv', sep='\t')
control_5 = pd.read_csv('Real-dataset-2-paired/control_4.tsv', sep='\t')
test_1 = pd.read_csv('Real-dataset-2-paired/test_1.tsv', sep='\t')
test_2 = pd.read_csv('Real-dataset-2-paired/test_2.tsv', sep='\t')
test_3 = pd.read_csv('Real-dataset-2-paired/test_3.tsv', sep='\t')
test_4 = pd.read_csv('Real-dataset-2-paired/test_4.tsv', sep='\t')
test_5 = pd.read_csv('Real-dataset-2-paired/test_5.tsv', sep='\t')

# Person 01
C1_p = control_1['5455178010_A.Detection Pval']
T1_p = test_1['5455178010_B.Detection Pval']

# Person 02
control_2['5455178010_E.Detection Pval'].mean()
test_2['5455178010_F.Detection Pval'].mean()

# Person 03
control_3['5455178010_I.Detection Pval'].mean()
test_3['5455178010_J.Detection Pval'].mean()

# Person 04
control_4['5522887032_E.Detection Pval'].mean()
test_4['5522887032_F.Detection Pval'].mean()

# Person 05
control_5['5522887032_E.Detection Pval'].mean()
test_5['5522887032_J.Detection Pval'].mean()

### Conducting t-test


