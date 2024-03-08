from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from A05_ML_df import p_val_df

# Load dataset 
p_val_df
X = p_val_df[['index','p_values']]
y = p_val_df['class']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize Logistic Regression classifier
lr_classifier = LogisticRegression(random_state=42)

# Train Logistic Regression classifier
lr_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_classifier.predict(X_test)
y_pred_t = lr_classifier.predict(X_train)

# Calculate accuracy
ts_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", ts_accuracy)

tr_accuracy = accuracy_score(y_train, y_pred_t)
print("Train Accuracy:", tr_accuracy)

cf_LR = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cf_LR)

### Full prediction
X_scaled = sc.fit_transform(X)

# Initialize Logistic Regression classifier
lr_classifier_full = LogisticRegression(random_state=42)

# Train Logistic Regression classifier on the entire dataset
lr_classifier_full.fit(X_scaled, y)

# Predict on the dataset
y_pred_full = lr_classifier_full.predict(X_scaled)

# Calculate accuracy
accuracy_full = accuracy_score(y, y_pred_full)
print("Full Dataset Accuracy:", accuracy_full)

# Compute confusion matrix
cf_matrix_full = confusion_matrix(y, y_pred_full)
print("Full Dataset Confusion Matrix:")
print(cf_matrix_full)

TN = cf_matrix_full[0, 0]
FP = cf_matrix_full[0, 1]
FN = cf_matrix_full[1, 0]
TP = cf_matrix_full[1, 1]

### ROC AUC

from sklearn.metrics import roc_curve, roc_auc_score

# Calculate probabilities for positive class
y_probs = lr_classifier_full.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr = FP/(FP+TN) , TP / (TP + FN) 

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_probs)

# Print ROC AUC
print("ROC AUC:", roc_auc)

# Plot ROC curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()



####################################################################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

# Load dataset
X = p_val_df[['index','p_values']]
y = p_val_df['class']

# Split dataset into train and test sets (stratified sampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize Logistic Regression classifier with class weights
lr_classifier = LogisticRegression(random_state=42, class_weight={0: class_weights[0], 1: class_weights[1]})

# Train Logistic Regression classifier
lr_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Compute confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf_matrix)

# Calculate probabilities for positive class
y_probs = lr_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC:", roc_auc)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
