from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train SVM classifier
svm_classifier.fit(X_train, y_train)
# Predict on the test set
y_pred = svm_classifier.predict(X_test)
y_pred_t = svm_classifier.predict(X_train)

# Calculate accuracy
ts_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", ts_accuracy)

tr_accuracy = accuracy_score(y_train, y_pred_t)
print("Test Accuracy:", tr_accuracy)

cf_SVM = confusion_matrix(y_test,y_pred)
cf_SVM

### Full prediction
X_scaled = sc.fit_transform(X)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train SVM classifier on the entire dataset
svm_classifier.fit(X_scaled, y)

# Predict on the dataset
y_pred = svm_classifier.predict(X_scaled)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix
cf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cf_matrix)