import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv(r'C:\Users\Vetri\OneDrive\Desktop\karthick J\Project\path_to_cleaned_data.csv')  # Use raw string (r) to avoid escape issues

# Check the columns in the dataset
print(f"Columns in the dataset: {df.columns}")

# Preprocess the data (you can modify this step based on your specific dataset)
# Example: Dropping rows with missing values
df = df.dropna()

# Define features and target
X = df.drop(columns=['survival_status'])  # Exclude the target column
y = df['survival_status']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
svm = SVC(probability=True)

# Voting Classifier (combines the predictions of the other models)
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('svm', svm)], voting='soft')

# Train the models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
voting_clf.fit(X_train, y_train)

# Predictions
log_reg_pred = log_reg.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
voting_pred = voting_clf.predict(X_test)

# Evaluation
log_reg_acc = accuracy_score(y_test, log_reg_pred)
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
voting_acc = accuracy_score(y_test, voting_pred)

# Classification reports
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))

print("Voting Classifier Classification Report:")
print(classification_report(y_test, voting_pred))

# Accuracy output
print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(f"Random Forest Accuracy: {rf_acc}")
print(f"SVM Accuracy: {svm_acc}")
print(f"Voting Classifier Accuracy: {voting_acc}")

# Visualize the accuracies of the models
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Voting Classifier']
accuracies = [log_reg_acc, rf_acc, svm_acc, voting_acc]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=accuracies, hue=models)  # Updated to use hue for colors
plt.title('Model Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.show()
