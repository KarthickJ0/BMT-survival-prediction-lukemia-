import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

# Load the cleaned dataset
data = pd.read_csv(r"C:\Users\Vetri\OneDrive\Desktop\karthick J\Project\path_to_cleaned_data.csv")

# Separate features and target variable
X = data.drop('survival_status', axis=1)  # Replace 'survival_status' with your target column
y = data['survival_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (important for models like Logistic Regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
logreg = LogisticRegression(max_iter=1000)  # Increased max_iter for Logistic Regression
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', random_state=42)

# Create a voting classifier (combining multiple models)
voting_clf = VotingClassifier(estimators=[
    ('logreg', logreg),
    ('rf', rf),
    ('svm', svm)
], voting='hard')

# Train the models
logreg.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train_scaled, y_train)
voting_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test_scaled)
y_pred_voting = voting_clf.predict(X_test_scaled)

# Print classification reports for each model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("Voting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

# Calculate and print accuracy for each model
logreg_acc = logreg.score(X_test_scaled, y_test)
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test_scaled, y_test)
voting_acc = voting_clf.score(X_test_scaled, y_test)

print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"Combined Model Accuracy (Voting): {voting_acc:.4f}")

# Plotting the accuracy of the models
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Voting Classifier']
accuracies = [logreg_acc, rf_acc, svm_acc, voting_acc]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

# Plotting a pie chart of the target variable distribution (survival_status)
target_counts = y.value_counts()

plt.figure(figsize=(6, 6))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title('Distribution of Survival Status')
plt.show()
