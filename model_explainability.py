import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the dataset and trained models
data = pd.read_csv(r"C:\Users\Vetri\OneDrive\Desktop\karthick J\Project\path_to_cleaned_data.csv") # Update with your dataset path
X = data.drop(columns=['survival_status'])  # Drop the target column
y = data['survival_status']  # Assuming 'survival_status' is your target variable

# Feature Scaling (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert X_scaled back to a DataFrame with original column names
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Load the saved models
logistic_model = joblib.load('logistic_regression_best_model.pkl')
rf_model = joblib.load('random_forest_best_model.pkl')
svm_model = joblib.load('svm_best_model.pkl')

# Use SHAP to explain Logistic Regression model
explainer_lr = shap.Explainer(logistic_model, X_scaled_df)
shap_values_lr = explainer_lr(X_scaled_df)

# Plot SHAP summary for Logistic Regression
shap.summary_plot(shap_values_lr.values, X_scaled_df)

# Use SHAP to explain Random Forest model
explainer_rf = shap.Explainer(rf_model, X_scaled_df)
shap_values_rf = explainer_rf(X_scaled_df)

# Plot SHAP summary for Random Forest
shap.summary_plot(shap_values_rf.values, X_scaled_df)

# **Fix for SVM Model**:
# Wrap the SVM model to output probabilities for SHAP
def predict_proba_svm(X):
    return svm_model.predict_proba(X)

# Use SHAP to explain SVM model (with probability output)
explainer_svm = shap.Explainer(predict_proba_svm, X_scaled_df)
shap_values_svm = explainer_svm(X_scaled_df)

# Plot SHAP summary for SVM
shap.summary_plot(shap_values_svm.values, X_scaled_df)

# Use SHAP to explain Ensemble model (Hard Voting, Soft Voting, or Stacking)
# For example, using the Random Forest model as base
explainer_ensemble = shap.Explainer(rf_model, X_scaled_df)
shap_values_ensemble = explainer_ensemble(X_scaled_df)

# Plot SHAP summary for Ensemble model
shap.summary_plot(shap_values_ensemble.values, X_scaled_df)

# Optional: Visualize SHAP for individual predictions
# For example, explain the first prediction
shap.initjs()
shap.force_plot(shap_values_lr[0], X_scaled_df.iloc[0], X_scaled_df.columns)  # Modify for any model

# Save SHAP plots as images (if needed)
shap.summary_plot(shap_values_lr.values, X_scaled_df, show=False)
plt.savefig('shap_logistic_summary.png')

shap.summary_plot(shap_values_rf.values, X_scaled_df, show=False)
plt.savefig('shap_rf_summary.png')

shap.summary_plot(shap_values_svm.values, X_scaled_df, show=False)
plt.savefig('shap_svm_summary.png')

shap.summary_plot(shap_values_ensemble.values, X_scaled_df, show=False)
plt.savefig('shap_ensemble_summary.png')

print("Model explainability plots saved!")
