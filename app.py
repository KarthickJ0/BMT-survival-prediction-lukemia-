import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Function to load the pre-trained models
def load_model(model_name):
    try:
        model = joblib.load(f'{model_name}.pkl')  # Ensure model files are named correctly
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# Function to preprocess and align features with the model
def preprocess_features(features, model):
    # Define encoding dictionary
    encoding_dict = {
        "Gender": {"Male": 0, "Female": 1},
        "HLA_typing": {"HLA-A": 0, "HLA-B": 1, "HLA-DR": 2, "Other": 3},
        "recipient_rh": {"Positive": 1, "Negative": 0},
        "donor_gender": {"Male": 0, "Female": 1},
        "donor_rh": {"Positive": 1, "Negative": 0},
        "transplant_type": {"Allogeneic": 0, "Autologous": 1},
        "chronic_GvHD": {"Yes": 1, "No": 0},
    }

    # Encode features
    for column, mapping in encoding_dict.items():
        if column in features.columns:
            features[column] = features[column].map(mapping)

    # Convert to numeric and align with model's required features
    features = features.apply(pd.to_numeric, errors='coerce')
    
    if model is not None:
        # Align features to match model's training columns
        required_features = model.feature_names_in_
        features = features.reindex(columns=required_features, fill_value=0)

    return features

# Function to make predictions
def predict(model, features):
    if model is not None:
        try:
            prediction = model.predict(features)
            probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            return prediction, probabilities
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None

# Function to plot SHAP summary
def plot_shap_summary(shap_values, feature_names):
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

# Function to handle SHAP explanations
def generate_shap_explanations(model, features):
    try:
        explainer = shap.KernelExplainer(model.predict_proba, features)
        shap_values = explainer.shap_values(features)
        return shap_values
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {e}")
        return None

# Streamlit UI
st.title("Survival Prediction for Leukemia")
st.write("Provide the required inputs to predict survival outcomes and view model insights.")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=30)
cd34_count = st.number_input("CD34+ Cell Count", min_value=0.0, value=10.0)
gender = st.selectbox("Gender", ["Male", "Female"])
hla_typing = st.selectbox("HLA Typing", ["HLA-A", "HLA-B", "HLA-DR", "Other"])
recipient_age_below_10 = st.selectbox("Recipient Age Below 10", [0, 1])
recipient_body_mass = st.number_input("Recipient Body Mass", min_value=0.0, value=70.0)
recipient_rh = st.selectbox("Recipient RH Factor", ["Positive", "Negative"])
donor_gender = st.selectbox("Donor Gender", ["Male", "Female"])
donor_rh = st.selectbox("Donor RH Factor", ["Positive", "Negative"])
transplant_type = st.selectbox("Transplant Type", ["Allogeneic", "Autologous"])
acute_gvhd_grade = st.selectbox("Acute GVHD Grade", [0, 1, 2, 3, 4])
chronic_gvhd = st.selectbox("Chronic GVHD", ["Yes", "No"])

# Collect all input features into a DataFrame
features = pd.DataFrame({
    "Age": [age],
    "CD34_count": [cd34_count],
    "Gender": [gender],
    "HLA_typing": [hla_typing],
    "recipient_age_below_10": [recipient_age_below_10],
    "recipient_body_mass": [recipient_body_mass],
    "recipient_rh": [recipient_rh],
    "donor_gender": [donor_gender],
    "donor_rh": [donor_rh],
    "transplant_type": [transplant_type],
    "acute_GvHD_grade": [acute_gvhd_grade],
    "chronic_GvHD": [chronic_gvhd],
})

# Load the selected model
model_option = st.selectbox("Select the Model", ["Logistic Regression", "SVM", "Random Forest", "Voting Classifier (Hard)", "Voting Classifier (Soft)"])
model_name_map = {
    "Logistic Regression": "logistic_regression_best_model",
    "SVM": "svm_best_model",
    "Random Forest": "random_forest_best_model",
    "Voting Classifier (Hard)": "voting_classifier_hard",
    "Voting Classifier (Soft)": "voting_classifier_soft",
}
model = load_model(model_name_map[model_option])

# Preprocess features
features = preprocess_features(features, model)

# Predict and display results
if st.button("Predict"):
    prediction, probabilities = predict(model, features)
    if prediction is not None:
        st.write(f"Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
        if probabilities is not None:
            labels = ['Not Survived', 'Survived']
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(probabilities[0], labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
            ax.axis('equal')
            st.pyplot(fig)

        # Generate SHAP explanations
        shap_values = generate_shap_explanations(model, features)
        if shap_values is not None:
            st.write("SHAP Summary:")
            plot_shap_summary(shap_values, features.columns)
