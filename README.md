# BMT-survival-prediction-lukemia-
Here’s a tailored project description for your GitHub repository, based on the details you’ve shared:

---

# Survival Prediction for Leukemia Patients Undergoing Bone Marrow Transplant (BMT) Using Machine Learning

This project aims to develop a machine learning-based system to predict survival outcomes for leukemia patients undergoing Bone Marrow Transplant (BMT). By leveraging clinical and transplant-related data, the system provides clinicians with actionable insights to enhance patient care and treatment strategies.

## Objectives:
- **Predict Survival Outcomes**: Estimate the likelihood of survival post-transplant.
- **Assess Relapse Risks**: Identify potential relapse risks based on patient data.
- **Support Clinical Decisions**: Aid healthcare professionals in selecting optimal treatment strategies, reducing risks like Graft-versus-Host Disease (GVHD), and improving overall transplant success rates.

## Features:
- **Data Preprocessing**: Includes handling missing data through techniques like KNN imputation, feature selection, and data normalization.
- **Model Development**: Utilizes machine learning models such as Logistic Regression, Random Forest, SVM,  optimized 
- **Multimodel Approach**: Combines multiple models to improve prediction accuracy and robustness.
- **Model Explainability**: Implements SHAP (SHapley Additive exPlanations) for interpreting model predictions.
- **Interactive User Interface**: A Streamlit-based application that allows users to input patient data and receive real-time predictions with visual feedback.

## Project Workflow:
1. **Data Preparation**: Clean and preprocess the dataset, addressing missing values and selecting relevant features.
2. **Model Training**: Train and validate multiple machine learning models to identify the best-performing algorithm.
3. **Explainability Analysis**: Use SHAP to explain model predictions and understand feature impacts.
4. **Deployment**: Develop a Streamlit app for real-time prediction and visualization.

## Installation:
1. Clone the repository:
   ```
   git clone https://github.com/KarthickJ0/survival-prediction-leukemia
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Dataset:
The project uses a clinical dataset that includes parameters such as patient age, CD34+ cell counts, HLA typing, and other transplant-related metrics. The dataset is preprocessed to ensure quality inputs for model training.

## Future Enhancements:
- **Relapse Prediction Module**: Adding functionality to predict patient relapse risks.
- **Batch Processing**: Enable predictions for multiple patients at once.
- **API Integration**: Develop an API for integration with healthcare systems.
- **Continuous Model Improvement**: Regularly retrain the model with new data to improve prediction accuracy over time.

## Contributions:
Contributions from the community are welcome. Please fork the repository and submit a pull request with your enhancements or fixes.

---

Feel free to adjust this description to reflect any additional details or updates specific to your project.
