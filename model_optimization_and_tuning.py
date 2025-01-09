import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the preprocessed dataset
data = pd.read_csv(r"C:\Users\Vetri\OneDrive\Desktop\karthick J\Project\path_to_cleaned_data.csv")

# Verify the columns to confirm the target and features
print("Columns in the dataset:", data.columns)

# Define features (X) and target (y) for survival prediction
X = data.drop(columns=['survival_status'])  # Features
y = data['survival_status']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids for optimization
model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 20, None],
            'criterion': ['gini', 'entropy']
        }
    },
    'SVM': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }
    }
}

# Run GridSearchCV for each model and store results
best_models = {}
for model_name, mp in model_params.items():
    print(f"Training and tuning: {model_name}")
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_models[model_name] = clf.best_estimator_
    print(f"Best parameters for {model_name}: {clf.best_params_}")

# Evaluate each model on the test set
for model_name, model in best_models.items():
    print(f"\nEvaluating model: {model_name}")
    y_pred = model.predict(X_test)
    print(f"Classification Report for {model_name}:\n", classification_report(y_test, y_pred))

# Save the best models (optional, for reuse)
import joblib
for model_name, model in best_models.items():
    joblib.dump(model, f"{model_name.replace(' ', '_').lower()}_best_model.pkl")
    print(f"Saved {model_name} as {model_name.replace(' ', '_').lower()}_best_model.pkl")
