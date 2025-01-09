import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:/Users/Vetri/OneDrive/Desktop/karthick J/Project/dataset/bone-marrow-dataset.csv')

# Data Preprocessing

# Handle missing values
data = data.replace('?', None)  # Replace '?' with None for consistency
data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric (if possible)

# Print statistics of missing data before dropping
print("Missing data statistics before dropping:")
print(data.isnull().sum())

# Drop rows with missing values only in critical columns (e.g., survival_status)
data = data.dropna(subset=['survival_status'])

# Print statistics after dropping rows
print("\nMissing data statistics after dropping rows:")
print(data.isnull().sum())

# Check which columns are available in the dataset
print("\nColumns in the dataset:")
print(data.columns)

# Refined list of categorical columns (exclude continuous columns like age, CD34, etc.)
categorical_columns = ['donor_ABO', 'donor_CMV', 'recipient_gender', 'recipient_ABO', 'recipient_rh', 
                       'disease', 'disease_group', 'gender_match', 'ABO_match', 'CMV_status',
                       'HLA_match', 'HLA_mismatch', 'antigen', 'allel', 'HLA_group_1', 'risk_group', 
                       'stem_cell_source', 'tx_post_relapse', 'relapse', 'extensive_chronic_GvHD', 
                       'survival_status']

# Check for empty or missing categorical columns
for col in categorical_columns:
    if col in data.columns:
        if data[col].isnull().all():
            print(f"Column '{col}' is empty (all values are NaN). It will be skipped.")
        else:
            # Convert categorical columns to numeric values using label encoding
            data[col] = pd.Categorical(data[col]).codes
    else:
        print(f"Column '{col}' is missing in the dataset.")

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['survival_status'])
y = data['survival_status']

# Check if the dataset is not empty
if len(X) == 0 or len(y) == 0:
    raise ValueError("The dataset is empty after preprocessing. Check missing values handling.")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training using RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Model prediction
y_pred = rf_model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
