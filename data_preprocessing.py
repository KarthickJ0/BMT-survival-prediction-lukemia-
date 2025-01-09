import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('C:/Users/Vetri/OneDrive/Desktop/karthick J/Project/dataset/bone-marrow-dataset.csv')


# Display the first few rows to understand the structure
print(data.head())

# ---------------------------------------------------------------
# 1. Handle Missing Values
# ---------------------------------------------------------------

# Display columns with missing values
print("Missing values per column:")
print(data.isnull().sum())

# Fill missing numerical values with the mean of the column (can be adjusted)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing categorical values with the most frequent value (mode)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# ---------------------------------------------------------------
# 2. Remove Duplicate Rows
# ---------------------------------------------------------------

# Remove any duplicate rows in the dataset
data = data.drop_duplicates()

# Check again if any duplicates remain
print(f"Number of duplicate rows removed: {data.duplicated().sum()}")

# ---------------------------------------------------------------
# 3. Correct Invalid Data
# ---------------------------------------------------------------

# Example: Remove rows where 'donor_age' is less than 0 (invalid)
data = data[data['donor_age'] >= 0]

# Example: Handle negative or unrealistic values in other columns (you can add more checks as needed)
data = data[data['recipient_age'] >= 0]

# ---------------------------------------------------------------
# 4. Standardize Data Formats
# ---------------------------------------------------------------

# Convert categorical columns to numeric labels using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Convert any columns that should be datetime types (if applicable)
# Example: If 'donation_date' column exists and should be datetime
# data['donation_date'] = pd.to_datetime(data['donation_date'])

# ---------------------------------------------------------------
# 5. Outlier Detection and Handling
# ---------------------------------------------------------------

# Using Z-score method to detect outliers in numerical columns
from scipy import stats

# Calculate Z-scores
z_scores = stats.zscore(data[numerical_cols])

# Identify outliers with Z-scores > 3 or < -3
outliers = (abs(z_scores) > 3).all(axis=1)

# Remove outliers
data = data[~outliers]

# Alternatively, using IQR (Interquartile Range) for outlier detection
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers using IQR
outliers_iqr = (data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR))

# Remove outliers
data = data[~outliers_iqr.any(axis=1)]

# ---------------------------------------------------------------
# 6. Feature Engineering (Optional - Customization for Your Project)
# ---------------------------------------------------------------

# Create new features based on domain knowledge (if needed)
# Example: You can create interaction features or polynomial features, e.g.:
# data['age_risk'] = data['recipient_age'] * data['risk_group']

# One-hot encoding for categorical features if necessary
# data = pd.get_dummies(data, drop_first=True)

# ---------------------------------------------------------------
# 7. Final Check
# ---------------------------------------------------------------

# Check for any remaining missing values
print("Final missing values per column after preprocessing:")
print(data.isnull().sum())

# Display the cleaned data
print(data.head())

# Save the cleaned data to a new CSV file
data.to_csv('path_to_cleaned_data.csv', index=False)

