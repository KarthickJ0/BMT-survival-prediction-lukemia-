import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('C:/Users/Vetri/OneDrive/Desktop/karthick J/Project/dataset/bone-marrow-dataset.csv')

# Handle missing values by replacing '?' with NaN
data = data.replace('?', None)

# Drop columns with all missing values
data = data.dropna(axis=1, how='all')

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns using the mean strategy
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# Impute missing values for categorical columns using the most frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# After imputation, check for any remaining missing values
print("\nMissing data after imputation:")
print(data.isnull().sum())

# You can now proceed with the rest of your model pipeline (e.g., splitting data, training models)
