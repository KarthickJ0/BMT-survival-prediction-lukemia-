import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/Vetri/OneDrive/Desktop/karthick J/Project/dataset/bone-marrow-dataset.csv')

# Replace '?' with NaN for consistency in missing data handling
data = data.replace('?', np.nan)

# Check missing data before imputation
print("Missing data before imputation:")
print(data.isnull().sum())

# Initialize KNNImputer (you can also try other imputation strategies here)
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN imputation
data_imputed_knn = knn_imputer.fit_transform(data.select_dtypes(include=[np.number]))

# Convert imputed data back to DataFrame
data_imputed_knn = pd.DataFrame(data_imputed_knn, columns=data.select_dtypes(include=[np.number]).columns)

# Replace imputed values back into original data
data[data_imputed_knn.columns] = data_imputed_knn

# Apply mean imputation for columns with remaining missing data
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':  # For categorical columns, apply mode imputation
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:  # For numerical columns, apply mean imputation
            data[column].fillna(data[column].mean(), inplace=True)

# Check missing data after imputation
print("\nMissing data after imputation:")
print(data.isnull().sum())
