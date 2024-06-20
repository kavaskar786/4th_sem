import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
file_path = 'HealthDataset.csv'
data = pd.read_csv(file_path)

# EDA Techniques

# 1. Shape of the Dataset
print("Shape of the dataset:", data.shape)

# 2. Data Types
print("Data types:\n", data.dtypes)

# 3. Summary Statistics
print("Summary statistics:\n", data.describe())

# 4. Missing Values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 5. Value Counts for Categorical Columns
for col in data.select_dtypes(include=['object']).columns:
    print(f"Value counts for {col}:\n", data[col].value_counts())

# 6. Distribution of Numerical Columns
data.hist(figsize=(10, 10))
plt.show()

# 7. Boxplots for Outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.show()

# 8. Correlation Matrix
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
print("Correlation Matrix:\n", correlation_matrix)

# 9. Heatmap for Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 10. Pairplot for Relationships
sns.pairplot(numeric_data)
plt.show()

# Pre-processing Techniques

# 1. Handling Missing Values
data.fillna(data.mean(), inplace=True)

# 2. Log Transformation
data_transformed = data.copy()
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].skew() > 1:
        data_transformed[col] = np.log1p(data[col])

# 3. Feature Engineering
if 'Weight' in data.columns and 'Height' in data.columns:
    data['BMI'] = data['Weight'] / (data['Height']/100)**2

# 4. Outlier Detection using Z-score
z_scores = zscore(data.select_dtypes(include=[np.number]))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_cleaned = data[filtered_entries]

# 5. Handling Outliers (Replacing with Median)
for col in data.select_dtypes(include=[np.number]).columns:
    median = data[col].median()
    std = data[col].std()
    outliers = (data[col] - median).abs() > 3 * std
    data.loc[outliers, col] = median

# 6. Data Encoding (One-Hot Encoding)
data_encoded = pd.get_dummies(data, drop_first=True)

# 7. Standardization
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_encoded)

# 8. Normalization
normalizer = MinMaxScaler()
data_normalized = normalizer.fit_transform(data_encoded)

# 9. Removing Duplicates
data_deduped = data.drop_duplicates()

# 10. Scaling Numerical Features
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=[np.number]).columns
data_scaled = data.copy()
data_scaled[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Convert processed data to DataFrame
data_standardized = pd.DataFrame(data_standardized, columns=data_encoded.columns)
data_normalized = pd.DataFrame(data_normalized, columns=data_encoded.columns)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Display final processed data
print("Standardized Data:\n", data_standardized.head())
print("Normalized Data:\n", data_normalized.head())
print("Scaled Data:\n", data_scaled.head())
print("Data processing complete.")
