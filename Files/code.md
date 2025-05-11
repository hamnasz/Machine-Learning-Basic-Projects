Below is the complete and proper Python code for performing linear regression on the Wine Quality dataset from the UCI Machine Learning Repository. The code follows ten essential steps for data preprocessing and model application, with detailed comments for clarity. It is executable as-is, assuming the required libraries are installed.

---

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Loading
# Load the dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')  # Specify semicolon as separator based on dataset format
print("Data loaded successfully. Shape:", df.shape)

# Step 2: Data Inspection
# Examine the first few rows, data types, and summary statistics
print("\nFirst few rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Step 3: Data Cleaning
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)
# No imputation needed as this dataset typically has no missing values

# Step 4: Feature Selection
# Select features (all columns except 'quality') and target ('quality')
X = df.drop('quality', axis=1)
y = df['quality']

# Step 5: Outlier Handling
# Identify and remove outliers using the IQR method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Create a mask to filter out outliers
mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis=1)
X = X[mask]
y = y[mask]
print("\nData shape after removing outliers:", X.shape)

# Step 6: Data Splitting
# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 7: Data Transformation
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)        # Transform testing data (no fitting)

# Step 8: Model Training
# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 9: Model Evaluation
# Predict on the testing data
y_pred = model.predict(X_test_scaled)
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Step 10: Results Visualization
# Visualize predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Wine Quality")
# Add a diagonal line for perfect predictions
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.show()
```

---

### Explanation of Each Step

#### 1. **Data Loading**
- The dataset is loaded from the URL using `pd.read_csv()`. The Wine Quality dataset uses semicolons (`;`) as separators, so `sep=';'` is specified.

#### 2. **Data Inspection**
- The first few rows (`head()`), data types (`info()`), and summary statistics (`describe()`) are printed to verify the data's structure and contents.

#### 3. **Data Cleaning**
- Missing values are checked with `isnull().sum()`. This dataset is typically complete, so no imputation is needed.

#### 4. **Feature Selection**
- Features (`X`) are all columns except 'quality', which is the target variable (`y`).

#### 5. **Outlier Handling**
- Outliers are removed using the Interquartile Range (IQR) method. Rows where any feature falls outside 1.5 * IQR from the quartiles are excluded.

#### 6. **Data Splitting**
- The dataset is split into training (80%) and testing (20%) sets using `train_test_split`, with a fixed `random_state` for reproducibility.

#### 7. **Data Transformation**
- Features are standardized using `StandardScaler`. The scaler is fitted only on the training data to prevent data leakage, then applied to both training and testing sets.

#### 8. **Model Training**
- A `LinearRegression` model is trained on the scaled training data.

#### 9. **Model Evaluation**
- Predictions are made on the test set, and performance is assessed using Mean Squared Error (MSE) and R-squared (R²).

#### 10. **Results Visualization**
- A scatter plot compares actual vs. predicted quality scores, with a red dashed line indicating perfect predictions.

---

### Notes
- **Libraries Required**: Ensure you have `pandas`, `numpy`, `matplotlib`, and `scikit-learn` installed (`pip install pandas numpy matplotlib scikit-learn`).
- **Target Variable**: The 'quality' column is discrete (integers, typically 3-8). Linear regression assumes a continuous target, so results may not be optimal. For discrete targets, consider ordinal regression (not implemented here as per the request).
- **Outlier Removal**: The IQR method may remove a significant portion of the data. Adjust the multiplier (e.g., from 1.5 to 3) if too many rows are lost.

This code provides a robust foundation for linear regression analysis on the Wine Quality dataset, with proper preprocessing and evaluation steps.