# Step 3: Data Preprocessing

## Introduction

Data preprocessing is a crucial step in any machine learning project. It involves preparing and cleaning the raw data before feeding it into a machine learning model. This step typically includes handling missing values, detecting and removing outliers, encoding categorical variables, normalizing or standardizing numerical features, and engineering new features. Proper preprocessing improves the quality of the data and helps machine learning models achieve better performance.

---

## Code

```python
train_data = pd.read_csv('Data/train.csv', index_col='Id')
test_data = pd.read_csv('Data/test.csv', index_col='Id')
X_train = train_data.drop(['SalePrice'], axis=1)
y = train_data.SalePrice
X = pd.concat([X_train, test_data], axis=0)
print("Train data's size: ", X_train.shape)
print("Test data's size: ", test_data.shape)
numCols = list(X_train.select_dtypes(exclude='object').columns)
print(f"There are {len(numCols)} numerical features:\n", numCols)
catCols = list(X_train.select_dtypes(include='object').columns)
print(f"There are {len(catCols)} numerical features:\n", catCols)
train_data.head()
train_data.tail()
train_data.shape
train_data.sample()
train_data.info()
train_data.describe()
train_data.isnull().sum()
numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])
numeric_cols.fillna(numeric_cols.mean(), inplace=True)
train_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)
missing_values = train_data.isnull().sum()
print(missing_values)
numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])
numeric_cols.fillna(numeric_cols.mean(), inplace=True)
for col in non_numeric_cols.columns:
    non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)
train_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)
missing_values = train_data.isnull().sum()
print(missing_values)
train_data.isnull().sum()
train_data.dropna(inplace=True)
missing_values =train_data.isnull().sum()
print(missing_values)
train_data.drop_duplicates(inplace=True)
train_data.shape
numeric_cols = train_data.select_dtypes(include=[np.number])
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1
train_data_cleaned = train_data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
numeric_cols.boxplot()
plt.title("Before Outlier Removal")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
train_data_cleaned.select_dtypes(include=[np.number]).boxplot()
plt.title("After Outlier Removal")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
train_data_cleaned.head()
from sklearn.preprocessing import MinMaxScaler
numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])
scaler = MinMaxScaler()
scaled_numeric_data = scaler.fit_transform(numeric_cols)
scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)
scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols.reset_index(drop=True)], axis=1)
print(scaled_data.shape)
print()
print('*' * 60)
scaled_data.head()
from sklearn.preprocessing import StandardScaler
numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])
scaler = StandardScaler()
scaled_numeric_data = scaler.fit_transform(numeric_cols)
scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)
scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols.reset_index(drop=True)], axis=1)
print(scaled_data.shape)
print()
print('*' * 60)
scaled_data.head()
train_data["LandContour"].unique()
train_data.Neighborhood.unique()
from sklearn.preprocessing import StandardScaler
cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
data1 = pd.get_dummies(cat_features)
data1
cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
data1 = pd.get_dummies(train_data, columns=cat_features)
scaled_data = pd.concat([train_data, data1], axis=1)
print(scaled_data.shape)
print()
print('*' * 70)
scaled_data.head()
from sklearn.decomposition import PCA
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
numeric_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O']
train_data = pd.get_dummies(train_data, columns=cat_features)
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features].values)
pca = PCA(n_components=15)
train_data_pca = pca.fit_transform(train_data)
print(train_data_pca.shape)
print(train_data_pca[:5])
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(train_data[numeric_features[0]], train_data[numeric_features[1]], alpha=0.5)
plt.title('Original train_Data')
plt.xlabel(numeric_features[0])
plt.ylabel(numeric_features[1])
pca = PCA(n_components=15)
train_data_pca = pca.fit_transform(train_data)
plt.subplot(1, 2, 2)
plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], alpha=0.5)
plt.title('PCA Transformed train_Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()
```

---

## Explanation Table

| Code Line (or Block) | Simple Explanation |
|----------------------|-------------------|
| `train_data = pd.read_csv('Data/train.csv', index_col='Id')` | Loads the training data from a CSV file using the 'Id' column as the index. |
| `test_data = pd.read_csv('Data/test.csv', index_col='Id')` | Loads the test data using the 'Id' column as the index. |
| `X_train = train_data.drop(['SalePrice'], axis=1)` | Removes the 'SalePrice' column from training data to get input features. |
| `y = train_data.SalePrice` | Sets the target variable as the 'SalePrice' column. |
| `X = pd.concat([X_train, test_data], axis=0)` | Combines training and test features for preprocessing. |
| `print("Train data's size: ", X_train.shape)` | Prints the shape (rows, columns) of the training data features. |
| `print("Test data's size: ", test_data.shape)` | Prints the shape of the test data. |
| `numCols = list(X_train.select_dtypes(exclude='object').columns)` | Finds all columns with numeric data types. |
| `print(f"There are {len(numCols)} numerical features:\n", numCols)` | Prints the count and names of numeric features. |
| `catCols = list(X_train.select_dtypes(include='object').columns)` | Finds all columns with categorical (object) data types. |
| `print(f"There are {len(catCols)} numerical features:\n", catCols)` | Prints the count and names of categorical features. |
| `train_data.head()` | Displays the first five rows of the training data. |
| `train_data.tail()` | Displays the last five rows of the training data. |
| `train_data.shape` | Shows the shape (rows, columns) of the training data. |
| `train_data.sample()` | Shows a random row from the training data. |
| `train_data.info()` | Prints information about data types and non-null values in each column. |
| `train_data.describe()` | Provides statistics (mean, std, min, etc.) for numeric columns. |
| `train_data.isnull().sum()` | Counts missing values in each column. |
| `numeric_cols = train_data.select_dtypes(include=[np.number])` | Selects only the numeric columns from training data. |
| `non_numeric_cols = train_data.select_dtypes(exclude=[np.number])` | Selects only the non-numeric (categorical) columns. |
| `numeric_cols.fillna(numeric_cols.mean(), inplace=True)` | Fills missing numeric values with their mean. |
| `train_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)` | Combines numeric and non-numeric columns back into one DataFrame. |
| `missing_values = train_data.isnull().sum()` | Counts missing values in each column after filling. |
| `print(missing_values)` | Prints the number of missing values per column. |
| `for col in non_numeric_cols.columns: non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)` | Fills missing categorical values with the most frequent value (mode) of each column. |
| `train_data.dropna(inplace=True)` | Removes any remaining rows with missing values. |
| `train_data.drop_duplicates(inplace=True)` | Removes any duplicate rows from the data. |
| `Q1 = numeric_cols.quantile(0.25); Q3 = numeric_cols.quantile(0.75); IQR = Q3 - Q1` | Calculates quartiles and the interquartile range for outlier detection. |
| `train_data_cleaned = train_data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]` | Removes rows with outlier values beyond 1.5*IQR from Q1 or Q3. |
| `plt.figure... plt.show()` (boxplots) | Shows boxplots before and after outlier removal for visualization. |
| `from sklearn.preprocessing import MinMaxScaler` | Imports the scaler for normalization. |
| `scaler = MinMaxScaler(); scaled_numeric_data = scaler.fit_transform(numeric_cols)` | Normalizes numeric columns to a 0-1 range. |
| `scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)` | Converts normalized data back to a DataFrame. |
| `scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols.reset_index(drop=True)], axis=1)` | Combines normalized numeric and original categorical columns. |
| `from sklearn.preprocessing import StandardScaler` | Imports the scaler for standardization. |
| `scaler = StandardScaler(); scaled_numeric_data = scaler.fit_transform(numeric_cols)` | Standardizes numeric columns to have mean 0 and variance 1. |
| `scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)` | Converts standardized data back to a DataFrame. |
| `scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols.reset_index(drop=True)], axis=1)` | Combines standardized numeric and original categorical columns. |
| `cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']` | Lists all categorical feature names. |
| `data1 = pd.get_dummies(cat_features)` | Incorrectly tries to create dummy variables from a list; should be from DataFrame. |
| `data1 = pd.get_dummies(train_data, columns=cat_features)` | Converts categorical columns into one-hot encoded columns. |
| `scaled_data = pd.concat([train_data, data1], axis=1)` | Combines original and one-hot encoded data. |
| `from sklearn.decomposition import PCA` | Imports PCA for dimensionality reduction. |
| `train_data.fillna(train_data.mean(numeric_only=True), inplace=True)` | Fills missing numeric values with mean before PCA. |
| `cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']` | Lists categorical features again. |
| `numeric_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O']` | Lists numeric features again. |
| `train_data = pd.get_dummies(train_data, columns=cat_features)` | One-hot encodes categorical features. |
| `scaler = StandardScaler(); train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features].values)` | Standardizes numeric features. |
| `pca = PCA(n_components=15); train_data_pca = pca.fit_transform(train_data)` | Applies PCA to reduce features to 15 principal components. |
| `print(train_data_pca.shape); print(train_data_pca[:5])` | Prints the shape and first five rows of PCA-transformed data. |
| `plt.figure... plt.show()` (scatter plots) | Visualizes the data before and after PCA transformation. |

---

## Expected Outputs Summary

- The shapes of training and test datasets, as well as lists of numeric and categorical features, will be printed.
- Summary statistics, info, and missing value counts for the training data will be displayed.
- Missing values will be filled, and remaining missing data and duplicates will be removed.
- Outliers will be detected and removed based on the interquartile range (IQR), with visual boxplots shown before and after removal.
- Data will be normalized and standardized, and their shapes will be printed.
- Categorical features will be identified and converted into numerical format using one-hot encoding.
- Dimensionality reduction using PCA will be performed, with the shape and a sample of the transformed data displayed.
- Scatter plots will show the data before and after PCA.
- The dataset will be ready for further machine learning modeling after these preprocessing steps.

---
