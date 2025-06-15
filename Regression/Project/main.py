
import numpy as np

# In[2]:


import pandas as pd

# In[3]:


import matplotlib.pyplot as plt

# In[4]:


import seaborn as sns

# In[5]:


from sklearn.preprocessing import StandardScaler, LabelEncoder

# In[6]:


from sklearn.decomposition import PCA

# In[7]:


from imblearn.over_sampling import SMOTE

# In[8]:


from sklearn.preprocessing import LabelEncoder

# In[9]:


from scipy import stats

# In[10]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline
sns.set_style('darkgrid')

# **Load data**

# In[ ]:


train_data = pd.read_csv('Data/train.csv', index_col='Id')
test_data = pd.read_csv('Data/test.csv', index_col='Id')

X_train = train_data.drop(['SalePrice'], axis=1)
y = train_data.SalePrice

X = pd.concat([X_train, test_data], axis=0)

# **Describe data**

# In[12]:


print("Train data's size: ", X_train.shape)
print("Test data's size: ", test_data.shape)

# In[13]:


numCols = list(X_train.select_dtypes(exclude='object').columns)
print(f"There are {len(numCols)} numerical features:\n", numCols)

# In[14]:


catCols = list(X_train.select_dtypes(include='object').columns)
print(f"There are {len(catCols)} numerical features:\n", catCols)

# **Data dictionary** can be found [here](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/Data/data_description.txt).

# <a name="1-Introduction-to-Dataset"></a>
# # 1.Introduction to Dataset

# ### **Reading Data**

# In[15]:


train_data.head()

# In[16]:


train_data.tail()

# In[17]:


train_data.shape

# In[18]:


train_data.sample()

# In[19]:


train_data.info()

# In[20]:


train_data.describe()

# ### **Data Cleaning**

# **Handling Missing Values**

# In[21]:


train_data.isnull().sum()

# **Separate numeric and non-numeric columns**

# In[22]:


numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])

# **Fill missing values in numeric columns with the mean**

# In[23]:


numeric_cols.fillna(numeric_cols.mean(), inplace=True)

# **Combine back with non-numeric columns**

# In[24]:


train_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)

# **Check for any remaining missing values**

# In[25]:


missing_values = train_data.isnull().sum()
print(missing_values)

# In[26]:


numeric_cols = train_data.select_dtypes(include=[np.number])
non_numeric_cols = train_data.select_dtypes(exclude=[np.number])

numeric_cols.fillna(numeric_cols.mean(), inplace=True)

for col in non_numeric_cols.columns:
    non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)

train_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)

missing_values = train_data.isnull().sum()
print(missing_values)

# In[27]:


train_data.isnull().sum()

# In[28]:


train_data.dropna(inplace=True)

missing_values =train_data.isnull().sum()
print(missing_values)


# In[29]:


train_data.drop_duplicates(inplace=True)
train_data.shape

# **Outlier Detection and Removal**

# In[30]:


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

# In[31]:


plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
train_data_cleaned.select_dtypes(include=[np.number]).boxplot()
plt.title("After Outlier Removal")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# In[32]:


train_data_cleaned.head()

# **Data Transformation**

# **Normalization**:
# Normalization rescales the data to a fixed range, typically [0, 1] or [-1, 1].

# In[33]:


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


# **Standardization** :
# Standardization rescales the data so that it has a mean of 0 and a standard deviation of 1.

# In[34]:


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

# **One-Hot Encoding**

# In[35]:


train_data["LandContour"].unique()

# In[36]:


train_data.Neighborhood.unique()

# In[37]:


from sklearn.preprocessing import StandardScaler
cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
data1 = pd.get_dummies(cat_features)
data1

# In[38]:


cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
data1 = pd.get_dummies(train_data, columns=cat_features)
scaled_data = pd.concat([train_data, data1], axis=1)
print(scaled_data.shape)
print()
print('*' * 70)
scaled_data.head()

# **Data Reduction**

# In[39]:


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

# In[40]:


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

# **Handling Imbalanced Data**
#     
# - Resampling Techniques
# - Oversampling

# In[41]:


train_data.LotArea.value_counts(True)

# In[42]:


train_data = pd.read_csv('Data/train.csv')
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# In[43]:


if 'HouseStyle' not in train_data.columns:
    raise KeyError("'HouseStyle' column is not present in the dataset.")

# In[44]:


house_style = train_data['HouseStyle'].copy()

# In[45]:


cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
numeric_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O' and feature != 'HouseStyle']

# In[46]:


train_data = pd.get_dummies(train_data, columns=cat_features)
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features].values)

# In[47]:


print(train_data.columns.tolist())

# In[48]:


house_style_cols = [col for col in train_data.columns if col.startswith('HouseStyle_')]
y = train_data[house_style_cols].idxmax(axis=1).str.replace('HouseStyle_', '')
X = train_data.drop(columns=house_style_cols)

# In[49]:


if len(y.unique()) <= 1:
    raise ValueError("The target 'y' needs to have more than 1 class. Got 1 class instead.")

# In[50]:


print("Before SMOTE:", X.shape, y.shape)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# In[51]:


train_data_resampled = pd.concat(
    [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['HouseStyle'])],
    axis=1
)

# In[52]:


print("After SMOTE:", X_resampled.shape, y_resampled.shape)
print(train_data_resampled.head())

# **Undersampling**

# In[53]:


train_data = pd.read_csv('Data/train.csv')
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# In[54]:


cat_features = [feature for feature in train_data.columns if train_data[feature].dtype == 'O']
numeric_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O']

# In[55]:


train_data = pd.get_dummies(train_data, columns=cat_features)

# In[56]:


scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features].values)

# In[57]:


if train_data['SalePrice'].dtype != 'int64' and train_data['SalePrice'].dtype != 'bool':
    train_data['SalePrice'] = (train_data['SalePrice'] > 0.5).astype(int)

# In[58]:


X = train_data.drop(columns=['SalePrice'])
y = train_data['LotArea']

# In[59]:


if y.dtype == 'O':
    le = LabelEncoder()
    y = le.fit_transform(y)

# In[60]:


print(X.shape, y.shape)

# In[61]:


upper_limit = train_data['SalePrice'].quantile(0.99)
data = train_data[train_data['SalePrice'] <= upper_limit]

# In[62]:


data['SalePrice'] = np.log1p(data['SalePrice'])

# In[63]:


data_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['SalePrice'])], axis=1)
data_resampled.head()

# <a name="2-eda"></a>
# ## 2. Exploratory Data Analysis

# In[64]:


train_data = pd.read_csv('Data/train.csv', index_col='Id')
test_data = pd.read_csv('Data/test.csv', index_col='Id')

X_train = train_data.drop(['SalePrice'], axis=1)
y = train_data.SalePrice

X = pd.concat([X_train, test_data], axis=0)

# <a name="2.1-saleprice"></a>
# ### 2.1. Sale Price 

# In[65]:


plt.figure(figsize=(8,6))
sns.distplot(y)
title = plt.title("House Price Distribution")

# The distribution of `SalePrice` is right-skewed. Let's check its Skewness and Kurtosis statistics.

# In[66]:


print(f"""Skewness: {y.skew()}
Kurtosis: {y.kurt()}""")

# <a name="2.2-numerical"></a>
# ### 2.2. Numerical Features

# Top 10 numerical variables highly correlated with `SalePrice`:

# In[67]:


corr_mat = train_data.select_dtypes(include='number').corr()['SalePrice'].sort_values(ascending=False)
print(corr_mat.head(11))

# What are the top 10 features selected by [**Recursive Feature Elimination**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)?

# In[68]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=10, step=1)
selector = rfe.fit(X_train.fillna(0).select_dtypes(exclude='object'), y)
selectedFeatures = list(
    X_train.select_dtypes(exclude='object').columns[selector.support_])
selectedFeatures

# According to above analyses, **Overall Quality, Living Area, Number of Full Baths, Size of Garage and Year Built** are some of the most important features in determining house price. Let's take a closer look at them.

# **Overall Quality**

# Overall quality is the most important feature in both analyses. It is clear that higher quality makes the house more expensive.

# In[69]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=train_data, palette='GnBu')
title = plt.title('House Price by Overall Quality')

# **Living Area**
# 
# Living area has a linear relationship with house price. In the scatter plot below, we can clearly see some ***outliers*** in the data, especially the two houses in the lower-right corner with living area greater than ***4000 sqft*** and price lower than ***$200,000***.

# In[ ]:


def plotCorrelation(variables):
    """
    1. Print correlation of two variables
    2. Create jointplot of two variables
    """
    print("Correlation: ", train_data[[variables[0], variables[1]]].corr().iloc[1, 0])

    sns.jointplot(
        x=variables[0],
        y=variables[1],
        data=train_data,
        kind='reg',
        height=7,
        scatter_kws={'s': 10},
        marginal_kws={'kde': True}
    )

plotCorrelation(['GrLivArea', 'SalePrice'])


# **GarageCars**
# 
# Interestingly, houses with garage which can hold 4 cars are cheaper than houses with 3-car garage.

# In[71]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=train_data, palette='GnBu')
title = plt.title('House Price by Garage Size')

# **Year Built**
# 
# The age of the house also plays an important role in its price. Newer houses have higher average prices. There are several houses built before 1900 having a high price.

# In[72]:


plt.figure(figsize=(15, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_data)
title = plt.title('House Price by Year Built')

# <a name="2.2-categorical"></a>
# ### 2.3. Categorical Variables

# Using **ANOVA**, I have identified 15 categorical features having p-values lower than ***0.01***:

# In[73]:


sigCatCols = [
    'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'RoofMatl', 'ExterQual', 'BsmtQual',
    'BsmtExposure', 'KitchenQual', 'Functional', 'GarageQual', 'PoolQC'
]

# Let's explore some of them.

# In[74]:


def visualizeCatFeature(feature):
    """
    Visualize the relationship between `SalePrice` and categorical feature using box plots
    """
    # Descending order of levels sorted by median SalePrice
    featOrder = train_data.groupby(
        [feature]).median().SalePrice.sort_values(ascending=False).index

    # Create box plot
    sns.boxplot(x=feature,
                y='SalePrice',
                data=train_data,
                order=featOrder,
                palette='GnBu_r')

# **Neighborhood**
# 
# There is a big difference in house prices among neighborhood in Ames. The top 3 expensive neighborhoods are **NridgHt, NoRidge and StoneBr** with median house prices of approximately $300,000, three times as high as the median of the 3 cheapest neighborhoods, which are **BrDale, DOTRR and MeadowV**.

# In[75]:


def visualizeCatFeature(col):
    """
    Visualize the median SalePrice for each category in the specified column.
    """
    cat_price = train_data.groupby(col)['SalePrice'].median().sort_values()
    
    plt.bar(cat_price.index, cat_price.values)
    plt.xlabel(col)
    plt.ylabel('Median SalePrice')


# In[76]:


plt.figure(figsize=(12, 6))
visualizeCatFeature('SalePrice')
plt.title('House Price by SalePrice')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Roof Material**
# 
# Houses using **Wood Shingles roof** are the most expensive with price ranging from \\$300,000 to \\$450,000. There are also a lot of expensive houses using **Standard Composite Shingle roof**.

# In[77]:


plt.figure(figsize=(8, 6))
visualizeCatFeature('RoofMatl')
title = plt.title('House Price by Roof Material')

# **Kitchen Quality**
# 
# Kitchen Quality is another important feature to predict house price. There is a very big difference in price between houses with different kitchen quality. For example, the average price difference between a house with a **good** kitchen and one with an **excellent** kitchen is about $120,000.

# In[78]:


plt.figure(figsize=(8, 6))
visualizeCatFeature('KitchenQual')
title = plt.title('House Price by Kitchen Quality')

# <a name="2.4-correlations"></a>
# ### 2.4. Correlations

# In[ ]:


numeric_data = train_data.select_dtypes(include='number')
corr_mat = numeric_data.corr()

high_corr_mat = corr_mat[abs(corr_mat) >= 0.5]

# Plot correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(high_corr_mat,
            annot=True,
            fmt='.1f',
            cmap='GnBu',
            vmin=0.5,
            vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# There is multicollinearity in our training data. Below features are highly correlated:
#     - GarageCars and GarageArea
#     - GarageYrBlt and YearBuilt
#     - 1stFlrSF and TotalBsmtSF
#     - GrLivArea and TotRmsAbvGrd
# Multicolliniearity has a negative impact on our prediction models and makes standard errors of our estimates increase. Therefore, for each pair of highly correlated features, I will remove a feature that has a lower correlation with `SalePrice`.

# <a name="2.5-missing"></a>
# ### 2.5. Missing Values

# Most machine learning algorithms give an error when we train them on data with missing values. Therefore, it's important to identify them before deciding how to handle them (drop features or impute missing value).

# In[80]:


missing_data_count = X.isnull().sum()
missing_data_percent = X.isnull().sum() / len(X) * 100

missing_data = pd.DataFrame({
    'Count': missing_data_count,
    'Percent': missing_data_percent
})
missing_data = missing_data[missing_data.Count > 0]
missing_data.sort_values(by='Count', ascending=False, inplace=True)

print(f"There are {missing_data.shape[0]} features having missing data.\n")
print("Top 10 missing value features:")
missing_data.head(10)

# In[81]:


plt.figure(figsize=(12, 6))
sns.barplot(y=missing_data.head(18).index,
            x=missing_data.head(18).Count,
            palette='GnBu_r')
title = plt.title("Missing Values")

# With some basic understandings of the data set and features, let's move to data preprocessing and modeling steps.

# <a name="3-data-preprocessing"></a>
# ## 3. Data Preprocessing and Feature Engineering

# <a name="3.1-missing-values"></a>
# ### 3.1. Missing Values

# In[82]:


missing_data_count = X.isnull().sum()
missing_data_percent = X.isnull().sum() / len(X) * 100
missing_data = pd.DataFrame({
    'Count': missing_data_count,
    'Percent': missing_data_percent
})
missing_data = missing_data[missing_data.Count > 0]
missing_data.sort_values(by='Count', ascending=False, inplace=True)

# In[83]:


missing_data.head(10)

# There are 34 features that have missing values. I will divide them into three groups based on the data description:
#    - **Group 1 - Categorical variables where `NA` means no feature:** `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `MasVnrType` (15 variables)
#    
#     For this group I will impute `NA` with `'None'`.
#     
#    - **Group 2 - Numerical variables where `NA` means no feature:** `GarageArea`, `GarageCars`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `MasVnrArea` (10 variables)
#    
#     For this group I will impute `NA` with `0`.
#     
#    - **Group 3 - Other variables:** `Functional`, `MSZoning`, `Electrical`, `KitchenQual`, `Exterior1st`, `Exterior2nd`, `SaleType`, `Utilities`, `LotFrontage`, `GarageYrBlt` (9 variables)
#         - I will impute `Functional`, `MSZoning`, `Electrical`, `KitchenQual`, `Exterior1st`, `Exterior2nd`, `SaleType`, `Utilities` with their *modes*,
#         - impute `LotFrontage` with its *mean*,
#         - impute `GarageYrBlt` with `YearBuilt`.

# In[84]:


from sklearn.impute import SimpleImputer

# Group 1:

# In[85]:


group_1 = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

# In[86]:


X[group_1] = X[group_1].fillna("None")

# Group 2:

# In[87]:


group_2 = [
    'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

# In[88]:


X[group_2] = X[group_2].fillna(0)

# Group 3:

# In[89]:


group_3a = [
    'Functional', 'MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st',
    'Exterior2nd', 'SaleType', 'Utilities'
]

# In[90]:


imputer = SimpleImputer(strategy='most_frequent')

# In[91]:


X[group_3a] = pd.DataFrame(imputer.fit_transform(X[group_3a]), index=X.index)

# In[92]:


X.LotFrontage = X.LotFrontage.fillna(X.LotFrontage.mean())

# In[93]:


X.GarageYrBlt = X.GarageYrBlt.fillna(X.YearBuilt)

# Let's check whether there is any missing value left:

# In[94]:


sum(X.isnull().sum())

# Great! All missing values have been handled.

# <a name="3.2-outliers"></a>
# ### 3.2. Outliers

# Because regression models are very sensitive to outlier, we need to be aware of them. Let's examine outliers with a scatter plot.

# In[95]:


sns.set_style('darkgrid')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_data)
title = plt.title('House Price vs. Living Area')

# There are two observations lying separately from the rest. They have large living area but low price. They are the outliers that we are looking for. I will delete them from the training set.

# In[96]:


outlier_index = train_data[(train_data.GrLivArea > 4000)
                           & (train_data.SalePrice < 200000)].index

# In[97]:


X.drop(outlier_index, axis=0, inplace=True)

# In[98]:


y.drop(outlier_index, axis=0, inplace=True)

# <a name="3.3-feature-engineering"></a>
# ### 3.3. Feature Engineering

# <a name="3.3.1-create-new-variables"></a>
# #### 3.3.1. Create New Variables

# In this step I will create new features from weaker features in the training data. For example, the surface area of each floor has low correlation with house price; however, when we sum them up, the relationship becomes much stronger. In fact, `TotalSqFeet` becomes the strongest feature in the dataset. The new features I will create are **total square feet, total number of bathrooms, age of the house, whether the house was remodeled, and whether the house was sold in the same year it was built.**

# In[99]:


X['totalSqFeet'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

# In[100]:


X['totalBathroom'] = X.FullBath + X.BsmtFullBath + 0.5 * (X.HalfBath + X.BsmtHalfBath)

# In[101]:


X['houseAge'] = X.YrSold - X.YearBuilt

# In[102]:


X['reModeled'] = np.where(X.YearRemodAdd == X.YearBuilt, 0, 1)

# In[103]:


X['isNew'] = np.where(X.YrSold == X.YearBuilt, 1, 0)

# <a name="3.3.2-label-encoding"></a>
# #### 3.3.2. Label Encoding

# Ordinal categorical features are label encoded.

# Ordinal categorical columns

# In[104]:


label_encoding_cols = [
    "Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "BsmtQual", "ExterCond", "ExterQual", "FireplaceQu", "Functional",
    "GarageCond", "GarageQual", "HeatingQC", "KitchenQual", "LandSlope",
    "LotShape", "PavedDrive", "PoolQC", "Street", "Utilities"
]

# Apply Label Encoder

# In[105]:


label_encoder = LabelEncoder()

# In[106]:


for col in label_encoding_cols:
    X[col] = label_encoder.fit_transform(X[col])

# <a name="3.3.3-transform-variables"></a>
# #### 3.3.3. Transform Numerical Variables to Categorical Variables

# Because I have calculated age of houses, `YearBuilt` is no longer needed. However, `YrSold` could have a large impact on house price (e.g. In economic crisis years, house price could be lower). Therefore, I will transform it into categorical variables.
# 
# Like `YrSold`, some numerical variables don't have any ordinal meaning (e.g. `MoSold`, `MSSubClass`). I will transform them into categorical variables.

# In[107]:


to_factor_cols = ['YrSold', 'MoSold', 'MSSubClass']

# In[108]:


for col in to_factor_cols:
    X[col] = X[col].apply(str)

# <a name="3.4-skewness"></a>
# ### 3.4. Skewness and Normalizing Variables

# Normal distribution is one of the assumption that linear regression relies on. Therefore, transfoming skewed data will help our models perform better.
# 
# First, let's examine the target variable `SalePrice` with Distribution plot and Quantile-Quantile plot.

# **Target variable**

# In[109]:


from scipy.stats import norm

# In[110]:


def normality_plot(X):
    """
    1. Draw distribution plot with normal distribution fitted curve
    2. Draw Quantile-Quantile plot 
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.distplot(X, fit=norm, ax=axes[0])
    axes[0].set_title('Distribution Plot')

    axes[1] = stats.probplot((X), plot=plt)
    plt.tight_layout()

# In[111]:


normality_plot(y)

# One of the methods to normalize right-skewed data is using log transformation because big values will be pulled to the center. However, log(0) is Nan, so I will use log(1+X) to fix skewness instead.

# In[112]:


y = np.log(1 + y)

# And this is `SalePrice` after log transformation. The sknewness has been fixed.

# In[113]:


normality_plot(y)

# In the next step I will examine skewness in the rest of numerical variables and use log transformation to fix them,

# **Fixing skewness in other numerical variables**

# If skewness is less than -1 or greater than 1, the distribution is **highly skewed**.
# 
# If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is **moderately skewed**.
# 
# If skewness is between -0.5 and 0.5, the distribution is **approximately symmetric**.
# 
# Below are skewed features in our original train data.

# In[114]:


numeric_data = train_data.select_dtypes(include='number')

skewness = numeric_data.skew().sort_values(ascending=False)
skewness[abs(skewness) > 0.75]


# Let's check normality of `GrLivArea`:

# In[115]:


normality_plot(X.GrLivArea)

# In[ ]:


skewed_cols = list(skewness[abs(skewness) > 0.5].index)

skewed_cols = [
    col for col in skewed_cols if col not in ['MSSubClass', 'SalePrice']
]

for col in skewed_cols:
    X[col] = np.log(1 + X[col])

# Below is normality of `GrLivArea` after log-transformation. Skewness has been fixed.

# In[117]:


normality_plot(X.GrLivArea)

# <a name="3.5-feature-scaling"></a>
# ### 3.5. Feature Scaling

# Except for Decision Tree and Random Forest, it is highly recommended to standardize the data set before running machine learning algorithms since optimization methods and gradient descent run and converge faster on similarly scaled features.
# 
# However, outliers can often influence the sample mean and standard deviation in a negative way, and models like Lasso and Elastic Net are very sensitive to outliers. In such cases, the median and the interquartile range often give better results. I will use [**RobustScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to transform the training data.

# In[118]:


from sklearn.preprocessing import RobustScaler
numerical_cols = list(X.select_dtypes(exclude=['object']).columns)
scaler = RobustScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# <a name="3.6-one-hot-encoding"></a>
# ### 3.6. One-hot Encoding

# In[119]:


X = pd.get_dummies(X, drop_first=True)
print("X.shape:", X.shape)

# After preprocessing the train and test data, I split them again to perform modeling.

# In[120]:


ntest = len(test_data)
X_train = X.iloc[:-ntest, :]
X_test = X.iloc[-ntest:, :]
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

# <a name="4-modeling"></a>
# ## 4. Modeling

# In model evaluation, it's a common practice to split the entire training data into 2 sets of data (train and test). However, a model may work very well on a set of test data but have a poor performance on other sets of unseen data.
# 
# A solution to this problem is a procedure called [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) (CV). In the example below, under the basic approach, called k-fold CV, the training set is split into `5` smaller sets. Then, for each fold, a model is trained using the other `4` folds and evaluated on the remaining fold. The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
# 
# <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" width="400" height="200" alt="CV">
# 
# I will write a function to get the **Root Mean Squared Logarithmic Error (RMSLE)** for my models using cross-validation. There is one note here: because I have transformed the target variable to *log(1+y)* , the **Mean Squared Error** for *log(1+y)* is the **Mean Squared Logarithmic Error** for `SalePrice`.

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score

n_folds = 5


def getRMSLE(model):
    """
    Return the average RMSLE over all folds of training data.
    """
    kf = KFold(n_folds, shuffle=True, random_state=42)

    rmse = np.sqrt(-cross_val_score(
        model, X_train, y, scoring="neg_mean_squared_error", cv=kf))

    return rmse.mean()

# <a name="4.1-regularized-regression"></a>
# ### 4.1. Regularized Regressions

# In[122]:


from sklearn.linear_model import Ridge, Lasso

# <a name="4.1.1-ridge"></a>
# #### 4.1.1. Ridge Regression

# In the regularized linear regression (Ridge), we try to minimize:
# 
# $$ J(\theta) = \frac{1}{2m} \left( \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right)^2 \right) + \frac{\lambda}{2m} \left( \sum_{j=1}^n \theta_j^2 \right)$$
# 
# where $\lambda$ is a regularization parameter which controls the degree of regularization (thus, help preventing overfitting). The regularization term puts a penalty on the overall cost J. As the magnitudes of the model parameters $\theta_j$ increase, the penalty increases as well.
# 
# I will find the $\lambda$ that gives me the smallest **RMSLE** from cross-validation:

# In[123]:


lambda_list = list(np.linspace(20, 25, 101))

rmsle_ridge = [getRMSLE(Ridge(alpha=lambda_)) for lambda_ in lambda_list]
rmsle_ridge = pd.Series(rmsle_ridge, index=lambda_list)

rmsle_ridge.plot(title="RMSLE by lambda")
plt.xlabel("Lambda")
plt.ylabel("RMSLE")

print("Best lambda:", rmsle_ridge.idxmin())
print("RMSLE:", rmsle_ridge.min())

# In[124]:


ridge = Ridge(alpha=22.9)

# <a name="4.1.2-lasso"></a>
# #### 4.1.2. Lasso Regression

# Lasso Regression is very similar to Ridge regression. One difference is that in the regularization term, instead of using **sum of squared of $\theta$**, we use **sum of absolute value of $\theta$**:
# 
# $$ J(\theta) = \frac{1}{2m} \left( \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right)^2 \right) + \frac{\lambda}{2m} \left( \sum_{j=1}^n |\theta_j| \right)$$
# 
# Another big difference is that Ridge Regresion can only shrink parameters close to zero while Lasso Regression can shrink some parameters all the way to 0. Therefore, we can use Lasso Regression to perform feature selection and regression.
# 
# With the same method above, the best `lambda_` for my Lasso model is **0.00065**.

# In[125]:


lambda_list = list(np.linspace(0.0006, 0.0007, 11))
rmsle_lasso = [
    getRMSLE(Lasso(alpha=lambda_, max_iter=100000)) for lambda_ in lambda_list
]
rmsle_lasso = pd.Series(rmsle_lasso, index=lambda_list)

rmsle_lasso.plot(title="RMSLE by lambda")
plt.xlabel("Lambda")
plt.ylabel("RMSLE")

print("Best lambda:", rmsle_lasso.idxmin())
print("RMSLE:", rmsle_lasso.min())

# In[126]:


lasso = Lasso(alpha=0.00065, max_iter=100000)

# <a name="4.2-xgboost"></a>
# ### 4.2. XGBoost
# Following this [complete guide](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/?fbclid=IwAR1NTAXqgYzjOOFw3qOV5DrcItwNoM73iPvWggnuyVR1PbvORiEUjRunipo) of parameter tuning for XGBoost, I respectively tune and find the best parameter for  `n_estimators` `max_depth` `min_child_weight` `gamma` `subsample` `colsample_bytree` `reg_alpha` `reg_lambda` `learning_rate`.

# In[127]:


from xgboost import XGBRegressor

# In[ ]:


from xgboost import XGBRegressor

xgb = XGBRegressor(
    learning_rate=0.05,
    n_estimators=2100,
    max_depth=2,
    min_child_weight=2,
    gamma=0,
    subsample=0.65,
    colsample_bytree=0.46,
    scale_pos_weight=1,
    reg_alpha=0.464,
    reg_lambda=0.8571,
    random_state=7,
    n_jobs=2,
    verbosity=0 
)

getRMSLE(xgb)

# <a name="4.3-lightgbm"></a>
# ### 4.3. LightGBM

# LightGBM is a powerful gradient boosting framework based on decision tree algorithm. Like XGBoost, LightGBM has a high performance on large data sets  but much faster training speed than XGBoost does. Following [this guide](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/?fbclid=IwAR3uYr9U1VDaqh_jEn1cjvMyjEWVHKMaDm_Q9yD1y08OkGBywRR0qpuhhtw), I have tuned the parameters `num_leaves` `min_data_in_leaf` `max_depth` `bagging_fraction` `feature_fraction` `max_bin`. As you can see in the RMSLE reported below, for this data set LightGBM has better performance than XGBoost.

# In[129]:


from lightgbm import LGBMRegressor

# In[130]:


from lightgbm import LGBMRegressor

X_train.columns = X_train.columns.str.replace(' ', '_')

lgb = LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    n_estimators=730,
    num_leaves=8,
    min_data_in_leaf=4,
    max_depth=3,
    max_bin=55,
    bagging_fraction=0.78,
    bagging_freq=5,
    feature_fraction=0.24,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_sum_hessian_in_leaf=11,
    verbosity=-1
)

getRMSLE(lgb)

# <a name="4.4-averaging-model"></a>
# ### 4.4. Averaging Model

# Regularized regression and gradient boosting work very differently and they may perform well on different data points. Thus it is a good practice to get average predictions from these models. Below I create a new class for my averaging model.

# In[ ]:


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

class AveragingModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # Create clone models
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack(
            [model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)

# In[132]:


avg_model = AveragingModel(models=(ridge, lasso, xgb, lgb))
getRMSLE(avg_model)

# The RMSLE score of the averaging model is much better than any of base models. I will use this model as my final model. In the last step, I will train my final model on the whole training data, make predictions from the test data and save my output.

# In[ ]:


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

my_model = avg_model
my_model.fit(X_train, y)
predictions = my_model.predict(X_test)
final_predictions = np.exp(predictions) - 1
output = pd.DataFrame({'Id': test_data.index, 'SalePrice': final_predictions})
output.to_csv('submission.csv', index=False)

# <a name="5-conclusion"></a>
# ## 5. Conclusion

# In this project, I have conducted a detailed EDA to understand the data and important features. Based on exploratory analysis, I performed data preprocessing and feature engineering. Finally, I train regularized regression models (Ridge, Lasso), XGBoost and LightGBM, and take average predictions from these models to predict final price of each house. By the time I write this notebook, my best model has **Mean Absolute Error** of **12293.919**, ranking 95/15502, approximately top 0.6% in the Kaggle leaderboard.
