# Step 4: Implementing Linear Regression and Multiple Linear Regression

## Introduction

Linear Regression is one of the simplest and most widely used algorithms in machine learning for predicting a continuous target variable based on one or more input features. Simple Linear Regression uses one independent variable, while Multiple Linear Regression uses two or more. The goal is to find the best-fitting straight line (or hyperplane in multiple dimensions) that predicts the target variable.

---

## Code

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=10, step=1)
selector = rfe.fit(X_train.fillna(0).select_dtypes(exclude='object'), y)
selectedFeatures = list(
    X_train.select_dtypes(exclude='object').columns[selector.support_])
selectedFeatures
plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=train_data, palette='GnBu')
title = plt.title('House Price by Overall Quality')
def plotCorrelation(variables):
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
plt.figure(figsize=(8, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=train_data, palette='GnBu')
title = plt.title('House Price by Garage Size')
plt.figure(figsize=(15, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_data)
title = plt.title('House Price by Year Built')
sigCatCols = [
    'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'RoofMatl', 'ExterQual', 'BsmtQual',
    'BsmtExposure', 'KitchenQual', 'Functional', 'GarageQual', 'PoolQC'
]
def visualizeCatFeature(feature):
    featOrder = train_data.groupby(
        [feature]).median().SalePrice.sort_values(ascending=False).index
    sns.boxplot(x=feature,
                y='SalePrice',
                data=train_data,
                order=featOrder,
                palette='GnBu_r')
def visualizeCatFeature(col):
    cat_price = train_data.groupby(col)['SalePrice'].median().sort_values()
    plt.bar(cat_price.index, cat_price.values)
    plt.xlabel(col)
    plt.ylabel('Median SalePrice')
plt.figure(figsize=(12, 6))
visualizeCatFeature('SalePrice')
plt.title('House Price by SalePrice')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
visualizeCatFeature('RoofMatl')
title = plt.title('House Price by Roof Material')
plt.figure(figsize=(8, 6))
visualizeCatFeature('KitchenQual')
title = plt.title('House Price by Kitchen Quality')
numeric_data = train_data.select_dtypes(include='number')
corr_mat = numeric_data.corr()
high_corr_mat = corr_mat[abs(corr_mat) >= 0.5]
plt.figure(figsize=(15, 10))
sns.heatmap(high_corr_mat,
            annot=True,
            fmt='.1f',
            cmap='GnBu',
            vmin=0.5,
            vmax=1)
plt.title('Correlation Heatmap')
plt.show()
```

---

## Explanation Table

| Code Line (or Block) | Simple Explanation |
|----------------------|-------------------|
| `from sklearn.feature_selection import RFE` | Imports RFE for feature selection. |
| `from sklearn.linear_model import LinearRegression` | Imports Linear Regression model. |
| `estimator = LinearRegression()` | Creates a Linear Regression model object. |
| `rfe = RFE(estimator, n_features_to_select=10, step=1)` | Sets up RFE to select 10 best features using Linear Regression. |
| `selector = rfe.fit(X_train.fillna(0).select_dtypes(exclude='object'), y)` | Fits RFE on numeric features and target, filling missing values with 0. |
| `selectedFeatures = list(X_train.select_dtypes(exclude='object').columns[selector.support_])` | Gets the names of selected features. |
| `selectedFeatures` | Displays the selected features. |
| `plt.figure(figsize=(8, 6)); sns.boxplot(x='OverallQual', y='SalePrice', data=train_data, palette='GnBu')` | Draws a boxplot to visualize relationship between Overall Quality and Sale Price. |
| `title = plt.title('House Price by Overall Quality')` | Adds title to the plot. |
| `def plotCorrelation(variables): ...` | Defines a function to plot correlation and regression line between two variables. |
| `plotCorrelation(['GrLivArea', 'SalePrice'])` | Plots correlation and regression line between living area and sale price. |
| `plt.figure(figsize=(8, 6)); sns.boxplot(x='GarageCars', y='SalePrice', data=train_data, palette='GnBu')` | Boxplot to show relationship between number of garage cars and sale price. |
| `plt.figure(figsize=(15, 6)); sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_data)` | Scatterplot to show how house price changes with year built. |
| `sigCatCols = [...]` | List of significant categorical columns for analysis. |
| `def visualizeCatFeature(feature): ...` | Defines function for plotting the effect of categorical features. |
| `plt.figure(figsize=(12, 6)); visualizeCatFeature('SalePrice')` | Visualizes median sale price by category. |
| `plt.title('House Price by SalePrice'); plt.xticks(rotation=45); plt.tight_layout(); plt.show()` | Adds title and adjusts layout for the plot. |
| `plt.figure(figsize=(8, 6)); visualizeCatFeature('RoofMatl'); title = plt.title('House Price by Roof Material')` | Plots the impact of roof material on house price. |
| `plt.figure(figsize=(8, 6)); visualizeCatFeature('KitchenQual'); title = plt.title('House Price by Kitchen Quality')` | Plots the impact of kitchen quality on house price. |
| `numeric_data = train_data.select_dtypes(include='number')` | Selects all numeric columns. |
| `corr_mat = numeric_data.corr()` | Calculates correlation matrix for numeric features. |
| `high_corr_mat = corr_mat[abs(corr_mat) >= 0.5]` | Filters for strong correlations (>=0.5). |
| `plt.figure(figsize=(15, 10)); sns.heatmap(high_corr_mat, annot=True, fmt='.1f', cmap='GnBu', vmin=0.5, vmax=1)` | Visualizes strong correlations using a heatmap. |
| `plt.title('Correlation Heatmap'); plt.show()` | Adds title and shows the heatmap plot. |

---

## Expected Outputs Summary

- The code will display the top 10 features most relevant to predicting house sale price using linear regression.
- Boxplots and scatterplots will visually show how features like quality, garage size, year built, and other categorical variables impact sale price.
- Correlation plots and heatmaps will help in understanding which features are strongly related to the target variable.
- The selected features and their relationships to sale price can be used for building both simple and multiple linear regression models.
