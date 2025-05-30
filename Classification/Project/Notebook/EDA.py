
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import plotly.express as px
import matplotlib.pyplot as plt

data_df = pd.read_csv("/workspaces/Machine-Learning-Basic-Projects/Classification/Project/Data/churn.csv")

# In[2]:


def dataoveriew(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())
    
dataoveriew(data_df, 'Overview of the dataset')

# The data fall under two categories:
# - 17 Categorical features:
#     - CustomerID: Customer ID unique for each customer
#     - gender: Whether the customer is a male or a female
#     - SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
#     - Partner: Whether the customer has a partner or not (Yes, No)
#     - Dependent: Whether the customer has dependents or not (Yes, No)
#     - PhoneService: Whether the customer has a phone service or not (Yes, No)
#     - MultipeLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
#     - InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
#     - OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
#     - OnlineBackup: Whether the customer has an online backup or not (Yes, No, No internet service)
#     - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
#     - TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
#     - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
#     - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
#     - Contract: The contract term of the customer (Month-to-month, One year, Two years)
#     - PaperlessBilling: The contract term of the customer (Month-to-month, One year, Two years)
#     - PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
#     
# ***
#     
# - 3 Numerical features:
# 
#     - Tenure: Number of months the customer has stayed with the company 
#     - MonthlyCharges: The amount charged to the customer monthly
#     - TotalCharges: The total amount charged to the customer
#       
# ***
# 
# - Prediction feature:
#     - Churn: Whether the customer churned or not (Yes or No)
#     
#     
# These features can also be sub-divided into:
# 
# - Demographic customer information
# 
#     - gender , SeniorCitizen , Partner , Dependents
# 
# - Services that each customer has signed up for
# 
#     - PhoneService , MultipleLines , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies, 
#     
# - Customer account information
# 
#     - tenure , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges

# ### Explore Target variable

# In[3]:


import plotly.express as px

target_instance = data_df["Churn"].value_counts().reset_index()
target_instance.columns = ['Category', 'Count']

fig = px.pie(
    target_instance,
    values='Count',
    names='Category',
    color='Category',
    color_discrete_sequence=["#FFFF99", "#FFF44F"],  # canary, lemon
    color_discrete_map={"No": "#E30B5C", "Yes": "#50C878"},  # raspberry, emerald
    title='Distribution of Churn'
)
fig.show()

# We are trying to predict users that left the company in the previous month. It is a binary classification problem with an unbalance target.
# - Churn: No - 73.5%
# - Churn: Yes - 26.5%

# ### Explore Categorical features

# In[4]:


def bar(feature, df=data_df ):
    # Groupby the categorical feature
    temp_df = df.groupby([feature, 'Churn']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})
    # Calculate the value counts of each distribution and it's corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]
    # Defining string formatting for graph annotation
    # Numeric section
    def num_format(list_instance):
        formatted_str = ''
        for index,num in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{num}%, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{num}% & '
            else:
                formatted_str=formatted_str+f'{num}%'
        return formatted_str
    # Categorical section
    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{cat}, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{cat} & '
            else:
                formatted_str=formatted_str+f'{cat}'
        return formatted_str

    # Running the formatting functions
    num_str = num_format(percentage)
    cat_str = str_format(categories)
    
    # Define your custom color maps
    churn_colors = ["#FFFF99", "#FFF44F"]  # canary, lemon
    gender_colors = {"Female": "#E30B5C", "Male": "#50C878"}  # raspberry, emerald

    # Use gender map if feature is gender, otherwise use churn map
    if feature.lower() == "gender":
        fig = px.bar(
            temp_df,
            x=feature,
            y='Count',
            color=feature,
            title=f'Churn rate by {feature}',
            barmode="group",
            color_discrete_map=gender_colors
        )
    else:
        fig = px.bar(
            temp_df,
            x=feature,
            y='Count',
            color='Churn',
            title=f'Churn rate by {feature}',
            barmode="group",
            color_discrete_sequence=churn_colors
        )

    fig.add_annotation(
        text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=1.4,
        y=1.3,
        bordercolor='black',
        borderwidth=1)
    fig.update_layout(
        margin=dict(r=400),
    )
    return fig.show()

# In[5]:


#Gender feature plot
bar('gender')
#SeniorCitizen feature plot 
data_df.loc[data_df.SeniorCitizen==0,'SeniorCitizen'] = "No"   #convert 0 to No in all data instances
data_df.loc[data_df.SeniorCitizen==1,'SeniorCitizen'] = "Yes"  #convert 1 to Yes in all data instances
bar('SeniorCitizen')
#Partner feature plot
bar('Partner')
#Dependents feature plot
bar('Dependents')

# ***
# **Demographic analysis Insight**: 
# Gender and partner are even distributed with approximate percentage values. The difference in churn is slightly higher in females but the diffreence is negligible. There is a higher proportion of churn amongst younger customers (where SeniorCitizen is No), customers with no partners and customers with no dependents. These analysis on demographic section of data highlights on-senior citizens with no partners and dependents describe a particular segment of customers that are likely to churn.
# ***

# In[6]:


bar('PhoneService')
bar('MultipleLines')
bar('InternetService')
bar('OnlineSecurity')
bar('OnlineBackup')
bar('DeviceProtection')
bar('TechSupport')
bar('StreamingTV')
bar('StreamingMovies')

# ***
# **Services that each customer has signed up for Insight**:
# These category of features shows significant variations across their values. If a customer does not have a phone service, he/she cannot have multiple lines. About 90.3% of the customers have phone services and have the higher rate to churn. Customers who have Fibre optic as internet service are more likely to churn, this can happen due to high prices, competition, customer service, and many other reasons. Fiber optic service is much more expensive than DSL which may be one of the reasons why customers churn. Customers with  OnlineSecurity ,OnlineBackup ,DeviceProtection and TechSupport  are more unlikely to churn. Streaming service is not predictive for churn as it evenly distributed to yes and no options.
# ***

# In[7]:


bar('Contract')
bar('PaperlessBilling')
bar('PaymentMethod')

# **Payment**:
# ***
# The shorter the contract the higher churn rate as those with longer plans face additional barriers when cancelling prematurely. This clearly explains the motivation for companies to have long-term relationship with their customers. Churn Rate is higher for the customers who opted for paperless billing, About 59.2% of the customers make paperless billing. Customers who pay with electronic check are more likely to churn and this kind of payment is more common than other payment types.
# ***

# ### Explore Numeric features

# In[8]:


data_df.dtypes

# In[9]:


try:
    data_df['TotalCharges'] = data_df['TotalCharges'].astype(float)
except ValueError as ve:
    print (ve)

# In[10]:


data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
#Fill the missing values with with the median value
data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())

# In[11]:


# Defining the histogram plotting function
def hist(feature):
    group_df = data_df.groupby([feature, 'Churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Churn', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    fig.show()

# In[12]:


hist('tenure')
hist('MonthlyCharges')
hist('TotalCharges')

# ***
# **Customer account information**: The tenure histogram is rightly skewed and shows that majority of customers has been with the telecom company for just the first few months (0-9 months) and the highest rate of churn is also in that first few months (0-9months). 75% of customers who end up leaving Telcom company  do so within their first 30 months. The monthly charge histogram shows that clients with higher monthly charges have a higher churn rate (This suggests that discounts and promotions can be an enticing reason for customers to stay). The total charge trend is quite depict due to variation in frequency.
# Lets bin the numeric features into 3 sections based on quantiles (low, medium and high to get more information from it).
# ***

# In[13]:


#Create an empty dataframe
bin_df = pd.DataFrame()

#Update the binning dataframe
bin_df['tenure_bins'] =  pd.qcut(data_df['tenure'], q=3, labels= ['low', 'medium', 'high'])
bin_df['MonthlyCharges_bins'] =  pd.qcut(data_df['MonthlyCharges'], q=3, labels= ['low', 'medium', 'high'])
bin_df['TotalCharges_bins'] =  pd.qcut(data_df['TotalCharges'], q=3, labels= ['low', 'medium', 'high'])
bin_df['Churn'] = data_df['Churn']

#Plot the bar chart of the binned variables
bar('tenure_bins', bin_df)
bar('MonthlyCharges_bins', bin_df)
bar('TotalCharges_bins', bin_df)

# ***
# Based on binning, the low tenure and high monthly charge bins have higher churn rates as supported with the previous analysis. While the low Total charge bin has a higher churn rate. 
# ***

# ### Data preprocessing

# In[14]:


# The customerID column isnt useful as the feature us used for identification of customers. 
data_df.drop(["customerID"],axis=1,inplace = True)

# Encode categorical features

#Defining the map function
def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})

## Encoding target feature
data_df['Churn'] = data_df[['Churn']].apply(binary_map)

# Encoding gender category
data_df['gender'] = data_df['gender'].map({'Male':1, 'Female':0})

#Encoding other binary category
binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
data_df[binary_list] = data_df[binary_list].apply(binary_map)

#Encoding the other categoric features with more than two categories
data_df = pd.get_dummies(data_df, drop_first=True)

# In[15]:


# Checking the correlation between features
corr = data_df.corr()

fig = px.imshow(corr,width=1000, height=1000)
fig.show()

# Correlation is a statistical term is a measure on linear relationship with two variables. Features with high correlation are more linearly dependent and have almost the same effect on the dependent variable. So when two features have a high correlation, we can drop one of the two features.

# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

#Change variable name seperators to '_'
all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in data_df.columns]

#Effect the change to the dataframe column names
data_df.columns = all_columns

#Prepare it for the GLM formula
glm_columns = [e for e in all_columns if e not in ['customerID', 'Churn']]
glm_columns = ' + '.join(map(str, glm_columns))

#Fiting it to the Generalized Linear Model
glm_model = smf.glm(formula=f'Churn ~ {glm_columns}', data=data_df, family=sm.families.Binomial())
res = glm_model.fit()
print(res.summary())

# In[ ]:


np.exp(res.params)

# In[ ]:


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data_df['tenure'] = sc.fit_transform(data_df[['tenure']])
data_df['MonthlyCharges'] = sc.fit_transform(data_df[['MonthlyCharges']])
data_df['TotalCharges'] = sc.fit_transform(data_df[['TotalCharges']])

# #### Creating a baseline model

# In[ ]:


# Import Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Split data into train and test sets
from sklearn.model_selection import train_test_split
X = data_df.drop('Churn', axis=1)
y = data_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)  

# In[ ]:


def modeling(alg, alg_name, params={}):
    model = alg(**params) #Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Performance evaluation
    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ",acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ",pre_score)
        rec_score = recall_score(y_true, y_pred)                            
        print("recall: ",rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ",f_score)

    print_scores(alg, y_test, y_pred)
    return model

# In[ ]:


# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression')

# In[ ]:


# Feature selection to improve model building
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
log = LogisticRegression()
rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)

# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))

# In[ ]:


#Saving dataframe with optimal features
X_rfe = X.iloc[:, rfecv.support_]

#Overview of the optimal features in comparison with the intial dataframe
print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())

# In[ ]:


# Splitting data with optimal features
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.3, random_state=50)  

# In[ ]:


# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression Classification')

# In[ ]:


### Trying other machine learning algorithms: SVC
svc_model = modeling(SVC, 'SVC Classification')

# In[ ]:


#Random forest
rf_model = modeling(RandomForestClassifier, "Random Forest Classification")

# In[ ]:


#Decision tree
dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")

# In[ ]:


#Naive bayes 
nb_model = modeling(GaussianNB, "Naive Bayes Classification")

# In[ ]:


## Improve best model by hyperparameter tuning
# define model
model = LogisticRegression()

# define evaluation
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
from scipy.stats import loguniform
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 1000)

# define search
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(X_rfe, y)
summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# In[ ]:


params = result.best_params_
params

# In[ ]:


#Improving the Logistic Regression model
log_model = modeling(LogisticRegression, 'Logistic Regression Classification', params=params)

# In[ ]:


#Saving best model 
import joblib
#Sava the model to disk
filename = 'model.sav'
joblib.dump(log_model, filename)

# In[ ]:




# In[ ]:



