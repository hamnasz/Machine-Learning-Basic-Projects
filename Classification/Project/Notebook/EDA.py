import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import plotly.express as px
import matplotlib.pyplot as plt
data_df = pd.read_csv("/workspaces/Machine-Learning-Basic-Projects/Classification/Project/Data/churn.csv")
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
import plotly.express as px
target_instance = data_df["Churn"].value_counts().reset_index()
target_instance.columns = ['Category', 'Count']
fig = px.pie(
    target_instance,
    values='Count',
    names='Category',
    color='Category',
    color_discrete_sequence=["#FFFF99", "#FFF44F"],
    color_discrete_map={"No": "#E30B5C", "Yes": "#50C878"},
    title='Distribution of Churn'
)
fig.show()
def bar(feature, df=data_df ):
    temp_df = df.groupby([feature, 'Churn']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]
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
    num_str = num_format(percentage)
    cat_str = str_format(categories)
    churn_colors = ["#FFFF99", "#FFF44F"]
    gender_colors = {"Female": "#E30B5C", "Male": "#50C878"}
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
bar('gender')
data_df.loc[data_df.SeniorCitizen==0,'SeniorCitizen'] = "No"
data_df.loc[data_df.SeniorCitizen==1,'SeniorCitizen'] = "Yes"
bar('SeniorCitizen')
bar('Partner')
bar('Dependents')
bar('PhoneService')
bar('MultipleLines')
bar('InternetService')
bar('OnlineSecurity')
bar('OnlineBackup')
bar('DeviceProtection')
bar('TechSupport')
bar('StreamingTV')
bar('StreamingMovies')
bar('Contract')
bar('PaperlessBilling')
bar('PaymentMethod')
data_df.dtypes
try:
    data_df['TotalCharges'] = data_df['TotalCharges'].astype(float)
except ValueError as ve:
    print (ve)
data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())
def hist(feature):
    group_df = data_df.groupby([feature, 'Churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Churn', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    fig.show()
hist('tenure')
hist('MonthlyCharges')
hist('TotalCharges')
bin_df = pd.DataFrame()
bin_df['tenure_bins'] =  pd.qcut(data_df['tenure'], q=3, labels= ['low', 'medium', 'high'])
bin_df['MonthlyCharges_bins'] =  pd.qcut(data_df['MonthlyCharges'], q=3, labels= ['low', 'medium', 'high'])
bin_df['TotalCharges_bins'] =  pd.qcut(data_df['TotalCharges'], q=3, labels= ['low', 'medium', 'high'])
bin_df['Churn'] = data_df['Churn']
bar('tenure_bins', bin_df)
bar('MonthlyCharges_bins', bin_df)
bar('TotalCharges_bins', bin_df)
data_df.drop(["customerID"],axis=1,inplace = True)
def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})
data_df['Churn'] = data_df[['Churn']].apply(binary_map)
data_df['gender'] = data_df['gender'].map({'Male':1, 'Female':0})
binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
data_df[binary_list] = data_df[binary_list].apply(binary_map)
data_df = pd.get_dummies(data_df, drop_first=True)
corr = data_df.corr()
fig = px.imshow(corr,width=1000, height=1000)
fig.show()
import statsmodels.api as sm
import statsmodels.formula.api as smf
all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in data_df.columns]
data_df.columns = all_columns
glm_columns = [e for e in all_columns if e not in ['customerID', 'Churn']]
glm_columns = ' + '.join(map(str, glm_columns))
glm_model = smf.glm(formula=f'Churn ~ {glm_columns}', data=data_df, family=sm.families.Binomial())
res = glm_model.fit()
print(res.summary())
np.exp(res.params)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data_df['tenure'] = sc.fit_transform(data_df[['tenure']])
data_df['MonthlyCharges'] = sc.fit_transform(data_df[['MonthlyCharges']])
data_df['TotalCharges'] = sc.fit_transform(data_df[['TotalCharges']])
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
X = data_df.drop('Churn', axis=1)
y = data_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
def modeling(alg, alg_name, params={}):
    model = alg(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
log_model = modeling(LogisticRegression, 'Logistic Regression')
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
log = LogisticRegression()
rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()
print("The optimal number of features: {}".format(rfecv.n_features_))
X_rfe = X.iloc[:, rfecv.support_]
print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.3, random_state=50)
log_model = modeling(LogisticRegression, 'Logistic Regression Classification')
svc_model = modeling(SVC, 'SVC Classification')
rf_model = modeling(RandomForestClassifier, "Random Forest Classification")
dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")
nb_model = modeling(GaussianNB, "Naive Bayes Classification")
model = LogisticRegression()
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
from scipy.stats import loguniform
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 1000)
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
result = search.fit(X_rfe, y)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
params = result.best_params_
log_model = modeling(LogisticRegression, 'Logistic Regression Classification', params=params)
import joblib
filename = 'model.sav'
joblib.dump(log_model, filename)