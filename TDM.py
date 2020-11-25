# PROBLEM
# ACME is a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection,
# sentiment classification and customer intention prediction and classification.
#
# We are interested in developing a robust machine learning system that leverages information coming from call center data.
#
# Ultimately, at ACME we are looking to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving
# machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.
#
# Data Description:
#
# The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription,
# in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can
# withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.
#
# Attributes:
#
# age : age of customer (numeric)
#
# job : type of job (categorical)
#
# marital : marital status (categorical)
#
# education (categorical)
#
# default: has credit in default? (binary)
#
# balance: average yearly balance, in euros (numeric)
#
# housing: has a housing loan? (binary)
#
# loan: has personal loan? (binary)
#
# contact: contact communication type (categorical)
#
# day: last contact day of the month (numeric)
#
# month: last contact month of year (categorical)
#
# duration: last contact duration, in seconds (numeric)
#
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#
# Output (desired target):
#
# y - has the client subscribed to a term deposit? (binary)
#
#
# Goal(s):
#
# Predict if the customer will subscribe (yes/no) to a term deposit (variable y)


# Data Understanding

# imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from datetime import datetime
import datetime as dt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# To display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)

def load_tdm_data():
    df = pd.read_csv(r'datasets/term-deposit-marketing-2020.csv')
    return df

df = load_tdm_data()

df_bck = df.copy()

# EDA

# OVERVIEW

print(df.head())

print(df.tail())

print(df.info())

print(df.columns)

print(df.shape)

print(df.index)

print(df.describe().T)

print(df.isnull().values.any())

print(df.isnull().sum().sort_values(ascending=False))


# Drop default column

df.drop('default', axis=1, inplace=True)

# CATEGORICAL VARIABLE ANALYSIS

# WHAT ARE THE NAMES OF CATEGORICAL VARIABLES?

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical Variable count: ', len(cat_cols))
print(cat_cols)

# for col in cat_cols:
#     df[col] = df[col].astype('category')

# HOW MANY CLASSES DO CATEGORICAL VARIABLES HAVE?

print(df[cat_cols].nunique())

def cats_summary(data, categorical_cols, number_of_classes=15):
    var_count = 0  # count of categorical variables will be reported
    vars_more_classes = []  # categorical variables that have more than a number specified.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # choose according to class count
                print(pd.DataFrame({var: data[var].value_counts(), "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                sns.countplot(x=var, data=data)
                plt.show()
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)

cats_summary(df, cat_cols)

# There is an imbalance problem for y column!!!

# Convert y to 1 and 0
df['y'] = df['y'].replace('yes',1)
df['y'] = df['y'].replace('no',0)

# NUMERICAL VARIABLE ANALYSIS

print(df.describe().T)

# NUMERICAL VARIABLES COUNT OF DATASET?

df.info()

num_cols = [col for col in df.columns if df[col].dtypes != 'O']
print('Numerical Variables Count: ', len(num_cols))
print('Numerical Variables: ', num_cols)

# Histograms for numerical variables?

def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")

hist_for_nums(df, num_cols)

# DISTRIBUTION OF "y" VARIABLE

print(df["y"].value_counts()) #inbalancing problem!

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target);
    facet.add_legend()

# TARGET ANALYSIS BASED ON CATEGORICAL VARIABLES

def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 15 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
        plot_categories(df, cat=var, target='y')
        plt.show()

target_summary_with_cat(df, "y")

# TARGET ANALYSIS BASED ON NUMERICAL VARIABLES

def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")

target_summary_with_nums(df, "y")


def correlation_matrix(df):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()

correlation_matrix(df)

def find_correlation(dataframe, corr_limit=0.30):
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "y":
            pass

        else:
            correlation = dataframe[[col, "y"]].corr().loc[col, "y"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col)
            else:
                low_correlations.append(col)
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df)

print('Variables have low correlation with target:')
print('-' * 44)
print(low_corrs)
print('Variables have high correlation with target:')
print('-' * 44)
print(high_corrs)

# WORK WITH OUTLIERS

# LOF (Local Outlier Factor) has been applied

clf = LocalOutlierFactor(n_neighbors = 20, contamination=0.1)

clf.fit_predict(df[df[num_cols].columns.difference(["y"])])

df_scores = clf.negative_outlier_factor_

# np.sort(df_scores)[0:1000]

threshold = np.sort(df_scores)[100]

outlier_tbl = df_scores > threshold

press_value = df[df_scores == threshold]
outliers = df[~outlier_tbl]

res = outliers.to_records(index = False)
res[:] = press_value.to_records(index = False)

df[~outlier_tbl] = pd.DataFrame(res, index = df[~outlier_tbl].index)

# Is there any missing values?

print(df.isnull().values.any())  # NO!

# LABEL ENCODING

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

df = label_encoder(df)

# ONE-HOT ENCODING

def one_hot_encoder(dataframe, category_freq=20, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe

df = one_hot_encoder(df)

# SCALING

def Calc_IQR(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    return interquantile_range


def my_robust_scaler(data, col):
    IQR = Calc_IQR(data, col)
    data[col] = data[col].apply(lambda x: (x - data[col].median())
                / IQR)

Robust_columns = list(df[num_cols].columns.difference(["y"]))

# Call Function

for col in Robust_columns:
    my_robust_scaler(df, col)

# MODELLING

y = df["y"]
X = df.drop(["y"], axis=1)

# Base model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('Base LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm.predict(X_test))), '\n')
print(classification_report(y_test, lgbm.predict(X_test)))
lgbm_y_pred = lgbm.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Subscribed", "Not subscribed"] , yticklabels = ["Subscribed", "Not subscribed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# FIX IMBALANCED DATA PROBLEM

X_80, X_20 = train_test_split(df, test_size=0.2, random_state=123) # Splitting 20 percent of data for prediction

# Shuffling data
shuffled_df = X_80.sample(frac=1,random_state=4)

# Create a seperate dataframe for customers subscribed term project
yes_df = shuffled_df.loc[shuffled_df['y'] == 1]

# % 92.76 of dataset is consisting of no!
# Pick random samples (specified count) from majority class
no_df = shuffled_df.loc[shuffled_df['y'] == 0].sample(n=int(df.y.value_counts()[1]*0.78),random_state=123)


# Concat datasets of two classes
balanced_df = pd.concat([yes_df, no_df])

print((balanced_df.y.value_counts()/balanced_df.y.count()))

plt.bar(['No', 'Yes'], balanced_df.y.value_counts().values, facecolor = 'brown', edgecolor='brown', linewidth=0.5, ls='dashed')
sns.set(font_scale=1)
plt.title('Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()

# Base model for balanced dataframe

y = balanced_df["y"]
X = balanced_df.drop(["y"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('LGBM Base Balanced Data Accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm.predict(X_test))), '\n')
print(classification_report(y_test, lgbm.predict(X_test)))
lgbm_y_pred = lgbm.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Subscribed", "Not Subscribed"] , yticklabels = ["Subscribed", "Not Subscribed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('LightGBM Base Balanced Data Model')
plt.show()
# Accuracy decreased, but, our precision and recall scores increased and this is more important for the model being trustworthy!

# Model Tuning

lgbm_params = {
        'n_estimators': [50, 100, 500],
        'subsample': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.08, 0.01, 0.05, 0.1, 0.2],
        "min_child_samples": [10, 20, 30]}

lgbm_model = LGBMClassifier(random_state=123)

# print('LGBM Tuned Starting Time: ', datetime.now())

gs_cv = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X, y)

# print('LGBM Tuned Ending Time: ', datetime.now())

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)


print(gs_cv.best_params_)

print('LGBM Tuned Accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm_tuned.predict(X_test))), '\n')
print('LGBM Tuned Best params: ', gs_cv.best_params_, '\n')
print(classification_report(y_test, lgbm_tuned.predict(X_test)))
lgbm_y_pred = lgbm_tuned.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Subscribed", "Not subscribed"] , yticklabels = ["Subscribed", "Not Subscribed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('LightGBM Tuned Balanced Data Model')
plt.show()

# Feature Importance

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.figure(figsize=(10, 10))
# plt.xlabel('Feature Importance Scores')
# plt.ylabel('Features')
# plt.title("Feature Importance Levels")
plt.show()
plt.savefig('lgbm_importances.png')

# RESULTS
# Base LGBM accuracy: 0.942
#               precision    recall  f1-score   support
#            0       0.96      0.98      0.97      7462
#            1       0.59      0.43      0.50       538
#     accuracy                           0.94      8000
#    macro avg       0.77      0.71      0.73      8000
# weighted avg       0.93      0.94      0.94      8000


# LGBM Base Balanced Data Accuracy: 0.872
#               precision    recall  f1-score   support
#            0       0.89      0.84      0.87       448
#            1       0.85      0.90      0.88       467
#     accuracy                           0.87       915
#    macro avg       0.87      0.87      0.87       915
# weighted avg       0.87      0.87      0.87       915

# LGBM Tuned Accuracy: 0.926
# LGBM Tuned Best params:  {'learning_rate': 0.08, 'max_depth': 6, 'min_child_samples': 20, 'n_estimators': 100, 'subsample': 0.01}
#               precision    recall  f1-score   support
#            0       0.96      0.88      0.92       448
#            1       0.90      0.97      0.93       467
#     accuracy                           0.93       915
#    macro avg       0.93      0.92      0.93       915
# weighted avg       0.93      0.93      0.93       915


# BONUS1 - FEATURE IMPORTANCE
# When we check correlation matrix, and also feature importance levels of light gbm model, it is very clear that,
# THE MOST IMPORTANT AND AT THE SAME TIME THE ONLY HIGHLY CORRELATED FEATURE WITH TARGET IS DURATION!!!

# BONUS2 - CUSTOMER SEGMENTATION
# Now, my aim is to segment customers same as creating a RFM segmentation. Choosing clustering algorithms like K-means or hierarchical clustering,is an option
# as well, however RFM is more interesting for me that's why my preference is RFN segmentation
# Here I've created dummy customer IDs. Recency could be obtained. (last call date)
# Frequency will be the same with campaign column and it will tell us about the frequency of contacts performed during this campaign.
# The most important feature of this data set that is highly correlated with target is Duration, that's why we should precisely use this information, 
# therefore i'll take it as it as monetary column(Last contact duration in sec. )

df = load_tdm_data()

# Add dummy customer IDs
df['Customer_ID'] = list(range(1,40001))

# Convert month to number
df.head()
df['month'] = df['month'].replace('jan', 1)
df['month'] = df['month'].replace('feb', 2)
df['month'] = df['month'].replace('mar', 3)
df['month'] = df['month'].replace('apr', 4)
df['month'] = df['month'].replace('may', 5)
df['month'] = df['month'].replace('jun', 6)
df['month'] = df['month'].replace('jul', 7)
df['month'] = df['month'].replace('aug', 8)
df['month'] = df['month'].replace('sep', 9)
df['month'] = df['month'].replace('oct', 10)
df['month'] = df['month'].replace('nov', 11)
df['month'] = df['month'].replace('dec', 12)


# Function to Create datetime column
def create_date(a, b):
    return dt.datetime(2020, b, a)

df['Date_'] = df.apply(lambda row: create_date(row['day'],row['month']), axis=1)

# RECENCY

df['Date_'].min() # Timestamp('2020-01-28 00:00:00')

df['Date_'].max() # Timestamp('2020-12-27 00:00:00')

today_date = dt.datetime(2020, 12, 27) # Assumed today's date is the nearest call date in data

df.groupby("Customer_ID").agg({"Date_":"max"}).head() # customers' nearest visiting dates


today_date - df.groupby(df["Customer_ID"]).agg({"Date_":'max'})
# Broadcasting feature has been used for subtracting nearest invoice date from today's date

temp_df = (today_date - df.groupby("Customer_ID").agg({"Date_":"max"}))

temp_df.rename(columns = {"Date_":"Recency"}, inplace = True) # Recency for all customers

recency_df = temp_df["Recency"].apply(lambda x: x.days) #day value calculation for all customers

recency_df = pd.DataFrame({'Recency': recency_df})

recency_df.head()

# FREQUENCY

freq_df = pd.DataFrame({'Frequency': df['campaign']})

# DURATION (Monetary)

duration_df = pd.DataFrame({'Duration': df['duration']})

duration_df.head()

print(recency_df.shape,freq_df.shape,duration_df.shape)

rfd = pd.concat([recency_df, freq_df, duration_df],  axis=1) # RFD dataframe

rfd = rfd.dropna()

for col in rfd.columns:
    rfd[col] = rfd[col].astype('int')

rfd.describe().T

def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")

hist_for_nums(rfd, rfd.columns)

rfd.isnull().values.any()

rfd.info()

rfd.head()

rfd['Recency'] = rfd['Recency'].astype('int64')
rfd['Frequency'] = rfd['Frequency'].astype('int64')
rfd['Duration'] = rfd['Duration'].astype('int64')

# Labeling RFD scores according to RFD metrics

rfd["RecencyScore"] = pd.qcut(rfd["Recency"], 5, labels = [5, 4 , 3, 2, 1])
# Sorted and calculated 5 quantile values for Recency  and labeled (descending, this is important!)

rfd["FrequencyScore"]= pd.qcut(rfd["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
# Sorted and calculated 5 quantile values for Frequency  and labeled (ascending, this is important!)

rfd["DurationScore"] = pd.qcut(rfd['Duration'], 5, labels = [1, 2, 3, 4, 5])
# Sorted and calculated 5 quantile values for Duration  and labeled (ascending, this is important!)

# Calculate Recency Score
rfd["RFD_SCORE"] = (rfd['RecencyScore'].astype(str) +
                    rfd['FrequencyScore'].astype(str) +
                    rfd['DurationScore'].astype(str))

# RFM Map via Regular Expressions, For this mapping we should definitely specify segments with the help of business domain, this is just an example..
seg_map = {
    r'[1-5][1-2][1-2]': 'Hibernating',
    r'[1-5][1-2][3-4]': 'At Risk',
    r'[1-5][1-2]5': 'Can\'t Loose',
    r'[1-5]3[1-2]': 'About to Sleep',
    r'[1-5]33': 'Need Attention',
    r'[1-5][3-4][4-5]': 'Loyal Customers',
    r'[1-5]41': 'Promising',
    r'[1-5]51': 'New Customers',
    r'[1-5][4-5][2-3]': 'Potential Loyalists',
    r'[1-5]5[4-5]': 'Champions'
}
rfd['Segment'] = rfd['RFD_SCORE'].replace(seg_map, regex=True)  # Replace Segment values with Regex map

print(rfd.groupby('Segment').agg({'Recency': 'count'}))

# Now it is possible to reach to customers whichever segment they belong to, export them and we may take the proper action! like below
print(rfd[rfd['Segment'] == 'Champions'].index)  # Customer IDs of Champions

Champions_df = pd.DataFrame()

Champions_df['Champion_CustomerID'] = rfd[rfd['Segment'] == 'Champions'].index

Champions_df.to_csv("Champion_customers.csv", index = False) # Save Champion Customers as dataframe to share with people



