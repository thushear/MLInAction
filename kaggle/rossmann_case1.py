# Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# dataFrame
rossmann_df = pd.read_csv('./input/rossman/train.csv')
store_df = pd.read_csv('./input/rossman/store.csv')
test_df = pd.read_csv('./input/rossman/test.csv')
# Preview

test_df["Open"][test_df["Open"] != test_df["Open"]] = (test_df["DayOfWeek"] != 7).astype(int)

# Date

# Create Year and Month columns
rossmann_df['Year'] = rossmann_df['Date'].apply(lambda x: int(str(x)[:4]))
rossmann_df['Month'] = rossmann_df['Date'].apply(lambda x: int(str(x)[5:7]))

test_df['Year'] = test_df['Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))

# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)
# this column will be useful in analysis and visualization
rossmann_df['Date'] = rossmann_df['Date'].apply(lambda x: (str(x)[:7]))
test_df['Date'] = test_df['Date'].apply(lambda x: (str(x)[:7]))
# StateHoliday


rossmann_df["StateHoliday"].loc[rossmann_df["StateHoliday"] == 0] = "0"
# .... continue with StateHoliday

# After
rossmann_df["StateHoliday"] = rossmann_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test_df["StateHoliday"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# Using store_df

# Merge store_df with average store sales & customers
average_sales_customers = rossmann_df.groupby('Store')[["Sales", "Customers"]].mean()

sales_customers_df = DataFrame({'Store': average_sales_customers.index,
                                'Sales': average_sales_customers["Sales"],
                                'Customers': average_sales_customers["Customers"]},
                               columns=['Store', 'Sales', 'Customers'])
store_df = pd.merge(sales_customers_df, store_df, on='Store')

# print(store_df["CompetitionDistance"])

store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())

# Notice that test_df has only year=2015, and months 8 & 9

# drop Year and Month
rossmann_df.drop(["Year", "Month"], axis=1, inplace=True)
test_df.drop(["Year", "Month"], axis=1, inplace=True)

# Create dummy varibales for DayOfWeek
day_dummies_rossmann = pd.get_dummies(rossmann_df['DayOfWeek'], prefix='Day')
print('day_dummies_rossmann\n', day_dummies_rossmann.head())
day_dummies_rossmann.drop(['Day_7'], axis=1, inplace=True)
print('day_dummies_rossmann\n', day_dummies_rossmann.head())
day_dummies_test = pd.get_dummies(test_df['DayOfWeek'], prefix='Day')
day_dummies_test.drop(['Day_7'], axis=1, inplace=True)

rossmann_df = rossmann_df.join(day_dummies_rossmann)
print('rossmann_df\n', rossmann_df.head())
test_df = test_df.join(day_dummies_test)

rossmann_df.drop(['DayOfWeek'], axis=1, inplace=True)
test_df.drop(['DayOfWeek'], axis=1, inplace=True)

# remove all rows(store,date) that were closed
rossmann_df = rossmann_df[rossmann_df["Open"] != 0]

# drop unnecessary columns, these columns won't be useful in prediction
rossmann_df.drop(["Open", "Customers", "Date"], axis=1, inplace=True)

# save ids of closed stores, because we will assign their sales value to 0 later(see below)
closed_store_ids = test_df["Id"][test_df["Open"] == 0].values

# remove all rows(store,date) that were closed
test_df = test_df[test_df["Open"] != 0]

# drop unnecessary columns, these columns won't be useful in prediction
test_df.drop(['Open', 'Date'], axis=1, inplace=True)

# Loop through each store,
# train the model using the data of current store, and predict it's sales values.

rossmann_dic = dict(list(rossmann_df.groupby('Store')))
test_dic = dict(list(test_df.groupby('Store')))
submission = Series()
scores = []

for i in test_dic:
    print('i:\n', i)
    # current store
    store = rossmann_dic[i]
    print('store:\n', store)
    # define training and testing sets
    X_train = store.drop(["Sales", "Store"], axis=1)
    Y_train = store["Sales"]
    X_test = test_dic[i].copy()

    store_ids = X_test["Id"]
    X_test.drop(["Id", "Store"], axis=1, inplace=True)
    print('X_train:\n', X_train)
    print('Y_train:\n', Y_train)
    print('X_test:\n', X_test)
    # Linear Regression
    lreg = LinearRegression()
    lreg.fit(X_train, Y_train)
    Y_pred = lreg.predict(X_test)
    print('Y_pred:\n', Y_pred)
    scores.append(lreg.score(X_train, Y_train))

    # Xgboost
    # params = {"objective": "reg:linear",  "max_depth": 10}
    # T_train_xgb = xgb.DMatrix(X_train, Y_train)
    # X_test_xgb  = xgb.DMatrix(X_test)
    # gbm = xgb.train(params, T_train_xgb, 100)
    # Y_pred = gbm.predict(X_test_xgb)

    # append predicted values of current store to submission
    submission = submission.append(Series(Y_pred, index=store_ids))

# append rows(store,date) that were closed, and assign their sales value to 0
submission = submission.append(Series(0, index=closed_store_ids))
print('submission:\n', submission.head())
# save to csv file
submission = pd.DataFrame({"Id": submission.index, "Sales": submission.values})
submission.head()
print('scores:\n', scores)
# submission.to_csv('rossmann.csv', index=False)