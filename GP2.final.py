# IE598 Machine Learning in Finance, Fall 2018
# Final Group Project
# Authors: Joseph Loss, Ruozhong Yang, Fengkai Xu, Biao Feng, and Yuchen Duan

# source code available at https://github.com/chicago-joe/IE598_F18_MLF_GP
####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import pipeline
from sklearn.linear_model import LinearRegression
import scipy as sp
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
df = pd.read_excel('C:\\Users\\43739\\OneDrive\\us\\IE598\\group_project\\MLF_GP2_EconCycle.xlsx',sep=',')
cols = ['T1Y Index', 'T2Y Index', 'T3Y Index', 'T5Y Index', 'T7Y Index', 'T10Y Index', 'CP1M', 'CP3M', 'CP6M',
        'CP1M_T1Y',
        'CP3M_T1Y', 'CP6M_T1Y', 'USPHCI', 'PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD']

## Exploratory Data Analysis
df.dropna(inplace=True)
print(df.shape, df.info(), df.describe(), df.head())

CPTcols = ['CP1M_T1Y', 'CP3M_T1Y', 'CP6M_T1Y', 'USPHCI']
sns.pairplot(df[CPTcols], dropna=True, )
# plt.tight_layout()
# plt.savefig('E:\Study\Courses\Fall 2018\IE 598\IE598 Homework\Group Project\scatter_GP2_.png',dpi = 500)
plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=False,
                 annot=True,
                 square=False,
                 fmt='.1f',
                 annot_kws={'size': 8},
                 yticklabels=cols,
                 xticklabels=cols)
# plt.tight_layout()
# plt.savefig('E:\Study\Courses\Fall 2018\IE 598\IE598 Homework\Group Project\heatmap_rate_GP2_.png',dpi = 15000)
plt.show()

## 3- month prediction and model
X = np.array(df.drop(['USPHCI', 'PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD'], 1))
y = np.array(df['PCT 3MO FWD'])
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.1, random_state=42)

# feature importance
feat_labels = cols[:-4]
forest = RandomForestRegressor(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
print("3MO FWD RATE - Feature Importance")
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


print('\n')
plt.title('Feature Importance PCT 3MO FWD')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#Selection
model = SelectFromModel(forest, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
print(X_test.shape)
print(X_train.shape)

#LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=1, color='black', lw=2)
plt.xlim([0, 1])
plt.savefig('LinearRegression.png', dpi=300)
plt.show()
print('(LR)MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('(LR)R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
print('(LR)Slope: %.3f' % reg.coef_[0])
print('(LR)Intercept: %.3f' % reg.intercept_)


## 6 month prediction and model
X = np.array(df.drop(['USPHCI', 'PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD'], 1))
y = np.array(df['PCT 6MO FWD'])
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.1, random_state=42)
feat_labels = cols[:-4]

forest = RandomForestRegressor(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

print("6MO FWD RATE - Feature Importance")
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


print('\n')
plt.title('Feature Importance PCT 6MO FWD')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#Selection
model = SelectFromModel(forest, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
print(X_test.shape)
print(X_train.shape)

#RidgeRegression
alpha_space = np.logspace(-3, 0, 4)
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha   
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=1, color='black', lw=2)
    plt.xlim([0, 1])
    plt.savefig('Ridge(alpha='+str(alpha)+' ).png', dpi=300)
    plt.show()
    print('Ridgealpha: %.3f' %(alpha))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print('Slope: %.3f' % ridge.coef_[0])
    print('Intercept: %.3f' % ridge.intercept_)
    
## 9-month prediction & model
X = np.array(df.drop(['USPHCI', 'PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD'], 1))
y = np.array(df['PCT 9MO FWD'])
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.1, random_state=42)
feat_labels = cols[:-4]

forest = RandomForestRegressor(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

print("9MO FWD RATE - Feature Importance")
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
print('\n')
plt.title('Feature Importance: PCT 9MO FWD ')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#Selection
model = SelectFromModel(forest, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
print(X_test.shape)
print(X_train.shape)

#LassoRegression
alpha_space = np.logspace(-6, -3, 4)    
lasso = Lasso(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    lasso.alpha = alpha   
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=1, color='black', lw=2)
    plt.savefig('Lasso(alpha='+str(alpha)+' ).png', dpi=300)
    plt.xlim([0, 1])
    plt.show()
    print('Lassoalpha: %.6f' %(lasso.alpha))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print('Slope: %.3f' % lasso.coef_[0])
    print('Intercept: %.3f' % lasso.intercept_)

## # Part 5 - Ensemble Learning
# Set seed for reproducibility
SEED = 1

# Split dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.1, random_state=42)

# Instantiate a GradientBoostingRegressor 'gbr'
gbr = GradientBoostingRegressor(max_features=4, learning_rate=0.1, n_estimators=500,
                                subsample=0.3, random_state=42)
gbr.fit(X_train, y_train)
# Predict the test set labels
y_pred = gbr.predict(X_test)

# Evaluate the test set RMSE
mse = MSE(y_test, y_pred)
rsquared = r2_score(y_test, y_pred)

# Print the test set RMSE
print('\n')
print('Test set MSE: {:.2f}'.format(mse))
print('Test set R-Squared: {:.2f}'.format(rsquared))
