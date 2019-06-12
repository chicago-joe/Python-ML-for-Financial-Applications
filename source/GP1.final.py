# IE598 Machine Learning in Finance, Fall 2018
# University of Illinois at Urbana-Champaign
#
# Final Group Project
#
# Authors: Joseph Loss, Ruozhong Yang, Fengkai Xu, Biao Feng, and Yuchen Duan
#
# source code available at https://github.com/chicago-joe/Machine-Learning-in-Finance-Final-Project
# --------------------------------------------------------------------------------
# Model Outline:
# 1) Exploratory Data Analysis
# 2) Preprocessing, feature extraction, feature selection
# 3) Model fitting and evaluation, (you should fit at least 3 different machine learning models)
# 4) Hyperparameter tuning
# 5) Ensembling
# --------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import six
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1) Exploratory Data Analysis
df = pd.read_excel('C:\\Users\\jloss\\PyCharmProjects\\Machine-Learning-in-Finance-Final-Project\\data\\GP1_CreditScore.xlsx', sep = ',')
df.shape
df.info()
df.head()
df.describe()

cols = ['Sales/Revenues', 'Gross Margin', 'EBITDA', 'EBITDA Margin', 'Net Income Before Extras',
        'Total Debt', 'Net Debt', 'LT Debt', 'ST Debt', 'Cash', 'Free Cash Flow', 'Total Debt/EBITDA',
        'Net Debt/EBITDA', 'Total MV', 'Total Debt/MV', 'Net Debt/MV', 'CFO/Debt', 'CFO',
        'Interest Coverage', 'Total Liquidity', 'Current Liquidity', 'Current Liabilities',
        'EPS Before Extras', 'PE', 'ROA', 'ROE', 'InvGrd']

# correlation matrix
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 0.5)
hm = sns.heatmap(cm,
                 cbar = True,
                 annot = True,
                 square = True,
                 fmt = '.2f',
                 annot_kws = { 'size':3 },
                 yticklabels = cols,
                 xticklabels = cols)
plt.savefig('correlation matrix.png', dpi = 3000)
plt.show()

X = df[['Sales/Revenues', 'Gross Margin', 'EBITDA', 'EBITDA Margin', 'Net Income Before Extras',
        'Total Debt', 'Net Debt', 'LT Debt', 'ST Debt', 'Cash', 'Free Cash Flow', 'Total Debt/EBITDA',
        'Net Debt/EBITDA', 'Total MV', 'Total Debt/MV', 'Net Debt/MV', 'CFO/Debt','CFO',
        'Interest Coverage', 'Total Liquidity', 'Current Liquidity', 'Current Liabilities',
        'EPS Before Extras', 'PE', 'ROA', 'ROE']].values

y = df['InvGrd'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 2) Preprocessing, feature extraction, feature selection
# Select the feature(with importance)
forest = RandomForestClassifier(criterion = 'gini', n_estimators = 100, random_state = 42, n_jobs = 2)
forest.fit(X_train_std, y_train)

print(forest.feature_importances_)
print(X_train_std.shape)

model = SelectFromModel(forest, prefit = True)
X_train_std = model.transform(X_train_std)
X_test_std = model.transform(X_test_std)

print(X_test_std.shape)
print(X_train_std.shape)

# 3) Model fitting and evaluation, (you should fit at least 3 different machine learning models)

# 4) Hyperparameter tuning
# KNN
knn = KNeighborsClassifier()
params_knn = {
    'n_neighbors':range(1, 101)
}
grid_knn = GridSearchCV(estimator = knn,
                        param_grid = params_knn,
                        scoring = 'accuracy',
                        cv = 10,
                        n_jobs = -1)

grid_knn.fit(X_train_std, y_train)
best_model_knn = grid_knn.best_estimator_

print(best_model_knn.score(X_test_std, y_test))

# Random Forest
forest = RandomForestClassifier()
params_forest = {
    'criterion':['gini'],
    'n_estimators':range(1, 101),
    'random_state':[42]
}
grid_forest = GridSearchCV(estimator = forest,
                           param_grid = params_forest,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_forest.fit(X_train_std, y_train)
best_model_forest = grid_forest.best_estimator_

print(best_model_forest.score(X_test_std, y_test))

# Decision Tree
tree = DecisionTreeClassifier()
params_tree = {
    'criterion':['gini'],
    'max_depth':range(1, 101),
    'random_state':[42]
}
grid_tree = GridSearchCV(estimator = tree,
                         param_grid = params_tree,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs = -1)

grid_tree.fit(X_train_std, y_train)
best_model_tree = grid_tree.best_estimator_

print(best_model_tree.score(X_test_std, y_test))

# Logistic Regression
lr = LogisticRegression(max_iter = 1000,solver = 'lbfgs')
params_lr = {
    'C':range(1, 101),
    'random_state':[42]
}
grid_lr = GridSearchCV(estimator = lr,
                       param_grid = params_lr,
                       scoring = 'accuracy',
                       cv = 10,
                       n_jobs = -1)

grid_lr.fit(X_train_std, y_train)
best_model_lr = grid_lr.best_estimator_

print(best_model_lr.score(X_test_std, y_test))

# 5) Ensembling
# Majority Vote Classifier
class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    """ A majority vote ensemble classifier
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble
    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).
    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.
    """

    def __init__(self, classifiers, vote = 'classlabel', weights = None):

        self.classifiers = classifiers
        self.named_classifiers = { key:value for key, value
                                   in _name_estimators(classifiers) }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        y : array-like, shape = [n_samples]
            Vector of target class labels.
        Returns
        -------
        self : object
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis = 1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                    lambda x:
                    np.argmax(np.bincount(x,
                                          weights = self.weights)),
                    axis = 1,
                    arr = predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)
        return avg_proba

    def get_params(self, deep = True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep = True)):
                    out['%s__%s' % (name, key)] = value
            return out

# Ensembling
clf1 = grid_knn.best_estimator_
clf2 = grid_forest.best_estimator_
clf3 = grid_tree.best_estimator_

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
clf_labels = ['KNN', 'RandomForest', 'Decision tree']

print('10-fold cross validation:\n')

mv_clf = MajorityVoteClassifier(classifiers = [pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf, X = X_train_std, y = y_train, cv = 10, scoring = 'roc_auc')

    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# muti
X = df[['Sales/Revenues', 'Gross Margin', 'EBITDA', 'EBITDA Margin', 'Net Income Before Extras',
        'Total Debt', 'Net Debt', 'LT Debt', 'ST Debt', 'Cash', 'Free Cash Flow', 'Total Debt/EBITDA',
        'Net Debt/EBITDA', 'Total MV', 'Total Debt/MV', 'Net Debt/MV', 'CFO/Debt','CFO', 'Interest Coverage',
        'Total Liquidity', 'Current Liquidity', 'Current Liabilities', 'EPS Before Extras', 'PE', 'ROA', 'ROE']].values

y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify = y)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# KNN
knn = KNeighborsClassifier()
params_knn = {
    'n_neighbors':range(1, 101)
}
grid_knn = GridSearchCV(estimator = knn,
                        param_grid = params_knn,
                        scoring = 'accuracy',
                        cv = 10,
                        n_jobs = -1)

grid_knn.fit(X_train_std, y_train)
best_model_knn_muti = grid_knn.best_estimator_

print('muti=' + str(best_model_knn_muti.score(X_test_std, y_test)))

# RandomForest
forest = RandomForestClassifier()
params_forest = {
    'criterion':['gini'],
    'n_estimators':range(1, 101),
    'random_state':[42]
}
grid_forest = GridSearchCV(estimator = forest,
                           param_grid = params_forest,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_forest.fit(X_train_std, y_train)
best_model_forest_muti = grid_forest.best_estimator_

print('muti=' + str(best_model_forest_muti.score(X_test_std, y_test)))

# DecisionTree
tree = DecisionTreeClassifier()
params_tree = {
    'criterion':['gini'],
    'max_depth':range(1, 101),
    'random_state':[42]
}
grid_tree = GridSearchCV(estimator = tree,
                         param_grid = params_tree,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs = -1)

grid_tree.fit(X_train_std, y_train)
best_model_tree_muti = grid_tree.best_estimator_

print('muti=' + str(best_model_tree_muti.score(X_test_std, y_test)))

# logistic Regression
lr = LogisticRegression(max_iter = 1000, solver = 'lbfgs')
params_lr = {
    'C':range(1, 101),
    'random_state':[42]
}
grid_lr = GridSearchCV(estimator = lr,
                       param_grid = params_lr,
                       scoring = 'accuracy',
                       cv = 10,
                       n_jobs = -1)

grid_lr.fit(X_train_std, y_train)
best_model_lr_muti = grid_lr.best_estimator_

print('muti=' + str(best_model_lr_muti.score(X_test_std, y_test)))

# This part select the feature use for fitting(but seem to make no sense)
# If you want to use it, run it before the StandardScaler after the model

