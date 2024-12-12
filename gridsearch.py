from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

X_filename = 'converted_fc_plustestdata.csv'
# X_filename = 'converted_avg_according_to_row_1159.csv'
y_filename = 'gender_new.csv'

Xtmp = pd.read_csv(X_filename, header=None)
Xtmp = Xtmp.values
X = np.array(Xtmp)
ytmp = pd.read_csv(y_filename, header=None)
ytmp = ytmp.values
y = np.array(ytmp)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)
# X_test = imp.transform(X_test)
# sc = StandardScaler()
# X = sc.fit_transform(X)

qt = QuantileTransformer(output_distribution='normal')
X = qt.fit_transform(X)

pca = PCA(n_components=100)
X = pca.fit_transform(X)

'''
# RF
param1 = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
gsearch1 =GridSearchCV(estimator= RandomForestClassifier(), param_grid=param1, scoring='f1_macro', cv=5)
gsearch1.fit(X, y)
print(gsearch1.best_params_, gsearch1.best_score_)  # 100

param2 = {'max_depth': [1, 2, 3, 4, 5, 6]}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param2, scoring='f1_macro', cv=5)
gsearch2.fit(X, y)
print(gsearch2.best_params_, gsearch2.best_score_)  # 6

param3 = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=100, max_depth=6), param_grid=param3, scoring='f1_macro', cv=5)
gsearch3.fit(X, y)
print(gsearch3.best_params_, gsearch3.best_score_)  # leaf: 7; split: 3
'''

param1 = {'C': [0.0009,0.0008,0.0007,0.0006,0.001,0.0011,0.0012]}
gsearch1 =GridSearchCV(LogisticRegression(), param_grid=param1, cv=5)
gsearch1.fit(X, y)
print(gsearch1.best_params_, gsearch1.best_score_)