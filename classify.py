from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

X_filename = 'converted_fc.csv'
y_filename = 'gender.csv'

Xtmp = pd.read_csv(X_filename, header=None)
# 热力图
# corrmat = Xtmp.corr()
# fig = sns.heatmap(corrmat, square = True)
# fig.get_figure().savefig('row_avg_heatmap.png')
Xtmp = Xtmp.values
X = np.array(Xtmp)
# 折线图
# x_axis = range(0, 200)
# y_axis = np.average(Xtmp, axis=0)
# plt.plot(x_axis, y_axis)
# plt.savefig('rowavg_pic.png')
ytmp = pd.read_csv(y_filename, header=None)
ytmp = ytmp.values
y = np.array(ytmp)


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# mms = MinMaxScaler()
# X_train = mms.fit_transform(X_train)
# X_test = mms.transform(X_test)

# pt = PowerTransformer()
# X_train = pt.fit_transform(X_train)
# X_test = pt.transform(X_test)

# qt = QuantileTransformer(output_distribution='normal')
# X_train = qt.fit_transform(X_train)
# X_test = qt.transform(X_test)

sc = StandardScaler()
X = sc.fit_transform(X)

# 使用PCA降低数据维度
# pca = PCA(n_components=100)
# X = pca.fit_transform(X)

# pca = PCA(n_components=100)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# detector = EllipticEnvelope()
# detector.fit(X_train)
# detector.predict(X_train)
#
# X_train[0, :] = np.nan
# imputer = KNNImputer()
# imputer.fit_transform(X_train)


# XGB
xgb = XGBClassifier()
scores_xgb = cross_val_score(xgb, X, y, cv=10, scoring='accuracy')
print(f'xgb accuracy: {scores_xgb.mean()}')
# xgb.fit(X_train, y_train)
# train_xgb = xgb.score(X_train, y_train)
# acc_xgb = round(xgb.score(X_test, y_test) * 100, 2)

# logistic regression
lr = LogisticRegression(C=0.001)
# lr.fit(X_train, y_train)
# train_log = lr.score(X_train, y_train)
# acc_log = round(lr.score(X_test, y_test) * 100, 2)
# print('acc:')
# print(acc_log)
# print('train_acc:')
# print(train_log)
scores_lr = cross_val_score(lr, X, y, cv=10, scoring='accuracy')
print(f'lr accuracy: {scores_lr.mean()}')

# KNN
knn = KNeighborsClassifier(n_neighbors=16)
scores_knn = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(f'knn accuracy: {scores_knn.mean()}')
# knn.fit(X_train, y_train)
# train_knn = knn.score(X_train, y_train)
# acc_knn = round(knn.score(X_test, y_test) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
scores_gaussian = cross_val_score(gaussian, X, y, cv=10, scoring='accuracy')
print(f'gaussian accuracy: {scores_gaussian.mean()}')
# gaussian.fit(X_train, y_train)
# train_gaussian = gaussian.score(X_train, y_train)
# acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)

# SVM
svc = SVC(kernel='linear', C=0.08, random_state=42)
scores_svc = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(f'svc accuracy: {scores_svc.mean()}')
# svc.fit(X_train, y_train)
# train_svc = svc.score(X_train, y_train)
# y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_test, y_test) * 100, 2)
# print('acc:')
# print(acc_svc)
# print('train_acc:')
# print(train_svc)

# Linear SVC
linear_svc = LinearSVC()
scores_lsvc = cross_val_score(linear_svc, X, y, cv=10, scoring='accuracy')
print(f'lsvc accuracy: {scores_lsvc.mean()}')
# linear_svc.fit(X_train, y_train)
# train_lsvc = linear_svc.score(X_train, y_train)
# acc_linear_svc = round(linear_svc.score(X_test, y_test) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier(random_state=42)
scores_sgd = cross_val_score(sgd, X, y, cv=10, scoring='accuracy')
print(f'sgd accuracy: {scores_sgd.mean()}')
# sgd.fit(X_train, y_train)
# train_sgd = sgd.score(X_train, y_train)
# acc_sgd = round(sgd.score(X_test, y_test) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')
print(f'dt accuracy: {scores_dt.mean()}')
# decision_tree.fit(X_train, y_train)
# train_decision_tree = decision_tree.score(X_train, y_train)
# acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=3, max_leaf_nodes=7, random_state=42)
scores_rf = cross_val_score(random_forest, X, y, cv=10, scoring='accuracy')
print(f'rf accuracy: {scores_rf.mean()}')
# random_forest.fit(X_train, y_train)
# train_random_forest = random_forest.score(X_train, y_train)
# random_forest.score(X_test, y_test)
# acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
#
# models = pd.DataFrame({
#     'Model': ['XGB', 'Support Vector Machines', 'KNN',
#               'Logistic Regression', 'Random Forest',
#               'Naive Bayes', 'Perceptron',
#               'Stochastic Gradient Decent', 'Linear SVC',
#               'Decision Tree'],
#     'F1 Score': [scores_xgb.mean(), scores_svc.mean(), scores_knn.mean(), scores_lr.mean(),
#               scores_rf.mean(), scores_gaussian.mean(),
#               scores_sgd.mean(), scores_lsvc.mean(), scores_dt.mean()]
#     # 'train_Score': [train_xgb, train_svc, train_knn, train_log,
#     #           train_random_forest, train_gaussian,
#     #           train_sgd, train_lsvc, train_decision_tree]
# })
# print(models.sort_values(by='F1 Score', ascending=False))

# SVM_csv = LGB.predict(X_testcsv)
# LGB_Pred_testcsv= LGB_Pred_testcsv.astype("bool")
# print(LGB_Pred_testcsv)
# output = pd.DataFrame({'PassengerId': PassengerId, 'Transported': LGB_Pred_testcsv})
#
# output.to_csv('submission.csv', index=False)

# df_result = pd.DataFrame()
# model_list = [xgb, svc, knn, lr, random_forest,
#               gaussian, perceptron, sgd, linear_svc, decision_tree]
# for i, x in enumerate(model_list):
#     scores = cross_validate(x, X, y, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'])
#     df_result.loc[i, 'test_precision_macro'] = np.mean(scores['test_precision_macro'])
#     df_result.loc[i, 'test_recall_macro'] = np.mean(scores['test_recall_macro'])
#     df_result.loc[i, 'test_f1_macro'] = np.mean(scores['test_f1_macro'])
# df_result.index = pd.Series(['XGB', 'Support Vector Machines', 'KNN',
#                                    'Logistic Regression', 'Random Forest',
#                                    'Naive Bayes', 'Perceptron',
#                                    'Stochastic Gradient Decent', 'Linear SVC',
#                                    'Decision Tree'])
# print(df_result)

