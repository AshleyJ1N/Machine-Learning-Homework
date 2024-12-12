from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

X_filename = f'F:\\pyprojects\\fMRI_classification\\converted_fc.csv'
y_filename = f'F:\\pyprojects\\fMRI_classification\\age.csv'

Xtmp = pd.read_csv(X_filename, header=None)

Xtmp = Xtmp.values
X = np.array(Xtmp)
ytmp = pd.read_csv(y_filename, header=None)
ytmp = ytmp.values
y = np.array(ytmp)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# pt = PowerTransformer()
# X_train = pt.fit_transform(X_train)
# X_test = pt.transform(X_test)
# y_train = pt.fit_transform(y_train)
# y_test = pt.transform(y_test)

qt = QuantileTransformer(output_distribution='normal')
X = qt.fit_transform(X)
# X_train = qt.fit_transform(X_train)
# X_test = qt.transform(X_test)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# mms = MinMaxScaler()
# X_train = mms.fit_transform(X_train)
# X_test = mms.transform(X_test)

# 使用PCA降低数据维度
pca = PCA(n_components=100)
X = pca.fit_transform(X)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# 计算解释方差
# explained_variance = pca.explained_variance_
# 归一化解释方差
# explained_variance_ratio = explained_variance / np.sum(explained_variance)
# 计算解释方差的累积和
# cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
# 绘制维度数量与解释方差累积和的函数图
# plt.plot(np.arange(1, len(cumulative_explained_variance_ratio)+1), cumulative_explained_variance_ratio, 'b-')
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.title('Explained Variance Cumulative Sum')
# plt.grid(True)
# plt.show()

# SVR
# svr = SVR(kernel='linear', C=1)
# scores_svr = cross_val_score(svr, X, y, cv=10, scoring='r2')
# print(f'SVC r2: {scores_svr.mean()}')
# svr.fit(X_train, y_train)
# y_pred_svr = svr.predict(X_test)
# mse_svr = mean_squared_error(y_test, y_pred_svr)
# rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
# r2_svr = r2_score(y_test, y_pred_svr)
# print('svr:')
# print(f'MSE: {mse_svr:.2f}')
# print(f'RMSE: {rmse_svr:.2f}')
# print(f'R2: {r2_svr:.2f}')

# KNN
# from sklearn.neighbors import KNeighborsRegressor
# knn = KNeighborsRegressor()
# scores_knn = cross_val_score(knn, X, y, cv=10, scoring='r2')
# print(f'KNN r2: {scores_knn.mean()}')
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)
# mse_knn = mean_squared_error(y_test, y_pred_knn)
# rmse_knn = mean_squared_error(y_test, y_pred_knn, squared=False)
# r2_knn = r2_score(y_test, y_pred_knn)
# print('knn:')
# print(f'MSE: {mse_knn:.2f}')
# print(f'RMSE: {rmse_knn:.2f}')
# print(f'R2: {r2_knn:.2f}')

# ridge
# from sklearn.linear_model import Ridge
# ridge = Ridge()
# scores_ridge = cross_val_score(ridge, X, y, cv=10, scoring='r2')
# print(f'Ridge r2: {scores_ridge.mean()}')
# ridge.fit(X_train, y_train)
# y_pred_ridge = ridge.predict(X_test)
# mse_ridge = mean_squared_error(y_test, y_pred_ridge)
# rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)
# r2_ridge = r2_score(y_test, y_pred_ridge)
# print('ridge:')
# print(f'MSE: {mse_ridge:.2f}')
# print(f'RMSE: {rmse_ridge:.2f}')
# print(f'R2: {r2_ridge:.2f}')

# krr
# from sklearn.kernel_ridge import KernelRidge
# krr = KernelRidge()
# scores_krr = cross_val_score(krr, X, y, cv=10, scoring='r2')
# print(f'KernelRridge r2: {scores_krr.mean()}')
# krr.fit(X_train, y_train)
# y_pred_krr = krr.predict(X_test)
# mse_krr = mean_squared_error(y_test, y_pred_krr)
# rmse_krr = mean_squared_error(y_test, y_pred_krr, squared=False)
# r2_krr = r2_score(y_test, y_pred_krr)
# print('krr:')
# print(f'MSE: {mse_krr:.2f}')
# print(f'RMSE: {rmse_krr:.2f}')
# print(f'R2: {r2_krr:.2f}')

# lasso
# from sklearn.linear_model import Lasso
# lasso = Lasso()
# scores_lasso = cross_val_score(lasso, X, y, cv=10, scoring='r2')
# print(f'lasso r2: {scores_lasso.mean()}')
# lasso.fit(X_train, y_train)
# y_pred_lasso = lasso.predict(X_test)
# mse_lasso = mean_squared_error(y_test, y_pred_lasso)
# rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
# r2_lasso = r2_score(y_test, y_pred_lasso)
# print('lasso:')
# print(f'MSE: {mse_lasso:.2f}')
# print(f'RMSE: {rmse_lasso:.2f}')
# print(f'R2: {r2_lasso:.2f}')

# MLP
# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor()
# scores_mlp = cross_val_score(mlp, X, y, cv=10, scoring='r2')
# print(f'MLP r2: {scores_mlp.mean()}')
# mlp.fit(X_train, y_train)
# y_pred_mlp = mlp.predict(X_test)
# mse_mlp = mean_squared_error(y_test, y_pred_mlp)
# rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)
# r2_mlp = r2_score(y_test, y_pred_mlp)
# print('mlp:')
# print(f'MSE: {mse_mlp:.2f}')
# print(f'RMSE: {rmse_mlp:.2f}')
# print(f'R2: {r2_mlp:.2f}')

# decision tree
# from sklearn.tree import DecisionTreeRegressor
# dtr = DecisionTreeRegressor()
# scores_dt = cross_val_score(dtr, X, y, cv=10, scoring='r2')
# print(f'decision tree r2: {scores_dt.mean()}')
# dtr.fit(X_train, y_train)
# y_pred_dtr = dtr.predict(X_test)
# mse_dtr = mean_squared_error(y_test, y_pred_dtr)
# rmse_dtr = mean_squared_error(y_test, y_pred_dtr, squared=False)
# r2_dtr = r2_score(y_test, y_pred_dtr)
# print('dtr:')
# print(f'MSE: {mse_dtr:.2f}')
# print(f'RMSE: {rmse_dtr:.2f}')
# print(f'R2: {r2_dtr:.2f}')

# extra tree
# from sklearn.tree import ExtraTreeRegressor
# etr = ExtraTreeRegressor()
# scores_etr = cross_val_score(etr, X, y, cv=10, scoring='r2')
# print(f'extra tree r2: {scores_etr.mean()}')
# etr.fit(X_train, y_train)
# y_pred_etr = etr.predict(X_test)
# mse_etr = mean_squared_error(y_test, y_pred_etr)
# rmse_etr = mean_squared_error(y_test, y_pred_etr, squared=False)
# r2_etr = r2_score(y_test, y_pred_etr)
# print('etr:')
# print(f'MSE: {mse_etr:.2f}')
# print(f'RMSE: {rmse_etr:.2f}')
# print(f'R2: {r2_etr:.2f}')

# rf
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# scores_rf = cross_val_score(rf, X, y, cv=10, scoring='r2')
# print(f'ramdom forest r2: {scores_rf.mean()}')
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
# r2_rf = r2_score(y_test, y_pred_rf)
# print('rf:')
# print(f'MSE: {mse_rf:.2f}')
# print(f'RMSE: {rmse_rf:.2f}')
# print(f'R2: {r2_rf:.2f}')

# adaboost
# from sklearn.ensemble import AdaBoostRegressor
# abr = AdaBoostRegressor()
# scores_abr = cross_val_score(abr, X, y, cv=10, scoring='r2')
# print(f'adaboost r2: {scores_abr.mean()}')
# abr.fit(X_train, y_train)
# y_pred_abr = abr.predict(X_test)
# mse_abr = mean_squared_error(y_test, y_pred_abr)
# rmse_abr = mean_squared_error(y_test, y_pred_abr, squared=False)
# r2_abr = r2_score(y_test, y_pred_abr)
# print('ada:')
# print(f'MSE: {mse_abr:.2f}')
# print(f'RMSE: {rmse_abr:.2f}')
# print(f'R2: {r2_abr:.2f}')

# gbr
# from sklearn.ensemble import GradientBoostingRegressor
# gbr = GradientBoostingRegressor()
# scores_gbr = cross_val_score(gbr, X, y, cv=10, scoring='r2')
# print(f'gbr r2: {scores_gbr.mean()}')
# gbr.fit(X_train, y_train)
# y_pred_gbr = gbr.predict(X_test)
# mse_gbr = mean_squared_error(y_test, y_pred_gbr)
# rmse_gbr = mean_squared_error(y_test, y_pred_gbr, squared=False)
# r2_gbr = r2_score(y_test, y_pred_gbr)
# print('gbr:')
# print(f'MSE: {mse_gbr:.2f}')
# print(f'RMSE: {rmse_gbr:.2f}')
# print(f'R2: {r2_gbr:.2f}')

# bagging
# from sklearn.ensemble import BaggingRegressor
# br = BaggingRegressor()
# scores_br = cross_val_score(br, X, y, cv=10, scoring='r2')
# print(f'bagging r2: {scores_br.mean()}')
# br.fit(X_train, y_train)
# y_pred_br = br.predict(X_test)
# mse_br = mean_squared_error(y_test, y_pred_br)
# rmse_br = mean_squared_error(y_test, y_pred_br, squared=False)
# r2_br = r2_score(y_test, y_pred_br)
# print('br:')
# print(f'MSE: {mse_br:.2f}')
# print(f'RMSE: {rmse_br:.2f}')
# print(f'R2: {r2_br:.2f}')

# LR
lr = LinearRegression()
scores_lr = cross_val_score(lr, X, y, cv=10, scoring='r2')
print(f'lr r2: {scores_lr.mean()}')
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
# r2_lr = r2_score(y_test, y_pred_lr)
# print('LR:')
# print(f'MSE: {mse_lr:.2f}')
# print(f'RMSE: {rmse_lr:.2f}')
# print(f'R2: {r2_lr:.2f}')
