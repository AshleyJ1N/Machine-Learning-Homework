import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")

# 读取测试文件
X_test_filename = 'converted_fc.csv'

Xtmp = pd.read_csv(X_test_filename, header=None)
Xtmp = Xtmp.values
X_test = np.array(Xtmp)

# 数据预处理
# 分类任务预处理
# 插补缺失值
imp = pickle.load(open('imp.pkl', 'rb'))  # 读取
X_test = imp.transform(X_test)
# 标准化
sc = pickle.load(open('sc.pkl', 'rb'))  # 读取
X_test_classify = sc.transform(X_test)

# 回归任务预处理
# 标准化，并转换为正态分布
qt = pickle.load(open('qt.pkl', 'rb'))  # 读取
X_test_regression = qt.transform(X_test)
# 使用PCA降低数据维度
pca = pickle.load(open('pca.pkl', 'rb'))  # 读取
X_test_regression = pca.transform(X_test_regression)

# 性别预测
lr_c = joblib.load('classify_model')
y_gender = lr_c.predict(X_test_classify)

# 年龄预测
lr_r = joblib.load('regression_model')
y_age = lr_r.predict(X_test_regression)

# 保存预测结果
y_age = pd.DataFrame(y_age)
y_age.to_csv('pred_age.csv', header=None, index=False)
y_gender = pd.DataFrame(y_gender)
y_gender.to_csv('pred_gender.csv', header=None, index=False)
