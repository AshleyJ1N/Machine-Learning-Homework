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

# 读取训练文件
X_filename = 'converted_fc.csv'
gender_filename = 'gender.csv'
age_filename = 'age.csv'

Xtmp = pd.read_csv(X_filename, header=None)
Xtmp = Xtmp.values
X = np.array(Xtmp)

ytmp = pd.read_csv(gender_filename, header=None)
ytmp = ytmp.values
gender = np.array(ytmp)

ytmp = pd.read_csv(age_filename, header=None)
ytmp = ytmp.values
age = np.array(ytmp)

# 数据预处理
# 分类任务预处理
# 插补缺失值
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)
pickle.dump(imp, open('imp.pkl', 'wb'))  # 保存
# 标准化
sc = StandardScaler()
X_classify = sc.fit_transform(X)
pickle.dump(sc, open('sc.pkl', 'wb'))  # 保存

# 回归任务预处理
# 标准化，并转换为正态分布
qt = QuantileTransformer(output_distribution='normal')
X_regression = qt.fit_transform(X)
pickle.dump(qt, open('qt.pkl', 'wb'))  # 保存
# 使用PCA降低数据维度
pca = PCA(n_components=100)
X_regression = pca.fit_transform(X_regression)
pickle.dump(pca, open('pca.pkl', 'wb'))  # 保存

# 性别预测
lr_c = LogisticRegression(C=0.0009)
lr_c.fit(X_classify, gender)

# 年龄预测
lr_r = LinearRegression()
lr_r.fit(X_regression, age)

# 模型保存
joblib.dump(lr_c, 'classify_model')
joblib.dump(lr_r, 'regression_model')
