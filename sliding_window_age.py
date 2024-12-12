from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

r2_score = []
for slide in tqdm(range(0, 738)):
    # 文件读取
    X_filename = f'slidingwindow{slide+1}to{slide+3}.csv'
    y_filename = f'age.csv'

    Xtmp = pd.read_csv(X_filename, header=None)
    Xtmp = Xtmp.values
    X = np.array(Xtmp)

    ytmp = pd.read_csv(y_filename, header=None)
    ytmp = ytmp.values
    y = np.array(ytmp)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    # 转为正态分布
    qt = QuantileTransformer(output_distribution='normal')
    X = qt.fit_transform(X)

    # logistic regression
    lr = LinearRegression()
    scores = cross_val_score(lr, X, y, cv=10, scoring='r2')

    r2_score.append(scores.mean())

# 折线图
# x_axis = range(0, 738)
# y_axis = r2_score
# plt.plot(x_axis, y_axis)
# plt.savefig('sliding_window_r2_age.png')

r2_score = pd.DataFrame(r2_score)
r2_score.to_csv('sliding_window_r2_age.csv')
