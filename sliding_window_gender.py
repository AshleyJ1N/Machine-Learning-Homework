from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

f1_score = []
for slide in range(0, 738):
    # 文件读取
    X_filename = f'slidingwindow{slide+1}to{slide+3}.csv'
    y_filename = 'gender.csv'

    Xtmp = pd.read_csv(X_filename, header=None)
    Xtmp = Xtmp.values
    X = np.array(Xtmp)
    ytmp = pd.read_csv(y_filename, header=None)
    ytmp = ytmp.values
    y = np.array(ytmp)

    # 热力图
    # fig = sns.heatmap(Xtmp, square=True)
    # fig.get_figure().savefig(f'sliding{slide+1}to{slide+3}_heatmap.png')

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    # 标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X)


    # logistic regression
    lr = LogisticRegression(C=0.0009)
    scores = cross_val_score(lr, X, y, cv=10, scoring='f1_macro')

    f1_score.append(scores.mean())

# 折线图
x_axis = range(0, 738)
y_axis = f1_score
plt.plot(x_axis, y_axis)
plt.savefig('sliding_window_f1_gender.png')

f1_score = pd.DataFrame(f1_score)
f1_score.to_csv('sliding_window_f1_gender.csv')

