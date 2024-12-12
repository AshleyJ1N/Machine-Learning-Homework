import pandas as pd
import numpy as np

# 加载数据集
Xtrain_filepath = 'F:\\硕士作业存档\\研一下\\数据分析与机器学习\\作业一-fMRI分类\\train\\'


X = np.zeros((1140, 19900))
x_corr = np.zeros((1, 19900))
# r
j = 0
for i in range(0, 1171):
    filename = f'sub_{i+1}.csv'
    xtmp = pd.read_csv(Xtrain_filepath + filename, header=None)
    x_tmp = xtmp.values
    x_tmp = np.array(x_tmp)
    # 若数据不是200*740：切下一个被试
    if((x_tmp.shape[0] != 200) | (x_tmp.shape[1] != 740)):
        print("continue")
        continue
    k = 0
    for iter1 in range(0, 200):
        for iter2 in range(iter1 + 1, 200):
            A = pd.Series(np.array(x_tmp[iter1, :]))
            B = pd.Series(np.array(x_tmp[iter2, :]))
            x_corr[0, k] = round(A.corr(B), 4)
            k = k + 1
    X[j, :] = x_corr[0, :]
    j = j + 1

X = pd.DataFrame(X)
X.to_csv('converted_fc.csv', header=False, index=False)
