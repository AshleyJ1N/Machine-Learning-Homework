import numpy as np
import pandas as pd

# 读取测试数据
X = np.zeros((96, 19900))
x_corr = np.zeros((1, 19900))
j = 0
for i in range(1172, 1363, 2):
    filename = f'sub_{i}.csv'
    xtmp = pd.read_csv(filename, header=None)
    x_tmp = xtmp.values
    x_tmp = np.array(x_tmp)
    # 若数据不是200*740：切下一个被试
    if ((x_tmp.shape[0] != 200) | (x_tmp.shape[1] != 740)):
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
X.to_csv('converted_ROI_FC.csv', header=False, index=False)

'''
# 读取测试数据，少版
X = np.zeros((96, 200))
j = 0
for i in range(1172, 1363, 2):
    filename = f'sub_{i}.csv'
    xtmp = pd.read_csv(filename, header=None)
    x_tmp = xtmp.values
    x_tmp = np.array(x_tmp)
    # 若数据少于：切下一个被试
    if ((x_tmp.shape[0] != 200) | (x_tmp.shape[1] != 740)):
        print("continue")
        continue
    x_tmp = x_tmp[0:200, 0:740]
    x_avg = np.average(x_tmp, axis=1)
    X[j, :] = x_avg
    j = j + 1

X = pd.DataFrame(X)
X.to_csv('converted_ROI_new96.csv', header=False, index=False)
'''