import pandas as pd
import numpy as np
from tqdm import tqdm

# 加载数据集
Xtrain_filepath = 'D:\\wyj_pyproject\\作业一-fMRI分类\\data\\train\\'

# 每三个时间点为一个滑动窗口，计算三个时间点下200个ROI的相关
for slide in tqdm(range(11, 20)):
    X = np.zeros((1140, 19900))
    j = 0
    for i in tqdm(range(0, 1171)):
        filename = f'sub_{i + 1}.csv'
        xtmp = pd.read_csv(Xtrain_filepath + filename, header=None)
        x_tmp = xtmp.values
        # x_tmp = x_tmp[0:200, 0:740]
        x_tmp = np.array(x_tmp)
        # 若数据不是200*740：切下一个被试
        if ((x_tmp.shape[0] != 200) | (x_tmp.shape[1] != 740)):
            continue
        k = 0
        x_corr = np.zeros((1, 19900))
        for iter1 in range(0, 200):
            for iter2 in range(iter1 + 1, 200):
                A = pd.Series(np.array(x_tmp[iter1, slide:slide+3]))
                B = pd.Series(np.array(x_tmp[iter2, slide:slide+3]))
                x_corr[0, k] = round(A.corr(B), 4)
                k = k + 1
        X[j, :] = x_corr
        j = j + 1
    X = pd.DataFrame(X)
    X.to_csv(f'slidingwindow{slide+1}to{slide+3}.csv', header=False, index=False)




'''
# 平均数
j = 0
for i in range(0, 1171):
    filename = f'sub_{i+1}.csv'
    xtmp = pd.read_csv(Xtrain_filepath + filename, header=None)
    x_tmp = xtmp.values
    x_tmp = x_tmp[0:200, 0:740]
    x_tmp = np.array(x_tmp)
    # 若数据不是200*740：切下一个被试
    if((x_tmp.shape[0] != 200) | (x_tmp.shape[1] != 740)):
        print("continue")
        continue
    x_avg = np.average(x_tmp, axis=0)
    X[j, :] = x_avg
    j = j + 1

X = pd.DataFrame(X)
X.to_csv('converted_avg_according_to_col_1159.csv', header=False, index=False)
'''