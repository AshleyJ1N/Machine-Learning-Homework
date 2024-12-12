
文件夹内代码介绍：

【1】reduce_dimension_FC.py文件用于做数据预处理，将1171个被试的原始数据使用功能连接（Funtional Connectivity）的方式做降维。

【2】classify.py文件用9个不同的模型对原始数据做十折交叉验证，评估不同模型的性能。

【3】regression.py文件用9个不同的模型对原始数据做十折交叉验证，评估不同模型的性能。

【4】gridsearch.py文件用于做网格搜索，寻找不同模型的最佳参数。

【5】train.py文件用于对数据进行预处理（缺失值填补、归一化和标准化）并用LogisticRegression进行训练，保存模型参数。

【6】test.py文件读取保存好的模型，并输入测试集，预测性别和年龄。

（以下为plus作业的代码）
【7】sliding_window_prepocessing.py文件用于对被试的原始数据预处理，将其转换成738个时间窗下的1140个被试的ROI模态。

【8】sliding_window_age.py和sliding_window_gender.py文件用于对预处理后的每一个时间窗下的模态进行学习和预测，获得十折交叉验证后的指标。

【9】draw_pics.py用于将上一个文件跑出来的结果可视化，将分类和预测的结果做min-max归一化，并画一条折线图。


以下为其他文件的介绍：
【1】imp.pkl、pca.pkl、qt.pkl、sc.pkl分别用来保存回归和分类任务预处理过程中的参数。

【2】classify_model和regression_model分别用来保存回归和分类任务中使用的模型参数。