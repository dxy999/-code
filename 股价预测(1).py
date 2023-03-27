#绘制2016年汇率数据的时序图
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'D:\data\rates.csv', index_col = 'date',encoding='ANSI')
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
data.plot()
plt.show()

#纯随机性和平稳性检验
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data) #自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data['rates'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data['rates']))

#差分转换
D_data = data.diff().dropna() #对原数据进行1阶差分，删除非法值
D_data.columns = ['汇率差分']
D_data.plot() #时序图
plot_acf(D_data) #自相关图
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data) #偏自相关图
print('差分序列－ADF－检验结果为：', ADF(D_data[u'汇率差分'])) #平稳性检测

# ARIMA模型
from statsmodels.tsa.arima_model import ARIMA
data['rates'] = data['rates'].astype(float)
pmax = int(len(D_data)/10) #一般阶数不超过length/10
qmax = int(len(D_data)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try: #存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data, (p,1,q)).fit().aic)
        except:
            tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))

#使用ARIMA模型预测下2017年第一个交易日人民币是否升值
model = ARIMA(data, (p,1,q)).fit() #建立ARIMA(1,4,1)模型
model.summary2() #给出模型报告
model.forecast(1)#由此可得，使用ARIMA模型预测2017的第一个交易日人民币贬值

#使用循环神经网络预测下2017年第一个交易日人民币是否升值
#从文件中读出原始序列数据
import pandas as pd
filename=r'D:\研一\数据科学\data\rates.csv'
data=pd.read_csv(filename)
data.head()

from matplotlib import pyplot as plt
temp=data["rates"]
temp_10days=temp[:2000]#使用全部数据节点
temp_10days.plot()
plt.show()

#数据预处理
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
data_process=data.drop('date',axis = 1)#删除时间列
data_process = ss.fit_transform(data_process) #数据标准化

#构造样本数据集
import numpy as np
sample=10
lookback=1*2*6 
delay=2*6 
X=np.zeros((sample,lookback,data_process.shape[-1])) #(5000,720,14)
y=np.zeros((sample,))
#随机生成时刻 
min_index=lookback 
max_index=len(data_process)-delay-1 
rows=np.random.randint(min_index,max_index,size=sample)
#生成X和y数据
for j,row in enumerate(rows):
 #print(j,row)
 indices=np.arange(row-lookback,row)
 X[j]=data_process[indices,:]
 y[j]=data_process[row+delay,:][0]
 
 from keras.models import Sequential
from keras.layers import Dense,LSTM
model=Sequential()
#LSTM输出维度为32，也就是将输入14维的特征转换为32维的特征。
#模型只使用一层LSTM，只需要返回最后结点的输出
# X.shape[-1]是最后轴的维度大小14
model.add(LSTM(32,input_shape=(None,X.shape[-1])))
model.add(Dense(1))
#模型只预测1个值，全连接层输出结点数为1; 回归问题不使用激活函数
#神经网络编译
from keras.optimizers import RMSprop
#损失函数为平均绝对误差（MAE） 
model.compile(optimizer=RMSprop(),loss='mae')

y_predict=model.predict(X)
plt.plot(y[0:100], color='red', label='Real')
plt.plot(y_predict[0:100], color='blue', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='rate')
plt.legend()
plt.show()

#查看预测值
tt=np.zeros((y_predict.shape[0],14))
#tt[:,0]=tt[:,0]+y_predict 
for i in range(y_predict.shape[0]):
 tt[i,0]=y_predict[i,0]
ss.inverse_transform(tt)[:,0][0:1]#在循环神经网络预测下,2017年第一个交易日人民币与实际值比较结果为贬值