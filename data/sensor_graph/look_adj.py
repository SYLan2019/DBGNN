# import numpy as np
# import pickle
# f1=open('adj_mx.pkl','rb')
# adj_metr=pickle.load(f1,encoding='bytes')
# f2=open('adj_mx_bay.pkl','rb')
# adj_bay=pickle.load(f2,encoding='bytes')
# print('')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ... 创建一个简单的时间序列
hours = pd.date_range('2023-01-01', periods=24, freq='H')
values = range(24)

# 绘制图表
plt.plot(hours, values)

# ... 设置x轴刻度为时间
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# 显示图表
plt.show()


