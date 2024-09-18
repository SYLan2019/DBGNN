import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 创建地图对象
map = Basemap(projection='robin', lon_0=0, resolution='c')

# 加载地图数据
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgreen', lake_color='aqua')
map.drawcoastlines()

# 示例数据
longitudes = [1.5, 1.7, 1.9, 2.1, 2.3, 2.5]  # 经度列表
latitudes = [45.5, 45.3, 45.2, 45.5, 45.2, 45.3]  # 纬度列表
values = [10, 20, 30, 50, 80, 100]  # 对应每个点的值

# 将列表转换为 NumPy 数组
values = np.array(values)

# 根据值将颜色范围分为三部分
color_range = np.where(values <= 50, 0, np.where(values <= 80, 1, 2))
colors = ['blue', 'green', 'red']

# 遍历每个点并绘制
for lon, lat, val, col in zip(longitudes, latitudes, values, color_range):
    x, y = map(lon, lat)  # 转换经纬度到地图坐标系
    map.plot(x, y, marker='o', color=colors[col])  # 绘制点

plt.title('Scatterplot with Color Dependent on Value')
plt.show()