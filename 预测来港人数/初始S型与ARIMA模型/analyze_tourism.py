import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data/processed/processed_tourism_data.csv')

# 将date列转换为datetime类型
df['date'] = pd.to_datetime(df['date'])

# 创建图形
plt.figure(figsize=(15, 8))
plt.plot(df['date'], df['total arrivals'], 'b-', label='实际数据')

# 添加垂直线标注重要时间点
plt.axvline(x=pd.to_datetime('2019-07-01'), color='r', linestyle='--', label='社会运动开始')
plt.axvline(x=pd.to_datetime('2020-01-01'), color='g', linestyle='--', label='疫情开始')
plt.axvline(x=pd.to_datetime('2023-01-01'), color='y', linestyle='--', label='恢复期开始')

# 设置图表标题和标签
plt.title('香港旅客人数时间序列（2013-2025）', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('旅客人数', fontsize=12)
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('tourism_analysis.png')
plt.close()

# 计算基本统计信息
# 分析疫情前的数据（2013-2019.06）
pre_covid = df[df['date'] < '2019-07-01']
print("\n=== 疫情前数据统计（2013-2019.06）===")
print(f"平均月访客量: {pre_covid['total arrivals'].mean():,.0f}")
print(f"最大月访客量: {pre_covid['total arrivals'].max():,.0f}")
print(f"最小月访客量: {pre_covid['total arrivals'].min():,.0f}")

# 分析恢复期数据（2023-2025）
recovery = df[df['date'] >= '2023-01-01']
print("\n=== 恢复期数据统计（2023-2025）===")
print(f"平均月访客量: {recovery['total arrivals'].mean():,.0f}")
print(f"最大月访客量: {recovery['total arrivals'].max():,.0f}")
print(f"最小月访客量: {recovery['total arrivals'].min():,.0f}")

# 计算季节性指数
pre_covid['month'] = pre_covid['date'].dt.month
seasonal_index = pre_covid.groupby('month')['total arrivals'].mean() / pre_covid['total arrivals'].mean()
print("\n=== 季节性指数（基于疫情前数据）===")
print(seasonal_index)

# 分析恢复趋势
recovery['months_since_2023'] = (recovery['date'] - pd.to_datetime('2023-01-01')).dt.total_seconds() / (30*24*60*60)
slope, intercept, r_value, p_value, std_err = stats.linregress(recovery['months_since_2023'], recovery['total arrivals'])
print("\n=== 恢复期趋势分析 ===")
print(f"月均增长率: {slope:,.0f}")
print(f"R平方值: {r_value**2:.3f}")

# 计算2019年上半年的平均水平
baseline_2019 = df[(df['date'] >= '2019-01-01') & (df['date'] < '2019-07-01')]['total arrivals'].mean()
print(f"\n2019年上半年平均月访客量: {baseline_2019:,.0f}")
print(f"当前恢复水平: {(recovery['total arrivals'].mean() / baseline_2019 * 100):.1f}%") 