import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('data/processed/processed_tourism_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 计算2019上半年基准水平
baseline_2019 = df[(df['date'] >= '2019-01-01') & (df['date'] < '2019-07-01')]['total arrivals'].mean()

# 获取季节性因子
pre_covid = df[df['date'] < '2019-07-01'].copy()
pre_covid['month'] = pre_covid['date'].dt.month
seasonal_factors = pre_covid.groupby('month')['total arrivals'].mean() / pre_covid['total arrivals'].mean()

# 创建预测日期范围（2026-2030）
future_dates = pd.date_range(start='2026-01-01', end='2030-12-31', freq='M')
forecast_df = pd.DataFrame({'date': future_dates})
forecast_df['month'] = forecast_df['date'].dt.month
forecast_df['year'] = forecast_df['date'].dt.year

# 定义恢复曲线函数（使用S形曲线）
def recovery_curve(x, max_value, current_value, years):
    return max_value - (max_value - current_value) * np.exp(-0.5 * x / years)

# 生成预测
current_level = df[df['date'] >= '2025-01-01']['total arrivals'].mean()
years = forecast_df['year'] - 2025
forecast_df['baseline_forecast'] = recovery_curve(years, baseline_2019, current_level, 3)

# 应用季节性因子
forecast_df['seasonal_factor'] = forecast_df['month'].map(seasonal_factors)
forecast_df['final_forecast'] = forecast_df['baseline_forecast'] * forecast_df['seasonal_factor']

# 创建可视化
plt.figure(figsize=(15, 8))

# 绘制历史数据
plt.plot(df['date'], df['total arrivals'], 'b-', label='历史数据', alpha=0.6)

# 绘制预测数据
plt.plot(forecast_df['date'], forecast_df['final_forecast'], 'r--', label='预测数据')

# 添加参考线
plt.axhline(y=baseline_2019, color='g', linestyle=':', label='2019上半年平均水平')

# 设置图表
plt.title('香港旅客人数预测（2026-2030）', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('旅客人数', fontsize=12)
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('tourism_forecast.png')
plt.close()

# 输出预测结果统计
print("\n=== 预测结果统计（2026-2030）===")
yearly_forecast = forecast_df.groupby('year')['final_forecast'].agg(['mean', 'min', 'max'])
for year in yearly_forecast.index:
    print(f"\n{year}年预测：")
    print(f"年均访客量: {yearly_forecast.loc[year, 'mean']:,.0f}")
    print(f"最高月访客量: {yearly_forecast.loc[year, 'max']:,.0f}")
    print(f"最低月访客量: {yearly_forecast.loc[year, 'min']:,.0f}")
    print(f"占2019基准水平: {(yearly_forecast.loc[year, 'mean']/baseline_2019*100):.1f}%") 