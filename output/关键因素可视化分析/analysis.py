import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from scipy import stats

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.sans-serif'] = ['Arial']

# 读取数据
df = pd.read_csv('output/combined_monthly_data.csv')

# 将日期列转换为datetime类型
df['日期'] = pd.to_datetime(df['日期'])

# 创建时期标记
df['period'] = 'pre_covid'  # 默认为疫情前
df.loc[df['年份'].isin([2020, 2021, 2022]), 'period'] = 'covid'  # 疫情期间
df.loc[df['年份'] >= 2023, 'period'] = 'post_covid'  # 疫情后

# 创建图表：时间序列趋势
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('Tourism Industry Trends in Hong Kong', fontsize=16, y=1.02)

# 定义颜色
pre_covid_color = '#2ecc71'    # 绿色
covid_color = '#e74c3c'        # 红色
post_covid_color = '#3498db'   # 蓝色
alpha_covid = 0.3

# 创建不同时期的掩码
pre_covid_mask = df['period'] == 'pre_covid'
covid_mask = df['period'] == 'covid'
post_covid_mask = df['period'] == 'post_covid'

def plot_smoothed_data(ax, df, x_col, y_col, masks, colors, labels, alpha_values):
    for mask, color, label, alpha in zip(masks, colors, labels, alpha_values):
        # 获取当前时期的数据
        data = df[mask].sort_values(by=x_col)
        x = (data[x_col] - data[x_col].min()).dt.total_seconds()
        y = data[y_col].values
        
        if len(x) > 3:  # 只有当数据点足够时才进行平滑
            # 创建更密集的时间点进行插值
            x_smooth = np.linspace(x.min(), x.max(), 300)
            
            # 使用三次样条插值
            try:
                f = interp1d(x, y, kind='cubic')
                y_smooth = f(x_smooth)
                
                # 将时间转回datetime
                x_dates = pd.to_datetime(data[x_col].min() + pd.Timedelta(seconds=float(x_smooth[0])))
                x_smooth_dates = [x_dates + pd.Timedelta(seconds=float(t)) for t in x_smooth - x_smooth[0]]
                
                # 绘制平滑曲线
                ax.plot(x_smooth_dates, y_smooth, color=color, alpha=alpha, linewidth=2, label=label)
                
                # 绘制原始数据点
                ax.scatter(data[x_col], y, color=color, alpha=alpha, s=30)
            except:
                # 如果插值失败，就直接绘制原始数据
                ax.plot(data[x_col], y, color=color, alpha=alpha, linewidth=2, marker='o', 
                       markersize=4, label=label)
        else:
            # 数据点太少，直接绘制原始数据
            ax.plot(data[x_col], y, color=color, alpha=alpha, linewidth=2, marker='o', 
                   markersize=4, label=label)

# 定义绘图参数
masks = [pre_covid_mask, covid_mask, post_covid_mask]
colors = [pre_covid_color, covid_color, post_covid_color]
labels = ['Pre-COVID Period', 'COVID-19 Period', 'Post-COVID Period']
alphas = [1.0, alpha_covid, 1.0]

# 访客人数趋势
plot_smoothed_data(axes[0, 0], df, '日期', '访客人数', masks, colors, labels, alphas)
axes[0, 0].set_title('Number of Visitors', pad=20)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend()

# 演唱会场次趋势
plot_smoothed_data(axes[0, 1], df, '日期', '演唱会场次', masks, colors, labels, alphas)
axes[0, 1].set_title('Number of Concerts', pad=20)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Count')
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()

# 酒店房价趋势
plot_smoothed_data(axes[1, 0], df, '日期', '酒店房价', masks, colors, labels, alphas)
axes[1, 0].set_title('Hotel Room Price', pad=20)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Price (HKD)')
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()

# 酒店入住率趋势
plot_smoothed_data(axes[1, 1], df, '日期', '酒店入住率', masks, colors, labels, alphas)
axes[1, 1].set_title('Hotel Occupancy Rate', pad=20)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Rate (%)')
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend()

# 旅游业收入趋势
plot_smoothed_data(axes[2, 0], df, '日期', '旅游业收入', masks, colors, labels, alphas)
axes[2, 0].set_title('Tourism Revenue', pad=20)
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Revenue (Billion HKD)')
axes[2, 0].grid(True, linestyle='--', alpha=0.7)
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].legend()

# 航空旅客趋势
plot_smoothed_data(axes[2, 1], df, '日期', '航空旅客', masks, colors, labels, alphas)
axes[2, 1].set_title('Air Passengers', pad=20)
axes[2, 1].set_xlabel('Date')
axes[2, 1].set_ylabel('Count')
axes[2, 1].grid(True, linestyle='--', alpha=0.7)
axes[2, 1].tick_params(axis='x', rotation=45)
axes[2, 1].legend()

plt.tight_layout()
plt.savefig('trends.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算相关系数矩阵（使用非疫情期间的数据）
correlation_columns = ['访客人数', '演唱会场次', '酒店房价', '烟花表演', '酒店入住率', '旅游业收入']
non_covid_mask = df['period'].isin(['pre_covid', 'post_covid'])
correlation_matrix = df[non_covid_mask][correlation_columns].corr()

# 重命名列以用于相关性矩阵
column_mapping = {
    '访客人数': 'Visitors',
    '演唱会场次': 'Concerts',
    '酒店房价': 'Hotel Price',
    '烟花表演': 'Fireworks',
    '酒店入住率': 'Occupancy Rate',
    '旅游业收入': 'Tourism Revenue'
}

correlation_matrix.index = [column_mapping[col] for col in correlation_matrix.index]
correlation_matrix.columns = [column_mapping[col] for col in correlation_matrix.columns]

# 绘制相关系数热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdYlBu_r',
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.3f',
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Tourism Indicators (Excluding COVID-19 Period)', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印相关系数矩阵
print("\nCorrelation Matrix (Excluding COVID-19 Period):")
print(correlation_matrix)

# 计算每个变量的基本统计信息
df_stats = df[non_covid_mask][correlation_columns].describe()
df_stats.columns = [column_mapping[col] for col in df_stats.columns]
print("\nDescriptive Statistics (Excluding COVID-19 Period):")
print(df_stats)

# 在相关性分析之前添加烟花表演的专门分析
print("\n烟花表演对旅游指标的影响分析（非疫情期间）：")
non_covid_data = df[non_covid_mask]

metrics = ['访客人数', '酒店入住率', '旅游业收入', '酒店房价']
for metric in metrics:
    # 计算有无烟花表演时的平均值
    with_fireworks = non_covid_data[non_covid_data['烟花表演'] == 1][metric].mean()
    without_fireworks = non_covid_data[non_covid_data['烟花表演'] == 0][metric].mean()
    
    # 进行t检验
    t_stat, p_value = stats.ttest_ind(
        non_covid_data[non_covid_data['烟花表演'] == 1][metric],
        non_covid_data[non_covid_data['烟花表演'] == 0][metric]
    )
    
    print(f"\n{metric}:")
    print(f"有烟花表演时的平均值: {with_fireworks:.2f}")
    print(f"无烟花表演时的平均值: {without_fireworks:.2f}")
    print(f"差异百分比: {((with_fireworks - without_fireworks) / without_fireworks * 100):.2f}%")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")

# 创建箱型图来比较有无烟花表演时的各项指标
plt.figure(figsize=(15, 10))
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Impact of Fireworks Shows on Tourism Metrics (Excluding COVID-19 Period)', fontsize=16, y=1.02)

for idx, (metric, title) in enumerate(zip(metrics, 
    ['Number of Visitors', 'Hotel Occupancy Rate', 'Tourism Revenue', 'Hotel Room Price'])):
    i, j = idx // 2, idx % 2
    sns.boxplot(data=non_covid_data, x='烟花表演', y=metric, ax=axes[i, j])
    axes[i, j].set_title(title)
    axes[i, j].set_xlabel('Fireworks Show (0: No, 1: Yes)')

plt.tight_layout()
plt.savefig('fireworks_impact.png', dpi=300, bbox_inches='tight')
plt.close() 