import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read data
df = pd.read_csv('data/processed/processed_tourism_data_with_dummies.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Log transformation
df['log_arrivals'] = np.log1p(df['total arrivals'])

# Split data
pre_covid = df[:'2019-05']
recovery = df['2023-03':]

print("=== Data Period Statistics ===")
print("\nPre-COVID Period:", pre_covid.index[0].strftime('%Y-%m'), "to", pre_covid.index[-1].strftime('%Y-%m'))
print("Recovery Period:", recovery.index[0].strftime('%Y-%m'), "to", recovery.index[-1].strftime('%Y-%m'))

# STL decomposition for pre-COVID data
print("\n=== Pre-COVID STL Decomposition ===")
stl_pre = STL(pre_covid['log_arrivals'], period=12)
result_pre = stl_pre.fit()

# Calculate strength of seasonality and trend
total_var_pre = np.var(result_pre.resid)
seasonal_strength_pre = max(0, 1 - total_var_pre / np.var(result_pre.seasonal + result_pre.resid))
trend_strength_pre = max(0, 1 - total_var_pre / np.var(result_pre.trend + result_pre.resid))

print("\nPre-COVID Data Characteristics:")
print(f"Seasonal Strength: {seasonal_strength_pre:.3f}")
print(f"Trend Strength: {trend_strength_pre:.3f}")

# Plot STL decomposition for pre-COVID data
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle('STL Decomposition of Hong Kong Tourism Data (Pre-COVID)', fontsize=16, y=1.02)

# Original data
ax1.plot(pre_covid.index, pre_covid['log_arrivals'], color='#2E86AB')
ax1.set_title('Original Data (Log-transformed)', pad=10)
ax1.set_xlabel('')

# Trend
ax2.plot(pre_covid.index, result_pre.trend, color='#A23B72')
ax2.set_title('Trend Component', pad=10)
ax2.set_xlabel('')

# Seasonal
ax3.plot(pre_covid.index, result_pre.seasonal, color='#F18F01')
ax3.set_title('Seasonal Component', pad=10)
ax3.set_xlabel('')

# Residual
ax4.plot(pre_covid.index, result_pre.resid, color='#C73E1D')
ax4.set_title('Residual Component', pad=10)

plt.tight_layout()
plt.savefig('pre_covid_stl.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze monthly seasonal pattern
seasonal_df = pd.DataFrame({
    'month': pre_covid.index.month,
    'seasonal': result_pre.seasonal
})
monthly_pattern = seasonal_df.groupby('month')['seasonal'].mean()

# Plot monthly seasonal pattern
plt.figure(figsize=(12, 6))
ax = monthly_pattern.plot(kind='bar', color='#2E86AB', width=0.8)
plt.title('Monthly Seasonal Pattern in Hong Kong Tourism (Pre-COVID)', pad=20)
plt.xlabel('Month')
plt.ylabel('Seasonal Effect (Log Scale)')

# Rotate x-axis labels and adjust their position
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
           rotation=45, ha='right')

# Add value labels on bars with adjusted position
for i, v in enumerate(monthly_pattern):
    # 调整标签位置：正值在上方，负值在下方，距离根据数值大小调整
    offset = 0.02 if v >= 0 else -0.02
    y_pos = v + offset
    
    # 为较大的负值特别调整位置
    if v < -0.1:
        y_pos = v - 0.03
    
    ax.text(i, y_pos, f'{v:+.3f}', 
            ha='center', 
            va='bottom' if v >= 0 else 'top',
            fontsize=10)

# Adjust y-axis limits to make room for labels
y_min, y_max = plt.ylim()
plt.ylim(y_min - 0.02, y_max + 0.02)

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('pre_covid_seasonal_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPre-COVID Monthly Seasonal Effects (Sorted):")
sorted_effects = monthly_pattern.sort_values(ascending=False)
for month, effect in sorted_effects.items():
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    print(f"{month_name}: {effect:+.4f}")

# Analyze recovery period
print("\n=== Recovery Period Analysis ===")
stl_recovery = STL(recovery['log_arrivals'], period=12)
result_recovery = stl_recovery.fit()

# Plot recovery trend
plt.figure(figsize=(15, 8))
plt.plot(recovery.index, np.expm1(result_recovery.trend), 
         color='#2E86AB', linewidth=2.5, label='Recovery Trend')
plt.axhline(y=pre_covid['total arrivals'].mean(), color='#A23B72', 
            linestyle='--', linewidth=2, label='Pre-COVID Average')

plt.title('Hong Kong Tourism Recovery Trend Analysis', pad=20)
plt.xlabel('Date')
plt.ylabel('Number of Visitors')
plt.legend(frameon=True, facecolor='white', framealpha=1)
plt.grid(True, alpha=0.3)

# Format y-axis to millions
def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M'
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

plt.tight_layout()
plt.savefig('recovery_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate recovery period metrics
recovery_latest = np.expm1(result_recovery.trend[-1])
pre_covid_avg = pre_covid['total arrivals'].mean()
recovery_rate = recovery_latest / pre_covid_avg * 100

print("\nRecovery Analysis Results:")
print(f"Latest Trend Value: {recovery_latest:,.0f}")
print(f"Pre-COVID Average: {pre_covid_avg:,.0f}")
print(f"Recovery Rate: {recovery_rate:.1f}%")

# Calculate monthly growth rates
recovery_monthly_growth = recovery['total arrivals'].pct_change() * 100
avg_monthly_growth = recovery_monthly_growth.mean()

print(f"\nAverage Monthly Growth Rate during Recovery: {avg_monthly_growth:.1f}%")
print("\nLast 6 Months Growth Rates:")
for date, growth in recovery_monthly_growth[-6:].items():
    print(f"{date.strftime('%Y-%m')}: {growth:+.1f}%") 