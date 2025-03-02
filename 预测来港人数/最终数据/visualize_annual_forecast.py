import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 16
})

# Read data
historical_data = pd.read_csv('data/processed/processed_tourism_data_with_dummies.csv')
forecast_data = pd.read_csv('底层SARIMAX模型/预测数据/predict/tourism_forecast_results.csv', index_col=0)
forecast_data.index = pd.to_datetime(forecast_data.index)

# Process historical data (exclude 2025)
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data['year'] = historical_data['date'].dt.year
historical_data = historical_data[historical_data['year'] < 2025]  # Exclude 2025 data
annual_historical = historical_data.groupby('year')['total arrivals'].sum().reset_index()

# Calculate pre-pandemic average (2013-2019)
pre_pandemic_data = historical_data[
    (historical_data['date'].dt.year >= 2013) & 
    (historical_data['date'].dt.year <= 2019)
].groupby('year')['total arrivals'].sum()
average_value = pre_pandemic_data.mean()
print(f"\nPre-pandemic Average (2013-2019): {average_value:,.0f} ({average_value/1e6:.1f}M)")

# Calculate annual sums for forecast data (2025-2030)
annual_sums = []
annual_ci = []

for year in range(2025, 2031):
    year_data = forecast_data[forecast_data.index.year == year].copy()
    if not year_data.empty:
        annual_sum = year_data['predicted'].astype(float).sum()
        annual_lower = year_data['lower_ci'].astype(float).sum()
        annual_upper = year_data['upper_ci'].astype(float).sum()
        
        annual_sums.append({'year': year, 'predicted': annual_sum})
        annual_ci.append({'year': year, 'lower_ci': annual_lower, 'upper_ci': annual_upper})

annual_forecast = pd.DataFrame(annual_sums)
annual_forecast_ci = pd.DataFrame(annual_ci)

# Create figure with specific dimensions
fig, ax = plt.subplots(figsize=(15, 8))

# Color scheme
colors = {
    'historical': '#2E86AB',  # Blue
    'forecast': '#D95032',    # Red
    'confidence': '#FFE5D9',  # Light Orange
    'reference': '#2C3E50',   # Dark Blue-Gray
    'transition': '#A8A8A8'   # Gray for transition line
}

# Calculate y-axis range
y_min = 0
y_max = 80  # Set fixed maximum to 80 million

# Plot historical data
ax.plot(annual_historical['year'], annual_historical['total arrivals'] / 1e6,
        color=colors['historical'], linewidth=2.5, marker='o', markersize=6,
        label='Historical Data (2013-2023)')

# Create smooth transition between 2023 and 2025
last_historical_year = annual_historical['year'].max()
last_historical_value = annual_historical[annual_historical['year'] == last_historical_year]['total arrivals'].values[0] / 1e6
first_forecast_value = annual_forecast.iloc[0]['predicted'] / 1e6

# Create transition points and plot
transition_years = [last_historical_year, 2024, 2025]
transition_values = [last_historical_value, 
                    (last_historical_value * 0.4 + first_forecast_value * 0.6),  # 2024 transition value
                    first_forecast_value]

ax.plot(transition_years, transition_values,
        color=colors['transition'], linewidth=2, linestyle='--',
        alpha=0.7)

# Plot forecast data
ax.plot(annual_forecast['year'], annual_forecast['predicted'] / 1e6,
        color=colors['forecast'], linewidth=2.5, marker='s', markersize=6,
        label='Forecast (2025-2030)')

# Plot confidence interval
ax.fill_between(annual_forecast_ci['year'],
                annual_forecast_ci['lower_ci'] / 1e6,
                annual_forecast_ci['upper_ci'] / 1e6,
                color=colors['confidence'], alpha=0.5,
                label='95% Confidence Interval')

# Add reference line for pre-pandemic average
reference_value = average_value / 1e6
ax.axhline(y=reference_value, color=colors['reference'], linestyle='--',
           label=f'Pre-pandemic Average ({reference_value:.1f}M)', alpha=0.7)

# Add text annotation for reference line in the middle of the plot
mid_year = (2013 + 2030) / 2  # Calculate middle year
ax.text(mid_year, reference_value - 2, f'Pre-pandemic Average: {reference_value:.1f}M',
        va='top', ha='center', color=colors['reference'],
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

# Customize grid
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Set axis ranges
ax.set_xlim(2012.5, 2030.5)  # Add padding to x-axis
ax.set_ylim(0, y_max)

# Set labels and title
ax.set_title('Annual Visitor Arrivals Forecast for Hong Kong (2013-2030)',
             pad=20, fontweight='bold')
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Number of Visitors (Millions)', fontweight='bold')

# Customize x-axis
ax.set_xticks(range(2013, 2031))
ax.set_xticklabels(range(2013, 2031), rotation=45)

# Set y-axis ticks every 10 million
y_ticks = np.arange(0, y_max + 10, 10)
ax.set_yticks(y_ticks)
ax.set_ylim(0, y_max)

# Format y-axis labels
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))

# Add major gridlines
ax.grid(True, which='major', linestyle='--', alpha=0.3)

# Enhance legend
ax.legend(loc='upper left', frameon=True, framealpha=0.95,
          edgecolor='none', fontsize=10)

# Add annotations for forecast years (2025-2030)
for _, row in annual_forecast.iterrows():
    value = row['predicted'] / 1e6
    # 为2025年的数据点特别处理
    if row['year'] == 2025:
        xytext_offset = (0, -15)  # 向下偏移15个单位
        va_position = 'top'  # 改为顶部对齐
    else:
        xytext_offset = (0, 10)  # 其他年份保持不变
        va_position = 'bottom'
    
    ax.annotate(f'{value:.1f}M', 
                xy=(row['year'], value),
                xytext=xytext_offset,
                textcoords='offset points',
                ha='center',
                va=va_position,
                fontsize=9)

# Add key statistics
max_historical = annual_historical['total arrivals'].max() / 1e6
max_forecast = annual_forecast['predicted'].max() / 1e6
stats_text = (
    f'Historical Peak: {max_historical:.1f}M\n'
    f'Forecast Peak (2030): {max_forecast:.1f}M'
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
        va='top', fontsize=10)

# Add watermark
plt.figtext(0.99, 0.01, 'Hong Kong Tourism Forecast Model v1.0',
            ha='right', va='bottom', alpha=0.5, fontsize=8)

# Adjust layout
plt.tight_layout()

# Save high-resolution figure
plt.savefig('annual_forecast_visualization.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("\nVisualization has been generated: annual_forecast_visualization.png") 