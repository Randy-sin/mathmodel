import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Read data
df = pd.read_csv('data/processed/processed_tourism_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate baseline
baseline_2019 = df[(df['date'] >= '2019-01-01') & (df['date'] < '2019-07-01')]['total arrivals'].mean()

# Get seasonal factors
pre_covid = df[df['date'] < '2019-07-01'].copy()
pre_covid['month'] = pre_covid['date'].dt.month
seasonal_factors = pre_covid.groupby('month')['total arrivals'].mean() / pre_covid['total arrivals'].mean()

# Create future dates
future_dates = pd.date_range(start='2026-01-01', end='2030-12-31', freq='M')
forecast_df = pd.DataFrame({'date': future_dates})
forecast_df['month'] = forecast_df['date'].dt.month
forecast_df['year'] = forecast_df['date'].dt.year

# S-curve forecast function
def recovery_curve(x, max_value, current_value, years):
    return max_value - (max_value - current_value) * np.exp(-0.5 * x / years)

# Generate S-curve forecast
current_level = df[df['date'] >= '2025-01-01']['total arrivals'].mean()
years = forecast_df['year'] - 2025
forecast_df['s_curve_forecast'] = recovery_curve(years, baseline_2019, current_level, 3)
forecast_df['seasonal_factor'] = forecast_df['month'].map(seasonal_factors)
forecast_df['s_curve_final'] = forecast_df['s_curve_forecast'] * forecast_df['seasonal_factor']

# ARIMA model (using pre-covid and recovery data)
normal_periods = df[(df['date'] < '2019-07-01') | (df['date'] >= '2023-01-01')].copy()
normal_periods = normal_periods.set_index('date')['total arrivals']
model = ARIMA(normal_periods, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
arima_forecast = results.forecast(steps=len(future_dates))
forecast_df['arima_forecast'] = arima_forecast.values

# Create visualization
plt.figure(figsize=(15, 10))

# Plot historical data
plt.plot(df['date'], df['total arrivals'], color='#2E86C1', 
         label='Historical Data', linewidth=2, alpha=0.7)

# Plot forecasts
plt.plot(forecast_df['date'], forecast_df['s_curve_final'], 
         color='#E74C3C', label='S-Curve Forecast', 
         linestyle='--', linewidth=2)
plt.plot(forecast_df['date'], forecast_df['arima_forecast'], 
         color='#27AE60', label='ARIMA Forecast', 
         linestyle='--', linewidth=2)

# Add reference line
plt.axhline(y=baseline_2019, color='#7F8C8D', linestyle=':', 
            label='2019 H1 Average Level', alpha=0.5)

# Highlight important periods
plt.axvspan(pd.to_datetime('2019-07-01'), pd.to_datetime('2020-01-01'), 
            color='#FAD7A0', alpha=0.2, label='Social Events')
plt.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2023-01-01'), 
            color='#D5F5E3', alpha=0.2, label='COVID-19 Period')

# Customize the plot
plt.title('Hong Kong Tourism Arrivals Forecast (2026-2030)', 
          fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Monthly Visitors', fontsize=12)

# Format y-axis to show millions
def millions_formatter(x, pos):
    return f'{int(x/1000000)}M'
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

# Enhance grid and legend
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
          fontsize=10, framealpha=0.9)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Add text annotations for key statistics
max_s_curve = forecast_df['s_curve_final'].max()
max_arima = forecast_df['arima_forecast'].max()
plt.annotate(f'S-Curve Peak: {int(max_s_curve/1000000)}M',
             xy=(forecast_df['date'].iloc[-12], max_s_curve),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, color='#E74C3C')
plt.annotate(f'ARIMA Peak: {int(max_arima/1000000)}M',
             xy=(forecast_df['date'].iloc[-12], max_arima),
             xytext=(10, -20), textcoords='offset points',
             fontsize=10, color='#27AE60')

# Adjust layout and save
plt.tight_layout()
plt.savefig('enhanced_forecast.png', dpi=300, bbox_inches='tight')
plt.close()

# Print comparison statistics
print("\n=== Model Comparison (2026-2030) ===")
print("\nS-Curve Model Forecast:")
s_curve_stats = forecast_df.groupby('year')['s_curve_final'].agg(['mean', 'min', 'max'])
arima_stats = forecast_df.groupby('year')['arima_forecast'].agg(['mean', 'min', 'max'])

for year in s_curve_stats.index:
    print(f"\n{year}:")
    print(f"S-Curve Model:")
    print(f"  Average: {int(s_curve_stats.loc[year, 'mean']):,} visitors/month")
    print(f"  Peak: {int(s_curve_stats.loc[year, 'max']):,} visitors")
    print(f"  % of 2019 baseline: {(s_curve_stats.loc[year, 'mean']/baseline_2019*100):.1f}%")
    print(f"\nARIMA Model:")
    print(f"  Average: {int(arima_stats.loc[year, 'mean']):,} visitors/month")
    print(f"  Peak: {int(arima_stats.loc[year, 'max']):,} visitors")
    print(f"  % of 2019 baseline: {(arima_stats.loc[year, 'mean']/baseline_2019*100):.1f}%") 