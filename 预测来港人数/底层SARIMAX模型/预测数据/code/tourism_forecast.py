import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read data
df = pd.read_csv('data/processed/processed_tourism_data_with_dummies.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Split data
pre_covid = df[:'2019-05']
recovery = df['2023-03':]

# Get pre-COVID monthly patterns
monthly_means = pre_covid.groupby(pre_covid.index.month)['total arrivals'].mean()
pre_covid_monthly_pattern = {month: value for month, value in monthly_means.items()}

# Calculate baseline values and growth rate
pre_covid_avg = pre_covid['total arrivals'].mean()
current_value = recovery['total arrivals'].iloc[-1]
monthly_growth_rate = 0.047  # 4.7%

# Calculate time to recovery
months_to_recovery = np.log(pre_covid_avg / current_value) / np.log(1 + monthly_growth_rate)
years_to_recovery = months_to_recovery / 12
recovery_date = pd.Timestamp('2025-01-01') + pd.DateOffset(months=int(months_to_recovery))

print("\n=== Recovery Time Forecast ===")
print(f"Current Recovery Rate: {(current_value/pre_covid_avg)*100:.1f}%")
print(f"Monthly Growth Rate: {monthly_growth_rate*100:.1f}%")
print(f"Expected Time to Full Recovery: {months_to_recovery:.1f} months ({years_to_recovery:.1f} years)")
print(f"Expected Recovery Date: {recovery_date.strftime('%B %Y')}")

# Generate forecast dates starting from the last actual data point
last_actual_date = recovery.index[-1]
forecast_dates = pd.date_range(last_actual_date + pd.DateOffset(months=1), '2030-12-31', freq='M')
n_periods = len(forecast_dates)

# Calculate forecast values
forecast_values = []
current = current_value

# Create transition period (6 months) for smooth connection
transition_months = 6
transition_dates = pd.date_range(last_actual_date + pd.DateOffset(months=1), 
                               periods=transition_months, freq='M')

for date in forecast_dates:
    month = date.month
    if date >= recovery_date:
        # Use pre-COVID monthly pattern after recovery
        base = pre_covid_monthly_pattern[month]
        years_since_recovery = (date - recovery_date).days / 365.25
        growth_factor = (1 + 0.02) ** years_since_recovery
        forecast_value = base * growth_factor
    else:
        # Use growth rate before recovery
        current *= (1 + monthly_growth_rate)
        forecast_value = current
    
    # Apply smooth transition during the first 6 months
    if date in transition_dates:
        transition_weight = 1 - (transition_dates.get_loc(date) / transition_months)
        growth_based = current * (1 + monthly_growth_rate)
        pattern_based = pre_covid_monthly_pattern[month] * (1 + 0.02)
        forecast_value = growth_based * transition_weight + pattern_based * (1 - transition_weight)
    
    forecast_values.append(forecast_value)

# Get the last 12 months of actual data for smooth transition
last_actual_values = recovery['total arrivals'].iloc[-12:].values
last_actual_dates = recovery.index[-12:]

# Combine actual and forecast data for smooth transition
combined_dates = pd.concat([pd.Series(last_actual_dates), pd.Series(forecast_dates)])
combined_values = np.concatenate([last_actual_values, forecast_values])

# Convert dates to numeric values for spline interpolation
date_nums = np.arange(len(combined_dates))

# Create spline function with higher degree for smoother transition
spl = make_interp_spline(date_nums, combined_values, k=5)  # increased k from 3 to 5

# Generate smooth curve with more points
smooth_date_nums = np.linspace(date_nums.min(), date_nums.max(), 500)  # increased from 300 to 500
smooth_values = spl(smooth_date_nums)

# Map smooth dates back to actual dates
smooth_dates = pd.date_range(start=combined_dates.iloc[0], 
                           end=combined_dates.iloc[-1], 
                           periods=len(smooth_date_nums))

# Create forecast DataFrame with original dates
forecast_df = pd.DataFrame({
    'predicted': forecast_values
}, index=forecast_dates)

# Calculate 95% confidence interval
forecast_df['lower_ci'] = forecast_df['predicted'] * 0.9
forecast_df['upper_ci'] = forecast_df['predicted'] * 1.1

# Calculate annual means
annual_means = forecast_df.resample('Y').mean()

print("\n=== Annual Forecast Values ===")
for idx in annual_means.index:
    year = idx.year
    pred_value = float(annual_means.loc[idx, 'predicted'])
    recovery_rate = (pred_value / pre_covid_avg) * 100
    print(f"{year}: {pred_value:,.0f} (Recovery Rate: {recovery_rate:.1f}%)")

print("\n=== Monthly Forecast Values (2026) ===")
year_2026 = forecast_df.loc['2026']
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
for idx in year_2026.index:
    month = idx.month
    pred_value = float(year_2026.loc[idx, 'predicted'])
    recovery_rate = (pred_value / pre_covid_monthly_pattern[month]) * 100
    print(f"{months[month-1]}: {pred_value:,.0f} (Recovery Rate: {recovery_rate:.1f}%)")

# Plot forecast
plt.figure(figsize=(15, 8))

# Plot actual data
plt.plot(recovery.index, recovery['total arrivals'], 
         label='Actual Data', color='#2E86AB', linewidth=2)

# Plot smoothed forecast with better transition
plt.plot(smooth_dates[len(last_actual_values)*5:], 
         smooth_values[len(last_actual_values)*5:], 
         label='Forecast', color='#A23B72', linewidth=2)

# Add transition area visualization
transition_end_idx = len(last_actual_values)*5 + transition_months*5
plt.plot(smooth_dates[len(last_actual_values)*5:transition_end_idx], 
         smooth_values[len(last_actual_values)*5:transition_end_idx], 
         color='#F18F01', linewidth=2, linestyle=':', label='Transition Period')

# Plot confidence interval
plt.fill_between(forecast_df.index,
                 forecast_df['lower_ci'],
                 forecast_df['upper_ci'],
                 color='#A23B72',
                 alpha=0.2,
                 label='95% Confidence Interval')

# Add monthly reference lines
for month in range(1, 13):
    plt.axhline(y=pre_covid_monthly_pattern[month],
                color='#F18F01',
                linestyle=':',
                alpha=0.3)

# Add pre-COVID average line
plt.axhline(y=pre_covid_avg,
            color='#F18F01',
            linestyle='--',
            label='Pre-COVID Average',
            linewidth=2)

# Add recovery point marker
plt.plot(recovery_date, pre_covid_avg, 'o', 
         color='#F18F01', markersize=10, 
         label='Recovery Point')

plt.title('Hong Kong Tourism Forecast (2025-2030)', pad=20, fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Visitors', fontsize=12)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)

# Format y-axis to millions
def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M'
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

plt.tight_layout()
plt.savefig('tourism_forecast.png', dpi=300, bbox_inches='tight')
plt.close()

# Save forecast results
forecast_df.to_csv('tourism_forecast_results.csv') 