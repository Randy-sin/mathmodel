import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Read data
df = pd.read_csv('data/processed/processed_tourism_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate baseline and prepare data
baseline_2019 = df[(df['date'] >= '2019-01-01') & (df['date'] < '2019-07-01')]['total arrivals'].mean()

# Get seasonal factors
pre_covid = df[df['date'] < '2019-07-01'].copy()
pre_covid['month'] = pre_covid['date'].dt.month
seasonal_factors = pre_covid.groupby('month')['total arrivals'].mean() / pre_covid['total arrivals'].mean()

# Prepare validation data (using 2023-2025 as validation period)
validation_data = df[df['date'] >= '2023-01-01'].copy()
validation_dates = validation_data['date']
actual_values = validation_data['total arrivals'].values

# S-curve forecast
def recovery_curve(x, max_value, current_value, years):
    return max_value - (max_value - current_value) * np.exp(-0.5 * x / years)

# Generate S-curve predictions for validation period
months_since_2023 = np.arange(len(validation_data)) / 12
current_level = df[df['date'] < '2023-01-01']['total arrivals'].iloc[-12:].mean()
s_curve_base = recovery_curve(months_since_2023, baseline_2019, current_level, 3)
validation_data['month'] = validation_data['date'].dt.month
s_curve_pred = s_curve_base * validation_data['month'].map(seasonal_factors)

# ARIMA predictions
normal_periods = df[(df['date'] < '2019-07-01') | (df['date'] >= '2023-01-01')].copy()
normal_periods = normal_periods.set_index('date')['total arrivals']
model = ARIMA(normal_periods, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
arima_pred = results.predict(start=validation_dates.min(), end=validation_dates.max())

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

s_curve_metrics = calculate_metrics(actual_values, s_curve_pred)
arima_metrics = calculate_metrics(actual_values, arima_pred)

# Create visualization
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

# Plot 1: Predictions vs Actual
ax1 = plt.subplot(gs[0, :])
ax1.plot(validation_dates, actual_values, 'b-', label='Actual Values', linewidth=2)
ax1.plot(validation_dates, s_curve_pred, 'r--', label='S-Curve Prediction', linewidth=2)
ax1.plot(validation_dates, arima_pred, 'g--', label='ARIMA Prediction', linewidth=2)
ax1.set_title('Model Predictions vs Actual Values (2023-2025)', fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Monthly Visitors', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Format y-axis to show millions
def millions_formatter(x, pos):
    return f'{int(x/1000000)}M'
ax1.yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

# Plot 2: Metrics Comparison
metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
s_curve_values = [s_curve_metrics[m] for m in metrics_names]
arima_values = [arima_metrics[m] for m in metrics_names]

# Normalize metrics for better visualization
normalized_metrics = pd.DataFrame({
    'Metric': metrics_names * 2,
    'Model': ['S-Curve'] * 5 + ['ARIMA'] * 5,
    'Value': s_curve_values + arima_values
})

ax2 = plt.subplot(gs[1, 0])
sns.barplot(data=normalized_metrics, x='Metric', y='Value', hue='Model', ax=ax2)
ax2.set_title('Model Performance Metrics', fontsize=14)
ax2.set_ylabel('Value', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Error Distribution
ax3 = plt.subplot(gs[1, 1])
s_curve_errors = actual_values - s_curve_pred
arima_errors = actual_values - arima_pred
sns.kdeplot(data=s_curve_errors, label='S-Curve Errors', ax=ax3)
sns.kdeplot(data=arima_errors, label='ARIMA Errors', ax=ax3)
ax3.set_title('Error Distribution', fontsize=14)
ax3.set_xlabel('Error', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.legend()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed metrics
print("\n=== Model Evaluation Metrics ===")
print("\nS-Curve Model:")
for metric, value in s_curve_metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:,.2f}")

print("\nARIMA Model:")
for metric, value in arima_metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:,.2f}")

# Calculate overall score (weighted average of normalized metrics)
def calculate_overall_score(metrics):
    # Normalize and weight the metrics
    weights = {'MSE': 0.1, 'RMSE': 0.2, 'MAE': 0.2, 'MAPE': 0.2, 'R²': 0.3}
    max_metrics = {'MSE': max(s_curve_metrics['MSE'], arima_metrics['MSE']),
                  'RMSE': max(s_curve_metrics['RMSE'], arima_metrics['RMSE']),
                  'MAE': max(s_curve_metrics['MAE'], arima_metrics['MAE']),
                  'MAPE': max(s_curve_metrics['MAPE'], arima_metrics['MAPE']),
                  'R²': 1}
    
    score = 0
    for metric, weight in weights.items():
        if metric == 'R²':
            score += weight * metrics[metric]
        else:
            score += weight * (1 - metrics[metric]/max_metrics[metric])
    return score * 100

s_curve_score = calculate_overall_score(s_curve_metrics)
arima_score = calculate_overall_score(arima_metrics)

print("\n=== Overall Model Scores (0-100) ===")
print(f"S-Curve Model: {s_curve_score:.1f}")
print(f"ARIMA Model: {arima_score:.1f}") 