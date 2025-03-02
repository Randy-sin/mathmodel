import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# Set font for displaying text
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Read data
data = pd.DataFrame({
    'Year': [2015, 2016, 2017, 2018, 2019, 2023, 2024],
    'Concert Count': [15, 116, 99, 122, 93, 84, 72],
    'Hotel Price': [1447.50, 1287.08, 1286.75, 1375.42, 1206.00, 1395.33, 1316.30],
    'Fireworks Shows': [1, 3, 3, 2, 2, 3, 2],
    'Hotel Occupancy Rate': [85.00, 85.92, 86.92, 87.92, 85.92, 82.25, 84.20],
    'Tourism Revenue': [20.00, 20.83, 19.00, 21.05, 21.33, 14.83, 17.28],
    'Visitor Count': [9809779, 56654903, 58472157, 65147555, 55912609, 33999660, 36678799]
})

# Create directory to save charts
import os
if not os.path.exists('visualization_results'):
    os.makedirs('visualization_results')

# Set chart style
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-pastel')

# Figure 1: Visitor count trend over time
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Visitor Count'] / 1000000, marker='o', linewidth=2, markersize=10)
plt.title('Hong Kong Tourism Visitor Count Trend (2015-2024)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Visitor Count (Millions)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(data['Year'], rotation=45)

# Add data labels
for x, y in zip(data['Year'], data['Visitor Count'] / 1000000):
    plt.text(x, y + 1, f'{y:.2f}', ha='center', va='bottom', fontsize=10)

# Highlight pandemic period
plt.axvspan(2019.5, 2022.5, alpha=0.2, color='red')
plt.text(2021, max(data['Visitor Count'] / 1000000) * 0.8, 'Pandemic Period\n(Data Excluded)', 
         ha='center', va='center', fontsize=12, color='red', 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.6))

plt.tight_layout()
plt.savefig('visualization_results/visitor_trend.png', dpi=300)

# Figure 2: Correlation coefficients
correlation_data = {
    'Factor': ['Concert Count', 'Hotel Occupancy Rate', 'Fireworks Shows', 'Tourism Revenue', 'Hotel Price', 'High-Speed Rail Passengers'],
    'Correlation': [0.951, 0.653, 0.554, 0.405, -0.662, -0.108]
}
corr_df = pd.DataFrame(correlation_data)
corr_df = corr_df.sort_values('Correlation', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(corr_df['Factor'], corr_df['Correlation'], color=['#2ecc71' if x >= 0 else '#e74c3c' for x in corr_df['Correlation']])
plt.title('Correlation Coefficients with Visitor Count', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.7, axis='x')

# Add data labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_x_pos = width + 0.03 if width > 0 else width - 0.08
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{corr_df["Correlation"].iloc[i]:.3f}', 
             va='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('visualization_results/correlation_factors.png', dpi=300)

# Figure 3: Elasticity coefficients
elasticity_data = {
    'Factor': ['Hotel Price', 'Concert Count', 'Hotel Occupancy Rate'],
    'Elasticity': [-100.01, 24.91, 8.98]
}
elas_df = pd.DataFrame(elasticity_data)

plt.figure(figsize=(10, 6))
bars = plt.bar(elas_df['Factor'], elas_df['Elasticity'], color=['#e74c3c', '#2ecc71', '#3498db'])
plt.title('Elasticity Coefficients for Visitor Count', fontsize=16)
plt.ylabel('Elasticity Coefficient', fontsize=14)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Add data labels
for bar in bars:
    height = bar.get_height()
    label_y_pos = height + 5 if height > 0 else height - 10
    plt.text(bar.get_x() + bar.get_width()/2, label_y_pos, f'{height:.2f}', 
             ha='center', va='center' if height > 0 else 'top', fontsize=12, 
             color='black' if height > 0 else 'white')

plt.tight_layout()
plt.savefig('visualization_results/elasticity_coefficients.png', dpi=300)

# Figure 4: Actual vs Predicted values
prediction_data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2023, 2024],
    'Actual Visitor Count': [9809779, 56654903, 58472157, 65147555, 55912609, 33999660, 36678799],
    'Predicted Visitor Count': [10581290, 60484250, 56514220, 64389110, 55028540, 32736100, 36941970]
}
pred_df = pd.DataFrame(prediction_data)

plt.figure(figsize=(12, 6))
plt.plot(pred_df['Year'], pred_df['Actual Visitor Count'] / 1000000, marker='o', linewidth=2, markersize=8, label='Actual')
plt.plot(pred_df['Year'], pred_df['Predicted Visitor Count'] / 1000000, marker='s', linewidth=2, linestyle='--', markersize=8, label='Predicted')
plt.title('Actual vs Predicted Visitor Count', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Visitor Count (Millions)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(pred_df['Year'], rotation=45)

# Calculate and display MAPE
mape = np.mean(np.abs((pred_df['Actual Visitor Count'] - pred_df['Predicted Visitor Count']) / pred_df['Actual Visitor Count'])) * 100
plt.text(2020, max(pred_df['Actual Visitor Count'] / 1000000) * 0.5, f'MAPE: {mape:.2f}%', 
         ha='center', va='center', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig('visualization_results/actual_vs_predicted.png', dpi=300)

# Figure 5: Resource allocation recommendation
resource_data = {
    'Investment Area': ['Hotel Price Subsidies', 'Concert Support', 'Hotel Service Quality'],
    'Allocation Percentage': [70, 20, 10]
}
res_df = pd.DataFrame(resource_data)

plt.figure(figsize=(10, 7))
colors = ['#3498db', '#2ecc71', '#f1c40f']
plt.pie(res_df['Allocation Percentage'], labels=res_df['Investment Area'], autopct='%1.1f%%', 
        startangle=90, colors=colors, shadow=True, explode=(0.1, 0, 0))
plt.title('Recommended Resource Allocation', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.tight_layout()
plt.savefig('visualization_results/resource_allocation.png', dpi=300)

# Figure 6: Trends of various factors over time
plt.figure(figsize=(14, 10))

# Create 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Concert count trend
axs[0, 0].plot(data['Year'], data['Concert Count'], marker='o', linewidth=2, color='#3498db')
axs[0, 0].set_title('Concert Count Trend', fontsize=14)
axs[0, 0].set_xlabel('Year', fontsize=12)
axs[0, 0].set_ylabel('Number of Concerts', fontsize=12)
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].set_xticks(data['Year'])
axs[0, 0].set_xticklabels(data['Year'], rotation=45)

# Hotel price trend
axs[0, 1].plot(data['Year'], data['Hotel Price'], marker='o', linewidth=2, color='#e74c3c')
axs[0, 1].set_title('Hotel Price Trend', fontsize=14)
axs[0, 1].set_xlabel('Year', fontsize=12)
axs[0, 1].set_ylabel('Price (HKD)', fontsize=12)
axs[0, 1].grid(True, linestyle='--', alpha=0.7)
axs[0, 1].set_xticks(data['Year'])
axs[0, 1].set_xticklabels(data['Year'], rotation=45)

# Hotel occupancy rate trend
axs[1, 0].plot(data['Year'], data['Hotel Occupancy Rate'], marker='o', linewidth=2, color='#2ecc71')
axs[1, 0].set_title('Hotel Occupancy Rate Trend', fontsize=14)
axs[1, 0].set_xlabel('Year', fontsize=12)
axs[1, 0].set_ylabel('Occupancy Rate (%)', fontsize=12)
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].set_xticks(data['Year'])
axs[1, 0].set_xticklabels(data['Year'], rotation=45)
axs[1, 0].set_ylim(80, 90)  # Set y-axis range to make changes more visible

# Tourism revenue trend
axs[1, 1].plot(data['Year'], data['Tourism Revenue'], marker='o', linewidth=2, color='#f39c12')
axs[1, 1].set_title('Tourism Revenue Trend', fontsize=14)
axs[1, 1].set_xlabel('Year', fontsize=12)
axs[1, 1].set_ylabel('Revenue (Billion HKD)', fontsize=12)
axs[1, 1].grid(True, linestyle='--', alpha=0.7)
axs[1, 1].set_xticks(data['Year'])
axs[1, 1].set_xticklabels(data['Year'], rotation=45)

plt.tight_layout()
plt.savefig('visualization_results/factors_trends.png', dpi=300)

# Figure 7: Scatter plot matrix - Relationships between factors and visitor count
factors = ['Concert Count', 'Hotel Price', 'Hotel Occupancy Rate', 'Visitor Count']
scatter_data = data.rename(columns={
    '演唱会场次': 'Concert Count',
    '酒店房价': 'Hotel Price',
    '酒店入住率': 'Hotel Occupancy Rate',
    '访客人数': 'Visitor Count'
})[factors]

plt.figure(figsize=(12, 10))
sns.pairplot(scatter_data, height=2.5, aspect=1.2, 
             plot_kws=dict(alpha=0.7, edgecolor='k', linewidth=0.5, s=100))
plt.suptitle('Scatter Plot Matrix of Factors vs Visitor Count', fontsize=16, y=1.02)
plt.savefig('visualization_results/scatter_matrix.png', dpi=300)

print("All visualization charts have been generated and saved to the 'visualization_results' directory") 