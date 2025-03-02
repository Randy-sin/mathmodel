import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

def calculate_metrics(actual, predicted):
    """计算评估指标"""
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mape, rmse

def rolling_forecast(train_data, test_data, seasonal_period=12):
    """执行滚动预测"""
    results = []
    train = train_data.copy()
    
    for i in range(len(test_data)):
        # 训练模型
        model = SARIMAX(train, 
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, seasonal_period))
        model_fit = model.fit()
        
        # 预测下一个时间点
        forecast = model_fit.forecast(steps=1)
        
        # 记录结果
        actual = test_data.iloc[i]
        predicted = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
        results.append({
            'date': test_data.index[i],
            'actual': actual,
            'predicted': predicted
        })
        
        # 更新训练数据
        train = pd.concat([train, pd.Series([actual], index=[test_data.index[i]])])
    
    return pd.DataFrame(results)

def plot_rolling_forecast_results(results):
    """绘制滚动预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 绘制实际值和预测值
    plt.plot(results['date'], results['actual'], 
             label='Actual', color='#2E86AB', linewidth=2)
    plt.plot(results['date'], results['predicted'], 
             label='Predicted', color='#A23B72', linewidth=2, linestyle='--')
    
    # 计算并显示评估指标
    mape, rmse = calculate_metrics(results['actual'], results['predicted'])
    plt.title(f'Rolling Forecast Validation Results\nMAPE: {mape:.2f}%, RMSE: {rmse:.2f}', pad=20)
    
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化y轴为百万
    def millions_formatter(x, pos):
        return f'{x/1e6:.1f}M'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
    
    plt.tight_layout()
    plt.savefig('rolling_forecast_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def residual_analysis(model_fit):
    """执行残差分析"""
    # 获取残差
    residuals = model_fit.resid
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Residual Diagnostics', fontsize=16, y=1.02)
    
    # 1. 残差时间序列图
    axes[0, 0].plot(residuals.index, residuals, color='#2E86AB')
    axes[0, 0].set_title('Residual Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    
    # 2. 残差直方图
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='#A23B72')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    
    # 计算最大滞后期数（不超过样本量的50%）
    max_lags = min(int(len(residuals) * 0.5) - 1, 24)
    
    # 3. ACF图
    plot_acf(residuals, ax=axes[1, 0], lags=max_lags, alpha=0.05)
    axes[1, 0].set_title('Autocorrelation Function (ACF)')
    
    # 4. PACF图
    plot_pacf(residuals, ax=axes[1, 1], lags=max_lags, alpha=0.05)
    axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回残差数据
    return pd.DataFrame({
        'date': residuals.index,
        'residual': residuals.values
    })

def main():
    # 读取数据
    df = pd.read_csv('data/processed/processed_tourism_data_with_dummies.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # ===== 第一部分：疫情前数据验证 =====
    print("\n=== 疫情前数据验证 ===")
    pre_covid_data = df[:'2019-05']['total arrivals']
    
    # 使用前3年数据作为训练集，其余作为测试集
    train_end_date = '2015-12-31'
    train_data = pre_covid_data[:'2015-12-31']
    test_data = pre_covid_data['2016-01-01':]
    
    print(f"\n训练数据期间: {train_data.index[0].strftime('%Y-%m')} 到 {train_data.index[-1].strftime('%Y-%m')}")
    print(f"测试数据期间: {test_data.index[0].strftime('%Y-%m')} 到 {test_data.index[-1].strftime('%Y-%m')}")
    print(f"训练集样本数: {len(train_data)} 个月")
    print(f"测试集样本数: {len(test_data)} 个月")
    
    # 执行滚动预测
    print("\n执行滚动预测验证...")
    forecast_results = rolling_forecast(train_data, test_data)
    plot_rolling_forecast_results(forecast_results)
    
    # 计算并打印评估指标
    mape, rmse = calculate_metrics(forecast_results['actual'], 
                                 forecast_results['predicted'])
    print(f"\n疫情前模型评估指标:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    
    # 保存疫情前验证结果
    results_df = pd.DataFrame({
        'date': forecast_results['date'],
        'actual': forecast_results['actual'],
        'predicted': forecast_results['predicted'],
        'error': forecast_results['actual'] - forecast_results['predicted'],
        'error_percentage': (forecast_results['actual'] - forecast_results['predicted']) / forecast_results['actual'] * 100
    })
    results_df.to_csv('pre_covid_validation_results.csv', index=False)
    
    # ===== 第二部分：恢复期数据验证 =====
    print("\n=== 恢复期数据验证 ===")
    # 使用疫情前全部数据作为训练集，恢复期数据作为测试集
    recovery_train_data = df[:'2019-05']['total arrivals']
    recovery_test_data = df['2023-03':]['total arrivals']
    
    print(f"\n训练数据期间: {recovery_train_data.index[0].strftime('%Y-%m')} 到 {recovery_train_data.index[-1].strftime('%Y-%m')}")
    print(f"测试数据期间: {recovery_test_data.index[0].strftime('%Y-%m')} 到 {recovery_test_data.index[-1].strftime('%Y-%m')}")
    print(f"训练集样本数: {len(recovery_train_data)} 个月")
    print(f"测试集样本数: {len(recovery_test_data)} 个月")
    
    # 执行恢复期滚动预测
    print("\n执行恢复期滚动预测验证...")
    recovery_forecast_results = rolling_forecast(recovery_train_data, recovery_test_data)
    
    # 保存图表为不同的文件名
    plt.figure(figsize=(15, 10))
    plt.plot(recovery_forecast_results['date'], recovery_forecast_results['actual'], 
             label='Actual', color='#2E86AB', linewidth=2)
    plt.plot(recovery_forecast_results['date'], recovery_forecast_results['predicted'], 
             label='Predicted', color='#A23B72', linewidth=2, linestyle='--')
    
    # 计算并显示评估指标
    recovery_mape, recovery_rmse = calculate_metrics(
        recovery_forecast_results['actual'], 
        recovery_forecast_results['predicted']
    )
    plt.title(f'Recovery Period Forecast Validation Results\nMAPE: {recovery_mape:.2f}%, RMSE: {recovery_rmse:.2f}', pad=20)
    
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化y轴为百万
    def millions_formatter(x, pos):
        return f'{x/1e6:.1f}M'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
    
    plt.tight_layout()
    plt.savefig('recovery_forecast_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n恢复期模型评估指标:")
    print(f"MAPE: {recovery_mape:.2f}%")
    print(f"RMSE: {recovery_rmse:.2f}")
    
    # 保存恢复期验证结果
    recovery_results_df = pd.DataFrame({
        'date': recovery_forecast_results['date'],
        'actual': recovery_forecast_results['actual'],
        'predicted': recovery_forecast_results['predicted'],
        'error': recovery_forecast_results['actual'] - recovery_forecast_results['predicted'],
        'error_percentage': (recovery_forecast_results['actual'] - recovery_forecast_results['predicted']) / recovery_forecast_results['actual'] * 100
    })
    
    # 按月份统计恢复期预测误差
    monthly_errors = recovery_results_df.copy()
    monthly_errors['month'] = monthly_errors['date'].dt.month
    monthly_error_stats = monthly_errors.groupby('month').agg({
        'error_percentage': ['mean', 'std', 'count']
    }).round(2)
    
    print("\n恢复期各月份预测误差统计：")
    months = ['一月', '二月', '三月', '四月', '五月', '六月', 
             '七月', '八月', '九月', '十月', '十一月', '十二月']
    for month in range(1, 13):
        if month in monthly_error_stats.index:
            mean_error = monthly_error_stats.loc[month, ('error_percentage', 'mean')]
            std_error = monthly_error_stats.loc[month, ('error_percentage', 'std')]
            count = monthly_error_stats.loc[month, ('error_percentage', 'count')]
            print(f"{months[month-1]}: 平均误差 {mean_error:+.2f}%, 标准差 {std_error:.2f}%, 样本数 {count:.0f}")
    
    # 保存恢复期结果
    recovery_results_df.to_csv('recovery_validation_results.csv', index=False)
    monthly_error_stats.to_csv('recovery_monthly_error_stats.csv')
    
    print("\n分析完成！结果已保存为以下文件：")
    print("1. pre_covid_validation_results.csv - 疫情前验证结果")
    print("2. recovery_validation_results.csv - 恢复期验证结果")
    print("3. recovery_monthly_error_stats.csv - 恢复期月度误差统计")
    print("4. rolling_forecast_validation.png - 疫情前预测验证图")
    print("5. recovery_forecast_validation.png - 恢复期预测验证图")

if __name__ == "__main__":
    main() 