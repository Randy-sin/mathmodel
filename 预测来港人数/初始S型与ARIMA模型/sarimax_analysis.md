# SARIMAX模型参数优化分析报告

## 1. 非疫情期间数据分析

### 1.1 平稳性检验
- **ADF检验**：
  - 统计量：-1.6335
  - p值：0.4656 > 0.05，表明数据不平稳
  - 需要进行差分处理

- **KPSS检验**：
  - 统计量：0.5350
  - p值：0.0338 < 0.05，拒绝趋势平稳的原假设
  - 进一步确认数据不平稳

### 1.2 差分分析
- 原始数据 ADF p值：0.4656
- 一阶差分 ADF p值：0.1241
- 季节性差分 ADF p值：0.4368
- 双重差分 ADF p值：0.0022

结论：数据需要进行季节性差分以达到平稳。

### 1.3 最优模型参数
- SARIMAX(1,0,1)(0,1,1,12)
- AIC值：2046.35
- 参数解释：
  - 非季节性部分：ARIMA(1,0,1)
    - AR(1)：一阶自回归
    - 无需一阶差分
    - MA(1)：一阶移动平均
  - 季节性部分：(0,1,1,12)
    - 无季节性AR项
    - 需要季节性差分
    - 季节性MA(1)项
    - 12个月的季节性周期

## 2. 疫情期间数据分析

### 2.1 平稳性检验
- **ADF检验**：
  - 统计量：-1.3943
  - p值：0.5850 > 0.05，表明数据不平稳
  - 需要进行差分处理

- **KPSS检验**：
  - 统计量：0.3582
  - p值：0.0952 > 0.05，不能拒绝趋势平稳的原假设
  - 数据表现出一定的趋势平稳性

### 2.2 差分分析
- 原始数据 ADF p值：0.5850
- 一阶差分 ADF p值：0.0000
- 季节性差分 ADF p值：0.1299
- 双重差分 ADF p值：0.0016

结论：数据需要进行一阶差分以达到平稳。

### 2.3 最优模型参数
- SARIMAX(1,1,2)(0,0,1,12)
- AIC值：909.27
- 参数解释：
  - 非季节性部分：ARIMA(1,1,2)
    - AR(1)：一阶自回归
    - 需要一阶差分
    - MA(2)：二阶移动平均
  - 季节性部分：(0,0,1,12)
    - 无季节性AR项
    - 无需季节性差分
    - 季节性MA(1)项
    - 12个月的季节性周期

## 3. 建议与实施

### 3.1 模型选择建议
1. **非疫情期间**：使用SARIMAX(1,0,1)(0,1,1,12)
   - 重点关注季节性变化
   - 使用季节性差分处理年度周期性
   - 通过AR(1)和MA(1)项捕捉短期波动

2. **疫情期间**：使用SARIMAX(1,1,2)(0,0,1,12)
   - 重点关注趋势变化
   - 使用一阶差分处理趋势不平稳性
   - 通过MA(2)项更好地捕捉短期波动的复杂性

### 3.2 实施注意事项
1. 在模型拟合时需要注意：
   - 对外生变量进行标准化处理
   - 处理缺失值和异常值
   - 考虑节假日效应的影响

2. 在预测时需要注意：
   - 对差分结果进行还原
   - 考虑预测区间的不确定性
   - 定期更新模型参数 