# 香港旅游业资源优化分配模型

## 项目概述

本项目通过数据分析和数学建模方法，研究香港旅游业的关键影响因素，并构建资源优化分配模型，为香港旅游业的可持续发展提供决策支持。

## 主要内容

1. **数据分析**：分析2015-2025年香港旅游业相关数据，排除疫情期间(2020-2022年)的异常数据
2. **趋势研究**：研究访客人数、酒店价格、高铁乘客量等因素的变化趋势
3. **相关性分析**：识别影响旅游收入的关键因素
4. **多元回归模型**：构建旅游收入预测模型
5. **资源优化分配**：基于弹性系数构建资源优化分配模型
6. **政策建议**：提出具体的资源分配和实施策略

## 文件结构

- `final_report.md`：完整的研究报告
- `analysis_report.md`：数据分析报告
- `resource_allocation_model.py`：资源优化分配模型代码
- `output/`：输出结果目录
  - `tourism_recommendations.md`：政策建议
  - `sensitivity_analysis.png`：敏感性分析图表
  - `correlation_heatmap.png`：相关性热力图
  - `关键因素可视化分析/`：可视化分析结果
- `combined_monthly_data.csv`：合并后的月度数据
- `Patronage_20250301.csv`：高铁乘客量数据

## 主要发现

1. 访客人数是影响旅游收入的最显著因素（相关系数0.680）
2. 酒店入住率是第二重要的因素（相关系数0.449）
3. 高铁开通对旅游业有积极影响（相关系数0.321）
4. 酒店价格与旅游收入呈负相关（系数-0.215）

## 资源优化分配建议

基于数据分析和优化模型，建议香港旅游业资源分配比例如下：

- **交通运输业**：10.00%
- **住宿业**：40.00%
- **文化娱乐业**：50.00%

## 使用方法

### 环境要求

- Python 3.8+
- 依赖包：pandas, numpy, matplotlib, seaborn, scipy, statsmodels

### 运行代码

1. 安装依赖：
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

2. 运行资源优化分配模型：
```bash
python resource_allocation_model.py
```

3. 查看结果：
结果将保存在`output`目录下，包括：
- 政策建议文档（tourism_recommendations.md）
- 敏感性分析图表（sensitivity_analysis.png）
- 相关性热力图（correlation_heatmap.png）

## 未来工作

1. 细化各行业内部的资源分配
2. 考虑季节性因素的影响
3. 探索更多元化的旅游产品开发策略
4. 构建动态调整机制，适应市场变化

## 联系方式

如有任何问题或建议，请联系项目负责人。 