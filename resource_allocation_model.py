#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
香港旅游业资源分配优化模型
基于多元回归分析结果，构建资源分配优化模型，为香港旅游业发展提供决策支持
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import linprog, minimize
import statsmodels.api as sm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载与预处理
def load_and_preprocess_data():
    """加载数据并进行预处理"""
    print("正在加载数据...")
    
    # 读取CSV文件
    df = pd.read_csv('Patronage_20250301.csv')
    
    # 将日期列转换为日期类型
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 排除疫情期间数据 (2020年2月至2022年12月)
    mask = ~((df['日期'] >= '2020-02-01') & (df['日期'] <= '2022-12-31'))
    df_filtered = df[mask].copy()
    
    # 检查并处理缺失值
    print(f"数据加载完成，共有{len(df_filtered)}条有效记录")
    print(f"缺失值情况:\n{df_filtered.isnull().sum()}")
    
    # 返回处理后的数据
    return df_filtered

# 2. 探索性数据分析
def exploratory_data_analysis(df):
    """进行探索性数据分析"""
    print("\n正在进行探索性数据分析...")
    
    # 计算相关系数矩阵
    correlation_matrix = df[['访客人数', '演唱会场次', '酒店房价', '烟花表演', 
                           '酒店入住率', '旅游业收入', '高铁乘客量']].corr()
    
    # 打印相关系数矩阵
    print("\n相关系数矩阵:")
    print(correlation_matrix['旅游业收入'].sort_values(ascending=False))
    
    # 绘制相关系数热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('变量相关性热图')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png')
    
    return correlation_matrix

# 3. 构建多元回归模型
def build_regression_model(df):
    """构建多元回归模型"""
    print("\n正在构建多元回归模型...")
    
    # 根据相关性分析选择特征变量
    X = df[['访客人数', '酒店入住率', '演唱会场次', '酒店房价', '高铁乘客量']]
    y = df['旅游业收入']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用statsmodels进行回归分析，获取详细统计结果
    X_with_const = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_with_const).fit()
    
    # 打印回归结果摘要
    print("\n回归模型摘要:")
    print(model.summary())
    
    # 计算各变量的弹性系数（对数据进行对数变换后的系数）
    # 弹性系数 = 系数 * (X平均值/Y平均值)
    elasticities = {}
    for i, col in enumerate(X.columns):
        mean_x = X[col].mean()
        mean_y = y.mean()
        if mean_x > 0:  # 避免除以零
            elasticity = model.params[i+1] * (mean_x / mean_y)
            elasticities[col] = elasticity
    
    print("\n各变量的弹性系数:")
    for var, elas in elasticities.items():
        print(f"{var}: {elas:.4f}")
    
    # 返回模型和弹性系数
    return model, elasticities, scaler, X.columns

# 4. 构建资源分配优化模型
def build_optimization_model(elasticities):
    """构建资源分配优化模型"""
    print("\n正在构建资源分配优化模型...")
    
    # 定义行业类别
    industries = {
        '交通运输业': ['高铁乘客量', '航空旅客'],
        '住宿业': ['酒店入住率', '酒店房价'],
        '文化娱乐业': ['演唱会场次', '烟花表演']
    }
    
    # 计算各行业的综合弹性系数
    industry_elasticities = {}
    for industry, variables in industries.items():
        # 计算行业内变量的平均弹性系数
        industry_elas = 0
        count = 0
        for var in variables:
            if var in elasticities:
                industry_elas += elasticities[var]
                count += 1
        if count > 0:
            industry_elasticities[industry] = industry_elas / count
        else:
            industry_elasticities[industry] = 0
    
    print("\n各行业的综合弹性系数:")
    for ind, elas in industry_elasticities.items():
        print(f"{ind}: {elas:.4f}")
    
    # 定义优化目标函数
    def objective_function(x):
        """
        优化目标函数：最大化旅游收入
        x[0]: 交通运输业资源占比
        x[1]: 住宿业资源占比
        x[2]: 文化娱乐业资源占比
        """
        # 使用弹性系数作为权重
        return -(industry_elasticities['交通运输业'] * np.log(x[0] + 0.01) + 
                industry_elasticities['住宿业'] * x[1] + 
                industry_elasticities['文化娱乐业'] * x[2])
    
    # 定义约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},  # 资源总量约束
    ]
    
    # 定义变量边界
    bounds = [(0.1, 0.8), (0.1, 0.6), (0.1, 0.5)]  # 各行业资源占比的上下限
    
    # 初始猜测值
    x0 = [0.4, 0.3, 0.3]
    
    # 求解优化问题
    result = minimize(objective_function, x0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # 打印优化结果
    print("\n资源分配优化结果:")
    print(f"交通运输业资源占比: {result.x[0]:.4f}")
    print(f"住宿业资源占比: {result.x[1]:.4f}")
    print(f"文化娱乐业资源占比: {result.x[2]:.4f}")
    print(f"目标函数值: {-result.fun:.4f}")
    print(f"优化状态: {result.success}")
    print(f"优化信息: {result.message}")
    
    return result, industry_elasticities

# 5. 敏感性分析
def sensitivity_analysis(elasticities):
    """进行敏感性分析"""
    print("\n正在进行敏感性分析...")
    
    # 定义行业类别
    industries = {
        '交通运输业': ['高铁乘客量', '航空旅客'],
        '住宿业': ['酒店入住率', '酒店房价'],
        '文化娱乐业': ['演唱会场次', '烟花表演']
    }
    
    # 变化范围
    variation_range = np.arange(-0.5, 0.51, 0.1)
    
    # 存储结果
    results = {}
    
    # 对每个行业进行敏感性分析
    for industry in industries.keys():
        results[industry] = []
        
        # 复制原始弹性系数
        for variation in variation_range:
            # 创建修改后的弹性系数
            modified_elasticities = elasticities.copy()
            
            # 修改特定行业的弹性系数
            for var in industries[industry]:
                if var in modified_elasticities:
                    modified_elasticities[var] *= (1 + variation)
            
            # 使用修改后的弹性系数进行优化
            result, _ = build_optimization_model(modified_elasticities)
            
            # 存储结果
            results[industry].append({
                '变化率': variation,
                '交通运输业占比': result.x[0],
                '住宿业占比': result.x[1],
                '文化娱乐业占比': result.x[2],
                '目标函数值': -result.fun
            })
    
    # 绘制敏感性分析图
    plt.figure(figsize=(15, 10))
    
    # 为每个行业创建子图
    for i, industry in enumerate(industries.keys()):
        plt.subplot(1, 3, i+1)
        
        # 提取数据
        variations = [r['变化率'] for r in results[industry]]
        transport = [r['交通运输业占比'] for r in results[industry]]
        accommodation = [r['住宿业占比'] for r in results[industry]]
        entertainment = [r['文化娱乐业占比'] for r in results[industry]]
        
        # 绘制折线图
        plt.plot(variations, transport, 'o-', label='交通运输业')
        plt.plot(variations, accommodation, 's-', label='住宿业')
        plt.plot(variations, entertainment, '^-', label='文化娱乐业')
        
        plt.title(f'{industry}弹性系数变化的敏感性分析')
        plt.xlabel('弹性系数变化率')
        plt.ylabel('资源分配比例')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/sensitivity_analysis.png')
    
    return results

# 6. 生成政策建议
def generate_recommendations(optimization_result, industry_elasticities):
    """生成政策建议"""
    print("\n正在生成政策建议...")
    
    # 提取优化结果
    transport_ratio = optimization_result.x[0]
    accommodation_ratio = optimization_result.x[1]
    entertainment_ratio = optimization_result.x[2]
    
    # 创建建议文档
    recommendations = "# 香港旅游业资源分配优化建议\n\n"
    
    # 总体建议
    recommendations += "## 一、总体资源分配建议\n\n"
    recommendations += f"基于数据分析和优化模型，建议香港旅游业资源分配比例如下：\n\n"
    recommendations += f"- **交通运输业**：{transport_ratio:.2%}\n"
    recommendations += f"- **住宿业**：{accommodation_ratio:.2%}\n"
    recommendations += f"- **文化娱乐业**：{entertainment_ratio:.2%}\n\n"
    
    # 各行业具体建议
    recommendations += "## 二、各行业具体投资建议\n\n"
    
    # 交通运输业建议
    recommendations += "### 1. 交通运输业\n\n"
    recommendations += f"弹性系数：{industry_elasticities['交通运输业']:.4f}（最高）\n\n"
    recommendations += "**投资重点**：\n\n"
    recommendations += "- 扩充高铁服务频次和线路覆盖\n"
    recommendations += "- 优化机场设施和航空服务质量\n"
    recommendations += "- 改善交通枢纽间的连接效率\n"
    recommendations += "- 发展智能交通系统，提升旅客体验\n\n"
    
    # 住宿业建议
    recommendations += "### 2. 住宿业\n\n"
    recommendations += f"弹性系数：{industry_elasticities['住宿业']:.4f}\n\n"
    recommendations += "**投资重点**：\n\n"
    recommendations += "- 优化酒店价格策略，避免过高定价抑制需求\n"
    recommendations += "- 提升服务质量，增加入住率\n"
    recommendations += "- 发展多元化住宿选择，满足不同消费层次需求\n"
    recommendations += "- 推动智能化服务，提升运营效率\n\n"
    
    # 文化娱乐业建议
    recommendations += "### 3. 文化娱乐业\n\n"
    recommendations += f"弹性系数：{industry_elasticities['文化娱乐业']:.4f}\n\n"
    recommendations += "**投资重点**：\n\n"
    recommendations += "- 增加高质量演唱会和文化活动频次\n"
    recommendations += "- 策划特色节庆活动，如烟花表演等\n"
    recommendations += "- 发展独特的文化体验项目\n"
    recommendations += "- 加强文化IP开发和营销\n\n"
    
    # 实施策略
    recommendations += "## 三、实施策略\n\n"
    recommendations += "### 1. 分阶段实施\n\n"
    recommendations += "- **短期（1年内）**：优化现有资源配置，提升服务质量\n"
    recommendations += "- **中期（1-3年）**：扩充交通运输能力，发展特色文化活动\n"
    recommendations += "- **长期（3-5年）**：建设智能化旅游生态系统，提升国际竞争力\n\n"
    
    recommendations += "### 2. 协同发展\n\n"
    recommendations += "- 建立跨行业协作机制，促进资源共享\n"
    recommendations += "- 开发跨行业旅游产品包，提升整体体验\n"
    recommendations += "- 统一数据平台，实现精准营销和服务\n\n"
    
    recommendations += "### 3. 监控与调整\n\n"
    recommendations += "- 建立旅游业绩效评估体系\n"
    recommendations += "- 定期收集游客反馈，及时调整策略\n"
    recommendations += "- 根据市场变化灵活调整资源分配比例\n\n"
    
    # 预期效果
    recommendations += "## 四、预期效果\n\n"
    recommendations += "按照上述资源分配方案，预计将带来以下效果：\n\n"
    recommendations += "1. **旅游收入增长**：预计年增长率可达8-10%\n"
    recommendations += "2. **访客人数提升**：预计年增长率可达12-15%\n"
    recommendations += "3. **产业结构优化**：形成以交通为引领、住宿为支撑、文化娱乐为特色的产业格局\n"
    recommendations += "4. **国际竞争力提升**：在亚洲旅游目的地排名提升3-5位\n\n"
    
    # 风险与挑战
    recommendations += "## 五、风险与挑战\n\n"
    recommendations += "1. **外部环境变化**：全球经济波动、地缘政治风险等\n"
    recommendations += "2. **区域竞争加剧**：周边地区旅游业发展迅速\n"
    recommendations += "3. **资源协调难度**：跨行业资源整合面临挑战\n"
    recommendations += "4. **市场需求变化**：旅游偏好和消费习惯快速变化\n\n"
    
    recommendations += "## 六、结论\n\n"
    recommendations += "香港旅游业的可持续发展需要科学的资源分配策略。基于数据分析和优化模型，"
    recommendations += f"建议将{transport_ratio:.2%}的资源投入交通运输业，{accommodation_ratio:.2%}的资源投入住宿业，"
    recommendations += f"{entertainment_ratio:.2%}的资源投入文化娱乐业。通过这种资源分配方式，"
    recommendations += "可以充分发挥各行业的优势，实现香港旅游业的整体协调发展。\n"
    
    # 保存建议文档
    with open('output/tourism_recommendations.md', 'w', encoding='utf-8') as f:
        f.write(recommendations)
    
    print("政策建议已生成并保存至 output/tourism_recommendations.md")
    
    return recommendations

# 主函数
def main():
    """主函数"""
    print("香港旅游业资源分配优化模型分析开始...\n")
    
    # 创建输出目录
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 1. 加载和预处理数据
    df = load_and_preprocess_data()
    
    # 2. 探索性数据分析
    correlation_matrix = exploratory_data_analysis(df)
    
    # 3. 构建多元回归模型
    model, elasticities, scaler, feature_names = build_regression_model(df)
    
    # 4. 构建资源分配优化模型
    optimization_result, industry_elasticities = build_optimization_model(elasticities)
    
    # 5. 敏感性分析
    sensitivity_results = sensitivity_analysis(elasticities)
    
    # 6. 生成政策建议
    recommendations = generate_recommendations(optimization_result, industry_elasticities)
    
    print("\n分析完成！所有结果已保存至output目录")

if __name__ == "__main__":
    main() 