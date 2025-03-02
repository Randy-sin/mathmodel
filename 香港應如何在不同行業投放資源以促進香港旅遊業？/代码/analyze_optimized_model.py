import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取年度数据
df = pd.read_csv("yearly_tourism_data.csv")
print("数据概览：")
print(df)

# 计算相关系数
print("\n相关系数矩阵（与访客人数的相关性）：")
corr = df.corr()["访客人数"].sort_values(ascending=False)
print(corr)

# 只选择相关性较高的变量（绝对值大于0.6）
selected_features = []
for feature, correlation in corr.items():
    if abs(correlation) > 0.6 and feature != "访客人数":
        selected_features.append(feature)

print(f"\n选择的特征变量（相关性绝对值>0.6）: {selected_features}")

# 准备建模数据
X = df[selected_features]
y = df["访客人数"]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = sm.add_constant(X_scaled)

# 建立模型
model = sm.OLS(y, X_scaled).fit()

# 输出模型结果
print("\n优化模型摘要：")
print(model.summary())

# 计算弹性系数
print("\n弹性系数（标准化后的影响力）：")
elasticities = {}
for i, var in enumerate(X.columns):
    elasticities[var] = model.params[i+1] * np.mean(X[var]) / np.mean(y)
    
for var, elasticity in sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{var}: {elasticity:.4f}")

# 计算预测值和残差
df["预测访客人数"] = model.predict(X_scaled)
df["残差"] = df["访客人数"] - df["预测访客人数"]

# 输出预测结果
print("\n预测结果：")
print(df[["年份", "访客人数", "预测访客人数", "残差"]])

# 计算模型评估指标
print("\n模型评估指标：")
print(f"R²: {model.rsquared:.4f}")
print(f"调整后的R²: {model.rsquared_adj:.4f}")
print(f"F统计量: {model.fvalue:.4f}")
print(f"F统计量的p值: {model.f_pvalue:.4f}")
print(f"均方误差(MSE): {np.mean(model.resid**2):.4f}")
print(f"平均绝对误差(MAE): {np.mean(np.abs(model.resid)):.4f}")
print(f"平均绝对百分比误差(MAPE): {np.mean(np.abs(model.resid/y))*100:.4f}%")

# 输出变量的标准化系数
print("\n标准化系数（直接比较各变量的影响力）：")
for i, var in enumerate(X.columns):
    print(f"{var}: {model.params[i+1]:.4f}")

# 比较原始数据和预测数据
print("\n原始数据与预测数据比较：")
comparison = pd.DataFrame({
    "年份": df["年份"],
    "实际访客人数": df["访客人数"],
    "预测访客人数": df["预测访客人数"],
    "预测误差": df["残差"],
    "预测误差百分比": (df["残差"] / df["访客人数"]) * 100
})
print(comparison)

# 输出模型方程
print("\n模型方程：")
equation = f"访客人数 = {model.params[0]:.2f}"
for i, var in enumerate(X.columns):
    equation += f" + {model.params[i+1]:.2f} × {var}"
print(equation)

# 保存优化模型的结果
with open("optimized_model_results.txt", "w") as f:
    f.write("优化模型分析结果\n")
    f.write("=================\n\n")
    f.write(f"选择的特征变量: {selected_features}\n\n")
    f.write(f"模型R²: {model.rsquared:.4f}\n")
    f.write(f"调整后的R²: {model.rsquared_adj:.4f}\n")
    f.write(f"F统计量: {model.fvalue:.4f}\n")
    f.write(f"F统计量的p值: {model.f_pvalue:.4f}\n")
    f.write(f"均方误差(MSE): {np.mean(model.resid**2):.4f}\n")
    f.write(f"平均绝对误差(MAE): {np.mean(np.abs(model.resid)):.4f}\n")
    f.write(f"平均绝对百分比误差(MAPE): {np.mean(np.abs(model.resid/y))*100:.4f}%\n\n")
    f.write("模型方程：\n")
    f.write(equation)
    
print("\n优化模型分析结果已保存到 optimized_model_results.txt") 