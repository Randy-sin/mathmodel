import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# 读取年度数据
df = pd.read_csv("yearly_tourism_data.csv")
print("数据概览：")
print(df)

# 计算相关系数
print("\n相关系数矩阵（与访客人数的相关性）：")
corr = df.corr()["访客人数"].sort_values(ascending=False)
print(corr)

# 准备建模数据
X = df[["演唱会场次", "酒店房价", "烟花表演", "酒店入住率", "高铁乘客量"]]
y = df["访客人数"]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = sm.add_constant(X_scaled)

# 建立模型
model = sm.OLS(y, X_scaled).fit()

# 输出模型结果
print("\n模型摘要：")
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