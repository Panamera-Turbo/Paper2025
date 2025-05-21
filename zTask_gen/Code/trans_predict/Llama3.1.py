import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 给定的预测结果和标准值
predictions = [
    218.55128198862076,
    230.53003415465355,
    223.62081795930862,
    223.62081795930862,
    217.90933892130852,
    233.92052590847015,
    237.41092383861542,
    217.90933892130852,
    217.90933892130852,
    223.62081795930862
]
true_value = 248

# 将标准值转换为与预测值相同长度的数组
y_true = np.full(len(predictions), true_value)
y_pred = np.array(predictions)

# 计算各项指标
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

# 打印结果
print(f"MAE (平均绝对误差): {mae:.4f}")
print(f"MSE (均方误差): {mse:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")
print(f"MAPE (平均绝对百分比误差): {mape:.4f}%")
print(f"R² (决定系数): {r2:.4f}")
