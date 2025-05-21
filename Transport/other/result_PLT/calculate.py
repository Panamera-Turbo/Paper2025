# 数据
import numpy as np
tpe_mae = np.array([20.68, 18.29, 19.54, 14.20])
no_tpe_dwa_mae = np.array([20.71, 18.38, 19.64, 14.29])
tpe_rmse = np.array([33.33, 29.82, 32.85, 23.49])
no_tpe_dwa_rmse = np.array([33.56, 29.95, 33.08, 23.63])
tpe_mape = np.array([13.34, 12.49, 8.25, 9.28])
no_tpe_dwa_mape = np.array([13.42, 12.57, 8.31, 9.35])

# 计算MAE、RMSE、MAPE的平均改进百分比
mae_improvement = ((no_tpe_dwa_mae - tpe_mae) / no_tpe_dwa_mae) * 100
rmse_improvement = ((no_tpe_dwa_rmse - tpe_rmse) / no_tpe_dwa_rmse) * 100
mape_improvement = ((no_tpe_dwa_mape - tpe_mape) / no_tpe_dwa_mape) * 100

# 计算平均改进百分比
mae_avg_improvement = np.mean(mae_improvement)
rmse_avg_improvement = np.mean(rmse_improvement)
mape_avg_improvement = np.mean(mape_improvement)


print(f"{mae_avg_improvement, rmse_avg_improvement, mape_avg_improvement}")