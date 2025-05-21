import numpy as np

# 仅DWA 和 无TPE和DWA (STID)的数据
dwa_mae = np.array([20.30, 18.13, 19.21, 13.61])
no_tpe_dwa_mae = np.array([20.71, 18.38, 19.64, 14.29])
dwa_rmse = np.array([32.92, 29.43, 31.52, 22.52])
no_tpe_dwa_rmse = np.array([33.56, 29.95, 33.08, 23.63])
dwa_mape = np.array([12.97, 12.12, 7.92, 8.91])
no_tpe_dwa_mape = np.array([13.42, 12.57, 8.31, 9.35])

# 计算改进百分比
mae_improvement = ((no_tpe_dwa_mae - dwa_mae) / no_tpe_dwa_mae) * 100
rmse_improvement = ((no_tpe_dwa_rmse - dwa_rmse) / no_tpe_dwa_rmse) * 100
mape_improvement = ((no_tpe_dwa_mape - dwa_mape) / no_tpe_dwa_mape) * 100

# 计算平均改进百分比
mae_avg_improvement = np.mean(mae_improvement)
rmse_avg_improvement = np.mean(rmse_improvement)
mape_avg_improvement = np.mean(mape_improvement)


print(f"{(mae_avg_improvement, rmse_avg_improvement, mape_avg_improvement)}")
