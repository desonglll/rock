# jrc.py
# 计算裂隙轮廓线 JRC 值

import numpy as np

def compute_JRC(x_mm, y_mm):
    """
    计算裂隙轮廓线的JRC值
    输入:
        x_mm: np.array, 水平坐标 (mm)
        y_mm: np.array, 垂直坐标 (mm)
    输出:
        JRC值 (float)
    """
    x_mm = np.asarray(x_mm)
    y_mm = np.asarray(y_mm)
    N = len(x_mm)
    if N < 2:
        return np.nan

    # 等间距差分法近似 Z2
    dy_dx_sq = ((y_mm[1:] - y_mm[:-1]) / (x_mm[1:] - x_mm[:-1] + 1e-8))**2
    Z2 = np.sqrt(np.mean(dy_dx_sq))

    # 巴顿经验公式计算JRC
    JRC = 51.85 * Z2**0.6 - 10.37
    return JRC
