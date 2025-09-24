import numpy as np

def compute_JRC(x_mm, y_mm):
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    N = len(x_mm)
    if N < 2:
        return np.nan
    dy_dx_sq = ((y_mm[1:] - y_mm[:-1]) / (x_mm[1:] - x_mm[:-1] + 1e-8))**2
    Z2 = np.sqrt(np.mean(dy_dx_sq))
    JRC = 51.85 * Z2**0.6 - 10.37
    if JRC < 0 or np.isnan(JRC) or np.isinf(JRC):
        return np.nan
    return JRC
