import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
import os

# 正弦函数模型
def sine_model(x, R, P, beta, C):
    return R * np.sin(2 * np.pi * x / P + beta) + C

# 提取裂隙边缘点
def extract_edge_points(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 二值化（裂隙黑色）
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # 边缘检测
    edges = cv2.Canny(binary, 50, 150)
    # 获取边缘点坐标
    points = np.column_stack(np.where(edges > 0))
    # OpenCV: points[:,0]=y, points[:,1]=x -> 转换为 (x, y)
    points = np.array([[x, y] for y, x in points])
    return points

# 聚类分裂隙
def cluster_points(points, eps=5, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:  # 噪声
            continue
        cluster_pts = points[labels == label]
        clusters.append(cluster_pts)
    return clusters

# 拟合每条裂隙的正弦曲线
# 正弦拟合带边界
def fit_sine_to_cluster(cluster):
    x_data = cluster[:, 0]
    y_data = cluster[:, 1]
    R0 = max((y_data.max() - y_data.min()) / 2, 1e-3)
    P0 = max(x_data.max() - x_data.min(), 1e-3)
    beta0 = 0  # 默认 0 在 [-π, π] 内
    C0 = y_data.mean()

    try:
        popt, _ = curve_fit(
            sine_model,
            x_data,
            y_data,
            p0=[R0, P0, beta0, C0],
            bounds=([0, 1e-3, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf]),
            maxfev=5000
        )
        return popt
    except Exception as e:
        print(f"Warning: cluster拟合失败, {e}")
        return None
# 主函数
def process_images(image_dir, output_csv='table1.csv'):
    records = []
    for img_file in os.listdir(image_dir):
        if not img_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(image_dir, img_file)
        points = extract_edge_points(img_path)
        clusters = cluster_points(points)
        for i, cluster in enumerate(clusters):
            params = fit_sine_to_cluster(cluster)
            if params is None:
                continue
            R, P, beta, C = params

            # 文件名 -> 图像编号
            base = os.path.splitext(img_file)[0]  # attach_2_1
            parts = base.split('_')  # ["attach", "2", "1"]
            if len(parts) >= 3:
                image_name = f"图{parts[1]}-{parts[2]}.jpg"
            else:
                image_name = img_file  # 兜底，防止异常文件名

            records.append({
                '图像编号': image_name,
                '裂隙编号': i+1,
                '振幅R(mm)': R,
                '周期P(mm)': P,
                '相位β(rad)': beta,
                '中心线位置C(mm)': C
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"表1已保存到 {output_csv}")

# 使用示例
process_images('./images', output_csv='result.csv')
