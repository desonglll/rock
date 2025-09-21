# connectivity_3d.py
# 多钻孔裂隙连通性分析与补钻孔推荐

import numpy as np

def crack_distance(crack1, crack2):
    """
    计算两条裂隙的最小距离
    crack1, crack2: dict，包含 'x_mm' 和 'y_mm'
    """
    x1, y1 = crack1['x_mm'], crack1['y_mm']
    x2, y2 = crack2['x_mm'], crack2['y_mm']

    # 如果任意一条裂隙为空
    if len(x1) == 0 or len(x2) == 0:
        return np.inf  # 返回无穷大表示不可连通

    X1, X2 = np.meshgrid(x1, x2)
    Y1, Y2 = np.meshgrid(y1, y2)
    dist = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
    return np.min(dist)

def process_all(crack_dict, threshold=50):
    """
    进行多钻孔裂隙连通性分析
    输入:
        crack_dict: {hole_id: [crack1, crack2, ...]}
        threshold: 连通距离阈值 (mm)
    输出:
        summary: list, 连通分析结果
        suggested_holes: list, 推荐补钻孔位置 (x,y,z)
    """
    summary = []
    holes = sorted(crack_dict.keys())

    # 连通性矩阵
    connected_pairs = []

    for i, h1 in enumerate(holes):
        for j, h2 in enumerate(holes):
            if j <= i:
                continue
            cracks1 = crack_dict[h1]
            cracks2 = crack_dict[h2]
            for c1 in cracks1:
                for c2 in cracks2:
                    d = crack_distance(c1, c2)
                    if d < threshold:
                        connected_pairs.append({
                            'hole1': h1,
                            'hole2': h2,
                            'distance': d,
                            'crack1': c1['id'],
                            'crack2': c2['id'],
                            'crack1_coords': (np.mean(c1['x_mm']), np.mean(c1['y_mm'])),
                            'crack2_coords': (np.mean(c2['x_mm']), np.mean(c2['y_mm']))
                        })

    summary = connected_pairs

    # 根据不确定性区域计算补钻孔位置
    uncertain_pairs = []
    for pair in connected_pairs:
        if pair['distance'] > threshold * 0.7:  # 接近阈值的高不确定性
            c1_coords = pair['crack1_coords']  # 确保 crack 字典包含空间坐标
            c2_coords = pair['crack2_coords']
            center = tuple((np.array(c1_coords) + np.array(c2_coords)) / 2)
            uncertain_pairs.append((center, pair['distance']))

    # 按不确定性程度排序（distance 越大越不确定）
    uncertain_pairs.sort(key=lambda x: -x[1])

    # 选择前3个中心点作为补钻孔位置
    suggested_holes = [center for center, _ in uncertain_pairs[:3]]

    return summary, suggested_holes
