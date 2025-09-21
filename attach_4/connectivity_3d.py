import numpy as np

def crack_distance(crack1, crack2):
    x1, y1 = crack1['x_mm'], crack1['y_mm']
    x2, y2 = crack2['x_mm'], crack2['y_mm']
    if len(x1) == 0 or len(x2) == 0:
        return np.inf
    X1, X2 = np.meshgrid(x1, x2)
    Y1, Y2 = np.meshgrid(y1, y2)
    dist = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
    return np.min(dist)

def process_all(crack_dict, threshold=50):
    summary = []
    holes = sorted(crack_dict.keys())
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

    uncertain_pairs = []
    for pair in connected_pairs:
        if pair['distance'] > threshold * 0.7:
            c1_coords = pair['crack1_coords']
            c2_coords = pair['crack2_coords']
            center = tuple((np.array(c1_coords) + np.array(c2_coords)) / 2)
            uncertain_pairs.append((center, pair['distance']))

    uncertain_pairs.sort(key=lambda x: -x[1])
    suggested_holes = [center for center, _ in uncertain_pairs[:3]]

    return summary, suggested_holes
