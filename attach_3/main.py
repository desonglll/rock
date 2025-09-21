# file: jrc_analysis.py
import os

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from skimage import measure
from skimage.morphology import skeletonize

# ------- 用户参数（根据数据修改） -------
CIRCUMFERENCE_MM = 94.25  # 钻孔周长（mm），题目给定


# 如果每张图像的周向像素不同，可基于图像宽度计算 px_per_mm = width_px / CIRCUMFERENCE_MM
# ------------------------------------------------

def pixel2mm_scale(img_width_px):
    return img_width_px / CIRCUMFERENCE_MM  # px per mm


# --------------- 辅助函数 -----------------
def load_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    # 确保二值 0/255
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def extract_centerline_from_mask(mask):
    """
    从二值裂隙掩码提取每个 x(像素) 的中心 y(像素) 值（如果有多段，则返回最长连通段）。
    返回 (x_px, y_px) numpy arrays.
    """
    h, w = mask.shape
    # skeleton 保留细线
    sk = skeletonize(mask // 255).astype(np.uint8)  # 0/1
    # 找连通组件，选择最长的一段
    labels = measure.label(sk, connectivity=2)
    if labels.max() == 0:
        return np.array([]), np.array([])
    best_label = None
    best_len = 0
    for lab in range(1, labels.max() + 1):
        coords = np.column_stack(np.where(labels == lab))  # (row(y), col(x))
        if coords.shape[0] > best_len:
            best_len = coords.shape[0]
            best_label = lab
    coords = np.column_stack(np.where(labels == best_label))
    # coords[:,0]=y, coords[:,1]=x
    # For each unique x, take mean y (centerline)
    xs = np.unique(coords[:, 1])
    x_list, y_list = [], []
    for x in xs:
        ys = coords[coords[:, 1] == x][:, 0]
        y_list.append(ys.mean())
        x_list.append(x)
    return np.array(x_list), np.array(y_list)


# 若裂隙宽度很大，也可用 top/bottom edges -> choose center = (y_top+y_bottom)/2
def extract_centerline_via_edges(mask):
    # edges per column: take topmost and bottommost crack pixel, mean as center
    h, w = mask.shape
    x_list, y_list = [], []
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if ys.size == 0:
            continue
        y_center = (ys.min() + ys.max()) / 2.0
        x_list.append(x)
        y_list.append(y_center)
    return np.array(x_list), np.array(y_list)


# ------- 三种采样方法 -------
def sample_equal_x(x_px, y_px, N):
    """等间距 x 采样：在 x 范围均匀采 N 点（或使用已有点插值）"""
    if len(x_px) == 0: return np.array([]), np.array([])
    x_min, x_max = x_px.min(), x_px.max()
    x_new = np.linspace(x_min, x_max, N)
    y_new = np.interp(x_new, x_px, y_px)
    return x_new, y_new


def sample_arc_length(x_px, y_px, N):
    """按弧长等距采样（先插值到高分辨率，再按累积弧长重采样）"""
    if len(x_px) == 0: return np.array([]), np.array([])
    # 插值到密样本
    t = x_px
    y = y_px
    xs_dense = np.linspace(t.min(), t.max(), max(5 * len(t), 200))
    ys_dense = np.interp(xs_dense, t, y)
    # 计算累积弧长（像素）
    seg = np.sqrt(np.diff(xs_dense) ** 2 + np.diff(ys_dense) ** 2)
    s = np.concatenate([[0], np.cumsum(seg)])
    s_total = s[-1]
    s_target = np.linspace(0, s_total, N)
    xs_resampled = np.interp(s_target, s, xs_dense)
    ys_resampled = np.interp(xs_resampled, xs_dense, ys_dense)
    return xs_resampled, ys_resampled


def sample_curvature_adaptive(x_px, y_px, N, k=0.5):
    """
    曲率自适应采样：在高曲率区域增加采样密度（简易实现）。
    k 控制曲率权重（0~1）。
    """
    if len(x_px) == 0: return np.array([]), np.array([])
    xs_dense = np.linspace(x_px.min(), x_px.max(), max(5 * len(x_px), 200))
    ys_dense = np.interp(xs_dense, x_px, y_px)
    # 估算二阶导近似曲率 proxy = abs(d2y/dx2)
    dy = np.gradient(ys_dense, xs_dense)
    d2y = np.gradient(dy, xs_dense)
    curvature = np.abs(d2y)
    # weight = 1 + k * (normalized curvature)
    w = 1.0 + k * (curvature / (curvature.max() + 1e-12))
    cumulative = np.cumsum(w)
    cumulative = cumulative / cumulative[-1]
    target = np.linspace(0, 1, N)
    xs_resampled = np.interp(target, cumulative, xs_dense)
    ys_resampled = np.interp(xs_resampled, xs_dense, ys_dense)
    return xs_resampled, ys_resampled


# ------- 计算 Z2 和 JRC -------
def compute_Z2_JRC(x_px, y_px, img_width_px):
    """
    输入：像素坐标 arrays x_px, y_px (对应的轮廓离散点，按 x 升序)
    img_width_px 用于 px->mm mapping: px_per_mm = img_width_px / CIRCUMFERENCE_MM
    返回 Z2 (无单位) 和 JRC (经验公式)
    """
    if len(x_px) < 2:
        return np.nan, np.nan
    px_per_mm = pixel2mm_scale(img_width_px)  # px/mm
    # 把 x,y 转 mm（注意：y 方向单位同样视为 mm）
    x_mm = x_px / px_per_mm
    y_mm = y_px / px_per_mm
    # 使用等间距离散公式的近似（式(3)）
    diffs = np.diff(y_mm) / np.diff(x_mm)  # dy/dx per segment
    # 公式使用 sqrt(1/N * sum(diff^2))
    Z2 = np.sqrt(np.mean(diffs ** 2))
    # 经验公式 (2)
    JRC = 51.85 * (Z2 ** 0.6) - 10.37
    return Z2, JRC


# ------- 可选：拟合正弦（输出 R,P,beta,C） -------
def sine_model(x, R, P, beta, C):
    return R * np.sin(2 * np.pi * x / (P + 1e-12) + beta) + C


def fit_sine(x_px, y_px, img_width_px):
    """
    拟合正弦，返回像素单位参数转换为 mm。若拟合失败返回 None。
    """
    if len(x_px) < 5:
        return None
    # 先把 x,y 转 mm
    px_per_mm = pixel2mm_scale(img_width_px)
    x_mm = x_px / px_per_mm
    y_mm = y_px / px_per_mm
    R0 = (y_mm.max() - y_mm.min()) / 2
    P0 = max(x_mm.max() - x_mm.min(), CIRCUMFERENCE_MM)  # 初始周期可取孔周长
    beta0 = 0
    C0 = y_mm.mean()
    try:
        popt, _ = curve_fit(
            sine_model, x_mm, y_mm,
            p0=[R0, P0, beta0, C0],
            bounds=([0, 10, -np.pi, 0], [200, 500, np.pi, 1000]),  # 合理上下限
            maxfev=20000
        )
        R, P, beta, C = popt
        return dict(R=R, P=P, beta=beta, C=C)
    except Exception as e:
        # 拟合失败
        return None


# ------- 主处理函数（每张图） -------
def analyze_image(mask_path, image_name, output_fig_dir=None, sample_Ns=[50, 100, 200]):
    """
    返回字典：包含每种采样策略和 N 下的 Z2/JRC，以及拟合参数（若可得）
    """
    mask = load_mask(mask_path)
    h, w = mask.shape
    # 尝试通过 skeleton 或 edges 提取中心线（优先 skeleton）
    x_px, y_px = extract_centerline_from_mask(mask)
    if len(x_px) < 5:
        x_px, y_px = extract_centerline_via_edges(mask)
    if len(x_px) < 5:
        return None  # 无有效轮廓

    results = []
    for N in sample_Ns:
        # 三种采样
        for method_name, sampler in [('equal_x', sample_equal_x),
                                     ('arc_length', sample_arc_length),
                                     ('curvature', lambda x, y, N: sample_curvature_adaptive(x, y, N, k=0.7))]:
            xs_s, ys_s = sampler(x_px, y_px, N)
            if len(xs_s) < 2:
                continue
            Z2, JRC = compute_Z2_JRC(xs_s, ys_s, w)
            sine_p = fit_sine(xs_s, ys_s, w)
            results.append({
                'image': image_name,
                'method': method_name,
                'N': N,
                'Z2': Z2,
                'JRC': JRC,
                'sine': sine_p
            })

    # optional: 保存示意图（原图+采样点+拟合曲线）
    if output_fig_dir:
        os.makedirs(output_fig_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.imshow(mask, cmap='gray')
        ax.set_title(image_name)
        # plot original centerline pts
        ax.scatter(x_px, y_px, s=2, c='cyan', label='centerline')
        # show first sampling of each method for visualization
        colors = {'equal_x': 'r', 'arc_length': 'g', 'curvature': 'y'}
        for r in results:
            if r['N'] == sample_Ns[0]:  # show only first N to avoid clutter
                xs, ys = None, None
                if r['method'] == 'equal_x':
                    xs, ys = sample_equal_x(x_px, y_px, r['N'])
                elif r['method'] == 'arc_length':
                    xs, ys = sample_arc_length(x_px, y_px, r['N'])
                else:
                    xs, ys = sample_curvature_adaptive(x_px, y_px, r['N'], k=0.7)
                ax.scatter(xs, ys, s=10, c=colors[r['method']], label=r['method'])
                # 如果有拟合则画拟合线
                if r['sine'] is not None:
                    s = r['sine']
                    px_per_mm = pixel2mm_scale(w)
                    x_mm = np.linspace(xs.min() / px_per_mm, xs.max() / px_per_mm, 300)
                    y_fit_mm = sine_model(x_mm, s['R'], s['P'], s['beta'], s['C'])
                    x_fit_px = x_mm * px_per_mm
                    y_fit_px = y_fit_mm * px_per_mm
                    ax.plot(x_fit_px, y_fit_px, c=colors[r['method']], linewidth=1)
        ax.legend()
        plt.savefig(os.path.join(output_fig_dir, image_name + '_viz.png'), dpi=200)
        plt.close()

    return results


# ------- 批量处理文件夹并输出表2 -------
def process_folder(mask_dir, out_csv='table2.csv', fig_out='figures', sample_Ns=[50, 100, 200]):
    records = []
    fail_list = []
    for fn in sorted(os.listdir(mask_dir)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        mask_path = os.path.join(mask_dir, fn)
        res = analyze_image(mask_path, fn, output_fig_dir=fig_out, sample_Ns=sample_Ns)
        if res is None:
            fail_list.append(fn)
            continue
        # for table2 we choose one representative method (e.g. arc_length, N=100) or record multiple rows
        # 这里把每张图的 arc_length N=100 作为代表并写入表2；若无则选其他可用记录
        chosen = None
        for r in res:
            if r['method'] == 'arc_length' and r['N'] == 100:
                chosen = r
                break
        if chosen is None:
            chosen = res[0]
        s = chosen['sine']
        fn_fmt = fn
        if fn_fmt.startswith("attach_3_") and fn_fmt.endswith(".jpg"):
            num = fn_fmt.replace("attach_3_", "").replace(".jpg", "")
            fn_fmt = f"图3-{num}.jpg"

        rec = {
            '图像编号': fn_fmt,
            '裂隙编号': 1,
            '振幅R (mm)': s['R'] if s else np.nan,
            '周期P (mm)': s['P'] if s else np.nan,
            '相位β (rad)': s['beta'] if s else np.nan,
            '中心线位置C (mm)': s['C'] if s else np.nan,
            'JRC值': chosen['JRC']
        }
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"表2已保存到 {out_csv}")
    if fail_list:
        print("以下文件未找到有效轮廓，需人工检查：", fail_list)


# --------------- 运行示例 ---------------
if __name__ == '__main__':
    MASK_DIR = './pred_masks'  # 把你的附件3掩码放这里
    process_folder(MASK_DIR, out_csv='p3_result.csv', fig_out='attach3_figs', sample_Ns=[50, 100, 200])
