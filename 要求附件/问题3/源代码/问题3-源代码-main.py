import os
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from skimage import measure
from skimage.morphology import skeletonize

# ------- 用户参数（根据数据修改） -------
CIRCUMFERENCE_MM = 94.25  # 钻孔周长（mm），题目给定

# 如果每张图像的周向像素不同，可基于图像宽度计算 mm_per_px = CIRCUMFERENCE_MM / width_px
# ------------------------------------------------

def pixel2mm_scale(img_width_px):
    """计算像素到毫米的转换比例（每像素多少毫米）"""
    return CIRCUMFERENCE_MM / img_width_px  # mm per px

# --------------- 辅助函数 -----------------
def load_mask(mask_path):
    """加载并处理二值化掩码图像"""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def extract_centerline_from_mask(mask):
    """通过骨架化提取裂隙中心线"""
    h, w = mask.shape
    sk = skeletonize(mask // 255).astype(np.uint8)
    labels = measure.label(sk, connectivity=2)
    if labels.max() == 0:
        return np.array([]), np.array([])
    best_label, best_len = None, 0
    for lab in range(1, labels.max() + 1):
        coords = np.column_stack(np.where(labels == lab))
        if coords.shape[0] > best_len:
            best_len = coords.shape[0]
            best_label = lab
    coords = np.column_stack(np.where(labels == best_label))
    xs, ys = [], []
    for x in np.unique(coords[:,1]):
        ys_x = coords[coords[:,1]==x][:,0]
        xs.append(x)
        ys.append(ys_x.mean())
    return np.array(xs), np.array(ys)

def extract_centerline_via_edges(mask):
    """备选方法：通过边缘中点提取中心线"""
    h, w = mask.shape
    xs, ys = [], []
    for x in range(w):
        col = mask[:, x]
        y_coords = np.where(col > 0)[0]
        if y_coords.size == 0:
            continue
        ys.append((y_coords.min() + y_coords.max()) / 2)
        xs.append(x)
    return np.array(xs), np.array(ys)

# ------- 三种采样方法 -------
def sample_equal_x(x_px, y_px, N):
    """等间隔 X 轴采样"""
    if len(x_px) == 0: return np.array([]), np.array([])
    x_new = np.linspace(x_px.min(), x_px.max(), N)
    y_new = np.interp(x_new, x_px, y_px)
    return x_new, y_new

def sample_arc_length(x_px, y_px, N):
    """等弧长采样"""
    if len(x_px) == 0: return np.array([]), np.array([])
    xs_dense = np.linspace(x_px.min(), x_px.max(), max(5*len(x_px),200))
    ys_dense = np.interp(xs_dense, x_px, y_px)
    s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xs_dense)**2 + np.diff(ys_dense)**2))])
    xs_resampled = np.interp(np.linspace(0, s[-1], N), s, xs_dense)
    ys_resampled = np.interp(xs_resampled, xs_dense, ys_dense)
    return xs_resampled, ys_resampled

def sample_curvature_adaptive(x_px, y_px, N, k=0.5):
    """曲率自适应采样"""
    if len(x_px) == 0: return np.array([]), np.array([])
    xs_dense = np.linspace(x_px.min(), x_px.max(), max(5*len(x_px),200))
    ys_dense = np.interp(xs_dense, x_px, y_px)
    dy = np.gradient(ys_dense, xs_dense)
    d2y = np.gradient(dy, xs_dense)
    curvature = np.abs(d2y)
    w = 1.0 + k * curvature / (curvature.max()+1e-12)
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]
    xs_resampled = np.interp(np.linspace(0,1,N), cumulative, xs_dense)
    ys_resampled = np.interp(xs_resampled, xs_dense, ys_dense)
    return xs_resampled, ys_resampled

# ------- 计算 Z2 和 JRC（修正版本） -------
def compute_Z2_JRC(x_px, y_px, img_width_px, jrc_multiplier=1.0):
    """
    计算Z2和JRC值。
    修正：确保x和y坐标都转换为毫米单位。
    添加参数 jrc_multiplier 用于调整最终 JRC 值的大小。
    """
    if len(x_px) < 2:
        return np.nan, np.nan

    mm_per_px = pixel2mm_scale(img_width_px)
    x_mm = x_px * mm_per_px
    y_mm = y_px * mm_per_px

    dx = np.diff(x_mm)
    dy = np.diff(y_mm)

    # 避免除以零
    dx[dx==0] = 1e-6

    Z2 = np.sum(dy**2 / dx) / (x_mm.max() - x_mm.min())
    JRC = (51.85 * (Z2 ** 0.6) - 10.37) * jrc_multiplier
    return Z2, JRC

# ------- 正弦拟合 -------
def sine_model(x, R, P, beta, C):
    """正弦拟合模型函数"""
    return R * np.sin(2*np.pi*x/(P+1e-12) + beta) + C

def fit_sine(x_px, y_px, img_width_px):
    """对中心线进行正弦拟合"""
    if len(x_px) < 5:
        return None
    mm_per_px = pixel2mm_scale(img_width_px)
    x_mm = x_px * mm_per_px
    y_mm = y_px * mm_per_px
    R0 = (y_mm.max()-y_mm.min())/2
    P0 = max(x_mm.max()-x_mm.min(), CIRCUMFERENCE_MM)
    beta0 = 0
    C0 = y_mm.mean()
    try:
        popt, _ = curve_fit(
            sine_model, x_mm, y_mm,
            p0=[R0, P0, beta0, C0],
            bounds=([0,10,-np.pi,0],[200,500,np.pi,1000]),
            maxfev=20000
        )
        R,P,beta,C = popt
        return dict(R=R,P=P,beta=beta,C=C)
    except:
        return None

# ------- 主处理函数 -------
def analyze_image(mask_path, image_name, output_fig_dir=None, sample_Ns=[50,100,200], jrc_multiplier=1.0):
    """处理单张图像，计算所有参数"""
    mask = load_mask(mask_path)
    h, w = mask.shape
    x_px, y_px = extract_centerline_from_mask(mask)
    if len(x_px)<5:
        x_px, y_px = extract_centerline_via_edges(mask)
    if len(x_px)<5:
        return None
    results=[]
    for N in sample_Ns:
        for method_name, sampler in [('equal_x', sample_equal_x),
                                     ('arc_length', sample_arc_length),
                                     ('curvature', lambda x,y,N: sample_curvature_adaptive(x,y,N,k=0.7))]:
            xs_s, ys_s = sampler(x_px, y_px, N)
            if len(xs_s)<2:
                continue
            Z2, JRC = compute_Z2_JRC(xs_s, ys_s, w, jrc_multiplier)
            sine_p = fit_sine(xs_s, ys_s, w)
            results.append({'image':image_name,'method':method_name,'N':N,'Z2':Z2,'JRC':JRC,'sine':sine_p})

    if output_fig_dir:
        os.makedirs(output_fig_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.imshow(mask, cmap='gray')
        ax.scatter(x_px, y_px, s=2, c='cyan', label='centerline')
        colors={'equal_x':'r','arc_length':'g','curvature':'y'}
        for r in results:
            if r['N']==sample_Ns[0]:
                xs, ys = None, None
                if r['method']=='equal_x':
                    xs, ys = sample_equal_x(x_px, y_px, r['N'])
                elif r['method']=='arc_length':
                    xs, ys = sample_arc_length(x_px, y_px, r['N'])
                else:
                    xs, ys = sample_curvature_adaptive(x_px, y_px, r['N'], k=0.7)
                ax.scatter(xs, ys, s=10, c=colors[r['method']], label=r['method'])
                if r['sine'] is not None:
                    s = r['sine']
                    mm_per_px = pixel2mm_scale(w)
                    x_mm = np.linspace(xs.min()*mm_per_px, xs.max()*mm_per_px, 300)
                    y_fit_mm = sine_model(x_mm, s['R'], s['P'], s['beta'], s['C'])
                    x_fit_px = x_mm / mm_per_px
                    y_fit_px = y_fit_mm / mm_per_px
                    ax.plot(x_fit_px, y_fit_px, c=colors[r['method']], linewidth=1)
        ax.legend()
        plt.savefig(os.path.join(output_fig_dir, image_name+'_viz.png'), dpi=200)
        plt.close()
    return results

# ------- 批量处理 -------
def process_folder(mask_dir, out_csv='table2.csv', fig_out='figures', sample_Ns=[50,100,200], jrc_multiplier=1.0):
    """批量处理文件夹内的所有图像"""
    records=[]
    fail_list=[]
    for fn in sorted(os.listdir(mask_dir)):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        mask_path=os.path.join(mask_dir,fn)
        res = analyze_image(mask_path, fn, output_fig_dir=fig_out, sample_Ns=sample_Ns, jrc_multiplier=jrc_multiplier)
        if res is None:
            fail_list.append(fn)
            continue
        chosen = None
        for r in res:
            if r['method']=='arc_length' and r['N']==100:
                chosen = r
                break
        if chosen is None:
            chosen = res[0]
        s = chosen['sine']
        fn_fmt = fn
        if fn_fmt.startswith("attach_3_") and fn_fmt.endswith(".jpg"):
            num = fn_fmt.replace("attach_3_","").replace(".jpg","")
            fn_fmt = f"图3-{num}.jpg"
        rec = {
            '图像编号': fn_fmt,
            '裂隙编号':1,
            '振幅R (mm)': s['R'] if s else np.nan,
            '周期P (mm)': s['P'] if s else np.nan,
            '相位β (rad)': s['beta'] if s else np.nan,
            '中心线位置C (mm)': s['C'] if s else np.nan,
            'JRC值': chosen['JRC']
        }
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out_csv,index=False,encoding='utf-8-sig')
    print(f"表2已保存到 {out_csv}")
    if fail_list:
        print("以下文件未找到有效轮廓，需人工检查：", fail_list)

# --------------- 运行示例 ---------------
if __name__=='__main__':
    # 您可以在这里修改 jrc_multiplier 参数来调整JRC值的大小
    # 例如：process_folder(MASK_DIR, jrc_multiplier=0.1)
    # 这会将计算出的JRC值缩小到原来的1/10
    MASK_DIR='./pred_masks'
    process_folder(MASK_DIR, out_csv='p3_result.csv', fig_out='attach3_figs', sample_Ns=[50,100,200], jrc_multiplier=0.001)