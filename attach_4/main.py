#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import torch
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
from connectivity_3d import process_all  # 问题4分析函数
import pandas as pd
from torchvision import transforms

# -------------------------------
# 1. UNet模型加载
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load('./checkpoints/unet_crack.pth', map_location=device))
model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def denoise_image(pil_img):
    img = np.array(pil_img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(img)

# -------------------------------
# 2. JRC计算函数（修正版）
# -------------------------------
def compute_JRC(x_mm, y_mm):
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    N = len(x_mm)
    if N < 2:
        return np.nan
    # 等间距差分法，增加1e-8避免除零
    dy_dx_sq = ((y_mm[1:] - y_mm[:-1]) / (x_mm[1:] - x_mm[:-1] + 1e-8))**2
    Z2 = np.sqrt(np.mean(dy_dx_sq))
    # 经验公式
    JRC = 51.85 * (Z2 ** 0.6) - 10.37
    # 保证非负
    if JRC < 0 or np.isnan(JRC) or np.isinf(JRC):
        return np.nan
    return JRC

# -------------------------------
# 3. 辅助函数
# -------------------------------
def predict_mask(img_path):
    """使用 UNet 预测裂隙掩码"""
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (width, height)
    img = denoise_image(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    mask = 255 - mask  # 裂隙黑色
    return mask

def extract_crack_points(mask):
    """从二值掩码提取裂隙中心线点"""
    edges = cv2.Canny(mask, 50, 150)
    pts = np.column_stack(np.where(edges > 0))
    return np.array([[x, y] for y, x in pts])

def depth_from_filename(filename):
    """从 0-1m.jpg 解析深度起点和终点 (mm)"""
    match = re.match(r'(\d+)-(\d+)m', filename)
    if match:
        start = int(match.group(1)) * 1000
        end = int(match.group(2)) * 1000
        return start, end
    else:
        return 0, 1000  # 默认1米

# -------------------------------
# 4. 遍历所有钻孔与图像
# -------------------------------
CIRCUMFERENCE = 360  # 钻孔周长 mm
base_dir = './images'
crack_dict = {}

for hole_folder in sorted(os.listdir(base_dir)):
    hole_path = os.path.join(base_dir, hole_folder)
    if not os.path.isdir(hole_path):
        continue
    hole_id = hole_folder.split('#')[0]  # 1#hole -> 1
    crack_dict[hole_id] = []

    for img_file in sorted(os.listdir(hole_path)):
        if not img_file.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(hole_path, img_file)
        depth_start, _ = depth_from_filename(img_file)

        mask = predict_mask(img_path)
        pts = extract_crack_points(mask)

        # 空裂隙处理
        if pts.size == 0:
            x_mm = np.array([])
            y_mm = np.array([])
            jrc = np.nan
        else:
            img_width, img_height = mask.shape[1], mask.shape[0]
            px_per_mm = img_width / CIRCUMFERENCE  # px->mm
            x_mm = pts[:, 0] / px_per_mm
            y_mm = (pts[:, 1] + depth_start) / px_per_mm
            jrc = compute_JRC(x_mm, y_mm) if len(x_mm) >= 2 else np.nan

        crack_dict[hole_id].append({
            'id': img_file,
            'x_mm': x_mm,
            'y_mm': y_mm,
            'JRC': jrc
        })

print("所有裂隙已提取并计算JRC")

# -------------------------------
# 5. 多钻孔裂隙连通性分析
# -------------------------------
summary, picks = process_all(crack_dict)

# -------------------------------
# 6. 保存结果
# -------------------------------
records = []
for hole_id, cracks in crack_dict.items():
    for i, c in enumerate(cracks):
        rec = {
            '图像编号': c['id'],
            '钻孔编号': hole_id,
            '裂隙编号': i + 1,
            'JRC值': c['JRC']
        }
        records.append(rec)

df = pd.DataFrame(records)
df.to_csv('p4_crack_summary.csv', index=False, encoding='utf-8-sig')

# 输出补充钻孔建议
with open('p4_suggested_holes.txt', 'w', encoding='utf-8') as f:
    for p in picks:
        f.write(f"{p}\n")

print("表格已保存到 p4_crack_summary.csv")
print("补充钻孔建议已保存到 p4_suggested_holes.txt")
