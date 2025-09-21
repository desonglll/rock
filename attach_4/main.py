#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import torch
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
from jrc import compute_JRC
from connectivity_3d import process_all
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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def denoise_image(pil_img):
    img = np.array(pil_img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(img)

# -------------------------------
# 2. 辅助函数
# -------------------------------
def predict_mask(img_path):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size
    img = denoise_image(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    mask = 255 - mask
    return mask

def extract_crack_points(mask):
    edges = cv2.Canny(mask, 50, 150)
    pts = np.column_stack(np.where(edges > 0))
    return np.array([[x, y] for y, x in pts])

def depth_from_filename(filename):
    match = re.match(r'(\d+)-(\d+)m', filename)
    if match:
        start = int(match.group(1)) * 1000
        end = int(match.group(2)) * 1000
        return start, end
    else:
        return 0, 1000

# -------------------------------
# 3. 遍历所有钻孔与图像
# -------------------------------
CIRCUMFERENCE = 360
base_dir = './images'
crack_dict = {}

for hole_folder in sorted(os.listdir(base_dir)):
    hole_path = os.path.join(base_dir, hole_folder)
    if not os.path.isdir(hole_path):
        continue
    hole_id = hole_folder.split('#')[0]
    crack_dict[hole_id] = []

    for img_file in sorted(os.listdir(hole_path)):
        if not img_file.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(hole_path, img_file)
        depth_start, _ = depth_from_filename(img_file)

        mask = predict_mask(img_path)
        pts = extract_crack_points(mask)

        if pts.size == 0:
            x_mm = np.array([])
            y_mm = np.array([])
            jrc = np.nan
        else:
            img_width, img_height = mask.shape[1], mask.shape[0]
            px_per_mm = img_width / CIRCUMFERENCE
            x_mm = pts[:, 0] / px_per_mm
            y_mm = pts[:, 1] / px_per_mm + depth_start  # depth加在mm单位
            jrc = compute_JRC(x_mm, y_mm)

        crack_dict[hole_id].append({
            'id': img_file,
            'x_mm': x_mm,
            'y_mm': y_mm,
            'JRC': jrc
        })

print("所有裂隙已提取并计算JRC")

# -------------------------------
# 4. 多钻孔裂隙连通性分析
# -------------------------------
summary, picks = process_all(crack_dict)

# -------------------------------
# 5. 保存结果
# -------------------------------
records = []
for hole_id, cracks in crack_dict.items():
    for i, c in enumerate(cracks):
        records.append({
            '图像编号': c['id'],
            '钻孔编号': hole_id,
            '裂隙编号': i + 1,
            'JRC值': c['JRC']
        })

df = pd.DataFrame(records)
df.to_csv('p4_crack_summary.csv', index=False, encoding='utf-8-sig')

with open('p4_suggested_holes.txt', 'w', encoding='utf-8') as f:
    for p in picks:
        f.write(f"{p}\n")

print("表格已保存到 p4_crack_summary.csv")
print("补充钻孔建议已保存到 p4_suggested_holes.txt")
