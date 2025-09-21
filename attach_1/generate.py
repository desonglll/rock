import os
import json
import cv2
import numpy as np

# 输入 JSON 文件夹
json_dir = './labels'
# 输出二值掩码文件夹
mask_dir = './masks'
os.makedirs(mask_dir, exist_ok=True)

# 遍历所有 JSON 文件
for json_file in os.listdir(json_dir):
    if not json_file.endswith('.json'):
        continue

    json_path = os.path.join(json_dir, json_file)
    base_name = os.path.splitext(json_file)[0]
    out_mask = os.path.join(mask_dir, f'{base_name}.jpg')

    # 打开 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 创建空白掩码
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 遍历所有标注对象
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)  # 裂隙=255

    # 保存二值掩码
    cv2.imwrite(out_mask, mask)
    print(f"已保存 {out_mask}")

print("全部 JSON 文件已转换为二值掩码。")
