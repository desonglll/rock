import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms
from PIL import Image

# 输出掩码目录
mask_dir = './pred_masks'
os.makedirs(mask_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 去噪函数
def denoise_image(pil_img):
    img = np.array(pil_img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(img)


# 数据增强/归一化
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 1. 定义网络结构
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

# 2. 加载权重
checkpoint_path = "./checkpoints/unet_crack.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)


# 批量预测生成掩码
model.eval()
with torch.no_grad():
    for img_name in os.listdir('./images'):
        if img_name.startswith('.'):
            continue  # 忽略隐藏文件
        img_path = os.path.join('./images', img_name)
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size  # 原图宽高 (width, height)

        img = denoise_image(img)

        img_tensor = transform(img).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255

        # Resize 回原图大小
        pred_mask = cv2.resize(pred_mask, orig_size, interpolation=cv2.INTER_NEAREST)
        pred_mask = 255 - pred_mask
        cv2.imwrite(os.path.join(mask_dir, img_name), pred_mask)

print("全部预测掩码已保存到 pred_masks/ 文件夹")
