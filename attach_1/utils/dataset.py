import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# 去噪函数
def denoise_image(pil_img):
    img = np.array(pil_img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(img)

# 数据增强
class RandomAugment:
    def __call__(self, image, mask):
        # 随机水平翻转
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        # 随机垂直翻转
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        # 随机旋转 ±10°
        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        # 对比度抖动
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        return image, mask

# 数据集
class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, augment=None, denoise=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        self.denoise = denoise
        self.images = sorted(f for f in os.listdir(images_dir) if not f.startswith('.'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 去噪
        if self.denoise:
            image = denoise_image(image)

        # 数据增强
        if self.augment:
            image, mask = self.augment(image, mask)

        # transform -> ToTensor, Resize 等
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
