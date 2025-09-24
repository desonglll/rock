import os

import cv2
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from utils.dataset import CrackDataset, RandomAugment, denoise_image

# 输出掩码目录
mask_dir = './pred_masks'
os.makedirs(mask_dir, exist_ok=True)


# 数据增强/归一化
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

augment = RandomAugment()
dataset = CrackDataset(
    images_dir='./images',
    masks_dir='./masks',
    transform=transform,
    augment=RandomAugment(),
    denoise=True  # 开启去噪
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(device)  # 把模型也放到 GPU

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice

criterion = nn.BCEWithLogitsLoss()  # 可单独用BCE
dice_loss = DiceLoss()

def combined_loss(outputs, masks):
    return 0.5 * criterion(outputs, masks) + 0.5 * dice_loss(outputs, masks)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss / len(dataloader):.4f}")

# 保存模型权重
os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), './checkpoints/unet_crack.pth')
print("模型权重已保存到 ./checkpoints/unet_crack.pth")

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
