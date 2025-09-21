import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib import rcParams


images_dir = './images'
masks_dir = './pred_masks'
output_dir = './paper_figures'
os.makedirs(output_dir, exist_ok=True)

rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS可用 'Arial Unicode MS'
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 选择三张图
selected_imgs = ['attach_1_01.jpg', 'attach_1_02.jpg', 'attach_1_03.jpg']  # 对应图1-1, 图1-2, 图1-3

for img_name in selected_imgs:
    orig = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
    mask = Image.open(os.path.join(masks_dir, img_name)).convert('L')

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(orig)
    axes[0].set_title('原图')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('识别结果')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, img_name))
    plt.close()
