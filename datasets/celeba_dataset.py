import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
# 修改为绝对导入:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mask_utils import inverse_rectangle_mask

class CelebADataset(Dataset):
    def __init__(self, root, img_size=512):
        self.root = root
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        # 先缩放原图，保持比例，最长边为512
        img.thumbnail((self.img_size, self.img_size), Image.LANCZOS)
        padded_img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
        w, h = img.size
        left = (self.img_size - w) // 2
        top = (self.img_size - h) // 2
        padded_img.paste(img, (left, top))
        img = self.transform(padded_img)  # [C, 512, 512]

        # 在填充后的图片加上mask
        mask = inverse_rectangle_mask(self.img_size, self.img_size)
        white = torch.ones_like(img)
        masked_img = img * mask + white * (1 - mask)

        return masked_img, mask, img

# 在 celeba_dataset.py 文件末尾添加以下代码

import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(tensor):
    """
    将归一化的张量转换回图像格式
    
    Args:
        tensor (torch.Tensor): 归一化的图像张量，形状为[C, H, W]
        
    Returns:
        PIL.Image: 转换后的图像
    """
    # 反归一化: denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # 限制值范围在[0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # 转换为numpy数组并调整维度顺序
    img_array = tensor.permute(1, 2, 0).numpy()
    # 转换为PIL图像
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    return img

def test_celeba_visualization(dataset, num_images=10, save_path="./celeba_visualization.png"):
    """
    测试并可视化CelebA数据集中的图片，保存到文件
    
    Args:
        dataset (CelebADataset): CelebA数据集实例
        num_images (int): 显示的图片数量
        save_path (str): 保存图片的路径
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 确保显示数量不超过数据集大小
    num_images = min(num_images, len(dataset))
    
    # 创建图形
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    # 如果只有一行，调整axes的形状
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    print(f"显示CelebA数据集中的前{num_images}张图片:")
    
    for i in range(num_images):
        # 获取数据
        masked_img, mask, clean_img = dataset[i]
        
        # 转换张量为图像
        masked_pil = tensor_to_image(masked_img)
        clean_pil = tensor_to_image(clean_img)
        mask_pil = Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8), mode='L')
        
        # 显示原始图像
        axes[i, 0].imshow(clean_pil)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # 显示掩码图像
        axes[i, 1].imshow(masked_pil)
        axes[i, 1].set_title(f'Masked Image {i+1}')
        axes[i, 1].axis('off')
        
        # 显示掩码
        axes[i, 2].imshow(mask_pil, cmap='gray')
        axes[i, 2].set_title(f'Mask {i+1}')
        axes[i, 2].axis('off')
        
        print(f"  图片 {i+1}: {dataset.filenames[i]}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CelebA可视化结果已保存到: {save_path}")

def test_celeba_dataset():
    """
    测试CelebA数据集加载和可视化功能
    """
    try:
        # 创建数据集实例
        dataset = CelebADataset(root='../data/celeba/img_align_celeba', img_size=512)  # 根据实际路径调整
        print(f"CelebA数据集加载成功，共包含 {len(dataset)} 张图片")
        
        # 显示前几张图片
        test_celeba_visualization(dataset, num_images=5, save_path="./celeba_dataset_sample.png")
        
    except Exception as e:
        print(f"测试CelebA数据集时出错: {e}")

if __name__ == '__main__':
    # 如果需要测试数据集，可以取消下面的注释
    test_celeba_dataset()
    pass