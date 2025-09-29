import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random

class SMAPDataset(Dataset):
    def __init__(self, root, img_size=512, target_size=8000):
        """
        初始化SMAP地图数据集
        
        Args:
            root (str): 数据集根目录路径
            img_size (int): 输出图像的尺寸（正方形）
            target_size (int): 目标数据集大小（用于数据增强）
        """
        self.root = root
        self.img_size = img_size
        self.target_size = target_size
        self.filenames = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 定义图像变换，与CelebADataset保持一致
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 定义数据增强变换
        self.augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=30),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        # 通过索引映射到原始图像，实现数据增强
        original_idx = idx % len(self.filenames)
        
        # 获取图片路径
        path = os.path.join(self.root, self.filenames[original_idx])
        img = Image.open(path).convert("RGB")
        
        # 首先移除右侧黑色区域
        img = self._remove_black_region(img)
        
        # 保持宽高比的情况下缩放图像到目标尺寸范围内
        # 使用thumbnail方法，与CelebADataset保持一致
        img.thumbnail((self.img_size, self.img_size), Image.LANCZOS)
        
        # 创建正方形画布并居中放置图像
        padded_img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
        w, h = img.size
        left = (self.img_size - w) // 2
        top = (self.img_size - h) // 2
        padded_img.paste(img, (left, top))
        
        # 应用数据增强（除了原始图像外）
        if idx >= len(self.filenames):
            # 先转换为RGBA以支持透明度处理
            padded_img = padded_img.convert("RGBA")
            # 应用增强
            padded_img = self.augmentation(padded_img)
            # 转换回RGB并保持白色背景
            white_bg = Image.new("RGB", padded_img.size, (255, 255, 255))
            white_bg.paste(padded_img, mask=padded_img.split()[-1] if len(padded_img.split()) == 4 else None)
            padded_img = white_bg
        
        # 应用变换
        img_tensor = self.transform(padded_img)  # [C, img_size, img_size]
        
        return img_tensor
    
    def _remove_black_region(self, img):
        """
        移除图片右侧的黑色标注区域
        
        Args:
            img (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 裁剪后的图像
        """
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 将图片转换为灰度图来检测黑色区域
        if len(img_array.shape) == 3:
            # RGB图片转灰度
            gray_img = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            # 已经是灰度图
            gray_img = img_array
            
        # 从右向左查找第一个非黑色像素列
        threshold = 30  # 黑色像素阈值
        
        # 初始化裁剪边界
        right_crop_boundary = img_array.shape[1]
        
        # 逐列从右向左检查
        for col in range(img_array.shape[1] - 1, -1, -1):
            # 检查该列是否大部分为黑色
            column = gray_img[:, col]
            black_pixels_ratio = np.sum(column < threshold) / len(column)
            
            # 如果黑色像素比例小于阈值，则认为找到了边界
            if black_pixels_ratio < 0.02:
                right_crop_boundary = col + 1  # 保留这一列
                break
        
        # 裁剪图片
        cropped_img = img.crop((0, 0, right_crop_boundary, img_array.shape[0]))
        return cropped_img

def preprocess_smap_images(input_folder, output_folder, target_size=8000):
    """
    预处理SMAP图像，通过数据增强扩充到指定数量并保存
    
    Args:
        input_folder (str): 输入图像文件夹路径
        output_folder (str): 输出图像文件夹路径
        target_size (int): 目标图像数量
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建数据集实例用于处理
    dataset = SMAPDataset(input_folder, target_size=target_size)
    
    print(f"开始预处理图像，目标数量: {target_size}...")
    
    # 获取原始文件名列表
    original_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i in range(target_size):
        try:
            # 获取处理后的图像张量
            img_tensor = dataset[i]
            
            # 反归一化: denormalize from [-1, 1] to [0, 1]
            img_tensor = (img_tensor + 1) / 2
            # 限制值范围在[0, 1]
            img_tensor = torch.clamp(img_tensor, 0, 1)
            # 转换为numpy数组并调整维度顺序
            img_array = img_tensor.permute(1, 2, 0).numpy()
            # 转换为PIL图像
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            
            # 生成文件名
            if i < len(original_filenames):
                # 原始图像
                filename = original_filenames[i]
            else:
                # 增强图像
                original_idx = i % len(original_filenames)
                name, ext = os.path.splitext(original_filenames[original_idx])
                filename = f"{name}_aug_{i//len(original_filenames)}.jpg"
            
            # 保存图像
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, "JPEG", quality=95)
            
            if (i + 1) % 500 == 0:
                print(f"已处理 {i + 1}/{target_size} 张图像")
                
        except Exception as e:
            print(f"处理第 {i} 张图像时出错: {e}")
    
    print(f"预处理完成！共处理 {target_size} 张图像，保存至 {output_folder}")

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

def test_dataset_visualization(dataset, num_images=10, save_path="./smap_visualization.png"):
    """
    测试并可视化数据集中的图片，保存到文件
    
    Args:
        dataset (SMAPDataset): 数据集实例
        num_images (int): 显示的图片数量
        save_path (str): 保存图片的路径
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 确保显示数量不超过数据集大小
    num_images = min(num_images, len(dataset))
    
    # 创建图形
    fig, axes = plt.subplots(num_images, 1, figsize=(5, 5*num_images))
    
    # 如果只有一行，调整axes的形状
    if num_images == 1:
        axes = [axes]
    
    print(f"显示数据集中的前{num_images}张图片:")
    
    for i in range(num_images):
        # 获取数据
        img_tensor = dataset[i]
        
        # 转换张量为图像
        img_pil = tensor_to_image(img_tensor)
        
        # 显示图像
        axes[i].imshow(img_pil)
        if i < len(dataset.filenames):
            axes[i].set_title(f'Original Image {i+1}: {dataset.filenames[i]}')
        else:
            original_idx = i % len(dataset.filenames)
            axes[i].set_title(f'Augmented Image {i+1}: {dataset.filenames[original_idx]} (aug)')
        axes[i].axis('off')
        
        print(f"  图片 {i+1}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {save_path}")

def test_preprocessing():
    """
    测试预处理功能
    """
    try:
        # 测试预处理
        input_folder = '../data/mapdata'
        output_folder = '../data/mapdata_processed'
        
        print("测试SMAP图像预处理...")
        preprocess_smap_images(input_folder, output_folder, target_size=8000)
        
        # 测试可视化
        print("生成预处理结果可视化...")
        dataset = SMAPDataset(input_folder, target_size=100)  # 小规模测试
        test_dataset_visualization(dataset, num_images=10, save_path="./smap_preprocessing_sample.png")
        
    except Exception as e:
        print(f"测试预处理时出错: {e}")

if __name__ == '__main__':
    test_preprocessing()