# datasets_smap.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 修改为绝对导入:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mask_utils import inverse_rectangle_mask

class SMAPDataset(Dataset):
    def __init__(self, root, img_size=512):
        """
        初始化SMAP地图数据集
        
        Args:
            root (str): 数据集根目录路径
            img_size (int): 输出图像的尺寸（正方形）
        """
        self.root = root
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 定义图像变换，与CelebADataset保持一致
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 获取图片路径
        path = os.path.join(self.root, self.filenames[idx])
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
        
        # 应用变换
        img_tensor = self.transform(padded_img)  # [C, img_size, img_size]
        
        # 创建掩码，与CelebADataset保持一致
        mask = inverse_rectangle_mask(self.img_size, self.img_size)
        white = torch.ones_like(img_tensor)
        masked_img = img_tensor * mask + white * (1 - mask)
        
        return masked_img, mask, img_tensor
    
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

def test_dataset_visualization(dataset, num_images=10, save_path="./dataset_visualization.png"):
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
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    # 如果只有一行，调整axes的形状
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    print(f"显示数据集中的前{num_images}张图片:")
    
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
    print(f"可视化结果已保存到: {save_path}")

# 保留原有的处理函数，用于预处理图片
def crop_image_remove_black_region(image_path, output_path=None):
    """
    裁剪图片，移除右侧黑色标注区域，保留左侧地图部分
    
    Args:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径，如果为None则覆盖原图
    
    Returns:
        tuple: (原始宽度, 裁剪后宽度)
    """
    # 打开图片
    with Image.open(image_path) as img:
        # 转换为numpy数组以便处理
        img_array = np.array(img)
        
        # 将图片转换为灰度图来检测黑色区域
        if len(img_array.shape) == 3:
            # RGB图片转灰度
            gray_img = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            # 已经是灰度图
            gray_img = img_array
            
        # 从右向左查找第一个非黑色像素列
        # 黑色像素值接近0，我们设定一个阈值来判断是否为黑色
        threshold = 30  # 阈值，可根据实际情况调整
        
        # 初始化裁剪边界
        right_crop_boundary = img_array.shape[1]
        
        # 逐列从右向左检查
        for col in range(img_array.shape[1] - 1, -1, -1):
            # 检查该列是否大部分为黑色
            column = gray_img[:, col]
            black_pixels_ratio = np.sum(column < threshold) / len(column)
            
            # 如果黑色像素比例小于某个阈值，则认为找到了边界
            if black_pixels_ratio < 0.02:
                right_crop_boundary = col + 1  # 保留这一列
                break
        
        # 裁剪图片
        cropped_img = img.crop((0, 0, right_crop_boundary, img_array.shape[0]))
        
        # 保存裁剪后的图片
        if output_path is None:
            output_path = image_path
        cropped_img.save(output_path)
        
        return img_array.shape[1], right_crop_boundary

def process_folder_images(folder_path, output_folder=None):
    """
    处理文件夹中的所有图片，保持宽高比
    
    Args:
        folder_path (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径，如果为None则覆盖原图
    """
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 如果指定了输出文件夹，则创建它
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(folder_path, filename)
            
            # 确定输出路径
            if output_folder:
                output_path = os.path.join(output_folder, filename)
            else:
                output_path = None  # 覆盖原图
            
            try:
                original_width, cropped_width = crop_image_remove_black_region(input_path, output_path)
                print(f"{filename}: {original_width} -> {cropped_width} (裁剪掉了 {original_width - cropped_width} 像素)")
                processed_count += 1
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
    
    print(f"总共处理了 {processed_count} 张图片")

def main():
    input_folder = '../data/mapdata'
    output_folder = '../data/mapdata_cropped'  # 可选：指定输出文件夹
    
    print("开始处理图片，移除右侧黑色标注区域...")
    process_folder_images(input_folder, output_folder)
    print("处理完成！")



def test_dataset():
    """
    测试数据集加载和可视化功能
    """
    try:
        # 创建数据集实例
        dataset = SMAPDataset(root='../data/mapdata_cropped', img_size=512)
        print(f"数据集加载成功，共包含 {len(dataset)} 张图片")
        
        # 显示前十张图片
        test_dataset_visualization(dataset, num_images=5)
        
    except Exception as e:
        print(f"测试数据集时出错: {e}")

if __name__ == '__main__':
    # 如果需要测试数据集，可以取消下面的注释
    test_dataset()
    #main()