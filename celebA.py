from torchvision import datasets, transforms
import os

def download_celeba(root='./data'):
    """下载 CelebA 数据集"""
    try:
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 下载并加载 CelebA 数据集
        print("📥 开始下载 CelebA 数据集...")
        celeba_dataset = datasets.CelebA(
            root=root,
            split='all',
            download=True,
            transform=transform
        )
        print("✅ CelebA 数据集下载完成")
        return celeba_dataset
        
    except Exception as e:
        print(f"❌ 下载过程中出现错误: {e}")
        print("💡 请尝试以下解决方案:")
        print("1. 运行 'pip install gdown' 安装 gdown 库")
        print("2. 确保网络连接正常")
        print("3. 如果问题持续存在，可能需要手动下载数据集")
        return None

if __name__ == "__main__":
    dataset = download_celeba('./data')
    if dataset is not None:
        print(f"数据集大小: {len(dataset)} 张图像")