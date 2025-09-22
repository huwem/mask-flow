# datasets/celeba_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from utils.mask_utils import inverse_rectangle_mask  # 更改导入的函数


class CelebADataset(Dataset):
    def __init__(self, root, img_size=256):
        self.root = root
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.pad = T.Pad(padding=0, fill=1)  # 默认先不填充，后面动态设置

    def __len__(self):
        return len(self.filenames)

# ...existing code...

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)  # [C, H, W]

        _, h, w = img.shape
        pad_h = max(0, self.img_size - h)
        pad_w = max(0, self.img_size - w)
        # 统一填充到目标尺寸
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        pad_transform = T.Pad(padding=padding, fill=1)
        img = pad_transform(img)
        # 强制裁剪到目标尺寸，保证所有图片都是 [C, img_size, img_size]
        img = img[:, :self.img_size, :self.img_size]

        mask = inverse_rectangle_mask(self.img_size, self.img_size)
        white = torch.ones_like(img)
        masked_img = img * mask + white * (1 - mask)

        return masked_img, mask, img
# ...existing code...