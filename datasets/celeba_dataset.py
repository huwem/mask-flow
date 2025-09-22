import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
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