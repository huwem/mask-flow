# utils/mask_utils.py
import torch
import random

def random_rectangle_mask(H=256, W=256, max_size=0.4):
    mask = torch.ones(1, H, W)
    h, w = int(H * max_size), int(W * max_size)
    rh = random.randint(0, H - h)
    rw = random.randint(0, W - w)
    mask[0, rh:rh+h, rw:rw+w] = 0
    return mask

def inverse_rectangle_mask(H=256, W=256, min_visible_ratio=0.6, max_visible_ratio=0.7):
    """
    创建反向遮罩：遮住大部分图片，只留下单块区域可见
    
    Args:
        H, W: 图像高度和宽度
        min_visible_ratio: 最小可见区域比例
        max_visible_ratio: 最大可见区域比例
    """
    # 初始化全为0的遮罩（完全遮挡）
    mask = torch.zeros(1, H, W)
    
    # 计算可见区域大小
    visible_ratio = random.uniform(min_visible_ratio, max_visible_ratio)
    visible_h = int(H * visible_ratio)
    visible_w = int(W * visible_ratio)
    
    # 随机确定可见区域位置
    rh = random.randint(0, H - visible_h)
    rw = random.randint(0, W - visible_w)
    
    # 在遮罩中设置可见区域为1（不遮挡）
    mask[0, rh:rh+visible_h, rw:rw+visible_w] = 1
    
    return mask