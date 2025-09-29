# utils/mask_utils.py
import torch
import random

def inverse_rectangle_mask(H, W, min_visible_ratio=0.15, max_visible_ratio=0.4):
    """
    创建反向遮罩：遮住大部分图片，只在中间四分之一区域留下单块可见区域
    
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
    
    # 定义中间四分之一区域的边界
    # 中心区域为图像中心的四分之一
    center_h_start = H // 4
    center_h_end = 3 * H // 4
    center_w_start = W // 4
    center_w_end = 3 * W // 4
    
    # 确保可见区域不会超出中心区域
    visible_h = min(visible_h, center_h_end - center_h_start)
    visible_w = min(visible_w, center_w_end - center_w_start)
    
    # 在中心四分之一区域内随机确定可见区域位置
    rh = random.randint(center_h_start, center_h_end - visible_h)
    rw = random.randint(center_w_start, center_w_end - visible_w)
    
    # 在遮罩中设置可见区域为1（不遮挡）
    mask[0, rh:rh+visible_h, rw:rw+visible_w] = 1
    
    return mask