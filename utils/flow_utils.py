import torch
import torch.nn.functional as F

import torch
import numpy as np

def linear_growth_mask(mask, t, img_size=512):
    """
    mask: [1, H, W] 或 [1, 1, H, W]，初始掩码（1=可见，0=遮挡），中心可见
    t: [B]，时间步，t ∈ [0,1]
    img_size: 图像大小
    返回: continuous_mask [B, 1, H, W]，随时间线性扩展的掩码
    """
    # 确保 mask 是 [1, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1, H, W]

    B = t.size(0)
    H, W = img_size, img_size

    # 扩展 t 到 [B, 1, 1, 1]
    t_expand = t.view(B, 1, 1, 1)

    # 获取原始可见区域的边界（假设是矩形）
    # 找到 mask 中值为 1 的像素坐标
    coords = torch.where(mask[0] == 1)
    if len(coords[0]) == 0:
        raise ValueError("Initial mask has no visible region.")

    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()

    # 中心点
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # 原始半宽和半高
    orig_h_half = (y_max - y_min) // 2
    orig_w_half = (x_max - x_min) // 2

    # 最大可扩展到全图
    max_h_half = center_y  # 向上/下最多到边缘
    max_w_half = center_x  # 向左/右最多到边缘

    # 防止除零
    max_h_half = max(max_h_half, 1)
    max_w_half = max(max_w_half, 1)

    # 当前扩展的半宽/半高（线性增长）
    curr_h_half = orig_h_half + t_expand * (max_h_half - orig_h_half)  # [B,1,1,1]
    curr_w_half = orig_w_half + t_expand * (max_w_half - orig_w_half)

    # 创建网格
    y_grid = torch.arange(H, device=t.device).view(1, 1, H, 1).float()
    x_grid = torch.arange(W, device=t.device).view(1, 1, 1, W).float()

    # 判断是否在当前可见区域内
    in_y = (y_grid >= (center_y - curr_h_half)) & (y_grid <= (center_y + curr_h_half))
    in_x = (x_grid >= (center_x - curr_w_half)) & (x_grid <= (center_x + curr_w_half))

    continuous_mask = in_y & in_x  # [B, 1, H, W]
    continuous_mask = continuous_mask.float()

    return continuous_mask



def compute_flow_vector_continuous(x0, x1, mask, t, alpha=10.0):
    """
    计算连续mask插值下的真实流场
    x0: 噪声图像 [B, C, H, W]
    x1: 真实图像 [B, C, H, W]
    mask: 原始掩码 [B, 1, H, W]
    t: [B]，时间步
    alpha: 控制sigmoid陡峭程度
    返回: vt_true [B, C, H, W]
    """
    B, C, H, W = x0.shape
    t_expand = t.view(B, 1, 1, 1)
    # 连续mask
    M_t = torch.sigmoid(alpha * (t_expand - 0.5)) * mask  # [B, 1, H, W]
    # 对t求导
    dM_dt = alpha * torch.sigmoid(alpha * (t_expand - 0.5)) * (1 - torch.sigmoid(alpha * (t_expand - 0.5))) * mask
    # vt_true
    vt_true = dM_dt * (x1 - x0)
    return vt_true

def flow_matching_loss(model, clean_img, masked_img, mask=None, num_time_samples=20, alpha=10.0):
    B, C, H, W = clean_img.shape
    device = clean_img.device
    losses = []
    for _ in range(num_time_samples):
        t = torch.rand(B, device=device)
        x0 = masked_img
        x1 = clean_img
        if mask is not None:
            continuous_mask = linear_growth_mask(mask, t, alpha=alpha).to(device)
        else:
            print("Warning: No mask provided, using all ones mask.")
        xt = continuous_mask * x1 + (1 - continuous_mask) * x0
        vt_true = compute_flow_vector_continuous(x0, clean_img, mask, t, alpha=alpha)
        vt_pred = model(xt, t, masked_img)
        loss = F.mse_loss(vt_pred, vt_true)
        losses.append(loss)
    return torch.stack(losses).mean()