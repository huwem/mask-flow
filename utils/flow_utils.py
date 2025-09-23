import torch
import torch.nn.functional as F

def linear_growth_mask(mask, t, img_size=512, return_deriv=True):
    """
    mask: [1, H, W] 或 [1, 1, H, W]，初始掩码（1=可见，0=遮挡）
    t: [B]，时间步，t ∈ [0,1]
    img_size: 图像大小
    return_deriv: 是否返回 dm/dt
    返回: 
        continuous_mask [B, 1, H, W]
        dm_dt (optional) [B, 1, H, W] 掩码对时间的导数
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]

    B = t.size(0)
    H, W = img_size, img_size

    t_expand = t.view(B, 1, 1, 1)  # shape: [B,1,1,1]

    coords = torch.where(mask[0, 0] == 1)
    if len(coords[0]) == 0:
        raise ValueError("Initial mask has no visible region.")

    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    orig_h_half = (y_max - y_min) // 2
    orig_w_half = (x_max - x_min) // 2

    max_h_half = min(center_y, H - 1 - center_y)
    max_w_half = min(center_x, W - 1 - center_x)

    max_h_half = max(max_h_half, 1)
    max_w_half = max(max_w_half, 1)

    # 线性插值：h(t) = h0 + t * (h1 - h0)
    curr_h_half = orig_h_half + t_expand * (max_h_half - orig_h_half)  # [B,1,1,1]
    curr_w_half = orig_w_half + t_expand * (max_w_half - orig_w_half)  # [B,1,1,1]

    # 导数 dh/dt = (max_h_half - orig_h_half), dw/dt = (max_w_half - orig_w_half)
    dh_dt = (max_h_half - orig_h_half)  # scalar
    dw_dt = (max_w_half - orig_w_half)  # scalar

    # 创建坐标网格
    y_grid = torch.arange(H, device=t.device).view(1, 1, H, 1).float()  # [1,1,H,1]
    x_grid = torch.arange(W, device=t.device).view(1, 1, 1, W).float()  # [1,1,1,W]

    dy = torch.abs(y_grid - center_y)  # [1,1,H,W]
    dx = torch.abs(x_grid - center_x)  # [1,1,H,W]

    # 当前时刻的掩码：矩形框内为1
    in_y = (dy <= curr_h_half)  # [B,1,H,W]
    in_x = (dx <= curr_w_half)  # [B,1,H,W]
    continuous_mask = (in_y & in_x).float()  # [B,1,H,W]

    if not return_deriv:
        return continuous_mask

    # -------------------------------
    # 计算 dm/dt = d(in_y)/dt * in_x + in_y * d(in_x)/dt
    # 但由于是矩形，边界处才有非零导数
    # 我们近似为：仅在矩形边缘宽度为1的环上，且导数为 dh_dt 或 dw_dt
    # 更简单的方法：使用解析梯度（符号微分）
    # 注意：PyTorch 不直接支持 step 函数的梯度，所以我们用 smooth approximation
    # 或者我们只关心“质量守恒”的总流动，可以返回一个稀疏的边界梯度
    # 这里采用一种简化方式：只计算边界处的梯度（数值近似）
    # --------------------------------

    # 方法：有限差分近似 dm/dt
    eps = 1e-4
    t_upper = torch.clamp(t_expand + eps, 0.0, 1.0)
    t_lower = torch.clamp(t_expand - eps, 0.0, 1.0)

    h_upper = orig_h_half + t_upper * (max_h_half - orig_h_half)
    h_lower = orig_h_half + t_lower * (max_h_half - orig_h_half)
    w_upper = orig_w_half + t_upper * (max_w_half - orig_w_half)
    w_lower = orig_w_half + t_lower * (max_w_half - orig_w_half)

    in_y_upper = (dy <= h_upper).float()
    in_y_lower = (dy <= h_lower).float()
    in_x_upper = (dx <= w_upper).float()
    in_x_lower = (dx <= w_lower).float()

    m_upper = in_y_upper * in_x_upper
    m_lower = in_y_lower * in_x_lower

    dm_dt = (m_upper - m_lower) / (2 * eps)  # 数值导数 [B,1,H,W]

    return continuous_mask, dm_dt


def compute_velocity_field(x0, x1, continuous_mask, dm_dt, t, mask):
    """
    x0: [B, C, H, W] 噪声/masked图像
    x1: [B, C, H, W] 清晰图像
    continuous_mask: [B, 1, H, W] 当前时刻的可见掩码
    dm_dt: [B, 1, H, W] 掩码对时间的导数
    t: [B]
    mask: 原始掩码（可选）
    返回: vt_true [B, C, H, W]
    """
    # 根据插值路径求导
    vt_true = dm_dt * (x1 - x0)  # [B,1,H,W] * [B,C,H,W] -> broadcast to [B,C,H,W]
    return vt_true

def flow_matching_loss(model, clean_img, masked_img, mask=None, num_time_samples=20):
    B, C, H, W = clean_img.shape
    device = clean_img.device
    losses = []

    t_values = torch.linspace(0.0, 1.0, num_time_samples, device=device)
    
    for i in range(num_time_samples):
        t = t_values[i].expand(B)  # [B]
        t.requires_grad_()  # 如果你想让 t 参与自动微分（可选）

        x0 = masked_img
        x1 = clean_img
        
        if mask is not None:
            # 获取掩码和其时间导数
            continuous_mask, dm_dt = linear_growth_mask(mask, t, img_size=H, return_deriv=True)
            continuous_mask = continuous_mask.to(device)
            dm_dt = dm_dt.to(device)
        else:
            raise ValueError("Mask must be provided.")

        # 构造 xt
        xt = continuous_mask * x1 + (1 - continuous_mask) * x0  # [B,C,H,W]

        # 计算真实速度场
        vt_true = compute_velocity_field(x0, x1, continuous_mask, dm_dt, t, mask)

        # 模型预测
        vt_pred = model(xt, t, masked_img)  # 假设 model 接受 t 作为输入

        # 损失：只在有流动的地方计算（即边界扩展区域）
        # 注意：dm_dt 在扩展前沿非零，其他地方为0，所以天然加权
        loss = F.mse_loss(vt_pred, vt_true, reduction='mean')
        losses.append(loss)

    return torch.stack(losses).mean()