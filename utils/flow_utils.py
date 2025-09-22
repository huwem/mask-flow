import torch
import torch.nn.functional as F

def linear_growth_mask(mask, t, img_size=512):
    """
    mask: [1, H, W] 或 [1, 1, H, W]，初始掩码（1=可见，0=遮挡），中心可见
    t: [B]，时间步，t ∈ [0,1]
    img_size: 图像大小
    返回: continuous_mask [B, 1, H, W]，随时间线性扩展的掩码
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]

    B = t.size(0)
    H, W = img_size, img_size

    t_expand = t.view(B, 1, 1, 1)

    coords = torch.where(mask[0] == 1)
    if len(coords[0]) == 0:
        raise ValueError("Initial mask has no visible region.")

    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    orig_h_half = (y_max - y_min) // 2
    orig_w_half = (x_max - x_min) // 2

    max_h_half = center_y
    max_w_half = center_x

    max_h_half = max(max_h_half, 1)
    max_w_half = max(max_w_half, 1)

    curr_h_half = orig_h_half + t_expand * (max_h_half - orig_h_half)
    curr_w_half = orig_w_half + t_expand * (max_w_half - orig_w_half)

    y_grid = torch.arange(H, device=t.device).view(1, 1, H, 1).float()
    x_grid = torch.arange(W, device=t.device).view(1, 1, 1, W).float()

    in_y = ((y_grid >= (center_y - curr_h_half)) & (y_grid <= (center_y + curr_h_half))).float()
    in_x = ((x_grid >= (center_x - curr_w_half)) & (x_grid <= (center_x + curr_w_half))).float()

    continuous_mask = in_y * in_x
    return continuous_mask


def compute_velocity_field(x0, x1, continuous_mask, t):
    """
    计算连续mask插值下的真实流场
    x0: 噪声图像 [B, C, H, W]
    x1: 真实图像 [B, C, H, W]
    continuous_mask: 随时间变化的掩码 [B, 1, H, W]
    t: 时间步 [B]
    返回: vt_true [B, C, H, W]
    """
    B, _, _, _ = x0.shape
    t_expand = t.view(B, 1, 1, 1)

    # 计算 dM/dt
    d_curr_h_dt = (continuous_mask.shape[2] // 2 - (continuous_mask.sum(dim=(2, 3)).mean(dim=-1) / continuous_mask.shape[3]).unsqueeze(-1).unsqueeze(-1))
    d_curr_w_dt = (continuous_mask.shape[3] // 2 - (continuous_mask.sum(dim=(2, 3)).mean(dim=-1) / continuous_mask.shape[2]).unsqueeze(-1).unsqueeze(-1))

    dM_dt = (d_curr_h_dt + d_curr_w_dt) / 2

    vt_true = dM_dt * (x1 - x0)
    return vt_true


def flow_matching_loss(model, clean_img, masked_img, mask=None, num_time_samples=20):
    B, C, H, W = clean_img.shape
    device = clean_img.device
    losses = []

    for _ in range(num_time_samples):
        t = torch.rand(B, device=device)
        x0 = masked_img
        x1 = clean_img
        
        if mask is not None:
            continuous_mask = linear_growth_mask(mask, t, img_size=H).to(device)
        else:
            continuous_mask = torch.ones_like(masked_img[:, :1])

        xt = continuous_mask * x1 + (1 - continuous_mask) * x0
        vt_true = compute_velocity_field(x0, x1, continuous_mask, t)
        vt_pred = model(xt, t, masked_img)
        loss = F.mse_loss(vt_pred, vt_true)
        losses.append(loss)

    return torch.stack(losses).mean()