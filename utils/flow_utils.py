import torch
import torch.nn.functional as F

def generate_continuous_mask(mask, t, alpha=10.0):
    """
    mask: [B, 1, H, W]，原始掩码（1为可见，0为遮挡）
    t: [B]，时间步
    alpha: 控制sigmoid陡峭程度
    返回连续mask [B, 1, H, W]
    """
    B, _, H, W = mask.shape
    t_expand = t.view(B, 1, 1, 1)
    # 连续mask：sigmoid方式
    continuous_mask = torch.sigmoid(alpha * (t_expand - 0.5)) * mask
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

def flow_matching_loss(model, clean_img, masked_img, mask=None, num_time_samples=5, alpha=10.0):
    B, C, H, W = clean_img.shape
    device = clean_img.device
    losses = []
    for _ in range(num_time_samples):
        t = torch.rand(B, device=device)
        x0 = torch.randn_like(clean_img)
        if mask is not None:
            continuous_mask = generate_continuous_mask(mask, t, alpha=alpha).to(device)
        else:
            continuous_mask = torch.ones(B, 1, H, W, device=device)
        xt = continuous_mask * clean_img + (1 - continuous_mask) * x0
        vt_true = compute_flow_vector_continuous(x0, clean_img, mask, t, alpha=alpha)
        vt_pred = model(xt, t, masked_img)
        # 损失只在掩码区域计算
        mask_expand = mask.expand_as(vt_true)
        loss = F.mse_loss(vt_pred[mask_expand == 1], vt_true[mask_expand == 1])
        losses.append(loss)
    return torch.stack(losses).mean()