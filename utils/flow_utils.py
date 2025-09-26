import torch
import torch.nn.functional as F


def compute_velocity_field(x0, x1):
    """
    根据线性插值路径计算真实流场（速度场）
    x0: [B, C, H, W] 噪声/遮挡图像
    x1: [B, C, H, W] 真实图像
    continuous_mask: [B, 1, H, W] 当前时刻的可见掩码 (本实现中不使用)
    dm_dt: [B, 1, H, W] 掩码对时间的导数 (本实现中不使用)
    t: [B] 时间步
    mask: 原始掩码 (本实现中不使用)
    返回: vt_true [B, C, H, W] 真实速度场
    """
    # 使用线性插值路径计算速度场
    # 对于线性插值 x_t = (1 - t) * x_0 + t * x_1
    # 速度场 v_t = dx_t/dt = -x_0 + x_1 = (x_1 - x_0)
    vt_true = x1 - x0
    return vt_true


def flow_matching_loss(model, clean_img, masked_img):
    """

    model: 流模型
    clean_img: 真实图像 [B, C, H, W]
    masked_img: 遮挡图像 [B, C, H, W]
    """
    B, C, H, W = clean_img.shape
    device = clean_img.device

    # 随机采样单个时间点，范围在[0, 1]之间
    t = torch.rand(B, device=device)

    # 使用标准高斯噪声作为起点
    x0 = torch.randn(B, C, H, W, device=device)
    x1 = clean_img
    
    # 使用线性插值构造 xt
    # 线性插值: xt = (1 - t) * x0 + t * x1
    t_expand = t.view(B, 1, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1

    # 计算真实速度场 (不使用掩码)
    vt_true = compute_velocity_field(x0, x1)

    # 模型预测速度场 (使用masked_img作为条件)
    vt_pred = model(xt, t, masked_img)

    # 计算流匹配损失 (MSE)
    flow_loss = F.mse_loss(vt_pred, vt_true, reduction='mean')
    

    return flow_loss
