import torch
import matplotlib.pyplot as plt
from utils.flow_utils import linear_growth_mask

def save_inpainting_result(model, batch, mask, device, filename, num_steps=50):
    model.eval()
    x_cond, x_true = batch
    x_cond = x_cond.to(device)
    x_true = x_true.to(device)
    mask = mask.to(device)
    B, C, H, W = x_true.shape

    # 初始化为带掩码的输入（这是推理的起点）
    x = x_cond.clone().to(device)
    
    with torch.no_grad():
        for i in range(num_steps, 0, -1):
            t = torch.full((B,), i / num_steps, device=device)
            
            # 在推理过程中，我们不能使用 x_true，应该只使用当前的 x 和 mask
            # xt 应该是当前状态 x，而不是基于 x_true 计算的
            xt = x  # 当前状态就是 xt
            
            # 预测速度场
            vt = model(xt, t, x_cond)
            
            # 更新 x（欧拉步进）
            x = x + vt * (1.0 / num_steps)
            
            # 应用掩码约束：保持已知区域不变
            # 在已知区域（mask=1），保持 x_cond 的值不变
            x = mask * x_cond + (1 - mask) * x

    pred = x.cpu()
    x_cond_np = (x_cond.cpu() + 1) / 2
    pred_np = (pred + 1) / 2
    true_np = (x_true.cpu() + 1) / 2

    fig, axes = plt.subplots(3, B, figsize=(B * 3, 9))
    if B == 1:
        axes = axes[:, None]
    for i in range(B):
        axes[0, i].imshow(x_cond_np[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title("Masked")
        axes[0, i].axis("off")

        axes[1, i].imshow(pred_np[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title("Inpainted")
        axes[1, i].axis("off")

        axes[2, i].imshow(true_np[i].permute(1, 2, 0).clamp(0, 1))
        axes[2, i].set_title("Ground Truth")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    model.train()