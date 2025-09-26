import torch
import matplotlib.pyplot as plt

def save_inpainting_result(model, batch, mask, device, filename, num_steps=10):
    model.eval()
    x_cond, x_true = batch
    x_cond = x_cond.to(device)
    x_true = x_true.to(device)
    mask = mask.to(device)
    B, C, H, W = x_true.shape

    # 为每张图像单独处理
    preds = []
    
    with torch.no_grad():
        for b in range(B):  # 对batch中的每张图像单独处理
            # 取出单张图像
            x_cond_single = x_cond[b:b+1]  # 保持batch维度 [1, C, H, W]
            x_true_single = x_true[b:b+1]
            
            # 初始化 x 为标准高斯噪声
            x = torch.randn_like(x_cond_single, device=device)
            
            # 流匹配模型从 t=0 到 t=1 进行积分
            for i in range(num_steps):
                # 时间从 0 到 1 均匀分布 (单个时间值)
                t = torch.full((1,), i / num_steps, device=device)

                xt = x  # 当前状态就是 xt
                
                # 预测速度场 (使用masked_img作为条件)
                vt = model(xt, t, x_cond_single)

                # 前向欧拉积分步长
                dt = 1.0 / num_steps
                x = x + vt * dt
            
            preds.append(x)
    
    # 合并所有预测结果
    pred = torch.cat(preds, dim=0)

    pred = pred.cpu()
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