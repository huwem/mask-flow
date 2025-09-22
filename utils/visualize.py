import torch
import matplotlib.pyplot as plt
from utils.flow_utils import generate_continuous_mask

def save_inpainting_result(model, batch, mask, device, filename, num_steps=50, alpha=10.0):
    model.eval()
    x_cond, x_true = batch
    x_cond = x_cond.to(device)
    x_true = x_true.to(device)
    mask = mask.to(device)
    B, C, H, W = x_true.shape

    # 初始化为带掩码的输入
    x = x_cond.clone().to(device)
    
    with torch.no_grad():
        for i in range(num_steps, 0, -1):
            t = torch.full((B,), i / num_steps, device=device)  
            continuous_mask = generate_continuous_mask(mask, t, alpha=alpha)
            xt = continuous_mask * x_true + (1 - continuous_mask) * x
            vt = model(xt, t, x_cond)
            x = x + vt * (1.0 / num_steps)

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