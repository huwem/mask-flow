import torch
from models.conditional_unet import ConditionalUNet
from PIL import Image
from torchvision import transforms
import numpy as np
from utils.flow_utils import generate_continuous_mask
from utils.mask_utils import random_rectangle_mask, inverse_rectangle_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch_180.pth", map_location=device))
model.eval()

img = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img).unsqueeze(0).to(device)

# 使用 mask_utils 生成掩码
mask = random_rectangle_mask(64, 64, max_size=0.4)  # 或 inverse_rectangle_mask(64, 64)
mask = mask.to(device)
mask = mask.unsqueeze(0) if mask.dim() == 3 else mask  # [1, 1, H, W]
masked_img = img * mask

x = torch.randn_like(img)
num_steps = 100
alpha = 10.0

with torch.no_grad():
    for i in range(num_steps, 0, -1):
        t = torch.full((1,), i / num_steps, device=device)
        continuous_mask = generate_continuous_mask(mask, t, alpha=alpha)
        xt = continuous_mask * img + (1 - continuous_mask) * x
        vt = model(xt, t, masked_img)
        x = x + vt * (1.0 / num_steps)

def tensor_to_pil(t):
    t = (t[0].cpu() + 1) / 2
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

Image.fromarray(tensor_to_pil(masked_img)).save("results/masked.png")
Image.fromarray(tensor_to_pil(x)).save("results/inpainted.png")
print("✅ 推理完成，结果已保存。")