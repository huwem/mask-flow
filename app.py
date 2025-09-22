# app.py
import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import random

# 导入模型（确保路径正确）
from models.conditional_unet import ConditionalUNet

# 检查是否有预训练权重
MODEL_PATH = "checkpoints/flow_inpaint.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"请先训练模型并保存到 {MODEL_PATH}")

# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def create_inverse_mask(image, min_visible_ratio=0.1, max_visible_ratio=0.3):
    """在图像上创建反向遮挡：遮住大部分图片，只留下单块区域可见"""
    H, W = image.shape[0], image.shape[1]
    
    # 创建全遮挡的遮罩
    mask = np.zeros((H, W), dtype=np.float32)
    
    # 计算可见区域大小
    visible_ratio = random.uniform(min_visible_ratio, max_visible_ratio)
    visible_h = int(H * visible_ratio)
    visible_w = int(W * visible_ratio)
    
    # 随机确定可见区域位置
    rh = random.randint(0, H - visible_h)
    rw = random.randint(0, W - visible_w)
    
    # 设置可见区域
    mask[rh:rh+visible_h, rw:rw+visible_w] = 1
    
    return mask

def preprocess_image(image):
    """转换为张量"""
    transform = lambda x: (x.astype(np.float32) / 255.0) * 2 - 1  # [0,255] -> [-1,1]
    return torch.tensor(transform(image)).permute(2, 0, 1).unsqueeze(0)

def postprocess_tensor(tensor):
    """转回图像"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1) / 2  # [-1,1] -> [0,1]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

@torch.no_grad()
def inpaint_image(input_img, steps=100):
    # 调整大小
    input_img = Image.fromarray(input_img).resize((64, 64))
    input_np = np.array(input_img)
    
    # 创建反向遮挡图（遮住大部分图片，只留下单块区域可见）
    mask = create_inverse_mask(input_np)
    masked_img = input_np * mask[..., None]

    # 预处理
    x_cond = preprocess_image(masked_img).to(device)
    x = torch.randn(1, 3, 64, 64).to(device)

    # 流匹配反向采样
    for i in range(steps, 0, -1):
        t = torch.full((1,), i / steps, device=device)
        dt = 1.0 / steps
        vt = model(x, t, x_cond)
        x = x - vt * dt

    # 后处理
    result = postprocess_tensor(x)
    original = np.array(input_img)
    masked = (masked_img).astype(np.uint8)

    return original, masked, result

# 构建 Gradio 界面
demo = gr.Interface(
    fn=inpaint_image,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=[
        gr.Image(type="numpy", label="原始图像"),
        gr.Image(type="numpy", label="遮挡图像"),
        gr.Image(type="numpy", label="补全结果")
    ],
    title="🎨 FlowInpaint - 基于流匹配的图像修复",
    description="上传一张人脸图像，系统将自动遮挡大部分区域并进行修复，只保留一小块可见区域。",
    examples=["test.jpg"],  # 准备一张测试图
    cache_examples=False
)

# 启动
if __name__ == "__main__":
    demo.launch()