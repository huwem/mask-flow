import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积块：Conv2d + GELU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Module):
    """上采样模块：最近邻插值 + 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class Downsample(nn.Module):
    """下采样模块：平均池化 + 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, width=64, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.width = width

        # 时间嵌入层 - 修改输入维度为 1
        self.time_embed = nn.Sequential(
            nn.Linear(1, width),  # 改为 1 而不是 width
            nn.GELU(),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 4),
        )

        # 条件图像嵌入：1x1 卷积扩展通道
        self.cond_embed = nn.Conv2d(3, width, kernel_size=1)

        # 输入头：x + time_emb + cond_emb → 初始特征
        self.head = nn.Conv2d(3 + width * 4 + width, width, kernel_size=1)

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        # 各阶段通道数
        enc_channels = [width, width*2, width*4, width*8]  # [64, 128, 256, 512]

        for i in range(len(enc_channels)):
            blocks = nn.ModuleList()
            in_ch = enc_channels[i]
            # 第一个 block 输入通道取决于是否是第一层
            for j in range(num_blocks):
                blocks.append(ConvBlock(in_ch if j == 0 else in_ch, in_ch))
            self.enc_blocks.append(blocks)

            if i < len(enc_channels) - 1:
                self.downsample.append(Downsample(enc_channels[i], enc_channels[i+1]))

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(width*8, width*8)  # 512 → 512

        # --- Decoder ---
        self.upsample = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # 解码器上采样层
        self.upsample = nn.ModuleList([
            Upsample(width * 8, width * 4),  # 512 -> 256
            Upsample(width * 4, width * 2),  # 256 -> 128
            Upsample(width * 2, width),      # 128 -> 64
        ])

        # 解码器块：第一个 block 处理拼接（in=2*target），后续保持 target 通道
        dec_in_channels = [width*8, width*4, width*2]  # 拼接后通道: 512, 256, 128
        dec_out_channels = [width*4, width*2, width]   # 目标输出: 256, 128, 64

        for i in range(len(dec_in_channels)):
            in_ch = dec_in_channels[i]
            out_ch = dec_out_channels[i]
            blocks = nn.ModuleList()

            # 第一个 block: 将拼接后的高维特征映射到目标通道
            blocks.append(ConvBlock(in_ch, out_ch))

            # 剩余 block: 在目标通道上进行特征精炼
            for j in range(1, num_blocks):
                blocks.append(ConvBlock(out_ch, out_ch))

            self.dec_blocks.append(blocks)

        # --- 输出头 ---
        self.tail = nn.Conv2d(width, 3, kernel_size=1)

    def forward(self, x, t, x_cond=None):
        """
        Args:
            x: 当前噪声图像 [B, 3, H, W]
            t: 时间步 [B]
            x_cond: 条件图像 [B, 3, H, W]
        Returns:
            vt: 预测速度 [B, 3, H, W]
        """
        B = x.shape[0]

        # 时间嵌入 - 确保时间张量形状正确
        if t.dim() == 1:
            t = t.unsqueeze(1)  # 将 [B] 转换为 [B, 1]
        t_emb = self.time_embed(t.float())  # [B, 256]
        t_emb = t_emb.view(B, -1, 1, 1)  # [B, 256, 1, 1]

        # 条件图像嵌入
        if x_cond is None:
            x_cond = torch.zeros_like(x)
        cond_emb = self.cond_embed(x_cond)  # [B, 64, H, W]

        # 融合 x, t_emb, cond_emb
        t_emb_up = F.interpolate(t_emb, size=x.shape[-2:], mode='nearest')  # [B, 256, H, W]
        h = torch.cat([x, t_emb_up, cond_emb], dim=1)  # [B, 3+256+64, H, W]
        h = self.head(h)  # [B, 64, H, W]

        # 存储编码器输出用于跳跃连接
        enc_outputs = []

        # --- Encoder Forward ---
        for i in range(len(self.enc_blocks)):
            for block in self.enc_blocks[i]:
                h = block(h)
            enc_outputs.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)

        # --- Bottleneck ---
        h = self.bottleneck(h)

        # --- Decoder Forward ---
        for i in range(len(self.upsample)):
            h = self.upsample[i](h)
            skip = enc_outputs[len(self.enc_blocks) - 2 - i]
            h = torch.cat([h, skip], dim=1)  # 拼接跳跃连接

            # 通过该阶段的多个 ConvBlock
            for block in self.dec_blocks[i]:
                h = block(h)

        # --- Output ---
        h = self.tail(h)
        return h


# === 测试代码（可选）===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet(width=64, num_blocks=2).to(device)
    print(f"Model initialized with width={model.width}")

    # 模拟输入
    x = torch.randn(4, 3, 256, 256).to(device)
    t = torch.rand(4).to(device)
    x_cond = torch.randn(4, 3, 256, 256).to(device)

    with torch.no_grad():
        vt = model(x, t, x_cond)
        print(f"Input: {x.shape}")
        print(f"Output: {vt.shape}")  # Should be [4, 3, 256, 256]
        assert vt.shape == x.shape, "Output shape mismatch!"
        print("✅ Forward pass successful!")