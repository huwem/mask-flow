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


class AttentionBlock(nn.Module):
    """注意力模块"""
    def __init__(self, channels, num_heads=8, head_dim=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        if head_dim is None:
            self.head_dim = channels // num_heads
        else:
            self.head_dim = head_dim
            self.num_heads = channels // head_dim
            
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # 计算Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # 每个形状为 [B, num_heads, head_dim, H*W]
        
        # 注意力计算
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力权重
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class ResBlock(nn.Module):
    """残差块，包含卷积和可选的注意力"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.act2 = nn.GELU()
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
            
        # 注意力机制（可选）
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        
        # 第一个卷积块
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        # 第二个卷积块
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        # 残差连接
        h = h + residual
        
        # 注意力机制
        h = self.attention(h)
        
        return h


class ImprovedConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, width=64, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.width = width

        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 4),
        )

        # 条件图像嵌入
        self.cond_embed = nn.Conv2d(3, width, kernel_size=1)

        # 输入头
        self.head = nn.Conv2d(3 + width * 4 + width, width, kernel_size=1)

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        # 各阶段通道数
        enc_channels = [width, width*2, width*4, width*8]  # [64, 128, 256, 512]

        for i in range(len(enc_channels)):
            blocks = nn.ModuleList()
            in_ch = enc_channels[i]
            
            # 在较高分辨率添加注意力机制
            use_attention = (enc_channels[i] >= width * 4)  # 在256通道及以上使用注意力
            
            for j in range(num_blocks):
                out_ch = in_ch if j == 0 else in_ch
                in_ch_actual = in_ch if j == 0 and i == 0 else out_ch
                blocks.append(ResBlock(in_ch_actual, out_ch, use_attention=use_attention))
            self.enc_blocks.append(blocks)

            if i < len(enc_channels) - 1:
                self.downsample.append(Downsample(enc_channels[i], enc_channels[i+1]))

        # --- Bottleneck ---
        # 瓶颈层使用注意力机制
        self.bottleneck = nn.ModuleList([
            ResBlock(width*8, width*8, use_attention=True),
            ResBlock(width*8, width*8, use_attention=True)
        ])

        # --- Decoder ---
        self.upsample = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # 解码器上采样层
        self.upsample = nn.ModuleList([
            Upsample(width * 8, width * 4),  # 512 -> 256
            Upsample(width * 4, width * 2),  # 256 -> 128
            Upsample(width * 2, width),      # 128 -> 64
        ])

        # 解码器块
        dec_in_channels = [width*8, width*4, width*2]  # 拼接后通道: 512, 256, 128
        dec_out_channels = [width*4, width*2, width]   # 目标输出: 256, 128, 64

        for i in range(len(dec_in_channels)):
            in_ch = dec_in_channels[i]
            out_ch = dec_out_channels[i]
            blocks = nn.ModuleList()
            
            # 在较高分辨率添加注意力机制
            use_attention = (out_ch >= width * 2)  # 在128通道及以上使用注意力

            # 第一个 block: 处理拼接后的高维特征
            blocks.append(ResBlock(in_ch, out_ch, use_attention=use_attention))

            # 剩余 block: 在目标通道上进行特征精炼
            for j in range(1, num_blocks):
                blocks.append(ResBlock(out_ch, out_ch, use_attention=use_attention))

            self.dec_blocks.append(blocks)

        # --- 输出头 ---
        self.tail = nn.Sequential(
            nn.GroupNorm(min(32, width // 4), width),
            nn.GELU(),
            nn.Conv2d(width, 3, kernel_size=1)
        )

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

        # 时间嵌入
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
        for block in self.bottleneck:
            h = block(h)

        # --- Decoder Forward ---
        for i in range(len(self.upsample)):
            h = self.upsample[i](h)
            skip = enc_outputs[len(self.enc_blocks) - 2 - i]
            h = torch.cat([h, skip], dim=1)  # 拼接跳跃连接

            # 通过该阶段的多个 ResBlock
            for block in self.dec_blocks[i]:
                h = block(h)

        # --- Output ---
        h = self.tail(h)
        return h


# === 测试代码 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedConditionalUNet(width=64, num_blocks=2).to(device)
    print(f"Model initialized with width={model.width}")

    # 模拟输入
    x = torch.randn(4, 3, 64, 64).to(device)  # 使用较小尺寸便于测试
    t = torch.rand(4).to(device)
    x_cond = torch.randn(4, 3, 64, 64).to(device)

    with torch.no_grad():
        vt = model(x, t, x_cond)
        print(f"Input: {x.shape}")
        print(f"Output: {vt.shape}")  # Should be [4, 3, 64, 64]
        assert vt.shape == x.shape, "Output shape mismatch!"
        print("✅ Forward pass successful!")