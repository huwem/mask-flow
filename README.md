# Flow-Inpaint: 基于流匹配的掩码插值图像修复

## 项目简介

本项目实现了一种基于流匹配（Flow Matching）的图像修复方法。与传统线性插值不同，我们采用**动态掩码插值**：在训练过程中，掩码区域会随时间步$t$逐渐变化，模型学习如何逐步恢复被遮挡的图像内容。

## 新插值原理

- **动态掩码插值**：  
  对于每个时间步$t$，生成一个动态掩码$M_t$，可见区域比例与$t$相关。插值公式如下：
  $$
  x_t = M_t \cdot x_1 + (1 - M_t) \cdot x_0
  $$
  其中 $x_1$ 为原始图像，$x_0$ 为噪声，$M_t$ 为随$t$变化的掩码。

- **训练目标**：  
  让模型预测流场$v_t$，指导如何从噪声逐步恢复到完整图像，损失函数为预测流与真实流的均方误差。

## 主要特性

- 支持CelebA人脸数据集
- 动态掩码插值训练，提升修复区域的自然度
- 训练与推理均支持mask控制
- TensorBoard可视化训练过程

## 运行方法

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据集  
   将CelebA数据集放入 `data/celeba/img_align_celeba/` 目录。

3. 训练模型
   ```bash
   python mini_train.py
   ```

4. 推理/修复
   ```bash
   python sample.py
   ```

## 代码结构

- `mini_train.py`：训练主程序，采用动态掩码插值
- `sample.py`：推理脚本，支持自定义mask
- `utils/flow_utils.py`：流匹配损失与动态mask插值实现
- `datasets/celeba_dataset.py`：数据集加载
- `models/conditional_unet.py`：条件UNet模型

## 掩码插值示意

训练时，掩码区域随$t$逐步扩展，模型学习逐步填充缺失内容：

```
t=0.1   t=0.5   t=0.9
[██░░] [████] [██████]
```

## 数据集准备

请下载 CelebA 数据集并解压到 `data/celeba/img_align_celeba/` 目录。  
如需使用其它数据集，请参考 `datasets/celeba_dataset.py` 修改数据加载逻辑。

## 训练与推理参数说明

- 训练参数可在 `config/config.yaml` 中调整，如 batch_size、learning_rate、num_epochs 等。
- 推理脚本 `sample.py` 支持自定义遮罩区域，具体见代码注释。

## 结果可视化

训练过程中会自动保存修复结果图片到 `results/` 目录，并可通过 TensorBoard 查看训练日志：

```bash
tensorboard --logdir runs/
```

## 联系与反馈

如有问题或建议，请在 GitHub Issues 留言。

## 致谢

本项目参考了流匹配与扩散模型相关论文与开源实现。

