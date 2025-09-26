# train.py
import torch
from torch.utils.data import DataLoader, Subset
from models.conditional_unet import ConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def main():
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置：设置epoch数为200
    config['num_epochs'] = 2000

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 创建TensorBoard日志目录
    writer = SummaryWriter(log_dir='runs/flow_inpaint_training')
    
    # 记录配置信息
    writer.add_text('Config', str(config))

    # 创建数据集和数据加载器
    print("Loading dataset...")
    try:
        full_dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
        # 只使用前256张图片
        dataset = Subset(full_dataset, range(min(256, len(full_dataset))))
        print(f"Dataset loaded with {len(dataset)} samples ")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # 加速数据传输到GPU
    )

    # 初始化模型
    print("Initializing model...")
    model = ConditionalUNet(in_channels=3, width=config.get('model_width', 128))
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # 设置优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs']
    )

    print(f"🚀 开始训练 ({config['num_epochs']} epochs)...")
    
    # 用于记录训练损失
    train_losses = []
    
    # 固定一批验证样本用于可视化
    val_sample = None

    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 遍历所有batch
        for i, (masked, mask, clean) in enumerate(dataloader):
            # 确保所有张量都在相同设备上
            masked, clean, mask = masked.to(device, non_blocking=True), clean.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            # 在第一个epoch保存验证样本
            if epoch == 0 and val_sample is None:
                val_sample = (masked[:4].clone(), clean[:4].clone())
                val_mask = mask[:4].clone()
            
            # 检查输入是否有效
            if torch.isnan(masked).any() or torch.isnan(clean).any():
                print(f"NaN values detected in batch {i}, skipping...")
                continue
                
            try:
                # 计算流匹配损失
                loss = flow_matching_loss(model, clean, masked, mask=mask)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss value in batch {i}, skipping...")
                    continue
                    
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 记录每个batch的损失
                writer.add_scalar('Batch/Loss', loss.item(), epoch * len(dataloader) + i)
                
                # 打印进度
                if (i + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{config['num_epochs']}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"Error in batch {i}: {e}")
                print("Skipping this batch...")
                continue

        # 计算平均损失
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            

        # 定期保存检查点和可视化结果（每100个epoch保存一次）
        if (epoch + 1) % 100 == 0:
            # 保存模型检查点
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # 生成可视化结果
            with torch.no_grad():
                try:
                    # 保存当前batch的可视化结果
                    save_inpainting_result(
                        model,
                        (masked[:4], clean[:4]),
                        mask[:4],
                        device,
                        f"{config['results_dir']}/epoch_{epoch+1}.png"
                    )
                    
                    # 保存固定样本的可视化结果，便于比较训练过程
                    if val_sample is not None:
                        val_masked, val_clean = val_sample
                        val_masked, val_clean = val_masked.to(device), val_clean.to(device)
                        save_inpainting_result(
                            model,
                            (val_masked, val_clean),
                            val_mask,
                            device,
                            f"{config['results_dir']}/fixed_sample_epoch_{epoch+1}.png"
                        )
                    
                    # 将图像结果添加到TensorBoard
                    if os.path.exists(f"{config['results_dir']}/epoch_{epoch+1}.png"):
                        result_image = plt.imread(f"{config['results_dir']}/epoch_{epoch+1}.png")
                        writer.add_image('Training Results', 
                                       np.transpose(result_image, (2, 0, 1)), 
                                       epoch, 
                                       dataformats='CHW')
                                       
                except Exception as e:
                    print(f"Failed to save visualization: {e}")

    # 保存最终模型
    try:
        final_model_path = config.get('model_save_path', "checkpoints/final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ 训练完成，最终模型已保存至 {final_model_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")


        

    
    writer.close()
    print("Training completed and TensorBoard logs saved.")

if __name__ == "__main__":
    main()