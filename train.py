# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import yaml
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.CA_unet import ImprovedConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result

def main():
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    seed = config.get('seed', 42)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    # 替换配置中的变量占位符
    for key in config:
        if isinstance(config[key], str) and '${seed}' in config[key]:
            config[key] = config[key].replace('${seed}', str(seed))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 创建TensorBoard日志目录
    writer = SummaryWriter(log_dir=config['tensorboard_log_dir'])
    
    # 记录配置信息
    writer.add_text('Config', str(config))

    # 创建数据集和数据加载器
    print("Loading dataset...")
    try:
        full_dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
        # 限制数据集大小
        max_dataset_size = config.get('max_dataset_size', 8000)
        if len(full_dataset) > max_dataset_size:
            dataset = Subset(full_dataset, range(min(max_dataset_size, len(full_dataset))))
            print(f"Dataset limited to {len(dataset)} samples")
        else:
            dataset = full_dataset
            print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size = 64, 
        shuffle=True, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )

    # 初始化模型
    print("Initializing model...")
    model = ImprovedConditionalUNet(
        in_channels=config.get('in_channels', 3),
        width=32
    )
    model = model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 设置优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs']
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 20),
            gamma=config.get('gamma', 0.5)
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs']
        )
    
    # 创建梯度缩放器用于混合精度训练
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print(f"🚀 开始训练 ({config['num_epochs']} epochs)...")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Dataset size: {len(dataset)}")
    
    # 用于记录训练损失
    train_losses = []
    
    # 固定一批验证样本用于可视化
    val_sample = None
    val_mask = None

    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 遍历所有batch
        for i, (masked, mask, clean) in enumerate(dataloader):
            # 确保所有张量都在相同设备上
            masked = masked.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # 在第一个epoch保存验证样本
            if epoch == 0 and val_sample is None:
                val_sample = (masked[:4].clone(), clean[:4].clone())
                val_mask = mask[:4].clone()
            
            # 检查输入是否有效
            if torch.isnan(masked).any() or torch.isnan(clean).any():
                print(f"NaN values detected in batch {i}, skipping...")
                continue
                
            try:
                # 使用自动混合精度
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    loss = flow_matching_loss(model, clean, masked)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss value in batch {i}, skipping...")
                    continue
                    
                # 反向传播使用缩放器（如果使用CUDA）
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # 更新权重
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

        # 计算平均损失
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录epoch级别的损失和学习率
            writer.add_scalar('Epoch/Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed with no valid batches.")
        
        # 更新学习率
        scheduler.step()

        # 定期保存检查点和可视化结果
        if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
            # 保存模型检查点
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss if num_batches > 0 else None,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # 生成可视化结果
            model.eval()
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
                    if val_sample is not None and val_mask is not None:
                        val_masked, val_clean = val_sample
                        val_masked, val_clean = val_masked.to(device), val_clean.to(device)
                        save_inpainting_result(
                            model,
                            (val_masked, val_clean),
                            val_mask,
                            device,
                            f"{config['results_dir']}/fixed_sample_epoch_{epoch+1}.png"
                        )
                    
                except Exception as e:
                    print(f"Failed to save visualization: {e}")
            model.train()

        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终模型
    try:
        final_model_path = config.get('model_save_path', "checkpoints/final_model.pth")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ 训练完成，最终模型已保存至 {final_model_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    writer.close()
    print("Training completed and TensorBoard logs saved.")

if __name__ == "__main__":
    main()