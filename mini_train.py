import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from models.CA_unet import ImprovedConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def setup_devices():
    """设置多GPU环境"""
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # 使用多个GPU
        device_ids = list(range(torch.cuda.device_count()))  # 使用所有可用GPU
        device = torch.device("cuda:0")  # 主设备
        print(f"Using devices: {[f'cuda:{i}' for i in device_ids]}")
        return device, device_ids
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_ids = [0]
        print("Using single GPU: cuda:0")
        return device, device_ids
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device, None

def setup_ddp(rank, world_size):
    """设置DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def main_worker(gpu, ngpus_per_node, config):
    """DDP工作进程主函数"""
    # 设置DDP
    setup_ddp(gpu, ngpus_per_node)
    
    # 设置设备
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    
    # 设置随机种子
    seed = config.get('seed', 42)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 替换配置中的变量占位符
    for key in config:
        if isinstance(config[key], str) and '${seed}' in config[key]:
            config[key] = config[key].replace('${seed}', str(seed))
    
    # 只在主进程中创建目录和TensorBoard
    if gpu == 0:
        os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
        os.makedirs(config['results_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=config['tensorboard_log_dir'])
        writer.add_text('Config', str(config))
    else:
        writer = None

    # 创建数据集
    print(f"GPU {gpu}: Loading dataset...")
    try:
        full_dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
        dataset = Subset(full_dataset, range(min(8000, len(full_dataset))))
        if gpu == 0:
            print(f"Dataset loaded with {len(dataset)} samples ")
    except Exception as e:
        print(f"GPU {gpu}: Failed to load dataset: {e}")
        return
    
    # 使用DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=ngpus_per_node, rank=gpu)
    
    # 根据GPU数量调整batch size
    base_batch_size = min(config['batch_size'], 24)
    effective_batch_size = base_batch_size * ngpus_per_node
    
    dataloader = DataLoader(
        dataset, 
        batch_size=base_batch_size,  # 每个GPU的batch size
        shuffle=False,  # DistributedSampler会处理shuffle
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型
    print(f"GPU {gpu}: Initializing model...")
    model = ImprovedConditionalUNet(in_channels=3, width=config.get('model_width', 64)) 
    model = model.to(device)
    
    # 使用DistributedDataParallel
    model = DDP(model, device_ids=[gpu])
    print(f"GPU {gpu}: Model parallelized with DDP")
    
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
    
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler('cuda')

    if gpu == 0:
        print(f"🚀 开始训练 ({config['num_epochs']} epochs)...")
    
    # 用于记录训练损失
    train_losses = []
    
    # 固定一批验证样本用于可视化（只在主GPU上）
    val_sample = None
    val_mask = None

    # 训练循环
    for epoch in range(config['num_epochs']):
        sampler.set_epoch(epoch)  # 重要：确保每个epoch数据shuffle
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 遍历所有batch
        for i, (masked, mask, clean) in enumerate(dataloader):
            # 确保所有张量都在相同设备上
            masked, clean, mask = masked.to(device), clean.to(device), mask.to(device)

            # 在第一个epoch保存验证样本（只在主GPU上）
            if gpu == 0 and epoch == 0 and val_sample is None:
                val_sample = (masked[:4].clone(), clean[:4].clone())
                val_mask = mask[:4].clone()
            
            # 检查输入是否有效
            if torch.isnan(masked).any() or torch.isnan(clean).any():
                print(f"GPU {gpu}: NaN values detected in batch {i}, skipping...")
                continue
                
            try:
                # 使用自动混合精度
                with autocast(device_type='cuda'):
                    loss = flow_matching_loss(model, clean, masked)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"GPU {gpu}: Invalid loss value in batch {i}, skipping...")
                    continue
                    
                # 反向传播使用缩放器
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新权重
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 记录每个batch的损失（只在主GPU上）
                if gpu == 0 and writer is not None:
                    writer.add_scalar('Batch/Loss', loss.item(), epoch * len(dataloader) + i)
                
                # 打印进度（每10个batch打印一次以节省输出，只在主GPU上）
                if gpu == 0 and (i + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{config['num_epochs']}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"GPU {gpu}: Error in batch {i}: {e}")
                print("Skipping this batch...")
                torch.cuda.empty_cache()
                continue

        # 计算平均损失
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            
            # 记录epoch级别损失和学习率（只在主GPU上）
            if gpu == 0 and writer is not None:
                writer.add_scalar('Epoch/Loss', avg_loss, epoch)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            # 更新学习率
            scheduler.step()
            
            if gpu == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. Avg Loss: {avg_loss:.4f}")

        # 定期保存检查点和可视化结果（只在主GPU上）
        if gpu == 0 and (epoch + 1) % 5 == 0:
            # 保存模型检查点
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # 使用module获取原始模型
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # 生成可视化结果
            model.eval()  # 切换到评估模式
            with torch.no_grad():
                try:
                    # 保存当前batch的可视化结果
                    save_inpainting_result(
                        model.module,  # 使用module获取原始模型
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
                            model.module,  # 使用module获取原始模型
                            (val_masked, val_clean),
                            val_mask,
                            device,
                            f"{config['results_dir']}/fixed_sample_epoch_{epoch+1}.png"
                        )
                    
                except Exception as e:
                    print(f"Failed to save visualization: {e}")
            model.train()  # 恢复训练模式

        # 显存清理
        torch.cuda.empty_cache()

    # 保存最终模型（只在主GPU上）
    if gpu == 0:
        try:
            final_model_path = config.get('model_save_path', "checkpoints/final_model.pth")
            torch.save(model.module.state_dict(), final_model_path)  # 使用module获取原始模型
            print(f"✅ 训练完成，最终模型已保存至 {final_model_path}")
        except Exception as e:
            print(f"Failed to save final model: {e}")

        if writer is not None:
            writer.close()
        print("Training completed and TensorBoard logs saved.")

def main():
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取GPU数量
    ngpus_per_node = torch.cuda.device_count()
    print(f"Available GPUs: {ngpus_per_node}")
    
    if ngpus_per_node > 1:
        # 使用多GPU DDP训练
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # 单GPU训练
        main_worker(0, 1, config)

if __name__ == "__main__":
    main()