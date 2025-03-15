import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
import wandb
from .losses import DynamicContrastiveLoss, SupConLoss, TripletLoss
from .evaluator import Evaluator

class ContrastiveTrainer:
    """对比学习训练器，实现两阶段训练策略"""
    
    def __init__(self, model, train_loader, val_loader=None, 
                 optimizer=None, scheduler=None, contrastive_loss=None,
                 device="cuda", log_dir="logs", checkpoint_dir="checkpoints",
                 use_wandb=False):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            contrastive_loss: 对比损失函数
            device: 设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            use_wandb: 是否使用wandb记录实验
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化损失函数
        self.contrastive_loss = contrastive_loss if contrastive_loss else DynamicContrastiveLoss()
        
        # 初始化优化器
        if optimizer is None:
            # 区分参数组，对编码器和生成器使用不同的学习率
            encoder_params = list(model.discriminator.parameters())
            generator_params = list(model.generator.parameters())
            
            self.optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': 3e-5},
                {'params': generator_params, 'lr': 5e-5}
            ])
        else:
            self.optimizer = optimizer
            
        # 初始化学习率调度器
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10)
        else:
            self.scheduler = scheduler
            
        # 初始化评估器
        self.evaluator = Evaluator(model, device=device)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        
    def train_epoch(self, stage="pretrain"):
        """
        训练一个epoch
        
        Args:
            stage: 训练阶段，"pretrain"或"finetune"
        """
        self.model.train()
        epoch_loss = 0
        epoch_gen_loss = 0
        epoch_contrast_loss = 0
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} ({stage})")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备上
            pos_input_ids = batch['pos_input_ids'].to(self.device)
            pos_attention_mask = batch['pos_attention_mask'].to(self.device)
            pos_token_type_ids = batch['pos_token_type_ids'].to(self.device)
            
            # 处理负样本
            neg_input_ids_list = []
            neg_attention_mask_list = []
            neg_token_type_ids_list = []
            
            for neg_encoding in batch['neg_encodings']:
                neg_input_ids_list.append(neg_encoding['input_ids'].to(self.device))
                neg_attention_mask_list.append(neg_encoding['attention_mask'].to(self.device))
                neg_token_type_ids_list.append(neg_encoding['token_type_ids'].to(self.device))
                
            neg_input_ids = torch.stack(neg_input_ids_list, dim=0)
            neg_attention_mask = torch.stack(neg_attention_mask_list, dim=0)
            neg_token_type_ids = torch.stack(neg_token_type_ids_list, dim=0)
            
            # 根据训练阶段决定是否冻结部分模型
            if stage == "pretrain":
                # 预训练阶段：冻结生成器，只训练判别器
                for param in self.model.generator.parameters():
                    param.requires_grad = False
                for param in self.model.discriminator.parameters():
                    param.requires_grad = True
                    
                # 只计算对比损失
                pos_embeddings, neg_embeddings = self.model.discriminator(
                    pos_input_ids, pos_attention_mask, pos_token_type_ids,
                    neg_input_ids.view(-1, neg_input_ids.size(-1)), 
                    neg_attention_mask.view(-1, neg_attention_mask.size(-1)),
                    neg_token_type_ids.view(-1, neg_token_type_ids.size(-1))
                )
                
                # 计算对比损失
                batch_size = pos_embeddings.size(0)
                neg_count = neg_embeddings.size(0) // batch_size
                
                # 重塑负样本嵌入以匹配批次结构
                neg_embeddings = neg_embeddings.view(batch_size, neg_count, -1)
                
                # 为每个正样本选择最难的负样本（最相似的）
                hardest_negatives = []
                for i in range(batch_size):
                    similarities = torch.matmul(
                        pos_embeddings[i:i+1], 
                        neg_embeddings[i].transpose(0, 1)
                    )
                    hardest_idx = similarities.argmax().item()
                    hardest_negatives.append(neg_embeddings[i, hardest_idx])
                
                hardest_negatives = torch.stack(hardest_negatives)
                
                # 计算三元组损失
                triplet_loss = TripletLoss(margin=1.0)
                contrast_loss = triplet_loss(
                    pos_embeddings, 
                    pos_embeddings,  # 自身作为正样本
                    hardest_negatives
                )
                
                # 在预训练阶段，总损失就是对比损失
                loss = contrast_loss
                gen_loss = torch.tensor(0.0).to(self.device)
                
            else:  # finetune阶段
                # 微调阶段：联合训练生成器和判别器
                for param in self.model.generator.parameters():
                    param.requires_grad = True
                for param in self.model.discriminator.parameters():
                    param.requires_grad = True
                
                # 准备生成器输入
                input_ids = batch.get('input_ids', pos_input_ids).to(self.device)
                attention_mask = batch.get('attention_mask', pos_attention_mask).to(self.device)
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pos_input_ids=pos_input_ids,
                    pos_attention_mask=pos_attention_mask,
                    pos_token_type_ids=pos_token_type_ids,
                    neg_input_ids=neg_input_ids.view(-1, neg_input_ids.size(-1)),
                    neg_attention_mask=neg_attention_mask.view(-1, neg_attention_mask.size(-1)),
                    neg_token_type_ids=neg_token_type_ids.view(-1, neg_token_type_ids.size(-1))
                )
                
                # 获取损失
                loss = outputs['loss']
                gen_loss = outputs['generator_loss']
                contrast_loss = outputs.get('contrastive_loss', torch.tensor(0.0).to(self.device))
                
                # 动态调整对比损失权重
                if isinstance(self.contrastive_loss, DynamicContrastiveLoss):
                    self.contrastive_loss.update_weights(
                        self.global_step, 
                        len(self.train_loader) * 5  # 假设总共5个epoch
                    )
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'gen_loss': gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0,
                'contrast_loss': contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0
            })
            
            # 累计损失
            epoch_loss += loss.item()
            epoch_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0
            epoch_contrast_loss += contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/gen_loss': gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0,
                    'train/contrast_loss': contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0,
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
                
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        avg_gen_loss = epoch_gen_loss / len(self.train_loader)
        avg_contrast_loss = epoch_contrast_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'gen_loss': avg_gen_loss,
            'contrast_loss': avg_contrast_loss
        }
    
    def validate(self):
        """验证模型性能"""
        # 使用评估器评估模型
        metrics = self.evaluator.evaluate(self.val_loader)
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log({
                'val/drr': metrics['document_retention_rate'],
                'val/em': metrics['exact_match'],
                'val/f1': metrics['f1'],
                'val/contrast_acc': metrics['contrast_accuracy']
            }, step=self.global_step)
            
        return metrics
    
    def train(self, pretrain_epochs=10, finetune_epochs=5):
        """
        训练模型
        
        Args:
            pretrain_epochs: 预训练轮次
            finetune_epochs: 微调轮次
        """
        self.logger.info("开始训练")
        
        # 预训练阶段
        self.logger.info(f"预训练阶段: {pretrain_epochs} 轮")
        for epoch in range(pretrain_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(stage="pretrain")
            
            self.logger.info(
                f"预训练 Epoch {epoch}: "
                f"Loss={train_metrics['loss']:.4f}, "
                f"Contrast Loss={train_metrics['contrast_loss']:.4f}"
            )
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.logger.info(
                    f"验证: DRR={val_metrics['document_retention_rate']:.4f}, "
                    f"EM={val_metrics['exact_match']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}, "
                    f"Contrast Acc={val_metrics['contrast_accuracy']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics['document_retention_rate'] > self.best_metric:
                    self.best_metric = val_metrics['document_retention_rate']
                    self.save_checkpoint(os.path.join(
                        self.checkpoint_dir, 'best_pretrain_model.pt'))
            
            # 保存检查点
            self.save_checkpoint(os.path.join(
                self.checkpoint_dir, f'pretrain_epoch_{epoch}.pt'))
                
        # 微调阶段
        self.logger.info(f"微调阶段: {finetune_epochs} 轮")
        self.best_metric = float('-inf')  # 重置最佳指标
        
        for epoch in range(finetune_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(stage="finetune")
            
            self.logger.info(
                f"微调 Epoch {epoch}: "
                f"Loss={train_metrics['loss']:.4f}, "
                f"Gen Loss={train_metrics['gen_loss']:.4f}, "
                f"Contrast Loss={train_metrics['contrast_loss']:.4f}"
            )
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.logger.info(
                    f"验证: DRR={val_metrics['document_retention_rate']:.4f}, "
                    f"EM={val_metrics['exact_match']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}, "
                    f"Contrast Acc={val_metrics['contrast_accuracy']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics['document_retention_rate'] > self.best_metric:
                    self.best_metric = val_metrics['document_retention_rate']
                    self.save_checkpoint(os.path.join(
                        self.checkpoint_dir, 'best_finetune_model.pt'))
            
            # 保存检查点
            self.save_checkpoint(os.path.join(
                self.checkpoint_dir, f'finetune_epoch_{epoch}.pt'))
                
        self.logger.info("训练完成")
        
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }, path)
        self.logger.info(f"检查点已保存到 {path}")
        
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.logger.info(f"检查点已从 {path} 加载") 