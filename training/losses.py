import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    监督对比损失函数
    参考: Supervised Contrastive Learning (Khosla et al., 2020)
    """
    
    def __init__(self, temperature=0.07):
        """
        初始化
        
        Args:
            temperature: 温度参数，控制分布的平滑程度
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels=None, mask=None):
        """
        计算对比损失
        
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
            mask: 掩码，指示哪些样本对应考虑 [batch_size, batch_size]
            
        Returns:
            对比损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        if labels is not None and mask is None:
            # 如果提供了标签但没有掩码，则根据标签创建掩码
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # 默认情况下，每个样本只与自身为正样本
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            
        # 计算特征之间的相似度
        contrast_feature = features
        anchor_feature = features
        
        # 计算点积
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # 对角线上的元素是自身相似度，应该排除
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        mask = mask * logits_mask
        
        # 计算正样本对的数量
        pos_per_sample = mask.sum(1)
        pos_per_sample = torch.clamp(pos_per_sample, min=1e-8)
        
        # 计算对比损失
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算正样本对的平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample
        
        # 损失是负的log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss

class TripletLoss(nn.Module):
    """
    三元组损失函数
    """
    
    def __init__(self, margin=1.0):
        """
        初始化
        
        Args:
            margin: 边界值
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        
        Args:
            anchor: 锚点样本
            positive: 正样本
            negative: 负样本
            
        Returns:
            三元组损失
        """
        distance_pos = torch.sum((anchor - positive) ** 2, dim=1)
        distance_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        losses = torch.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()

class DynamicContrastiveLoss(nn.Module):
    """
    动态对比损失，根据训练进度调整权重
    """
    
    def __init__(self, sup_loss=None, triplet_loss=None, 
                 sup_weight=0.5, triplet_weight=0.5,
                 annealing_strategy="linear"):
        """
        初始化
        
        Args:
            sup_loss: 监督对比损失
            triplet_loss: 三元组损失
            sup_weight: 监督对比损失权重
            triplet_weight: 三元组损失权重
            annealing_strategy: 权重调整策略
        """
        super(DynamicContrastiveLoss, self).__init__()
        self.sup_loss = sup_loss if sup_loss else SupConLoss()
        self.triplet_loss = triplet_loss if triplet_loss else TripletLoss()
        
        self.sup_weight = sup_weight
        self.triplet_weight = triplet_weight
        self.annealing_strategy = annealing_strategy
        
        self.current_step = 0
        self.total_steps = 1000  # 默认值，可在训练中更新
        
    def update_weights(self, current_step, total_steps):
        """更新损失权重"""
        self.current_step = current_step
        self.total_steps = total_steps
        
        if self.annealing_strategy == "linear":
            # 线性调整权重
            progress = current_step / total_steps
            self.sup_weight = 0.3 + 0.4 * progress  # 从0.3增加到0.7
            self.triplet_weight = 0.7 - 0.4 * progress  # 从0.7减少到0.3
            
    def forward(self, anchor, positive, negative, labels=None):
        """
        计算动态对比损失
        
        Args:
            anchor: 锚点样本
            positive: 正样本
            negative: 负样本
            labels: 标签
            
        Returns:
            加权对比损失
        """
        # 计算监督对比损失
        features = torch.cat([anchor.unsqueeze(1), positive.unsqueeze(1)], dim=1)
        features = features.view(-1, features.shape[-1])
        
        if labels is not None:
            # 创建新标签，使得每对正样本共享同一标签
            new_labels = torch.repeat_interleave(torch.arange(anchor.shape[0]), 2)
            sup_loss = self.sup_loss(features, new_labels)
        else:
            sup_loss = self.sup_loss(features)
            
        # 计算三元组损失
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        
        # 加权组合
        total_loss = self.sup_weight * sup_loss + self.triplet_weight * triplet_loss
        
        return total_loss, sup_loss, triplet_loss 