import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from .contrastive import ContrastivePathDiscriminator

class ContrastiveRAToT(nn.Module):
    """集成对比学习的检索增强思维树模型"""
    
    def __init__(self, 
                 generator_name="t5-base", 
                 discriminator_name="bert-base-uncased",
                 projection_dim=128,
                 alpha=0.5):
        """
        初始化模型
        
        Args:
            generator_name: 生成器模型名称
            discriminator_name: 判别器模型名称
            projection_dim: 投影维度
            alpha: 生成损失与对比损失的权重平衡系数
        """
        super(ContrastiveRAToT, self).__init__()
        
        # 生成器模型（用于生成推理路径）
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)
        
        # 对比判别器（用于评估推理路径）
        self.discriminator = ContrastivePathDiscriminator(discriminator_name, projection_dim)
        
        # 损失权重
        self.alpha = alpha
        
    def forward(self, 
                input_ids, attention_mask,
                decoder_input_ids=None,
                labels=None,
                pos_input_ids=None, pos_attention_mask=None, pos_token_type_ids=None,
                neg_input_ids=None, neg_attention_mask=None, neg_token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            decoder_input_ids: 解码器输入ID
            labels: 标签
            pos_input_ids: 正样本输入ID
            pos_attention_mask: 正样本注意力掩码
            pos_token_type_ids: 正样本token类型ID
            neg_input_ids: 负样本输入ID
            neg_attention_mask: 负样本注意力掩码
            neg_token_type_ids: 负样本token类型ID
            
        Returns:
            损失和生成的输出
        """
        # 生成器前向传播
        generator_outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        
        # 如果提供了对比样本，则计算对比损失
        contrastive_loss = None
        if pos_input_ids is not None and neg_input_ids is not None:
            pos_embeddings, neg_embeddings = self.discriminator(
                pos_input_ids, pos_attention_mask, pos_token_type_ids,
                neg_input_ids, neg_attention_mask, neg_token_type_ids
            )
            
            # 计算对比损失
            from training.losses import TripletLoss
            triplet_loss_fn = TripletLoss(margin=0.5)
            
            # 处理批次维度
            batch_size = pos_embeddings.size(0)
            if neg_embeddings.size(0) > batch_size:
                # 如果有多个负样本，重塑为 [batch_size, neg_count, dim]
                neg_count = neg_embeddings.size(0) // batch_size
                neg_embeddings = neg_embeddings.view(batch_size, neg_count, -1)
                
                # 对每个负样本计算损失并取平均
                contrastive_loss = 0
                for i in range(neg_count):
                    contrastive_loss += triplet_loss_fn(
                        pos_embeddings,  # 锚点
                        pos_embeddings,  # 正样本
                        neg_embeddings[:, i, :]  # 负样本
                    )
                contrastive_loss /= neg_count
            else:
                # 单个负样本情况
                contrastive_loss = triplet_loss_fn(
                    pos_embeddings,  # 锚点
                    pos_embeddings,  # 正样本
                    neg_embeddings   # 负样本
                )
        
        # 总损失 = 生成损失 + alpha * 对比损失
        loss = generator_outputs.loss
        if contrastive_loss is not None:
            loss = loss + self.alpha * contrastive_loss
            
        return {
            'loss': loss,
            'generator_loss': generator_outputs.loss,
            'contrastive_loss': contrastive_loss,
            'logits': generator_outputs.logits
        }
    
    def generate(self, input_ids, attention_mask, **kwargs):
        """生成推理路径"""
        return self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def evaluate_path(self, question_ids, question_mask, 
                      path_ids, path_mask, path_token_type_ids=None):
        """评估推理路径的质量"""
        # 编码问题和路径
        path_embedding = self.discriminator.encoder(
            path_ids, path_mask, path_token_type_ids
        )
        
        # 返回路径的嵌入表示，可用于后续排序
        return path_embedding 