import torch
import torch.nn as nn
from .encoder import ReasoningPathEncoder

class ContrastivePathDiscriminator(nn.Module):
    """对比推理路径判别器，区分正确路径与误导路径"""
    
    def __init__(self, model_name="bert-base-uncased", projection_dim=128):
        """
        初始化对比判别器
        
        Args:
            model_name: 预训练模型名称
            projection_dim: 投影维度
        """
        super(ContrastivePathDiscriminator, self).__init__()
        self.encoder = ReasoningPathEncoder(model_name, projection_dim)
        
    def forward(self, pos_input_ids, pos_attention_mask, pos_token_type_ids=None,
                neg_input_ids=None, neg_attention_mask=None, neg_token_type_ids=None):
        """
        前向传播
        
        Args:
            pos_input_ids: 正样本输入ID
            pos_attention_mask: 正样本注意力掩码
            pos_token_type_ids: 正样本token类型ID
            neg_input_ids: 负样本输入ID
            neg_attention_mask: 负样本注意力掩码
            neg_token_type_ids: 负样本token类型ID
            
        Returns:
            正样本嵌入和负样本嵌入
        """
        # 编码正样本
        pos_embeddings = self.encoder(
            pos_input_ids, 
            pos_attention_mask, 
            pos_token_type_ids
        )
        
        # 如果提供了负样本，则编码负样本
        neg_embeddings = None
        if neg_input_ids is not None:
            neg_embeddings = self.encoder(
                neg_input_ids, 
                neg_attention_mask, 
                neg_token_type_ids
            )
            
        return pos_embeddings, neg_embeddings
    
    def compute_similarity(self, emb1, emb2):
        """计算嵌入向量之间的余弦相似度"""
        return torch.matmul(emb1, emb2.t()) 