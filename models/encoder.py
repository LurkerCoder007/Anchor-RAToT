import torch
import torch.nn as nn
from transformers import AutoModel

class ReasoningPathEncoder(nn.Module):
    """推理路径编码器，将推理路径编码为嵌入向量"""
    
    def __init__(self, model_name="bert-base-uncased", projection_dim=128):
        """
        初始化编码器
        
        Args:
            model_name: 预训练模型名称
            projection_dim: 投影维度
        """
        super(ReasoningPathEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # 投影头，将编码器输出映射到对比空间
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, projection_dim)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """前向传播"""
        # 编码输入
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 使用[CLS]标记的表示作为序列表示
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # 投影到对比空间
        projected = self.projection(sequence_output)
        
        # 归一化嵌入向量
        normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
        
        return normalized 