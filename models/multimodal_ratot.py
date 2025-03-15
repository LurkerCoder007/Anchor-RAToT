"""
多模态推理路径评估器

扩展对比推理路径评估器，支持多模态推理路径，
包括文本、图像和表格数据。
"""

class MultimodalPathEncoder(nn.Module):
    """多模态推理路径编码器"""
    
    def __init__(self, text_encoder, image_encoder, table_encoder, fusion_dim=768):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.table_encoder = table_encoder
        
        # 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(text_encoder.config.hidden_size + 
                     image_encoder.config.hidden_size + 
                     table_encoder.config.hidden_size, 
                     fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, text_inputs, image_inputs=None, table_inputs=None):
        """前向传播"""
        # 编码文本
        text_outputs = self.text_encoder(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]
        
        # 编码图像（如果有）
        if image_inputs is not None:
            image_outputs = self.image_encoder(**image_inputs)
            image_embeddings = image_outputs.pooler_output
        else:
            image_embeddings = torch.zeros(text_embeddings.size(0), 
                                          self.image_encoder.config.hidden_size,
                                          device=text_embeddings.device)
        
        # 编码表格（如果有）
        if table_inputs is not None:
            table_outputs = self.table_encoder(**table_inputs)
            table_embeddings = table_outputs.pooler_output
        else:
            table_embeddings = torch.zeros(text_embeddings.size(0),
                                          self.table_encoder.config.hidden_size,
                                          device=text_embeddings.device)
        
        # 融合多模态特征
        combined = torch.cat([text_embeddings, image_embeddings, table_embeddings], dim=1)
        fused_embeddings = self.fusion(combined)
        
        return fused_embeddings