import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
from .negative_sampler import AdversarialNegativeSampler

class ReasoningPathDataset(Dataset):
    """推理路径数据集，包含正样本和负样本生成"""
    
    def __init__(self, data_path, tokenizer, max_length=512, negative_ratio=1, 
                 sampler=None, retrieval_pool_path=None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            negative_ratio: 每个正样本生成的负样本数量
            sampler: 负样本生成器
            retrieval_pool_path: 检索池文件路径
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_ratio = negative_ratio
        
        # 加载检索池
        self.retrieval_pool = []
        if retrieval_pool_path and os.path.exists(retrieval_pool_path):
            with open(retrieval_pool_path, 'r') as f:
                self.retrieval_pool = json.load(f)
        
        # 初始化负样本生成器
        self.sampler = sampler if sampler else AdversarialNegativeSampler(
            tokenizer_name=tokenizer.name_or_path)
    
    def _load_data(self, data_path):
        """加载数据"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据样本，包括正样本和生成的负样本"""
        item = self.data[idx]
        question = item['question']
        correct_path = item['reasoning_path']
        
        # 生成负样本
        negative_paths = []
        for _ in range(self.negative_ratio):
            strategy = random.choice(["entity", "logic", "document", "mixed"])
            neg_path = self.sampler.generate_negative_sample(
                correct_path, self.retrieval_pool, strategy)
            negative_paths.append(neg_path)
        
        # 编码正样本
        pos_encoding = self.tokenizer(
            question, correct_path,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码负样本
        neg_encodings = []
        for neg_path in negative_paths:
            neg_encoding = self.tokenizer(
                question, neg_path,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            neg_encodings.append({
                'input_ids': neg_encoding['input_ids'].squeeze(0),
                'attention_mask': neg_encoding['attention_mask'].squeeze(0),
                'token_type_ids': neg_encoding.get('token_type_ids', torch.zeros_like(neg_encoding['input_ids'])).squeeze(0)
            })
        
        return {
            'question': question,
            'positive_path': correct_path,
            'negative_paths': negative_paths,
            'pos_input_ids': pos_encoding['input_ids'].squeeze(0),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(0),
            'pos_token_type_ids': pos_encoding.get('token_type_ids', torch.zeros_like(pos_encoding['input_ids'])).squeeze(0),
            'neg_encodings': neg_encodings
        }

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 