import random
import numpy as np
import copy
import re

def add_noise_to_retrieval_pool(original_pool, noise_ratio=0.3, seed=42):
    """
    向检索池添加噪声文档
    
    Args:
        original_pool: 原始检索池文档列表
        noise_ratio: 噪声比例 (0.0-1.0)
        seed: 随机种子
        
    Returns:
        添加噪声后的检索池
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 复制原始检索池
    noisy_pool = copy.deepcopy(original_pool)
    
    # 计算要添加的噪声文档数量
    original_size = len(original_pool)
    noise_size = int(original_size * noise_ratio / (1 - noise_ratio))
    
    # 生成噪声文档
    noise_docs = []
    for i in range(noise_size):
        noise_type = random.choice(["mix", "entity_swap", "irrelevant"])
        
        if noise_type == "mix" and len(original_pool) >= 2:
            # 混合两个文档的内容
            doc1, doc2 = random.sample(original_pool, 2)
            sentences1 = doc1.split('.')
            sentences2 = doc2.split('.')
            
            # 交替选择句子
            mixed_sentences = []
            for j in range(max(len(sentences1), len(sentences2))):
                if j < len(sentences1) and sentences1[j].strip():
                    mixed_sentences.append(sentences1[j])
                if j < len(sentences2) and sentences2[j].strip():
                    mixed_sentences.append(sentences2[j])
                    
            noise_doc = '.'.join(mixed_sentences)
            noise_docs.append(noise_doc)
            
        elif noise_type == "entity_swap" and original_pool:
            # 替换实体
            doc = random.choice(original_pool)
            
            # 简单的实体提取和替换
            entities = re.findall(r'\b[A-Z][a-z]+\b', doc)
            if len(entities) >= 2:
                unique_entities = list(set(entities))
                if len(unique_entities) >= 2:
                    entity1, entity2 = random.sample(unique_entities, 2)
                    noise_doc = doc.replace(entity1, "TEMP_ENTITY")
                    noise_doc = noise_doc.replace(entity2, entity1)
                    noise_doc = noise_doc.replace("TEMP_ENTITY", entity2)
                    noise_docs.append(noise_doc)
            
        elif noise_type == "irrelevant":
            # 生成完全不相关的文档
            irrelevant_topics = [
                "The solar system consists of the Sun and eight planets.",
                "Machine learning is a subset of artificial intelligence.",
                "The Great Wall of China is over 13,000 miles long.",
                "The human genome contains approximately 3 billion base pairs.",
                "The Industrial Revolution began in Great Britain in the 18th century."
            ]
            noise_doc = random.choice(irrelevant_topics)
            noise_docs.append(noise_doc)
    
    # 添加噪声文档到检索池
    noisy_pool.extend(noise_docs)
    
    return noisy_pool

def create_noisy_datasets(dataset_class, original_dataset, retrieval_pool, 
                          noise_levels=[0.1, 0.3, 0.5, 0.7], **dataset_args):
    """
    创建不同噪声级别的数据集
    
    Args:
        dataset_class: 数据集类
        original_dataset: 原始数据集
        retrieval_pool: 原始检索池
        noise_levels: 噪声级别列表
        dataset_args: 数据集初始化参数
        
    Returns:
        不同噪声级别的数据集字典 {noise_level: dataset}
    """
    noisy_datasets = {}
    
    # 对每个噪声级别创建数据集
    for noise_level in noise_levels:
        # 生成带噪声的检索池
        noisy_pool = add_noise_to_retrieval_pool(
            retrieval_pool, noise_ratio=noise_level)
        
        # 创建新的数据集实例，使用带噪声的检索池
        noisy_dataset = copy.deepcopy(original_dataset)
        noisy_dataset.retrieval_pool = noisy_pool
        
        noisy_datasets[noise_level] = noisy_dataset
        
    return noisy_datasets 