import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import re

def calculate_document_retention_rate(pred_citations, gold_citations):
    """
    计算文档保留率
    
    Args:
        pred_citations: 预测的文档引用列表
        gold_citations: 金标文档引用列表
        
    Returns:
        文档保留率
    """
    if not gold_citations:
        return 0.0
        
    total_correct = 0
    total_gold = 0
    
    for pred, gold in zip(pred_citations, gold_citations):
        # 计算正确引用的文档数
        correct = len(set(pred).intersection(set(gold)))
        total_correct += correct
        total_gold += len(gold)
        
    return total_correct / total_gold if total_gold > 0 else 0.0

def calculate_qa_metrics(predictions, ground_truths):
    """
    计算QA评估指标
    
    Args:
        predictions: 预测答案列表
        ground_truths: 标准答案列表
        
    Returns:
        包含EM和F1的字典
    """
    if not ground_truths:
        return {"exact_match": 0.0, "f1": 0.0}
        
    # 精确匹配
    exact_matches = [
        1 if pred.strip() == gt.strip() else 0
        for pred, gt in zip(predictions, ground_truths)
    ]
    em = sum(exact_matches) / len(exact_matches)
    
    # F1分数
    f1_scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = set(pred.lower().split())
        gt_tokens = set(gt.lower().split())
        
        if not gt_tokens:
            f1_scores.append(1.0 if not pred_tokens else 0.0)
            continue
            
        if not pred_tokens:
            f1_scores.append(0.0)
            continue
            
        common = len(gt_tokens.intersection(pred_tokens))
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(gt_tokens) if gt_tokens else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    return {"exact_match": em, "f1": avg_f1}

def calculate_contrast_metrics(pos_embeddings, neg_embeddings):
    """
    计算对比学习评估指标
    
    Args:
        pos_embeddings: 正样本嵌入
        neg_embeddings: 负样本嵌入
        
    Returns:
        包含对比学习指标的字典
    """
    if isinstance(pos_embeddings, torch.Tensor):
        pos_embeddings = pos_embeddings.detach().cpu().numpy()
    if isinstance(neg_embeddings, torch.Tensor):
        neg_embeddings = neg_embeddings.detach().cpu().numpy()
    
    # 计算余弦相似度矩阵
    pos_norm = np.linalg.norm(pos_embeddings, axis=1, keepdims=True)
    neg_norm = np.linalg.norm(neg_embeddings, axis=1, keepdims=True)
    
    pos_embeddings_normalized = pos_embeddings / pos_norm
    neg_embeddings_normalized = neg_embeddings / neg_norm
    
    # 正样本之间的相似度
    pos_pos_sim = np.matmul(pos_embeddings_normalized, pos_embeddings_normalized.T)
    
    # 正负样本之间的相似度
    pos_neg_sim = np.matmul(pos_embeddings_normalized, neg_embeddings_normalized.T)
    
    # 计算对比准确率
    # 对于每个正样本，检查它是否与所有其他正样本的相似度都高于与任何负样本的相似度
    n_pos = pos_embeddings.shape[0]
    n_neg = neg_embeddings.shape[0]
    
    # 创建标签和预测分数用于ROC-AUC计算
    labels = []
    scores = []
    
    # 对每个正样本
    for i in range(n_pos):
        # 与其他正样本的相似度
        for j in range(n_pos):
            if i != j:
                labels.append(1)
                scores.append(pos_pos_sim[i, j])
        
        # 与负样本的相似度
        for j in range(n_neg):
            labels.append(0)
            scores.append(pos_neg_sim[i, j])
    
    # 计算ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    
    # 计算PR-AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    # 计算对比准确率
    correct = 0
    total = 0
    
    for i in range(n_pos):
        for j in range(n_pos):
            if i != j:
                for k in range(n_neg):
                    if pos_pos_sim[i, j] > pos_neg_sim[i, k]:
                        correct += 1
                    total += 1
    
    contrast_accuracy = correct / total if total > 0 else 0.0
    
    return {
        "contrast_accuracy": contrast_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "avg_pos_pos_sim": np.mean(pos_pos_sim) - np.mean(np.diag(pos_pos_sim)),  # 排除自身相似度
        "avg_pos_neg_sim": np.mean(pos_neg_sim)
    }

def calculate_reasoning_complexity_metrics(reasoning_paths):
    """
    计算推理复杂度指标
    
    Args:
        reasoning_paths: 推理路径列表
        
    Returns:
        包含复杂度指标的字典
    """
    # 步骤数
    steps_count = []
    # 实体数
    entity_count = []
    # 关系数
    relation_count = []
    
    for path in reasoning_paths:
        # 计算步骤数
        steps = [s for s in path.split('.') if s.strip()]
        steps_count.append(len(steps))
        
        # 计算实体数（假设大写开头的词是实体）
        entities = re.findall(r'\b[A-Z][a-z]+\b', path)
        entity_count.append(len(set(entities)))
        
        # 计算关系数（简化为动词数量）
        # 这里使用一个简单的启发式方法，实际应用中可能需要NLP工具
        words = path.lower().split()
        relations = [w for w in words if w.endswith('ed') or w.endswith('ing')]
        relation_count.append(len(set(relations)))
    
    return {
        "avg_steps": np.mean(steps_count),
        "avg_entities": np.mean(entity_count),
        "avg_relations": np.mean(relation_count),
        "max_steps": max(steps_count),
        "complexity_score": np.mean([s * e for s, e in zip(steps_count, entity_count)])
    } 