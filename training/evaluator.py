import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re

class Evaluator:
    """模型评估器，计算各种评估指标"""
    
    def __init__(self, model, device="cuda"):
        """
        初始化评估器
        
        Args:
            model: 模型
            device: 设备
        """
        self.model = model
        self.device = device
        
    def evaluate(self, data_loader):
        """
        评估模型性能
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            包含各种指标的字典
        """
        self.model.eval()
        
        # 初始化指标
        document_citations = []
        gold_document_citations = []
        predictions = []
        ground_truths = []
        contrast_preds = []
        contrast_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # 获取输入数据
                input_ids = batch.get('input_ids', batch['pos_input_ids']).to(self.device)
                attention_mask = batch.get('attention_mask', batch['pos_attention_mask']).to(self.device)
                
                # 生成答案
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )
                
                # 解码生成的文本
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("t5-base")  # 假设使用T5
                generated_texts = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                
                # 提取文档引用
                for text in generated_texts:
                    # 假设文档引用格式为[Doc1], [Doc2]等
                    citations = re.findall(r'\[Doc\d+\]', text)
                    document_citations.append(citations)
                    
                # 获取金标文档引用（假设在batch中）
                if 'gold_citations' in batch:
                    gold_document_citations.extend(batch['gold_citations'])
                    
                # 获取答案和标准答案
                if 'answers' in batch:
                    ground_truths.extend(batch['answers'])
                    predictions.extend([
                        re.sub(r'\[Doc\d+\]', '', text).strip() 
                        for text in generated_texts
                    ])
                
                # 评估对比学习性能
                pos_input_ids = batch['pos_input_ids'].to(self.device)
                pos_attention_mask = batch['pos_attention_mask'].to(self.device)
                pos_token_type_ids = batch['pos_token_type_ids'].to(self.device)
                
                # 处理负样本
                for i, neg_encoding in enumerate(batch['neg_encodings']):
                    neg_input_ids = neg_encoding['input_ids'].to(self.device)
                    neg_attention_mask = neg_encoding['attention_mask'].to(self.device)
                    neg_token_type_ids = neg_encoding['token_type_ids'].to(self.device)
                    
                    # 获取嵌入
                    pos_embed = self.model.discriminator.encoder(
                        pos_input_ids, pos_attention_mask, pos_token_type_ids)
                    neg_embed = self.model.discriminator.encoder(
                        neg_input_ids, neg_attention_mask, neg_token_type_ids)
                    
                    # 计算相似度
                    similarity = torch.sum(pos_embed * neg_embed, dim=1)
                    
                    # 预测：相似度低于阈值的为负样本
                    threshold = 0.5
                    preds = (similarity < threshold).int().cpu().numpy()
                    
                    # 所有样本的标签都是1（负样本）
                    labels = np.ones_like(preds)
                    
                    contrast_preds.extend(preds)
                    contrast_labels.extend(labels)
        
        # 计算文档保留率
        drr = self._calculate_document_retention_rate(
            document_citations, gold_document_citations)
        
        # 计算QA准确率
        em, f1 = self._calculate_qa_accuracy(predictions, ground_truths)
        
        # 计算对比学习准确率
        contrast_acc = accuracy_score(contrast_labels, contrast_preds)
        
        return {
            'document_retention_rate': drr,
            'exact_match': em,
            'f1': f1,
            'contrast_accuracy': contrast_acc
        }
    
    def _calculate_document_retention_rate(self, pred_citations, gold_citations):
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
    
    def _calculate_qa_accuracy(self, predictions, ground_truths):
        """
        计算QA准确率
        
        Args:
            predictions: 预测答案列表
            ground_truths: 标准答案列表
            
        Returns:
            精确匹配率和F1分数
        """
        if not ground_truths:
            return 0.0, 0.0
            
        # 精确匹配
        exact_matches = [
            1 if pred.strip() == gt.strip() else 0
            for pred, gt in zip(predictions, ground_truths)
        ]
        em = sum(exact_matches) / len(exact_matches)
        
        # F1分数（简化版，基于词重叠）
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
            precision = common / len(pred_tokens)
            recall = common / len(gt_tokens)
            
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
            
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        return em, avg_f1
    
    def evaluate_noise_robustness(self, data_loaders_with_noise):
        """
        评估噪声鲁棒性
        
        Args:
            data_loaders_with_noise: 包含不同噪声级别的数据加载器字典
                格式: {noise_level: data_loader}
                
        Returns:
            不同噪声级别下的性能指标
        """
        results = {}
        
        for noise_level, loader in data_loaders_with_noise.items():
            metrics = self.evaluate(loader)
            results[noise_level] = metrics
            
        return results 