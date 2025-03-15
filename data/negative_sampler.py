import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re

class AdversarialNegativeSampler:
    """对抗性负样本生成器，通过扰动正确推理路径生成误导性样本"""
    
    def __init__(self, tokenizer_name="bert-base-uncased", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(tokenizer_name).to(device)
        self.device = device
        self.model.eval()
    
    def get_entity_embeddings(self, texts):
        """获取文本中实体的嵌入表示"""
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            # 使用[CLS]表示作为文本嵌入
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings
    
    def entity_replacement(self, correct_path, retrieval_pool, replace_ratio=0.3):
        """
        实体替换策略：用检索池中的相似但矛盾的实体替换正确路径中的关键实体
        
        Args:
            correct_path: 正确推理路径，包含文档和推理步骤
            retrieval_pool: 检索池中的文档列表
            replace_ratio: 替换比例
        
        Returns:
            扰动后的推理路径
        """
        # 提取正确路径中的关键实体
        path_entities = self._extract_entities(correct_path)
        if not path_entities:
            return correct_path
        
        # 从检索池中提取候选实体
        pool_entities = []
        for doc in retrieval_pool:
            pool_entities.extend(self._extract_entities(doc))
        
        if not pool_entities:
            return correct_path
        
        # 计算实体嵌入
        all_entities = path_entities + pool_entities
        embeddings = self.get_entity_embeddings(all_entities)
        
        path_embeds = embeddings[:len(path_entities)]
        pool_embeds = embeddings[len(path_entities):]
        
        # 为每个路径实体找到语义相似但不同的实体
        replacements = {}
        for i, path_entity in enumerate(path_entities):
            # 计算相似度
            similarities = np.dot(path_embeds[i], pool_embeds.T)
            # 选择相似度适中的实体（不要太相似也不要太不相似）
            candidates = [(pool_entities[j], similarities[j]) 
                          for j in range(len(pool_entities)) 
                          if 0.5 < similarities[j] < 0.8]
            
            if candidates:
                # 按相似度排序并选择
                candidates.sort(key=lambda x: x[1], reverse=True)
                replacements[path_entity] = candidates[0][0]
        
        # 执行替换
        perturbed_path = correct_path
        num_to_replace = max(1, int(len(replacements) * replace_ratio))
        for entity, replacement in list(replacements.items())[:num_to_replace]:
            perturbed_path = perturbed_path.replace(entity, replacement)
            
        return perturbed_path
    
    def logic_perturbation(self, correct_path, steps_delimiter="\n"):
        """
        逻辑扰动策略：颠倒推理步骤顺序或删除必要前提
        
        Args:
            correct_path: 正确推理路径，包含推理步骤
            steps_delimiter: 步骤分隔符
        
        Returns:
            扰动后的推理路径
        """
        # 分割推理步骤
        steps = correct_path.split(steps_delimiter)
        if len(steps) <= 1:
            return correct_path
            
        # 随机选择扰动类型
        perturbation_type = random.choice(["reorder", "skip"])
        
        if perturbation_type == "reorder" and len(steps) >= 3:
            # 颠倒相邻步骤顺序
            idx = random.randint(0, len(steps) - 2)
            steps[idx], steps[idx + 1] = steps[idx + 1], steps[idx]
        elif perturbation_type == "skip" and len(steps) >= 3:
            # 删除中间步骤（保留首尾）
            idx = random.randint(1, len(steps) - 2)
            steps.pop(idx)
            
        return steps_delimiter.join(steps)
    
    def document_noise(self, correct_path, retrieval_pool, doc_markers=None):
        """
        文档噪声模拟：用检索到的无关文档替换正确路径中的关键文档
        
        Args:
            correct_path: 正确推理路径，包含文档引用
            retrieval_pool: 检索池中的文档列表
            doc_markers: 文档标记格式，如["[Doc", "]"]
        
        Returns:
            扰动后的推理路径
        """
        if not doc_markers:
            doc_markers = ["[Doc", "]"]
            
        # 提取文档引用
        start_marker, end_marker = doc_markers
        doc_start_indices = [i for i in range(len(correct_path)) 
                            if correct_path[i:i+len(start_marker)] == start_marker]
        
        if not doc_start_indices or not retrieval_pool:
            return correct_path
            
        # 随机选择一个文档引用进行替换
        idx = random.choice(doc_start_indices)
        end_idx = correct_path.find(end_marker, idx) + len(end_marker)
        
        if end_idx <= idx:
            return correct_path
            
        # 随机选择检索池中的文档
        replacement_doc = random.choice(retrieval_pool)
        # 格式化为文档引用格式
        formatted_doc = f"{start_marker}{len(retrieval_pool)}{end_marker}"
        
        # 替换文档引用
        perturbed_path = correct_path[:idx] + formatted_doc + correct_path[end_idx:]
        
        return perturbed_path
    
    def generate_llm_adversarial_sample(self, correct_path, question=None):
        """
        使用大型语言模型生成对抗性负样本
        
        Args:
            correct_path: 正确的推理路径
            question: 原始问题
            
        Returns:
            对抗性负样本
        """
        # 如果没有初始化LLM，则使用简单的替代方法
        if not hasattr(self, 'llm') or self.llm is None:
            return self.generate_negative_sample(correct_path, [], "mixed")
        
        # 构建提示
        prompt = f"""
        问题: {question if question else '未提供问题'}
        
        正确的推理路径:
        {correct_path}
        
        请生成一个看起来合理但实际包含错误的推理路径。这个路径应该:
        1. 保持与原始路径相似的结构和风格
        2. 包含微妙的逻辑错误或事实错误
        3. 看起来很有说服力，难以立即识别出错误
        
        错误的推理路径:
        """
        
        # 调用LLM生成对抗样本
        try:
            response = self.llm(prompt, max_tokens=len(correct_path.split()) * 2)
            adversarial_path = response.strip()
            return adversarial_path
        except Exception as e:
            print(f"LLM生成对抗样本失败: {e}")
            # 回退到基于规则的方法
            return self.generate_negative_sample(correct_path, [], "mixed")

    def generate_counterfactual_sample(self, correct_path):
        """
        生成反事实负样本，保持推理结构但改变关键事实
        
        Args:
            correct_path: 正确的推理路径
            
        Returns:
            反事实负样本
        """
        # 分割推理步骤
        steps = correct_path.split('.')
        
        # 选择要修改的步骤
        if len(steps) <= 1:
            return self.generate_negative_sample(correct_path, [], "entity")
        
        # 选择1-3个步骤进行修改
        num_steps_to_modify = min(random.randint(1, 3), len(steps))
        steps_to_modify = random.sample(range(len(steps)), num_steps_to_modify)
        
        # 修改选定的步骤
        for i in steps_to_modify:
            if not steps[i].strip():
                continue
            
            # 提取数字并反转
            numbers = re.findall(r'\d+', steps[i])
            for num in numbers:
                if random.random() < 0.7:  # 70%的概率修改数字
                    # 生成一个不同的数字
                    new_num = str(int(num) + random.randint(1, 10))
                    steps[i] = steps[i].replace(num, new_num)
            
            # 添加否定词或移除否定词
            if "not" in steps[i] or "n't" in steps[i]:
                if random.random() < 0.7:
                    steps[i] = steps[i].replace("not ", "").replace("n't", "")
            else:
                words = steps[i].split()
                if len(words) > 2:
                    verb_pos = random.randint(0, len(words)-1)
                    if random.random() < 0.3:  # 30%的概率添加否定词
                        words.insert(verb_pos, "not")
                        steps[i] = " ".join(words)
        
        # 重新组合推理路径
        counterfactual_path = '.'.join(steps)
        return counterfactual_path

    def generate_negative_sample(self, correct_path, retrieval_pool=None, strategy=None):
        """
        生成负样本
        
        Args:
            correct_path: 正确推理路径
            retrieval_pool: 检索池
            strategy: 负样本生成策略，可选["entity", "logic", "document", "mixed", "counterfactual", "llm_adversarial"]
        
        Returns:
            生成的负样本
        """
        if strategy is None:
            strategy = random.choice(["entity", "logic", "document", "mixed", 
                                     "counterfactual", "llm_adversarial"])
        
        if strategy == "entity":
            return self.entity_replacement(correct_path, retrieval_pool)
        elif strategy == "logic":
            return self.logic_perturbation(correct_path)
        elif strategy == "document":
            return self.document_noise(correct_path, retrieval_pool)
        elif strategy == "counterfactual":
            return self.generate_counterfactual_sample(correct_path)
        elif strategy == "llm_adversarial":
            return self.generate_llm_adversarial_sample(correct_path)
        elif strategy == "mixed":
            # 组合多种扰动策略
            perturbed = self.entity_replacement(correct_path, retrieval_pool)
            perturbed = self.logic_perturbation(perturbed)
            return perturbed
        else:
            raise ValueError(f"Unknown negative sampling strategy: {strategy}")
    
    def _extract_entities(self, text):
        """简单的实体提取（实际应用中可使用NER工具）"""
        # 这里使用简化版，实际应用中应使用NER工具
        words = text.split()
        # 假设大写开头的词可能是实体
        entities = [word for word in words if word and word[0].isupper()]
        return list(set(entities)) 