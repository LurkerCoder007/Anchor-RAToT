"""
自适应负样本生成器

根据模型当前状态动态调整负样本生成策略，
生成更具挑战性的负样本。
"""

class AdaptiveNegativeSampler:
    """自适应负样本生成器"""
    
    def __init__(self, base_sampler, model):
        self.base_sampler = base_sampler
        self.model = model
        self.strategy_weights = {
            "entity": 0.2,
            "logic": 0.2,
            "document": 0.2,
            "counterfactual": 0.2,
            "llm_adversarial": 0.2
        }
        self.strategy_scores = {k: 0.0 for k in self.strategy_weights}
        self.strategy_counts = {k: 0 for k in self.strategy_weights}
    
    def generate_negative_samples(self, correct_path, retrieval_pool, n_samples=3):
        """生成多个负样本"""
        samples = []
        strategies = []
        
        # 根据当前权重选择策略
        for _ in range(n_samples):
            strategy = self._sample_strategy()
            neg_sample = self.base_sampler.generate_negative_sample(
                correct_path, retrieval_pool, strategy)
            samples.append(neg_sample)
            strategies.append(strategy)
        
        return samples, strategies
    
    def update_strategy_weights(self, strategies, losses):
        """根据损失更新策略权重"""
        # 更新每种策略的得分
        for strategy, loss in zip(strategies, losses):
            self.strategy_scores[strategy] += loss.item()
            self.strategy_counts[strategy] += 1
        
        # 计算平均得分
        avg_scores = {k: self.strategy_scores[k] / max(1, self.strategy_counts[k])
                     for k in self.strategy_scores}
        
        # 更新权重（难度越高的策略权重越大）
        total_score = sum(avg_scores.values())
        if total_score > 0:
            self.strategy_weights = {k: v / total_score for k, v in avg_scores.items()}
    
    def _sample_strategy(self):
        """根据权重采样策略"""
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return np.random.choice(strategies, p=weights)