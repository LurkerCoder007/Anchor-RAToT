"""
搜索引擎集成

将对比推理路径评估器集成到搜索引擎中，
提高搜索结果的可解释性和准确性。
"""

class ExplainableSearchEngine:
    """可解释搜索引擎"""
    
    def __init__(self, retriever, path_generator, path_evaluator):
        self.retriever = retriever
        self.path_generator = path_generator
        self.path_evaluator = path_evaluator
    
    def search(self, query, top_k=10):
        """搜索并返回带解释的结果"""
        # 检索相关文档
        documents = self.retriever.retrieve(query, top_k=top_k*2)
        
        results = []
        for doc in documents:
            # 生成解释路径
            path = self.path_generator.generate(query, doc)
            
            # 评估路径质量
            score = self.path_evaluator.evaluate(path)
            
            results.append({
                "document": doc,
                "explanation_path": path,
                "confidence": score
            })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results[:top_k]