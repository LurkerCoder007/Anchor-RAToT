"""
教育应用

将对比推理路径评估器应用于教育场景，
帮助学生理解和改进推理过程。
"""

class ReasoningTutor:
    """推理辅导系统"""
    
    def __init__(self, path_evaluator, feedback_generator):
        self.path_evaluator = path_evaluator
        self.feedback_generator = feedback_generator
    
    def evaluate_student_reasoning(self, problem, student_solution):
        """评估学生的推理过程"""
        # 评估推理路径
        score = self.path_evaluator.evaluate(student_solution)
        
        # 生成反馈
        if score > 0.8:
            feedback = self.feedback_generator.generate_positive_feedback(student_solution)
        else:
            # 分析错误
            error_analysis = self.path_evaluator.analyze_errors(student_solution)
            feedback = self.feedback_generator.generate_corrective_feedback(
                student_solution, error_analysis)
        
        return {
            "score": score,
            "feedback": feedback
        }