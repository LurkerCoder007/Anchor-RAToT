import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import torch
import seaborn as sns
from matplotlib.colors import ListedColormap

def visualize_embeddings(pos_embeddings, neg_embeddings, save_path=None, title="Embedding Space Visualization"):
    """
    可视化嵌入空间
    
    Args:
        pos_embeddings: 正样本嵌入 [n_pos, dim]
        neg_embeddings: 负样本嵌入 [n_neg, dim]
        save_path: 保存路径
        title: 图表标题
    """
    # 确保输入是numpy数组
    if isinstance(pos_embeddings, torch.Tensor):
        pos_embeddings = pos_embeddings.detach().cpu().numpy()
    if isinstance(neg_embeddings, torch.Tensor):
        neg_embeddings = neg_embeddings.detach().cpu().numpy()
    
    # 合并嵌入
    all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
    
    # 创建标签
    labels = np.concatenate([
        np.ones(pos_embeddings.shape[0]),  # 正样本标签为1
        np.zeros(neg_embeddings.shape[0])  # 负样本标签为0
    ])
    
    # 使用t-SNE降维
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # 分离正负样本
    n_pos = pos_embeddings.shape[0]
    pos_2d = embeddings_2d[:n_pos]
    neg_2d = embeddings_2d[n_pos:]
    
    # 设置样式
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # 创建自定义颜色映射
    colors = ["#FF5555", "#5555FF"]  # 红色和蓝色
    cmap = ListedColormap(colors)
    
    # 绘制散点图
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', s=100, alpha=0.7, label='正确路径', edgecolors='w')
    plt.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', s=100, alpha=0.7, label='误导路径', edgecolors='w')
    
    # 添加标题和图例
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_noise_robustness(results, metric_name='document_retention_rate', save_path=None):
    """
    绘制噪声鲁棒性曲线
    
    Args:
        results: 不同噪声级别的结果字典 {noise_level: metrics}
        metric_name: 要绘制的指标名称
        save_path: 保存路径
    """
    # 提取噪声级别和对应的指标值
    noise_levels = sorted(results.keys())
    metric_values = [results[level][metric_name] for level in noise_levels]
    
    # 设置样式
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 绘制曲线
    plt.plot(noise_levels, metric_values, 'o-', linewidth=2, markersize=8)
    
    # 添加标题和标签
    metric_display_names = {
        'document_retention_rate': '文档保留率',
        'exact_match': '精确匹配率',
        'f1': 'F1分数',
        'contrast_accuracy': '对比准确率'
    }
    
    display_name = metric_display_names.get(metric_name, metric_name)
    plt.title(f'噪声鲁棒性: {display_name}', fontsize=16)
    plt.xlabel('噪声比例', fontsize=14)
    plt.ylabel(display_name, fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(noise_levels)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for x, y in zip(noise_levels, metric_values):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_similarity_matrix(pos_embeddings, neg_embeddings, save_path=None):
    """
    可视化相似度矩阵
    
    Args:
        pos_embeddings: 正样本嵌入
        neg_embeddings: 负样本嵌入
        save_path: 保存路径
    """
    if isinstance(pos_embeddings, torch.Tensor):
        pos_embeddings = pos_embeddings.detach().cpu().numpy()
    if isinstance(neg_embeddings, torch.Tensor):
        neg_embeddings = neg_embeddings.detach().cpu().numpy()
    
    # 计算余弦相似度
    def cosine_similarity(a, b):
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a_normalized = a / a_norm
        b_normalized = b / b_norm
        return np.matmul(a_normalized, b_normalized.T)
    
    pos_pos_sim = cosine_similarity(pos_embeddings, pos_embeddings)
    pos_neg_sim = cosine_similarity(pos_embeddings, neg_embeddings)
    neg_neg_sim = cosine_similarity(neg_embeddings, neg_embeddings)
    
    # 创建完整的相似度矩阵
    n_pos = pos_embeddings.shape[0]
    n_neg = neg_embeddings.shape[0]
    
    full_sim = np.zeros((n_pos + n_neg, n_pos + n_neg))
    full_sim[:n_pos, :n_pos] = pos_pos_sim
    full_sim[:n_pos, n_pos:] = pos_neg_sim
    full_sim[n_pos:, :n_pos] = pos_neg_sim.T
    full_sim[n_pos:, n_pos:] = neg_neg_sim
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(full_sim, dtype=bool)
    mask[np.diag_indices_from(mask)] = True  # 掩盖对角线
    
    sns.heatmap(full_sim, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    # 添加分隔线
    plt.axhline(y=n_pos, color='k', linestyle='-', linewidth=2)
    plt.axvline(x=n_pos, color='k', linestyle='-', linewidth=2)
    
    # 添加标签
    plt.title("嵌入相似度矩阵", fontsize=16)
    
    # 设置刻度标签
    pos_labels = [f"正样本{i+1}" for i in range(n_pos)]
    neg_labels = [f"负样本{i+1}" for i in range(n_neg)]
    all_labels = pos_labels + neg_labels
    
    plt.xticks(np.arange(len(all_labels)) + 0.5, all_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(all_labels)) + 0.5, all_labels, rotation=0)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相似度矩阵已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_attention_weights(model, input_ids, attention_mask, token_type_ids=None, 
                               tokenizer=None, save_path=None):
    """
    可视化注意力权重
    
    Args:
        model: 模型
        input_ids: 输入ID
        attention_mask: 注意力掩码
        token_type_ids: token类型ID
        tokenizer: 分词器
        save_path: 保存路径
    """
    if not hasattr(model, 'discriminator') or not hasattr(model.discriminator, 'encoder'):
        print("模型不支持注意力可视化")
        return
    
    # 确保输入是批次形式
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(0)
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        outputs = model.discriminator.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
    
    # 获取最后一层的注意力权重
    attention = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
    attention = attention.mean(dim=1)  # 平均所有注意力头 [batch_size, seq_len, seq_len]
    
    # 解码token
    if tokenizer is not None:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    else:
        tokens = [f"Token_{i}" for i in range(input_ids.shape[1])]
    
    # 只保留有效的token（非padding）
    valid_length = attention_mask[0].sum().item()
    attention = attention[0, :valid_length, :valid_length]
    tokens = tokens[:valid_length]
    
    # 绘制注意力热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.cpu().numpy(), xticklabels=tokens, yticklabels=tokens, 
                cmap="YlGnBu", square=True)
    
    plt.title("注意力权重可视化", fontsize=16)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重可视化已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(train_metrics, val_metrics=None, metrics_to_plot=None, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_metrics: 训练指标字典 {epoch: {metric_name: value}}
        val_metrics: 验证指标字典 {epoch: {metric_name: value}}
        metrics_to_plot: 要绘制的指标列表
        save_path: 保存路径
    """
    if metrics_to_plot is None:
        # 默认绘制所有指标
        metrics_to_plot = list(next(iter(train_metrics.values())).keys())
    
    # 每个指标一个子图
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    
    # 如果只有一个指标，确保axes是列表
    if n_metrics == 1:
        axes = [axes]
    
    # 提取轮次
    epochs = sorted(train_metrics.keys())
    
    # 中文指标名称映射
    metric_display_names = {
        "loss": "损失",
        "generator_loss": "生成器损失",
        "contrastive_loss": "对比损失",
        "document_retention_rate": "文档保留率",
        "exact_match": "精确匹配率",
        "f1": "F1分数",
        "contrast_accuracy": "对比准确率",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC"
    }
    
    # 绘制每个指标
    for i, metric in enumerate(metrics_to_plot):
        # 提取训练指标
        train_values = [train_metrics[epoch].get(metric, float('nan')) for epoch in epochs]
        
        # 绘制训练曲线
        axes[i].plot(epochs, train_values, 'o-', label='训练')
        
        # 如果有验证指标，也绘制
        if val_metrics is not None:
            val_values = [val_metrics[epoch].get(metric, float('nan')) for epoch in epochs]
            axes[i].plot(epochs, val_values, 's-', label='验证')
        
        # 设置标题和标签
        display_name = metric_display_names.get(metric, metric)
        axes[i].set_title(f"{display_name}随轮次变化", fontsize=14)
        axes[i].set_xlabel('轮次', fontsize=12)
        axes[i].set_ylabel(display_name, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend()
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close() 