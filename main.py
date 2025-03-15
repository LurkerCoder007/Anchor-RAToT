import argparse
import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, set_seed
import wandb
import json
from datetime import datetime

from data.dataset import ReasoningPathDataset, create_dataloader
from models.ratot import ContrastiveRAToT
from training.trainer import ContrastiveTrainer
from training.evaluator import Evaluator
from training.losses import DynamicContrastiveLoss
from utils.noise_generator import create_noisy_datasets
from utils.visualization import visualize_embeddings, plot_noise_robustness

def parse_args():
    parser = argparse.ArgumentParser(description="对比推理路径评估器")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, required=True, help="验证数据路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--retrieval_pool", type=str, required=True, help="检索池路径")
    
    # 模型参数
    parser.add_argument("--generator_model", type=str, default="t5-base", help="生成器模型")
    parser.add_argument("--discriminator_model", type=str, default="bert-base-uncased", help="判别器模型")
    parser.add_argument("--projection_dim", type=int, default=128, help="投影维度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="预训练轮次")
    parser.add_argument("--finetune_epochs", type=int, default=5, help="微调轮次")
    parser.add_argument("--lr_encoder", type=float, default=3e-5, help="编码器学习率")
    parser.add_argument("--lr_generator", type=float, default=5e-5, help="生成器学习率")
    parser.add_argument("--negative_ratio", type=int, default=3, help="每个正样本的负样本数量")
    parser.add_argument("--alpha", type=float, default=0.5, help="对比损失权重")
    parser.add_argument("--temperature", type=float, default=0.07, help="温度系数")
    
    # 噪声鲁棒性测试参数
    parser.add_argument("--test_noise", action="store_true", help="是否测试噪声鲁棒性")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7], 
                        help="噪声级别列表")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="contrastive-ratot", help="wandb项目名称")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--visualize", action="store_true", help="是否可视化嵌入空间")
    
    return parser.parse_args()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    visualization_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"contrastive-ratot-{args.generator_model.split('/')[-1]}-{args.discriminator_model.split('/')[-1]}"
        )
    
    # 加载分词器
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    discriminator_tokenizer = AutoTokenizer.from_pretrained(args.discriminator_model)
    
    # 加载检索池
    with open(args.retrieval_pool, "r") as f:
        retrieval_pool = json.load(f)
    
    # 创建数据集
    train_dataset = ReasoningPathDataset(
        data_path=args.train_data,
        tokenizer=discriminator_tokenizer,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
        retrieval_pool_path=args.retrieval_pool
    )
    
    val_dataset = ReasoningPathDataset(
        data_path=args.val_data,
        tokenizer=discriminator_tokenizer,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
        retrieval_pool_path=args.retrieval_pool
    )
    
    test_dataset = ReasoningPathDataset(
        data_path=args.test_data,
        tokenizer=discriminator_tokenizer,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
        retrieval_pool_path=args.retrieval_pool
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ContrastiveRAToT(
        generator_name=args.generator_model,
        discriminator_name=args.discriminator_model,
        projection_dim=args.projection_dim,
        alpha=args.alpha
    ).to(device)
    
    # 创建损失函数
    contrastive_loss = DynamicContrastiveLoss(
        temperature=args.temperature,
        sup_weight=0.5,
        triplet_weight=0.5
    )
    
    # 创建训练器
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        contrastive_loss=contrastive_loss,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb
    )
    
    # 训练模型
    print("开始训练模型...")
    trainer.train(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs
    )
    
    # 加载最佳模型
    best_model_path = os.path.join(checkpoint_dir, "best_finetune_model.pt")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
    
    # 评估模型
    print("在测试集上评估模型...")
    evaluator = Evaluator(model, device=device)
    test_metrics = evaluator.evaluate(test_loader)
    
    print("测试集结果:")
    print(f"文档保留率: {test_metrics['document_retention_rate']:.4f}")
    print(f"精确匹配率: {test_metrics['exact_match']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"对比准确率: {test_metrics['contrast_accuracy']:.4f}")
    
    # 保存测试结果
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    
    # 噪声鲁棒性测试
    if args.test_noise:
        print("进行噪声鲁棒性测试...")
        
        # 创建不同噪声级别的数据集
        noisy_datasets = create_noisy_datasets(
            ReasoningPathDataset,
            test_dataset,
            retrieval_pool,
            noise_levels=args.noise_levels
        )
        
        # 创建数据加载器
        noisy_loaders = {}
        for noise_level, dataset in noisy_datasets.items():
            noisy_loaders[noise_level] = create_dataloader(
                dataset, batch_size=args.batch_size, shuffle=False)
        
        # 评估噪声鲁棒性
        noise_results = evaluator.evaluate_noise_robustness(noisy_loaders)
        
        # 保存结果
        with open(os.path.join(args.output_dir, "noise_robustness_results.json"), "w") as f:
            json.dump({str(k): v for k, v in noise_results.items()}, f, indent=4)
        
        # 绘制噪声鲁棒性曲线
        plot_noise_robustness(
            noise_results,
            metric_name='document_retention_rate',
            save_path=os.path.join(visualization_dir, "noise_robustness_drr.png")
        )
        
        plot_noise_robustness(
            noise_results,
            metric_name='f1',
            save_path=os.path.join(visualization_dir, "noise_robustness_f1.png")
        )
    
    # 可视化嵌入空间
    if args.visualize:
        print("可视化嵌入空间...")
        
        # 获取一批数据
        batch = next(iter(test_loader))
        pos_input_ids = batch['pos_input_ids'].to(device)
        pos_attention_mask = batch['pos_attention_mask'].to(device)
        pos_token_type_ids = batch['pos_token_type_ids'].to(device)
        
        # 获取第一个负样本
        neg_encoding = batch['neg_encodings'][0]
        neg_input_ids = neg_encoding['input_ids'].to(device)
        neg_attention_mask = neg_encoding['attention_mask'].to(device)
        neg_token_type_ids = neg_encoding['token_type_ids'].to(device)
        
        # 获取嵌入
        with torch.no_grad():
            pos_embeddings, neg_embeddings = model.discriminator(
                pos_input_ids, pos_attention_mask, pos_token_type_ids,
                neg_input_ids, neg_attention_mask, neg_token_type_ids
            )
        
        # 可视化
        visualize_embeddings(
            pos_embeddings,
            neg_embeddings,
            save_path=os.path.join(visualization_dir, "embedding_space.png"),
            title="推理路径嵌入空间"
        )
    
    print(f"所有结果已保存到 {args.output_dir}")
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 