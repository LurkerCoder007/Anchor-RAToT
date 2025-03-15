# 项目结构
ratot_contrastive/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # 数据集加载与处理
│   └── negative_sampler.py # 负样本生成策略
├── models/
│   ├── __init__.py
│   ├── encoder.py          # 路径编码器
│   ├── contrastive.py      # 对比学习模块
│   └── ratot.py            # RAToT集成
├── training/
│   ├── __init__.py
│   ├── losses.py           # 损失函数
│   ├── trainer.py          # 训练器
│   └── evaluator.py        # 评估器
├── utils/
│   ├── __init__.py
│   └── metrics.py          # 评估指标
├── config.py               # 配置文件
└── main.py                 # 主程序入口 