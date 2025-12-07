# Hyperspectral Pipeline (Standalone)

这是一个独立的高光谱（HSI）处理流程，借鉴 CrossTimeNet 的分阶段思想，但与原始时间序列项目解耦：

- **阶段 1：Tokenizer 训练** — `train_hsi_tokenizer.py`
  - 训练光谱 TCN VQ-VAE（逐像元光谱）与空间 CNN VQ-VAE（图像块），产出两个码本检查点。
- **阶段 2：BERT 预训练** — `pretrain_hsi_bert.py`
  - 冻结或微调 tokenizer，做掩码 token 预测以对齐光谱/空间码本。
- **阶段 3：下游微调** — `finetune_hsi_downstream.py`
  - 载入预训练权重，训练分类头（附带异常检测头）。

## 统一参数管理
`hsi_params.py` 使用 dataclass 提供 `TokenizerConfig`、`PretrainConfig`、`FinetuneConfig` 三套配置，支持通过 `--config your.json` 覆盖默认值。

示例 JSON：
```json
{
  "tokenizer": {"num_epochs": 20, "save_dir": "./runs/tokenizer"},
  "pretrain": {"mask_ratio": 0.4, "save_dir": "./runs/pretrain"},
  "finetune": {"num_classes": 15, "save_dir": "./runs/finetune"}
}
```

## 数据与 DataLoader
`hsi_data.py` 定义 `HSICubeDataset`：
- 从 `torch.save` 生成的张量文件加载高光谱立方体及标签。
- 可选均值方差归一化；支持按 split 名称加载不同文件列表。

## 核心模型
`hsi_model.py` 提供：
- `SpectralTCNVQVAE`：TCN + VQ 对逐像元光谱量化。
- `SpatialCNNVQVAE`：CNN + VQ 对空间补丁量化。
- `HSIBert`：拼接光谱/空间 token 做 BERT 掩码建模；包含分类头与异常检测头。

## 快速开始
假设已有 `train.pt`/`val.pt` 立方体张量：
```bash
# 阶段 1：训练 tokenizer
python train_hsi_tokenizer.py --config configs/hsi.json --split train

# 阶段 2：BERT 自监督预训练
python pretrain_hsi_bert.py --config configs/hsi.json --freeze_tokenizers

# 阶段 3：下游分类微调
python finetune_hsi_downstream.py --config configs/hsi.json --train_split train --val_split val
```

> 提示：将本目录视作独立项目使用，不需要修改 CrossTimeNet 的时间序列代码。
