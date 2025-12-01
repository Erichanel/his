# SSPQ-CrossTimeNet-HSI 项目需求说明书

> 面向 Codex / Copilot 自动生成代码的技术规格文档

---

## 1. 项目名称与目标

**项目名称**：SSPQ-CrossTimeNet-HSI（高光谱异常检测）

**总体目标**：构建一个三阶段的高光谱异常检测流水线：

1. **Stage 1 – SSPQ Tokenizer**
   - 输入：高光谱图像 patch，形状 `S × S × B`（S 为空间尺寸，B 为光谱维度）。
   - 输出：离散 **Joint Token ID 序列**（SSPQ tokens），对应每个 patch 的光谱–空间码。
   - 方法：光谱 TCN VQ‑VAE + 空间 CNN VQ‑VAE，训练两个码本 Kₛ（spectral）和 Kₓ（spatial），联合 token 定义为 `joint_id = k + K_s * m`。

2. **Stage 2 – Cross-Domain Self-Supervised Pre-Training（BERT + Structured MTP）**
   - 输入：来自多个场景 / 传感器的 SSPQ token 序列。
   - Backbone：BERT 风格 Transformer（基于 `transformers.BertForMaskedLM`）。
   - 任务：结构化 Masked Token Prediction（MTP），使用光谱段掩码 + 空间邻域掩码，mask ratio ≈ 0.3–0.45。
   - 输出：跨传感器共享的高光谱 token encoder 参数（通用 HSI 编码器）。

3. **Stage 3 – Target Task Fine-Tuning & Anomaly Detection**
   - 输入：目标场景的 SSPQ token 序列。
   - 模型：使用 Stage 2 得到的“通用 HSI 编码器”，可以完全冻结或通过 LoRA 微调。
   - 顶层加入多个线性 head：
     - LM‑NLL head：基于 masked LM 的负对数似然得分。
     - Recon head：基于 Stage 1 解码器的重建误差。
     - One‑Class head：基于深度表征的一类分类器。
   - 输出：分辨率为 `H × W` 的像素级异常得分图和多 head 的中间分数。

---

## 2. 工程目录结构（建议）

项目根目录推荐结构如下，Codex 可按照此结构生成代码文件：

```text
project_root/
├── configs/
│   ├── tokenizer.yaml          # Stage1 相关超参
│   ├── pretrain.yaml           # Stage2 相关超参
│   └── anomaly.yaml            # Stage3 相关超参
├── data/
│   ├── datasets.py             # HSI 数据集 & patch 抽取
│   └── transforms.py           # 归一化、增强等
├── sspq_tokenizer/
│   ├── spectral_vqvae.py       # SpectralTCNVQVAE
│   ├── spatial_vqvae.py        # SpatialCNNVQVAE
│   ├── joint_tokenizer.py      # SSPQTokenizer
│   └── train_tokenizer.py      # Stage1 训练脚本
├── encoder/
│   ├── hsi_bert_encoder.py     # HSIEncoderBERT + UniversalHSIModel
│   └── pretrain_mtp.py         # Stage2 预训练脚本
├── anomaly/
│   ├── heads.py                # LMNLLHead / ReconHead / OneClassHead
│   ├── detector.py             # HSIAnomalyDetector
│   ├── train_anomaly.py        # Stage3 训练脚本
│   └── inference.py            # 推理 & 生成 anomaly map
├── utils/
│   ├── config.py               # 解析 yaml / 命令行
│   ├── logging.py              # 打印 & 日志
│   ├── checkpoint.py           # 保存/加载模型
│   ├── metrics.py              # AUC 等指标
│   └── distributed.py          # DDP / 多卡支持（可选）
├── main_pretrain.py            # 命令行入口：Stage1+2
└── main_anomaly.py             # 命令行入口：Stage3
```

---

## 3. 数据格式与预处理

### 3.1 原始高光谱数据

- 单个场景数据：`cube`
  - 类型：`numpy.ndarray` 或 `torch.Tensor`
  - 形状：`(H, W, B)`。
- 可选：同尺寸 `gt_mask`
  - 形状：`(H, W)`，值为 `{0,1}`（0 表示正常，1 表示异常），用于有标签评估或监督微调。

为支持多传感器 / 多场景预训练，配置文件中使用列表描述各数据源：

```yaml
pretrain_datasets:
  - name: sensor1_sceneA
    data_path: path/to/sceneA.npz   # 内含 cube / gt 等
    domain_id: 0
  - name: sensor2_sceneB
    data_path: path/to/sceneB.npz
    domain_id: 1
```

### 3.2 Patch 抽取与归一化

#### 类 `HSIPatchExtractor`

- 初始化参数：
  - `patch_size: int` (S)
  - `stride: int`（默认为 `patch_size // 2`，允许 patch 重叠）。
- 方法：
  - `extract_patches(cube: np.ndarray) -> (patches, coords)`
    - `patches` 形状：`(N_patches, B, S, S)`（内部统一采用 `[C, H, W]` 排列）。
    - `coords`：`(N_patches, 2)`，每行是 patch 左上角坐标 `(row, col)`。

#### 归一化与增强

- 函数 `normalize_cube(cube, method='minmax'|'zscore') -> cube_norm`。
- 可选数据增强（仅 Stage1/2 训练用）：随机旋转、翻转、Gaussian noise 等，定义在 `data/transforms.py`。

---

## 4. Stage 1：SSPQ Tokenizer 模块需求

Stage1 包含两个独立训练的 VQ‑VAE 分支：光谱分支（Spectral TCN）和空间分支（Spatial CNN），分别学习光谱码本 Kₛ 和空间码本 Kₓ。

### 4.1 光谱分支：`SpectralTCNVQVAE`

文件：`sspq_tokenizer/spectral_vqvae.py`

```python
class SpectralTCNVQVAE(nn.Module):
    def __init__(
        self,
        num_bands: int,        # B
        hidden_dim: int,       # 如 64
        codebook_size: int,    # K_s
        num_layers: int,
        kernel_size: int,
        commitment_beta: float,
    ): ...
```

- 输入：`x`，形状 `[batch, B, S, S]`。
- 内部处理：
  - 对空间维做平均池化：`x_mean = x.mean(dim=(-1, -2))`，得到 `[batch, B]`。
  - 将 `[batch, B]` reshape 为 `[batch, 1, B]` 或 `[batch, B, 1]` 输入 TCN 编码器。
- 输出：
  - `recon`: `[batch, B]` 或 `[batch, B, 1]`。
  - `quantized`: `[batch, T_s, D_s]`（一般取 `T_s=1`）。
  - `indices`: `[batch, T_s]`，离散 index 范围 `[0, K_s-1]`。
  - `loss_dict`：包含 `recon_loss`, `vq_loss`, `commitment_loss`。

### 4.2 空间分支：`SpatialCNNVQVAE`

文件：`sspq_tokenizer/spatial_vqvae.py`

```python
class SpatialCNNVQVAE(nn.Module):
    def __init__(
        self,
        num_bands: int,          # B
        hidden_dim: int,
        codebook_size: int,      # K_x
        encoder_channels: list,  # 如 [64, 128, 256]
        commitment_beta: float,
    ): ...
```

- 输入：`x`，形状 `[batch, B, S, S]`。
- 处理：
  - 使用 `1×1` 卷积将光谱维 B 压到较小维度。
  - 使用多层 2D CNN + pooling 得到空间 latent feature map。
  - 将 latent map 展平为 `[batch, T_x, D_x]`，`T_x` 为空间 token 数（可以是 `S×S` 或下采样后的尺寸）。
- 输出：
  - `recon`: `[batch, B, S, S]`。
  - `quantized`: `[batch, T_x, D_x]`。
  - `indices`: `[batch, T_x]`，离散 index 范围 `[0, K_x-1]`。
  - `loss_dict`：同样包含 `recon_loss`, `vq_loss`, `commitment_loss`。

### 4.3 Joint Tokenizer：`SSPQTokenizer`

文件：`sspq_tokenizer/joint_tokenizer.py`

```python
class SSPQTokenizer(nn.Module):
    def __init__(self, spectral_cfg, spatial_cfg):
        self.spectral_vqvae = SpectralTCNVQVAE(**spectral_cfg)
        self.spatial_vqvae = SpatialCNNVQVAE(**spatial_cfg)
        self.K_s = spectral_cfg["codebook_size"]
        self.K_x = spatial_cfg["codebook_size"]
        self.joint_vocab_size = self.K_s * self.K_x
```

#### 主要方法

1. `forward(patches: Tensor, return_recon: bool = False) -> dict`

- 输入：`patches` `[batch, B, S, S]`。
- 流程：
  1. `spec_out = spectral_vqvae(patches)`。
  2. `spat_out = spatial_vqvae(patches)`。
  3. 设 `T = min(T_s, T_x)`，对齐前 T 个位置：
     - `k = spec_out["indices"][:, :T]`
     - `m = spat_out["indices"][:, :T]`
     - `joint_ids = k + K_s * m`。
- 输出字典：

```python
{
    "joint_ids": LongTensor[batch, T],
    "spectral_ids": LongTensor[batch, T_s],
    "spatial_ids": LongTensor[batch, T_x],
    "loss": spectral_loss + spatial_loss,
    "recon_s": spec_out.get("recon"),   # 可选
    "recon_x": spat_out.get("recon"),   # 可选
}
```

2. `decode_joint_ids(joint_ids: LongTensor) -> recon_patches`

- 通过 `joint_ids` 恢复 `(k, m)`：

```python
m = joint_ids // K_s
k = joint_ids % K_s
```

- 使用两个 decoder 分支得到重建结果，可按简单的加权平均或拼接后卷积融合重建 patch。

3. `save_codebooks(path)` / `load_codebooks(path)`

- 序列化 / 加载两个 VQ‑VAE 的码本和相关参数。

### 4.4 Stage1 训练脚本：`train_tokenizer.py`

```python
def train_sspq_tokenizer(config_path: str):
    # 1. 读取 configs/tokenizer.yaml
    # 2. 构建 HSIPatchExtractor & Dataset & DataLoader
    # 3. 初始化 SSPQTokenizer
    # 4. 用 Adam / AdamW + AMP 训练
    # 5. 按 epoch 保存 checkpoint
    ...
```

配置字段示例：

```yaml
patch_size: 11
stride: 5
K_s: 256
K_x: 256
hidden_dim: 64
num_layers: 4
kernel_size: 3
commitment_beta: 0.25
batch_size: 64
num_epochs: 50
lr: 1e-3
```

训练输出：`checkpoints/tokenizer/best.pth`，包含 `SSPQTokenizer` 的 `state_dict` 和配置。

---

## 5. Stage 2：Cross-Domain Self-Supervised Pre-Training

Stage2 使用 Stage1 生成的 joint token 序列，基于 BERT 风格 Transformer 执行跨域自监督预训练，目标是学习通用的 HSI token 编码器。

### 5.1 模型：`HSIEncoderBERT`

文件：`encoder/hsi_bert_encoder.py`

```python
class HSIEncoderBERT(nn.Module):
    def __init__(
        self,
        joint_vocab_size: int,     # K_s * K_x
        num_domains: int,          # 传感器/场景数量
        bert_model_name_or_path: str,  # 如 'bert-base-uncased'
        load_pretrained_lm: bool,
        mask_ratio: float,
        max_seq_len: int,
    ): ...
```

- 组件：
  - `self.encoder: BertForMaskedLM`（来自 `transformers`）。
  - `self.token_embedding: nn.Embedding(joint_vocab_size + 1, hidden_size)`，最后一个 ID 作为 `mask_token_id`。
  - `self.domain_embedding: nn.Embedding(num_domains, hidden_size)`。
  - `self.position_embedding: nn.Embedding(max_seq_len, hidden_size)`。
  - `self.dropout: nn.Dropout`。

#### 方法 `mask_tokens(joint_ids, mask_ratio, strategy="structured")`

- 输入：`joint_ids` `[batch, T]`。
- 输出：
  - `masked_input_ids` `[batch, T]`。
  - `labels` `[batch, T]`（被 mask 的位置为原始 id，其他位置为 `-100`）。
- `strategy="structured"`：
  - 优先在光谱连续段和空间相邻 token 上成块 mask。
  - 实现上可以先随机采样若干连续 index 段，再扩展到其空间邻居。

#### 方法 `forward(tokens, domain_ids, pretrain=True)`

- 输入：
  - `tokens: LongTensor` `[batch, T]`。
  - `domain_ids: LongTensor` `[batch]`。
- `pretrain=True`：
  1. 调用 `mask_tokens` 生成 `masked_input_ids` 与 `labels`。
  2. 计算嵌入：

```python
tok_emb = token_embedding(masked_input_ids)
dom_emb = domain_embedding(domain_ids).unsqueeze(1)  # [batch, 1, hidden]
pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0)
pos_emb = position_embedding(pos_ids)
inputs_embeds = tok_emb + dom_emb + pos_emb
```

  3. 调用 `self.encoder(inputs_embeds=inputs_embeds, labels=labels)`。
  4. 返回：`loss`, `logits`, `hidden_states`（最后一层）。

- `pretrain=False`：
  - 不 mask，直接构造 `inputs_embeds`，返回最后一层 `hidden_states`。

### 5.2 封装：`UniversalHSIModel`

```python
class UniversalHSIModel(nn.Module):
    def __init__(self, tokenizer: SSPQTokenizer, encoder: HSIEncoderBERT):
        self.tokenizer = tokenizer
        self.encoder = encoder

    def encode_patches(self, patches, domain_ids, pretrain: bool = False):
        # patches: [batch, B, S, S]
        tok_out = self.tokenizer(patches, return_recon=False)
        joint_ids = tok_out["joint_ids"]          # [batch, T]
        if pretrain:
            loss, logits, hidden = self.encoder(joint_ids, domain_ids, pretrain=True)
            return {"loss": loss, "logits": logits, "hidden": hidden, "joint_ids": joint_ids}
        else:
            hidden = self.encoder(joint_ids, domain_ids, pretrain=False)
            return {"hidden": hidden, "joint_ids": joint_ids}
```

### 5.3 Stage2 训练脚本：`pretrain_mtp.py`

```python
def pretrain_universal_hsi(config_path: str):
    # 1. 读取 configs/pretrain.yaml
    # 2. 加载 SSPQTokenizer checkpoint 并冻结参数
    # 3. 初始化 HSIEncoderBERT 和 UniversalHSIModel
    # 4. 为每个 domain 构建 DataLoader（输出 patches, domain_id）
    # 5. 使用 AMP + 线性 warmup 学习率调度进行训练
    # 6. 保存最佳 encoder checkpoint
    ...
```

典型配置字段：

```yaml
bert_model_name_or_path: "bert-base-uncased"
load_pretrained_lm: true
num_domains: 3
mask_ratio: 0.35
max_seq_len: 256
pretrain_batch_size: 32
pretrain_epochs: 30
learning_rate_pretrain: 1e-4
num_warmup_steps: 2000
```

输出：`checkpoints/encoder/universal_hsi_encoder.pth`。

---

## 6. Stage 3：HSI Anomaly Detection

Stage3 使用预训练的通用 HSI 模型进行目标场景异常检测，可选择完全冻结 encoder 或使用 LoRA 微调，顶层采用多种 linear head 组合。

### 6.1 Anomaly Heads（`anomaly/heads.py`）

#### 6.1.1 LM‑NLL Head：`LMNLLHead`

- 功能：基于 masked LM 的 token-level 负对数似然，作为异常得分。
- 接口：

```python
def compute_lm_nll_scores(
    encoder: HSIEncoderBERT,
    joint_ids: torch.LongTensor,   # [batch, T]
    domain_ids: torch.LongTensor,  # [batch]
    mask_strategy: str = "structured",
) -> torch.Tensor:                 # [batch, T]
    ...
```

- 思路：
  - 对输入 `joint_ids` 再次执行一次结构化 mask，调用 `encoder(..., pretrain=True)`。
  - 使用 `outputs.logits` 计算被 mask 位置的 `-log p(token)`，其余位置设为 0 或插值。

#### 6.1.2 Recon Head：`ReconHead`

- 功能：利用 SSPQTokenizer 的解码器计算重建误差。

```python
def compute_recon_error(
    tokenizer: SSPQTokenizer,
    patches: torch.Tensor,         # [batch, B, S, S]
    joint_ids: torch.LongTensor,   # [batch, T]
) -> torch.Tensor:                 # [batch] 或 [batch, T]
    ...
```

- 简化方案：
  - 对 patch 前向通过 tokenizer（含 decode）获得重建 patch。
  - 计算 MSE 或 L1 误差，聚合到 patch-level 或 token-level。

#### 6.1.3 One-Class Head：`OneClassHead`

```python
class OneClassHead(nn.Module):
    def __init__(self, hidden_size: int):
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # 可选：DeepSVDD 中心向量 c
```

- 输入：patch-level 表征 `z`，例如 `z = seq_embeds.mean(dim=1)`，`[batch, hidden_size]`。
- 训练：
  - 有标签：BCELoss / BCEWithLogitsLoss。
  - 无标签（仅正常样本）：可维护中心向量 `c`，最小化 `||f(x) - c||^2`。

### 6.2 汇总模型：`HSIAnomalyDetector`

文件：`anomaly/detector.py`

```python
class HSIAnomalyDetector(nn.Module):
    def __init__(
        self,
        universal_model: UniversalHSIModel,
        hidden_size: int,
        freeze_encoder: bool = True,
        use_lora: bool = False,
        head_weights: dict | None = None,
    ): ...
```

- 成员：
  - `self.universal_model`
  - `self.one_class_head`
  - `self.head_weights = {"lm_nll": 0.4, "recon": 0.3, "one_class": 0.3}`（默认）。

#### 方法 `forward(patches, domain_ids, mode="train"|"eval")`

- 流程：
  1. 通过 `universal_model.encode_patches(patches, domain_ids, pretrain=False)` 得到
     - `hidden`: `[batch, T, hidden_size]`
     - `joint_ids`。
  2. 计算各 head：
     - `lm_scores = compute_lm_nll_scores(...)` → `[batch, T]`
     - `recon_scores = compute_recon_error(...)` → `[batch]` 或 `[batch, T]`
     - `oc_scores = one_class_head(hidden.mean(dim=1))` → `[batch, 1]`
  3. 将三个分数归一化到相近尺度（如 z-score / min-max），再按照 `head_weights` 加权求和得到 `combined_score`。

- 返回：

```python
{
    "lm_nll": lm_scores,
    "recon": recon_scores,
    "one_class": oc_scores,
    "combined": combined_score,
}
```

#### 方法 `generate_anomaly_map(cube, domain_id)`

- 输入：`cube` `[H, W, B]`。
- 步骤：
  1. 使用 `HSIPatchExtractor` 抽取所有 patch 及其坐标。
  2. 将 patch 分 batch 输入 `forward`，得到每个 patch 的 `combined_score`。
  3. 构建大小为 `[H, W]` 的 score 累积矩阵和计数矩阵，将覆盖某像素的所有 patch 分数取平均。
  4. 返回最终 `anomaly_map` `[H, W]`。

### 6.3 Stage3 训练与推理脚本

#### `anomaly/train_anomaly.py`

```python
def train_anomaly_detector(config_path: str):
    # 1. 读取 configs/anomaly.yaml
    # 2. 加载 UniversalHSIModel checkpoint
    # 3. 根据配置是否冻结 encoder / 使用 LoRA
    # 4. 构建面向目标场景的 Patch Dataset（可从 gt_mask 派生 patch 标签）
    # 5. 训练 OneClassHead（以及可选的 LoRA 参数）
    # 6. 保存 anomaly_detector.pth
    ...
```

#### `anomaly/inference.py`

```python
def run_inference(config_path: str):
    # 1. 加载 anomaly_detector.pth
    # 2. 读取目标场景 cube
    # 3. 调用 HSIAnomalyDetector.generate_anomaly_map
    # 4. 保存 anomaly_map.npy / 可视化结果
    # 5. 如提供 gt_mask，则计算 AUC/PR 等指标
    ...
```

---

## 7. 配置与超参数

统一采用 yaml 配置或命令行参数，类似 CrossTimeNet 的 `args.py`：

```yaml
seed: 66
device: "cuda:0"

# BERT / LM
bert_model_name_or_path: "bert-base-uncased"
load_pretrained_lm: true

# Tokenizer
patch_size: 11
stride: 5
K_s: 256
K_x: 256
hidden_dim: 64
tokenizer_lr: 1e-3
tokenizer_epochs: 50

# Pretrain
mask_ratio: 0.35
pretrain_batch_size: 32
pretrain_epochs: 30
learning_rate_pretrain: 1e-4
num_warmup_steps: 2000

# Anomaly
anomaly_batch_size: 16
anomaly_epochs: 20
learning_rate_anomaly: 1e-4
freeze_encoder: true
use_lora: false
head_weights:
  lm_nll: 0.4
  recon: 0.3
  one_class: 0.3
```

配置解析由 `utils/config.py` 实现，支持命令行参数覆盖 yaml 中字段。

---

## 8. 非功能性需求

- **框架与依赖**
  - Python ≥ 3.9
  - PyTorch ≥ 2.0
  - `transformers`（BERT）
  - `numpy`, `scipy`（可选）, `tqdm`, `pyyaml`
- **性能**
  - 所有训练脚本优先使用 GPU（`cuda`），并支持 AMP（`torch.cuda.amp.autocast` + `GradScaler`）。
- **可扩展性**
  - 易于添加新的场景 / 传感器，只需在配置文件中新增 dataset 项并指定 `domain_id`。
  - Tokenizer、Encoder 与 AnomalyDetector 模块相互解耦，可替换任意一个模块而不影响整体架构。

---

## 9. 推荐训练 / 推理流程（命令行示例）

1. 训练 Stage1：SSPQ Tokenizer

```bash
python sspq_tokenizer/train_tokenizer.py     --config configs/tokenizer.yaml
```

2. 训练 Stage2：Universal HSI Encoder

```bash
python encoder/pretrain_mtp.py     --config configs/pretrain.yaml     --tokenizer_ckpt checkpoints/tokenizer/best.pth
```

3. 训练 Stage3：Anomaly Detector

```bash
python anomaly/train_anomaly.py     --config configs/anomaly.yaml     --encoder_ckpt checkpoints/encoder/universal_hsi_encoder.pth
```

4. 推理：生成像素级 anomaly map

```bash
python anomaly/inference.py     --config configs/anomaly.yaml     --detector_ckpt checkpoints/anomaly/anomaly_detector.pth     --data_path path/to/target_scene.npz
```

---

本文件即为可直接上传给 Codex / Copilot 的项目需求与接口说明文档，后续可按文件结构逐个让模型生成具体实现代码。
