# 需求文档：HSI-VQ-SSSE（9×9 patch，3×3 cube，GWPCA 64维，HSIMAE-style SSSE）

> 复制友好版（可直接粘贴到你的仓库 README / docs / 发给 Codex）。  
> 固定规格：patch=9×9，cube=3×3，GWPCA=8组×8主成分→64维。  
> token 网格：空间 P=9（3×3），光谱 T=8，总 tokens N=72。每个 token 对应一个 cube：3×3×8（flatten F=72）。

---

## 1. 目标与范围

### 1.1 目标
在现有仓库风格（configs/ + main_*.py + encoder/ + sspq_tokenizer/ + utils/）下实现：

1) **分类任务**：patch-level logits，支持少样本/常规训练。  
2) **异常检测**：输出 patch-level anomaly score + spatial anomaly map（3×3）。  
3) 保留模型特色：**SSPQ 双码本 VQ 离散 token + LM-NLL 异常分数**。  
4) Stage2 效仿 HSIMAE：**SSSE（空间/光谱分离 Transformer）+ 可分离位置编码 + 行/列一致性结构化 mask**。  
5) **禁止 joint_id**：使用 factorized token 预测（分别预测光谱码本索引 k 与空间码本索引 m）。

### 1.2 非目标（可选项）
- （可选）完整 VQ-VAE decoder 重建 9×9×64：可作为增强项，不是第一阶段必须。  
- （可选）大规模外部预训练数据 pipeline：先保证模块/脚本可跑通。

---

## 2. 固定输入与 tokenization（硬约束）

### 2.1 输入
- 原始 patch：`x_raw ∈ R[B, C_raw, 9, 9]`（C_raw 可变）  
- GWPCA 后：`x ∈ R[B, 64, 9, 9]`，其中 `64 = T*L = 8*8`

### 2.2 token 网格
- 空间切块：`cube_size s=3`，non-overlap stride=3  
  - `Hs=Ws=9/3=3`  
  - `P=Hs*Ws=9`
- 光谱分组：`T=8`，每组 `L=8`
- 每 token cube：`[3,3,8]`，flatten `F=72`
- token 网格：`[P=9, T=8]`，总 token 数 `N=72`

---

## 3. 仓库目录与模块规范（必须遵守）

建议新增/规范化以下文件（职责不可混淆）：

### 3.1 configs/
- `configs/pretrain_hsivq_ssse.yaml`
- `configs/classify_hsivq_ssse.yaml`
- `configs/anomaly_hsivq_ssse.yaml`

### 3.2 utils/
- `utils/gwpca.py`：GWPCA 到 64维（8组×8PC）
- `utils/cube_partition.py`：9×9×64 → cubes `[B,9,8,72]`
- `utils/pos_embed.py`：2D/1D sincos 位置编码
- `utils/masking.py`：HSIMAE-style 行/列一致性 mask
- （可选）`utils/metrics.py`：AUROC/AUPR/TopK mean 等

### 3.3 sspq_tokenizer/
- `sspq_tokenizer/vq.py`：VectorQuantizerEMA（或普通 VQ）
- `sspq_tokenizer/tokenizer.py`：SSPQTokenizer（双分支：空间 CNN + 光谱 1DConv）

### 3.4 encoder/
- `encoder/vit_blocks.py`：TransformerBlock（ViT-style）
- `encoder/ssse_encoder.py`：SSSEEncoder（SpatialEncoder + SpectralEncoder + FusionBlock）

### 3.5 models/
- `models/hsi_vq_ssse.py`：整模型组装（Tokenizer + SSSE + Heads）

### 3.6 main 脚本
- `main_pretrain.py / main_classify.py / main_anomaly.py`  
保持入口风格，只替换模型与调用链，保证能跑通。

---

## 4. Stage0：GWPCA（光谱 group-wise PCA）

### 4.1 目标
统一光谱维度到 `C_in=64`，确保跨数据集/传感器维度一致。

### 4.2 约束与接口
- groups = 8
- nc_per_group = 8
- 输出必须是 `[B,64,9,9]`（本任务 patch 级）

文件：`utils/gwpca.py`
- `apply_gwpca(x, group=8, nc_per_group=8, whiten=True) -> x_pca`
- x 允许输入：`[H,W,C_raw]` 或 `[B,C_raw,H,W]`  
- 产出：`[B,64,9,9]`（或对整图：`[B,64,H,W]`）

---

## 5. cube 切分：make_cubes（9×9×64 → [B,9,8,72]）

文件：`utils/cube_partition.py`  
函数：`make_cubes(x, patch_size=9, cube_size=3, T=8) -> (cubes, meta)`

### 5.1 输入
- `x`: `[B,64,9,9]`

### 5.2 输出
- `cubes`: `[B,P=9,T=8,F=72]`
- `meta`: `Hs=3, Ws=3, L=8, F=72`

### 5.3 切分规则（必须一致）
1) reshape 光谱分组：`[B,64,9,9] -> [B,T=8,L=8,9,9]`
2) 空间 non-overlap unfold（3×3，stride=3）：得到 `Hs*Ws=9` 个空间块  
3) cube: `[3,3,8]` flatten 为 `F=72`
4) 输出顺序：**先空间 P，再光谱 T** → `[B,P,T,F]`

---

## 6. Stage1：SSPQ Tokenizer（轻量 CNN + 轻量 1DConv）

> 目标：每个 token cube（3×3×8）→ 两个离散索引：k（光谱码本）与 m（空间码本）。  
> Token 数量不变：同一 (p,t) 位置对应一对 (k,m)。禁止合并 joint_id。

### 6.1 Tokenizer 必须输出
文件：`sspq_tokenizer/tokenizer.py`  
类：`SSPQTokenizer`

`forward(cubes:[B,9,8,72]) -> dict`
- `k_idx`: `[B,9,8]` (long), in `[0, Ks)`
- `m_idx`: `[B,9,8]` (long), in `[0, Kx)`
- `emb_s`: `[B,9,8,D]`
- `emb_x`: `[B,9,8,D]`
- `vq_loss_dict`: `{'spectral':..., 'spatial':..., 'total':...}`
- （可选）perplexity 指标

### 6.2 光谱分支（1DConv/轻TCN）——“谱型字典”
输入逻辑（建议实现为代码里的明确步骤）：
- 将 F=72 还原 cube：`[B,9,8,3,3,8]`（或在 make_cubes 同时返回 unflatten 版本）
- 空间聚合：`mean over 3×3 -> [B,9,8,8]`
- reshape：`[B*9*8, 1, 8]`
- 1DConv×2（kernel=3, padding=1）+ GELU
- flatten + Linear -> `z_e_s ∈ R[B,9,8,D_vq]`
- VQ（codebook size Ks）→ k_idx + emb_s

### 6.3 空间分支（2D CNN）——“空间模式字典”
- cube 重排为“8 通道 3×3 小图”：`[B*9*8, 8, 3, 3]`
- Conv2d×2（kernel=3, padding=1）+ GELU
- flatten + Linear -> `z_e_x ∈ R[B,9,8,D_vq]`
- VQ（codebook size Kx）→ m_idx + emb_x

### 6.4 VQ 实现要求
文件：`sspq_tokenizer/vq.py`
- 推荐 `VectorQuantizerEMA`
- 配置：`decay=0.99`，`beta=0.25`
- 输出：`z_q, indices, vq_loss, perplexity`

---

## 7. Stage2：HSIMAE 风格 SSSE Transformer（最关键提升点）

### 7.1 token 融合（禁止 joint_id）
- `tok = MLP(concat(emb_s, emb_x))` → `[B,9,8,D]`
- sum 作为可选配置，但默认 concat+MLP

### 7.2 可分离位置编码（Pos2D + Pos1D）
文件：`utils/pos_embed.py`
- `pos2d: [P=9, D]` 对 3×3 空间网格 sincos
- `pos1d: [T=8, D]` 对 8 光谱组 sincos
- `pos: [P,T,D] = pos2d[:,None,:] + pos1d[None,:,:]`
- `tok += pos`

### 7.3 HSIMAE-style 行/列一致性 mask
文件：`utils/masking.py`
函数：`hsimae_consistent_mask(B, P=9, T=8, mr_spa, mr_spe, device, seed=None) -> mask`
- 采样 `Mp`：`ceil(mr_spa*P)` 个空间位置
- 采样 `Mt`：`ceil(mr_spe*T)` 个光谱组
- `mask[p,t] = True if (p in Mp) OR (t in Mt)`
- 输出 mask: `[B,9,8]` bool，并返回实际 mask ratio 统计

### 7.4 SSSEEncoder（分离编码 + 融合）
文件：`encoder/ssse_encoder.py`
类：`SSSEEncoder`，输入/输出：
- 输入：`tok [B,9,8,D]`，`mask [B,9,8]`
- 输出：`z [B,9,8,D]`

mask token：
- learnable `mask_token [1,1,1,D]`
- `tok_masked = tok; tok_masked[mask]=mask_token`

SpatialEncoder（学空间）：
- reshape：`[B,9,8,D] -> [B,8,9,D] -> [B*8,9,D]`
- TransformerBlock × depth_spa
- reshape 回 `[B,9,8,D]` 得 `z_spa`

SpectralEncoder（学光谱）：
- reshape：`[B,9,8,D] -> [B*9,8,D]`
- TransformerBlock × depth_spe
- reshape 回 `[B,9,8,D]` 得 `z_spe`

融合：
- `z = FusionBlock(z_spa + z_spe)`（depth_fuse 个 block 或轻 MLP）

TransformerBlock：
文件 `encoder/vit_blocks.py`
- LN + MHA + residual
- LN + MLP(GELU) + residual
- batch_first=True
- 所有 reshape/permute 必须 assert 维度正确

---

## 8. 预训练目标：Factorized MTP（k/m 两头预测）

### 8.1 heads
- `head_k: Linear(D -> Ks)` 输出 logits_k `[B,9,8,Ks]`
- `head_m: Linear(D -> Kx)` 输出 logits_m `[B,9,8,Kx]`

### 8.2 loss（只在 mask=True 位置计算）
- `L_k = CE(logits_k[mask], k_idx[mask])`
- `L_m = CE(logits_m[mask], m_idx[mask])`
- `L_mtp = L_k + L_m`
- 预训练总 loss：`L_total = L_mtp + vq_loss_total`

### 8.3 NLL map（异常检测基础）
- `nll_k = -log softmax(logits_k)[..., k_true]`
- `nll_m = -log softmax(logits_m)[..., m_true]`
- `nll_map = nll_k + nll_m` → `[B,9,8]`
- loss 只对 mask 位置回传，但 nll_map 推荐对全位置都计算（更稳定的异常热图）

---

## 9. 下游任务输出

### 9.1 分类
- pooling：默认 mean over P,T → `[B,D]`
- `cls_head: Linear(D -> num_classes)`
- `L_cls = CE(logits, y)`
- 微调可联合：`L = L_cls + lambda_mtp*L_mtp + vq_loss_total`（可选）

### 9.2 异常检测
- `anomaly_score [B]`：mean(nll_map) 或 topk_mean
- `anomaly_map_spatial [B,3,3]`：对 T 平均后 reshape：
  - `nll_spatial = mean_t(nll_map) -> [B,9]`
  - reshape -> `[B,3,3]`

---

## 10. configs 字段（最少集）

必须字段：
- data: `patch_size=9`, `cube_size=3`, `pca_group=8`, `pca_nc_per_group=8`, `C_in=64`, `T=8`, `L=8`
- vq: `Ks`, `Kx`, `D_vq`, `vq_decay=0.99`, `vq_beta=0.25`
- encoder: `D`, `depth_spa`, `depth_spe`, `depth_fuse`, `heads`, `mlp_ratio`, `dropout`
- mask: `mr_spa`, `mr_spe`
- train: `lr`, `wd`, `epochs`, `batch_size`, `seed`, `lambda_mtp`

推荐默认（强起点）：
- `D=384`, `D_vq=256`, `Ks=512`, `Kx=512`
- `depth_spa=6`, `depth_spe=6`, `depth_fuse=2`
- `heads=8`, `mlp_ratio=4.0`
- `mr_spa=0.5`, `mr_spe=0.5`, `dropout=0.1`

---

## 11. main_* 脚本验收要求

### main_pretrain.py
- 随机数据跑通 forward/backward 一步
- 打印 shapes：
  - cubes `[B,9,8,72]`
  - tok `[B,9,8,D]`
  - logits_k `[B,9,8,Ks]`
  - nll_map `[B,9,8]`

### main_classify.py
- 输出 logits `[B,num_classes]`，loss 非 NaN

### main_anomaly.py
- 输出 anomaly_score `[B]`，anomaly_map_spatial `[B,3,3]`

---

## 12. 验收标准（必须）
- 不使用 joint_id；k/m 两头预测 + NLL(k)+NLL(m) 可工作  
- SSSE reshape 流程正确，能在 `[B,9,8,D]` 上完成分离编码  
- mask 是 HSIMAE-style 行/列一致性（不是普通随机 mask）  
- 模块化规范：utils/、encoder/、sspq_tokenizer/、models/ 分工清晰  
