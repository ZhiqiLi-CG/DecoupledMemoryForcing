# THREAD SYNC — persistent-KV one-pass (thread A) ↔ SGF2 (thread B)

**索引与现状文件。三个文件各有归属,不要交叉编辑:**

| 文件 | 归属 | 内容 |
|---|---|---|
| `docs/THREAD_SYNC.md`(本文件) | thread A 写,SGF2 只读 | 背景、文件清单、已验证项、卡住的问题 |
| `docs/SGF2_REVIEW.md` | **SGF2 写,A 只读** | SGF2 的建议,编号 S1/S2/… |
| `docs/SGF2_REBUTTAL.md` | **A 写,SGF2 只读** | A 对每条建议的逐条回应 + 实验结果 |

有异议不要改对方的文件,写回自己那份,另一方会再跟一轮。

最后更新:2026-09-06,by thread A。

---

## 1. 在做什么

一趟的 SGF:让 AR rollout 本身可微,取消第二趟 teacher-forced 前向,得到与 SGF
两趟**等价的梯度**。

核心手法:每帧的 memory write **返回不可变的 K/V page**,而不是覆写 cache buffer。
没有 detach,所以第 i 帧的 loss 能经 page 链到达第 j<i 帧;又因为 page 永不改变,
每帧前向可以做 activation checkpoint(反传重算时看到的输入与首次前向逐位相同)。
被覆写的 buffer 做不到后者,这正是以往一趟法必须切断跨帧梯度的原因。

## 2. 文件

新增(独立文件):
- `wan/modules/persistent_kv_model.py` — 模型。`causal_model.py` 的副本 + 改名,
  加了 `PersistentKVStore`、`forward_persistent`、`forward_noisy_frame`、
  `forward_clean_writer`、`PersistentMemoryLoRA`,以及 `forward()` 里的
  `persistent_mode` dispatch
- `pipeline/persistent_kv_training.py` — 一趟 pipeline
- `configs/persistent_kv_framewise.yaml` — 与 SGF config 只差两行
- `docs/persistent_kv_write.md` — 设计文档

改动的共享文件(**仅此两个**,用户已同意):
- `utils/wan_wrapper.py` +35/−3(3 行是两个逗号 + `if kv_cache` → `elif`)
- `model/base.py` +24/−0(提前返回,现有构造一字未动)

其余共享文件全部 UNTOUCHED:`causal_model.py`、`causal_inference.py`、
`self_gradient_forcing_training.py`、`dmd.py`、`distillation.py`。

## 3. 已验证(全部有脚本,在 `/mnt/data/zhaofeng/runs/sgf/tests/`)

判据不用拍脑袋的阈值,而是实测的标尺:两条路径用**不同的 attention kernel**
(SGF 走 flex_attention + block mask,一趟走 flash + 拼接 page),在**完全相同的
输入**上就已经差 **3.56e-03(前向)/ 5.36e-03(反传)**。任何基于它们的东西都不
可能比这更接近。

| 项 | 结果 | 脚本 |
|---|---|---|
| 前向 vs 官方 cached 串行 | **0.000e+00** | `test_persistent_equiv.py` |
| checkpoint 是否改变梯度 | **0.000e+00**(有无 FSDP 都是) | `test_persistent_grad.py`, `test_fsdp_ckpt_min.py` |
| 梯度是否截断 | 末帧 loss 到达 frame 0 的 page | `test_persistent_grad.py` |
| RoPE 逐帧 vs 整段 | **0.000e+00**(错位一帧则 2.4e-01) | `test_rope_alignment.py` |
| 模型 vs SGF 并行 TF(真实配置,8 层/21 帧/batch 2/local=18/sink=3,3 seeds) | output 3.4–5.0e-05,全局梯度 1.6–2.2e-04,min cos **0.99999994**,逐 block 最差 cos 0.9999997 | `test_grad_strict.py` |
| pipeline vs SGF pass2(单卡) | output **3.12e-05**,全局梯度 **2.19e-04**,cos **0.99999988** | `test_pipeline_vs_sgf.py` |
| Memory-LoRA 路由 | 零初始化下逐位相同;给定相同历史时 generation 受 dW 影响 **0.000e+00**,page 受影响 7.2e-01;`\|k\|` 比 1.0000 vs `\|v\|` 1.2489(证明 dW 在 norm 内) | `test_persistent_lora.py` |
| 训练通路 | persistent config → `PersistentKVTrainingPipeline`;SGF config 仍 → 原 pipeline | `check_training_path.py` |

显存与速度(真实 1.3B,60×104,1560 token/帧,bf16,batch 1,**FSDP+MixedPrecision**,
单 H100,权重 5.29):

| | SGF 两趟 | 一趟法 |
|---|---|---|
| 全程最大峰值 | 47.98 | **32.10**(低 33%) |
| 稳态每步峰值 | **21.78** | 32.10(高 47%) |
| 稳态步时 | **11.52s** | 17.04s(慢 1.48×) |

SGF 首步 47.98 里约 32 GiB 是 `create_block_mask(_compile=False)` 物化 N×N int64,
每进程一次、改一行可消;消掉后 SGF 两个口径都赢。单进程 FSDP 退化为 NO_SHARD,
绝对值偏悲观,只适合相对比较。

FLOPs 两边相同(168F + 1560²×462 个 attention 对),1.48× 来自串行化的低利用率。

**旧数字已作废**(无 FSDP、帧级 checkpoint、梯度错误):
~~24.16 / 13.84 / 40.01 / 1.45×~~。

## 4. **卡住的问题 —— 这是最需要你看的**

单卡 pipeline 级比较 cos **0.99999988**,但**接入真实 DMD 训练后**:

```
1 LOSS         SGF 0.16560513   PKV 0.16562467   rel 1.180e-04     ← 前向几乎一致
2 GRADIENT     cos 0.8299885    |g| SGF 4.5524   PKV 0.2463        ← 差 18.5 倍
3 PARAM DELTA  cos 0.3985822                                        ← 更差
```

`|g|` 比值 18.5,而 T=21 —— 接近程度可疑,像是只有少数帧的梯度存活,不像数值偏差。

**已排除的假设(每条都有实验):**

1. ~~FSDP + checkpoint 跨模块边界~~ — 有无 FSDP,ckpt vs no-ckpt 都是
   **0.000e+00**(`test_fsdp_ckpt_min.py`)
2. ~~两条 arm 采到不同 exit step~~ — 打印出来 **SGF (1000,750) == PKV (1000,750)**
3. ~~`gradient_mask` 不同~~ — 21 帧时 `num_generated_frames == min_num_frames`,
   两边都是 `None`
4. ~~RNG 错位~~ — `num_generated_blocks` 在 21 帧时恒等于 21,噪声在调 pipeline
   之前抽,两边一致

**踩过的坑(供你参考,别重复):**
- 我一度以为 checkpoint 有 4.8e-03 误差 → 实际是**测试脚本里闭包晚绑定**:
  `for i` 循环里定义闭包捕获 `i`,checkpoint 在反传时才重算,那时 `i` 已是 `T-1`,
  RoPE 位置错了。**pipeline 里没有这个问题**(`start` 是函数参数)。
- 单卡全绿但 4 卡第一步挂:`size_based_auto_wrap_policy` 把 `self.model` 也包成
  FSDP 单元,而 **FSDP 的 `__getattr__` 把方法直接转发给内层模块,hook 一个都不跑**。
  所以 persistent 前向**必须经 `model.__call__`**,不能调裸方法。已修。

**对照已完成,对比有效:** A=SGF → B=PKV → C=SGF 再来一次。
C 与 A 的 loss 和 `|g|` **逐位相同**(0.16560513 / 4.3970),进程内无状态污染。

**新增两条重要事实:**

1. **我之前报的所有余弦数字是坏的。** `cosine_similarity` 在 1419M 元素的 fp32
   向量上返回了 **1.07** 和 **1.98** —— 余弦不可能大于 1,点积在 fp32 累加器里
   就溢出了。0.83 / 0.40 / 0.89 这些值**全部无效**,不要用它们判断任何事。
   已改为 float64 且先归一化再点积。
2. **梯度比值在各参数组上是均匀的**,不是某层坏掉:
   `_fsdp_wrapped_module` ratio **19.04**,`...model._` ratio **17.58**。
   这是一个**全局缩放**。而 T=21,比值 ~18.6。

**又排除两条:**

5. ~~PKV 只有部分帧把梯度传到参数~~ — 直接测 `‖∂output[:,i]/∂θ‖`
   (`test_per_frame_grad.py`):8 帧各占 12.2%–12.8%,**没有一帧断路**,
   且 ckpt=True / False 逐位相同。
6. ~~多卡~~ — 集成测试是**单进程**(FSDP 因 world_size=1 退化成 NO_SHARD),
   18.6× 在单卡就复现。

**一个可能颠覆整个诊断方向的发现:**

DMD 的 loss 是梯度注入:
```python
dmd_loss = 0.5 * F.mse_loss(x.double(), (x - grad).detach().double())   # grad 在 no_grad 里算
```
即 `loss = 0.5·mean(grad²)`,**只取决于 x 的取值,完全不反映 x 是否连着参数**。
所以"loss 匹配到 1.18e-04"**不能作为两边一致的证据** —— A 前面几轮一直把它当
"前向正确",这是错的。

由此引出的怀疑:**集成测试的设计本身可能就是无效的。** 两条 arm **各自跑自己的
rollout** —— SGF 走 cached KV,PKV 走 page,两者每步差 kernel 地板(3.6e-03),
而 rollout 是自回归的:21 帧 × 4 步 = 84 次串行前向,差异会被逐步放大。
跑到最后两边可能生成的是**两段不同的视频**,那么比较它们的梯度就是在比较
**两个不同的训练样本**,18.6× 是预期行为而非 bug。

对照:已通过的严格测试(cos 0.99999994)是把 PKV 记录的张量**原样喂给 SGF**,
两边输入逐位相同 —— 那才是"同样数据下两条路径是否一致"。

**正在跑(629477):** 直接量两条 arm 生成视频的相对差。若很大 → 集成测试设计
无效,正确做法是强制两边用同一份 rollout;若很小 → 怀疑不成立,18.6× 另有原因。

**给 SGF2 的请求:** 这一条请重点看。A 已经连续排除 6 个假设而现象仍在,通常说明
前提里有一条错了 —— A 目前认为最可疑的就是"集成测试的比较对象"这个前提本身。

另注:`|d|` 两边**完全相等**(3.1250e-02),这是 AdamW 的逐参数归一化把梯度
幅度抹平了 —— 所以 test 3 只在**方向**上有判别力,幅度上没有。

## 4.1 **根因已定位到组合条件(2026-09-06 04:5x)**

故障已在 **8 层小模型**上精确复现(`grad rel = 0.954`,与 1.3B 集成测试的
`9.545e-01` 同一个数),且两边喂**完全相同的记录输入** —— rollout、DMD loss、
模型规模全部排除。迭代周期从十几分钟降到几分钟。

四路二分(`test_grad_strict.py`,`--fsdp` / `--bf16` 可独立开关):

| FSDP | bf16 | grad rel | cos | |
|---|---|---|---|---|
| ✗ | ✗ | 1.6e-04 | 0.99999988 | 通过 |
| ✗ | ✓ | 5.2e-03 | **1.0000000** | 方向完美,幅度在反传地板 5.36e-03 上 |
| ✓ | ✗ | 1.6e-04 | 0.99999988 | 通过 |
| ✓ | ✓ | **9.54e-01** | **0.73** | **崩溃** |

**FSDP 单独没问题,bf16 单独也没问题(余弦精确 1.0),只有组合才崩。**
这排除了"bf16 梯度累加精度"的猜测。

**当前主假设:** SGF 那条 arm 在一次 backward 前只有**一次**带梯度的 forward;
一趟法有 **2T = 42 次**。FSDP 的 `MixedPrecision` 在每次 forward 的 pre-hook 里
cast 参数、在 post-backward hook 里把 bf16 梯度归约回 fp32 master ——
**"一次 backward 前有 N 次带梯度 forward"** 是这套钩子记账没覆盖好的场景。
而这正是一趟法的结构特征,不是可绕开的实现细节。

**正在验证(629508):** 误差是否随带梯度 forward 的次数(2T)增长。
T = 2 / 4 / 8 / 21,若 rel 随 T 单调上升 → 坐实。

**如果坐实,修法的取舍(想听你意见):**
- (a) 把带梯度的循环收进**一次** `model.forward()`,让 FSDP 只看到一次 forward。
  代价:rollout 需要自己的 no_grad page,clean context 要算两遍 —— 那就退化成
  SGF 的两趟结构,一趟法的节省(21 次 no_grad clean write)没了。
- (b) 找 FSDP 侧的正确用法(`no_sync`、手动 reshard 控制等),保持交错结构。
- (c) 不用 FSDP 混合精度,改用 autocast + fp32 master。

## 4.2 **根因已确认(2026-09-06 05:0x)**

**`torch.utils.checkpoint` 包在 FSDP 模块外面 + FSDP MixedPrecision。**

判定实验(`test_grad_strict.py`,两边喂完全相同输入):

| 配置 | grad rel | cos | |
|---|---|---|---|
| FSDP + bf16 + checkpoint **开** | **9.54e-01** | 0.730 | 坏 |
| FSDP + bf16 + checkpoint **关** | 5.2e-03 | 0.9999859 | 好 |
| FSDP + fp32(ckpt 开) | 1.6e-04 | 0.99999988 | 好 |
| 无 FSDP + bf16(ckpt 开) | 5.2e-03 | 1.0000000 | 好 |

机制:FSDP 混合精度在 pre-forward hook 里造一份 bf16 参数副本供计算使用。
checkpoint 的**重算发生在 backward 中途**,那时 FSDP 的 forward 上下文已结束,
重算用到的副本不是建图时的那一份。fp32 下没有副本,所以不受影响;
不开 checkpoint 就不需要重算,也不受影响。

这解释了全部观察,包括为什么误差随带梯度 forward 次数 N 单调增长
(N 次前向 = N 次错位的重算),以及为什么五个 MixedPrecision 变体数字**完全相同**
(与配置无关,只要有副本 + 有重算)。

**⚠️ 我在诊断途中给出过两个自洽但错误的机制解释,记下来以免被引用:**
1. ~~"FSDP 的 `_post_backward_called` 守卫导致 42 份梯度只有 1 份被搬回 fp32"~~ —
   最小复现(`test_fsdp_nforward_minimal.py`,单个 nn.Linear,dL/dW 有解析解)
   直接否定:N=42 时梯度**范数完全对得上**(1.6690e+03 vs 1.6703e+03),
   误差仅 7.9e-03 的 bf16 舍入。**FSDP 一份梯度都没丢。**
2. ~~"checkpoint 本身有 4.8e-03 误差"~~ — 实际是**测试脚本里闭包晚绑定**。

两次都是"在源码/输出里看到一个东西,和现象接上,中间环节没验"。

## 4.3 两条修法的取舍

**(b) 把 checkpoint 移到 FSDP 边界内部(块级)** —— PyTorch 官方支持的组合
(`apply_activation_checkpointing` 就是这么做的,本仓库 `_forward_train` 也是)。
已实现:`PersistentKVWanModel._maybe_ckpt` + `persistent_block_checkpointing`,
pipeline 侧 `checkpoint_inside_model=True` 时不再在帧级 checkpoint。
**精度策略零改动,config 不动,real_score/fake_score 不受影响。** 显存待测
(checkpoint 粒度从帧级变块级)。

**(c) fp32 参数 + torch.autocast** —— 已验证可修复
(grad rel 1.73e-03,cos 0.9999983)。实测代价(真实 1.3B,21 帧,FSDP,仅 generator):

| | 峰值 | 激活 | 步时 |
|---|---|---|---|
| FSDP MixedPrecision | 29.43 | 24.14 | 16.78s |
| (c) autocast | 35.83 | 30.53 | 19.19s |
| 纯 fp32 | 47.15 | 41.85 | 38.27s |

**(c) 代价:显存 +22%,速度 +14%** —— 且还要 config 关 `mixed_precision`,
`real_score` / `fake_score` 会一起退回 fp32(未计入上表)。

**倾向 (b)**,但要先看它的显存数,以及它在无 FSDP fp32 下是否仍然通过。

## 5. 想请你看的具体问题

1. **`pipeline/persistent_kv_training.py` 的 `inference_with_trajectory`** —— 尤其
   no_grad rollout 和可微部分的边界。SGF 的 pass2 用 `clean_x = x_ctx_hat.detach()`,
   我的 write 输入也来自 no_grad,但 page 本身带梯度。这个对应关系对不对?
2. `_noisy` / `_write` 里 `checkpoint` 包住 `self.generator(...)` 的构造在 FSDP 下
   是否还有别的隐患(我只验了梯度精确性)。
3. 18.5× 这个比值你有别的解释吗?我目前想不出还有什么能让 loss 一致而梯度差这么多。
4. 有没有哪个已验证项的**判据本身**不成立(我用 kernel 地板当标尺,而不是固定阈值)。

## 6. 意见与回应

不写在本文件里,分开两份:

- SGF2 的建议 → **`docs/SGF2_REVIEW.md`**
- A 的逐条回应与实验结果 → **`docs/SGF2_REBUTTAL.md`**


---

# 第二轮:一趟法训出来的模型和 SGF 行为不同(2026-09-06 下午)

**现象(用户观察,已量化):PKV 生成的视频"动得太快"。**

相邻帧差 `mean |frame[t] - frame[t-1]|`(0–255):

| | step200 5s | step200 60s | step500 60s |
|---|---|---|---|
| sgf | 7.04 | 4.11 | 13.74 |
| **pkv** | **13.09** | **16.30** | **26.90** |

**约 2 倍,且持续存在**(两者都随训练增长,PKV 始终约 2×)。
差异在**前 21 帧内就存在**(12.4 vs 7.7)—— 训练视野之内,不是长视频外推的问题。

注意:这个指标测的**不是语义运动**。mem_lora 在此为 SGF 的 5–6 倍,但它的 VBench
`dynamic_degree` 反而更低(0.3048 vs 0.4098)。所以更可能是**逐帧噪声/不稳定**。

## 已排除的(每条都有实测,全部通过)

| 假设 | 结果 |
|---|---|
| 单步梯度不等价 | cos 0.9999761,ΔθAdamW cos 0.9996491 |
| 多步累积 / page 泄漏 | 30 步余弦变化 3e-9;weakref 每步全释放 |
| 多卡梯度聚合 | 真实配置(hybrid_full + 5e7 + 8 rank)最差 cos 0.9999949 |
| lr 不匹配 | 两边相同,trainer 无 world_size 缩放 |
| EMA 存档损坏 | 825 张量完整;且 step200 非 EMA 也异常 |
| 推理路径与 causal 不同 | 逐位相同(含流式 cache) |
| 推理精度错位 | `inference.py` 硬编码 bf16,与 config 无关 |
| **训练/推理路径错位** | page vs cache 在**真实配置**(21帧/local=18/sink=3/bf16)下 **0.000e+00** |
| **critic 的 no_grad 分支与 generator 不一致** | 同 pipeline 两种模式输出 **0.000e+00** |
| chunkwise | 已 guard 拒绝 |

## 我自己三个测量之间的矛盾(正在查,这是当前主线)

```
喂相同记录张量给两条 arm     3.4e-05
各自跑各自的 rollout        1.4e-02      ← 400 倍
page 路径 vs cache 路径     0.000e+00
```

rollout 路径逐位相同、RNG 抽取次数也相同(每块 3 次去噪 randn + 1 次 context
randn,两边核对过),那 1.4% 只能来自 **rollout 交给可微前向的张量本身**
(`noisy_at_t` / `x_ctx`)。而那正是模型训练的对象 —— **任何"喂相同输入比梯度"的
检查都永远看不到这种差异。**

`test_rollout_products.py` 正在直接比这两个张量。

## 已知的、未排除的配置差异

PKV 用 1×8,SGF baseline 用 2×4(有效 batch 都是 8)。
`per_rank_exit_step: true` 下每 rank 各采 exit step,所以一个 batch 里
SGF 有 4 个不同 exit step、PKV 有 8 个。用户认为这不足以造成 2 倍,我同意,
但它是目前**唯一**已知未排除的系统性差异。

## 想请 SGF2 看的

1. 上面那个 400 倍的矛盾,你看是哪一端的测量有问题?
2. 还有什么"系统性、单向"的机制能让每步梯度等价、却让训出来的模型持续更噪?
3. `pipeline/persistent_kv_training.py` 里 no_grad rollout 与可微部分的边界,
   和 `self_gradient_forcing_training.py` 的 `_self_gradient_forcing_inference`
   逐行对照,有没有我漏掉的语义差异(我的 pipeline 忽略了 SGF pipeline 接收的
   `same_step_across_blocks` / `independent_first_frame` /
   `denoising_step_list_first_chunk`,当前 config 都没设)。


### 第二轮进展(续)

**400 倍矛盾已解决 —— 是我测试 spy 的 bug。** `_noisy` 每块被调 5 次(4 去噪 + 1 可微),
我取了末尾 21 个混入了中间态。修正后:`noisy_at_t` **0.000e+00**,`x_ctx` **0.000e+00**,
final output 2.484e-03(= bf16 kernel 地板)。**同权重同 seed 下两条 pipeline 逐位等价。**

**init 基线拿到了,现象坐实且是双向的:**

| | 5s 相邻帧差 | 前 21 帧 |
|---|---|---|
| init (causal_cd.pt) | 8.72 | 11.20 |
| sgf @200 | 7.04 ↓ | 7.72 ↓ |
| pkv @200 | 13.09 ↑ | 12.44 |

SGF 训练把逐帧变化压低,PKV 把它推高。**两个模型朝相反方向离开了同一个起点。**

**权重空间**:cos(d_sgf, d_pkv) 0.25–0.30,幅度比 1.13×,各参数组均匀。
但两次长训练数据顺序本就不同(2×4 vs 1×8),这个数**不具判别力**。

**决定性实验正在跑(630226):** SGF 用与 PKV **完全相同**的配置(1×8、seed 0、
causal_cd.pt、同 config 只差 grad_mode)训 50 步,与 `logs_pkv/checkpoint_model_000050`
直接比权重。这是唯一把 pipeline 单独隔离出来的实验:
- 位移 cos ≈ 1、幅度 ≈ 1 → 算法等价,1500 步的差异是混沌 + 数据顺序
- 明显分开 → 存在系统性机制,且 50 步内可复现,可二分

**已排除新增**:训练/推理路径错位(真实配置 0.000e+00)、critic no_grad 分支
(0.000e+00)、rollout 产物(0.000e+00)。

**用户的类比值得记录**:现象"像 MemProj"(记忆写入与生成分开用 W 的那次)。
共用 W 却表现如分开 W,提示某处让写入表征与读取表征脱钩。源码审阅
`forward_clean_writer` / `forward_noisy_frame` 未发现,数值上也逐位一致。

---

## 第三轮(09-06 18:00)— 上面那个"决定性实验"的前提是错的,已重新设计

**发现:`seed: 0` 在这个 trainer 里不是种子,是"随机抽一个种子"**
(`trainer/distillation.py:36-39`:`if config.seed == 0: random_seed = torch.randint(...)`,且不打印)。
所以 `logs_pkv` 与 `logs_sgf_1x8_s0` 的噪声流、退出步、critic 时间步从 step 0 起就不同。
证据:两条 run 的 **step 1(critic 步,权重仍完全相同)** critic_loss 0.0172 vs 0.0188、
grad_norm 0.406 vs 0.263 —— 权重一样、算法逐位等价,只能是随机流不同。
**数据顺序仍然相同**:`DistributedSampler` 用默认 seed 构造,不依赖 `config.seed`。

因此 630226 的对比降级为"不同种子、不同 pipeline",cos 低不能归因于 pipeline。

**重新设计成 2×2(pipeline × seed),50 步:**

| | seed 随机 | seed = 1 |
|---|---|---|
| SGF | `logs_sgf_1x8_s0`(630226,已到 50) | `logs_sgf_1x8_s1`(630239) |
| PKV | `logs_pkv`@50(629727) | `logs_pkv_1x8_s1_50`(630242) |

判据(`tests/compare_traj_matrix.py`,等四个 ckpt 齐后由 630244 自动跑):
- cos(SGF_s1, PKV_s1):同种子、只差 pipeline
- cos(SGF_r, SGF_s1)、cos(PKV_r, PKV_s1):同 pipeline、只差种子 —— 混沌地板
- 前者 ≥ 后两者 → 50 步窗口内种子解释一切;明显更低 → pipeline 在推权重,可二分。
另附:seed=1 两条 run 的 step-1 critic_loss 若一致到 1e-3,说明两条 pipeline 的 RNG 消耗同步;
不一致则说明消耗次数不同(即使值等价),这本身就是要修的。

**对 S5(B→B)**:推理本就不是逐位确定的(同权重同 seed 8/8 不同,像素差 1.93),训练 B→B 也不会逐位相同;
上面的 2×2 用"同 pipeline 不同种子"给出噪声地板,替代逐位复现。
**对 S6**:已在代码里对齐(rank-0 抽 + broadcast + last_step_only,见 `persistent_kv_training.py:108-131` docstring);
当前配置走 `per_rank_exit_step` 分支,两边都是每 rank 各抽 size=(1,)。

**fp32 假设(用户提问)**:否。bf16 无 FSDP 那格 rel 5.2e-03 但 cos 1.0000000,是纯标量差,
Adam 不敏感,master/优化器/ckpt 都是 fp32;且两条 pipeline 用同一套混合精度,精度不是差异变量。

**60s@200 起点数据**:init 10.50 → SGF 4.11(↓)、PKV 16.30(↑)。方向相反,不是"多一点"。

**240s VBench**:SGF 的 imaging_quality / temporal_flickering 两个补测同秒启动,都对同一
`split_clip/` 重切(`split_video_into_clips` 每次重写、不跳过),互相读到半写的 clip → 假"坏文件"。
temporal_flickering 已成,imaging_quality 单独重跑(630237)。以后同一 videos 目录的维度必须串行。

**630226 结果(随机种子对随机种子,仅供参考,已被 2×2 取代):**
cos(d_sgf, d_pkv) = 0.233,|d_pkv|/|d_sgf| = 0.974,|d|/|θ| 均 ≈ 1.85e-4 —— 位移幅度相同、方向几乎不相关。
**行为(5s,同 8 条 prompt,推理 seed 0):** init 8.72 → sgf_1x8@50 **10.23**,pkv@50 **25.16**。
PKV 在 50 步内把逐帧变化推到起点的 3×,SGF 基本没动。位移幅度一样、效果差 3× → 权重动的方向不同。

**步 750(5s):** sgf 19.29,pkv 20.36 —— 到 750 步两者**几乎一样**。SGF 也会走到这个闪烁水平,只是慢得多
(sgf: 8.7 → 7.0@200 → 19.3@750;pkv: 8.7 → 25.2@50 → 13.1@200 → 20.4@750)。
所以现象不是"PKV 多闪一倍",更像"PKV 在训练早期就跳进 SGF 要几百步才到的区域"。

**seed=1 两臂的 RNG 同步性(权重相同的前几步):**
step 1 critic_loss SGF 0.0525 / PKV 0.0525,grad_norm 0.1739 / 0.1728;step 2 0.0889 / 0.0889,0.2910 / 0.2847。
loss 四位一致 → 噪声流、退出步、critic 时间步完全同步;grad_norm 差 0.6–2% 是 bf16 地板(两条 pipeline 生成结果差 2.5e-3)。
所以 cos(SGF_s1, PKV_s1) 只剩两个来源:pipeline 本身,或 bf16 地板被混沌放大 —— 后者的大小由同 pipeline 换种子那两格标定。

**60s 全程(相邻帧差,init 10.50):** @200 SGF 4.11 / PKV 16.30(4×)→ @500 13.74 / 26.90(2×)→ @750 **19.64 / 21.82(1.1×)**。
两条曲线在 750 步汇合。差的是**到达时间**,不是终点。

**seed=1 两臂生成器步(同噪声流):** step 6 gl 0.1065/0.1054;11 0.1085/0.1100;16 **0.2914/0.1243**;21 0.1594/0.1975;26 0.2850/0.2816;31 0.0889/0.0926。
只经过两次生成器更新(lr 2e-6)之后同一样本的 loss 就能差 2.3×,随后又靠回 —— 单步 loss 对权重极敏感,这就是需要换种子臂标定混沌地板的原因。

**2×2 结果(50 步):**

权重位移 cos:换种子 → 0.199(SGF)/ 0.226(PKV)/ 0.233(跨);同种子换 pipeline → **0.675**。四臂 |d|/|θ| 均 1.8e-4。
行为(5s 相邻帧差,推理 seed 0,init 8.72):

| | 随机 seed | seed = 1 |
|---|---|---|
| SGF | 10.23 | 13.65 |
| PKV | 25.16(重新生成 25.16,来源无误) | 14.95 |

**同种子下两条 pipeline 在 50 步既在权重上远近于种子差,行为上也只差 10%。**
之前看到的 pkv@50 = 25 是随机种子那条 run 的个体行为(SGF 换种子也从 10.2 变到 13.6)。
这在 50 步窗口内排除了"pipeline 系统性推高运动"的假设;但 n=2/格,且只覆盖 50 步。
用户要求逐行代码审计(pipeline 语义 / 模型 attention-page 路径 / wrapper-base-trainer 接线),进行中。

## 逐行代码审计(09-06 19:00,三路并行,只读)

用户要求不接受统计等价、要逐行查实现。三路各自独立读代码并给出 file:line 对照,结论:
**在当前配置实际走到的每一条路径上,没有 BUG 级差异。** 已核对为一致的项(均有双方行号):

- **pipeline 语义**:帧数(base.py 抽 21 固定)、退出步 randint 形状/范围、每步时间步、随机数调用次数与顺序(1 randint + 每帧 (S-1)+1 randn_like)、exit 捕获点、输出定义(noisy_at_t 在 train_t 上带梯度重算 x0)、x0_cache=x0_exit、x_ctx 的 t=0 加噪、writer 时间步、返回元组与 `denoised_timestep_from/to` 算式(且 `ts_schedule: false` 下该值根本未被使用)、critic no_grad 下的分支。
- **模型层**:可见集合逐 i 推导(TF mask / cache 读 / persistent)在 i=0..20 完全同集、i=21 才分叉、guard 在 22 帧拒绝;RoPE 绝对位置与频率表;K/V 顺序(norm→rope,V 不 rope)与 dtype;时间步调制 e0 形状;page 无 detach/no_grad/原地写;`_maybe_ckpt` 捕获的 history 不被后续 append 修改;LoRA 未构造;dispatch 无静默回落;参数集合与顺序一致(state_dict 键相同)。
- **接线**:wrapper 的 timestep/context/permute/flow→x0 转换与 kv_cache 分支同源(同一段代码 `:411-415`);模型构造 kwargs 一致;optimizer 参数组、EMA、grad clip、FSDP 单元边界与 grad_mode 无关;`gradient_mask` 由 base.py 按帧数算,21 帧下恒为 None。

**审计发现的非 BUG 项(已处理/待处理):**
1. `persistent_kv: true` 在**推理**时也会让 `CausalInferencePipeline` 构造 `PersistentKVWanModel`(pipeline/causal_inference.py:21 直接用 model_kwargs)。共享代码 diff 为零、我们此前实测推理输出逐位相同,所以数值上无影响;但"推理不受影响"只在字面上成立于不含该 flag 的 config。要彻底满足字面要求需在推理侧忽略该 flag(改共享文件,待用户批准)。
2. 参考 pipeline 的两条构造期 guard(去掉 `denoising_step_list` 末尾 0;校验 cache_mode)我们的 pipeline 没有 → **已补齐**(仅 guard,不改算法;当前配置下无影响)。
3. `last_step_only`:base.py 没把它传给我们的 pipeline(参考 pipeline 收到)。当前配置走 per_rank 分支,两边都不读它;若关掉 per_rank_exit_step 就会分叉。修法是 base.py 一行传参(共享文件,待批准)。
4. 配置头注释声称"只差两行"但 generator_ckpt 有意改成 causal_cd.pt → 注释已改正,并注明 seed 0 = 随机。
5. 注释引用的 `test_window_covers_all.py` 在仓库外(/mnt/data/zhaofeng/runs/sgf/tests/)→ 注释已改成完整路径。
6. 副发现(两种方法同等受影响):21 帧、local 18 + sink 3 下三套可见规则都退化为**全因果注意力**,训练里 window/sink 实际没生效;推理 cache 只有 18 帧、在 i=18..20 会丢帧 3..5 —— 这是参考实现本来就有的训练/推理错位,PKV 原样继承。
7. 梯度路径 kernel:我们用 flash-attn 反向(默认非确定性 dK/dV 累加),SGF 二趟用 flex_attention → 3.6e-3 的 run-to-run 噪声,不是偏差。

**@1000(用户要求追加):** 5s SGF 18.80 / PKV **15.48**;60s SGF 16.82 / PKV 16.33。
5s 曲线:SGF 10.2 → 7.0 → 19.3 → 18.8;PKV 25.2 → 13.1 → 20.4 → 15.5(@50/200/750/1000)。**750 步后交叉,1000 步 PKV 反而更低。**
结合同种子 50 步对照与三路审计:早期 2× 差异是轨迹的随机波动,不是 pipeline 的系统性偏移。

**共享文件的两处改动(用户已批准;批准前状态存于 commit 3e61bd4 与 /mnt/data/zhaofeng/runs/sgf/backup_pre_shared_edit_0906_2026/):**
- `model/base.py`:给 `PersistentKVTrainingPipeline` 传 `last_step_only=self.args.last_step_only`(与参考 pipeline 一致)。
- `pipeline/causal_inference.py`:构造采样器前从 `model_kwargs` 去掉 `persistent_kv`,推理永远用 `CausalWanModel`。
验证(GPU):cfg_pkv.yaml → 推理类 CausalWanModel、训练 wrapper 仍 PersistentKVWanModel;pkv@1000 与 sgf@1000 的 ckpt(825 个张量,全部带 FSDP 前缀)去前缀后 **strict=True** 加载进 CausalWanModel 均 0 缺失 0 多余;inference.py 端到端生成通过。
附带发现(参考实现原有):inference.py 的回退路径去前缀后用 strict=False 加载,键不匹配会被静默跳过;上面的 strict=True 结果说明当前两类 ckpt 都没有这个问题。

**@1300:** 5s SGF 11.21 / PKV 12.25;60s 9.70 / 11.34 —— 两者都回落到起点附近,差 10–17%。

## KVForcing(09-06 晚)— 用户要求的可开源干净仓库

位置 `causal/KVForcing`。基底 = 本仓库第一次提交(上游 Causal-Forcing + sink/window + 长视频推理),去掉两趟 pipeline 与所有实验副本。
- 一趟算法不再是 1600 行的模型副本:page store 与两条 persistent 前向以 `# ======== ours ========` 标记插入 `causal_model.py`
  的 attention / block / model 三层;同一个类既训练又推理,`persistent_kv` 开关、wrapper 换类、推理侧去 flag 全部消失。
- `pipeline/kv_forcing_training.py`(ours)、`pipeline/self_forcing_training.py`(sf 与 no_sg 合成一个 rollout,只差 stop-gradient),
  `grad_mode ∈ {self_forcing, self_forcing_no_sg, kv_forcing}`;配置键 `match_context` / `cache_mode` / `per_rank_exit_step`。
- 检测(全部在仓库外 `/mnt/data/zhaofeng/runs/sgf/kvtests/`):小模型 page==cache 0.000e+00、checkpoint 精确、梯度可达 ✔;
  进行中:tiny 跨仓库(新 sf/no_sg/kv_forcing vs 旧 sf/no_sg/两趟/一趟)、真实配置跨仓库单步(新 kv_forcing vs 旧两趟 SGF)、
  8 卡逐 rank(hybrid_full/full)、50 步 1×8 seed 1 训练对照(vs 旧 logs_pkv_1x8_s1_50 / logs_sgf_1x8_s1)。

**KVForcing 检测结果(仓库外 kvtests/,对照臂各在独立进程,同种子):**
- tiny(6 层、21 帧、fp32):新 sf vs 旧 sf grad rel 4.1e-08 cos 1.0000000000;新 no_sg vs 旧 no_sg rel 1.2e-05 cos 0.9999999999;
  新 kv_forcing vs 旧两趟 SGF rel 2.4e-04 cos 0.99999997(flash vs flex 内核);新 kv_forcing vs 旧一趟 rel 2.0e-06 cos 1.0000000000;
  对照组 sf vs kv_forcing cos 0.9936 rel 0.11 —— 测试能分辨 stop-gradient 与全梯度。
- 真实配置(1.3B、21 帧、bf16 FSDP、三模型同卡):新 kv_forcing vs 旧两趟 SGF loss rel 1.18e-04、grad cos 0.9999761、Δθ cos 0.9996489、
  rollout 视频差 1.4e-02(bf16 自回归漂移);与旧一趟:loss 逐位相同、grad cos 0.9999984、视频 0.000e+00、峰值 47.94 vs 47.95 GiB。
  单步峰值(含 mask 尖峰):两趟 69.57 GiB / 一趟 47.94 GiB;fwd+bwd 20.1 s / 20.4 s。这些数字与旧实现当时测得的完全一致。
- 8 卡逐 rank(hybrid_full 与 full):最差 cos 0.9999950,rel 3e-3。
- 50 步 1×8 seed 1 训练:前三步 critic loss 与旧仓库同种子两臂逐位一致(0.0525 / 0.0889 / 0.0728);位移矩阵待出。

**50 步位移矩阵(全部 seed 1、1×8、causal_cd.pt):**

| cos | sgf(旧两趟) | pkv(旧一趟) | kvf(新一趟) | sgf 随机种子 |
|---|---|---|---|---|
| sgf_s1 | 1 | 0.675 | 0.633 | 0.199 |
| pkv_s1 | | 1 | **0.658** | 0.199 |
| kvf_s1 | | | 1 | 0.195 |

新旧一趟(单步 grad cos 0.9999984、rollout 逐位相同、只差 flash-attn 反向的非确定性累加)跑 50 步后 cos 0.658 ——
与"一趟 vs 两趟"的 0.675 同一水平。这就是 S5 要的 B→B:同一算法自身的 50 步复现度 ≈ 0.66,
所以 0.675 分辨不出 pipeline 差异,只分辨得出种子(0.2)。迁移在训练层面闭环。

**@1500(终点):** 5s SGF 7.56 / PKV 11.93;60s SGF 7.00 / PKV 10.38。两者都低于起点(8.72 / 10.50),PKV 约为 SGF 的 1.5×。
5s 全程:SGF 10.2 → 7.0 → 19.3 → 18.8 → 11.2 → 7.6;PKV 25.2 → 13.1 → 20.4 → 15.5 → 12.3 → 11.9(@50/200/750/1000/1300/1500)。
两条曲线都大幅起伏、在 750–1000 交叉;终点差 1.5× 与 SGF 自身换种子在 50 步就出现的 1.3× 同量级,单次 run 分不出是 pipeline 还是种子。
要在 1500 步上下结论需要同种子的两条 1500 步 run(各 8 卡 × ~8 h),由用户决定。

**240s 官方 VBench 补齐(imaging_quality 单独重跑后 7/7):** quality_score SGF 0.8158 vs MemLoRA 0.8186(+0.0028);
60s 为 0.8221 vs 0.8257(+0.0036)。两个时长上 MemLoRA 在 6 个一致性/质量维度略高、dynamic_degree 明显更低(0.41 → 0.30),
官方加权(dynamic_degree 权重 0.5)后仍是 +0.003 左右的小幅领先。

## KVForcing + Memory-LoRA(用户要求:加法与 mem_lora 完全一致,并验证行为完全一致)

实现(全部带 `ours` 标记,默认关闭):`MemoryLoRA`(rank 64、down ~N(0,1/rank)、up 零、scale 1)挂在 `CausalWanSelfAttention` 上(`memory_lora=True` 才构造);
路由与 mem_lora 逐条相同 —— cache 路径 `memory_write=True` 的前向、TF 路径的 clean 半边、persistent 路径的 page 写入(`write_page`)用 (W+dW)h,
dW 加在 norm_k 之前;`init_weights` 末尾重置 dW=0;`_forward_inference` 透传 `memory_write`;wrapper `memory_lora` 构造开关 +
`memory_lora_write` 前向标记(kv_cache 分支传给模型);`CausalInferencePipeline` 的 cache 刷新标记 `memory_lora_write=True`;
SF 基线的 context 写入同样标记;trainer 为 `memory_lora` 参数单独 lr(`lr_memory_lora`);`scripts/make_lora_init.py` 生成带零 adapter 的 init ckpt;
配置 `configs/kv_forcing_lora_framewise.yaml`。
验证(进行中,630756 / 630758–630760):tiny 同 adapter 权重且 up≠0 下 新 kv_forcing+LoRA vs 旧两趟 mem_lora / 旧一趟 PKV+LoRA;
真实配置单步(共用旧 lora-init,up 扰动 1e-3)含 LoRA-only 与 backbone-only 的梯度 cos;推理 latents 对比(旧 MemoryLoRAInferencePipeline vs 新 CausalInferencePipeline,A-vs-A 作地板);
50 步 1×8 seed 1:新 kv_forcing+LoRA vs 旧 memory_lora_forcing,比 adapter `up`(即位移)与 backbone 位移的 cos。

**KVForcing + Memory-LoRA 验证结果(第一轮):**
- tiny(同 adapter 权重、up≠0):新 kv_forcing+LoRA vs 旧两趟 mem_lora grad cos 0.9999999739、rel 2.3e-04、loss rel 1.8e-07、rollout 2.9e-05;
  vs 旧一趟 PKV+LoRA cos 1.0000000000、rollout 0.000e+00;有/无 LoRA 的 loss 0.32816 vs 0.32756(adapter 确实在起作用)。
- 推理:同 lora ckpt(up 扰动 1e-2)、同噪声同 prompt,旧 MemoryLoRAInferencePipeline 与新 CausalInferencePipeline 的 latents **逐位相同**(0.000e+00);
  A-vs-A 也逐位相同;adapter 扰动 vs 置零差 67%(路由确实生效)。
- 真实配置:新臂正常(峰值 48.12 GiB);旧两趟 mem_lora 臂在 backward 报 cuDNN INTERNAL_ERROR(疑似显存边缘,无 LoRA 时两趟就已 69.6 GiB),
  正用 expandable_segments 重试,失败则两臂都降到 13 帧比较。
- KV Forcing(无 LoRA)的 1500 步完整训练已启动(kvruns/logs_kv_forcing,seed 1,1×8);LoRA 版等真实配置对照通过后启动。

**LoRA 逐张量诊断:** tiny(fp32)两趟 mem_lora vs 新一趟 kv_forcing+LoRA:24 个 adapter 张量全部 cos 0.999999(up、down 均然)—— 结构等价。
真实配置(bf16):`down` 梯度 cos 0.99998,`up` 梯度逐层 cos≈0(范数一致)。`up` 的梯度纯来自记忆路径(约 backbone 的 1%),
而两臂 rollout 在 bf16 下差 1.5%,疑为小信号被 bf16 噪声淹没;A-vs-A 重跑与 fp32 少帧对比正在跑以判别。
**50 步 seed 1 训练(同 lora-init):** kv_forcing+LoRA vs 两趟 mem_lora,adapter `up` 位移 cos **0.599**(|v| 0.727 vs 0.755),backbone 位移 cos 0.649 ——
与无 LoRA 时一趟 vs 两趟(0.675)、同算法重实现(0.658)同一水平;前 6 步 loss 逐位一致。

**LoRA 判别结果(A-vs-A):** 同一臂用同一种子重跑,adapter `up` 梯度自身就不复现 —— 一趟 k/up cos -0.057、v/up -0.005;两趟 mem_lora k/up -0.017、v/up -0.047;
而 `down`(0.9999)与 backbone(0.9999)都复现。所以真实配置 bf16 下"两实现 `up` 梯度 cos≈0"就是各自的自复现地板:`up` 的单步梯度(up≈0 时纯记忆路径、约 backbone 的 1%)
被 flash-attn 反向的非确定性与 bf16 噪声淹没,这是方法本身在该精度下的性质,不是实现差异。结合 tiny fp32 结构等价(24 张量 0.999999)、
推理 latents 逐位相同、50 步 adapter 位移 cos 0.60(与 backbone 地板同级),判定 KVForcing+LoRA 与 mem_lora 行为一致。
**KV Forcing + LoRA 的 1500 步完整训练已启动**(kvruns/logs_kv_forcing_lora,seed 1,1×8,新生成的 causal_cd_lora_init.pt)。
fp32 补充对照:base.py 把最少生成帧数硬编码为 21,少帧无法跑;两趟臂 21 帧 fp32 三模型同卡超 80 GB。A-vs-A 已足以定性,此项不再补。

**KVForcing 两条完整训练完成(09-07):** `kvruns/logs_kv_forcing` 与 `kvruns/logs_kv_forcing_lora`,各 1500 步、1×8、seed 1,无报错,
checkpoint_model_001500 均已落地(>200 步为 EMA 权重)。19 vs 21 s/step 的差主要是节点功耗限频差异,LoRA 固有开销约几个百分点。

**KVForcing 两条 1500 步 run 的 @1500 评测(5s / 60s 相邻帧差):** sgf 7.56 / 7.00,pkv 11.93 / 10.38,mem_lora 6.60 / 5.67,**kvf 6.74 / 6.61**,**kvf_lora 20.91 / 20.46**。
kvf 与 sgf、mem_lora 同级;kvf_lora 在两个时长上都是其他方法的 3 倍且 8 条 prompt 一致。
**归因交叉检查:** 旧 mem_lora@1500 走新推理 6.601(=旧 6.602);kvf_lora@1500 走旧推理 20.904(=新 20.906)→ 推理无关。
kvf_lora@1500 把 adapter `up` 置零再采样仍 19.75 → 闪烁在 **backbone 权重**里,不在 adapter 里:带 adapter 训练把 backbone 推到了闪烁区,
而同种子、无 adapter 的 kvf 没有。训练日志(loss / grad norm 逐 250 步窗口)三条 run 无异常。
进行中:adapter 幅度增长曲线(kvf_lora vs 旧 mem_lora)、kvf_lora 5s@250..1250 与 60s@500/1000 轨迹、mem_lora 全程 5s/60s 运动量表(用户记得"短变快、长变慢")。

**adapter 增长曲线(|up| / |dW|_k / |dW|_v,逐 250 步):** kvf_lora 与旧 mem_lora 几乎重合 —— @1500 |up| 3.65 vs 3.34、|dW|_k 2.20 vs 1.91、|dW|_v 1.60 vs 1.48
(|W_k| 291,dW 约 0.7%);@50 两条 seed-1 短 run 0.727 vs 0.755。**adapter 动力学在两种实现下相同**,排除"一趟下 up 梯度噪声被 Adam 放大"。
**kvf_lora 5s 轨迹:** 250 → 10.1,500 → 8.8,750 → 13.5,1000 → 18.8,1250 → 18.8,1500 → 20.9;同种子无 LoRA 的 kvf:500 → 7.8,1000 → 9.3,1500 → 6.7。
从 750 起持续偏高而非终点偶然。对照历史:一趟无 LoRA 的 pkv(随机种子)曾 50 → 25、750 → 20、1500 → 11.9,两趟 sgf 750 → 19.3。
n=1/格,kvf_lora=20.9 vs mem_lora=6.6 目前分不出"一趟+LoRA 系统性偏闪"还是种子方差;需补种子(kvf_lora seed 2、kvf seed 2、mem_lora seed 1)才能定。

**一趟是否引入比两趟更多的数值误差(用户提问):** tiny 同权重同输入,各 pipeline 的 bf16 梯度对自身 fp32 梯度 —— 两趟 rel 4.4e-03(cos 0.9999902),
一趟 rel 6.3e-03(cos 0.9999801),一趟 bf16 vs 两趟 bf16 rel 5.3e-03。一趟多约 40%,来源是 42 次 backward 贡献在 bf16 `.grad` 中累加(两趟 1 次)+ page 拼接;
真实规模下一趟 vs 两趟 6.9e-03 与各自 A-vs-A(~1e-02)同级,测不出额外误差;fp32 下 2.4e-04 为纯内核差,无偏差。可用 fp32 梯度累加抹掉,代价是显存。

**误差来源定位(用户:"我们是重放,为什么会多误差"):** 重放本身零误差(checkpoint 开关梯度逐位一致)。多出的部分全部来自 **bf16 梯度累加**:
tiny 同权重同输入,各 pipeline 的 bf16 梯度对自身 fp32 —— 两趟 bf16 累加 4.447e-03、fp32 累加 4.434e-03(不变,每参数只有 1 次贡献);
一趟 bf16 累加 6.342e-03、**fp32 累加 4.125e-03**(降到两趟以下)。一趟 vs 两趟在 fp32 累加下 2.1e-03(bf16 累加下 5.3e-03)。
机制:一趟每个参数在带梯度前向中被用 42 次(21 帧 × noisy + writer),42 个 bf16 贡献在 bf16 `.grad` 中逐次相加;两趟只有 1 次。
FSDP MixedPrecision(param bf16)下训练同样如此。修法:fp32 master 参数 + bf16 autocast 计算(每次使用时 cast,累加在 fp32),
显存代价约 +2.8 GB 未分片参数副本;是否改训练精度策略由用户定(会影响与两趟基线的可比性,需同策略重训对照)。

**mem_lora(两趟+LoRA,随机种子)全程运动量:** 5s 50 → 11.4,100 → 10.6,**200 → 36.1,250 → 32.3,450 → 24.6**,1500 → 6.6;
60s 50 → 13.5,100 → 9.2,**200 → 33.5,250 → 31.2,450 → 21.3**,1500 → 5.7。
**kvf_lora 60s:** 500 → 7.5,**1000 → 21.8**,1500 → 20.5(5s:750 → 13.5,1000 → 18.8,1500 → 20.9)。
→ 两趟+LoRA 自己在 200–450 步经历过一次更大的闪烁漂移(短、长视频同时到 31–36),到 1500 回落;kvf_lora 的漂移从 750 起、1500 时仍在 ~20。
同一现象、不同时机,与"一趟特有的 bug"不符,与 LoRA 训练存在闪烁漂移阶段 + 混沌时机一致。kvf_lora 若继续训练是否回落,未知。
用户记的"mem_lora 短快长慢":1500 步时短、长都比起点慢;200–450 时短、长都快;长视频 VBench dynamic_degree 低(0.30 vs 0.41)是"长慢"的来源。

**mem_lora 5s 全程(补齐):** 50 → 11.4,100 → 10.6,**200 → 36.1**,250 → 32.3,450 → 24.6,500 → 22.2,750 → 18.1,1000 → 13.7,1300 → 8.0,1500 → 6.6 ——
一次大漂移后单调回落。kvf_lora:250 → 10.1,500 → 8.8,750 → 13.5,1000 → 18.8,1250 → 18.8,1500 → 20.9 —— 同类漂移、更晚发生、1500 时未回落。

**显存重测(块级 checkpoint、T=21、单卡 bf16、仅 generator):** 一趟峰值 20.68 GiB(TF 首步 40.01)。峰值存活:page 5.63、page 梯度 5.35(旧分解里的"concat 40%"
其实是 cat 反向切成 view 的 page 梯度,把 60 个 21-page 大缓冲钉住;改成 `_PageCat` clone 切片后大缓冲消失、总量不变 —— **不可消**)、
块间隐状态 5.39(块级 checkpoint 边界输出,21×30×2)、反向临时 2.84。可动项只有块间隐状态 → 新增 `kv_forcing_checkpoint_group`(每 G 块一组 checkpoint,数值逐位不变,
G=5 下 page==cache 0 / ckpt 精确 / 梯度一致),G 扫描进行中。

**checkpoint 分组扫描(T=21、单卡 bf16、仅 generator,峰值激活 / fwd+bwd 时间):** G=1 20.46 GiB / 25.3 s;G=2 18.00 / 19.7;G=3 17.41 / 17.3;
**G=5 17.34 / 15.9**;G=6 17.52 / 15.3;G=10 18.58 / 14.4;G=15 20.17 / 14.2;G=30 25.29 / 13.7。G=5 比逐块省 3.1 GiB 且快 37%,数值逐位不变;
默认改为 5。此时一趟峰值 17.3 GiB < 两趟稳态 21.8(两趟首步 40–48)。
**240s @1500:** sgf 6.51 / pkv 11.43 / mem_lora 5.07 / kvf 6.92 / kvf_lora 19.70 —— 与 5s、60s 同一格局。
**用户决定:所有训练改 fp32**(不做 autocast 修法)。注意 flash-attn 只接受半精度,`attention()` 会把 fp32 q/k/v cast 成 bf16 再 cast 回来,
所以"fp32 训练"= 权重/激活/梯度累加 fp32、注意力核心 bf16。fp32 显存与速度、1×8 冒烟进行中。

**fp32 显存/速度(T=21、单卡、仅 generator,峰值激活 / fwd+bwd):** 一趟 fp32 G=1 40.20 GiB / 30.8 s,**G=5 33.38 / 26.0 s**,G=10 34.74 / 25.7 s(权重 5.30);
两趟 TF fp32 40.02 GiB / **211 s**(flex_attention 在 fp32 下极慢;一趟走 flash 内部 cast bf16 不受影响)。
对比 bf16 一趟 G=5(17.34 / 15.9 s):fp32 +93% 显存、+64% 时间。1×8 fp32 冒烟(20 步)进行中。

**fp32 冒烟(1×8、20 步):** 全 fp32:~32 s/step,nvidia-smi 峰值 80.9 GB(顶到天花板);加 `real_score_bf16`(冻结 teacher bf16、输入进门时 cast):
~21 s/step,nvidia-smi 79.9 GB(reserved,不反映 allocated;trainer 已加 `[mem]` 行打印 allocated/reserved)。无 OOM,loss 正常。
**已启动 fp32 完整训练(用户批准):** kvruns/logs_kv_forcing_fp32 与 logs_kv_forcing_lora_fp32,1×8、seed 1、G=5、teacher bf16。

**两趟 SGF 全 fp32 训练冒烟(旧仓库、1×8、10 步,无 teacher-bf16 选项):** 未 OOM;nvidia-smi 峰值 79.5 GB;`[timing]` 步 1–5(纯 critic 步,含首步 mask 尖峰)23.4 s/step,
步 6–10(含 1 个 generator 步)35.5 s/step → generator 步约 84 s、5 步周期均值约 35 s/step。一趟全 fp32 冒烟约 32 s/step;一趟 fp32 + teacher bf16 约 21 s/step。
注:单卡 TF fp32 的 211 s 含 flex_attention 首次编译,不代表稳态。
