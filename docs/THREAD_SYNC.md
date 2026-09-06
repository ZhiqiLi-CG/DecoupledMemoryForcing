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
