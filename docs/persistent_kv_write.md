# Persistent Differentiable KV Write

一趟的自回归训练算法。目标:得到与 SGF 两趟**等价的梯度**,但只做一趟前向。

实现在 `wan/modules/persistent_kv_model.py`,**独立文件**,不修改任何共享代码。
`causal_model.py` / `causal_inference.py` / `model/base.py` / `model/dmd.py` /
`trainer/distillation.py` 全程未改动。

状态:核心算法冻结。数值等价与显存已验证(第 4、5 节),接线已完成,
**integration validation 待做**(第 4.1 节说明了已证明与未证明的边界)。

---

## 1. 要解决的问题

SGF 的两趟结构:

- **Pass 1** 整段 AR rollout,全程 `torch.no_grad()`
- **Pass 2** 一次并行 teacher-forced 前向,`clean_x = x_ctx_hat.detach()`

梯度不进入 rollout。代价是整段前向做了两次。

已有的一趟做法(Self Forcing、MDF)都靠**切断跨帧梯度**来省显存:KV cache 是一块
被反复覆写的 buffer,写进去的东西必须 `detach()`,否则 autograd 图会引用被后续帧
改写过的内存。所以第 i 帧的 loss 到不了第 j<i 帧。

## 2. 核心想法:不可变的 page

不覆写 buffer,而是让每一帧**返回**它写的 K/V,由一个只追加的容器持有:

```python
class PersistentKVStore:
    def append(self, layer_idx, k, v):    # 不 detach,不写 buffer
        self.k_pages[layer_idx].append(k)
        self.v_pages[layer_idx].append(v)

    def snapshot_all_layers(self, visible_ids):   # 返回 tuple,天然不可变
        ...
```

每个 page 一旦写入就再不改变。由此:

- **跨帧梯度是通的** —— 第 i 帧读的是第 j 帧那个具体的张量,autograd 图完整
- **可以做 activation checkpoint** —— 这是关键。`use_reentrant=False` 的
  checkpoint 在反传时**重算**前向,要求重算看到的输入与首次前向**逐位相同**。
  被覆写的 buffer 做不到这一点(重算时读到的是后续帧写进去的值),不可变的 page
  可以。没有不可变性就没有 checkpoint,没有 checkpoint 一趟法在 T=8 就 OOM。

## 3. 每帧两个前向

```python
for i in range(T):
    hist = store.snapshot_all_layers(all_written_so_far)
    out   = checkpoint(model.forward_noisy_frame,  latent_i, t_noisy, ctx, hist, i)
    pages = checkpoint(model.forward_clean_writer, latent_i, t_clean, ctx, hist, i)
    store.append_all_layers(pages)
```

- `forward_noisy_frame` — 带 head 和 unpatchify,产生这一帧的去噪预测。**只读**历史。
- `forward_clean_writer` — 不走 head,返回 `tuple(k0,v0,k1,v1,...)` 共 2×num_layers
  个 page。**只写**。

两者共用 `_embed_frame`(patchify + 时间/文本嵌入)。

注意力里因果性**不靠 mask**,靠"只拼接过去的 page":

```python
if len(history_k):
    k_attn = torch.cat((*history_k, k_rope), dim=1)   # persistent_kv_model.py:524
    v_attn = torch.cat((*history_v, v),      dim=1)   # persistent_kv_model.py:525
y = attention(q_rope, k_attn, v_attn)
```

这也是它不需要 `create_block_mask` 的原因 —— 见第 5 节的显存后果。

## 4. 验证

| 验证项 | 结果 | 脚本 |
|---|---|---|
| 前向 vs 官方 cached 串行 | 逐帧相对差 **0.000e+00**,page 内容 == KV cache | `test_persistent_equiv.py` |
| checkpoint 是否改变梯度 | 69 个参数张量最差相对差 **0.000e+00** | `test_persistent_grad.py` |
| 梯度是否截断 | 末帧 loss 到达 frame 0 的 page(5.33e-03),无零值 | 同上 |
| vs SGF 并行 TF | **等价**,见下方标尺 | `test_persistent_vs_sgf.py` |
| RoPE 逐帧 vs 整段对齐 | **0.000e+00**(错位一帧则为 2.4e-01) | `test_rope_alignment.py` |
| 两个 attention kernel 的分歧地板 | 前向 3.56e-03,反传 5.36e-03 | `test_kernel_floor.py` |
| Memory-LoRA 路由 | 见第 4.2 节,五项全过 | `test_persistent_lora.py` |
| **pipeline 级** vs SGF pass2 | output 3.12e-05,全局梯度 2.19e-04 / cos 0.99999988 | `test_pipeline_vs_sgf.py` |

`test_persistent_grad.py` 里 frame T-1 的 page 梯度为 0 是**正确行为**(它写在 loss
之后,没有后续帧读它),不是失败。探针有效的证据是 frame 0/1 非零。

### 等价性怎么判的

两条路径用**不同的 attention kernel**(SGF 走 flex_attention + block mask,一趟法走
flash + 拼接过去的 page),不可能逐位相等,所以任何固定阈值都是拍脑袋。改为先量出
标尺,再看观测值落在哪:

| | 相对差 |
|---|---|
| 真的结构性错误(rope 差一帧) | 2.4e-01 |
| 两个 kernel 单独的分歧,反传 | 5.36e-03 |
| 两个 kernel 单独的分歧,前向 | 3.56e-03 |
| 实测 output | **1.15e-04** |
| 实测 KV 梯度 | **3.13e-03** |
| 实测全局梯度向量 | **1.75e-04**,余弦 **0.99999988** |

即两条路径的差异比它们各自依赖的 kernel 之间的差异**还小 30 倍**。

逐张量最差是 1.154e-02(`blocks.0.cross_attn.k.bias`),但那个张量的梯度范数
1.865e-04,比模型最大梯度小 630 倍 —— 用最容易被舍入淹没的那一格代表全体没有意义,
训练跟随的是全局梯度方向。

两个失败的判据方法记录在此以免重犯:(a) 用"同一路径跑两遍"当噪声底 —— 两条路径都是
确定性的,底恒为 0,退化成要求逐位相等;(b) 用模型内的 rope-off-by-one 当阳性对照 ——
随机初始化的模型几乎不读位置,该对照只有 1.157e-04,和被检验的量同级,证明不了任何
事。rope 最终由 `test_rope_alignment.py` 直接单测解决(无模型、无阈值)。

### 4.1 pipeline 级等价 —— 已证明与未证明的边界

**已证明:** persistent differentiable path 与 SGF 的 differentiable Pass-2
computation 在数值上等价。

**尚未证明:** 整个训练更新与 SGF 等价。那需要接入训练后的 integration
validation:同 batch 同 RNG 的完整 training step 的 loss、optimizer 之前的全局
梯度、以及 AdamW step 之后的参数 delta 方向。在这三项通过之前,不能说
"the overall training update is equivalent to SGF"。

### 4.1.1 pipeline 级数值等价

`pipeline/persistent_kv_training.py` 结构严格对齐 `_self_gradient_forcing_inference`:
同样的 exit step 采样、同样的 context 加噪、同样的 x0 缓存模式。唯一变化是第二趟没了,
它的工作发生在 rollout 内部。

模型等价不等于接进去训练的东西等价 —— pipeline 还要采样 exit step、跑去噪循环、给
context 加噪、决定写什么。所以单独测:运行时**记录一趟 pipeline 喂给可微前向的确切
张量**,再把同样的张量交给 SGF 的 pass 2,两边 rollout 由构造相同,差异只可能来自
一趟结构本身。

| | 相对差 |
|---|---|
| output(DMD loss 消费的 x0) | **3.124e-05**(比前向 kernel 地板低 114 倍) |
| 全局参数梯度 | **2.193e-04**,余弦 **0.9999998808**(比反传地板低 24 倍) |

### 4.2 Memory-LoRA(`memory_lora=True`,默认关闭)

记忆写入器用 `(W + ΔW)`,生成路径用 `W`,无 stop-gradient:

```python
kk = self.k(x); vv = self.v(x)
if memory_write and self.memory_lora:      # 只在 memory writer 上生效
    kk = kk + self.k_memory_lora(x)
    vv = vv + self.v_memory_lora(x)
k = self.norm_k(kk)                        # dW 在 norm_k 内部
```

**路由在一趟法里是结构性的:** `forward_noisy_frame` 恒传 `memory_write=False`,
`forward_clean_writer` 恒传 `True`,两个独立函数各自写死。两趟版本需要在 TF 序列上
mask 掉 noisy 那一半、并往 cache 路径透传 `memory_write` 标志 —— 那条链路曾经因为一次
静默的 `str.replace` 失配而整条断掉、几小时后才暴露。这里没有那条链路。

| 检查 | 结果 |
|---|---|
| 零初始化下 True vs False | rel **0.000e+00**(逐位相同) |
| `init_weights()` 后 max\|up\| | **0.000e+00**(xavier 没冲掉零初始化) |
| 给定相同历史,generation 受 dW 影响 | rel **0.000e+00** |
| 同上,memory pages 受 dW 影响 | rel 7.203e-01 |
| dW 是否在 norm 内 | \|k\| 比值 **1.0000** vs \|v\| 比值 1.2489 |
| 梯度 | 适配器 4/4 + backbone 都有 |

`|k|` 不动而 `|v|` 变化,正说明 K 经 `norm_k` 重新归一化 —— 两趟版本里"dW 加在 norm
之后、把 key 推离归一化球面(均值 2.8%/最大 12.4%)"的 bug 在此结构上不可能发生。

**测试自身的一个坑(已修,值得记住):** `memory_lora=True` 会多构造两个 `nn.Linear`,
**在构造中途多消耗随机数**,使其后所有 backbone 权重初始化错位。只靠同一个 seed 建两个
模型会得到完全不同的骨干,其差异会被误读成"适配器生效了"——第一版测试就是这么报出
1.388e+00 的假失败。正确做法是让 LoRA 模型 `load_state_dict(plain.state_dict(),
strict=False)`。

## 5. 显存与速度

**测量条件:** Wan 1.3B(dim 1536 / 30 层 / 12 头),60×104 latent → 1560 token/帧,
batch 1,4 步去噪,**FSDP + MixedPrecision(与训练一致)**,单张 H100-80GB,
前向+反传,不含优化器状态。权重 5.29 GiB(FSDP 的 fp32 主副本)。
两边的梯度都已验证正确(见 §4)。

| | SGF 两趟 | 一趟法 | |
|---|---|---|---|
| **全程最大峰值**(会不会 OOM) | 47.98 | **32.10** | 一趟低 **33%** |
| 稳态每步峰值(batch 上限) | **21.78** | 32.10 | 一趟高 **47%** |
| 激活(稳态) | 16.49 | 26.80 | |
| 首步耗时 | 13.79s | 17.31s | |
| **稳态步时** | **11.52s** | 17.04s | 一趟慢 **1.48×** |

SGF 首步的 47.98 主要来自 `create_block_mask(..., _compile=False)` 在 eager 下
物化 N×N int64 网格(N=65520 → N²×8B = 31.98 GiB),`self.block_mask` 有缓存
所以每进程只发生一次,**改一行 `_compile=True` 即可消除**;消掉后 SGF 在两个
口径上都优于一趟法。这一点必须和上表一起说,否则"一趟低 33%"会被误读成算法优势。

**口径说明:单进程 FSDP 会退化成 NO_SHARD** —— 付了 fp32 主副本的代价却拿不到
分片收益,绝对值偏悲观。适合做两条路径的**相对**比较,不能当作真实 4 卡训练的
显存需求。

**FLOPs 两边相同**(线性部分都是 168F,attention 对都是 1560²×462),
1.48× 全部来自串行化的低利用率和 kernel 启动开销:SGF 的 pass 2 是一次 65520
token 的并行前向,一趟法是 42 次 1560 token 的串行前向。

**注意:以下旧数字已作废** —— 它们是在无 FSDP、帧级 checkpoint(梯度错误)下测的:
~~一趟 24.16 / SGF 稳态 13.84 / SGF 首步 40.01 / 1.45×~~。

峰值构成(T=12,allocator 追踪回放重建 99.55%,**在旧的帧级 checkpoint 配置下测的,
粒度已改为块级,需重测**):concat 40.2% / page 18.5% / 单帧重算激活 41.2% /
page 梯度 ≤2.3%。

## 6. 已知未决 / 下一步

- paged attention kernel(消 concat,估算 24.16 → 约 12 GiB,**未实现**)
- sink / FIFO / top-aligned RoPE
- 接入训练(Memory-LoRA 分支已实现并验证,见 4.2)

## 7. 相关脚本

全部在 `/mnt/data/zhaofeng/runs/sgf/tests/`:
`test_persistent_equiv.py` `test_persistent_grad.py` `test_persistent_memory.py`
`test_persistent_breakdown.py` `test_persistent_attrib.py` `test_persistent_vs_sgf.py`
`test_rope_alignment.py` `test_kernel_floor.py` `test_tf_breakdown.py` `test_tf_steady.py`
`memory_report.py`(汇总所有显存数据,从日志读取,不手填)


## 8. window / sink 的适用范围(不要误读)

persistent 路径 **没有实现** window 和 sink:`forward_persistent` 读取全部 page。

21 帧下这样做是安全的,但理由必须说准确。实测 `(sink=3, window=18)` 的
teacher-forcing mask 与全注意力 mask **逐位相同**(0 处不同;24 帧下 192 处不同)。
mask 相同意味着**没有任何 key 被丢弃 —— 窗口在这个长度上从未咬合**。

所以该测试证明的是「**窗口不咬合,因此读全历史与之等价**」,
**不是**「sink / FIFO eviction policy 已被验证」。后者在 persistent 路径里根本
不存在,没有东西可验证。一旦帧数 > `local_attn_size + sink_size`,eviction 就会
真正发生,届时必须先实现 window/sink 再单独验证。pipeline 里已有硬断言,
越界时直接抛错而不是静默降级。

## 9. 每个验证覆盖了什么(S9:绿灯不会自己外推)

一个通过的测试只对它实际跑过的 (设备数, 抽象层, 配置) 有效。这份表把覆盖范围写出来,
免得下次再出现「pipeline 级单卡全绿、4 卡 DMD 级崩掉」时找不到缺口在哪。

| 验证 | 设备 | 抽象层 | num_frame_per_block |
|---|---|---|---|
| 前向 vs cached 串行 | 1 | 模型 | 1 |
| checkpoint 精确性 | 1 | 模型 | 1 |
| RoPE 对齐 | 1 | 算子 | 1 |
| kernel / bf16 地板 | 1 | 算子 | — |
| 严格梯度(4 种精度×FSDP 组合) | 1 | 模型 | 1 |
| pipeline vs SGF pass2 | 1 | pipeline | 1 |
| 集成 1–3(真实 DMD) | 1 | 训练步 | 1 |
| 50 步冒烟 | **8** | 训练循环 | 1 |

**已知的零覆盖:**

- **`num_frame_per_block > 1`(chunkwise)** —— SGF 的 TF mask 里 clean token 对自己
  block 内其它 clean token 可见,而这里整块一次写入、块内可见性由
  `forward_clean_writer` 决定。block=1 时两者不可能不同,所以上表**没有一行**能外推
  到 chunkwise。pipeline 里已加硬 guard 直接 raise。
- **多卡下的逐 rank 行为** —— 8 卡只跑过 50 步冒烟,没有做过 per-rank 的 |g| / cos
  比对(S4b)。`per_rank_exit_step: true` 时各 rank 各采 exit step,聚合值可能被
  单个 rank 的异常主导。
- **page 的跨 step 生命周期** —— 70+ 步训练无 OOM 是间接证据,但没有直接测过
  `PersistentKVStore` 的 page 是否在 optimizer step 前被图持有(S10)。
