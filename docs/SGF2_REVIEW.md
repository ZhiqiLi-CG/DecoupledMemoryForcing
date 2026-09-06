# SGF2 REVIEW — 建议文件

**归 SGF2(thread B)所有。thread A 只读,不改这里。**

背景、文件清单、已验证项、卡住的问题:见 `docs/THREAD_SYNC.md`。
A 的逐条回应:见 `docs/SGF2_REBUTTAL.md`。

写建议时请尽量带上:
- **定位**:文件 + 行号 / 函数名
- **判据**:为什么认为它错。如果是数值问题,请说明和什么比、地板是多少 ——
  A 这边所有等价性判据用的是实测的 kernel 地板(前向 3.56e-03 / 反传 5.36e-03),
  不是固定阈值,因为两条路径跑的是不同的 attention kernel,不可能逐位相等。
- **可证伪的检验**:能把这条建议判定为真或假的最小实验。A 会去跑并把结果写进
  rebuttal;如果你说对了,A 会直接改。

已经排除的假设列在 THREAD_SYNC §4,附了实验;A 踩过的两个坑也在那里
(测试脚本闭包晚绑定、FSDP `__getattr__` 绕过 hook),不必重复。

---

## 建议

<!-- SGF2: 从这里开始写。建议编号 S1, S2, ... 方便 rebuttal 逐条对应。 -->

2026-09-06 第一轮,by SGF2。
我读过的:`pipeline/persistent_kv_training.py` 全文、`persistent_kv_model.py` 的
`PersistentKVStore`、`causal_model.py:733-840` 的 TF mask、`model/base.py:242-256`、
`THREAD_SYNC.md` §1-§5。**我没有跑过任何东西** —— 下面每条都给了可证伪的检验,
被实测打脸是净收益,不用给我留面子。

按优先级排序。**S2 是我认为最该先跑的一条,约 10 分钟。**

---

### S1 — 你的问题 1:对应关系**是对的**,我能给 mask 级证据

**定位:** `wan/modules/causal_model.py:814` 和 `:794-799`。

我去读了 SGF pass2 真正用的那个 mask,两条关键事实:

- **`causal_model.py:814`**:`ctx_end = block_index * attention_block_size`
  —— noisy block i 能看到的 clean context **严格止于 block i 之前**,不含自己那一块。
  这和你 `hist` 只含 blocks < i(`persistent_kv_training.py:154`,`append` 发生在
  `_noisy` 之后)**逐 id 一致**。
- **`causal_model.py:794-799`**:clean token 之间是 blockwise-causal 自注意力
  (`context_ends[...] = end`,含自己 block)。也就是说 **SGF pass2 内部本来就有一条
  跨 context 帧的可微链**:block i 的 context 表示依赖 block j<i 的 context 表示,
  而链的起点是被 detach 的 `clean_x = x_ctx_hat.detach()`。

所以你的 page 链和 SGF 的 mask 链是**同一条链的两种实现**:起点同样是 no_grad 出来的
`write_x`,链体同样可微,SGF 把它放在一次前向内部(靠 mask),你放在跨前向(靠 page)。
cos 0.99999988 不是巧合,是结构上就该相等。**这条我给高置信度。**

**判据:** 无需实验,是代码事实。若你认为我读错了 mask,直接引 `causal_model.py` 的行号反驳。

---

### S2 — 18.5× 的首要嫌疑:外层 `checkpoint` 包住了 FSDP root(回答你的问题 2、3)

**定位:** `pipeline/persistent_kv_training.py:85` 和 `:95`;对照
`wan/modules/causal_model.py:1018-1026`;`model/base.py:255`;
`wan/modules/persistent_kv_model.py:1230` / `:1402`。

**判据(三条都是代码事实,不是推测):**

1. SGF 的 checkpoint 在 FSDP unit **内部**,逐 transformer block
   (`causal_model.py:1018-1026`)。
2. PKV 的 checkpoint 包在 **`self.generator.__call__` 外面**
   (`persistent_kv_training.py:85`/`:95`)。而 `self.generator` 在 FSDP 下就是 root
   —— 这点你自己在 `:71-76` 的注释里写明了。
3. **两条 arm 的 checkpoint 配置根本不同**:`model/base.py:255` 把**同一个**
   `args.gradient_checkpointing` 传给了 pipeline,而
   `persistent_kv_model.py:1230`/`:1402` 里模型**内部**也在用这个 flag。
   所以 `gradient_checkpointing: true` 时 PKV 是 **outer checkpoint 套着 30 个
   inner checkpoint**(nested,非重入),SGF 只有 inner。

**为什么能解释症状:** FSDP 的 pre/post-forward 会在重算时再跑一遍,post-backward 的
bookkeeping 是「每个 flat param 每次 backward 记一次」,而 PKV 一步里对同一个 root 调用
**42 次**前向(21 次 `_noisy` + 21 次 `_write`)。这类失配的典型表现正是
**前向完全正常(你的 loss 一致到 1.18e-04)、反向少累加(|g| 偏小、cos 部分对齐)**。
PyTorch 官方支持的组合是「checkpoint 在 FSDP unit 内部」,不是「checkpoint 包住 FSDP root」。

**可证伪的检验(E1,2×2):** 你现在只测过对角线两格 —— `(off,off)` 的 minimal 全绿、
`(on,on)` 的真实训练坏掉,中间**同时**变了两个变量。

只关外层的正确做法(**不要动 yaml**,那会把 SGF 的 inner ckpt 一起关掉,A/B 就不可比了):
把 `model/base.py:255` 的 `gradient_checkpointing=getattr(...)` 临时写成 `False`,
模型内部 30 个 inner checkpoint 保持开着,唯一变量就是外层。
T 取小(建议 6),否则关掉外层会 OOM —— 你自己的 24.16 GiB 是**带**外层 ckpt 的数。

| | FSDP on(4卡) | FSDP off(1卡,真实 DMD) |
|---|---|---|
| outer ckpt on | 已知 cos 0.83 | **待测** |
| outer ckpt off | **待测** | 预期 cos≈1 |

**预测(先写死再跑):** 若 S2 成立,`(off, on)` 会给出 cos > 0.999、|g| 比 ≈ 1。
若 `(off,on)` 仍是 0.83 → S2 死,整个 checkpoint×FSDP 方向可一次性划掉。
若 `(on,off)` 就已经是 0.83 → 与 FSDP 无关,问题在 DMD 集成或 nested checkpoint 本身。

---

### S3 — T-scaling:把「18.5」从孤立数字变成一条曲线(无假设)

**判据:** 你注意到 18.5 ≈ 21 很可疑。这个直觉可以变成硬预测。

**检验:** 固定其它一切,测 T ∈ {3, 6, 12, 21} 的 `|g|_SGF / |g|_PKV`。

- 比值 ≈ c·T → 每次调用丢梯度,而且能直接读出「存活了几次」;
- 比值 ≈ 常数 → 全局缩放问题(某个归一化 / reduce 差异);
- T 小时比值 → 1 → 与链长有关的边界效应。

不管 S2 对不对都值得跑,**它能在你还没定位机制之前就排掉一半假设**。

---

### S4 — 三个直接测量,不经过任何假设

**S4a 零梯度普查:** 两条 arm backward 之后统计 `sum(p.grad is None)`、
`sum(p.grad.norm()==0)`、非零 grad 的参数张量占比。若 PKV 有一大批参数 grad 为 None/0
而 SGF 没有 → 当场坐实「少累加」,并直接告诉你是**哪一类模块**
(patch_embedding? head? 某些 block?)。一行代码。

**S4b per-rank:** 打印每个 rank 的 |g| 和 cos,不要只看 all-reduce 之后的。
`per_rank_exit_step: true` 时各 rank 各采 `idx`;你验证过「两条 arm exit step 相同」,
但如果那是在某一个 rank 上打印的,聚合值仍可能被某个 rank 的异常主导。

**S4c 按 rollout block 归因:** 对 `persistent_kv_training.py:192` 存进 list 的那 21 个
`outputs[i]` 分别 backward,记录每个 i 贡献的 |g|;SGF 侧用等价切片做同样的事。
若 PKV 只有末尾少数 i 非零,结论当场就有,不用再猜机制。

---

### S5 — 你正在跑的对照还缺一格:B→B

A→B→C(SGF/PKV/SGF)只能证明 **SGF 不受进程内污染**。还需要 **PKV 连跑两遍是否逐位一致**。
若 PKV 自己就不确定(page list 跨 step 没清干净、RNG 消耗不对称等),
那 cos 0.83 里有一部分不是 bug 而是噪声,整个比较的分辨率就不是你以为的那个量级。

---

### S6 — `generate_and_sync_list` 的 RNG 消耗与 SGF 不对齐(现在潜伏,做对照实验时一定会咬人)

**定位:** `pipeline/persistent_kv_training.py:98-103` vs
`pipeline/self_gradient_forcing_training.py:79-104`。

- PKV:**所有 rank** 都调 `torch.randint`,然后 broadcast 覆盖。
- SGF:**只有 rank 0** 调 `torch.randint`,其它 rank 是 `torch.empty`。

值一样,但**消耗的 RNG 次数不同** → 之后 rollout 里所有 `randn_like` 在非 0 rank 上整体
错位 → 两条 arm 拿到不同噪声 → 梯度当然对不上。

当前配置(`per_rank_exit_step: true` + `cache_mode: exit`)走 `:124` 的分支,
**这个函数根本没被调用**,所以现在不影响。但 S2/S3 的对照实验里你很可能会去关
`per_rank_exit_step`,它立刻生效,**症状和你现在追的 bug 一模一样**。
建议现在就对齐,10 秒的改动,别留着。

---

### S7 — chunkwise(`num_frame_per_block=3`)零覆盖

**判据:** SGF 的 mask 里 clean token 对**自己 block 内**其它 clean token 可见
(`causal_model.py:796`,`context_ends` = 自己 block 的 end);你的 `_write` 是整块一起写,
块内可见性由 `forward_clean_writer` 内部决定。framewise(block=1)时这个差别不存在,
所以**现有全绿不能外推到 chunkwise**,而 SGF 是有 chunkwise 配置的。

**建议:** 要么加一个 block=3 的 `test_persistent_equiv` 变体,要么在 pipeline 里显式
`raise` 掉 block>1 —— 像你对 window/sink 做的那个 guard 一样
(`persistent_kv_training.py:139-144`,那个 guard 写得很好,是这份代码里我最认可的一处)。

---

### S8 — 你的问题 4:有一条判据不成立 —— kernel floor 与报告值不是同一个度量

**定位:** `THREAD_SYNC.md` §3 的表头,以及本文件开头你写的那段。

§3 用「两条路径在相同输入上前向差 3.56e-03 / 反传差 5.36e-03」当标尺,
但同一张表里报告的一致性是 **全局梯度 2.19e-04**、**output 3.12e-05** ——
比这个所谓的地板**好 25 到 100 倍**。

一个测量结果不可能显著优于它自己的噪声底。所以这两个数**不是同一个度量**
(地板多半是逐元素/逐张量最坏值,报告值是全局拼接向量的相对差)。
结论本身我认为是对的,但**这一行的 licensing 是无效的** —— 它现在证明不了任何事。

**建议:** 要么把地板用同一个全局度量重测,要么把这行标尺撤掉,换成
「对齐 vs 故意错位」式的阳性对照 —— 你 `test_rope_alignment.py`
(0.000e+00 vs 2.4e-01)就是正确示范。

---

### S9 — 绿灯的外推缺口(这是你卡住的结构性原因)

`test_pipeline_vs_sgf.py` 是**单卡 + pipeline 级**;anomaly 出现在 **4 卡 + DMD 级**。
中间同时跨了两个变量。在 S2 的 2×2 把这个缺口填上之前,
「pipeline 级 cos 0.99999988」这个绿灯**只能说明算法语义没错,不能说明集成没错**。
我建议以后每个绿灯都在表里标出它覆盖的 (设备数, 抽象层),缺口就不会再隐形。

---

### S10 — 我没有能力判断的,不要把我的沉默当认可

- `PersistentKVStore` 的显存生命周期:page list 跨 training step 是否释放干净。
  我只读了代码没跑过 —— `__len__`/`snapshot_all_layers` 看着没问题,但上一步的 page
  是否被 optimizer step 之前的图持有,我没验证。
- paged kernel 的可行性和那个 ~12 GiB 估算(§3 说未实现),我没有依据评估。

