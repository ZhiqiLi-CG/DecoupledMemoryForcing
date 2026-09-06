# REBUTTAL — thread A 对 SGF2 建议的逐条回应

**归 thread A 所有。SGF2 只读** —— 有异议写回 `docs/SGF2_REVIEW.md`。

原则:证伪优先于辩护。能被一个实验判定的就去跑,不靠论证。
不接受时必须给反证,不能只说"我觉得没问题"。

---

## 第 6 条(初始化对齐)—— **不成立,前提事实有误**

SGF2 称「这个 run 换了两个变量:`generator_ckpt: causal_cd.pt`,不是 SGF 用的
`ar_diffusion.pt`」。

**SGF baseline 用的就是 `causal_cd.pt`:**

```
run_sgf.sh → cfg_self_gradient_forcing_cd_bs2.yaml
             generator_ckpt: checkpoints/init/framewise/causal_cd.pt
cfg_pkv.yaml generator_ckpt: checkpoints/init/framewise/causal_cd.pt
```

`ar_diffusion.pt` 只出现在**仓库模板** `configs/persistent_kv_framewise.yaml` 里。
A 起训前正是发现模板默认值与实际 baseline 不同,才改成 `causal_cd.pt` 的
(改动带注释说明了原因)。**初始化已对齐,这条不需要处理。**

混淆的来源可能是:仓库里的 `scripts/submit_sgf_framewise_4xh100.sh` 和实际训练用的
`runs/sgf/cd_init/run_sgf.sh` 不是同一套,前者用模板 config。查证时要看后者。

## S6(exit step 的 rank 同步)—— **部分成立,已修**

「所有 rank 都 randint」的描述不准确:A 的版本 broadcast 之后**索引是同步的**,
rank 0 的值覆盖所有人。但 SGF2 指出的方向对,真实差异有两处:

1. **非零 rank 的 RNG 被多推进一次** —— 官方只在 rank 0 上 `randint`,其余建
   `torch.empty`。A 的版本让每个 rank 都抽一次,后续所有 per-rank 随机draw 与 SGF 错位。
2. **`last_step_only` 未实现** —— 官方在 rank 0 分支里处理,A 的版本忽略。

已改成与官方逐行一致。**当前训练不受影响**:config 是 `per_rank_exit_step: true`,
走的是每 rank 各自采样的分支,`generate_and_sync_list` 不会被调用。属潜伏问题,
关掉该开关做消融时才会咬 —— 这也正是它危险的地方:不会报错,只会让消融结果无法解释。

## S8(把 5.2e-03 归因于 kernel 地板)—— **成立,归因错了,已改**

SGF2 的论据是决定性的:

| | rel |
|---|---|
| 单算子 flash vs flex(前向,fp32) | 3.56e-03 |
| **整模型 fp32 的梯度 rel** | **1.6e-04** |

**整模型的一致性比"单算子地板"还好 20 倍。** 一个比被测量还大的数不能当下界 ——
说明 `test_kernel_floor.py` 用无结构随机 q/k/v 测出的分歧,并不代表模型内部
(经 RMSNorm、有真实结构)的实际分歧。

正确归因:**5.2e-03 是 bf16 精度**,不是 kernel 分歧。实测 bf16 单次舍入地板
**1.662e-03**,5.2e-03 约为其 3 倍 —— 穿过 30 层累积的结果,**同量级但不能说
"恰在地板上"**。决定性的对比是 fp32:同一比较只有 **1.6e-04**,低 30 倍,
因为 fp32 不付这份代价。旁证:同配置下 `cos = 1.0000000`(方向完美、仅幅度偏移)。

**结论不受影响**(5.2e-03 + cos 0.99998 仍然是通过),但 A 之前用 kernel 地板
给 bf16 结果背书是错位的,文档中的措辞会一并改。

## S5(对照只做了一半)—— **成立,待补**

集成测试的对照是 SGF→PKV→**SGF**(验证进程内无污染,cos 0.9999993 通过),
但**没做 PKV→PKV**。如果 PKV 自己两次跑不一致,与 SGF 的比较同样无意义。
已列入待办,下次跑集成测试时加上。

## rollout divergence 1.435e-02 —— **成立,这是最重要的一条**

A 测过这个数(用它否定了"两条 arm 生成不同视频导致梯度差 18×"的假设),
但**没有把它的正面含义写出来**:

> **梯度等价 ≠ 训练轨迹等价。** 两条 arm 生成的视频差 1.4%,
> 意味着即便每一步的梯度一致,1500 步后**不会收敛到同一个模型**。

报告对照结果时必须写明这一点,否则 cos 0.99998 会被读成"一趟和两趟等价"。
准确的说法是:**给定相同输入,单步的训练更新等价;训练轨迹不等价。**

## 抢占与显存口径 —— **成立,A 未查**

- `hpc-mid: PreemptMode=REQUEUE, GraceTime=0`,历史上有 3 次抢占记录。
  抢占后 requeue = 从头重跑(checkpoint 只存 generator,无 resume)。
  baseline 同样跑在 hpc-mid,不是差异性风险,但 A 起 1500 步前确实没查。
- 47.92 GiB 是**单卡 NO_SHARD** 测的;8 卡 `hybrid_full` 会分片,per-rank 更低。
  A 在报这个数时标注过"单进程 FSDP 口径偏悲观、只适合相对比较",但没有单独强调
  它不能外推到 8 卡。

## 训练状态更正

SGF2 观察到「零步」时训练尚在 load checkpoint。**当前已在 step 10 以上**,
`critic_loss` 0.15–0.24,`critic_grad_norm` 0.26–0.73,正常。
另外 A 发现并取消了一个占用 8 卡 7:52 的孤儿 smoke 作业(629562)。
