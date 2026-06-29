# chunk_bwd_dqkwg CV 深融合 —— 性能劣化诊断与修复记录

> 分支: `communicate` ｜ 算子: `fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg`
> 截止提交: `bbdc6253 perf(dqkwg): inline mul1 for large cases`（精度全过）

---

## 0. 背景与目标

`communicate`(CV 深融合, ring/bounded 工作区版) 与 `main`(同一个 CV 融合 kernel, 但用全局 T 相关工作区 + 逐 stage `SyncAll` 调度) 对比,
部分形状 perf 劣化。**约束: CV 深融合不应比 main 慢(同步依赖更少、on-flight 中间数据应更轻),且只能改劣化形状、不动已优化的好形状。**

`main` 关键结构(读源码确认):
- **stage-major**: Part1(dw)[全 chunk] → `SyncAll` → Part2(mm5)[全 chunk] → … 每个 Part 只 streaming 写一张量。
- 全 7 个 Part 用 `BlockMmadTla<MmadPingpong>`(pingpong 双缓冲)。
- per-head CrossCore flag + `SyncAll<false>` 逐 Part 排空 FixPipe。
- dw 直接写输出 `ptrDw`(无 dw 工作区);**非 GVA**(q/k 必须 HV 头)。

`communicate` 关键结构:
- **chunk-interleaved group-major**: 每组 A/B/C/D 连着做。
- per-chunk raw 信用流水(`CrossCoreSetFlag/WaitFlag`),无 `SyncAll`。
- V≤128 用 `TileGemmDirect`(单 stage 无 pingpong);**V=256 用 `MatmulKernelTiled` = `BlockMmadTla<MmadPingpong>` 全 7 Part = 与 main 同款**。
- dw 走工作区 `wsDw` + `RepairDwChunkHeadBlock`;mul1 走工作区往返;dg_last 在 stage A 算(原意: 藏在 cube 的 2 个 matmul 后)。
- GVA(HV/HK 拆分)。

---

## 1. 全部尝试(按时间, 含无效的)

> 每次都在 NPU 上实测后回退/保留。**前 4 次全部失败**,把搜索空间收窄到了正确方向。

| # | 改动 | 提交 | 假设 | 结果 | 结论 |
|---|---|---|---|---|---|
| 1 | **per-head ready** 握手 | `e4f3fe63`→回退`8a02990f` | per-chunk ready 把 chunk 内 head 流水重叠砍了 | **case_11 死锁/超时** | 本算子用裸 `CrossCoreSetFlag`(无界计数,硬件上限低),per-head 未消费 ready ≈ N×HV(64) 溢出 → 丢信号死锁。**per-head 在此算子是死路**(除非换 `CrossCoreFlagWithReverse`)。 |
| 2 | **全局工作区寻址**(ring→global `(h*T+bos)*K`) | `bd7f1953`/`267e3fd1`→回退 | 环 slot 复用(WAR)比 main 全局唯一寻址更颠簸 L2(victim +29%) | **大 case 无改善, case_11/12 反而更差(+11%)** | **寻址不是病根**(BT=128 时全局 == main 布局仍 +25%)。全局 2x 内存反而拖累。 |
| 3 | **stage-major 调度** + 全局 | `5fe374b6`→回退 | group-major 跨 7 区交错写 → FixPipe 局部性差 | **step2_12 仍 +25%(略更差)** | **调度也不是病根**(BT=128 做到工作区+调度都和 main 逐字节一致, 仍劣化)。 |
| 4 | **preseed=1**(收小信用窗口/重叠) | `9b8b5f43`→回退`fbb084ca` | 重叠越深 → cube/vector 同时在不同 stage → L2 工作集越杂 → 颠簸 | **全面更差, 扩散到 23 个 case** | **重叠是好的**(CV 融合在正常工作)。"N=4 比 N=2 慢"是 #2 的 2x 内存, 不是重叠。 |
| 5 | **mul1 GM 往返诊断**(大 case 只读/写 1 个 block-row, golden-off) | `a31caffc` | mul1 的 [BT,BT] 往返(~3.2GB/step2_12)在和 cube FixPipe 抢 HBM 带宽 | **普通 case 全修好(<5%); step2 -6%(12: 22.56→16.55)** | ✅ **坐实**: mul1 往返流量是真瓶颈。main 内联算 mul1, communicate 多了这趟往返。 |
| 6 | **mul1 内联**(vector B 从输入 g 现读现算掩码, 不走 GM; 抽 `ComputeMul1HalfFp32` 复用 A 的 Part2 内核, 两个 row-half) | `bbdc6253` | 把诊断变成正确实现 | **精度全过, 收益达预期**(见 §3) | ✅ 已落地。 |

### 关键 msprof(step2_12, 环基线 vs main)
- cube 是瓶颈(FixPipe ~0.89 busy);communicate cube **FixPipe +4.4ms、MTE2 +2.4ms**,vector **scalar −7.3ms**;**MAC 两边一样**;**L2 victim +29%**。
- 即: 活从 vector(标量)挪到了 cube(访存),cube 在 GM 写带宽上被拖慢。FixPipe 归因(按输出元素数)显示 **7 个 Part 均匀 +14.6%**,不是某个 Part 单独爆 → 是**带宽/L2 颠簸**,不是计算量。

---

## 2. 已确定排除的(实测, 别再试)

| 假设 | 排除依据 |
|---|---|
| 工作区寻址(ring vs global) | step2_12 对此不变;BT=128 时全局做到 == main 布局仍 +25% |
| 调度(group-major vs stage-major) | stage-major 实测无改善 |
| 重叠/信用窗口深度 | preseed=1 反而全面更差 → 重叠有益 |
| matmul 类型(pingpong) | V=256 用的就是 main 同款 `BlockMmadTla<MmadPingpong>`,L1/L0 tile shape、`PackedTileCopyTla` 逐字节相同(只有 V≤128 用无 pingpong 的 TileGemmDirect, 那解释了 case_11/12 历史上的小幅劣化,已被 mul1 内联覆盖) |
| 输入读取量 | main 非 GVA, 跑 step2 时 q/k 按 HV 头读(比 communicate GVA 的 HK 读得更多)却更快 |
| per-head 握手 | 死锁(裸 flag 饱和) |

**核心机制(已坐实)**: cube 受 **GM 写带宽(L2 颠簸)** 限制;communicate 的**额外 vector GM 往返**在和 cube FixPipe 抢 HBM 带宽。削掉这些往返 → 释放带宽 → cube FixPipe 加速。mul1 往返是其中第一个(已削)。

---

## 3. 当前状态(mul1 内联后, 全部 golden 通过)

| case | current ms | main ms | 劣化 | shape 关键 |
|---|---|---|---|---|
| case_step2_12 | 29.43 | 25.46 | **+15.60%** | HV48/HK16, T65536, **BT128, V256**, varlen |
| case_step2_06 | 43.20 | 37.85 | **+14.13%** | HV64/HK2(n_ratio32), T65519, BT64, **V256**, varlen |
| case_step2_02 | 11.28 | 10.03 | **+12.44%** | HV63/HK21, T16384, BT64, **V256**, varlen |
| case_step2_08 | 20.90 | 18.72 | **+11.67%** | B16, HV63/HK21, T2048, BT64, **V256** |
| case_25       | 0.351 | 0.324 | +8.39% | B2, HV4, T512, BT128, **V256** —— **M=1 极小, 无流水** |
| case_step2_03 | 19.91 | 18.40 | **+8.20%** | HV32/HK8, T65536, BT128, **V256**, varlen |
| case_step2_07 | 1.58 | 1.48 | +6.63% | HV32/HK16, T4096, BT64, **V256** |

**普通 case(V=128, 含 case_11/12)全部 <5%, 已不在劣化表内。**

---

## 4. 剩余劣化的根因假设

### 4.1 决定性信号: **剩余劣化全部是 V=256**
- mul1 往返流量是 **V 无关**的([BT,BT]),所以削掉它**普通(V=128)全修好、step2(V=256)只部分修好**。
- 残余的 ~6~16% 是 **V=256 专属**的额外 HBM 流量/争用。case_25 是唯一例外(V256 但 M=1, 属固定开销/warmup, 见 §4.3)。

### 4.2 V=256 专属嫌疑(按可能性排序)
按同一机制(communicate 比 main 多的、且被 V=256 放大的 vector GM 往返, 在抢 cube FixPipe 带宽):

1. **【主嫌疑】dg_last 在 stage A 读 h+dh**
   - `dg_last = Σ_{K,V} h·dh`,每 head 读 `h[K,256]+dh[K,256]` ≈ 128KB(V=256), step2_12 全量 ≈ **3.2GB**。
   - communicate 在 **stage A** 算(藏在 cube 2 个 matmul 后);main 在 **stage D** 算。
   - stage A 同时还有: cube Part1(dw=dv@hᵀ)也读 h、Part2 写 mm5。→ **h 在 stage A 被 cube+vector 各读一次(V=256 大),vector 的 h+dh 大读和 cube 的 stage-A FixPipe 写抢带宽**。main 把这次大读放到 stage D(和 cube 的 dk_inner+mm7 并发),错开了 stage A 的写压力。
   - 这是"communicate 把活挪进 cube-bound 的 stage A"的优化对 V=256 的**反噬**(V=128 时 h+dh 小, 不显, 所以普通 case 不劣化)。

2. **mul0 行处理(`set_mul0RowNum(V==256?16:32)`)**: V=256 时每趟 16 行(vs 32)→ 2x 趟数。属 vector 计算/搬运, 需确认是否放大了某处 GM 流量。

3. **V=256 输入的 cube 重读**: v(Part3/5)、do(Part3/4)、h(Part1/4)在多个 stage 被 cube 重读;V=256 下这些 256 宽张量大, group-major 的"组内 L2 复用"对大 case 失效(工作集 > L2)→ 退回 HBM 重读。但 main 也重读(stage-major 也跨 Part 重读), 需诊断确认差异。

4. **dw 工作区往返**: communicate 写 `wsDw` + `RepairDwChunkHeadBlock`(读 dv+h, 只修 row0 前 16 列 ≈ 220MB, 小);main 直接写 `ptrDw`。**V 无关, 量小, 低优先级**。

### 4.3 case_25 单独说明
- B2/HV4/T512/BT128, **M=1, 每核 1 个 chunk, 无跨 chunk 流水**。+8.39% 属**固定开销/warmup/冷启动**(类似历史 case_13: mean 被一次冷启动 max 拉高, p50 往往已 ≤ main)。
- **不归属 V=256-带宽机制, 上面的修法不治它**。建议单独看 p50/中位数, 大概率非真实劣化。

---

## 5. 修复计划(诊断优先, 沿用第 5→6 步的成功套路)

> 原则: 先用 golden-off 廉价诊断坐实"哪一项 V=256 流量是瓶颈", 再做正确实现。**不再盲改**。

### 步骤 A —— 诊断 dg_last(主嫌疑)
- **改**(golden-off, 门控 `V==256 && largeMemBound`): 在 stage A 跳过 dg_last 的 h+dh 读+计算(`tensorSumFp32` 那段的 `DataCopy(gmDh/gmH)` 缩到 1 个 block-row, 或整段 `if(!diag)` 跳过)。stage D 读 `wsDgLast`(残留)→ golden 必挂, **只看 perf**。
- **判读**: step2(V=256)perf 明显向 main 收敛 → 确认 dg_last 在 stage A 的大读是瓶颈。

### 步骤 B —— 正确修复(若 A 确认)
- **把 dg_last 计算移回 stage D**(= main),门控 `V==256 && largeMemBound`,避免 stage A 的 h+dh 大读和 cube stage-A FixPipe 抢带宽。
  - 代价: stage D(vector-bound)多读 h+dh;但 main 证明这是更优的并发位置(和 cube 的 dk_inner/mm7 并发, 而非 dw/mm5 写并发)。
  - 需保留 `wsDgLast` 路径用于小 case(V=128/≤512MB 不变, 仍 stage A 算, 已不劣化)。
  - **风险**: dg_last 在 D 重算需要 h/dh(D 的 cube 也读 dh), 注意 UB 与读时序;小 case 字节不变。

### 步骤 C —— 若 dg_last 不是(或不止)
- 依次诊断 §4.2 的 #2(mul0)、#3(V=256 输入重读)。#3 若确认, 方向是让 V=256 输入在 cube 内更 streaming(或减少跨 stage 重读),但要小心别破坏 group-major 对其它好 case 的 L2 复用收益。

### 步骤 D —— case_25
- 仅核对 p50/中位数;若 mean 被冷启动拉高而 p50 ≤ main, 判为非真实劣化, 不改。

---

## 6. 经验/原则(供后续)

1. **诊断优先于盲改**: 前 4 次盲改(per-head/global/stage-major/preseed)全失败且浪费构建周期;第 5 步 golden-off 诊断一次坐实, 第 6 步正确实现一次成功。
2. **门控只改劣化形状**: 用 `mainWs = B·HV·T·K·2 + B·HV·T·BT·2 + B·HV·numChunks·4 > 512MB` 判据, host tiling 与 kernel **同公式**;小 case 字节不变。
3. **裸 `CrossCoreSetFlag` 不能 per-head**(计数饱和死锁);要 per-head 必须先换 `CrossCoreFlagWithReverse`(有界)。
4. **瓶颈在 cube GM 写带宽**: 真正有效的杠杆是**削减和 cube FixPipe 抢 HBM 带宽的额外 vector GM 往返**(mul1 已削, dg_last 是下一个嫌疑), 而非寻址/调度/重叠。
5. **机制可复用的实现技巧**: 把"落 GM 的中间量"改成"从输入现算"(mul1 从 g 现算), 用 vector 计算(被 idle 的 vector 藏住)换 GM 流量, 对 memory-bound 大 case 是净赚。
6. **mul1 内联实现要点**(已落地, 供参考): 抽 `ComputeMul1HalfFp32` 复用 A 的 Part2 逐行内核(同 buffer/repeat 参数/64×64 因果 mask), vector B 按 head 全 BT 调两次(两个 row-half, `BT_sub_start=0` 与 `vec0`)。A 的 mul1 是行优先(`repStride=16` 块 = 128 fp32 = 一行 BT), 故可直接乘 ds。B 的 UB 紧(BT=128 约 183KB/192KB)。
