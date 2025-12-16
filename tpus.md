---
layout: distill
title: "TPU 是怎么工作的"
# permalink: /main/
description: "TPU 的内部结构其实很简单：一个超强的矩阵乘法引擎 + 一大块显存 + 高速互连网络。搞懂这些，你就知道为什么有些模型跑得快、有些跑得慢了。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "第1部分：Roofline模型"

next_section_url: ../sharding
next_section_name: "第3部分：分片"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

toc:
  - name: TPU 长啥样？
  - name: TPU 之间怎么连接
  - name: 关键要点
  - subsections:
    - name: 规格速查表
  - name: 练习题
  - name: 附录
  - subsections:
    - name: "附录 A：TPU 内部细节"
    - name: "附录 B：脉动阵列原理"

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<p markdown=1 class="announce">想了解 GPU 的可以看新增的[第12章](../gpus)！</p>

## TPU 长啥样？

**TPU 本质上就是一个专门算矩阵乘法的计算单元（TensorCore）+ 一大块高速内存（HBM）。**<d-cite key="tpu_paper"></d-cite> 就这么简单。

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>TPU 芯片结构图：</b> 左边灰框是 TensorCore，里面有矩阵乘法单元（MXU）、向量单元（VPU）和片上内存（VMEM）。右边蓝条是大容量显存（HBM）。" %}

TensorCore 里面有三个关键部件：

### 1. MXU（矩阵乘法单元）—— 核心中的核心

MXU 是 TPU 的灵魂。它用一种叫"脉动阵列"的结构（见[附录 B](#附录-b脉动阵列原理)），每 8 个时钟周期就能完成一次 `bf16[8,128] × bf16[128,128] → f32[8,128]` 的矩阵乘法。<d-footnote>TPU v6e（Trillium）用的是 256×256 的 MXU，之前都是 128×128。</d-footnote>

- TPU v5e 单个 MXU 大约能做 5×10¹³ bf16 FLOPs/s（1.5GHz 时钟）
- 大多数 TPU 有 2-4 个 MXU，所以 TPU v5e 整体是 2×10¹⁴ bf16 FLOPs/s
- 低精度更快：int8 能达到 4×10¹⁴ OPs/s（翻倍）

### 2. VPU（向量处理单元）—— 干杂活的

除了矩阵乘法，还有很多"杂活"：ReLU 激活函数、向量加法、归约求和等。这些都是 VPU 干的。详见[附录 A](#附录-atpu-内部细节)。

### 3. VMEM（向量内存）—— 超快的小仓库

VMEM 是 TensorCore 内部的片上高速缓存，容量小（TPU v5e 只有 128MB）但**到 MXU 的带宽极高**。可以类比 CPU 的 L1/L2 缓存，但更大、由程序员控制。

**重点**：数据必须先从 HBM 搬到 VMEM，TensorCore 才能用它算东西。

---

**HBM（高带宽内存）** 就好理解了——就是我们平时说的"显存"：

- 容量大（[TPU v5e 有 16GB](https://cloud.google.com/tpu/docs/v5e#system_architecture)）
- 带宽还行（约 1-2 TB/s）
- 但比 VMEM 慢得多

计算时，数据流是这样的：**HBM → VMEM → MXU 计算 → VMEM → HBM**

---

### 矩阵乘法是怎么流水线起来的

TPU 做矩阵乘法时，会把整个过程流水线化：
1. 把权重和输入分块从 HBM 搬到 VMEM
2. 分块喂给 MXU 算
3. 结果分块写回 VMEM，再写回 HBM

**边搬边算**——搬运和计算重叠进行，这样 MXU 就不用干等数据了。

下面这个动画展示了逐元素乘法的过程（矩阵乘法类似）：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>动画：</b> 数据分块从 HBM 流入 VMEM，然后进 VPU 算，结果再流回去。边搬边算，形成流水线。" %}

<p markdown=1 class="takeaway">**一句话总结 TPU：** 把权重从 HBM 搬到 VMEM，再喂给脉动阵列，每秒能做约 200 万亿次乘加。性能瓶颈通常在数据搬运（HBM↔VMEM 和 VMEM↔MXU 的带宽），而不是计算本身。</p>

---

### VMEM 的妙用

VMEM 虽然小，但带宽比 HBM 高 20 多倍。这意味着：

- 如果数据能放进 VMEM，就不容易被带宽卡住
- 那些"算术强度"差的操作，如果能从 VMEM 跑，效率也能拉上来
- 矩阵乘法如果权重能放进 VMEM，临界 batch size 会大幅降低

问题是 VMEM 太小，这通常是个挑战。<d-footnote>有时候可以做"VMEM 预取"——在算注意力的时候，提前把下一层的 FFN 权重加载到 VMEM，掩盖搬运开销。但这要求权重够小或分片够细。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

---

### 芯片、核心、托盘的关系

**一个 TPU 芯片通常有两个核心，共享内存**（叫"megacore"模式，从 v4 开始）。老款 TPU（v3 及以前）的两个核心是独立的。推理芯片（如 v5e）每个芯片只有一个核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**4 个芯片组成一个"托盘"（tray）**，通过 PCIe 连到一个 CPU 主机。这就是你在 Colab 或 TPU-VM 里看到的 4 芯片/8 核心配置（通常当作 4 个逻辑核心用）。<d-footnote>推理芯片 v5e 每主机有 2 个托盘、8 个芯片（但每芯片只有 1 核），加起来也是 8 核。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽有限**——大约只有 HBM 带宽的 1/100。可以把数据卸载到主机 RAM，但很慢。

## TPU 之间怎么连接

单机不够用怎么办？把多张 TPU 连起来！

### ICI：芯片间的高速直连

**在同一个 Pod 内，TPU 通过 ICI（芯片间互连）直接相连**——不经过主机！

- 老款（v2、v3）和推理芯片（v5e、v6e）：连接 4 个最近邻居（2D 环面）
- v4 和 v5p：连接 6 个最近邻居（3D 环面）

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面的好处：任意两个节点的最大距离从 N 减到 N/2。还有"扭曲环面"（像莫比乌斯带一样缠绕）可以进一步缩短距离。

### Pod 能做多大？

**SuperPod（最大 Pod）**：
- TPU v4：`16×16×16`（4096 芯片）
- TPU v5p：`16×20×28`（8960 芯片）

这些大 Pod 由 `4×4×4` 的小立方体通过[光学交换机](https://arxiv.org/pdf/2208.10041)连接。<d-footnote>光学交换机的带宽和 ICI 一样，只是连接可以重新配置。</d-footnote>

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以申请小规模配置（如 `2×2×1`、`2×2×2`），但**没有环绕链路**，通信时间会翻倍。完整立方体倍数（如 `4×4×4`、`4×4×8`）才有环绕链路。

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 v6e 的 Pod 是单个 `16×16` 的 2D 环面，长边（16）才有环绕链路。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

---

### TPU vs GPU 的网络差异

这是 **TPU 和 GPU 的核心区别**：

- **TPU**：每个芯片只连最近的邻居（4 或 6 个），构成环面
- **GPU**：通过交换机层次连接，近似点对点（NVLink 域内 8 张或 72 张直连，更大规模需要 O(log N) 跳）

TPU 的方式更便宜、更简单、能扩展到更大规模。GPU 的方式延迟更低、任意两点通信更快。各有利弊。详见 [GPU 章节](../gpus#networking)。

---

### 带宽速度排行

以 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 为例：

| 连接类型 | 带宽（每芯片） | 备注 |
|---------|---------------|------|
| HBM ↔ VMEM | 2.5 TB/s | 最快 |
| ICI（每轴） | 90 GB/s（双向）<d-footnote>单向 45 GB/s × 2。完整环才能用满双向。</d-footnote> | v5p 有 3 轴 |
| PCIe | ~16 GB/s | 比 HBM 慢 100 倍 |
| DCN（出口） | ~6 GB/s | 最慢 |

**结论**：把模型拆到多卡上时，要小心别让通信拖慢计算。

---

### 多切片训练：跨 Pod 怎么办

一组通过 ICI 连接的 TPU 叫一个"**切片（Slice）**"。不同切片可以通过 **DCN**（数据中心网络）连接——比如连接不同 Pod 上的切片。

DCN 比 ICI 慢很多，数据还得绕道：TPU → PCIe → 主机 → 网络 → 目标主机 → PCIe → TPU。尽量减少等 DCN 的时间。

## 关键要点

1. **TPU 结构很简单**：矩阵乘法单元 + 显存 + ICI（超快）+ DCN（较慢）

2. **带宽速度排行**：HBM > ICI > PCIe > DCN

3. **TPU 只连最近邻居**：远距离通信需要跳转多个芯片

4. **权重矩阵要填充到 128×128**（v6 是 256×256）才能喂饱 MXU

5. **低精度更快**：int8 是 bf16 的 2 倍，int4 是 4 倍（VPU 操作仍是 fp32）

6. **避免让 MXU 等数据**：通信量要和各链路速度匹配

### 规格速查表

| 型号 | Pod 大小 | 单主机 | HBM/芯片 | HBM 带宽 | bf16 算力 | int8 算力 |
|:-----|:--------:|:------:|:--------:|:--------:|:---------:|:---------:|
| v3   | 32×32    | 4×2    | 32GB     | 0.9 TB/s | 140 TF/s  | 140 TF/s  |
| v4p  | 16³      | 2×2×1  | 32GB     | 1.2 TB/s | 275 TF/s  | 275 TF/s  |
| v5p  | 16×20×28 | 2×2×1  | 96GB     | 2.8 TB/s | 459 TF/s  | 918 TF/s  |
| v5e  | 16×16    | 4×2    | 16GB     | 0.8 TB/s | 197 TF/s  | 394 TF/s  |
| v6e  | 16×16    | 4×2    | 32GB     | 1.6 TB/s | 920 TF/s  | 1840 TF/s |

*TF/s = 10¹² FLOPs/s*

ICI 带宽（每链路）：

| 型号 | 单向 | 双向 |
|:-----|:----:|:----:|
| v3   | 100 GB/s | 200 GB/s |
| v4p  | 45 GB/s  | 90 GB/s  |
| v5p  | 90 GB/s  | 180 GB/s |
| v5e  | 45 GB/s  | 90 GB/s  |
| v6e  | 90 GB/s  | 180 GB/s |

<d-footnote>双向带宽 = 单向 × 2，完整环时可以用满。</d-footnote>

PCIe 约 16 GB/s/芯片（v6e 是 32），DCN 约 6 GB/s/芯片（v6e 是 12.5，v5e 是 3.125）。

## 练习题

这些数字看着枯燥，但用处很大——可以让你快速估算模型性能。

**题 1：推理延迟下界**

假设你要从一个 2000 亿参数的 bf16 模型采样，模型分布在 32 张 TPU v4p 上。把所有参数从 HBM 加载到 MXU 要多久？

{% details 点击查看答案 %}

参数量：`2×200×10⁹ = 400×10⁹` 字节（bf16 每参数 2 字节）
每芯片：`400×10⁹ / 32 = 12.5×10⁹` 字节
HBM 带宽：1.2×10¹² 字节/s
加载时间：`12.5×10⁹ / 1.2×10¹² ≈ 10ms`

**这就是采样延迟的理论下界**——每次采样都要加载所有参数，不可能比 10ms 更快。实际上，小 batch 时接近这个值。

{% enddetails %}

**题 2：数一数**

一个完整的 TPU v5e Pod 有：
1. 多少 CPU 主机？
2. 多少 TensorCore？
3. 总算力是多少？
4. 总显存是多少？

对 v5p Pod 也算一下。

{% details 点击查看答案 %}

**v5e**：
- Pod 是 16×16 = 256 芯片
- 每主机 4×2 = 8 芯片 → 256/8 = **32 主机**
- v5e 每芯片 1 核 → **256 TensorCore**
- 算力：256 × 2×10¹⁴ = **5.1×10¹⁶ bf16 FLOPs/s**（51 PF）
- 显存：256 × 16GB = **4TB**

**v5p**：
- Pod 是 16×20×28 = 8960 芯片
- 每主机 2×2×1 = 4 芯片 → 8960/4 = **2240 主机**
- v5p 每芯片 2 核 → **17920 TensorCore**
- 算力：8960 × 4.5×10¹⁴ = **4×10¹⁸ bf16 FLOPs/s**（4 EF）
- 显存：8960 × 96GB = **860TB**

{% enddetails %}

**题 3：从主机内存算矩阵乘法**

假设权重 `bf16[D, 4D]` 和激活 `bf16[B, D]` 都存在主机内存（不在 TPU 显存），你想用一张 TPU v6e 算矩阵乘法。假设 $B \ll D$。需要多大 batch 才能计算受限？（PCIe 带宽 1.5×10¹⁰ 字节/s）

{% details 点击查看答案 %}

- 计算量：$2B \cdot D \cdot 4D = 8BD^2$ FLOPs
- 计算时间：$8BD^2 / 9.2×10^{14}$
- 传输量：约 $8D^2$ 字节（权重为主，$B \ll D$）
- 传输时间：$8D^2 / 1.5×10^{10}$

计算受限条件：
$$\frac{8BD^2}{9.2×10^{14}} > \frac{8D^2}{1.5×10^{10}}$$

$$B > \frac{9.2×10^{14}}{1.5×10^{10}} ≈ 61000$$

需要 **6 万以上的 batch** 才能计算受限！PCIe 太慢了。

{% enddetails %}

**题 4：矩阵乘法需要多久**

在 1 张 TPU v5e 上，用 `int8[16384, 4096]` 的权重乘以 `int8[B, 4096]` 的激活：

1. 运行时间是 B 的什么函数？
2. 如果权重能放进 VMEM 呢？

{% details 点击查看答案 %}

**(1) 从 HBM 读：**
- FLOPs：$2 × 4096 × 16384 × B = 1.3×10^8 × B$
- $T_\text{计算} = 1.3×10^8 × B / 3.94×10^{14}$
- 传输量：$16384×4096 + 4096×B + 16384×B$ 字节
- $T_\text{通信} = (6.7×10^7 + 2×10^4×B) / 8.1×10^{11}$

计算受限条件：$B > 271$

**(2) 从 VMEM 读：**

VMEM 带宽约是 HBM 的 22 倍，临界点变成 $B > 11$。

{% enddetails %}

**题 5：ICI 传输**

4×4 的 TPU v5e 切片，把 `bf16[8, 128, 8192]` 从 (0,0) 发到 (3,3)。假设每跳延迟 1μs。

1. 第一个字节何时到达？
2. 整个传输要多久？

{% details 点击查看答案 %}

- 数据量：$2×8×128×8192 = 1.7×10^7$ 字节
- 4×4 切片没有环绕链路，(0,0) 到 (3,3) 需要 6 跳
- 第一个字节：**~6μs**（延迟）
- 带宽：可以从两个方向同时发（右和下），共 $2×45×10^9 = 9×10^{10}$ 字节/s
- 传输时间：$1.7×10^7 / 9×10^{10} ≈ 188μs$

总共约 **188μs**（带宽受限）。

{% enddetails %}

**题 6：综合挑战**

一个 `int8[128K, 128K]` 的大矩阵均匀分布在 TPU v5e 4×4 切片上，但卸载到了各芯片的主机内存。你想把它全部收集到 TPU(0,0) 然后乘以 `bf16[8, 128K]`。要多久？

{% details 点击查看答案 %}

矩阵约 16GB。4×4 切片有 2 个主机（每主机 8 芯片），每主机存 8GB。

方案：通过 ICI 收集比通过 DCN 更快。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每个芯片有独立 PCIe 到主机，简化图只画了一条。" %}

分步计算：

1. **PCIe 加载**：16GB / 16 芯片 = 1GB/芯片，带宽 1.5×10¹⁰ → 约 **66ms**

2. **ICI 收集**：TPU(0,0) 要收 15GB，2 个方向各 45 GB/s → 下界 **167ms**（实际可能更长）

3. **HBM → MXU**：16GB / 8.1×10¹¹ ≈ **19ms**

4. **计算**：$2×8×128K×128K = 2.7×10^{11}$ FLOPs / 1.97×10¹⁴ ≈ **1.3ms**

瓶颈在 ICI 收集。假设能部分重叠，总时间约 **170-200ms**。

{% enddetails %}

---

<h3 markdown=1 class="next-section">第 2 章完！接下来学习模型分片，[点击继续](../sharding)。</h3>

---

## 附录

### 附录 A：TPU 内部细节

这里更深入地介绍 TPU 内部。以 TPU v5p 为例。

### VPU 详解

VPU 是做"杂活"的向量单元：逐元素加法、ReLU、归约等。

**结构**：`(8, 128)` 的 2D SIMD 阵列
- 128 维叫 **lane**
- 8 维叫 **sublane**
- 每个 (lane, sublane) 有 4 个独立 ALU

**速度**：大多数指令 1 周期完成，2 周期延迟。

**VREGs（向量寄存器）**：v5p 每核 64 个，总共约 256KB。每周期可以从 VMEM 读 3 个、写 1 个。

**归约**：sublane 内归约很快（shuffle 几下就行），跨 lane 归约要用 XLU（慢）。

**小测验**：算一下 TPU v5p VPU 能做多少 FLOPs/s？（时钟 1.75GHz）

{% details 答案 %}
每周期：$8 × 128 × 4 × 2$（2 核）= 8192 FLOPs
总算力：$8192 × 1.75×10^9 = 1.4×10^{13}$ FLOPs/s

比 MXU 的 2×10¹⁴ 小 10 倍左右。

{% enddetails %}

**对比 GPU**：VPU 的每个 ALU 类似 CUDA 核心，每个 lane 类似一个 Warp 调度器。

### 标量核心

标量核心是 TPU 的"大脑"——取指令、控制 DMA、做标量运算。

注意：标量核心是单线程的，每周期只能发起一个 DMA 请求。

一个标量核心管着：1 个 VPU（4096 个 ALU）、4 个 MXU、2 个 XLU、多个 DMA 引擎。这种高度集中的控制是效率来源，但也限制了灵活性。

---

### 附录 B：脉动阵列原理

MXU 的核心是 **128×128 脉动阵列**（v6e 是 256×256）。

完全饱和时，每 8 周期完成一次 `bf16[8,128] @ bf16[128,128] → f32[8,128]`<d-footnote>8×128 的输入乘 128×128 的权重，输出 8×128。</d-footnote>。

**原理**：
- 16384（128×128）个 ALU 排成 2D 网格
- 权重从上方"流入"（RHS），输入从左边"流入"（LHS）
- 每个 ALU 做乘加，结果向下传递

看这个动画：

{% include figure.liquid path="assets/img/systolic-array.gif" %}

权重（蓝）先对角加载，输入（绿）再对角喂入。每帧里，重叠的蓝绿单元相乘，加上从上方传来的累积结果，然后向下传一格。

输出流出的过程：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

多组输入/权重的流水线：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

一开始有流水线气泡（等数据填满），之后就是无缝连续计算。

一个 2×3 矩阵乘法的简化动画：

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

**要点**：矩阵形状要大于 MXU 边长（128），否则会有大量浪费。多 MXU 时（v4/v5 有 4 个），需要更大的分块。

v6e 的 256×256 MXU 每周期 4 倍 FLOPs，但也需要更大的张量才能喂饱。

更多动画：[YouTube](https://www.youtube.com/watch?v=sJltBQ4MOHA)，[博客](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)
