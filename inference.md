---
layout: distill
title: "Transformer 推理全解"
# permalink: /main/
description: "推理和训练完全是两码事。训练只看吞吐量，推理还得管延迟。这章我们从『怎么生成一个 token』讲起，一直讲到『怎么搭建一个高效的推理引擎』。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "第6部分：训练 LLaMA"

next_section_url: ../applied-inference
next_section_name: "第8部分：服务 LLaMA"

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

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "推理基础：预填充与生成"
  - subsections:
    - name: "我们到底在优化什么？"
    - name: "矩阵乘法的瓶颈在哪？"
    - name: "注意力的瓶颈在哪？"
    - name: "延迟和吞吐量的理论公式"
    - name: "内存怎么算？"
    - name: "实例：LLaMA-2 13B"
  - name: "加速推理的各种技巧"
  - name: "多卡推理怎么分片？"
  - subsections:
    - name: "预填充的分片"
    - name: "生成的分片"
    - name: "KV 缓存怎么分？"
  - name: "推理引擎设计"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "实例：JetStream"
  - name: "练习题"
  - name: "附录"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 1);
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

## 推理基础：预填充与生成

模型训练好了，终于可以用了！

> 说实话，损失曲线降下来、benchmark 分数升上去，这些都是代理指标。真正有意思的时刻，是你看着模型一个字一个字往外蹦的时候。

### 朴素采样（不要这么干！）

采样的原理很简单：
1. 把一个序列喂给模型
2. 模型输出下一个 token 的概率分布
3. 从分布中采样一个 token
4. 把新 token 追加到序列，回到步骤 1

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>朴素采样。每生成一个 token 都要重新处理整个序列。FFN 是 O(n²)，注意力是 O(n³)。太慢了！" %}

问题是：**每生成一个 token，都要把前面所有 token 重新算一遍**。

生成 n 个 token：
- FFN：O(n²)
- 注意力：O(n³)

这显然不靠谱。

### KV 缓存来救场

聪明的做法是：**把中间结果存起来**。

具体来说，注意力机制里每个 token 的 Key 和 Value 投影是可以复用的。只要我们把它们存在一个叫 **KV 缓存** 的数据结构里，后续 token 就不用重新计算前面的 K 和 V 了。

有了 KV 缓存，推理分成两个阶段：

| 阶段 | 做什么 | 特点 |
|:-----|:-------|:-----|
| **预填充** | 一次性处理整个提示，生成 KV 缓存 | 可以并行，像训练一样 |
| **生成** | 一个一个吐 token，每次更新 KV 缓存 | 必须串行，一次一个 |

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>带 KV 缓存的高效采样。预填充（红色）处理提示并缓存 KV。生成（蓝色）每次只处理一个新 token，从缓存读取历史 KV。FFN 降到 O(n)，注意力降到 O(n²)。" %}

现在复杂度变成：
- FFN：O(n)
- 注意力：O(n²)

当你在 ChatGPT 里看到回答一个字一个字蹦出来，每个字（通常）就是一次单独的模型调用。

<p markdown=1 class="takeaway">**关键洞察**：预填充和生成是**完全不同的任务**！预填充像训练（批量并行），生成像一个超慢的循环（必须串行）。KV 缓存是推理特有的复杂性来源。</p>

---

### 我们到底在优化什么？

训练只关心一个指标：**吞吐量**（每秒处理多少 token）。

推理要复杂得多，因为多了一个维度：**延迟**。

| 场景 | 关心什么 |
|:-----|:---------|
| **批量推理**（评估、数据生成） | 只看成本，不管延迟 |
| **聊天/流式** | 首 token 要快（TTFT），生成速度要跟上阅读速度 |
| **边缘推理**（本地 llama.cpp） | 单用户，拼命压延迟 |

最大化硬件利用率仍然重要（省钱、降低 TTFT），但**高利用率不一定等于好体验**。

很多优化要在延迟、吞吐量、上下文长度、甚至模型质量之间做权衡。

---

### 更细致地看 Transformer

训练时我们把 Transformer 简化成"一堆矩阵乘法"。推理要更精细地分析。

Transformer 前向传播的主要组件：

1. **线性操作**
   - MLP：W_in、W_out（大）
   - 注意力投影：W_Q、W_K、W_V、W_O（小）
   - 特点：从 HBM 读参数和激活 → 算 FLOPs → 写回 HBM

2. **点积注意力**
   - 从 HBM 读 KV 缓存和 Q → 算内积和 softmax → 写回 HBM

3. **其他杂活**
   - LayerNorm、激活函数、采样、位置编码...
   - 基本可以和上面重叠或融合

接下来我们分别分析：在预填充和生成中，什么是瓶颈？

---

### 矩阵乘法的瓶颈在哪？

所有线性操作本质上都一样：bf16[B, D] × bf16[D, F]。

回顾[第1章](../roofline)的公式：

$$T_{计算} = \frac{2BDF}{C}$$

$$T_{通信} = \frac{2BD + 2DF + 2BF}{W_{hbm}}$$

当 B << D, F 时（批次小，模型大），分母约等于 2DF：

$$\frac{T_{计算}}{T_{通信}} \approx \frac{B \cdot W_{hbm}}{C}$$

要计算受限（FLOPs 是瓶颈），需要：

$$B > \frac{C}{W_{hbm}} = B_{crit}$$

| 硬件 | C/W_hbm | B_crit (bf16) |
|:-----|--------:|-------------:|
| TPU v5e | 1.97e14 / 8.2e11 | **240** |
| H100 | ~3.9e15 / 3.35e12 | ~280 |

<p markdown=1 class="takeaway">**要点**：矩阵乘法要计算受限，每副本的 token 批次大小必须超过 B_crit（TPU v5e 约 240）。</p>

#### 预填充 vs 生成

**预填充**：提示通常有几百甚至几千个 token，轻松超过 240。基本总是计算受限。

**生成**：每个请求一次只能生成一个 token！要达到 240，必须把 240 个请求批在一起。这意味着 240 个独立的 KV 缓存，实际上很难做到。

<p markdown=1 class="takeaway">**要点**：预填充基本总是计算受限。生成要达到计算受限，必须把很多请求批在一起，这很难！</p>

#### 量化的影响

如果权重量化到 int8（激活仍是 bf16）：
- 通信量减半 → B_crit 减半（约 120）

如果 FLOPs 也用 int8：
- 算力翻倍 → B_crit 又翻倍（回到 240）

所以：B_crit = β × α_hbm，其中 β = 参数位数 / 激活位数。

---

### 注意力的瓶颈在哪？

这里事情变得有趣了，因为 KV 缓存来搅局。

假设用 Flash Attention（不实体化注意力矩阵）：

**读取**：
- Q 激活：bf16[B, T, D] → 2BTD 字节
- KV 缓存：2 × bf16[B, S, D] → 4BSD 字节

**计算**：
- QK 乘法：2BSTD FLOPs
- AV 乘法：2BSTD FLOPs

**算术强度**：

$$\text{强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

#### 预填充（S = T）

自注意力，S = T：

$$\text{强度} = \frac{T^2}{2T} = \frac{T}{2}$$

只要序列长度超过 480（TPU v5e），就能计算受限。一般没问题。

#### 生成（T = 1）

每次只处理一个新 token：

$$S \gg T=1 \implies \text{强度} \approx \frac{S \cdot 1}{S+1} \approx 1$$

**强度恒定为 1**！不管批次多大、序列多长，都改变不了。

每次都要把整个 KV 缓存从 HBM 读一遍，却只做很少的计算。

<p markdown=1 class="takeaway">**要点**：预填充的注意力可以计算受限（序列够长就行）。生成的注意力**永远是内存带宽受限的**。</p>

为什么？因为每个请求都有自己的 KV 缓存。批次变大 → KV 缓存变多 → 内存读取同比例增加。没有复用，就没有收益。

---

### 延迟和吞吐量的理论公式

这是全章最重要的公式，请务必记住。

#### 生成步骤时间（小批次，带宽受限）

$$T_{step} = \frac{B \times \text{KV 缓存大小} + \text{参数大小}}{W_{hbm}}$$

#### 生成吞吐量

$$\text{Tokens/s} = \frac{B}{T_{step}} = \frac{B \times W_{hbm}}{B \times \text{KV 大小} + \text{参数大小}}$$

#### 一般情况（可能计算受限）

$$T_{step} = \underbrace{\frac{B \times \text{KV 大小}}{W_{hbm}}}_{\text{注意力（永远带宽受限）}} + \underbrace{\max\left(\frac{2B \times \text{参数}}{C}, \frac{\text{参数}}{W_{hbm}}\right)}_{\text{MLP（可能计算受限）}}$$

<b markdown=1 style="color: #57cf57;">小测验</b>：在 TPU v5e 4×4 上服务 30B 模型（int8），8192 上下文，100kB/token 的 KV 缓存，批次大小 4。最小步骤延迟是多少？批次 256 呢？

{% details 答案 %}

int8 参数 = 30GB
每序列 KV 缓存 = 100kB × 8192 = 819MB
16 芯片总带宽 = 16 × 8.1e11 = 1.3e13 B/s

**批次 4**（带宽受限）：
$$T = \frac{4 \times 819e6 + 30e9}{1.3e13} = 2.5ms$$

**批次 256**（MLP 计算受限）：
$$T = \frac{256 \times 819e6}{1.3e13} + \frac{2 \times 256 \times 30e9}{16 \times 1.97e14} = 16ms + 5ms = 21ms$$

{% enddetails %}

#### 延迟 vs 吞吐量的权衡

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图：</b>不同 PaLM 模型的延迟-吞吐量帕累托前沿。小批次快但效率低，大批次效率高但延迟大。int8 权重改善延迟但不改善最大吞吐量。" %}

<p markdown=1 class="takeaway">**要点**：关心吞吐量就用大批次（超过 B_crit ≈ 240）。关心延迟就用小批次。可能需要更大拓扑来支撑大批次。</p>

---

### 内存怎么算？

拿 LLaMA-2 13B 做例子：

| 参数 | 值 |
|:-----|:---|
| L | 40 |
| D | 5,120 |
| F | 13,824 |
| N (Q 头数) | 40 |
| K (KV 头数) | 40 |
| H | 128 |

**参数内存**：
- FFN：D² × 2.7 × 3 × L = 8.5B
- 注意力：(2×D×N×H + 2×D×K×H) × L = 4.2B
- 词表：2 × V × D = 0.3B
- 总计：**13B 参数**

bf16 = 26GB。量化可以更小。没有优化器、没有梯度。激活可以忽略（Flash Attention）。

**KV 缓存**（重点！）：

$$\text{KV 大小} = 2 \times \text{bytes} \times H \times K \times L \times T$$

LLaMA-2 13B，8192 序列，bf16：

$$8192 \times 40 \times 128 \times 40 \times 2 \times 2 = 6.7\text{GB}$$

**一个 KV 缓存就 6.7GB！4 个就超过参数了！**

这就是为什么 KV 缓存是推理的大麻烦。

---

### 实例：LLaMA-2 13B 吞吐量建模

在 8×TPU v5e（128GB HBM，6.5TB/s 带宽，1600TF/s）上：

| 批次 | KV 缓存 (GB) | 总内存 (GB) | 步骤时间 (ms) | 吞吐量 (tok/s) |
|-----:|-----------:|----------:|------------:|--------------:|
| 1 | 6.7 | 32.7 | 5.0 | 200 |
| 8 | 53.6 | 79.6 | 12.1 | 659 |
| 16 | 107.2 | 133.2 | 20.3 | 788 |
| 32 | 214.4 | 240.4 | 36.7 | 873 |
| 64 | 428.8 | 454.8 | 69.3 | 923 |
| 240 | 1608 | 1634 | 249 | 964 |

问题：
- 批次 16 就 OOM 了（>128GB）
- 收益递减严重

**如果 KV 缓存小 5 倍**（比如用 8 个 KV 头配 40 个 Q 头）：

| 批次 | KV 缓存 (GB) | 总内存 (GB) | 步骤时间 (ms) | 吞吐量 (tok/s) |
|-----:|-----------:|----------:|------------:|--------------:|
| 1 | 1.3 | 27.3 | 4.2 | 240 |
| 8 | 10.7 | 36.7 | 5.6 | 1429 |
| 16 | 21.4 | 47.4 | 7.2 | 2212 |
| 32 | 42.9 | 68.9 | 10.5 | 3048 |
| 64 | 85.8 | 111.8 | 17.0 | 3757 |
| 240 | 321.6 | 347.6 | 53.0 | 4529 |

延迟更好，吞吐量更高，批次能开更大。LLaMA-3 正是这么做的（32 个 Q 头，8 个 KV 头）。

<p markdown=1 class="takeaway">**要点**：KV 缓存大小对推理性能影响巨大。小 KV = 更大批次 + 更低延迟 + 更高吞吐量。</p>

---

## 加速推理的各种技巧

既然 KV 缓存是罪魁祸首，大家想了很多办法来压缩它：

### 1. 分组多查询注意力（GQA/MQA）

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

- **MHA**：每个 Q 头配一个 KV 头
- **GQA**：多个 Q 头共享一个 KV 头（如 LLaMA-3：4:1）
- **MQA**：所有 Q 头共享一个 KV 头

效果：KV 缓存减少 Q:KV 倍数。模型质量对此相对不敏感。

### 2. 混合局部/全局注意力

- 局部注意力只看固定窗口内的 token
- 混合使用：大部分层用局部，少数层用全局
- 超过窗口长度的部分，KV 缓存大小可以大幅减少

### 3. 跨层共享 KV

- 相邻层共享同一份 KV 缓存
- 减少存储，但可能需要多次读取（不一定改善延迟）

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="<b>图：</b>左：纯全局注意力。右：局部/全局混合 + 跨层共享。来源：Character.ai 博客。" %}

### 4. 量化

- 参数：int8、int4、fp8
- KV 缓存：也可以量化
- 好处：省内存、省带宽、降低 B_crit
- 额外好处：可以在训练后应用，不需要重新训练

### 5. Paged Attention

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid" caption="<b>图：</b>Paged Attention 把 KV 缓存存在页表里，像操作系统管理内存一样。来源：vLLM 论文。" %}

- 传统做法：为每个请求预分配最大长度的 KV 缓存
- Paged Attention：按需分配，用多少存多少
- 运行时优化，对架构透明

<p markdown=1 class="takeaway">**要点**：这些优化叠加起来，可以把 KV 缓存压缩一个数量级，推理成本也能降一个数量级。</p>

---

## 多卡推理怎么分片？

### 预填充的分片

从 roofline 角度，**预填充几乎和训练一样**。

可以用的技术：
- 张量并行（Megatron）
- 序列并行（够长的话）
- 流水线并行
- 甚至 FSDP

分片策略：
1. **先做张量并行**：直到 ICI 受限（约 4-8 路）
2. **再做序列并行**：像数据并行，但在序列维度切分

<p markdown=1 class="takeaway">**要点**：预填充的分片和训练几乎一样。张量并行到 ICI 瓶颈，然后序列并行。</p>

### 生成的分片

生成就难办多了：

- 批次小，很难达到计算受限
- 延迟敏感
- 对通信开销更敏感

**不能用的策略**：

| 策略 | 为什么不行 |
|:-----|:----------|
| FSDP | 我们是带宽受限的，不能通过 ICI 移动权重（太慢） |
| 数据并行 | 复制权重没意义，不如直接开多个副本 |
| 序列并行 | 每次只有一个 token，没序列可切 |

**只剩下张量并行**。

好消息是：因为我们是带宽受限的，可以做更激进的张量并行来改善延迟！

在训练中，ICI 瓶颈是 FLOPs 和 ICI 通信的比较。
在生成中，瓶颈是 HBM 带宽和 ICI 通信的比较。

$$T_{HBM} = \frac{2DF}{Y \cdot W_{hbm}}$$

$$T_{ICI} = \frac{2BD}{W_{ici}}$$

要 ICI 不成瓶颈：

$$Y < \frac{F}{B \cdot \beta}$$

其中 β = W_hbm / W_ici ≈ 8（TPU v5e/v6e）。

例如：F=16384，B=32 → 可以做到 64 路张量并行！

<p markdown=1 class="takeaway">**要点**：生成只能用张量并行的变体。目标是移动激活而不是 KV/参数。带宽受限时可以比训练做更多路张量并行。</p>

### KV 缓存怎么分？

KV 缓存也需要分片，而且尽量不要复制（太大了）。

分片策略：
1. **先沿头维度切**（Megatron 风格）：最多切 K 路
2. **再沿批次维度切**：KV[2, B_Z, S, K_Y, H]

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图：</b>(a) 纯张量并行的 MHA。(b) KV 缓存批次分片的 MQA。需要额外的 AllToAll 在张量分片和批次分片之间转换。" %}

代价：每层两次 AllToAll（Q 从张量分片转批次分片，输出再转回来）。

如果批次太小或上下文太长，还可以沿序列维度切 KV。

---

## 推理引擎设计

知道了怎么高效执行单次预填充和生成，还需要设计一个**推理引擎**来把它们串起来。

### 最简单的方案（不推荐）

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" %}

聚集一批请求 → 预填充 → 生成直到全部完成 → 下一批

问题：
1. **TTFT 差**：后来的用户要等前面所有人的预填充
2. **生成效率低**：短序列完成后，批次槽空着
3. **预填充浪费**：要填充到最长序列
4. **分片耦合**：预填充和生成被迫用同样的分片

这种方案只适合：边缘设备（单用户）或早期原型。

### 交错方案

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

预填充批次大小 1（立即返回），生成批多个请求。

优点：
- TTFT 好（不用等其他人）
- 生成吞吐量高（大批次）
- 预填充不用填充

缺点：
- **用户 A 的生成会被用户 B 的预填充打断**
- 延迟抖动严重

### 分离式服务（推荐）

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

预填充和生成**跑在不同的 TPU/GPU 上**。

工作流程：
1. 预填充服务器处理提示，生成 KV 缓存
2. 通过网络发送 KV 缓存到生成服务器
3. 生成服务器把多个 KV 缓存批在一起生成

优点：
- **低延迟可扩展**：用户请求不会互相阻塞
- **专业化**：预填充和生成可以用不同的分片/拓扑
- **弹性扩展**：可以独立扩展预填充和生成的容量

缺点：
- KV 缓存要走网络（但通常可接受）
- 系统更复杂

<p markdown=1 class="takeaway">**要点**：高吞吐量、低延迟的服务，通常要把预填充和生成分离到不同服务器。预填充批次 1，生成批多个请求。</p>

---

### 连续批处理

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

核心思想：
- 编译一个 prefill 函数和一个 generate 函数
- 用调度器动态管理请求队列
- 有空槽就插入新请求，完成了就移出

这样可以保持生成批次始终饱满。

---

### 前缀缓存

预填充很贵。能不能少做一点？

观察：相同前缀的请求，KV 缓存是一样的！

例如：
- "I like dogs" 和 "I like cats"
- 前两个 token 的 KV 完全相同

应用场景：
1. **多轮对话**：每轮只需要预填充新增的部分
2. **Few-shot 提示**：系统指令可以缓存起来复用

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b>用 LRU Trie 实现前缀缓存。共享前缀可以避免重复存储。来源：Character.ai 博客。" %}

实现要点：
- 本地缓存在 HBM 或主机内存
- 用 Trie 结构存储，LRU 驱逐
- 需要亲和性路由（后续请求到同一个副本）

---

### 实例：JetStream

Google 开源的推理引擎 [JetStream](https://github.com/google/JetStream)：

核心组件：
- **预填充引擎**：在独立 TPU 切片上
- **生成引擎**：在独立 TPU 切片上
- **传输线程**：协调 KV 缓存从预填充传到生成

Engine 接口：
- `prefill(tokens)` → 返回 KV 缓存
- `insert(kv_cache)` → 插入到生成批次
- `generate(batch)` → 为每个请求生成一个 token

还有 [PyTorch 版本](https://github.com/google/jetstream-pytorch)。

---

## 练习题

用这个虚构的模型练习：

| 参数 | 值 |
|:-----|:---|
| L | 64 |
| D | 4,096 |
| F | 16,384 |
| N (Q 头) | 32 |
| K (KV 头) | 8 |
| H | 256 |
| V | 32,128 |

**问题 1**：参数量和 KV 缓存大小

{% details 答案 %}

**参数**：
- MLP：L × D × F × 3 = 64 × 4096 × 16384 × 3 = 12.9B
- 注意力：L × 2 × D × H × (N + K) = 64 × 2 × 4096 × 256 × 40 = 5.4B
- 词表：D × V = 0.13B
- 总计：**18.4B**

**KV 缓存**（int8）：
2 × L × K × H = 2 × 64 × 8 × 256 = **262KB/token**

{% enddetails %}

**问题 2**：在 TPU v5e 4×4 上能开多大批次？（int8，128k 上下文）

{% details 答案 %}

每序列 KV = 262KB × 128K = 33.5GB
16 TPU × 16GB = 256GB 总 HBM
可用 = 256 - 18.4 (参数) = 237.6GB
最大批次 = 237.6 / 33.5 ≈ **7**

如果 K=1：最大批次 ≈ **56**

{% enddetails %}

**问题 3**：加载参数的理论最小时间

{% details 答案 %}

18.4B 字节 ÷ (16 × 8.1e11 B/s) = **1.4ms**

这是步骤延迟的下限。

{% enddetails %}

**问题 4**：预填充和生成怎么分片？

{% details 提示 %}

1. 4×4 ICI 结构是什么样的？
2. 张量并行的 roofline 界限？
3. KV 缓存怎么分片？

{% enddetails %}

**问题 5**：改成 MoE（E=16 专家，k=2 激活）

{% details 答案 %}

(1) 总参数 = 64 × 4096 × (3×16×16384 + 2×256×40) + 131K = **212B**
激活参数 = 64 × 4096 × (3×2×16384 + 2×256×40) + 131K = **31.2B**

(2) B_crit = 240 × (16/2) = **1920 tokens**

(3) KV 缓存不变（注意力没变）

(4) FLOPs = 2 × 激活参数 × T = 2 × 31.2B × T

{% enddetails %}

---

<h3 markdown=1 class="next-section">下一章我们看[如何实际服务 LLaMA](../applied-inference)！</h3>

---

## 附录

### 附录 A：B=240 规则的验证

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid" %}
{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid" %}

实测确实在批次 240 左右看到拐点。

### 附录 B：2D 权重静止分片

当拓扑很大时，可以同时沿 D 和 F 分片权重，让每块接近正方形。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid" %}

通信量随 √N 下降，比 1D Megatron 更好。当 N > 81 时值得考虑。

### 附录 C：延迟受限通信

当数据量很小时，通信时间被延迟（而非带宽）主导。

临界点：buffer < W_ici × 1μs ≈ 45KB

对于 BS=16, D=8192 的 int8 激活：16×8192=131KB，已经延迟受限了。

### 附录 D：推测采样

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}
{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

核心思想：用小模型快速生成草稿，大模型并行验证。

- 小模型生成 K 个 token
- 大模型一次验证 K 个
- 接受正确的，拒绝错误的

为什么快？
- 生成本来是带宽受限的
- 批量验证可以更好利用算力
- 平均每步得到 >1 个 token

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>Chinchilla 70B 用 4B 草稿模型的推测采样效果。自然语言（XSum）最优提前 3-4 个 token，代码（HumanEval）可以更激进。" %}

<p markdown=1 class="takeaway">**要点**：推测采样用吞吐量换延迟。在批次受限时（KV 缓存大、硬件小），可能两者都赢。</p>
