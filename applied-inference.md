---
layout: distill
title: "实战：在 TPU 上服务 LLaMA 3-70B"
# permalink: /main/
description: "真刀真枪干一次。这章我们算一算：服务 LLaMA 3-70B 要多少卡？KV 缓存有多大？批次能开多大？延迟和吞吐量怎么权衡？一起来练练手。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "第7部分：推理"

next_section_url: ../profiling
next_section_name: "第9部分：性能分析"

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
  - name: "服务 LLaMA：基础计算"
  - subsections:
    - name: "吞吐量怎么算"
    - name: "预填充怎么算"
  - name: "延迟 vs 吞吐量：可视化"
  - name: "练习题"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
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

## 开场白

和训练实战章节一样，这章要你**自己动手算**。

每个问题先别看答案，用纸笔推一遍。这样才能真正理解前面章节的公式是怎么用的。

---

## 服务 LLaMA：基础计算

### 模型配置回顾

先把 LLaMA 3-70B 的参数表搬过来：

| 参数 | 符号 | 值 |
|:-----|:----:|---:|
| 层数 | L | 80 |
| 隐藏维度 | D | 8,192 |
| FFN 维度 | F | 28,672 |
| Q 头数 | N | 64 |
| KV 头数 | K | 8 |
| 头维度 | H | 128 |
| 词表大小 | V | 128,256 |

### 用什么硬件？

简单原则：**FLOPs/美元** 最高的。

| 硬件 | bf16 FLOPs/s | 单价（$/h） | FLOPs/$ |
|:-----|-------------:|----------:|--------:|
| H100 | 9.9e14 | $10.8 | 3.3e17 |
| TPU v5p | 4.59e14 | $4.2 | 3.9e17 |
| TPU v5e | 1.97e14 | $1.2 | **5.8e17** |

TPU v5e 性价比最高，是专门的推理芯片。

但注意：每个 TPU v5e 只有 **16GB HBM**，必须激进分片。

---

### 问题 1：KV 缓存多大？

LLaMA 3-70B 的 KV 缓存，每个 token 占多少内存？（假设 int8）

{% details 答案 %}

$$\text{每 token KV} = 2 \times K \times H \times L = 2 \times 8 \times 128 \times 80 = 160\text{KB}$$

**160KB/token**！

如果序列长度 32k：
$$160\text{KB} \times 32768 = 5.3\text{GB/序列}$$

批次 240：
$$5.3\text{GB} \times 240 = 1.3\text{TB}$$

对比模型参数才 70GB（int8），KV 缓存才是内存大户！

需要约 86 张 TPU v5e 才能放下这么多内存。

{% enddetails %}

---

### 问题 2：最小需要多少卡？

目标：批次 32，序列长度 8192，全部 int8。

{% details 答案 %}

**KV 缓存**：160KB × 8192 × 32 = 41.9GB

**参数**：70GB（int8，每参数 1 字节）

**总计**：41.9 + 70 = **112GB**

最少需要：112 / 16 = 7 张 TPU v5e

实际拓扑：**4×2**（8 张），或保守起见 **4×4**（16 张）

{% enddetails %}

---

### 问题 3：延迟和吞吐量是多少？

在 4×2 和 4×4 上，这个配置的理论性能？（bf16 FLOPs，int8 权重）

{% details 答案 %}

首先确认我们是带宽受限还是计算受限。

int8 权重 + bf16 FLOPs 的临界批次大小 = 120（[第7章](../inference)）。

批次 32 < 120，所以**带宽受限**。

**4×2（8 张）**：

$$T_{step} = \frac{112\text{GB}}{8 \times 8.1 \times 10^{11}} = 17\text{ms}$$

吞吐量 = 32 / 0.017 = **1882 tok/s**（总）

= **235 tok/s/chip**

**4×4（16 张）**：

$$T_{step} = \frac{112\text{GB}}{16 \times 8.1 \times 10^{11}} = 8.5\text{ms}$$

吞吐量保持 235 tok/s/chip（内存总量不变，只是加载更快）

**结论**：更大拓扑改善延迟，但不改善每芯片吞吐量。

{% enddetails %}

---

## 吞吐量怎么算

优化吞吐量 = 让 MXU 尽量忙起来 = 尽量计算受限。

### 问题 4：多大批次才能计算受限？

| 配置 | 临界批次大小 B_crit |
|:-----|------------------:|
| bf16 权重 + bf16 FLOPs | 240 |
| int8 权重 + bf16 FLOPs | 120 |
| int8 权重 + int8 FLOPs | 240 |

为什么 int8 权重 + bf16 FLOPs 反而更好？

{% details 答案 %}

$$B_{crit} = \frac{C / W_{hbm}}{\text{每参数字节数} / \text{每激活字节数}}$$

int8 权重：分母从 2 变成 1，B_crit 减半。

int8 FLOPs：分子翻倍（TPU int8 算力是 bf16 的 2 倍），抵消了。

**实践中**：量化权重比量化计算更容易做到无损，所以 int8 权重 + bf16 FLOPs 很常见。

{% enddetails %}

---

### 问题 5：最小拓扑能服务吗？

不同量化下，服务 LLaMA 3-70B 的最小配置：

{% details 答案 %}

| dtype | 参数大小 | KV/token | 最小 TPU | 实际拓扑 | 剩余 HBM | 最大批次 (8k) |
|:------|-------:|--------:|--------:|--------:|--------:|------------:|
| bf16 | 140GB | 324KB | 9 | 4×4 (16) | 116GB | 43 |
| int8 | 70GB | 162KB | 5 | 4×2 (8) | 58GB | 43 |
| int4 | 35GB | 81KB | 3 | 2×2 (4) | 29GB | 43 |

**int4 可以在 4 张卡上跑 LLaMA 70B！**

但批次只有 43，远低于 B_crit，利用率很低。

想要高吞吐量，需要更大拓扑来放更多 KV 缓存。

{% enddetails %}

---

### 问题 6：最大批次时的延迟？

把 HBM 填满时：

{% details 答案 %}

**延迟下限** = 把 16GB HBM 全读一遍的时间：

$$T = \frac{16\text{GB}}{8.2 \times 10^{11}} = 19\text{ms/step}$$

如果中位生成长度 512 token，每次请求约 **9.7 秒**。

小批次时可以更快（比如 int4 参数只需读 35GB，约 10ms）。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：延迟下限 = 参数大小 / (芯片数 × HBM 带宽)。当 KV 缓存小时，每层就是"读权重 → 算 → 丢权重"的循环。</p>

---

### 问题 7：每芯片吞吐量是多少？

假设中位生成长度 512 token：

{% details 答案 %}

$$\text{QPS/chip} = \frac{B}{T_{step} \times 512 \times N_{chips}}$$

最大批次 43，步骤时间 19ms，代入：

| dtype | QPS/chip |
|:------|--------:|
| bf16 | 0.27 |
| int8 | 0.55 |
| int4 | 1.11 |

这是理论上限。实际可能只有一半（激活内存等开销）。

{% enddetails %}

---

### 问题 8：拓扑翻倍会怎样？

{% details 答案 %}

更大拓扑 = 更多 HBM = 更大批次 = 更高每芯片吞吐量！

| dtype | 新拓扑 | 新批次 | QPS/chip |
|:------|:------|-------:|--------:|
| bf16 | 4×8 | 140 | 0.44 |
| int8 | 4×4 | ~90 | 0.90 |
| int4 | 2×4 | ~80 | 1.80 |

**最小拓扑 ≠ 最优拓扑**！

如果 KV 缓存是瓶颈，用更大拓扑能显著提升吞吐量。

{% enddetails %}

---

### 问题 9：分片策略是什么？

4×8 拓扑，bf16 服务：

{% details 答案 %}

生成只能用**张量并行**。

能做多少路？回顾 ICI 瓶颈条件：

$$Y < \frac{F \times M_Y}{2200}$$

LLaMA 3-70B：F = 28672，双轴分片（M_Y = 2）：

$$Y < \frac{28672 \times 2}{2200} = 26$$

所以最多约 16 路不会 ICI 受限，**4×4 可以，4×8 不行**。

**但是**！当带宽受限时（批次小），可以更激进：

$$Y < \frac{F}{8 \times B}$$

批次 64 时：Y < 28672 / (8 × 64) = 56。可以 4×8！

验算 4×8 各项时间：
- T_ici = 11μs
- T_hbm = 18μs（权重加载）
- T_math = 4μs

HBM 带宽仍是瓶颈，ICI 没问题。✓

**量化的额外好处**：int8/int4 权重更小，在更大批次下也能保持张量并行不受 ICI 限制。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：张量并行的极限取决于 d_ff 和分片轴数，通常 8-32 路。带宽受限时可以做更多路来降低延迟。</p>

---

## 预填充怎么算

### 问题 10：预填充延迟

8192 token 的预填充，16 张 TPU v5e，40% MFU：

{% details 答案 %}

预填充是计算受限的（长序列）。

$$T = \frac{2 \times 70 \times 10^9 \times 8192}{16 \times 1.97 \times 10^{14} \times 0.4} = 0.91\text{s}$$

**将近 1 秒！** 这是 TTFT 的主要来源。

{% enddetails %}

---

### 问题 11：连续批处理的驱逐率

假设：
- 中位预填充长度：8192
- 中位生成长度：4096
- 生成批次：32

每步有多少请求完成？多少 KV token 被驱逐？

{% details 答案 %}

每步完成的请求：32 / 4096 = 0.008（约 125 步完成一个）

每个完成的请求占用的 KV 长度：8192 + 4096 = 12288

每步驱逐的 KV token：32 × 12288 / 4096 = **96 token/step**

通用公式：B × (P + G) / G

{% enddetails %}

---

### 问题 12：预填充和生成服务器的比例

分离式部署，配置：
- 预填充：910ms（8k token）
- 生成：19ms/step，批次 32，中位生成 512 token

需要多少预填充服务器配多少生成服务器？

{% details 答案 %}

这是流水线平衡问题。

**输入速率**（预填充）：P / 0.91 序列/秒

**消费速率**（生成）：32 × G / (0.019 × 512) 序列/秒

平衡：P / 0.91 = 32 × G / 9.73

解得：**P ≈ 3G**

需要 3 倍于生成服务器的预填充服务器！

（这说明预填充是瓶颈。实践中可能用更大的预填充拓扑来改善 MFU。）

{% enddetails %}

---

## 延迟 vs 吞吐量：可视化

以 LLaMA 70B 在 TPU v5e 4×4 上为例，看看不同批次大小的效果。

**滑块控制序列长度**：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

观察要点：

1. **成本 vs 延迟的权衡极其剧烈**：延迟翻倍可以换来 100 倍成本下降！

2. **吞吐量在 B=120 附近趋平**：这是 int8 权重 + bf16 FLOPs 的临界点。

3. **长上下文时达不到饱和**：因为 HBM 放不下那么大的批次。

4. **KV 加载主导延迟**：大批次时 KV 读取时间比参数读取还长。

---

### 延迟来源分解

把步骤时间拆成：参数加载、KV 加载、FLOPs。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

关键发现：

- **小批次**：参数加载主导
- **大批次**：KV 加载和 FLOPs 主导
- **长上下文**：KV 加载时间 > FLOPs 时间！

<p markdown=1 class="takeaway">**要点**：LLaMA 3-70B 在大多数配置下都是 KV 缓存带宽受限的。减少 KV 大小对生成吞吐量至关重要。</p>

{% details 代码参考 %}

```python
import numpy as np

num_chips = 16
param_count = 70e9  # int8 = 1 byte/param
sequence_length = 8192

hbm_bandwidth = 8.20e11  # v5e
flops = 1.97e14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80  # 2 * B * H * K * L

def get_max_batch_size(seq_len, param_size, max_chips):
    for bs in range(1, 1024):
        kv = kv_cache_size(seq_len * bs)
        if (kv + param_size) > max_chips * 16e9:
            return bs - 1
    return 1024

max_bs = get_max_batch_size(sequence_length, param_count, num_chips)
batch_sizes = np.arange(1, max_bs + 1)

# 各项时间
kv_time = kv_cache_size(sequence_length * batch_sizes) / (num_chips * hbm_bandwidth)
param_time = param_count / (num_chips * hbm_bandwidth)
flops_time = 2 * param_count * batch_sizes / (num_chips * flops)

# 总延迟
mlp_time = np.maximum(flops_time, param_time)
attn_time = kv_time  # 生成时永远带宽受限
latency = 1000 * (mlp_time + attn_time)  # ms

# 吞吐量
throughput = batch_sizes / (latency / 1000 * num_chips)  # tok/s/chip
```

{% enddetails %}

---

## 练习题

### 问题 1：LLaMA 3-405B

(a) 每个 token 的 FLOPs？单张 TPU v5e 的延迟下限（忽略通信）？

(b) 如果 N 张卡且计算受限？如果通信受限？

### 问题 2：LLaMA 3-8B 服务

int8 权重 + int8 KV 缓存，批次 240，序列 8k。

(a) 参数、KV 缓存、激活各占多少内存？
(b) 最小拓扑是什么？

### 问题 3：LLaMA 3-405B 服务设计

int8 权重 + bf16 FLOPs。

(a) 如果延迟硬性限制 15ms/token，最高吞吐量配置是什么？
(b) 理论最小步骤时间？

---

<h3 markdown=1 class="next-section">下一章我们看[如何用 Profile 工具分析性能](../profiling)！</h3>
