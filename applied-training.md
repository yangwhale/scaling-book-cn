---
layout: distill
title: "实战：在 TPU 上训练 LLaMA 3"
# permalink: /main/
description: "学了这么多理论，是时候动真格的了。这一章我们拿 LLaMA 3 开刀，手把手算一遍：模型多大、要多少卡、怎么切、训多久。纸上得来终觉浅，自己算一遍才踏实。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "第5部分：训练"

next_section_url: ../inference
next_section_name: "第7部分：推理"

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
  - name: "LLaMA 3 长什么样？"
  - name: "参数量和计算量"
  - name: "怎么分片训练"
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

这一章的风格和前面不一样——**我们要你自己动手算**。

每个问题我都会先藏起来答案，你可以先用纸笔算一算，再点开对答案。不要偷懒直接看答案，这样你会错过很多"原来如此"的顿悟时刻。

我们的目标是把前几章学的公式用起来，解决一个真实问题：**怎么在 TPU v5p 上训练 LLaMA 3？**

---

## LLaMA 3 长什么样？

LLaMA 3 家族<d-cite key="llama3"></d-cite>有三个主要尺寸：8B、70B、405B。我们主要拿 **70B** 来练手，8B 和 405B 留给你课后作业。

先来看看 70B 的"身材尺寸"（来自 [HuggingFace 配置](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)）：

| 参数 | 符号 | 值 |
|:-----|:----:|---:|
| 层数 | L | 80 |
| 隐藏维度 | D | 8,192 |
| FFN 维度 | F | 28,672 |
| 注意力头数 | N | 64 |
| KV 头数 | K | 8 |
| 每头维度 | H | 128 |
| 词表大小 | V | 128,256 |

这些数字怎么查？直接看 HuggingFace 上的 config.json：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

> **小技巧**：建议你自己做一个表格，把常见模型（LLaMA、Mistral、Qwen、DeepSeek 等）的参数都整理进去。这样比较不同模型的设计决策时，一目了然。

---

## 参数量和计算量

### 问题 1：验证 70B

拿着上面的表格，能算出来确实是 700 亿参数吗？

先别看答案，用[第4章](../transformers)的公式自己算算！

{% details 点击查看答案 %}

**拆解计算**：

| 模块 | 公式 | 数量 |
|:-----|:-----|-----:|
| **FFN** | D × F × 3 × L | 8192 × 28672 × 3 × 80 = **563 亿** |
| **注意力** | L × (2×D×N×H + 2×D×K×H) | 80 × (2×8192×64×128 + 2×8192×8×128) = **120 亿** |
| **词表** | 2 × V × D | 2 × 128256 × 8192 = **21 亿** |
| **总计** | | 563 + 120 + 21 = **704 亿** ✓ |

正如预期，FFN 占了大头（80%），注意力也不可忽略（17%），词表只占 3%。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：MLP 的三个大矩阵占了绝大部分参数。估算模型大小时，其他部分基本可以忽略。</p>

---

### 问题 2：每个 token 多少 FLOPs？

训练时，处理一个 token 需要多少次浮点运算？

{% details 点击查看答案 %}

还记得[第4章](../transformers)的结论吗？

$$\text{每 token FLOPs} \approx 6 \times \text{参数量}$$

所以：`6 × 70×10⁹ = 4.2×10¹¹ FLOPs/token`

换算一下：**每个 token 需要 0.42 TFLOPs**。

单张 TPU v5p 能跑 459 TFLOPs/s，所以理论上：
- 完美效率：`4.2×10¹¹ ÷ 4.59×10¹⁴ ≈ 1ms/token`

当然，实际达不到 100% 利用率，但给你一个数量级的概念。

{% enddetails %}

---

### 问题 3：训练总共多少 FLOPs？

LLaMA 3 训练了 **15 万亿个 token**。总计算量是多少？

{% details 点击查看答案 %}

简单乘法：

$$4.2 \times 10^{11} \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs}$$

这是什么概念？

- 6.3 **尧** FLOPs（Yotta，10²⁴）
- 单张 TPU 需要跑 `6.3×10²⁴ ÷ 4.59×10¹⁴ = 4.35 亿秒`
- 换算成年：**435 年**

所以我们需要很多很多 TPU 并行工作！

{% enddetails %}

---

### 问题 4：一个 Pod 训练多久？

假设用一整个 TPU v5p Pod（16×20×28 = 8960 芯片），以 40% MFU 训练。需要多长时间？

{% details 点击查看答案 %}

$$T = \frac{6.3 \times 10^{24}}{8960 \times 4.59 \times 10^{14} \times 0.4} = 3.8 \times 10^6 \text{ 秒}$$

换算：**44 天**

这个数字还挺合理的。40% MFU 是比较保守的估计，实际可能更高。

{% enddetails %}

---

### 问题 5：最少需要多少 TPU？

LLaMA 3-70B 用了约 400 万 token 的批次大小。理论上最少需要多少 TPU 才能跑起来？

（假设：bf16 参数 + fp32 优化器 + 每层 4 个检查点）

{% details 点击查看答案 %}

这是个**内存问题**。让我们算算需要多少内存：

| 用途 | 计算 | 大小 |
|:-----|:-----|-----:|
| 参数（bf16） | 2 × 70×10⁹ | 140 GB |
| 优化器（fp32，Adam） | 8 × 70×10⁹ | 560 GB |
| 激活检查点 | 2 × 8192 × 4×10⁶ × 4 × 80 | 20.9 TB |
| **总计** | | **~21.6 TB** |

注意到了吗？**激活检查点占了 96%！**即使只存 4 个检查点/层，也远超参数和优化器。

每张 TPU v5p 有 96GB HBM：

$$\frac{21.6 \times 10^{12}}{96 \times 10^9} = 225 \text{ 张}$$

理论上 225 张就够了！

**但是...**

这样训练需要 `44天 × 8960÷225 = 1752 天 ≈ 4.8 年`。

没人会这么干。所以我们用大集群不是因为内存不够，而是**需要更多算力来缩短训练时间**。

{% enddetails %}

---

### 问题 6：每张 TPU 用多少内存？

如果真的用 8960 张 TPU 训练，每张卡的内存使用是多少？

{% details 点击查看答案 %}

$$\frac{21.6 \text{ TB}}{8960} = 2.4 \text{ GB/卡}$$

才 2.4GB！连 96GB 的零头都不到！

即使激进一点（每层 12 个检查点），也只有 8GB/卡。

**结论**：大规模训练时，我们的内存利用率其实很低。瓶颈在计算，不在内存。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：理论上用很少的 TPU 就能训练大模型，只是会花很长时间。用大集群是为了快，不是为了装得下。</p>

---

## 怎么分片训练

现在进入正题：在 8960 芯片的 TPU v5p Pod 上，用 400 万 token 批次训练 LLaMA 3-70B，应该怎么分片？

### 问题 7：能只用 FSDP 吗？

假设不做序列并行，只用 FSDP。行不行？

{% details 点击查看答案 %}

**不行，而且原因可能出乎你意料。**

LLaMA 3 用长度 4096 的序列，400 万 token 意味着只有 1024 个序列。

FSDP 沿批次维度切分，所以最多只能切成 1024 份！

8960 张卡，但批次只能切 1024 份？剩下的卡没活干。

这是个很"学究"的限制，但确实存在。

{% enddetails %}

---

### 问题 8：加上序列并行呢？

如果允许在批次和序列两个维度都做 FSDP，能行吗？

{% details 点击查看答案 %}

现在可以切到 8960 份了。算算每卡批次大小：

$$\frac{4 \times 1024 \times 1024}{8960} = 468 \text{ token/卡}$$

回顾[第5章](../training)的结论：FSDP 在三轴 Pod 上的门槛是 **850 token/卡**。

468 < 850，**还是会被通信卡住**。

纯 FSDP 方案不可行。

{% enddetails %}

---

### 问题 9：混合策略呢？

FSDP + 张量并行，能保持计算受限吗？最优配置是什么？

{% details 点击查看答案 %}

**能！**

先检查门槛：混合策略的门槛是 `2550² / (2×28672) ≈ 113 token/卡`。

我们有 468 > 113，理论上可行！

最优 FSDP 分片数：

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \times 4.19 \times 10^6 \times 8960}{28672}} = 1618$$

实际配置（取 2 的幂）：
- **FSDP**：2048 路
- **张量并行**：4 路（8960 ÷ 2048 ≈ 4）

这个配置应该能跑得很好！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：LLaMA 3-70B 在完整 v5p Pod 上训练，最佳配置是数据并行（1024）+ 序列并行（2）+ 张量并行（4），总共约 2048 × 4 ≈ 8000 路。纯 FSDP 会被通信卡住，必须混合张量并行。</p>

---

## 练习题

学以致用，这两道题留给你自己做。

### 问题 1：扩展到 4 个 Pod

假设要在 **4 个 Pod**（约 36000 芯片）上用同样的批次大小训练 LLaMA 3-70B：

1. 应该用什么并行化方案？
2. 是计算受限还是通信受限？
3. 训练大约需要多久？

*提示：别忘了跨 Pod 走的是 DCN，带宽门槛不一样。*

---

### 问题 2：LLaMA 3-405B

拿出 [LLaMA 3.1-405B 的配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)，做以下分析：

**(a) 基础计算**
1. 整理出参数表（像上面 70B 那样）
2. 算算总参数量
3. 每个 token 多少 FLOPs？
4. 如果训练 15T token，总计算量是多少？

**(b) 扩展规划**
假设要在 **8 个 TPU v5p Pod** 上训练：
1. 应该用什么并行化方案？
2. 是计算受限还是通信受限？
3. 训练大约需要多久？

---

<h3 markdown=1 class="next-section">到这里，训练部分就结束了！下一章我们看[推理是怎么回事](../inference)。</h3>