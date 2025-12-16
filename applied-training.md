---
layout: distill
title: "在 TPU 上训练 LLaMA 3"
# permalink: /main/
description: "让我们仔细看看如何使用前几节学到的知识在 TPU v5p 上训练 LLaMA 3 模型。它们有多大？不同配置下训练的成本是多少？如何分片？让我们通过一些粗略估算来看看前几节的内容如何映射到真实模型上。"
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
  - name: "LLaMA 3 是什么样的？"
  - name: "计算参数和 FLOPs"
  - name: "如何为训练分片 LLaMA 3-70B"
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

*本节的目标是将前一节的结果应用到一个非常实际的问题：训练 LLaMA 3 系列（家族）模型。与前几节不同，我们希望你自己完成大部分工作。因此，我们隐藏了每个部分的答案，这样你可以先尝试回答。试着拿起笔自己算一算！*

### LLaMA 3 是什么样的？

LLaMA-3 模型系列<d-cite key="llama3"></d-cite>包括 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们将主要关注 70B，将 8B 和 405B 留给你在最后的练习部分探索。这是 LLaMA 3-70B 的架构，取自 LLaMA [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**                  | **值**    |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了展示这些信息多容易找到，这里是配置本身以及映射关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

*对于许多不同的开源 LLM，制作一个包含这些数字的大表格是很有用的，这样你可以快速比较它们做出的设计决策。*

### 计算参数和 FLOPs

**问题：** 从这个表格，我们能计算 LLaMA 3-70B 的参数数量吗？🤫 让我们应用[第4章](../transformers)的内容，看看能否得到 70B！

| 参数类型         | 公式                                                                                                                                              | 数量                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| FFW 参数         | d_model * d_ff * 3（用于 gelu + 输出投影）* n_layers                                                                                              | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                     |
| 词汇表参数       | 2（输入和输出嵌入）* n_embeddings * d_model                                                                                                       | 2 * 128,256 * 8,192 = **2.1e9**                               |
| 注意力参数       | n_layers * [ 2（用于 q 嵌入和拼接输出投影）* d_model * n_heads * d_qkv + 2（用于 k 和 v）* d_model * n_kv_heads * d_qkv]                          | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9**  |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                            |

太好了！我们得到了预期的数字。你会注意到，如预期的那样，FFW 参数完全主导了整体参数数量，尽管注意力也不可忽视。

<p markdown=1 class="takeaway">**要点**：MLP 块中的 3 个大权重矩阵比 Transformer 中所有其他数组大得多，以至于我们在推理模型内存或 FLOPs 时通常几乎可以忽略所有其他参数。对于 LLaMA 3-70B，它们占 70B 参数中的 56B。</p>

现在让我们看看 FLOPs！*记住[第4章](../transformers)中关于训练的一般规则。*

**问题：** LLaMA-3 每个 token 每个训练步骤执行多少 FLOPs？*这帮助我们确定整个训练过程的成本。*

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：如[第4章](../transformers)所示，我们每个 token 执行大约 $$6 \cdot \text{参数数量}$$ 次 FLOPs，所以这里大约是 `6 * 70e9 = 4.2e11` FLOPs / token。那大约是每 token 每步骤半个 TFLOP。假设我们是计算受限的，这在单个 TPU v5p 芯片上应该需要大约 `4.2e11 / 4.59E+14 = 1ms`，假设完美的 FLOPs 利用率。

{% enddetails %}

**问题：** LLaMA 3 训练了大约 15 万亿个 token。总共需要多少 FLOPs？

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：这很简单，就是 `4.2e11 * 15e12 = 6.3e24 FLOPs` 总计。6.3 yottaFLOPs（尧 FLOPs）。这是很多！在单个 TPU 上这需要 `6.3e24 / 4.59E+14 = 435 年`。这也是很多！

{% enddetails %}

**问题：** 假设我们想在一个完整的 TPU v5p pod 上训练，具有 16x20x28 = 8960 芯片。在 bfloat16 下以 40% MFU 训练需要多长时间，假设我们是计算受限的？

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：我们知道每个 TPU v5p 可以执行 4.59e14 FLOPs / 秒。以 40% MFU，这将需要大约 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`。**这大约是 44 天！** 这相当合理，假设我们实际上可以达到 40% MFU。

{% enddetails %}

**问题：** LLaMA 3-70B 使用大约 400万 token 的批次大小进行预训练。我们至少需要多少 TPU 才能用这个批次大小训练？*你可以假设 bfloat16 参数和 float32 优化器状态，并且每层检查点梯度 4 次。*

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：这个问题主要是关于内存使用，因为这是对可用计算的唯一严格限制。在训练期间，我们有三个 HBM 的主要用途：模型参数、优化器状态和梯度检查点。如果我们假设 bfloat16 权重、float32 优化器状态和一个*非常*保守的梯度检查点方案（每层 4 次），我们有：

| **参数**         | 2 * 70GB | ~140GB |
| **优化器状态**   | 8 * 70GB | ~560GB |
| **梯度检查点**   | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计**         |                         | ~21.6TB |

这里的总计约为 21.6TB。你注意到梯度检查点强烈主导了内存图景，即使是非常保守的检查点方案。我们技术上可以减少到每层 1 个检查点，或做微批处理，但这是一个合理的图景。在这些假设下，由于每个 TPU v5p 有 96GB 的 HBM，我们需要 `21.6e12 / 96e9 = 225` 个 TPU。这实际上并不多！

*为什么我们不这样做？* 因为它会花我们 `44 天 * 8960 / 225 = 1752 天` 来训练。那将近四年。**那是很多。** 尽管如此，这清楚地表明我们使用这些大型集群不是因为我们受内存限制，而是因为我们需要额外的 FLOPs。

{% enddetails %}

**问题：** 在与上题相同的假设下，如果我们使用 8960 个 TPU v5p 芯片，每个芯片将使用多少内存？

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：我们的总内存仍然约为 21.6TB，所以每芯片我们将使用约 2.4GB，这基本上是微不足道的。如果我们做更激进的检查点，例如每层 12 个检查点，我们仍然只会在每芯片 8GB。我们在这些规模的训练中远没有接近内存受限。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：技术上可以在非常小的拓扑上训练即使是非常大的模型，但需要注意的是它们可能需要很长时间。能够计算训练运行的总 FLOPs 允许我们通过假设适度的 MFU 和已知的拓扑来粗略估计其训练时间。</p>

### 如何为训练分片 LLaMA 3-70B

让我们继续使用上面的设置，假设我们想在 8960 芯片的 TPU v5p pod 上以 400万 token 批次大小（每批次 1024 个长度为 4096 的序列）训练 LLaMA 3-70B。让我们讨论这个模型的最佳分片策略是什么。

**问题：** 在上述假设下，我们可以仅使用 FSDP 训练我们的模型吗？首先，假设我们不能做任何序列/上下文并行。*这应该是你的第一个想法，因为它简单，如果有效的话不会引入额外的通信。*

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：这个答案会有点学究气。如上所述，LLaMA 3-70B 最初使用长度为 4K 的序列训练，所以 400万 token 的批次大小给我们一个*序列批次大小*为 1024。这意味着我们实际上只能做纯数据并行/FSDP 到 1024 芯片，*因为这就是我们可以进行数据并行的序列数量*。所以在"完全数据并行没有额外通信"的简单意义上，答案是否。下一个问题将回答这个问题的一个稍微不那么学究的版本。

{% enddetails %}

**问题：** 让我们放松不做任何序列分片的要求。如果我们允许自己在批次*和*序列轴上都做 FSDP，我们可以在 8960 芯片上仅使用 FSDP 训练 LLaMA 3-70B 吗？

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：现在我们允许自己也做序列/上下文并行，我们可以扩展得更多。首先让我们计算每设备批次大小。如果我们做 8960 路 FSDP，我们的每 TPU 批次大小为 `4 * 1024 * 1024 / 8960 = 468 tokens`。我们从上一节知道，当 $$\text{每设备批次大小} < 2550 / M_X$$ 时，我们会被 FSDP 的 ICI 限制。由于我们可以在完整的 3D pod 上分配 3 个轴，这将给我们一个下限 850，我们远低于这个值。**所以答案是否，即使有 3 个轴。我们将是确定的通信受限。**

{% enddetails %}

**问题：** 现在让我们看看混合张量并行和 FSDP。是否存在某种组合让我们保持计算受限？如果是，我们应该做多少 FSDP 和张量并行？

{% details 在你思考之后，点击这里查看答案！ %}

**答案**：首先让我们检查这是否能放得下。我们知道如果每芯片批次大小小于 $2550^2 / 2F = 113$，我们将是通信受限的。如上所述，我们略高于这个值。所以这很好！现在要选择 FSDP 的最优量，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到 2 的合理倍数，这给我们大约 2048 路 FSDP 和 4 路张量并行。这应该运行良好！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以在完整的 TPU v5p pod 上以 400万 token 批次大小训练 LLaMA-3，使用数据并行（1024 路）、序列并行（2 路）和张量并行（4 路）的混合，而不会是通信受限的。如果我们尝试做纯 FSDP 或 FSDP + 序列并行，我们将是通信受限的。我们在上一节中推导的方程非常实用。</p>

## 练习题

**问题 1 [将 LLaMA 70B 扩展到更多芯片]：** 假设我们想在 4 个 pod 上以相同的批次大小训练 LLaMA 3-70B。我们会使用什么并行化方案？我们是计算受限还是通信受限？训练大约需要多长时间？*确保使用正确的 roofline 界限。*

**问题 2 [LLaMA 405B]：**

(a) 使用 LLaMA 3-405B [配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)，写一个如上所述包含所有关键超参数的表格。这个模型总共有多少参数？每个训练步骤多少 FLOPs？如果我们训练 15T 个 token，我们执行多少 FLOPs？

(b) 假设我们想在 8 个 TPU v5p pod 上训练。我们会使用什么并行化方案？训练需要多长时间？我们是计算受限还是通信受限？

<h3 markdown=1 class="next-section">第 6 章到此结束。关于 Transformer 推理的第 7 章，请点击[这里](../inference)。</h3>