
---
layout: distill
title: "Transformer 推理全解"
# permalink: /main/
description: "对 Transformer 进行推理与训练可能非常不同。部分原因是推理增加了一个新的考虑因素：延迟。在本节中，我们将从单个模型采样一个新 token 一直到作为推理引擎的一部分在多个加速器切片上高效扩展大型 Transformer。"
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
  - name: "Transformer 推理基础"
  - subsections:
    - name: "我们实际上想优化什么？"
    - name: "线性操作：什么是瓶颈？"
    - name: "注意力呢？"
    - name: "LLM 延迟和吞吐量的理论估计"
    - name: "内存呢？"
    - name: "为 LLaMA 2-13B 建模吞吐量和延迟"
  - name: "提高生成吞吐量和延迟的技巧"
  - name: "将推理分布到多个加速器"
  - subsections:
    - name: "预填充"
    - name: "生成"
    - name: "分片 KV 缓存"
  - name: "设计有效的推理引擎"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "看一个实现：JetStream"
  - name: "练习题"
  - name: "附录"

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

## Transformer 推理基础

你已经训练好了一个 Transformer，现在想用它生成一些新序列。*归根结底，基准分数上升和损失曲线下降只是一些代理指标，用来判断一旦真正运行起来是否会发生有趣的事情！*<d-footnote>从历史上看，你可以在不接触推理的情况下对 Transformer 进行大量研究——LLM 损失、多选基准测试可以在没有适当的 KV 缓存或生成循环实现的情况下高效运行。这意味着，特别是在研究代码库中，推理代码路径中往往有很多唾手可得的优化机会。</d-footnote>

采样在概念上很简单。我们输入一个序列，我们最喜欢的 Transformer 会输出 $$\log p(\text{下一个token}_i \vert \text{之前的tokens})$$，即所有可能的下一个 token 的对数概率。我们可以从这个分布中采样并获得一个新 token。将这个 token 追加到序列并重复这个过程，我们就获得了一个 token 序列，它是提示的延续。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>从 Transformer 朴素采样。蓝色的 logits 给我们一个下一个 token 的分布，我们可以从中采样。注意每一步都重新处理整个前缀，导致算法运行时间为 $\Theta(n^2)$。" %}

我们刚刚描述了 Transformer 采样的朴素实现，虽然它能工作，**但我们在实践中从不这样做**，因为我们每次生成 token 时都在重新处理整个序列。这个算法在 FFW 上是 $$O(n^2)$$ 的，在注意力机制上是 $$O(n^3)$$ 的，用于生成 $$n$$ 个 token！

**我们如何避免这种情况？** 事实证明，我们可以从每次前向传播中保存一些中间激活值，而不是每次都做完整的前向传播，这让我们可以避免重新处理之前的 token。具体来说，由于一个给定的 token 在点积注意力期间只关注之前的 token，我们可以简单地将每个 token 的键和值投影写入一个名为 **KV 缓存**的新数据结构。一旦我们为过去的 token 保存了这些键/值投影，未来的 token 可以简单地计算它们的 $$q_i \cdot k_j$$ 乘积，而不需要对较早的 token 执行任何新的 FLOPs。太棒了！

考虑到这一点，推理有两个关键部分：

* <b style="color: red;">预填充</b>：给定一个长提示，我们同时处理提示中的所有 token，并将结果激活值（具体来说是键值投影）保存在 **"KV 缓存"**中。我们还保存最后一个 token 的 logits。
* <b style="color: blue;">生成</b>：给定一个 KV 缓存和之前的 logits，我们从 logits 中增量采样一个 token，将该 token 反馈给 Transformer，并为下一步生成一组新的 logits。我们还将该新 token 的 KV 激活值追加到 KV 缓存中。我们重复这个过程，直到遇到特殊的 `<EOS>` token 或达到某个最大长度限制。

这是带 KV 缓存的采样图示：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>带 KV 缓存的高效 Transformer 采样图示。<b style=\"color: red;\">预填充</b>处理我们的提示并将所有每 token 的键值激活值保存在缓存中。<b style=\"color: blue;\">生成</b>获取这个缓存（和最后一个 token 的 logits），采样一个新 token，并将该新 token 通过模型传递，关注 KV 缓存并将新 token 的键值投影保存回缓存。这在 MLP 块中是一个 $O(n)$ 算法。" %}

通过使用 KV 缓存采样，我们将生成 $n$ 个 token 的时间复杂度降低到 FFW 上的 $$O(n)$$ 和注意力上的 $$O(n^2)$$，因为我们永远不会重新处理之前的 token。然而，仍然需要多次前向传播来生成一个序列——这就是当你查询 Gemini 或 ChatGPT 时结果流回给你时发生的事情。每个 token（通常）是对一个大型模型的单独（但部分缓存）Transformer 调用。

我们很快会看到<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是非常不同的任务——Transformer 推理实际上是两个伪装的任务！与训练相比，KV 缓存也是一个新的且重要的复杂性来源。

### 我们实际上想优化什么？

在我们继续之前，值得强调推理的一个全新方面：延迟。在训练期间我们只关心吞吐量（**每芯片**每秒处理的总 token 数），而在推理期间我们必须担心我们生成 token 的速度（**首 token 时间（TTFT）**和**每 token 延迟**）。例如：

* **离线批量推理**用于评估和数据生成，只关心推理的批量成本，对单个样本的延迟不敏感。
* **聊天界面/流式任务**需要在规模上廉价运行，同时具有低 TTFT 并以足够快的速度生成 token 以超过人类阅读速度。
* **边缘推理**（例如你笔记本电脑上的 `llama.cpp`）只需要以尽可能低的延迟为一个用户服务，可能有严格的硬件限制。

最大化硬件利用率仍然很关键，有助于成本和 TTFT，但与训练不同，它*不一定*在所有上下文中都能为个别用户带来更好的体验。加速器、系统和模型架构层面的许多优化在延迟、吞吐量、上下文长度甚至模型质量之间做出权衡。

### 更细粒度地看 Transformer

到目前为止，我们主要将 Transformer 视为一堆前馈块。虽然从 FLOPs 和内存的角度来看这通常是合理的，但它不足以正确建模推理。<d-footnote>你会注意到本节的一件事是，推理比训练要求更高。我们通常有更少的 FLOPs，更少的批处理机会，以及对延迟更高的敏感性。KV 缓存也大大增加了推理的复杂性。</d-footnote>正如我们在[第4章](../transformers)中看到的，Transformer 前向传播的主要组件是：

1. **一堆线性操作**，包括 MLP（$W_{in}$、$W_{out}$）和注意力的 QKV 投影及输出投影（$W_Q$、$W_K$、$W_V$ 和 $W_O$）。这些都涉及从 HBM 读取参数和一批激活值，做一些 FLOPs，然后将结果写回 HBM。
2. **点积注意力**。我们需要从 HBM 读取一批键值投影和一批查询激活值，做几个内积和一些 softmax 操作，然后将注意力结果写回 HBM。
3. **其他一切**，包括应用层归一化、激活函数、token 采样、更新 KV 缓存和位置嵌入。这些确实需要一些 FLOPs，但被上述操作主导或融合到上述操作中。

在接下来的几节中，我们将在预填充和生成的上下文中查看每一个，并询问什么可能会成为我们性能的瓶颈。在单个加速器内，我们是计算受限还是内存受限？我们想强调预填充与生成的答案会有多么不同。

### 线性操作：什么是瓶颈？

我们所有的线性操作在概念上是相同的，无论它们在 MLP 块还是注意力中。它们的算术强度取决于批次大小。我们在[第1章](../roofline)中做过这个数学，但值得重复。让我们看看 $\text{bf16[B, D]}$ 批次与 $\text{bf16[D, F]}$ 矩阵的单个矩阵乘法。这可以是大的 MLP 块（$W_\text{in}$ 或 $W_\text{out}$）或较小的注意力投影之一（$W_Q$、$W_K$、$W_V$、$W_O$）。要做这个矩阵乘法，我们需要将这两个数组从 HBM 加载到 MXU，做乘法，然后将结果写回 HBM。如前所述，我们有：

$$T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}} = \frac{2BDF}{\text{加速器 FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{通信字节数}}{\text{带宽 Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{带宽 Bytes/s}}$$

TPU 或 GPU 可以在加载时进行计算来重叠这些，所以要成为计算受限的，我们需要 $$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中右边是我们硬件的算术强度。现在假设 $D$ 和 $F$ 与 $B$ 相比非常大（通常我们的批次最多是 500，而 $D$ 和 $F > 10k$），我们可以通过使用 $\small{2BD + 2DF + 2BF \approxeq 2DF}$ 来简化分母，这给我们

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们量化权重或使用更低精度的 FLOPs 进行矩阵乘法，这个临界批次大小可以改变。例如，如果我们将权重量化为 int8 或 fp8，$B_\text{crit}$ 减少 2 倍。如果我们用 int8 或 fp8 做 FLOPs，$B_\text{crit}$ 增加 2 倍。因此，如果我们令 $\beta = \text{每参数位数} / \text{每激活位数}$ 和 $\alpha_\text{hbm} = C / W_\text{hbm}$，我们的临界批次大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** Transformer 矩阵乘法是计算受限的*当且仅当*每副本**token**批次大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{每参数位数} / \text{每激活位数}) = \beta \cdot \alpha_\text{hbm}$。对于 TPU v5e 上的 bf16 激活，这是 240 个 token。对于 H100，这大约是 280 个 token。</p>

在训练期间，我们在所有矩阵乘法中都会有很高的强度，因为我们在非常大的批次上重用相同的权重。**这种高算术强度也延续到预填充，因为用户提示通常有数百甚至数千个 token 长。** 如前所述，TPU v5e 的硬件算术强度是 240，所以如果一个超过 240 个 token 的序列以 bf16 被馈入在此硬件上运行的稠密模型，我们预期会是计算受限的，一切都很好。比这更短的提示技术上可以批处理在一起以达到更高的利用率，但这通常不是必要的。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，所有矩阵乘法基本上总是计算受限的。因此，简单地最大化硬件利用率或 MFU（模型 FLOPs 利用率）足以最大化每芯片吞吐量（成本）和延迟（以 TTFT 的形式）。除非提示非常短，否则每提示级别的批处理只会增加延迟，对预填充吞吐量的改进很小。</p>

然而，在生成期间，对于每个请求，我们只能一次一个 token 地做前向传播，因为步骤之间存在顺序依赖！因此我们只能（容易地）通过将多个请求批处理在一起来达到良好的利用率，在批次维度上并行化。我们稍后会更多地谈论这个，但实际上将许多并发请求批处理在一起而不影响延迟是很难的。因此，**用生成来饱和硬件 FLOPs 要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成期间，总 token 批次大小必须大于 $B_{\text{crit}}$ 才能在线性/前馈操作上是计算受限的（TPU v5e 上 bf16 参数为 240）。因为生成是串行的、逐 token 的，这要求我们将多个请求批处理在一起，这很难！</p>

*值得注意的是这有多大！* 生成批次大小 240 意味着 240 个并发请求同时生成，对于稠密模型意味着 240 个独立的 KV 缓存。这意味着这在实践中很难实现，除了一些批量推理设置。相比之下，在预填充期间推送超过 240 个 token 是相当常规的，尽管随着稀疏性增加需要一些小心。

**注意这个确切的数字会因量化和硬件类型而异。** 加速器通常可以在更低精度下提供更多算术。例如，如果我们有 int8 参数但用 bf16 做计算，临界批次大小下降到 120。使用 int8 激活和 int8 参数，它跳回到 240，因为 TPU v5e 可以提供 400 TOPs/s 的 int8 x int8。

### 注意力呢？

当我们看点积注意力操作时，事情变得更加复杂，特别是因为我们必须考虑 KV 缓存。让我们只看一个纯多头注意力的注意力头。在单个 Flash Attention 融合中，我们<d-footnote>我们在这里通过忽略应用 softmax、掩码等的非矩阵乘法 FLOPs 来大大简化。它们应该与计算或 HBM 读取重叠，但在某些 TPU 代上可能非平凡。这些细节不会改变主要信息，即 KV 缓存通常是内存受限的。</d-footnote>：

1. 从 HBM 读取形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活值。
2. 从 HBM 读取 $KV$ 缓存，它是一对 $\text{bf16[B, S, D]}$ 张量。
3. 在 $$QK$$ 矩阵乘法中执行 $2BSTD$ FLOPs。使用 Flash Attention，我们不需要将 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM。
4. 在注意力 $$AV$$ 矩阵乘法中执行 $2BSTD$。
5. 将结果 $\text{bf16[B, T, D]}$ 张量写回 HBM。

综合起来，我们得到：

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$ 因为我们在做自注意力，所以这简化为 $T^2 / 2T = T / 2$。这很好，因为这意味着**预填充期间注意力的算术强度是 $\Theta(T)$**。这意味着相当容易对注意力变成计算受限。只要我们的序列长度相当大，我们就没问题！

但因为生成有一个微不足道的序列维度，且 $B$ 和 $D$ 维度抵消，我们可以做近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着我们无法做任何事情来改善生成期间注意力的算术强度。我们在加载大量 KV 缓存时只做很少量的 FLOPs。**所以我们在注意力期间基本上总是内存带宽受限的！**

<p markdown=1 class="takeaway">**要点：** 在预填充期间，对于任何合理的序列长度（大约 $\gt 480$ token），注意力通常是计算受限的，而在生成期间，我们的算术强度低且恒定，所以我们总是内存带宽受限的。</p>

*这在概念上是为什么？* 主要是因为我们在模型的线性部分是计算受限的，是因为参数（内存带宽密集型组件）被许多批次项重用。然而，每个批次项都有自己的 KV 缓存，所以更大的批次大小意味着更多的 KV 缓存。我们几乎*总是*在这里是内存受限的，除非架构被激进地调整。

这也意味着，一旦参数内存与 KV 缓存内存相当，增加批次大小的吞吐量收益递减。收益递减对你伤害的程度取决于单个序列的参数与 KV 缓存字节的比率，即大约 $2DF / SHK$ 的比率。由于 $HK\approx D$，这大致取决于 $F$ 与序列长度 $S$ 的比率。这也取决于使 KV 缓存更小的架构修改（我们马上会说更多）。

### LLM 延迟和吞吐量的理论估计

从这个数学，我们可以得到优化时应该瞄准的步骤时间的相当好的界限。**（注意：如果有一件事我们希望读者从这整章中带走，那就是以下内容）。** 对于生成期间的小批次大小（这很常见），我们可以通过假设在注意力和 MLP 块中都是内存带宽受限来下限我们的每步延迟：

$$\begin{equation*}
\text{理论最小步骤时间} = \frac{\text{批次大小} \times \text{KV 缓存大小} + \text{参数大小}}{\text{总内存带宽}}
\end{equation*}$$

类似地，对于吞吐量：

$$\begin{equation*}
\text{理论最大 Tokens/s} = \frac{\text{批次大小} \times \text{总内存带宽}}{\text{批次大小} \times \text{KV 缓存大小} + \text{参数大小}}
\end{equation*}$$

最终，随着批次大小增长，FLOPs 开始主导参数加载，所以在实践中我们有更一般的方程：

$$\begin{align}
\tiny \text{理论步骤时间（一般）} = \underbrace{\frac{\text{批次大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批次大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能是计算受限）}}
\end{align}$$

其中注意力组件（左）永远不会是计算受限的，因此不需要 FLOPs roofline。这些对于粗略计算相当有用，例如

<b markdown=1 style="color: #57cf57;">小测验：</b>假设我们想从一个 30B 参数的稠密模型在 TPU v5e 4x4 切片上以 int8 和 bf16 FLOPs、8192 上下文和每 token 100 kB 的 KV 缓存，以 4 个 token 的批次大小进行生成步骤。这个操作的合理延迟下限是多少？如果我们想采样 256 个 token 的批次呢？

{% details 点击这里查看答案。 %}

**答案：** 在 int8 中，我们的参数将使用 30e9 字节，根据给定的规格，我们的 KV 缓存每个将使用 `100e3 * 8192 = 819MB`。我们有 16 个芯片，每个有 `8.1e11` bytes/s 的带宽和 `1.97e14` bf16 FLOPs/s。根据上面的方程，由于我们有小批次大小，我们预期步骤时间至少是 `(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`。在 256 个 token 时，我们将很好地处于 MLP 块的计算受限区域，所以我们的步骤时间大约是 `(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里有一个明确的吞吐量和延迟之间的权衡。小批次快但不能很好地利用硬件。大批次慢但高效。这是为一些较老的 PaLM 模型计算的延迟-吞吐量帕累托前沿（来自 [ESTI 论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图：</b>几个 PaLM 模型的成本（即吞吐量）与延迟的帕累托前沿。注意芯片数量（C）和批次大小（B）如何沿帕累托前沿移动你，除了绿点（PaLM 540B 的 C:32 B:16），那里可用内存阻止了设置支持一个好的批次大小，导致吞吐量受损。注意吞吐量通常在批次大小 240 左右后趋于平缓。int8 权重提供了更好的延迟-吞吐量帕累托最优，但不是更好的最大吞吐量。" %}

我们不仅用批次大小作为旋钮来权衡延迟和吞吐量，如果我们发现自己受到 HBM 限制，我们也可能更喜欢更大的拓扑而不是更小的拓扑，以便我们可以容纳更大的批次。[下一节](../applied-inference)更详细地探讨这一点。

<p markdown=1 class="takeaway">**要点：** 如果你关心生成吞吐量，使用尽可能大的每芯片批次大小。任何超过 TPU 算术强度（$B_\text{crit}$，通常是 120 或 240）的每芯片批次大小都会最大化吞吐量。你可能需要增加拓扑来实现这一点。较小的批次大小将允许你以牺牲吞吐量为代价来改善延迟。</p>

{% details 从硬件角度来看，这有一些注意事项。点击这里查看一些细节。 %}

这一切都相当理论化。在实践中，我们经常由于几个原因没有看到尖锐的 roofline：

* 我们假设 HBM 读取将与 FLOPs 完美重叠是不现实的，因为我们的编译器（XLA）是有缺陷的。
* 对于分片模型，XLA 也经常无法有效地将我们模型分片矩阵乘法的 ICI 通信与 FLOPs 本身重叠，所以我们经常在 $$\text{BS}=32$$ 以上开始在线性操作上受到延迟影响。
* 大于理论 roofline 的批次大小仍然会看到一些吞吐量改善，因为不完美的重叠，但这是一个好的启发式。

{% enddetails %}

### 内存呢？

我们花了一些时间看带宽和 FLOPs，但没有看内存。由于我们的新数据结构——KV 缓存，推理时的内存图景看起来很不一样。对于这一节，让我们选择一个真实的模型（LLaMA 2-13B）来展示事情有多么不同：

| 超参数             | 值     |
| ------------------ | ------ |
| L (num_layers)     | 40     |
| D (d_model)        | 5,120  |
| F (ffw_dimension)  | 13,824 |
| N (num_heads)      | 40     |
| K (num_kv_heads)   | 40     |
| H (qkv_dim)        | 128    |
| V (num_embeddings) | 32,000 |

推理期间什么在使用内存？显然是我们的参数。计算它们，我们有：

| 参数类型         | 公式                                                                                                          | 大小（字节）                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| FFW 参数         | d_model<sup>2</sup> x ffw_multiplier x 3（用于 gelu + 输出投影）x n_layers                                    | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                        |
| 词汇表参数       | 2（输入和输出嵌入）x n_embeddings x d_model                                                                   | 2 x 32,000 x 5,120 = **0.3e9**                                  |
| 注意力参数       | [2（*q 和输出*）x d_model x n_heads x d_qkv + 2（*用于 k 和 v*）x d_model x n\_kv\_heads x d_qkv] x n_layers  | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9**  |

把这些参数加起来，我们得到 8.5e9 + 4.2e9 + 0.3e9 = **13e9 总参数**，正如预期。如前几节所示，在训练期间我们可能以 bfloat16 存储参数，优化器状态用 float32。这可能使用大约 100GB 的内存。这与我们的梯度检查点相比微不足道，后者可能使用几 TB。

**推理有何不同？** 在推理期间，我们存储一份参数副本，比如用 bfloat16。这使用 26GB——实际上通过量化我们通常可以做得比这好得多。没有优化器状态或梯度需要跟踪。因为我们不做检查点（为反向传播保留激活值），我们的激活占用对于预填充<d-footnote>特别是因为 Flash Attention，它避免了实体化我们的注意力矩阵</d-footnote>和生成都是可以忽略的。如果我们预填充 8k 个 token，单个激活只使用大约 `8,192 x 5,120 x 2 bytes = 80MB` 的内存。更长的预填充可以分解成许多更小的前向传播，所以对于更长的上下文也不是问题。生成使用的 token 更少，所以激活可以忽略。

**主要区别是 KV 缓存**。这些是所有过去 token 的键和值投影，大小仅受最大允许序列长度限制。$$T$$ 个 token 的总大小是

$$\text{KV 缓存大小} = 2 \cdot \text{每浮点数字节数} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个头的维度，$$K$$ 是 KV 头的数量，$$L$$ 是层数，2 来自同时存储键和值。

**这可以很快变得很大**，即使是适度的批次大小和上下文长度。对于 LLaMA-13B，bf16 下单个 8192 序列的 KV 缓存是

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**只需 4 个就超过了我们参数的内存使用！** 需要说明的是，LLaMA 2 没有针对更长上下文的 KV 缓存大小进行优化（并不总是这么糟糕，因为通常 $K$ 小得多，如 LLaMA-3），但这仍然说明问题。我们在内存或延迟估计中不能忽略这些。

### 为 LLaMA 2-13B 建模吞吐量和延迟

让我们看看如果我们尝试在 8xTPU v5e 上以不同批次大小完美高效地执行生成会发生什么，直到早先推导的临界批次大小（240）以获得最大理论吞吐量。

| 批次大小                          |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV 缓存内存 (GiB)                 |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存 (GiB)                      |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步骤时间 (ms)                 |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量 (tokens/s)             | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5e 给我们 128GiB 的 HBM，6.5TiB/s 的 HBM 带宽（每个 0.82TiB/s）和 1600TF/s 的计算。

对于这个模型，增加批次大小确实给我们更好的吞吐量，但我们遭受快速递减的收益。我们在批次大小 16 以上 OOM，需要多一个数量级的内存才能接近 240。更大的拓扑可以改善延迟，但我们在每芯片吞吐量上碰壁了。

假设我们保持参数总数相同，但神奇地使 KV 缓存小 5 倍（比如，使用 1:5 [GMQA](#提高生成吞吐量和延迟的技巧)，这意味着我们有 8 个 KV 头共享给 40 个 Q 头——详见下一节）。

| 批次大小                          |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV 缓存内存 (GiB)                 |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存 (GiB)                      |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步骤时间 (ms)                 |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量 (tokens/s)             | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

使用更小的 KV 缓存，我们仍然有递减的收益，但理论上每芯片吞吐量继续扩展到批次大小 240。我们可以容纳大得多的批次 64，延迟在所有批次大小上也始终更好。延迟、最大吞吐量和最大批次大小都显著改善！事实上，后来的 LLaMA 代使用了正是这种优化——LLaMA-3 8B 有 32 个查询头和 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：** 除了参数，KV 缓存的大小对模型的最终推理性能有很大影响。我们希望通过架构决策和运行时优化的组合来控制它。</p>

## 提高生成吞吐量和延迟的技巧

自原始的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762)以来，已经开发了许多技术来使模型更高效，通常专门针对 KV 缓存。一般来说，更小的 KV 缓存使得更容易增加生成步骤的批次大小和上下文长度而不影响延迟，并使围绕 Transformer 的系统（如请求缓存）更容易处理。忽略对质量的影响，我们可能会看到：

**分组多查询注意力（又名 GMQA、GQA）：** 我们可以减少 KV 头的数量，并在注意力机制中与许多 Q 头共享它们。在极端情况下，可以在所有 Q 头之间共享单个 KV 头。这将 KV 缓存减少了 Q:KV 比率的倍数（相对于纯 MHA），并且已经观察到模型的性能对这种变化相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效地增加了注意力计算的算术强度（参见[第4章](../transformers)的问题 4）。

**混合一些局部注意力层：** 局部注意力将上下文限制在一个中小尺寸的最大长度。在训练时间和预填充时间，这涉及将注意力矩阵掩码为对角条带而不是三角形。这有效地限制了局部层的 KV 缓存的最大长度。通过将一些局部层与一些全局层混合到模型中，在超过局部窗口的上下文处，KV 缓存的大小大大减少。

**跨层共享 KV：** 模型可以学习以某种模式跨层共享相同的 KV 缓存。虽然这确实减少了 KV 缓存大小，并在增加批次大小、缓存、离线存储等方面提供好处，但共享的 KV 缓存可能需要从 HBM 多次读取，*所以它不一定改善步骤时间。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="<b>左：</b>纯全局注意力的多层。<b>右：</b>一些全局/局部交错模式的示例，与相邻层共享。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。"%}

**量化：** 推理通常对参数和 KV 的精度不太敏感。通过量化参数和 KV 缓存（例如到 int8、int4、`fp8` 等），我们可以节省两者的内存带宽，减少达到计算 roofline 所需的批次大小，并节省内存以在更大的批次大小下运行。量化有一个额外的优势，即使模型没有用量化训练，它通常也可以在训练后应用。

**使用不规则 HBM 读取和 Paged Attention：** 我们在上面的计算中为每个 KV 缓存分配了 8k 的上下文，但通常不需要从内存中读取整个 KV 缓存——请求有各种长度分布，不使用模型的最大上下文，所以我们通常可以实现只读取 KV 缓存非填充部分的内核（例如 Flash Attention 变体）。

Paged Attention<d-cite key="paged"></d-cite> 是对此的改进，它将 KV 缓存存储在操作系统风格的页表中，基本上完全避免填充 KV 缓存。这增加了很多复杂性，但意味着每个批次只使用它需要的内存。这是一个运行时优化，所以它对架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b>在生成期间，单个 token（第四个）关注多个 KV 缓存块/页。通过分页 KV 缓存，我们避免加载或存储超过我们需要的内存。取自 <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention 论文</a>。" %}

<p markdown=1 class="takeaway">**总体看法：** 总的来说，这些 KV 缓存优化可以将 KV 缓存大小比标准 MHA Transformer 减少超过一个数量级。这可以导致 Transformer 整体成本的数量级改善。</p>

## 将推理分布到多个加速器

到目前为止，我们已经含糊地说了如何扩展到单个芯片之外。遵循[第5章](../training)，让我们探索可用的不同策略及其权衡。与往常一样，我们将分别查看预填充和生成。

### 预填充

从 roofline 的角度来看，**预填充几乎与训练相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（对于足够长的上下文）、流水线，甚至 FSDP 都是可行的！你只需要保留 KV 以便稍后进行生成。与训练一样，增加芯片数量使我们可以访问更多 FLOPs/s（可能降低 TTFT），但增加通信开销（可能降低每芯片吞吐量）。

**分片预填充的一般规则：** 这是预填充的一般规则集。我们假设我们只在单个序列上进行预填充（没有批次维度）：

1. *模型分片：* 我们通常首先做一些模型并行，直到我们变成 ICI 受限。如我们在[第5章](../training)中看到的，对于 1 个轴这大约是 $F / 2200$（通常大约 4-8 路分片）。
2. *序列并行：* 超过这个，我们做序列并行（像数据并行但在序列维度上分片）。虽然序列并行在注意力中引入一些额外通信，但在较长上下文时通常相当小。与训练一样，我们可以重叠通信和计算（分别使用集合矩阵乘法用于 Megatron 和环形注意力）。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，几乎任何在训练期间可以工作的分片都可以正常工作。做模型并行直到 ICI 界限，然后做序列并行。</p>

### 生成

生成比预填充更复杂。首先，更难获得大批次大小，因为我们需要将许多请求批处理在一起。延迟目标更低。综合起来，这些意味着我们通常更加内存受限，对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP 是不可能的：** 因为我们在从 HBM 加载参数和 KV 缓存到 MXU 时是内存受限的，我们不想通过比 HBM 慢几个数量级的 ICI 移动它们。*我们想移动激活而不是权重。* 这意味着类似 FSDP 的方法对于生成通常完全不可行。<d-footnote>在训练后不小心保留它是一种容易且常见的导致数量级回归的方式</d-footnote>

2. **没有理由做数据并行：** 纯数据并行没有帮助，因为它复制我们的参数并且不能帮助我们更快地加载参数。你最好启动模型的多个副本。<d-footnote>我们的意思是，启动多个具有较小批次大小的模型副本的服务器。模型级别的数据并行严格更差。</d-footnote>

3. **没有序列 = 没有序列分片。** 祝你序列分片顺利。

*这主要给我们留下了用于稠密模型生成的模型分片变体*。与预填充一样，我们可以做的最简单的事情是简单的模型并行（激活完全复制，MLP 的权重完全沿隐藏维度分片）直到我们变成 ICI 受限的 4-8 路。然而，由于我们经常是内存带宽受限的，我们实际上可以超越这个限制来改善延迟！

**关于生成的 ICI 界限的注释：** 在训练期间我们希望是计算受限的，所以我们的 roofline 看的是我们的 ICI 通信何时比 FLOPs 花费更长时间。然而，在生成期间，如果我们被参数加载的内存带宽限制，我们可以将模型分片增加到超过这个点，以最小的吞吐量成本（以 tokens/sec/chip 计）改善延迟。更多模型分片给我们更多 HBM 来加载权重，我们的 FLOPs 不重要。<d-footnote>在 FLOPs 时间没有成为瓶颈的意义上，所以我们需要担心的是 ICI 时间超过参数加载时间。</d-footnote>让我们看看在它成为瓶颈之前我们可以做多少模型并行。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。这个数字对于 TPU v5e 和 TPU v6e 通常大约是 8。这意味着例如如果 $F$ 是 16,384 且 $B$ 是 32，理论上我们可以做模型并行到 `16384 / (32 * 8) = 64` 路而不会对吞吐量有有意义的影响。这假设我们可以完全 64 路分片我们的 KV 缓存，这很困难：我们在下面讨论这个。

对于注意力层，我们也用 Megatron 风格在头上模型分片注意力 $$W_Q$$ 和 $$W_O$$。KV 权重相当小，复制它们通常比超过 $K$ 路分片更便宜。

<p markdown=1 class="takeaway">**要点：** 我们在生成期间的唯一选择是模型并行的变体。我们的目标是移动激活而不是 KV 缓存或参数，后者更大。当我们的批次大小很大时，我们做模型并行直到 FLOPs-ICI 界限（$F / \alpha$）。当我们的批次大小较小时，我们可以通过更多模型分片来改善延迟（以适度的吞吐量成本）。当我们想要模型分片超过我们 KV 头数时，我们也可以沿批次维度分片我们的 KV。</p>

### 分片 KV 缓存

**我们还有一个需要分片的额外数据结构——KV 缓存。** 同样，我们几乎总是更喜欢避免复制缓存，因为它是注意力延迟的主要来源。为此，我们首先沿头维度 Megatron 分片 KV。这限于 $K$ 路分片，所以对于头数较少的模型，我们尽可能多地分片头维度，然后沿批次维度分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这意味着 KV 缓存是完全分布的。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图：</b>注意力机制的比较，(a) 带纯模型分片的多头注意力和 (b) 带 KV 缓存批次分片的多查询注意力。注意我们需要每个注意力层两次额外的 AllToAll 来将激活从模型分片转移到批次分片，以便它们可以作用于 KV 缓存。" %}

这样做的成本是每个注意力层两次 AllToAll——一次将 Q 激活转移到批次分片以便我们可以用批次分片计算注意力，一次将批次分片的注意力输出转回纯模型分片。

{% details 这是完整算法！ %}

这里我们将写出在 $Y$ 和 $Z$ 上都有模型并行的完整注意力算法。我为同时使用 $K$ 表示键张量和 KV 头维度道歉。令 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（现有激活，从前一层未分片）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ...（现有 KV 缓存，批次分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你可以大致看出它是如何工作的。新的通信相当便宜，因为它们操作我们小的激活，而作为回报，我们节省了大量加载 KV 的内存带宽（它们是静止的）。

</div>

{% enddetails %}

* **序列分片：** 如果批次大小太小，或上下文很长，我们可以序列分片 KV 缓存。同样，我们在这里累积跨分片的注意力时支付一个集合成本。首先我们需要 AllGather Q 激活，然后以类似于 Flash Attention 的方式累积 KV。

## 设计有效的推理引擎

到目前为止，我们已经看了如何在隔离中高效地优化和分片单独的预填充和生成操作。要实际有效地使用它们，我们需要设计一个推理引擎，可以在延迟/吞吐量帕累托前沿的我们选择的点上馈送这两个操作。

最简单的方法是简单地运行一批预填充，然后一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>在最简单的设置中，请求被聚合，服务器在运行一批预填充和为所有序列调用生成函数直到完成之间交替。" %}

这很容易实现，是大多数代码库中的第一个推理设置，但它有多个缺点：

1. **延迟很糟糕。** 我们耦合了预填充和生成批次大小。在大预填充批次大小时首 token 时间（TTFT）很糟糕——你需要在任何用户看到任何 token 之前完成所有预填充。在小批次大小时生成吞吐量很糟糕。
2. **我们阻塞较短生成在较长生成上。** 许多序列会在其他序列之前完成，在生成期间留下空批次槽，进一步损害生成吞吐量。问题随着批次大小和生成长度增加而加剧。
3. **预填充被填充。** 预填充被填充到最长序列，我们浪费了大量计算。这有解决方案，但历史上 XLA 使得跳过这些 FLOPs 相当困难。同样这随着批次大小和预填充序列长度增加而变糟。
4. **我们被迫在预填充和生成之间共享分片。** 预填充和生成都在同一个切片上，这意味着我们对两者使用相同的拓扑和分片（除非你保留两份权重），这通常对性能没有帮助，例如生成需要更多的模型分片。

因此，这种方法仅推荐用于边缘应用（通常只关心为单个用户服务，使用每字节 FLOPs 较少的硬件）和 Transformer 代码库生命周期早期的快速迭代（由于其简单性）。

一个稍微好一点的方法是在批次大小 1 下执行预填充（计算受限但有合理延迟），但在生成期间将多个请求批处理在一起：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免批处理预填充浪费的 TTFT，同时保持生成吞吐量高。我们称之为**交错**配置，因为我们"交错"预填充和生成步骤。这对于评估等批量生成应用非常强大，其中吞吐量是主要目标。调度器可以配置为在任何生成槽打开时立即优先预填充，确保即使是非常大的生成批次大小也能保持高利用率。我们还可以避免将预填充填充到最大长度，因为它没有与另一个请求批处理。

主要缺点是当服务器正在执行预填充时，所有其他请求的生成暂停，因为所有计算资源都将被预填充消耗。用户 A 正在解码的响应将被用户 B 正在进行的预填充阻塞。这意味着即使 TTFT 改善了，token 生成平均会是抖动和缓慢的，这对许多应用来说不是好的用户体验——其他用户的预填充在请求的整体延迟的关键路径上。

为了解决这个问题，我们将解码和预填充分离。虽然 Transformer 推理可以在一个服务器上完成，但从延迟的角度来看，在两组 TPU/GPU 上执行这两个不同的任务通常更好。预填充服务器生成 KV 缓存，通过网络发送到生成服务器，后者将多个缓存批处理在一起并为每个生成 token。我们称之为**"解聚"**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了几个优势：

1. **规模化的低延迟**：用户的请求永远不会阻塞在另一个用户的请求上，除非预填充容量不足。请求应该立即被预填充，然后发送到生成服务器，然后立即插入生成缓冲区。如果我们预期许多并发请求进来，我们可以独立于生成服务器数量扩展预填充服务器数量，这样用户就不会在预填充队列中等待很长时间。

2. **专业化：** 预填充和生成的延迟最优参数分片策略/硬件拓扑通常非常不同（例如，更多模型并行对生成有用但对预填充没用）。限制两个操作使用相同的分片会损害两者的性能，而拥有两组权重使用内存。另外，通过将预填充移到自己的服务器，它不需要持有任何 KV 缓存，除了它当前正在处理的那个。这意味着我们有更多空闲内存用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是 KV 缓存现在需要通过网络传输。这通常是可以接受的，但再次提供了减少 KV 缓存大小的动机。

<p markdown=1 class="takeaway">**要点：** 对于延迟敏感的高吞吐量服务，我们通常必须将预填充和生成分离到单独的服务器，预填充在批次 1 下操作，生成将许多并发请求批处理在一起。</p>

### 连续批处理

上面的问题 (2) 激发了**连续批处理**的概念。我们优化并编译：

* 一些具有可变上下文长度的预填充函数，并将其插入某个 KV 缓冲区，某个最大批次大小和上下文长度/页数。
* 一个生成函数，接收 KV 缓存，并为所有当前活跃的请求执行生成步骤。

然后我们将这些函数与一个调度器结合，该调度器排队传入的请求，根据可用的生成槽调用预填充和生成，处理历史缓存（见下一节）并流式输出 token。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

由于预填充是昂贵的和计算受限的（给我们更少的余量），减少其成本的最佳方法之一是少做它。因为 LLM 是自回归的，查询 ["I", "like", "dogs"] 和 ["I", "like", "cats"] 产生的 KV 缓存在前两个 token 上是相同的。这意味着，原则上，如果我们先计算 "I like dogs" 缓存，然后计算 "I like cats" 缓存，我们只需要做 1/3 的计算。我们可以通过重用缓存来节省大部分工作。这在几个特定情况下特别强大：

1. **聊天机器人**：大多数聊天机器人对话涉及来回对话，严格追加到自身。这意味着如果我们可以保存每个对话轮次的 KV 缓存，我们可以跳过除最新 token 以外的所有计算。
2. **Few-shot 提示**：如果我们有任何类型的 few-shot 提示，这可以免费保存和重用。系统指令通常也有这种形式。

这之所以难做的唯一原因是内存限制。正如我们所见，KV 缓存很大（通常是几 GB），为了缓存有用，我们需要保留它们直到后续查询到达。通常，预填充服务器上任何未使用的 HBM 都可以用于本地缓存系统。此外，加速器通常在其 CPU 主机上有大量内存（例如，8xTPU v5e 服务器有 128GiB 的 HBM，但大约 450GiB 的主机 DRAM）。这个内存比 HBM 慢得多——通常太慢以至于无法做生成步骤——但对于缓存读取来说足够快。在实践中：

* 因为 KV 缓存是本地的，属于处理初始请求的 TPU 集合，我们需要某种形式的亲和性路由来确保后续查询到达同一个副本。这可能导致负载均衡问题。
* 更小的 KV 缓存是有帮助的（再次）——它使我们能够在相同的空间内保存更多 KV 缓存，并减少读取时间。
* KV 缓存及其查找可以很自然地存储在树或 trie 中。驱逐可以在 LRU 基础上进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b>实现为 LRU trie 的 KV 前缀缓存。我们可以通过共享前缀来避免重复 KV 内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。" %}

### 看一个实现：JetStream

Google 开源了一个实现这种逻辑的库，叫做 [JetStream](https://github.com/google/JetStream)。服务器有一组"预填充引擎"和"生成引擎"，通常在不同的 TPU 切片上，由单个控制器编排。预填充发生在"[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"中，而生成发生在"[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"中。我们还有一个"[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"来编排从预填充切片到生成切片的 KV 缓存复制。

Engine 接口（[在这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)实现）是任何 LLM 必须提供的通用接口。关键方法是：

* **prefill：** 接收一组输入 token 并生成一个 KV 缓存。
* **insert：** 接收一个 KV 缓存并将其插入生成正在从中生成的批处理 KV 缓存中。
* **generate：** 接收一组批处理 KV 缓存并为每个批次条目生成一个 token，为每个 token 将单个 token 的 KV 缓存追加到解码状态。

我们还有一个 JetStream 的 PyTorch 版本，可在[这里](https://github.com/google/jetstream-pytorch)获得。

## 练习题

我将为这一节发明一个基于 LLaMA-2 13B 的新模型。以下是详细信息：

| 超参数             | 值     |
| :----------------- | :----- |
| L (num_layers)     | 64     |
| D (d_model)        | 4,096  |
| F (ffw_dimension)  | 16,384 |
| N (num_heads)      | 32     |
| K (num_kv_heads)   | 8      |
| H (qkv_dim)        | 256    |
| V (num_embeddings) | 32,128 |

**问题 1：** 上述模型有多少参数？int8 下每个 token 的 KV 缓存有多大？*你可以假设我们共享输入和输出投影矩阵。*

{% details 点击这里查看答案。 %}

**参数数量：**

* MLP 参数数量：$L * D * F * 3$
* 注意力参数数量：$L * 2 * D * H * (N + K)$
* 词汇表参数：$D * V$（因为我们共享这些矩阵）

我们的总参数数量因此是 $L * D * (3F + 2H * (N + K)) + D * V$。代入上面的数字，我们有 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，这个模型有大约 184亿参数。

KV 缓存在 int8 中是每 token $2 * L * K * H$，即 `2 * 64 * 8 * 256 = 262kB` 每 token。

{% enddetails %}

**问题 2：** 假设我们想在 TPU v5e 4x4 切片上服务这个模型，并且可以完全分片我们的 KV 缓存到这个拓扑上。我们能容纳的最大批次大小是多少，假设我们对一切使用 int8 并希望支持 128k 序列？如果我们将 KV 头数降到 1 呢？

{% details 点击这里查看答案。 %}

我们的 KV 缓存在 int8 中每 token 大小为 $2 \cdot L \cdot K \cdot H$，或 `2 * 64 * 8 * 256 = 262kB`。对于 128k 序列，这意味着每批次条目 `262e3 * 128e3 = 33.5GB`。由于每个 TPU 有 16GB 的 HBM，包括我们的参数，我们能容纳的最大批次大小是 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果我们有 $K=1$，我们会有这个的 8 倍，即大约 56。

{% enddetails %}

**问题 3：** 假设参数在 TPU v5e 4x4 切片上完全分片，从 HBM 加载所有参数到 MXU 需要多长时间？假设 int8 参数。*这是每步延迟的一个好的下限。*

{% details 点击这里查看答案。 %}

我们总共有 184亿参数，int8 中是 18.4e9 字节。我们每芯片有 8.1e11 HBM 带宽，所以假设我们可以完全使用 HBM 带宽，大约需要 `18e9 / (8.1e11 * 16) = 1.3ms`。

{% enddetails %}

**问题 4：** 假设我们想在 TPU v5e 4x4 切片上使用 int8 FLOPs 和参数/激活来服务这个模型。我们如何为预填充和解码分片？*提示：也许先回答这些问题：*

1. 4x4 上的 ICI 是什么样的？
2. 张量并行的 roofline 界限是什么？
3. 我们如何分片 KV 缓存？

对于这种分片，生成的大致每步延迟是多少？

**问题 5：** 假设上述模型实际上是一个 MoE。MoE 模型实际上是一个具有 E 个 FFW 块副本的稠密模型。每个 token 通过 k 个 FFW 块，这 `k` 个被平均以产生输出。让我们使用上述设置的 `E=16` 和 `k=2`。

1. 它有多少总参数和激活参数？*激活意味着任何给定 token 使用的。*
2. 在 TPU v5e 上需要什么批次大小才能变成 FLOPs 受限？
3. 每个 token 的 KV 缓存有多大？
4. T 个 token 的前向传播涉及多少 FLOPs？

{% details 点击这里查看答案。 %}

(1) 作为 MoE，每个 MLP 块现在有 $3 * E * D * F$ 个参数，比稠密变体增加了 $E$。因此它现在有 $L * D * (3EF + 2H * (N + K)) + D * V$ 或 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 总参数，增加了大约 12 倍。对于激活参数，我们有 $k$ 而不是 $E$ 个激活参数，总共 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比稠密变体增加不到 2 倍。

(2) 因为我们有 $E$ 倍更多的参数却只有 $k$ 倍更多的 FLOPs，我们的 HBM roofline 增加了 $E/k$ 倍。这意味着在 TPU v5e 上我们需要大约 `240 * (16 / 2) = 1920` 个 token。

(3) KV 缓存大小保持不变，因为 MoE 特性不会改变注意力机制的任何东西。

(4) 这仍然是 $2ND$，其中 $D$ 是激活参数数量。因此这是 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题 6：** 对于 MoE，我们可以做"专家分片"，在我们网格的一个轴上拆分我们的专家。在我们的标准符号中，我们的第一个 FFW 权重形状为 `[E, D, F]`，我们将其分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 仅在训练期间用作我们的 FSDP 维度。假设我们想在 TPU v5e 上进行推理：

1. 上述模型在 Y=8、Z=16 的 TPU v5e 8x16 切片上的 HBM 权重加载时间是多少？每 TPU 有多少空闲 HBM 可用？
2. 我们能将模型放在的最小切片是什么？

**问题 7 [2D 模型分片]：** 这里我们将解决 [ESTI 论文](https://arxiv.org/pdf/2211.05102)所称的 2D 权重静止分片的数学。我们在附录 B 中简要描述了这一点，但先尝试做这个问题，看看你是否能解决数学问题。2D 权重静止分片的基本思想是沿 $D$ 和 $F$ 轴分片我们的权重，使每个块大致是方形的。这减少了通信负载并允许我们稍微扩展得更远。

这是 2D 权重静止的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

你的目标是算出这个算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它何时会优于传统的 3D 模型分片？

{% details 点击这里查看答案！ %}

让我们算出 $T_\text{math}$ 和 $T_\text{comms}$。我们所有的 FLOPs 都是完全分片的，所以如前所述我们有 $T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在是

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

其中我们注意到 AllReduce 的成本是两倍，我们按每个操作执行的轴数来缩放通信。假设我们可以自由选择拓扑并假设 $F=4D$（如 LLaMA-2），我们声称（通过一些基本微积分）$X$、$Y$ 和 $Z$ 的最优值是 $X = \sqrt{N / 8}$，$YZ = \sqrt{8N}$，所以总通信是

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，从上面复制，正常的 1D 模型并行会有 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$，所以什么时候新的通信更小？我们有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$

对于一般的 $F$，我们声称这个条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

所以这告诉我们如果我们有超过 81 个芯片，我们最好使用这种新方案。现在这是一个有点奇怪的结果，因为我们历史上发现自己在大约 ~20 路张量并行时 ICI 受限。但这里，即使我们是通信受限的，我们的总通信随着总芯片数继续减少！这告诉我们，我们可以继续增加芯片、增加批次大小、做更多参数扩展，并看到减少的延迟。

{% enddetails %}

<h3 markdown=1 class="next-section">第 7 部分到此结束！第 8 部分将看看我们如何在 TPU 上服务 LLaMA 3，请点击[这里](../applied-inference)。</h3>

## 附录

### 附录 A：批次大小 > 240 规则有多真实？

我们上面提供的简单规则——我们的批次大小必须大于 240 个 token 才能是计算受限的——大致是正确的，但忽略了 TPU 在其他操作没有使用所有可用 HBM 时预取权重的能力，比如在做设备间通信时。

这是一个经验图，显示了一个小型 Transformer 的层时间（微秒），d<sub>model</sub> 8192，d<sub>ff</sub> 32768，每层只有 2 个矩阵乘法。这来自[这个 Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到步骤时间在大约批次 240 之前增加得非常缓慢，然后线性增加。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这是实际吞吐量（tokens / us）。这相当清楚地表明了论点。由于我们的层在这里大约有 600M 参数分片 4 路，我们预期最小延迟大约是 365us。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型中，我们确实看到吞吐量增加直到大约每数据并行分片 BS240。

### 附录 B：2D 权重静止分片

随着拓扑增长，如果我们可以访问更高维度的网格（如 TPU 的），可以通过"**2D 权重分片**"进一步细化。通过引入第二个分片轴。我们称之为"**2D 权重静止**"，在 [Efficiently Scaling Transformer Inference 论文](https://arxiv.org/abs/2211.05102)中有更详细的描述。

因为我们在 Megatron 中只分片隐藏的 $$F$$ 维度，一旦芯片数量增长到用 1D 分片时它可能变得比 $$E$$（$$d_\text{model}$$ 维度）显著更小。这意味着在较大批次大小时，在 MLP 的第一层应用后对隐藏维度执行一部分集合操作可能更经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

这个图显示：

1. 1D 权重静止分片，又名纯 Megatron 分片，激活在 AllGather 后完全复制，权重在隐藏 F 维度上完全分片。
2. 2D 权重静止分片，权重在隐藏 F 和收缩 E 维度上分片，激活在 E 维度上分片。我们在第一层之前在 (yz) 轴上执行 AllGather，然后在 (x) 轴上 ReduceScatter。

对于注意力层，Megatron 风格分片对于较少数量的芯片也相对简单。然而，Megatron 在 $$n_\text{heads}$$ 维度上发生，这对可能的分片量设置了限制。用（分片隐藏维度换成分片 $$n_\text{heads}$$ 维度）修改 2D 分片，我们获得了进一步扩展的能力。

### 附录 C：延迟受限通信

回顾一下，在[第3章](../sharding)中，我们推导了在具有全双工带宽 WICI 和延迟 Tmin 的 1D 环链路上 X 个芯片上对每个 TPU 上大小为 B 的张量执行 AllGather 所需的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于大 B，挂钟时间保持相对恒定，因为当你向系统添加更多芯片时，你同时扩展执行操作所需的数据移动量和可用的总带宽。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于延迟优化推理期间移动的数据量相对较少，激活上的集合操作通常受延迟项限制（特别是对于小批次大小）。通过计算完成所需的跳数，可以很容易地可视化延迟。

在 TPU 上，如果通信的张量大小相关部分小于每跳 1 微秒（跳是两个相邻设备之间的通信），我们可能会被实际调度集合操作的固定开销瓶颈。使用 `4.5e10` 单向 ICI 带宽，ICI 通信在以下情况变成延迟受限：$$(\text{字节} / n_\text{分片}) / 4.5e10 < 1e-6$$。对于 8 路 Megatron 分片，这是当 `buffer_size < 360kB` 时。**这在推理期间实际上并不那么小：** 使用 `BS=16` 和 `D=8192`（int8），我们的激活将使用 `16*8192=131kB`，所以我们已经是延迟受限的。

<p markdown=1 class="takeaway">**要点：** 当 $$\text{总字节数} < W_{ICI} \times 1e-6$$ 时，我们的通信变成延迟受限的。例如，使用在 $$Y$$ 上的模型并行，当 $$Y > BD / 45,000$$ 时我们在 int8 中变成受限。</p>

这里可以与计算 roofline 做一个平行——我们正在承担一些小操作的固定成本（通信的延迟，矩阵乘法的内存带宽）。

### 附录 D：推测采样

当我们*真的*关心端到端延迟时，有一个额外的技巧我们可以使用，叫做推测采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。回顾一下，我们通常从大型 Transformer 逐个生成 token：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用推测采样，我们使用一个更小、更便宜的模型来生成 token，然后用大模型检查结果。这用*贪婪解码*最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个更小、更便宜的模型贪婪采样。理想情况下，我们使用一个训练来匹配较大模型的模型，例如通过蒸馏，但它可以简单到只是使用 n-gram 或 token 匹配一个小语料库的文本。
2. 在我们生成了 K 个 token 后，我们使用大模型为我们到目前为止生成的所有 token 计算下一个 token 的 logits。
3. 因为我们是贪婪解码的，我们可以简单地检查较小模型生成的 token 是否在所有可能的 token 中具有最高概率。如果其中一个 token 是错误的，我们取最长的正确前缀并将第一个错误的 token 替换为正确的 token，然后回到 (1)。如果所有 token 都是正确的，我们可以在回到 (1) 之前使用最后一个正确的 logit 来采样一个额外的 token。

**为什么这是延迟上的赢？** 这个方案仍然要求我们对每个 token 做相当于通过大模型一次前向传播的 FLOPs，但因为我们可以将一堆 token 批处理在一起，我们可以在一次前向传播中做所有这些 FLOPs，并利用我们*不是*计算受限的事实来免费评分更多 token。

每个接受的 token 平均在 FLOPs 上变得更昂贵（因为一些会被拒绝，我们必须调用一个草稿模型），但我们从硬件中挤出了更多 FLOPs，而小模型是便宜的，所以我们总体上赢了。我们还在多个步骤中共享 KV 缓存加载，所以**对于长上下文，推测解码也可以是吞吐量上的赢。** 因为一切都被大模型检查过，我们根本不改变采样分布（尽管对于非贪婪，确切的轨迹会不同）。

传统上，推测解码依赖于存在一个与目标模型具有相似采样分布的较小模型，例如 LLaMA-2 2B 用于 LLaMA-2 70B，这通常不存在。即使这是可用的，如果接受率低，较小的草稿模型仍然可能太昂贵。相反，在主模型内嵌入一个草稿模型可能会有帮助，例如通过向基础模型的后面某一层添加一个专用的草稿头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。因为这个头与主模型共享大部分参数，它运行更快且更接近匹配采样分布。

对于正常的自回归采样，token/s 与步骤时间相同。我们仍然受制于这里算术强度部分的理论最小步骤时间（事实上，推测采样步骤时间通常比正常自回归采样慢很多，但因为我们平均每步骤得到超过 1 个 token，我们可以获得更好的 tokens/s）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>这个图显示了 Chinchilla（DeepMind 的一个 70B 模型）使用 4B 参数草稿模型（小模型）的每步延迟和推测成功率。对于 XSum（一个自然语言数据集），理想的推测量是大约提前 3-4 个 token，而 HumanEval（一个编码数据集）更可预测，从更激进的推测中看到收益。" %}

**这对非贪婪解码如何工作？** 这更复杂一些，但本质上归结为一个 Metropolis-Hastings 启发的算法，其中我们有从 logits 导出的 $$P_{\text{草稿模型}}(\text{选择的token})$$ 和 $$P_{\text{目标模型}}(\text{选择的token})$$，如果这些概率的比率小于某个阈值，则概率性地拒绝选择的 token。

这[两篇](https://arxiv.org/abs/2211.17192)[论文](https://arxiv.org/abs/2302.01318)同时推导了这个，并有很好的例子说明这在实践中是如何工作的。

<p markdown=1 class="takeaway">**要点：** 推测采样是另一个用吞吐量换取更好的每 token 延迟的强大杠杆。然而，在批次大小受限的情况下（例如硬件占用小或 KV 缓存大），它变成双赢。</p>
