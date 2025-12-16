---
layout: distill
title: "Roofline 模型详解"
# permalink: /main/
description: "当我们在硬件上运行算法时，会受到三个因素的限制：计算机进行数学运算的速度（每秒运算数）、用于移动数据的带宽（每秒字节数）以及用于存储数据的总内存（字节数）。这些"Roofline"约束让我们能够估算给定计算的时间上界和下界。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "第0部分：导论"

next_section_url: ../tpus
next_section_name: "第2部分：TPU"

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

  - name: 时间都去哪了？
  - subsections:
    - name: "可视化 Roofline"
    - name: "矩阵乘法"
    - name: "网络通信 Roofline"
  - name: 练习题

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

## 时间都去哪了？

让我们从一个极其简单的问题开始：*为什么一个算法需要 50ms 而不是 50s 或 5ms*？在模型内部究竟发生了什么需要花费大量时间，我们应该期望它需要多长时间？

**计算：** 深度学习模型本质上是一系列矩阵乘法，每个都由浮点乘法和加法"运算"（FLOPs）组成。我们的加速器速度决定了这些计算需要多长时间：

$$\begin{equation}
T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}}
\end{equation}$$

例如，NVIDIA H100 可以执行约 9.89e14 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，一种机器学习中常用的 16 位浮点格式。</d-footnote> FLOPs/s，而 TPU v6e 可以执行 9.1e14 FLOPs/s。<d-footnote>H100 和 B200 通常只能达到标称峰值 FLOPs 的 80-85%，而 TPU 在正常使用中可以接近 95%。</d-footnote> 这意味着在 H100 上执行 1e12 FLOPs 大约需要 `1e12 / 9.89e14 = 1.01ms`，在 TPU v6e 上需要 `1e12 / 9.1e14 = 1.1ms`。<d-footnote>请注意，这些芯片的定价不同，这个比较没有按成本标准化。</d-footnote>

**芯片内通信：** *在加速器内部*，张量需要在片上内存（HBM）和计算核心之间传输。你会看到这个链路的带宽被称为"HBM 带宽"<d-footnote>NVIDIA 也称之为"内存带宽"。</d-footnote>。在 H100 上，[这大约是 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上[大约是 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们将模型*分布到多个加速器*时，张量经常需要在它们之间传输。我们的硬件通常有几种选择（ICI、DCN 和 PCIe），每种都有不同的带宽。

无论是芯片内通信还是芯片间通信，我们都以 bytes/s 来衡量，并用以下公式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{通信字节数}}{\text{网络/内存带宽 Bytes/s}}
\end{equation}$$

通常（但不总是），单个芯片内的计算可以与芯片内和芯片间的通信重叠。这意味着**我们可以用计算和通信时间的最大值来给训练和推理时间设定下界**。我们也可以**用它们的和来设定上界**。在实践中，我们针对最大值进行优化，因为代数更简单，而且我们通常可以通过重叠通信和计算来接近这个界限。如果我们以最大值为目标优化，那么上下界最多相差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。然后，我们通过建模"重叠区域"和开销来提高精度，这可以通过分析你的特定模型和目标系统来获得。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设可以完美重叠通信和计算，当 $T_\text{math} > T_\text{comms}$ 时，我们可以看到硬件的充分利用。我们称之为"计算受限"。当 $T_\text{comms} > T_\text{math}$ 时，我们往往是"通信受限"的，至少有一部分加速器 FLOPs/s 被浪费在等待数据传输上。判断一个操作是计算受限还是通信受限的一种方法是查看它的"*算术强度*"或"*运算强度*"。

**定义：** 算法的算术强度由它执行的总 FLOPs 与它需要通信的字节数之比给出——无论是芯片内还是芯片间。

$$\begin{equation}
\text{算术强度} = \frac{\text{计算 FLOPs}}{\text{通信字节数}}
\end{equation}$$

算术强度衡量给定操作的"每字节 FLOPs"。在一阶近似下，当我们的算术强度高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 较大，我们通常能使用大部分可用的 FLOPs。当情况相反时，我们在通信上花费更多时间，浪费了 FLOPs。这个交叉点发生的地方是我们硬件的"峰值算术强度"，即峰值加速器 FLOPs/s 与加速器带宽的比值。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{计算 FLOPs}} {\text{加速器 FLOPs/s}} > \frac{\text{通信字节数}}{\text{带宽 Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{计算 FLOPs}}{\text{通信字节数}} > \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{强度}(\text{计算}) > \text{强度}(\text{加速器}) & \\
\end{align*}$$

量 $\text{强度}(\text{加速器})$ 是加速器达到峰值 FLOPs/s 时的算术强度。**对于 TPU v5e MXU，这大约是 240 FLOPs/字节**，因为 TPU 可以执行 `1.97e14` FLOPs/s 并从 HBM 加载 `8.2e11` 字节/s。<d-footnote>MXU 是 TPU 上的矩阵乘法单元。我们在这里特别指出这一点是因为 TPU 还有其他加速器如 VPU，负责逐元素运算，具有不同的峰值 FLOPs/s。</d-footnote> 这意味着如果一个算法的算术强度低于 240 FLOPs/字节，它将受到字节加载的限制，因此我们无法充分利用硬件。<d-footnote>这只有在算法从 HBM 加载其权重并在 MXU 中运行时才成立。正如我们将在下一节讨论的，我们有时可以将参数存储在 VMEM 中，它具有更高的带宽。许多算法也在 VPU 中运行，具有不同的性能特征。</d-footnote> 让我们看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）</span>：** 要计算两个 bfloat16 精度向量的点积，`x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存加载 $x$ 和 $y$，每个有 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，并将 $2$ 字节写回 HBM
$$\begin{equation}
\text{强度}(\text{点积}) = \frac{\text{总 FLOPs}}{\text{总字节数}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时。所以点积的算术强度是 $\frac{1}{2}$，或者换句话说，点积每加载一字节执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的算术强度，我们将是通信受限的。<d-footnote>上面的 240 这个数字在这里不是正确的比较，因为如你将在下一节看到的，点积是在 VPU 而不是 MXU 上执行的。TPU v5p VPU 可以做大约 7e12 FLOPs/秒，所以它的临界强度约为 3，这意味着我们在这里仍然有些通信受限。无论如何，我们的强度低且恒定这一事实意味着在大多数硬件上很难做到计算受限。</d-footnote>

### 可视化 Roofline

我们可以使用 **Roofline 图**来可视化内存和计算之间的权衡，它绘制了算法在我们硬件上的峰值可实现 FLOPs/s（吞吐量）（y 轴）与该算法的算术强度（x 轴）之间的关系。这是一个双对数图示例：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图示：</b> 一个示例 Roofline 图，显示了两个具有不同算术强度（Algo 1 和 Algo 2）的算法及其在不同带宽（BW1 和 BW2）下的相应理论峰值吞吐量。在红色区域，算法在两种带宽下都是带宽受限的，浪费了硬件峰值 FLOPs/s 的一部分。黄色区域仅在较低带宽（BW1）下是带宽受限的。绿色区域在所有带宽下都是计算受限的。在这里，我们使用加速器的峰值 FLOPs/s，增加带宽或改善强度都不会带来好处。" %}

如上所示，随着强度增加（从左到右移动），我们最初看到算法性能（以 FLOPs/s 为单位）线性增加，直到达到硬件的临界算术强度，对于 TPU v5e 来说是 240。任何强度较低的算法都将是带宽（BW）受限的，受峰值内存带宽限制（以红色显示）。任何在右侧的算法将充分利用我们的 FLOPs（以绿色显示）。这里，Algo 1 是通信受限的，只使用了总硬件 FLOPs/s 的一部分。Algo 2 是计算受限的。我们通常可以通过增加算法的算术强度或增加可用的内存带宽（从 BW1 移动到 BW2）来改善算法的性能。

### 矩阵乘法

让我们看看我们即将成为最爱的算法：矩阵乘法（又称 matmul）。我们写作 $X * Y \rightarrow Z$，其中 $X$ 的形状是 $\text{bf16}[B, D]$，$Y$ 的形状是 $\text{bf16}[D, F]$，$Z$ 的形状是 $\text{bf16}[B, F]$。要做矩阵乘法，我们需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并写回 $2BF$ 字节。<d-footnote>技术上我们执行 $BF \times (2D - 1)$ FLOPs，但这足够接近了。这来自 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>虽然矩阵乘法的输出技术上是 float32，但我们通常在复制回 HBM 之前向下转换为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{强度}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设我们的"批量大小" $B$ 相对于 $D$ 和 $F$ 较小，我们可以得到一个很好的简化

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approxeq \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{强度}(\text{matmul}) > \text{强度}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 矩阵乘法来说，这是一个合理的假设，因为我们通常有一个本地（每副本）批量大小 $B < 1024$ 个 token（*不是序列*），但 $D$ 和 $F > 8000$。因此，当我们的每副本<d-footnote>我们说每副本是因为，如果我们进行某种模型分片以增加矩阵乘法中使用的芯片数量，我们会将可用的计算和内存带宽按相同的比例扩展。因此，临界批量大小对于模型权重的每个独立副本都是正确的。</d-footnote>批量大小大于 240 个 token 时，我们通常会变成计算受限，这是一个非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 对于 bfloat16 矩阵乘法要在大多数 TPU 上达到计算受限状态，我们需要每副本 token 批量大小大于 240。<d-footnote>请注意，这_不是_通常意义上的批量大小，通常意义上指的是序列的批量大小。事实证明，大多数 Roofline 纯粹取决于 token 数量，无论它们属于同一序列还是不同序列。例如，如果你在 128 个 GPU 上有 512 个 4096 token 序列的批量大小，你的总批量大小是 `512 * 4096 = 2M` 个 token，本地批量大小是 16k 个 token。</d-footnote></p>

这有一些值得注意的注意事项，我们将在下面的问题中探讨，特别是关于量化（例如，如果我们量化我们的激活但仍进行全精度 FLOPs），但这是一个值得记住的好规则。对于 GPU，这个数字略高（接近 300），但同样的结论通常成立。当我们[将大矩阵乘法分解为小矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，分块大小也很重要。<d-footnote>当我们做大型矩阵乘法时，我们需要将其分解为适合 VMEM/SMEM/TMEM（更高带宽的片上内存）的更小分块。这导致我们多次加载块，所以不再完全正确地说我们只加载 $O(N^2)$ 字节。考虑一个 $(m, k) \cdot (k, n)$ 矩阵乘法，分块大小为 $bm$、$bk$、$bm$。令 $tm = m / bm$ 等。那么总 FLOPs 是 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节是 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + 2 \cdot bm \cdot bn)$。忽略最后一项，我们的强度是 $bm \cdot bn / (bm + bn)$，这与上面类似。</d-footnote> 我们将在[下一节](../tpus)讨论更底层的 GPU 和 TPU 细节。

### 网络通信 Roofline

到目前为止，我们讨论的所有 Roofline 都是内存带宽 Roofline，_都在单个芯片内_。这不应该被视为规则。事实上，本书中我们关心的大多数 Roofline 涉及芯片之间的通信：通常是涉及跨多个 TPU 分片的矩阵的矩阵乘法。

举一个有些刻意的例子，假设我们想要乘以两个大矩阵 $X\sim \text{bfloat16[B, D]}$ 和 $Y \sim \text{bfloat16[D, F]}$，它们在 2 个 TPU/GPU 之间均匀分割（沿 $D$ 维度）。要做这个乘法（正如我们将在[第 3 节](../sharding)中看到的），我们可以在每个 TPU 上乘以每个矩阵的一半（TPU 0 上 `A = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 上 `B = X[:, D // 2:] @ Y[D // 2:, :]`），然后将生成的"部分和"复制到另一个 TPU 并加在一起。假设我们可以在每个方向复制 `4.5e10` 字节，并在每个芯片上执行 `1.97e14` FLOPs/s。$T_\text{math}$ 和 $T_\text{comms}$ 是多少？

$T_\text{math}$ 显然是之前的一半，因为每个 TPU 做一半的工作，即<d-footnote>我们忽略了将两个部分和加在一起所需的 FLOPs（另外 BF 次加法），但这基本上可以忽略不计。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{加速器 FLOPs/s}} = \frac{BDF}{1.97e14}$$

那么 $T_\text{comms}$ 呢？这现在指的是芯片之间的通信时间！这只是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{网络带宽}} = \frac{2BF}{4.5e10}$$

因此，当 $$\text{强度}(\text{matmul (2芯片)}) > \text{强度}(\text{TPU 相对于芯片间网络})$$ 时，我们变成计算受限的（现在是相对于芯片间网络），或等价地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 或 $D > 8755$。注意，与之前不同，临界阈值现在取决于 $D$ 而不是 $B$！试着思考为什么会这样。这只是一个例子，但我们强调这种 Roofline 对于知道何时可以跨多个 TPU 并行化操作至关重要。

## 练习题

**问题 1 [int8 矩阵乘法]：** 假设我们想在 int8 精度（每个参数 1 字节）而不是 bfloat16 下做矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$。<d-footnote>这里和整篇文章中，我们将使用符号 $A \cdot_D B$ 来表示乘法是在 D 维度上进行收缩。这是对 einsum 符号的滥用。</d-footnote>

1. 需要从内存加载多少字节？需要写回内存多少字节？
2. 总共执行多少 OPs？
3. 算术强度是多少？
4. $T_\text{math}$ 和 $T_\text{comms}$ 的 Roofline 估计是多少？整个操作运行时间的合理上界和下界是多少？

假设我们的 HBM 带宽是 `8.1e11` 字节/s，int8 峰值 OPs/s 是 `3.94e14`（大约是 bfloat16 的 2 倍）。

{% details 点击这里查看答案。 %}

1. 因为我们以 int8 存储参数，每个参数 1 字节，所以我们从 HBM 加载 $$BD + DF$$ 字节，写回 $$BF$$ 字节。
2. 这与 bfloat16 相同，但理论上 int8 OPs/s 应该更快。所以这仍然是 $2BDF$ FLOPs。
3. 算术强度是 $$2BDF / (BD + DF + BF)$$。如果我们做与上面相同的假设，即 $$B \ll D$$ 且 $$B \ll F$$，我们得到算术强度为 $$2B$$，这意味着我们的规则变成 $B > \text{HBM int8 算术强度} / 2$。使用给定的数字，这个 int8 强度是 `3.94e14 / 8.1e11 = 486`，所以规则是 $B > 486 / 2 = 243$。注意这基本没变！
4. $$T_\text{math} = 2BDF / 3.94e14$$，$$T_\text{comms} = (BD + DF + BF) / 8.1e11$$，所以合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 矩阵乘法]：** 在实践中，我们经常对权重和激活使用不同的量化，所以我们可能以非常低的精度存储权重，但保持激活（和计算）在更高的精度。假设我们想用 int8 量化权重，但保持激活（和计算）在 bfloat16。在什么批量大小下我们变成计算受限？假设 `1.97e14` bfloat16 FLOPs/s。

*提示：这具体指 `bfloat16[B, D] * int8[D, F] -> bfloat16[B, F]`，其中 $B$ 是"批量大小"。*

{% details 点击这里查看答案。 %}

再次假设 B 较小，我们有 2BDF bfloat16 FLOPs，但只有 DF 个权重（而不是 bfloat16 中的 2DF）。这意味着当 $$2B > 240$$ 或 $$B > 120$$ 时，我们变成计算受限。这低得多，意味着如果我们可以做 int8 权重量化（这相当容易做到）但仍然做 bfloat16 FLOPs，我们在效率上获得了有意义的提升（尽管 int8 OPs 会更好）。

{% enddetails %}

**问题 3：** 采用问题 2 的设置，为 $F = D = 4096$ 和 $F = D = 1024$ 绘制峰值 FLOPs/s 与 $B$ 的 Roofline 图。*使用加载的确切字节数，而不是近似值。*

{% details 点击这里查看答案。 %}

这是所要求的图：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意两个模型最终都达到了峰值硬件 FLOPs/s，但较大的 D/F 更早达到。D=F=1024 几乎使临界批量大小翻倍。生成此图的代码在这里：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('批量大小')
plt.ylabel('TPU v5e 上的峰值 bfloat16 FLOPs/s')
plt.grid()
```

{% enddetails %}

**问题 4：** 如果我们想执行 $\text{int8[B, D]} *_D \text{int8[B, D, F]} \rightarrow \text{int8[B, F]}$，其中我们想象为每个批次元素有一个不同的矩阵。这个操作的算术强度是多少？

{% details 点击这里查看答案。 %}

让我们首先看看总 FLOPs 和通信。

1. 总 FLOPs：FLOPs 基本相同，因为我们做相同数量的 $$BD \times DF$$ 矩阵乘法（这在第 4 节有更多讨论）。所以这只是 $$2BDF$$。
2. 总通信：这里我们有多得多的通信：$$BD + BDF + BF$$。
3. 因此，我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 主导分母，这大约是 $$2$$。所以它不再取决于批量大小，而是基本上恒定的。这很糟糕，因为这意味着无论如何我们基本上总是通信受限。

{% enddetails %}

**问题 5 [GPU 的内存 Roofline]：** 使用 [NVIDIA 为 H100 SXM 提供的规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法将变成计算受限的批量大小。*请注意，Tensor Core FLOPs 数字是真实值的两倍，因为它们只有在结构化稀疏性下才能实现。*

{% details 点击这里查看答案。 %}

从规格表，我们看到报告的 bfloat16 FLOPs 值是 `1.979e15` FLOPs/s，并有一个星号标注"带稀疏性"。没有稀疏性的真实值是这个的一半，意味着接近 `1e15` FLOPs/s。内存带宽是 3.35TB/s，或 `3.35e12` 字节/秒。因此 $B_\text{crit}$ 是 `1e15 / 3.35e12 = 298`，与 TPU 相当相似。

{% enddetails %}

<h3 markdown=1 class="next-section">第 1 部分到此结束！关于真实 TPU 如何处理 FLOPs 和通信的第 2 部分，[请点击这里](../tpus)。</h3>