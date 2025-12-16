---
layout: distill
title: "分片矩阵及其乘法"
# permalink: /main/
description: "当我们训练大型机器学习模型时，必须将其参数或输入分割（或"分片"）到多个加速器上。由于大语言模型主要由矩阵乘法组成，理解这一点归结为理解如何在矩阵分布在多个设备上时进行乘法。我们基于 TPU 通信原语的成本，开发了一套简单的分片矩阵乘法理论。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "第2部分：TPU"

next_section_url: ../transformers
next_section_name: "第4部分：Transformer 数学"

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
  - name: "分区符号与集合操作"
  - subsections:
    - name: "统一的分片符号"
    - name: "如何在代码中描述这些？"
  - name: "分片数组的计算"
  - subsections:
    - name: "情况1：两个乘数都没有分片的收缩维度"
    - name: "情况2：一个乘数有分片的收缩维度"
    - name: "情况3：两个乘数都有分片的收缩维度"
    - name: "情况4：两个乘数都有沿同一轴分片的非收缩维度"
  - name: "深入理解 TPU 通信原语"
  - subsections:
    - name: "最后一个通信原语：AllToAll"
    - name: "更多关于 ReduceScatter"
  - name: "我们学到了什么？"
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

## 分区符号与集合操作

当我们在一万个 TPU 或 GPU 上训练一个大语言模型时，我们仍然在抽象上做着与在一个设备上训练相同的计算。区别在于**我们的数组无法放入单个 TPU/GPU 的 HBM 中**，所以我们必须分割它们。<d-footnote>值得注意的是，我们也可能为了速度而选择并行化。即使我们可以放在更少的芯片上，扩展到更多芯片只是给我们更多的 FLOPs/s。例如，在推理期间，我们有时可以放在较小的拓扑上，但选择扩展到较大的拓扑以减少延迟。同样，在训练期间，我们经常扩展到更多芯片以减少步骤时间。</d-footnote> 我们称之为"*分片*"或"*分区*"我们的数组。扩展的艺术在于弄清楚如何分片我们的模型以保持计算效率。

这是一个分片在 4 个 TPU 上的示例 2D 数组 **A**：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图示：</b> 一个形状为 <b>A</b>[I, J] 的示例数组在 4 个设备上分片。两个维度都均匀分片在 2 个设备上，分片为 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每个 TPU 持有总内存的 1/4。" %}

注意分片数组仍然具有与未分片数组相同的*全局*或*逻辑形状*，比如 `(4, 128)`，但它也有一个*设备本地形状*，比如 `(2, 64)`，这告诉我们每个 TPU 实际持有的字节大小（在上图中，每个 TPU 持有总数组的 ¼）。现在我们将其推广到任意数组。

### 统一的分片符号

我们使用*命名轴符号*的变体来描述张量如何以块的形式跨设备分片：我们假设存在一个 2D 或 3D 的设备网格，称为**设备网格（device mesh）**，其中每个轴都被赋予了**网格轴名称**，例如 **X**、**Y 和 Z**。然后，我们可以通过描述数组的每个命名维度如何跨物理网格轴分区来指定矩阵数据在设备网格上的布局方式。我们称这种分配为**分片（sharding）**。

**示例（上图）**：对于上图，我们有：
* **网格：** 上面的设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，告诉我们有 4 个 TPU 在 2x2 网格中，轴名称为 $X$ 和 $Y$。
* **分片：** $A[I_X, J_Y]$，告诉我们沿网格轴 $X$ 分片第一个轴 $I$，沿网格轴 $Y$ 分片第二个轴 $J$。这个分片告诉我们每个分片持有 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$ 的数组。

综合起来，我们知道数组的本地形状（单个设备持有的分片大小）是 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $$\lvert I\rvert$$ 是 A 的第一维大小，$$\lvert J\rvert$$ 是 A 的第二维大小。

<b markdown=1 style="color: #048affff;">小测验 [沿 1 轴的 2D 分片]：</b> 考虑一个数组 `fp32[1024, 4096]`，分片为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每个设备持有多少数据？在 H100 上从 HBM 加载这个数组需要多长时间（假设每芯片 `3.4e12` 内存带宽）？

{% details 点击这里查看答案。 %}

$A[I_{XY}, J]$ 将第一个维度 (I) 沿 X 和 Y 硬件轴分片。在这个例子中，本地形状是 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定的例子，全局形状是 `fp32[1024, 4096]`，所以本地形状是 `fp32[64, 4096]`。

由于每个 GPU 有 `4 * 64 * 4096 = 1MiB` 字节，这大约需要 `1e6 / 3.4e12 = 294ns`，但由于这太小，由于各种开销，可能显著更多。

{% enddetails %}

**可视化这些分片：** 让我们通过查看一个在 4 个设备上分割的 2D 数据数组来可视化这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们将矩阵的*完全复制*形式简单地写成 $A[I, J]$，没有分片分配。这意味着*每个*设备都包含整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以用下标网格轴来表示这些维度之一已跨网格轴分区。例如 $A[I_X, J]$ 意味着 **I** 逻辑轴已跨 **X** 网格维度分区，但 **J** 维度*未*分区，块在 **Y** 网格轴上保持*部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 意味着 **I** 逻辑轴已跨 **X** 网格轴分区，**J** 维度已跨 **Y** 网格轴分区。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们在下图中说明其他可能性：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里 $A[I_{XY}, J]$ 意味着我们将 **X** 和 **Y** 网格轴视为一个更大的扁平化维度，并将 **I** 命名轴跨所有设备分区。多个网格轴下标的顺序很重要，因为它指定了跨网格分区的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后，请注意我们*不能*让多个命名轴沿*相同*的网格维度分片。例如 $A[I_X, J_X]$ 是一个无意义的、禁止的分片。一旦一个网格维度被用来分片数组的一个维度，它在某种意义上就被"用尽"了。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设 **A** 是一个形状为 `int8[128, 2048]`、分片为 $A[I_{XY}, J]$、网格为 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（共 32 个设备）的数组。**A** 每个设备使用多少内存？**A** 跨所有设备使用的总内存是多少？

{% details 点击这里查看答案。 %}

**答案：** 我们的数组 **A** 沿 X 和 Y 分片，沿 Z 复制，所以每个设备它的形状是 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为 `8 * 2048 = 16,384` 字节。因为它沿 Z 复制，而在 Z 平面内它完全沿 X 和 Y 分片，所以有 2 份原始数组的完整副本（每个 Z 平面一份）。所以跨所有设备的总大小是：原始数组大小 × Z 副本 = 128 * 2048 * 2 = 512 KiB 总计。或者，我们可以验证：32 个设备 × 16,384 字节/设备 = 512 KiB 总计。

{% enddetails %}

### 如何在代码中描述这些？

到目前为止我们一直避免谈论代码，但现在是偷看一下的好机会。JAX 使用一种命名分片语法，非常接近我们上面描述的抽象语法。我们将在[第10章](../jax-stuff)中更多地讨论这个，但这里是一个快速预览。你可以在 Google Colab 中[这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)玩这个，并分析结果看看 JAX 如何处理不同的分片。这个代码片段做了 3 件事：

1. 创建一个 **jax.Mesh**，将我们的 8 个 TPU 映射到一个 4x2 网格中，轴名称 'X' 和 'Y' 分配给两个轴。
2. 创建矩阵 A 和 B，其中 A 沿其两个维度分片，B 沿输出维度分片。
3. 编译并执行一个返回分片数组的简单矩阵乘法。

```py
import jax
import jax.numpy as jnp

# 创建我们的网格！我们在一个 TPU v2-8 4x2 切片上运行，名称为 'X' 和 'Y'。
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 一个帮助定义分片的小工具函数。PartitionSpec 是我们的
# 分片（从轴到名称的映射）。
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# 我们将 A 和 B 都沿非收缩维度分片，A 也沿收缩维度分片。
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以对这些分片数组执行矩阵乘法！out_shardings 告诉我们想要
# 输出如何分片。JAX/XLA 为我们处理其余的分片。
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 的酷炫之处在于这些数组表现得就像是未分片的！`B.shape` 会告诉我们全局或逻辑形状 (2048, 8192)。我们必须实际查看 `B.addressable_shards` 来看它是如何本地分片的。我们可以对这些数组执行操作，JAX 会尝试弄清楚如何广播或重塑它们来执行操作。例如，在上面的例子中，**A** 的本地形状是 `[2, 1024]`，**B** 的本地形状是 `[2048, 4096]`。JAX/XLA 会根据需要自动添加跨这些数组的通信来执行最终乘法。

## 分片数组的计算

如果你有一个分布在多个设备上的数据数组，并希望对其执行数学运算，与分片数据和计算相关的开销是什么？

显然，这取决于涉及的计算。

* 对于*逐元素*操作，在分布式数组上操作**没有开销**。
* 当我们希望对驻留在多个设备上的元素执行操作时，事情变得复杂。幸运的是，对于大多数机器学习，几乎所有计算都以矩阵乘法的形式进行，它们相对简单易于分析。

本节的其余部分将讨论如何乘以分片矩阵。在一阶近似中，这涉及移动矩阵的块，以便你可以完全乘以或求和每个块。**每种分片将涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 可以在没有任何通信的情况下相乘，因为*收缩维度*（J，我们实际求和的那个）是未分片的。然而，如果我们希望输出未分片（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），我们需要将 $A$ 和 $B$ 或 $C$ 复制到每个设备（使用 *AllGather*）。这两种选择有不同的通信成本，所以我们需要计算这个成本并选择最低的。

{% details 你可以用"分块矩阵乘法"来理解这一点。 %}

要理解这一点，回顾"分块矩阵"的概念可能会有帮助，即矩阵的嵌套矩阵：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法有一个很好的性质，当矩阵乘数以块的形式书写时，乘积可以按照标准规则以块矩阵乘法的形式书写：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着实现分布式矩阵乘法归结为在网络上移动这些分片块，对块执行*本地*矩阵乘法，并求和它们的结果。**问题是添加什么通信，以及它有多贵。**

{% enddetails %}

方便的是，我们可以将所有可能的分片归结为大约 4 种需要考虑的情况，每种都有一个关于我们需要添加什么通信的规则
1. **[情况1](#情况1两个乘数都没有分片的收缩维度)：** 两个输入都没有沿收缩维度分片。_我们可以在没有任何通信的情况下乘以本地分片。_
2. **[情况2](#情况2一个乘数有分片的收缩维度)：** 一个输入有分片的收缩维度。_我们通常沿收缩维度"AllGather"分片输入。_
3. **[情况3](#情况3两个乘数都有分片的收缩维度)：** 两个输入都沿收缩维度分片。_我们可以乘以本地分片，然后"AllReduce"结果。_
4. **[情况4](#情况4两个乘数都有沿同一轴分片的非收缩维度)：** 两个输入都有沿同一轴分片的非收缩维度。我们必须先 AllGather 两个输入之一才能继续。

你可以把这些看作是简单需要遵循的规则，但理解这些规则为什么成立以及它们有多贵也很有价值。我们现在将详细讨论每一个。

### 情况1：两个乘数都没有分片的收缩维度

**引理：** 当乘以分片矩阵时，计算是有效的，输出遵循输入的分片，*除非*收缩维度被分片或两个矩阵沿同一轴分片。例如，这工作正常

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

没有任何通信，并产生一个跨 X 和 Y 硬件维度分片的张量。试着想想为什么会这样。基本上，计算*独立*于分片，因为每个批次条目有一些本地的被收缩轴的块，它可以乘以并归约。这些情况中的任何一个都工作正常并遵循这个规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

因为 **A** 和 **B** 都没有分片的收缩维度 **J**，我们可以简单地执行输入的本地分块矩阵乘法，结果*已经*根据所需的输出分片进行了分片。当两个乘数的非收缩维度沿同一轴分片时，这不再成立（详见[无效分片](#情况4两个乘数都有沿同一轴分片的非收缩维度)部分）。

### 情况2：一个乘数有分片的收缩维度

让我们考虑当一个输入 **A** 沿收缩 **J** 维度分片而 **B** 完全复制时该怎么做：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地乘以 **A** 和 **B** 的本地块，因为我们需要对 **A** 的完整收缩维度求和，而它跨 X 轴分割。通常，我们首先"**AllGather**" **A** 的分片，使每个设备都有完整副本，然后才与 **B** 相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样实际的乘法可以在每个设备上完全完成。

<p markdown=1 class="takeaway">**要点：** 当乘以其中一个矩阵沿收缩维度分片的矩阵时，我们通常先 AllGather 它，使收缩不再分片，然后做本地矩阵乘法。</p>

请注意，当 **B** 也没有沿 X 分片时，我们也可以做本地部分矩阵乘法，然后求和（或 *AllReduce*）分片的部分和，这在某些情况下可能更快。见下面的[问题 4](#练习题)。

**什么是 AllGather？** AllGather 是我们将讨论的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。AllGather *移除分片*沿一个轴，并将分散在设备上的分片重新组装到该轴上的*每个*设备上。使用上面的符号，AllGather 从一组轴中移除一个下标，例如

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们不必为给定维度移除所有下标，例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是一个 AllGather，只是在单个轴上。还要注意，我们可能还希望使用 AllGather 来移除*非收缩*维度的分片，例如在矩阵乘法中：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以最初 AllGather **A** 来移除输入分片，或者我们可以做分片矩阵乘法，然后 AllGather 结果 **C**。

**AllGather 实际上是如何执行的？** 要围绕单个 TPU 轴（一个环）执行 1 维 AllGather，我们基本上让每个 TPU 将其分片传递到环上，直到每个设备都有一份副本。<d-footnote>GPU AllGather 也可以这样工作，你从节点中的 GPU 创建一个环，并按该（任意）顺序传递块。</d-footnote> 这是一个动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图示：</b> 展示如何围绕一组 8 个 TPU 或 GPU 设备执行 AllGather 的动画。每个设备从数组的 1/8 开始，最终得到完整副本。" %}

我们可以在一个方向或两个方向（上面显示的是两个方向）进行 AllGather。如果我们做一个方向，每个 TPU 发送大小为 $\text{bytes} / N$ 的块，绕环 $N - 1$ 跳。如果我们做两个方向，我们有 $\lfoor \frac{N}{2} \rfloor$ 跳，大小为 $2 \cdot \text{bytes} / N$。

**这需要多长时间？** 让我们取双向 AllGather 并计算需要多长时间。设 $$V$$ 是数组的字节数，$X$ 是收缩维度上的分片数。那么从上图，每跳在每个方向发送 $V / \lvert X\rvert$ 字节，所以每跳需要

$$T_{hop} = \frac{2 \cdot V}{X \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽。<d-footnote>分子中的因子 2 来自我们使用双向带宽。我们在每个方向发送 $V / X$，或总共 $2V / X$。</d-footnote> 我们需要发送总共 $\lvert X\rvert / 2$ 跳才能到达每个 TPU<d-footnote>技术上是 $\lfloor X / 2 \rfloor$</d-footnote>，所以总归约需要

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

注意这**不依赖于 $X$！** 这有点惊人，因为这意味着即使我们的 TPU 只是本地连接的，连接的局部性也不重要。我们只受每个链路速度的瓶颈限制。

<p markdown=1 class="takeaway">**要点：** 在吞吐量受限的情况下执行 AllGather（或 ReduceScatter 或 AllReduce）时，实际通信时间只取决于数组的大小和可用带宽，而不是我们的数组分片在多少设备上！</p>

**关于 ICI 延迟的说明：** 每次通过 ICI 链路的跳跃都有一些固有的开销，与数据量无关。这通常约为 1us。这意味着当我们的数组 $$A$$ 非常小，每跳花费不到 1us 时，我们可以进入一个"延迟受限"的状态，此时计算_确实_依赖于 $X$。

{% details 完整细节，请点击这里。 %}

设 $$T_\text{min}$$ 是单跳的最小时间。那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行 $X / 2$ 跳。对于大的归约或收集，我们完全是带宽受限的。我们发送了太多数据，每跳的开销基本上可以忽略不计。但对于小数组（例如，从模型采样时），这不可忽略，ICI 带宽就不相关了。我们纯粹受延迟限制。另一种说法是，给定一个特定的 TPU，例如 TPU v5e，其单向 ICI 带宽为 `4.5e10`，发送任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都将是延迟受限的。

{% enddetails %}

这是一个在 TPU v5e 8x16 切片上 AllGather 带宽的经验测量。数组沿 16 轴分片，所以它有一个完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图示：</b> TPU v5e 在 AllGather 期间的经验带宽和估计链路带宽。橙色的 BW 是实际 AllGather 的每秒字节数，而蓝色曲线显示根据集合操作已知成本计算的经验单向链路带宽。" %}

请注意，我们不仅达到了约 95% 的声称峰值带宽（`4.5e10`），而且我们在约 10MB 时达到峰值，当 16 路分片时每设备约 500kB（*附注*：这比 GPU 好得多）。

**当我们沿多个轴 AllGather 时会发生什么？** 当我们沿多个轴收集时，我们有多个 ICI 维度来执行收集。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 在两个硬件网格轴上操作。这使可用带宽增加了 $N_\text{axes}$ 倍。

当考虑延迟时，我们得到一般规则：

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度。

<b markdown=1 style="color:rgb(144, 92, 255);">小测验 2 [AllGather 时间]：</b> 使用[第2部分](../tpus)中的数字，在具有 2D 网格 `{'X': 8, 'Y': 4}` 的 TPU v5e 上执行 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 需要多长时间，$$E = 2048$$，$$F = 8192$$（bfloat16）？$$E=256, F=256$$ 呢？

{% details 点击这里查看答案。 %}

**答案：** 让我们首先计算一些基本量：

1) TPU v5e 每个 2 轴有 4.5e10 字节/秒的单向 ICI 带宽。
2) 在 bfloat16 中对于 (a)，我们有 $A[E_Y, F]$，所以每个设备持有形状为 bfloat16[512, 8192] 的数组，有 512 * 8192 * 2 = 8.4MB。总数组大小为 2048 * 8192 * 2 = 34MB。

*对于第(1)部分*，我们可以使用上面的公式。由于我们在一个轴上执行 AllGather，我们有 $T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$。为了检查我们不是延迟受限的，我们知道在大小为 4 的轴上，我们最多有 3 跳，所以我们的延迟边界大约是 3us，所以我们不接近。然而，TPU v5e 只有当一个轴大小为 16 时才有环绕连接，所以这里*我们实际上不能做完全双向的 AllGather*。我们需要 3 跳让数据从边缘到达另一边，所以理论上我们有更像 $T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这是**](https://imgur.com/a/RkvpRGQ)**来自[这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)的实际分析**，显示 $680 \mu s$，这是合理的，因为我们可能没有获得 100% 的理论带宽！*对于第(2)部分*，每个分片大小为 `64 * 256 * 2 = 32kB`。32e3 / 4.5e10 = 0.7us`，所以我们是延迟受限的。由于我们有 3 跳，这大约需要 3 * 1us = 3us。[实际上，更接近 8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

<p markdown=1 class="takeaway">**注意：** 当我们有像 `{'X': 16, 'Y': 4}` 这样的 2D 网格时，每个轴不必对应特定的_硬件_轴。这意味着例如上面的可以描述一个 4x4x4 TPU v5p 立方体，其中 2 个轴在 $X$ 轴上。这将在后面我们描述跨多个轴的数据并行时发挥作用。</p>

### 情况3：两个乘数都有分片的收缩维度

第三个基本情况是当两个乘数都沿其收缩维度分片，沿相同的网格轴：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下，*本地*分片块矩阵乘法至少*可以*执行，因为它们将共享相同的收缩索引集。但每个乘积将只代表最终期望乘积的*部分和*，沿 **X** 维度的每个设备将留下该最终期望乘积的不同*部分和*。这太常见了，以至于我们扩展我们的符号来明确标记这个条件：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

符号 **{ U<sub>X</sub> }** 读作"**沿 X 网格轴未归约**"，指的是操作在某种意义上是"不完整的"这一状态，它只会在最终求和后完成。$\cdot_\text{LOCAL}$ 语法意味着我们执行本地求和但保留结果未归约。

这可以看作是关于矩阵乘法和外积的以下结果：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中 ⊗ 是外积。因此，如果轴 **X** 上的 TPU **i** 有 **A** 的第 **i** 列和 **B** 的第 **i** 行，我们可以做本地矩阵乘法得到 $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。这个矩阵在每个条目中有 **A • B** 在该条目处的和的第 **i** 项。我们仍然需要对 **P**（我们沿网格轴 **X** 分片）执行那个求和，以获得完整的 **A • B**。如果我们按块（即分片）来写 **A** 和 **B**，然后对结果的每个分片求和，这同样有效。

我们可以使用完整的 **AllReduce** 跨 **X** 轴来解决这个问题：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce 移除部分和，导致沿该轴的*每个*设备具有相同的完全求和值。AllReduce 是我们将在本节讨论的几个关键通信中的第二个，第一个是 AllGather，其他是 ReduceScatter 和 AllToAll。AllReduce 接受一个具有未归约（部分求和）轴的数组，并通过将这些分片传递到未归约轴周围并累积结果来执行求和。签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

这意味着它只是移除 $\\{U_Y\\}$ 后缀，但否则保持结果不变。

**AllReduce 有多贵？** AllReduce 执行方式的一个心理模型是，每个设备将其分片发送给邻居，并求和它收到的所有分片。显然，这比 AllGather 更贵，因为每个"分片"的形状与完整数组相同。通常，**AllReduce 的成本是 AllGather 的两倍。** 看到这一点的一种方法是注意 **AllReduce** 可以表示为另外两个原语的组合：**ReduceScatter** 和 **AllGather**。像 AllReduce 一样，ReduceScatter 解决数组上的部分和，但导致输出沿给定维度"分散"或分区。AllGather 收集所有这些片段并沿该物理轴"取消分区/取消分片/复制"逻辑轴。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**ReduceScatter 呢？** 就像 AllReduce 移除一个下标（上面的 $F_Y \to F$），ReduceScatter 对一个未归约/部分求和的数组求和，然后将不同的逻辑轴沿相同的网格轴分散（分片）。$[F]\\{U_Y\\} \to [F_Y]$。动画展示了这是如何完成的：注意它与 AllGather 非常相似，但不是保留每个分片，而是将它们相加。因此，排除执行归约所需的时间，其延迟大致相同。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每跳的通信时间只是每分片字节数 $V / Y$ 除以带宽 $W_\text{ici}$，就像 AllGather 一样，所以我们有

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中 $$W_\text{ici}$$ 是双向带宽，只要我们有一个完整的环来归约。

### 情况4：两个乘数都有沿同一轴分片的非收缩维度

每个网格维度在分片张量时最多只能出现一次。执行上述规则有时会导致违反此规则的情况，例如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的，因为给定的分片，比如沿维度 **X** 的第 **i** 个，将有 **C** 的 **(i, i)** 分片，即对角条目。那么所有分片中没有足够的信息来恢复除了对角条目以外的任何东西，所以我们不能允许这种分片。

解决这个问题的方法是 AllGather 某些维度。这里我们有两种选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

无论哪种情况，结果在其形状中只会提到 **X** 一次。我们选择哪个取决于后续操作需要什么分片。

## 深入理解 TPU 通信原语

前面的 4 种情况介绍了用于执行分片矩阵乘法的几个"核心通信原语"：

1. **AllGather：** 从分片中移除一个下标，收集分片。
2. **ReduceScatter：** 通过沿该轴求和分片来移除数组的"未归约"后缀，使数组沿第二个轴分片。
3. **AllReduce：** 移除"未归约"后缀，使数组沿该轴未分片。

还有一个核心通信原语需要提及，它出现在混合专家（MoE）模型和其他计算中：**AllToAll**。

### 最后一个通信原语：AllToAll

最后一个基本集合操作，在考虑分片矩阵乘法时不会自然出现，但在实践中经常出现，是 **AllToAll** 集合操作，或更准确地说是*分片转置*或重新分片操作的特殊情况。例如

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

当不同区域的分片计算没有兼容的布局方案时，通常需要 AllToAll 来重新排列分片布局。在考虑分片混合专家模型时，它们自然出现。*你可以把 AllToAll 看作是将下标从一个轴移动到另一个轴*。因为 AllToAll 不需要将每个分片的所有数据复制到整个环上，它实际上比 AllGather *便宜*（差 ¼ 因子）<d-footnote>对于偶数大小的双向环，每个设备将向右发送 $(N/2 + (N/2-1) + … + 1)$ 块，向左发送 $((N/2-1) + … + 1)$ 块 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个块（即分片的分片）的大小是 $\text{bytes} / N^2$，所以每设备成本是 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。这个结果在所有设备上扩展，因为总带宽随设备数量扩展。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如果我们推广到 ND AllToAll，在 AxBxC 网格上 $V$ 字节数组的总成本是

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中像往常一样 $W_\text{ici}$ 是双向 ICI 带宽。对于 1D 网格，这简化为 $V / (4 \cdot W_\text{ici})$，是 AllReduce 成本的 1/4。在 2D 中，成本实际上随最小轴的大小缩小。

*旁注：如果你想要这个事实的粗略推导，从 1D 环 $\mathbb{Z} / N\mathbb{Z}$ 开始。如果我们随机选择一个源和目标节点，它们平均相距 N / 4 跳，给我们成本 $(V \cdot N) / (4 * N)$。现在如果我们考虑 ND 环，每个轴基本上是独立的。每个节点有 $1 / N$ 字节，平均需要跳跃其数据 $\max(A, B, C, …) / 4$ 跳。*

### 更多关于 ReduceScatter

ReduceScatter 是一个比初看起来更基本的操作，因为它实际上是 AllGather 的导数，反之亦然。即如果在前向传播中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

那么我们 ReduceScatter 反向模式导数 **A'**（它们在每个分片上通常会不同）来推导分片的 **A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，前向传播中的 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ 意味着反向传播中的 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

{% details 关于 AllGather 和 ReduceScatter 如何互为导数的细节，请点击这里。 %}

这源于广播和归约作为线性算子是转置关系的事实，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也称为[克罗内克积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体地，如果我们有向量 $x \in \mathbb{R}^n$，任意数量的设备 $p \in \mathbb{N}$，并设 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们可以用以下方式定义广播和归约，这应该与你对它们的直觉理解相匹配：

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

让我们看一个例子，其中 $n = 1$，$p = 2$。如果 $x = (7)$，我们有 $$\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$。这符合我们的预期，将 $\mathbb{R}^n$ 中的向量广播到 $\mathbb{R}^{pn}$。现在令 $y = (8, 9)$，我们有 $$\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$。这再次符合我们的预期，将 $\mathbb{R}^{p n}$ 中的向量归约到 $\mathbb{R}^{n}$ 中的向量。由于 $(A \otimes B)^T = A^T \otimes B^T$ 对任意两个矩阵 $A$ 和 $B$ 成立，我们看到 $\text{reduce} = \text{broadcast}^T$。我们将 AllGather 和 ReduceScatter 恢复为以下外积：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

这里我们把 $\mathbb{R}^{p^2 n}$ 看作 $\mathbb{R}^{p \times p n}$，所以我们的 $p$ 个设备每个有一个 $\mathbb{R}^{p n}$ 向量。我们建议用小例子玩一玩，比如 $n = 2$，$p = 3$，看看这些算子作为矩阵长什么样。使用相同的转置性质，我们再次得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然 $\text{ReduceScatter}^T = \text{AllGather}$。这种转置将在反向传播期间出现，因为如果我们有 $y = Ax$ 对某个线性算子 $A$（如 AllGather 或 ReduceScatter），那么在反向传播期间我们将有损失对 $y$ 的导数 $\frac{\partial L}{\partial y}$，我们得到 $\frac{\partial L}{\partial x}$ 为 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$。这展示了 AllGather 的导数将是 ReduceScatter，反之亦然。

{% enddetails %}

将 AllReduce 转换为 AllGather 和 ReduceScatter 还有一个方便的性质，我们可以推迟最终的 AllGather 到稍后的某个时刻。很常见的是，我们宁愿不支付将完整矩阵乘积复制到设备上的成本。相反，我们想在这种组合两个具有分片收缩维度的乘数的情况下保持分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以执行 ReduceScatter 而不是 AllReduce，然后可选地在稍后某个时间执行 AllGather，即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

请注意，ReduceScatter *引入*一个分片维度，因此在这种情况下可以自然地选择将新分片引入 **I** 或 **K** 命名维度。我们通常需要选择*哪个*命名维度来引入新的分片（尽管选择通常由更大的建模上下文强制）。这就是为什么我们使用语法 **ReduceScatter<sub>X,K</sub>** 来指定要分片的轴。

## 我们学到了什么？

* 数组的分片由一个**网格（Mesh）**指定，它为我们 TPU 网格的物理硬件轴命名，以及一个**分片（Sharding）**，它将网格轴名称分配给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J] 描述一个抽象数组 **A**，其第一个维度沿两个网格轴 X 和 Y 分片。结合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 或缩写 Mesh({'X': 4, 'Y': 8})，这告诉我们我们的数组沿第一个维度 32 路分片。

* **分片数组的算术与未分片数组完全一样工作，除非你沿分片轴执行收缩**。在那种情况下，我们必须引入一些通信。我们考虑四种情况：

  1. *两个数组都没有沿收缩维度分片*：不需要通信。
  2. *一个数组沿收缩维度分片*（或收缩维度沿不同轴分片）：我们在执行操作前 AllGather 其中一个输入。
  3. *两个数组都沿收缩维度相同分片*：我们本地乘以分片，然后执行 AllReduce 或 ReduceScatter。
  4. *两个数组沿相同网格轴沿非收缩维度分片*：我们首先 AllGather 其中一个输入。

* TPU 使用大约 **4 个核心通信原语**：
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A, B_X]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$（技术上不是原语，因为它结合了 ReduceScatter + AllGather）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 每个操作的成本和延迟**不依赖于轴的大小（只要它们是带宽受限的）**，而只依赖于输入数组的大小和链路带宽。对于单向 AllGather/ReduceScatter：

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴大小} - 1}{\text{轴大小}}
\longrightarrow \frac{\text{数据量}}{\text{带宽 (双向)}}$$

* AllReduce 由 ReduceScatter 后跟 AllGather 组成，因此成本是上述的 2 倍。AllToAll 只需要将分片传递到环的一部分，因此成本是 AllGather 的 ¼。这是总结：

| 操作              | 描述                                                                                               | 语法                             | 运行时间                                         |
| :---------------- | :------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 沿一个轴收集分片数组的所有分片，移除一个下标。                                                     | $[A_X, B] \to [A, B]$            | bytes / (双向 ICI 带宽 * 轴数)                   |
| **ReduceScatter** | 沿一个轴求和部分求和的数组，并沿另一个轴分片它（添加一个下标）。                                   | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与 AllGather 相同                                |
| **AllReduce**     | 沿一个轴求和部分求和的数组。移除一个 { U<sub>x</sub> }。结合 AllGather 和 ReduceScatter。          | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 收集（复制）一个轴并沿相同轴分片不同的维度。                                                       | $[A, B_X] \to [A_X, B]$          | AllGather / 4（对于双向环）                      |

## 练习题

*这里有一些基于本节内容的练习题。我们暂时不会包括所有答案，但会在可能的时候写更多答案。*

**问题 1 [复制分片]**: 一个数组分片为 $A[I_X, J, K, \ldots]$（即只沿 $X$ 分片），网格为 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。$A$ 跨所有芯片占用的总字节数与数组一份副本大小的比率是多少？

{% details 点击这里查看答案。 %}

我们的数组只沿 X 分片，X 大小为 4，所以实际上每个分片大小为 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于我们的数组沿 Y 和 Z 复制，总大小是 $Y \cdot Z \cdot \text{sizeof}(A)$，所以总大小与单芯片大小的比率是 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题 2 [AllGather 延迟]**: 在 TPU v4p 4x4x4 切片上，网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，$\text{AllGather}_X([B_X, D_Y])$ 应该需要多长时间？如果 $B=1024$ 且 $D=4096$（bfloat16）？$$\text{AllGather}_{XY}([B_X, D_Y])$$ 呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 呢？

{% details 点击这里查看答案。 %}

我们在所有轴上都有环绕链路，因为我们有一个完整的 `4x4x4` 立方体，所以我们有 9e10 双向带宽可用。

1. 因为我们只在一个轴上收集而另一个是分片的，我们实际上收集 $2BD / Y$ 字节在 1 个轴上。*如果你只考虑 Y 轴上的单个分片，沿 X 的 AllGather 看起来像一个未分片的 AllGather，有 1 / Y 的字节。*由于我们的 TPU v4p ICI 带宽是 9e10 字节/秒双向，这将需要 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 我们有之前两倍的带宽，但我们 AllGather 完整数组，所以 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远离 4us（每跳 1us）的延迟边界，所以我们没问题。

3. AllReduce 的成本是 AllGather 的两倍。每个分片大小为 $2BD / (X * Y)$，所以成本约为 $4BD / (X * Y * W)$，或大约 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

{% enddetails %}

**问题 3 [延迟受限的 AllGather]**: 假设我们执行 $\text{AllGather}_X([B_X])$，但 $B$ 非常小（比如 128）。在 TPU v4p 4x4x4 切片上，网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，bfloat16 下这应该需要多长时间？*提示：你可能是延迟受限的。*

{% details 点击这里查看答案。 %}

我们的 bfloat16 数组总共只使用 256 字节，每设备只有 64 字节。由于我们在 TPU v4p 上有大小为 4 的轴，我们有环绕链路，所以我们可以双向发送数组。以 `4.5e10` 的单向带宽，每跳大约需要 `64 / 4.5e10 ~ 0`，所以我们肯定是延迟受限的。计算跳数，我们只需 2 跳就可以完成整个收集，所以大约 2us 是一个好的估计。

{% enddetails %}

**问题 4 [矩阵乘法策略]**: 要执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，在本节中我们告诉你执行 $\text{AllGather}_X(Y[D_X, F])$ 并乘以完全复制的矩阵（情况 2，*策略 1*）。相反，你可以像 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$（情况 4，*策略 2*）那样乘以本地分片，然后 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。每个执行多少 FLOPs 和通信？哪个更好，为什么？

{% details 点击这里查看答案。 %}

让我们从基线（*策略 1*）开始。如我们所示，AllGather 的成本是 $2DF / W_\text{ici}$。一旦我们有了完全复制的数组，总计算时间是 $2BDF / C$（其中 $C$ 是我们的加速器 FLOPs/s，因为每个 TPU 做相同的 FLOPs）。所以我们有

$$T_\text{total (策略 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略 2）对 $2BF$ 字节做 AllReduce，成本为 $4BF / W_\text{ici}$，但做 $1 / X$ 少的 FLOPs（因为计算是分片的）。这意味着我们做 $2\cdot B\cdot D\cdot F / X$ FLOPs，得到的 AllReduce 通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节（bfloat16）。因此，我们*策略 2*（没有 AllGather，只是后面的 AllReduce）的总时间大约是

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*哪个更大？*策略 (2) 在 $D / (X \cdot C) > 2 / W_\text{ici}$ 时是计算受限的，或当 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时。我们可能合理地期望 $D \approx 8k$，所以这意味着大约 $X < 2$，这不太可能——因此我们基本上总是用策略 2 是通信受限的。对于基线（策略 1），当 $$B < C / W_\text{ici} = 2550$$ 时我们是通信受限的，这经常但不总是正确的。

所以如果 $B < 2550$，我们在两种情况下都是通信受限的，我们有

$$T_\text{comms for 策略 2} < T_\text{comms for 策略 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当 $D > 2B$ 且 $2B < 5100$ 时为真。这经常为真，所以如果我们的批量小，策略 2 有时可能更好。当我们的批量大（$B > 2550$）时，我们有

$$T_\text{comms for 策略 2} < T_\text{math for 策略 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当 $2 / W_\text{ici} < D / C$ 时为真，或当 $D > 2 * 2550 = 5100$ 时，对于大模型通常为真。所以这个替代策略对大模型通常更好，除非 $D$ 很小。

*为什么我们不总是这样做？*嗯，在实践中我们有时可能这样做，但通常一个矩阵乘法输入沿其收缩维度分片而另一个输入没有沿该轴分片的情况相当罕见。例如，如果我们做 FSDP（在[第5章](../training)中解释），我们会沿数据维度分片我们的参数，但我们的激活_也会沿数据分片_。所以在这个意义上这不会经常出现。

{% enddetails %}

**问题 5 [最小延迟]**: 假设我想在 TPU v5p 4x4x4 上做矩阵乘法 $A[I, J] \cdot_J B[J, K] \to C[I, K]$，延迟最低。假设输入可以任意分片，但结果应该完全复制。我的输入应该如何分片？总 FLOPs 和通信时间是多少？

{% details 点击这里查看（部分）答案。 %}

我们不会在这里提供完整答案，但我们将首先描述四个最可能的选项：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + 最后 AG
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + 最后 AG
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + 最后 AR
4. $A[I, J] \cdot B[J, K]$（完全复制）

我们还可以考虑沿不同网格轴分片不同轴，但这不太可能改变最终成本。对于除 (4) 以外的所有情况，每 TPU 的总 FLOPs 是相同的，但每个的通信不同。然后我们只需计算每个的通信成本，看看哪个最低。简而言之，(1) 和 (2) 同样好。

{% enddetails %}

**问题 6：** 假设我们想在 TPU v5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。我们执行什么通信？在通信和计算上花费多少时间？

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的设置，我们结合数据、张量和零分片。
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理的标准，我们做纯张量并行（+数据）。

**问题 7：** 一个典型的 Transformer 块有两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，其中 $F \gg D$。假设我们有批量大小 B。那么完整块是 $In[B, D] \cdot W_\text{in}[D, F]. \cdot W_\text{out}[F, D]$。让我们取 $D=8192$，$F=32768$，$B=128$，假设一切都是 bfloat16。假设我们在 TPU v5e 2x2 切片上运行，但假设每个 TPU 只有 300MB 的空闲内存。In、$W_\text{in}$、$W_\text{out}$ 和 Out 应该如何分片以保持在内存限制之下同时最小化总时间？在通信和 FLOPs 上花费多少时间？*提示：最终输出不需要完全复制，但应该与输入相同分片，以便"层"可以重复。*

{% details 点击这里查看（部分）答案。 %}

首先让我们考虑内存。我们的两个大矩阵每个使用 `2 * 8192 * 32768 = 536MB`。我们的激活 `In` 大小为 `128 * 8192 = 1MB`（小到不用担心）。由于每个设备只有 300MB 的空闲内存，我们显然需要分片我们的矩阵乘法。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B, D]$（这通常称为 FSDP）
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这称为张量并行）

第一个相当糟糕，因为我们需要先 AllGather 我们的大权重或我们的激活。第二个在开始时需要 AllGather，在结束时需要 ReduceScatter（比 AllReduce 便宜）。我把剩下的数学留作练习。

{% enddetails %}

**问题 8 [挑战]**: 使用上面的短代码片段作为模板，分配一个分片数组并使用 pmap 或 shard_map 对 4 个主要通信原语（AllGather、AllReduce、ReduceScatter 和 AllToAll）中的每一个进行基准测试。你会想使用 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们需要多长时间？

**问题 9 [分片矩阵乘法的另一种策略？]**: [上面](#情况2一个乘数有分片的收缩维度)我们声称当矩阵乘法只有一个输入沿其收缩维度分片时，我们应该 AllGather 分片矩阵并本地执行结果收缩。你可能想到的另一种策略是执行分片矩阵乘法，然后 AllReduce 结果（好像两个输入都沿收缩维度分片），即 $A[I, J_X] *_J B[J, K] \to C[I, K]$ 通过

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 显式写出矩阵 $A[N, M]$ 和 $B[M, K]$ 的这个算法，使用索引来精确显示什么计算在什么设备上完成。假设 $A$ 分片为 $A[I, J_X]$ 在 ND 设备上，你希望输出在所有设备上复制。
2. 现在假设你可以接受最终结果不在每个设备上复制，而是分片（沿 N 或 K 维度）。上面的算法会如何变化？
3. 纯看上面策略的通信成本（在 (b) 中，而不是 (a)），这个通信成本与我们首先 AllGather A 然后做矩阵乘法的算法的通信成本相比如何？

{% details 点击这里查看答案。 %}


1. 首先计算外积，将结果存储在 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$。注意重复的索引不是被收缩的那个，因为我们在做外积。这里的和遍历我们正在使用的特定设备上存储的 i 值集合。所以，例如，如果我们有大小为 16 的收缩轴和 4 个设备，那么在设备 0 上，i 的范围是 {0, 1, 2, 3}；在设备 1 上，i 的范围是 {4, 5, 6, 7}；在设备 2 上，i 的范围是 {8, 9, 10, 11}；在设备 3 上，i 的范围是 {12, 13, 14, 15}。然后 AllReduce 存在于每个设备上的 $O[N, K]$ 的部分和，以形成完整的 $O[N, K]$。
2. 在步骤 2 中不做 AllReduce，我们可以用更便宜的 ReduceScatter，沿任一轴：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如上面正文中所述，做 AllGather 的成本（当我们是吞吐量受限时）与 ReduceScatter 相同；它只是由我们正在处理的完整矩阵的大小给出。所以在 gather-then-matmul 算法中，这随 $NM$ 扩展（因为我们 $\text{AllGather}$ $A$）；在 matmul-then-reduce-scatter 算法中，这随 NK 扩展（因为我们 reduce-scatter $O$）。所以两种算法的通信成本比是 `M/K`。

{% enddetails %}

**问题 10：AllToAll 的乐趣：** 在上表中，注意到执行 AllToAll 的时间比执行 AllGather 或 ReduceScatter 的时间低 4 倍（在我们是吞吐量受限的情况下）。在这个问题中，我们将看到这个因子 4 来自哪里，以及如果我们只有单向 ICI 链路而不是双向 ICI 链路，这个因子会如何变化。

1. 让我们从单向情况开始。想象我们有 *D* 个设备在环形拓扑中，如果我们在 N x N 矩阵 *A* 上做 AllGather 或 ReduceScatter，它分片为 $A[I_X, J]$（假设 $D$ 整除 $N$ 简化）。描述这两个集合操作涉及的通信，并计算在整个算法期间通过**单个** ICI 链路传输的标量（浮点数或整数）总数。
2. 现在让我们考虑 AllToAll，仍然在单向 ICI 情况下。这种情况下的算法与 all-gather 情况有何不同？计算在这个算法中通过单个 ICI 链路传输的标量数量。
3. 你应该发现 (a) 和 (b) 答案之间的比率是一个好数字。用简单的术语解释这个因子来自哪里。
4. 现在让我们加入双向通信。这如何影响 all-gather 情况所需的总时间？
5. 添加双向通信如何影响 AllToAll 情况所需的总时间？
6. 现在简单解释双向环中 AllGather 时间和 AllToAll 时间之间的比率。

{% details 点击这里查看答案。 %}

(1) **解答：** 过程很简单：在算法的每一步中，每个设备将发送矩阵的单分片"条带"（总共 $$\frac{N}{D} \times N$$ 个元素）给最近的邻居。这发生 $$D-1$$ 次，因为每个分片需要通信给除了它开始所在的那个之外的所有设备。所以总共，每个设备传输 $$\frac{N^2(D-1)}{D}$$ 个标量，即流经单个 ICI 链路。

**答案：** $$N^2 (1-\frac{1}{D})$$，或当 $$D >> 1$$ 时简单地 $$N^2$$。

(2) **解答：** 从通信的角度来看，AllToAll 和 AllGather 之间的关键区别是，在 AllToAll 中，存在于特定设备上的整个分片不需要通信给每个其他设备。想象存储在特定设备（称其为设备 0）上的分片是 $$[A, B, C, D]$$（这里 A,B,C,D 是矩阵，我们为说明想象一个有 4 个设备的环）。现在矩阵 $$A$$ 不需要通信到任何地方，矩阵 $$B$$ 需要最终到达设备 1；矩阵 $$C$$ 到达设备 2；矩阵 $$D$$ 到达设备 3。所以在算法的第一步，我们发送 $$B$$、$$C$$ 和 $$D$$ 到设备 1；在下一步，设备 1 发送 $$C$$ 和 $$D$$ 到设备 2；在最后一步，设备 2 只发送 $$D$$ 到设备 3。这种情况下传输的参数总数是 $$(\text{A/B/C/D 的大小}) * (3 + 2 + 1)$$。A/B/C/D 的大小（在一般情况下）是 $$\frac{N^2}{D^2}$$，在一般情况下 $$(3 + 2 + 1)$$ 项变成 $$((D-1) + (D-2) + … + 1)$$，或 $$\frac{(D)(D-1)}{2}$$。所以通过单个 ICI 链路传输的总字节数是 $$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$，或当 $$D >> 1$$ 时简单地 $$\frac{N^2}{2}$$。

(3) **解答：** 因子简单地是 $$\frac{1}{2}$$，即在单向环拓扑上 AllToAll 的成本是 all-gather/ReduceScatter 的一半。回顾上面的推导，这最终来自这样一个事实：在 all-gather 情况下，我们传输相同大小的块 $$(D-1)$$ 次，即我们在做求和 $$ \text{tiny block size} * (D + D + D + … + D)$$，而在 AllToAll 情况下，我们在做求和 $$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$。因此因子 2 本质上来自 $$1 + 2 + \ldots + n = n(n+1)/2$$ 这个事实。

(4) **解答：** 任何一个链路必须承载的标量总数现在减少了 2 倍，因为在双向环中，每个"分片条带"可以同时双向发送。

(5) **解答：** 在这种情况下，与单向情况相比，我们赢得了 4 倍。这通过考虑单个分片条带中每个大小为(N2/D2)块的命运最容易看出，比如起源于设备 0 的那个。不是像单向情况那样将一个块发送距离 D-1，另一个块距离 D - 2 等等一直到 1，我们现在将条带分成向右或向左移动的块，最大移动距离是 floor(D/2)。所以对应的和现在变成 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，或在大 $$D$$ 极限下 $$D^2/8$$。与单向情况下的 $$D^2/2$$ 相比，我们看到我们赢得了 4 倍。

(6) **解答：** 在单向环中，我们看到 AllToAll 时间已经是 all-gather 时间的两倍快；这来自我们不需要将我们的完整条带发送到每一个设备这一事实。然后，当我们添加双向性时，我们看到对于 AllToAll 是 4 倍优势，对于 all-gather 只有 2 倍优势。将这些比率放在一起，我们得到我们追求的因子 4。

{% enddetails %}

<h3 markdown=1 class="next-section">第 3 部分到此结束！关于 Transformer 数学的第 4 部分，请点击[这里](../transformers)！</h3>
