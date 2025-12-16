---
layout: distill
title: "Transformer 数学完全指南"
# permalink: /main/
description: "本章将快速回顾 Transformer 架构，特别是如何计算 FLOPs、字节数和其他关键指标。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "第3部分：分片"

next_section_url: ../training
next_section_name: "第5部分：训练"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "计算点积"
  - subsections:
    - name: "前向和反向 FLOPs"
  - name: "Transformer 计算量统计"
  - name: "全局 FLOPs 和参数计算"
  - name: "其他数学知识"
  - subsections:
    - name: "稀疏性与混合专家模型"
    - name: "梯度检查点"
    - name: "键值（KV）缓存"
  - name: "本章要点"
  - name: "练习题"
  - name: "附录"
  - subsections:
    - name: "附录 A：Flash Attention 工作原理"

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

## 计算点积

让我们从向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$ 开始，它们的形状如下：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{数组}  & \textrm{形状} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end {array}
$$

- 向量点积 $$x \cdot y$$ 需要 $$P$$ 次*加法*和*乘法*，总共 $$2P$$ 次浮点运算。
- 矩阵-向量乘法 $$Ax$$ 沿 $$A$$ 的行执行 $$N$$ 次点积，总共 $$2NP$$ 次 FLOPs。
- 矩阵-矩阵乘法 $$AB$$ 对 $$B$$ 的每一列（共 $$M$$ 列）执行一次矩阵-向量乘法，总共 $$2NPM$$ 次 FLOPs。
- 一般情况下，如果我们有两个高维数组 $$C$$ 和 $$D$$，其中某些维度是<span style="color:red">收缩维度</span>，某些是<span style="color:blue">批次维度</span>（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$），那么这个收缩操作的 FLOPs 成本是所有 $$C$$ 和 $$D$$ 维度乘积的两倍，其中批次和收缩维度只计算一次（例如 $$2\blue{GH}IJMN\red{KL}$$）。注意，只有当一个维度同时出现在两个乘数中时，它才是批次维度。（另外注意，如果没有收缩维度，即只是逐元素乘法，则不需要乘以 2。）

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{FLOPs} & \textrm{数据量} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end {array}
$$

请特别注意：对于矩阵-矩阵乘法，*计算量*以 $$O(N^3)$$ 立方增长，而数据传输只以 $$O(N^2)$$ 平方增长——这意味着随着矩阵乘法规模的增大，我们*更容易*达到计算饱和的极限。这是非常不寻常的特性，这在很大程度上解释了为什么我们使用以矩阵乘法为主的架构——它们非常适合扩展！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向 FLOPs

在训练过程中，我们并不特别关心给定矩阵乘法的结果本身；我们真正关心的是它的导数。这意味着我们在反向传播过程中需要执行显著更多的 FLOPs。

假设 **B** 是一个更大网络中的某个矩阵，**A** 是我们的输入激活值，且 **C = A B**，那么损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

这是一个外积运算，需要 $2NPM$ 次 FLOPs 来计算（因为它沿 $N$ 维度收缩）。同样，损失对 **A** 的导数是：

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

这同样是 $2NPM$ 次 FLOPs，因为 **dL/dC** 是一个大小为 $$[N, M]$$ 的（协）向量。虽然这个量不是参数的导数，但它用于计算网络前面层的导数（例如，就像 dL/dC 用于计算上面的 dL/dB 一样）。

把这些加起来，我们看到**在训练过程中，总共有 6NPM 次 FLOPs**，而推理过程中是 2NPM：前向传播 2NPM，反向传播 4NPM。由于 PM 是矩阵中的参数数量，这就是著名的 $$6 \times \text{参数数量} \times \text{token数量}$$ 近似公式的最简单形式，用于估算 Transformer 训练时的 FLOPs：每个 token 需要 $$6 \times \text{参数数量}$$ 次 FLOPs。我们将在下面展示更精确的推导。

## Transformer 计算量统计

Transformer 是未来。好吧，至少是现在。也许几年前，它只是众多架构之一。但今天，值得了解这个架构的几乎每一个细节。我们不会重新介绍这个架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)可能是有用的参考资料。

这是 Transformer 解码器架构的基本图示：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>这个图展示了标准 Transformer 的一层，从上到下流动。我们使用单字母约定来描述 Transformer 中数组的形状和布局，收缩维度用红色显示，批次维度用蓝色显示。在给定操作中，输入形状在左上角给出，参数形状在右上角给出，结果形状在下方，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。" %}

**注释 [门控 einsum]**：上图使用了"[门控 einsum](https://arxiv.org/abs/2002.05202)"<d-cite key="glu"></d-cite>，其中我们将上投影矩阵分成两个矩阵（上图中的 $W_\text{In1}$ 和 $W_\text{In2}$），它们的输出逐元素相乘作为一种"门控函数"。并非所有 LLM 都使用这种方式，所以有时你会看到单个 $W_\text{In}$ 矩阵，MLP 参数总数为 2DF 而不是 3DF。通常在这种情况下，D 和 F 会相应放大以保持与 3 矩阵情况相同的参数数量。话虽如此，某种形式的门控 einsum 被 LLAMA、DeepSeek 和许多其他模型使用。

**注释 2 [MHA 注意力]**：对于自注意力，T 和 S 相同，但对于交叉注意力，它们可能不同。对于普通的多头注意力（MHA），N 和 K 相同，而对于[多查询注意力](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite> K=1，对于[分组多查询注意力](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite> K 只需要能整除 N 即可。

## 全局 FLOPs 和参数计算

下面我们将计算每层的 FLOPs，以避免到处都要写 **L** 因子。

### MLP 层

Transformer 的 MLP 通常由 2 个输入矩阵乘法（逐元素组合）和 1 个输出矩阵乘法组成：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数量} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力层

对于具有不同 **Q** 和 **KV** 头数的通用分组查询注意力情况，假设 **Q**、**K**、**V** 投影具有相同的头维度 H，我们估计 **QKVO** 矩阵乘法的成本：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数量} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力操作更加微妙，实际上是一个在 $$B$$、$$K$$ 维度上批处理的 $$TH \cdot HS$$ 矩阵乘法，然后是 softmax，再然后是一个同样在 $$B$$、$$K$$ 维度上批处理的 $$TS \cdot SH$$ 矩阵乘法。我们用蓝色高亮批次维度：

$$
\begin{array}{cc}
\textrm{操作} & \textrm{训练 FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注释 [因果掩码]**：大多数最新的 Transformer 使用因果掩码而不是完全双向注意力。在这种情况下，点积操作的有效 FLOPs 减少了 1/2。要在实践中实现这种减少，我们需要使用注意力内核，而不是朴素的 einsum。

### 其他操作

Transformer 中还有几个其他操作。LayerNorm 相对便宜，可以在一阶成本估计中忽略。还有最终巨大的（虽然不是每层都有的）解嵌入矩阵乘法。

$$
\begin{array}{ccc}
\textsf{操作} & \textsf{训练 FLOPs} & \textsf{参数量} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOPs 的一般经验法则

如果我们忽略短上下文训练中点积注意力的成本，那么所有层的总 FLOPs 是：

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{token 数量} * \textrm{参数数量}
\end{align*}
$$

这导出了一个著名的经验法则，用于估计稠密 Transformer 的 FLOPs 数量（忽略注意力 FLOPs）。（解嵌入是另一个简单的矩阵乘法，有 $6BSDV$ 次 FLOPs 和 $DV$ 个参数，也遵循同样的经验法则。）

### 注意力成本随上下文长度的变化

如果我们确实考虑上面的点积注意力，并假设 $$F=4D$$、$$D=NH$$（这是典型设置）且 $$N=K$$：

$$\small{\frac{\textrm{注意力 FLOPs}}{\textrm{矩阵乘法 FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

所以要点是：**点积注意力 FLOPs 只有在 T>8D 时才在训练中占主导地位**。对于 D ~ 8k，这将是 ~64K 个 token。这是有道理的，因为这意味着随着 MLP 大小的增加，注意力 FLOPs 变得不那么关键。对于大型模型，注意力的二次成本实际上并不是长上下文训练的巨大障碍。然而，对于较小的模型，例如 Gemma-27B，D=4608，这意味着注意力在大约 32k 序列长度时变得占主导地位。Flash Attention 也有助于减轻长上下文的成本，我们在[附录 A](#附录-a-flash-attention-工作原理) 中简要讨论。

## 其他数学知识

### 稀疏性与混合专家模型

我们不得不简要讨论混合专家模型（MoE）<d-cite key="moe"></d-cite>，它用一组可以动态路由的独立 MLP 替换标准 Transformer 中的单个稠密 MLP 块。粗略来说，**MoE 就是一个正常的稠密模型，每层有 E 个 MLP 块**，而不是只有一个。每个 token 激活这些专家中的 $k$ 个，通常 $k \ll E$。比率 $E / k$ 称为稀疏度，通常在 8 到 64 之间（例如 [DeepSeek v3](https://arxiv.org/pdf/2412.19437) 有效地使用 $k=8$，$E=256$）。与稠密版本相比，这将参数数量增加了 $O(E)$ 倍，同时每个 token 激活的参数数量乘以 $k$。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图：</b>一个有 $n$ 个专家的 MoE 层示例。门控专家将每个 token 路由到其中 $k$ 个，这 $k$ 个 MLP 的输出被求和。我们的参数数量是每个专家大小的 $n$ 倍，但每个 token 只使用 $k$ 个。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与稠密模型相比，MoE 引入了新的通信开销，主要是两次 AllToAll（一次在 MoE 块之前，一次之后），用于将 token 路由到正确的专家并将它们带回其原始设备。<d-footnote>严格来说，只有当我们沿与专家相同的轴进行数据或序列分片时才会发生这种情况。</d-footnote>然而，正如我们在上一节中看到的，每次 AllToAll 的成本仅为沿单个轴的可比 AllGather 的 1/4（对于双向环）。

### 梯度检查点

反向传播作为一种算法，用内存换取计算。反向传递不需要 $$O(n_\text{layers}^2)$$ 次 FLOPs，而是**需要 $$O(n_\text{layers})$$ 的内存**，保存前向传递期间生成的所有中间激活值。虽然这比二次计算要好，但在内存方面代价非常高：一个具有 $$B * T=4M$$（每批次总共 4M 个 token）、L=64 和 D=8192 的模型，如果要避免所有不必要的反向传递计算，必须保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的激活值（bfloat16）。20 大约来自计算上面 Transformer 图中的每个中间节点，例如：

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算，我们需要保存前向传递中的 $$g(x)$$ 和 $$\exp(g(x))$$。为了避免保存这么多内存，我们可以选择只保存一部分中间激活值。这里有几个我们使用的策略：

* **块重计算（Block remat）**：只保存每层的输入。这是我们使用的最激进的方法，每层只保存 1 个检查点，意味着在上面的例子中我们只保存 4.2TB。这迫使我们在反向传递中重复几乎所有前向传递的 FLOPs，意味着我们将 FLOPs 从 $$6ND$$ 增加到大约 $$8ND$$。
* **只保存大矩阵乘法输出**：另一个简单的策略是只保存大矩阵乘法的输出。这让我们可以避免在反向传递中重新计算任何大矩阵乘法，但仍然需要重新计算其他激活函数和注意力的部分。这将每层的 20 减少到接近 7。

这绝不是全面的。使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（你可以在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)阅读更多）。

### 键值（KV）缓存

正如我们将在[第7章](../inference)中看到的，LLM 推理有两个关键部分：预填充和生成。

* **预填充** 处理一个长提示并将其注意力激活值保存在键值缓存（KV Cache）中供生成使用，具体来说是注意力块中的键值投影。
* **生成** 将多个这样的 KV 缓存批处理在一起，并从每个缓存中采样 token。

每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 代表键和值。这相当大！int8 格式的键值缓存总大小是 $2SLKH$。对于一个中等大小的模型，具有 8k 上下文长度、64 层和 $KH = NH = D = 8192$，这是 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。你可以看到为什么我们希望使用 $K \ll N$ 的 GMQA。

## 本章要点

* Transformer 的整体参数和 FLOPs 相当容易计算，这里进行总结，假设使用 MHA（批次大小 B，词汇表大小 V，序列长度 T，D=d<sub>model</sub>，F=d<sub>ff</sub>）：

| 组件          | 每层参数量                | 每层训练 FLOPs                    |
| :------------ | :------------------------ | :-------------------------------- |
| **MLP**       | 3DF                       | 18BTDF                            |
| **注意力**    | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH     |
| **其他**      | D                         | BTD                               |
| **词汇表**    | DV（总计，非每层）        | 12BTDV                            |

* MLP 块的参数数量在总参数数量中占主导地位，只要序列长度 $T < 8D$，MLP 块也在 FLOPs 预算中占主导地位。
* 训练期间的总 FLOPs 预算可以很好地用 $$6 \cdot \text{参数数量} \cdot \text{token数量}$$ 来近似（对于合理的上下文长度）。
* 在推理期间，我们的 KV 缓存大约是每个缓存 $$2 \cdot S \cdot L \cdot N \cdot H$$，尽管架构修改通常可以减少这个值。

## 练习题

**问题 1：** 一个模型具有 $D=4096$，$F=4 \cdot D$，$V=32,000$，$L=64$，它有多少参数？注意力参数占比是多少？每个 token 的 KV 缓存有多大？*你可以假设 $N\cdot H=D$ 并使用 int8 KV 的多头注意力。*

{% details 点击这里查看答案。 %}

1. 总参数大约是 $$L \cdot (3DF + 4DNH + D) + 2DV$$。对于给定的数字，这是 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，即 160亿参数。
2. 注意力参数与总参数的比例一般是 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。所以大约 1/4 的参数用于注意力。
3. 每个 token 的 KV 缓存是 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$（int8），即 `512kB / token`。

{% enddetails %}

**问题 2：** 在 `{'X': 4, 'Y': 8, 'Z': 4}` 上执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 需要多少总 FLOPs？每个 TPU 执行多少 FLOPs？

{% details 点击这里查看答案。 %}

操作的总"理论" FLOPs 是 $$2 \cdot B \cdot D \cdot F$$。然而，因为计算没有在 Z 维度上分片，我们实际上多做了 Z 倍的 FLOPs，即总共 $$2 \cdot B \cdot D \cdot F \cdot Z$$ 次 FLOPs。由于计算在其他维度上分片，每个设备的总量大约是 $$2 \cdot B \cdot D \cdot F / (X \cdot Y)$$。

{% enddetails %}

**问题 3：** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

{% details 点击这里查看答案。 %}

按照上面的规则，我们有 I 和 J 作为收缩维度，K、L、M、N 和 O 作为非收缩维度。我们没有"批次维度"，所以这就是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，所有轴的乘积。如果我们有共享轴，它只会被计算一次。

{% enddetails %}

**问题 4：** 自注意力的算术强度是多少（忽略 Q/K/V/O 投影）？*用 Q 和 KV 长度 T 和 S 的函数给出答案。* 在什么上下文长度时注意力是计算受限的？给定我们 TPU 的 HBM 带宽，画出随着上下文长度增长，注意力相对于 FFW 块的有效相对成本。

{% details 点击这里查看答案。 %}

自注意力需要加载 $$Q$$、$$K$$ 和 $$V$$ 激活值，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，然后将结果写回 HBM。这将使用 Flash Attention 完成，所以这个数学有一些注意事项，但基本上在 bf16 中自注意力执行：

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节数是 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总 FLOPs 是 $$4BTSNH + O(BTSN)$$，算术强度是 $$4BTSKGH / (4BHK * (TG + S))$$。

所以基本上，在预填充期间我们有 $$S=T$$，所以算术强度是 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成期间，$$T=1$$，所以我们有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$（假设 $$S$$ 非常大）。取决于你如何理解这个问题，在预填充或训练期间，假设没有序列分片，自注意力在 S=240 时是计算受限的。在生成期间，我们永远不会是计算受限的，因为 $$G$$ 很小。然而，你可以看到增加 $$G$$ 会让我们更接近计算受限。

{% enddetails %}

**问题 5：** 在什么序列长度时，自注意力 FLOPs 等于 QKVO 投影 FLOPs？

{% details 点击这里查看答案。 %}

这纯粹是一个关于何时 $$24BTDNH == 12BT^2NH$$ 的问题。简化后我们得到 $$2D = T$$，所以例如对于 $$D=4096$$，这是 $$8192$$。这告诉我们，对于大多数合理的上下文长度，矩阵乘法 FLOPs 更大。

{% enddetails %}

**问题 6：** 假设我们在前向传递中只保存 Transformer 层中 7 个主要矩阵乘法的输出（Q、K、V、O + 三个 FFW 矩阵）。在反向传递中需要多少额外的 FLOPs 来"重新计算"？

{% details 点击这里查看答案。 %}

只保存七个矩阵乘法输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传递必须重新计算两个注意力矩阵乘法：

$$QK^{\top} \quad\text{和}\quad \operatorname{softmax}(QK^{\top})V.$$

每个是一个在 $B$ 个序列和 $N$ 个头上批处理的 $T \times T$ 矩阵乘法，所以额外的 FLOPs 是：

$$4 \; B \, T^{2} \, N \, H.$$

所有其他重新计算的操作只有 $O(BTD)$。

{% enddetails %}

**问题 7：** DeepSeek v3 说它在 14.8T 个 token 上训练了 2.79M H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。鉴于它有 370亿激活参数，他们大约达到了什么硬件利用率？*提示：注意他们使用了没有结构化稀疏性的 FP8 FLOPs。*

{% details 点击这里查看答案。 %}

从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表，我们发现带稀疏性的 FP8 性能是 3,026 TFLOPs/s，或者通常没有稀疏性时是这个值的一半（`1.513e15` FLOPs/s）。2.79M H800 小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总 FLOPs。给定激活参数数量为 370亿，这个训练运行应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着 FLOPs 利用率大约是 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8：** 混合专家（MoE）模型有 $E$ 个标准稠密 MLP 块的副本，每个 token 激活其中 $k$ 个专家。在 TPU v5e 上使用 int8 权重的 MoE 需要多大的批次大小（以 token 计）才能是计算受限的？对于 DeepSeek，它有 256 个（路由的）专家和 $k=8$，这个数字是多少？

{% details 点击这里查看答案。 %}

因为我们有 $E$ 个专家副本，在 int8 中，我们需要加载 $E \cdot D \cdot F$ 字节。因为每个 token 激活 $k$ 个专家，我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。要使用 bfloat16 FLOPs 达到计算受限，我们需要算术强度超过 240，这发生在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时。

因此，我们需要 $B > 120 \cdot E / k$ 才能是计算受限的。对于 DeepSeek，这给我们 $B > 120 \cdot 256 / 8 = 3840$。这在生成时是一个相当大的批次大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第 4 部分到此结束！关于 Transformer 训练扩展的第 5 部分，请点击[这里](../training)！</h3>

## 附录

### 附录 A：Flash Attention 工作原理

将 Transformer 扩展到非常长上下文的传统反对意见是，注意力 FLOPs 和内存使用随上下文长度呈二次增长。虽然注意力 QK 乘积确实有形状 $[B, S, T, N]$（其中 B 是批次大小，S 和 T 是 Q 和 K 的序列维度，N 是头数），但这个说法有一些严重的注意事项：

1. 正如我们在第 4 章中提到的，即使这是二次的，注意力 FLOPs 只在 $$S > 8 \cdot D$$ 时才占主导地位，特别是在训练期间，单个注意力矩阵的内存与所有权重和激活检查点相比很小，尤其是在分片时。
2. 我们不需要完全生成完整的注意力矩阵来计算注意力！我们可以计算局部和与最大值，避免生成超过一小块数组。虽然总 FLOPs 仍然是二次的，但我们大大减少了内存压力。

第二个观察最早由 [Rabe 等人 2021](https://arxiv.org/abs/2112.05682) 提出，后来在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao 等人 2022）中进一步阐述。基本思想是按 K/V 块计算注意力，我们计算局部 softmax 和一些辅助统计量，然后将它们传递给下一个块，该块将它们与其本地块组合。具体来说，我们计算：

1. **M：** 序列维度上 $$q \cdot k$$ 的运行最大值
2. **O：** 序列维度上的运行完整注意力 softmax
3. **L：** 运行分母 $$\sum_i (q \cdot k_i - \text{运行最大值})$$

有了这些，我们可以用恒定的内存量计算新的最大值、新的运行和以及新的输出。为了给出这是如何工作的一个粗略描述，注意力大致是这个操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

减去最大值是为了数值稳定性，可以添加而不影响结果，因为 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$。只看上面的分母，如果我们想象有两个连续的键向量块 $$K^1$$ 和 $$K^2$$，我们分别计算每个的局部 softmax 和 $$L^1$$ 和 $$L^2$$：

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

然后我们可以用以下公式将这些组合成这两个块的完整 softmax 和：

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ 和 } M^2 = \max_j Q \cdot K_j^2$$

这可以对完整的 softmax 也这样做，给我们一种累积任意大 softmax 和的方法。这是 Flash Attention 论文中的完整算法：

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件的角度来看，这让我们可以将 Q 块放入 VMEM（上面的算法称之为片上 SRAM），所以我们只需要在每次迭代时加载 KV 块，从而提高算术强度。我们还可以将运行统计量保留在 VMEM 中。

最后一个值得强调的微妙点是注意力 softmax 的一个性质，它被用来使 Flash VJP（反向模式导数）计算对于训练来说是实用的。如果我们将中间 softmax 数组定义为：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

在注意力中，我们从反向模式的 *dO* 和 *V* 数组获得 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将这个梯度反向传播到 Q 和 K 时：

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，允许我们用沿特征**深度**维度的局部收缩替换沿大键**长度**维度的收缩：

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这种替换对于能够实现 VJP 的序列块*局部*计算至关重要，并使得更巧妙的分片方案（如环形注意力）成为可能。
