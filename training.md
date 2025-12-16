
---
layout: distill
title: "如何并行化 Transformer 训练"
# permalink: /main/
description: "本章讨论 LLM 训练中使用的四种主要并行化方案：数据并行、全分片数据并行（FSDP）、张量并行和流水线并行。对于每种方案，我们计算在什么时候会被通信瓶颈限制。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "第4部分：Transformer"

next_section_url: ../applied-training
next_section_name: "第6部分：训练 LLaMA"

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
  - name: "什么是扩展？"
  - subsections:
    - name: "数据并行"
    - name: "全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "结合 FSDP 和张量并行"
    - name: "流水线并行"
    - name: "跨 Pod 扩展"
  - name: "TPU 上 LLM 训练的要点"
  - name: "练习题"
  - name: "附录"
  - subsections:
    - name: "附录 A：推导反向传播通信"

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

## 什么是扩展？

"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现吞吐量的等比例线性增长（我们称之为*强扩展*）。虽然单个芯片上的性能取决于内存带宽和 FLOPs 之间的权衡，但集群级别的性能取决于通过将芯片间通信与有用的 FLOPs 重叠来隐藏通信开销。这并非易事，因为增加芯片数量会增加通信负载，同时减少我们可用来隐藏通信的每设备计算量。正如我们在[第3章](../sharding)中看到的，分片矩阵乘法通常需要昂贵的 AllGather 或 ReduceScatter 操作，这可能会阻止 TPU 执行有用的工作。本节的目标是找出这些操作何时变得*过于昂贵*。

在本节中，我们将讨论四种常见的并行化方案：（纯）**数据并行、全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称为模型并行）以及（简要介绍的）**流水线并行**。对于每种方案，我们将展示产生的通信成本以及该成本何时开始成为计算成本的瓶颈。<d-footnote>我们将重点关注通信边界——因为虽然内存容量限制很重要，但在使用重计算（激活检查点）和预训练期间使用大量芯片时，它们通常不会限制我们。我们在这里也不讨论 MoE 的专家并行——它大大扩展了设计空间，这里只讨论稠密 Transformer 的基本情况。</d-footnote>在本节中，你可以只关注芯片间通信成本，因为只要我们有足够大的单芯片批次大小，从 HBM 到 MXU 的数据传输就已经与计算重叠了。

我们将使用以下符号来简化本节中的计算。

| 符号 | 含义（模型参数）                                               |
| :--- | :------------------------------------------------------------- |
| D    | **d**<sub>model</sub>（隐藏维度/残差流维度）                   |
| F    | **d**<sub>ff</sub>（前馈维度）                                 |
| B    | 批次维度（批次中的 token 数量；总量，非每设备）                |
| T    | 序列长度                                                       |
| L    | 模型层数                                                       |

| 符号 | 含义（硬件特性）                                                                    |
| :--- | :---------------------------------------------------------------------------------- |
| C    | 每芯片 FLOPS/s                                                                      |
| W    | 网络带宽（双向，常用下标如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$）                   |
| X    | 沿网格轴 X 的芯片数量                                                               |
| Y    | 沿另一个网格轴（标记为 Y）的芯片数量                                                |
| Z    | 沿第三个网格轴（标记为 Z）的芯片数量                                                |

为简单起见，**我们将 Transformer 近似为一堆 MLP 块**——正如我们在[第4章](../transformers)中看到的，对于较大的模型，注意力只占 FLOPs 的相对较小部分。我们还将忽略门控矩阵乘法，留下以下每层的简单结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图：</b>简化的 Transformer 层。我们将每个 FFW 块视为两个矩阵的堆叠：<b>W<sub>in</sub></b>: <code>bf16[D, F]</code>（上投影）和 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code>（下投影），输入为 <b>In</b>: <code>bf16[B, D]</code>。" %}

{% details 这是没有并行化的小型 Transformer 的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面层需要*)

</div>

我们提供这个作为与添加通信的算法的比较。

{% enddetails %}

以下是我们将讨论的 4 种并行化方案。每种方案可以通过上图中 **In**、**W<sub>in</sub>、W<sub>out</sub> 和 Out** 的分片方式来唯一定义。

**1. 数据并行：** *激活沿批次分片，参数和优化器状态在每个设备上复制。通信仅在反向传播期间发生。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 全分片数据并行（FSDP 或 ZeRO-3）：** *激活沿批次分片（类似纯数据并行），参数沿相同网格轴分片，并在前向传播中使用前即时 AllGather。优化器状态也沿批次分片。减少重复内存。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为 Megatron 分片或模型并行）：** *激活沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。在每个块之前和之后对激活进行 AllGather 和 ReduceScatter。与 FSDP 兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活被微批处理并沿层维度滚动。流水线阶段之间的通信是最小的（只是单跳移动激活）。借用符号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当你的模型即使以很小的批次大小（>240 个 token，以达到计算受限）也能放入单个芯片时，**你应该始终使用简单的数据并行。** 纯数据并行将我们的激活拆分到任意数量的 TPU 上，只要 TPU 数量小于我们的批次大小。前向传播不涉及通信，但在每个步骤结束时，**每个 TPU 对其本地梯度执行 AllReduce 以在更新参数之前同步它们。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b>纯数据并行示意图（前向传播）。我们的激活（左）完全沿批次维度分片，权重完全复制，因此每个 TPU 都有相同的权重副本。这意味着权重的总内存增加了 N 倍，但前向传播不需要通信。" %}

{% details 这是前向和反向传播的完整算法。为了紧凑，我们滥用符号将 dL/dOut 写成 dOut。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面层需要*)

</div>

我们忽略损失函数的细节，并缩写 $\text{Tmp} = W_\text{in} \cdot \text{In}$。注意，虽然我们的最终损失是平均值 **AllReduce**(Loss[B<sub>X</sub>])，但我们只需要在反向传播中计算 AllReduce 来平均权重梯度。

{% enddetails %}

注意前向传播没有通信——**都在反向传播中**！反向传播还有一个很好的特性，即 AllReduce 不在"关键路径"上，这意味着每个 AllReduce 可以在方便的时候执行，不会阻塞你执行后续操作。如果总通信成本超过总计算成本，总体通信成本*仍然可能成为瓶颈*，但从实现角度来看更宽容。我们将看到模型/张量并行没有这个特性。

**为什么要这样做？** 纯数据并行通过沿批次维度拆分激活来减少激活内存压力，允许我们几乎任意增加批次大小，只要我们有更多芯片来拆分批次维度。特别是在训练期间，当我们的激活通常主导内存使用时，这非常有帮助。

**为什么不这样做？** 纯数据并行不能减少模型参数或优化器状态的内存压力，这意味着对于参数 + 优化器状态无法放入单个 TPU 的有趣大规模模型，纯数据并行很少有用。为了给出规模感，如果我们使用 bf16 参数和 fp32 优化器状态以及 Adam<d-footnote>Adam 存储参数、一阶和二阶累加器。由于参数是 bfloat16，优化器状态是 float32，这给我们每个参数 `2 + 8 = 10` 字节。</d-footnote>训练，我们能放下的最大模型有 $$\text{TPU 内存} / 10$$ 个参数，例如在具有 96GB HBM 的 TPU v5p 芯片上使用纯数据并行，这大约是 90亿参数。

<p markdown=1 class="takeaway">**要点**：我们可以用 Adam 和纯数据并行训练的最大模型有 $$\text{参数数量} = \text{每设备 HBM} / 10$$。对于 TPU v5p，这大约是 90亿参数。<d-footnote>注意这不包括梯度检查点，所以这实际上没什么用。这是批次大小为 1 个 token 的绝对下限。</d-footnote></p>

*要使其在训练期间对真实模型有用，我们需要至少部分分片模型参数或优化器。*

**什么时候会被通信瓶颈限制？** 如上所示，我们每层有两次 AllReduce，每次大小为 $$2DF$$（对于 bf16 权重）。数据并行何时使我们受通信限制？

如上表所示，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批次被分割的分片数<d-footnote>我们假设这种分区是在 ICI 网格上完成的，所以相关的网络带宽是 $W_\text{ici}$</d-footnote>。让我们计算执行相关矩阵乘法所需的时间 $$T_\text{math}$$，以及所需的通信时间 $$T_\text{comms}$$。由于这种并行化方案在前向传播中不需要通信，我们只需要为反向传播计算这些量。

*通信时间：* 从前面的章节我们知道，在 1D 网格中执行 AllReduce 所需的时间仅取决于被 AllReduce 的数组的总字节数和 ICI 带宽 $W_\text{ici}$；具体来说，AllReduce 时间是 $2 \cdot \text{总字节数} / W_\text{ici}$。由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 都进行 AllReduce，我们每层有 2 次 AllReduce。每次 AllReduce 是针对一个权重矩阵，即一个有 $DF$ 个参数或 $2DF$ 字节的数组。综合起来，单层的 AllReduce 总时间是：

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：* 每层在前向传播中包含两次矩阵乘法，或在反向传播中包含四次矩阵乘法，每次需要 $2(B/X)DF$ 次 FLOPs。因此，对于反向传播中的单层，我们有：

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们可以重叠，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时，我们变成计算受限的，即当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结论是，要保持数据并行的计算受限状态，我们需要每设备批次大小 $$B / X$$ 超过 ICI 算术强度 $C / W_\text{ici}$。这最终是因为计算时间随每设备批次大小扩展，而通信时间与此量无关（因为我们传输的是模型权重）。注意 $B > C/W_\text{ici}$ 条件与单设备计算受限规则 $B > 240$ 的相似性；在那种情况下，规则也来自于计算时间随批次大小扩展，而数据传输大小（在 $B \ll F, D$ 的情况下）与批次大小无关。

让我们放入一些实际数字来获得规模感。对于 TPU v5p，`C=4.6e14`，1D 数据并行的 ICI `W=2 * 9e10`，所以**我们每芯片的批次大小必须至少为 2,550 才能避免通信受限**。由于我们可以在多个轴上进行数据并行，如果我们将 TPU v5p pod 的所有三个轴都用于纯数据并行，我们将带宽 $W_\text{ici}$ 增加 3 倍，可以缩减到每 TPU 只需 BS=850，或每批次每 pod（8960 芯片）760万 token！**这告诉我们，被纯数据并行瓶颈限制相当困难！**

<p markdown=1 class="takeaway">**注释 [上下文并行]：** 在本节中，$B$ 始终指**以 token 计**的总批次大小。然而，显然我们的批次由许多不同的序列组成，那么这是如何工作的？就 MLP 而言，**token 就是 token**！它们属于同一个序列还是两个不同的序列并不重要。所以我们或多或少可以自由地在批次和序列维度上进行数据并行：我们称之为上下文并行或序列并行，但你可以把它看作是另一种数据并行。注意力比 MLP 更棘手，因为我们做一些跨序列计算，但这可以通过在注意力期间收集 KV 或 Q 并仔细重叠 FLOPs 和通信来处理（通常使用称为"环形注意力"的技术）。在本节中，我们将完全忽略序列维度，并假设有一定程度的批次或序列并行。</p>

**关于多网格轴的注释：** 我们应该快速说明多轴如何影响可用带宽。当我们为给定的并行化策略使用多个网格轴时，我们获得更多带宽。

* **定义：** $M_X$（$M_Y$、$M_Z$ 等）是给定并行化策略跨越的硬件网格轴数量。
* **效果（带宽受限）：** 使用 $M$ 个轴提供（约 $M$ 倍）的聚合链路带宽，所以集合操作时间按 $\propto 1/M_X$ 缩放。

### 全分片数据并行（FSDP）

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

全分片数据并行（通常称为 FSDP 或 ZeRO 分片<d-cite key="zero"></d-cite>）将模型优化器状态和权重跨数据并行分片拆分，并根据需要高效地收集和分散它们。**与纯数据并行相比，FSDP 大幅减少每设备内存使用并节省反向传播 FLOPs，开销非常小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b>FSDP 将 Win 的收缩维度和 Wout 的输出维度沿数据维度分片。这减少了内存，但（从第3章）要求我们在执行矩阵乘法之前收集 W 的权重。注意激活（左）<it>没有沿收缩维度分片</it>，这迫使我们进行收集。<b>注意我们的权重优化器状态同样沿收缩维度分片。</b>" %}

你会记得（从[第3章](../sharding)）AllReduce 可以分解为 AllGather 和 ReduceScatter。这意味着，我们可以将权重和优化器状态跨芯片分片，而不是为标准数据并行执行完整的梯度 AllReduce，在前向传播期间每层 AllGather 它们，在反向传播期间 ReduceScatter 跨权重，而没有额外成本。

{% details 这是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**全分片数据并行（FSDP）：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上，可以在前一层期间完成*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃 W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上，可以在前一层期间完成*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步完成*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可以提前完成*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(这里可以丢弃 W<sub>out</sub>[F, D])*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = dTmp[B<sub>X</sub>, F] \*<sub>B</sub> In[B<sub>X</sub>, D]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(不在关键路径上，可以异步完成)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可以提前完成*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面层需要）(这里可以丢弃 W<sub>in</sub>[D, F]*)

</div>

{% enddetails %}

这也被称为"ZeRO 分片"，来自"零开销分片"，因为我们不执行任何不必要的计算或存储任何不必要的状态。ZeRO-{1,2,3} 用于分别指代以这种方式分片优化器状态、梯度和权重。由于它们都有相同的通信成本<d-footnote>严格来说，FSDP 在前向传播中添加了纯 DP 没有的通信，但这与反向传播成比例，所以不应该影响通信 roofline。关键是 ZeRO-3 将反向传播的 AllReduce 变成了 AllGather 和 ReduceScatter，它们的总通信量相同。</d-footnote>，我们基本上总是可以做 ZeRO-3 分片，它将参数、梯度和优化器状态跨一组设备分片。

**为什么要这样做？** 标准数据并行涉及大量重复工作。每个 TPU AllReduce 完整梯度，然后更新完整优化器状态（所有 TPU 上相同的工作），然后更新参数（同样完全重复）。对于 ZeRO 分片（分片梯度/优化器状态），你可以 ReduceScatter 梯度，而不是 AllReduce，只更新你的优化器状态分片，更新参数分片，然后根据前向传播需要 AllGather 参数。

**什么时候会被通信瓶颈限制？** 我们的相对 FLOPs 和通信成本与纯数据并行完全相同，因为反向传播中的每个 AllReduce 已变成 AllGather + ReduceScatter。回想一下，AllReduce 实现为 AllGather 和 ReduceScatter，每个成本的一半。这里我们建模前向传播，因为它与反向传播具有相同的 FLOPs 对通信比：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，当 $$B / X > C / W_\text{ici}$$ 时我们是计算受限的，即当每设备批次大小 $B/X$ 超过"ICI 算术强度"$C/W_\text{ici}$（v5p 为 `4.59e14 / 1.8e11 = 2550`）。这对我们来说很好，因为这意味着如果我们的每设备批次大小足够大到纯数据并行是计算受限的，我们可以——不用担心离开计算受限区域——简单地升级到 FSDP，为自己节省大量参数和优化器状态内存！虽然我们确实必须在前向传播中添加通信，但这个成本无关紧要，因为它只是与前向传播 FLOPs 重叠。

<p markdown=1 class="takeaway">**要点：** FSDP 和纯数据并行在 TPU v5 上当每设备批次大小小于 $2550 / M_X$ 时变成带宽受限的，其中 $M_X$ 是网格轴数。</p>

例如，DeepSeek-V2（最近发布训练批次大小信息的少数强模型之一）使用了约 4000万 token 的批次大小。**这将允许我们扩展到大约 47,000 芯片，或大约 5 个 TPU v5 pod，在达到带宽限制之前。**

对于 LLaMA-3 70B，它使用大约 `6.3e24 (15e12 * 70e9 * 6)` FLOPs 训练，我们可以将 1600万 token 的批次分割到大约 `16e6 / (2550 / 3) = 18,823` 芯片（大约 2 个 8960 芯片的 pod），每个芯片有 `4.59e14` FLOPs，以 50% 峰值 FLOPs 利用率（通常称为 MFU）运行，**大约 17 天内训练完成**。不错！但让我们探索如何做得更好。

<p markdown=1 class="takeaway">**关于临界批次大小的注释**：有点违反直觉的是，当我们的总批次大小减少（芯片数量固定）时，我们变得更加通信瓶颈。数据并行和 FSDP 让我们可以扩展到任意多的芯片，只要我们能不断增加批次大小！然而，在实践中，随着批次大小增加，我们往往会看到训练收益递减，因为我们的梯度几乎变得无噪声。我们有时还会看到训练不稳定。因此，在"无限计算"情况下找到最优分片方案的游戏通常从一个由扩展定律确定的固定批次大小和一个已知（大量）芯片数量开始，然后旨在找到一个允许我们将那个小批次大小放在这么多芯片上的分区。</p>

### 张量并行

**语法：** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用 $$Y$$ 以便最终与 FSDP 结合）

在全分片数据并行 AllReduce 中，我们跨芯片移动权重。我们也可以分片模型的前馈维度并在层期间移动激活——这被称为"1D 模型并行"或 Megatron 分片<d-cite key="megatron"></d-cite>。这可以解锁更小的每 pod 高效批次大小。下图显示了以这种方式分片的单个矩阵的示例：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图：</b>基本张量并行的示例。因为我们只在 Y 上分片激活（不像 FSDP 中我们在 X 上分片），我们在 X 上复制激活。使用我们的标准语法，这是 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为我们只沿一个收缩维度分片，我们通常在矩阵乘法前 AllGather 激活 <b>A</b>。" %}

如所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一个矩阵乘法前收集激活。当激活比权重小时，这比 ZeRO 分片便宜。** 这通常只有在添加一定量的 ZeRO 分片（减少收集的大小）时才成立。这是我们倾向于混合 ZeRO 分片和张量并行的原因之一。

{% details 这是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(在关键路径上)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(沿收缩维度没有分片，所以没有通信)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(在关键路径上)*
5.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(这里可以丢弃 dOut[B, D])*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(通过与前向传播 (1) 共享可以跳过)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = dTmp[B, F<sub>Y</sub>] \*<sub>B</sub> In[B, D]
7.  dIn[B, D] {U.Y} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前面层需要)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

张量并行的一个好处是它与 Transformer 前向传播中的两个矩阵很好地交互。朴素地，我们会在每个矩阵之后做一次 AllReduce。但这里我们首先做 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，然后做 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开始时 AllGather **In**，在结束时 ReduceScatter **Out**，而不是做一次 AllReduce。

**这有多昂贵？** 让我们只建模前向传播——反向传播只是这里每个操作的转置。在 1D 张量并行中，我们在第一个矩阵乘法前 AllGather 激活，在第二个之后 ReduceScatter，每次发送两个字节（bf16）。让我们弄清楚什么时候我们会被通信瓶颈限制。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意我们希望计算成本大于通信成本，我们得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此，例如对于 TPU v5p，bf16 下 $C / W_{ici} = 2550$，所以我们只能做到 $Y < F / 2550$ 的张量并行。当我们有多个 ICI 轴时，我们的 $T_\text{comms}$ 减少 $M_Y$ 倍，所以我们得到 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行变成通信受限的。对于大多数模型，这是 8 到 16 路张量并行。</p>

**注意这不依赖于计算的精度**，因为例如对于 int8，在 TPU v5p 上，$$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$，但通信量也减半，所以两个因子 2 抵消了。

**让我们思考一些例子：**

* 在 TPU v5p 上使用 LLaMA 3-70B，$$D = 8192,$$ $$F \approx 30,000$$，我们可以舒适地做 8 路张量并行，但在 16 路张量并行时会是通信受限的。模型 8 路分片所需的 F 是 20k。

* 对于 Gemma 7B，$$F \approx 50k$$，所以我们在 19 路张量并行时变成通信受限。这意味着我们可能可以做 16 路并且仍然看到好的性能。

### 结合 FSDP 和张量并行

**语法：** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 和张量并行的好处是它们可以结合。通过沿两个轴分片 **W<sub>in</sub>** 和 **W<sub>out</sub>**，我们既节省内存又节省计算。因为我们沿 X 分片 B，我们减少了模型并行 AllGather 的大小，因为我们沿 Y 分片 F，我们减少了 FSDP 的通信开销。这意味着两者的结合可以让我们达到比上面看到的更低的有效批次大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b>结合 FSDP 和张量并行的示意图。与其他情况不同，模型参数没有重复。" %}

{% details 这是混合 FSDP + 张量并行的完整算法。虽然我们有很多通信，但因为我们批次分片了激活和张量分片了权重，所有的 AllGather 和 ReduceScatter 都更小了！ %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
5.  Out[B<sub>X</sub>, D] {U.Y} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*
7.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(这里可以丢弃 dOut[B, D])*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(不在关键路径上 + 可以与前一层的 (2) 共享)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> In[B<sub>X</sub>, D]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
11. dIn[B<sub>X</sub>, D] {U.Y} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前面层需要)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

**FSDP 和 TP 的正确组合是什么？** 一个简单但关键的准则是 FSDP 移动权重，张量并行移动激活。这意味着随着批次大小缩小（特别是当我们做更多数据并行时），张量并行变得更便宜，因为每分片的激活更小。

* 张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$，随着 $$X$$ 增长而缩小。
* FSDP 执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$，随着 $$Y$$ 增长而缩小。

因此，通过结合两者，我们可以将每副本的最小批次大小进一步降低。我们可以用与上面相同的方式计算 FSDP 和 TP 的最优量：

令 $$X$$ 为用于 FSDP 的芯片数，$$Y$$ 为用于张量并行的芯片数。令 $$N$$ 为我们切片中的芯片总数，$$N=XY$$。令 $$M_X$$ 和 $$M_Y$$ 分别为我们做 FSDP 和 TP 的网格轴数（它们大致应该加起来为 3）。我们纯粹建模前向传播，因为它每 FLOP 的通信最多。然后将上面算法中的通信加起来，我们有

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样，我们的总 FLOPs 时间是

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为了简化分析，我们做两个假设：首先，我们允许 $X$ 和 $Y$ 取非整数值（只要它们是正的并满足 $XY=N$）；其次，我们假设我们可以完全重叠 $X$ 和 $Y$ 轴上的通信。在第二个假设下，总通信时间是

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们问在什么条件下我们会是计算受限的之前，让我们找到最小化总通信的 $X$ 和 $Y$ 的最优值。由于我们的 FLOPs 与 $X$ 和 $Y$ 无关，最优设置是那些简单地最小化通信的设置。为此，让我们用 $X$ 和 $N$（它是固定的，因为它是我们系统中的芯片数）而不是 $X$ 和 $Y$ 来写上面的 $T_\text{comms}$：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

因为 $T_\text{FSDP comms}$ 是 $X$ 的单调递增函数，$T_\text{TP comms}$ 是 $X$ 的单调递减函数，最大值必须在 $T_\text{FSDP comms} = T_\text{TP comms}$ 时最小化，这发生在

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这超级有用！这告诉我们，对于给定的 $B$、$F$ 和 $N$，什么量的 FSDP 是最优的。让我们感受一下规模。代入现实值，即 $N = 64$（对应于 4x4x4 芯片阵列）、$B=48,000$、$F=32768$，得到大约 $X\approx 13.9$。所以我们会选择 $X$ 为 16，$Y$ 为 4，接近我们计算的最优值。

<p markdown=1 class="takeaway">**要点：** 一般来说，在训练期间，FSDP 的最优量是 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。</p>

现在让我们回到我们一直在问所有并行化策略的问题：**在什么条件下我们会是计算受限的？** 由于我们可以重叠 FLOPs 和通信，当

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

时我们是计算受限的。

令 $\alpha \equiv C / W_\text{ici}$，即 ICI 算术强度，我们可以简化：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算 $X_{opt}$ 使左边的最大值相等，我们可以直接代入任一边（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步简化，我们发现

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左边与通信时间成正比，右边与计算时间成正比。注意，虽然计算时间随批次大小线性扩展（无论哪种并行化方式都是如此），但通信时间随批次大小的平方根扩展。因此，计算与通信时间的比率也随批次大小的平方扩展：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为确保这个比率大于 1 以便我们是计算受限的，我们需要

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为了得到近似数字，再次代入 $F=32,768$、$\alpha=2550$ 和 $M_X M_Y=2$（对于 3D 网格必须如此）。这给出大约 $B/N > 99$。与纯数据并行（或 FSDP）情况相比，这大约给我们赢得了 8 倍，在那种情况下假设 3D 网格，我们计算 $B/N$ 必须超过约 $850$ 才能计算受限。

<p markdown=1 class="takeaway">**要点：** 结合张量并行和 FSDP 允许我们将 $B/N$ 降到 $$2550^2 / 2F$$。这让我们可以处理每芯片低至 100 的批次大小，这大约比仅用 FSDP 能达到的小 8 倍。</p>

下面我们绘制了混合 FSDP + TP 的 FLOPs 对通信时间比率，将其与仅张量并行（TP）和仅数据并行（FSDP）在代表性的 4x4x4 芯片阵列上进行比较。虽然纯 FSDP 并行对于非常大的批次大小占主导地位，但在每芯片批次大小在大约 100 到 850 之间的区域，需要混合 FSDP + TP 策略才能保持计算受限。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b>TPU v5p 4x4x4 切片上最优混合 FSDP/TP 的 FLOPs 对通信时间比率，F=30k。如预期，张量并行与批次大小有固定比率；理想的混合 FSDP + TP 按 $\sqrt{B}$ 扩展，FSDP 按 $B$ 扩展。然而，在中间批次大小区域，只有 FSDP + TP 达到大于 1 的比率。"%}

这是 TPU v5p 16x16x16 的另一个例子，显示不同分片方案下 FLOPs 和通信时间随批次大小的变化。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图：</b>不同并行化方案的通信时间。黑色虚线是矩阵乘法 FLOPs 所需的时间，所以任何在这条线上方的曲线都是通信受限的。我们注意到所有策略在批次大小 6e5 以下都变成通信受限，这与我们预期的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 一致。" %}

黑色曲线是模型 FLOPs 花费的时间量，意味着任何这条线低于所有通信成本的批次大小都严格是通信受限的。你会注意到黑色曲线在大约 `4e5` 处与绿色曲线相交，正如预测的那样。

这是一个交互式动画来玩这个，显示不同批次大小的总计算时间和通信时间：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

你会注意到这与上面大体一致（最小值在 FSDP=256，TP=16 附近），加减一些由于每个轴数量略有不同的波动因素。

### 流水线并行

你可能会注意到我们在前面的章节中一直避免谈论流水线。流水线是 GPU 并行化的主导策略，但在 TPU 上不那么必要。简而言之，流水线训练涉及将模型的层拆分到多个设备上，并在前向和反向传播期间在流水线阶段之间传递激活。算法大致如下：

1. 在 TPU 0 上初始化数据，权重沿层维度分片（对于带 FSDP 和张量并行的流水线是 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上执行第一层，然后将结果激活复制到 TPU 1，重复直到到达最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对于最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段，重复直到到达 TPU 0。

{% details 这是一些（可运行的）Python 伪代码 %}

这个伪代码应该可以在 Cloud TPU VM 上运行。虽然它不是很高效或现实，但它让你感受数据如何在设备之间传播。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 假设每层只是一个矩阵乘法。
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# 假设 num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # 编造一些假的损失函数

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(0, num_layers, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i + 1], weights[i])
  dx, dw = f_vjp(dx)  # 计算 jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # 更新权重

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线有很多优点：它在流水线阶段之间有低通信成本，这意味着即使互连带宽很低，你也可以训练非常大的模型。这在 GPU 上通常非常有用，因为它们不像 TPU 那样通过 ICI 密集连接。

**为什么这很困难/烦人？** 你可能在上面的伪代码中注意到 TPU 0 几乎总是空闲的！它只在流水线的第一步和最后一步做工作。这段空闲期称为流水线气泡，处理起来非常烦人。通常我们首先尝试通过微批处理来缓解这一点，微批处理将多个小批次发送通过流水线，使 TPU 0 在总步骤时间的至少更大比例内保持利用。

第二种方法是仔细重叠前向矩阵乘法 $W_i @ x_i$、反向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$ 和 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于每个都需要一些 FLOPs，我们可以重叠它们以完全隐藏气泡。这是最近 DeepSeek v3 论文<d-cite key="DeepSeek3"></d-cite>中的一张图，显示了他们的"无气泡"流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b>DeepSeek v3 流水线调度（来自他们的<a href=\"https://arxiv.org/pdf/2412.19437\">最近论文</a>）。橙色是前向矩阵乘法，绿色是 dL/dx 矩阵乘法，蓝色是 dL/dW 矩阵乘法。通过优先处理反向 dL/dx 乘法，我们可以避免"搁浅"FLOPs。" %}

因为它对 TPU（有更大的互联 pod）不那么关键，我们不会深入探讨这个，但理解关键的流水线瓶颈是一个很好的练习。

### 跨 Pod 扩展

最大可能的 TPU 切片是具有 8960 芯片（和 2240 主机）的 TPU v5p SuperPod。当我们想要扩展超过这个大小时，我们需要跨越数据中心网络（DCN）边界。每个 TPU 主机配备一个或多个 NIC（网络接口卡），通过以太网将主机连接到其他 TPU v5p pod。如[TPU 章节](../tpus)所述，每个主机有大约 200Gbps（25GB/s）的全双工 DCN 带宽，即每 TPU 约 6.25GB/s 全双工（出口）带宽。

通常，当扩展超过单个 pod 时，我们在 ICI 域内做某种形式的模型并行或 FSDP，然后跨多个 pod 做纯数据并行。令 $N$ 为我们要扩展到的 TPU 数量，$M$ 为每个 ICI 连接切片的 TPU 数量。要在 DCN 上做 AllReduce，我们可以在 pod 集合上做环形规约，得到（在反向传播中）：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 扩展，因为与 ICI 不同，总带宽随着我们增长 ICI 域并获得更多 NIC 而增长。简化后，我们发现 $T_\text{math} > T_\text{comms}$ 当

$$\frac{B}{\text{切片}} > \frac{C}{W_\text{dcn}}$$

对于 TPU v5p，$\frac{C}{W_\text{dcn}}$ 大约是 `4.46e14 / 6.25e9 = 71,360`。这告诉我们，要有效地扩展到 DCN 上，需要每个 ICI 域的最小批次大小才能出口每个节点。

**这有多大问题？** 举一个具体例子，假设我们想在 TPU v5p 上以 200万 token 的 BS 训练 LLaMA-3 70B。LLaMA-3 70B 有 $F\approx 30,000$。从上面的章节，我们知道以下几点：

* 我们可以做张量并行到 $Y = M_Y \cdot F / 2550 \approxeq 11 \cdot M_Y$ 以上。
* 只要 $B / N > 2550 / M_X$，我们就可以做 FSDP。这意味着如果我们想用 BS=200万和 3 轴数据并行训练，我们最多只能使用约 2400 芯片，大约是 TPU v5p pod 的四分之一。
* 当我们结合 FSDP + 张量并行时，当 $B / N < 2550^2 / 2 * 30,000 = 108$ 时变成通信受限，所以这让我们可以扩展到大约 18k 芯片！然而，TPU v5p pod 的最大大小是 8k 芯片，所以超过那个我们必须使用 DCN。

简而言之，我们有一个很好的配方来用 BS=100万训练，使用大约 X（FSDP）= 1024 和 Y（TP）= 8，但用 BS=200万我们需要使用 DCN。如上所述，我们的 DCN 算术强度是 $\text{71,360}$，所以我们只需要确保每个 ICI 域的批次大小大于这个值。这对我们来说很简单，因为用 2 个 pod，我们的每 pod BS 将是 100万，每 TPU 批次大小是 111，这很好（可能有点接近，但理论上可行）。

<p markdown=1 class="takeaway">**要点：** 使用纯数据并行跨多个 TPU pod 扩展相当简单，只要我们的每 pod 批次大小至少是 71k token。</p>

## TPU 上 LLM 训练的要点

* 增加并行性或减少批次大小都倾向于使我们更加通信受限，因为它们减少了每芯片执行的计算量。

* 在合理的上下文长度（约 32k）内，我们可以将 Transformer 建模为一堆 MLP 块，并通过它们如何分片每层的两/三个主要矩阵乘法来定义几种并行化方案。

* 训练期间我们考虑 4 种主要并行化方案，每种都有自己的带宽和计算要求（数据并行、FSDP、张量并行）。

| **策略**                                     | **描述**                                                                                                                                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **数据并行**                                 | 激活沿批次分片，其他一切完全复制，我们在反向传播期间 all-reduce 梯度。                                                                                                                  |
| **FSDP**                                     | 激活、权重和优化器沿批次分片，权重在使用前收集，梯度 reduce-scatter。                                                                                                                   |
| **张量并行（又名 Megatron、模型并行）**      | 激活沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片，激活在 W<sub>in</sub> 前收集，结果在 W<sub>out</sub> 后 reduce-scatter。                                                        |
| **混合 FSDP + 张量并行**                     | 以上两者，FSDP 收集模型分片的权重。                                                                                                                                                     |

这是每种方法的"公式"：

$$\small
\begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 每种策略都有一个变成网络/通信受限的限制，基于其每设备计算和通信。这是每层的计算和通信，假设 $$X$$ 是 FSDP，$$Y$$ 是张量并行。

$$
\small
\begin{array}{ccc}
\text{策略} & \text{每层计算量} & \text{每层通信量} \\
& \text{（忽略门控 einsum）} & \text{（字节，前向 + 反向传播）}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态使用字节 = 10 倍参数数量。这意味着我们很少能在内存中放下超过几十亿参数。

* 当 $$\text{每分片批次大小} < C / W$$ 时，数据并行和 FSDP 变成通信受限，这是网络的算术强度。对于 ICI 这是 2,550，对于 DCN 这是 75,000。这可以通过更多并行轴增加。

* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行变成通信受限。**对于大多数模型，这在 8-16 路左右。** 这与批次大小无关。

* 混合 FSDP + 张量并行允许我们将批次大小降到低至 $$2550^2 / 2F \approx 100$$。这相当低。

* 跨 pod 的数据并行需要每 pod 最小批次大小约 75,000 才能不被 DCN 限制。

* 基本上，如果你的批次大小很大或模型很小，事情很简单。你可以做数据并行或 FSDP + 跨 DCN 的数据并行。中间部分是事情变得有趣的地方。

## 练习题

让我们用 LLaMA-2 13B 作为本节的基本模型。这是模型详情：

| 超参数     | 值     |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 有单独的嵌入和输出矩阵以及门控 MLP 块。

**问题 1：** LLaMA-2 13B 有多少参数（我知道这很傻，但做一下数学）？*注意，如在[Transformer 数学](../transformers)中，LLaMA-3 有 3 个大 FFW 矩阵，两个上投影和一个下投影。我们在本节中忽略了两个"门控"einsum 矩阵，但它们的行为与本节中的 W<sub>in</sub> 相同。*

{% details 点击这里查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词汇表参数：$$2VD$$ = `0.3e9`
* 总计：`8.5e9 + 4.2e9 + 0.39e9 = 13.1e9`，如预期！

{% enddetails %}

**问题 2：** 假设我们用 BS=1600万 token 和 Adam 训练。暂时忽略并行化，模型的参数、优化器状态和激活使用多少总内存？*假设我们以 bf16 存储参数，以 fp32 存储优化器状态，并在三个大矩阵乘法后检查点激活。*

{% details 点击这里查看答案。 %}

参数（bf16）和两个优化器状态（fp32，一阶和二阶动量累加器）使用的总内存是 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘法后的激活形状为 $BF$，最后一个为 $BD$（按上面的 Transformer 图），所以 bf16 的总内存是 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 或 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e6`。所有其他激活或多或少可以忽略。

{% enddetails %}

**问题 3：** 假设我们想在 TPU v5p 16x16x16 切片上以 32k 序列长度和 300万 token 的总批次大小训练。假设我们要使用 bfloat16 权重和 float32 优化器，如上所述。

1. 我们可以使用纯数据并行吗？为什么或为什么不？
2. 我们可以使用纯 FSDP 吗？为什么或为什么不？使用纯 FSDP，每个设备将使用多少内存（假设我们只在 3 个大 FFW 矩阵后做梯度检查点）。
3. 我们可以使用混合 FSDP + 张量并行吗？为什么或为什么不？如果可以，$X$ 和 $Y$ 应该是多少？每个设备将存储多少内存？仅使用 roofline FLOPs 估计并忽略注意力，在 40% MFU 下每个训练步骤需要多长时间？

{% details 点击这里查看答案。 %}

首先，让我们写下一些数字。32k 序列长度和 300万批次大小，我们的序列批次大小为 96。在 TPU v5p 16x16x16 切片上，我们有 `393TB` 的 HBM。

1. 我们不能使用纯数据并行，因为它在每个芯片上复制参数和优化器状态，这些已经大约是 130GB（来自问题2），这比我们每芯片的 HBM（96GB）还多。

2. 让我们首先纯粹看内存。将问题2中的 BS=1600万替换为 300万，我们得到 `~7.86e12` 总检查点激活，加上 1.3e11 优化器状态，这使我们几乎正好是 8e12 = 8TB。TPU v5p 切片总共有 `393TB` 的 HBM，所以我们安全地在 HBM 限制之下。接下来让我们看看我们是否会是通信或计算受限的。有 4096 芯片和 3 轴并行，我们可以做最小批次大小 `850 * 4096 = 3.48M` token。这略高于我们的 300万批次大小。所以我们实际上是通信受限的，这很遗憾。所以一般答案是**不，我们不能单独做 FSDP**。

3. 现在我们知道我们的主要关注是通信受限，让我们代入一些数字。首先，我们知道从上面，我们的混合 FSDP + 张量并行的每芯片批次大小需要高于 $2550^2 / 2F = 235$。这意味着理论上我们可以做到！让我们弄清楚每个的量。

我们有规则 $X_{opt} = \sqrt((F / B) * (M_X / M_Y) * N)$，所以这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，意味着我们将做大约 1024 路 DP 和 4 路 TP。每 TPU 内存将如 (2) 中所述，步骤时间将只是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第 5 部分到此结束！第 6 部分将此内容应用于真实的 LLaMA 模型，请点击[这里](../applied-training)！</h3>

## 附录

### 附录 A：推导反向传播通信

上面，我们将 Transformer 层前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。我们如何推导反向传播所需的通信？

这自然地从上一节中单个矩阵乘法 **Y = X * A** 的规则推导出来：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个，我们得到以下公式（令 Tmp[B, F] 代表 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub> = dTmp[B, F] *<sub>B</sub> Tmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

注意这些公式是数学陈述，没有提到分片。反向传播的工作是计算这四个量。因此，要弄清楚必要的通信，我们只需取上面四个方程中要进行矩阵乘法的所有量的分片（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>），这些由我们的并行化方案指定，并使用分片矩阵乘法的规则来弄清楚我们必须做什么通信。注意 dOut 与 Out 以相同的方式分片。