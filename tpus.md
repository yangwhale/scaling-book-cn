---
layout: distill
title: "如何理解 TPU"
# permalink: /main/
description: "本章节全面介绍 TPU 的工作原理、它们如何联网组成多芯片训练和推理系统，以及这如何影响我们最喜欢的算法的性能。对 GPU 用户也有很多有价值的内容！"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

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

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 什么是 TPU？
  - name: TPU 网络
  - name: 核心要点
  - subsections:
    - name: TPU 规格
  - name: 练习题
  - name: 附录
  - subsections:
    - name: "附录 A：TPU 内部细节"
    - name: "附录 B：脉动阵列如何工作？"

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

<p markdown=1 class="announce">你可能还会喜欢阅读关于 NVIDIA GPU 的新[第12章](../gpus)！</p>

## 什么是 TPU？

**TPU 基本上是一个专门用于矩阵乘法的计算核心（称为 TensorCore）连接到一堆快速内存（称为高带宽内存或 HBM）<d-cite key="tpu_paper"></d-cite>。** 这是一个示意图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图示：</b> TPU 芯片的基本组件。TensorCore 是左侧的灰色方框，包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。" %}

你可以把 TensorCore 基本上看作是一个非常擅长矩阵乘法的机器，但它还有一些其他值得注意的功能。TensorCore 有三个关键单元：

* **MXU**（矩阵乘法单元）是 TensorCore 的核心。对于大多数 TPU 代际，它每 8 个周期执行一次 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法，使用脉动阵列（参见<a href="#附录-b脉动阵列如何工作">附录 B</a>了解详情）。<d-footnote>TPU v6e（Trillium）有一个 256x256 的 MXU，而之前所有代际都使用 128x128。</d-footnote>
  * 在 TPU v5e 上，这大约是 `5e13` bf16 FLOPs/s（每个 MXU，1.5GHz）。大多数 TensorCore 有 2 个或 4 个 MXU，所以例如 TPU v5e 的总 bf16 FLOPs/s 是 `2e14`。
  * TPU 还支持更低精度的矩阵乘法，具有更高的吞吐量（例如，每个 TPU v5e 芯片可以做 `4e14` int8 OPs/s）。

* **VPU**（向量处理单元）执行一般的数学运算，如 ReLU 激活或向量之间的逐点加法或乘法。归约（求和）也在这里执行。<a href="#附录-atpu-内部细节">附录 A</a> 提供了更多细节。
* **VMEM**（向量内存）是位于 TensorCore 内部的片上暂存器，靠近计算单元。它比 HBM 小得多（例如，TPU v5e 上是 128 MiB），但到 MXU 的带宽高得多。VMEM 的工作方式有点像 CPU 上的 L1/L2 缓存，但更大且由程序员控制。HBM 中的数据需要复制到 VMEM 中，TensorCore 才能用它进行任何计算。

**TPU 在矩阵乘法方面非常非常快**。这主要是它们做的事情，而且做得很好。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 是迄今为止最强大的 TPU 之一，每核可以做 `2.5e14` bf16 FLOPs/秒，每芯片 `5e14` bf16 FLOPs/秒。一个 8960 芯片的 Pod 可以做 4 exaflops/秒。那是*很多*。那是世界上最强大的超级计算机之一。而 Google 有很多这样的。<d-footnote>TPU，特别是它们的脉动阵列，是如此强大的硬件加速器，因为矩阵乘法是少数几个使用 $O(n^3)$ 计算来处理 $O(n^2)$ 字节的算法之一。这使得普通 ALU 很容易受到计算而不是内存带宽的瓶颈。</d-footnote>

上图还包括一些其他组件，如 SMEM 和标量单元，用于控制流处理，在<a href="#附录-atpu-内部细节">附录 A</a> 中有简要讨论，但不是理解的关键。另一方面，HBM 很重要且相当简单：

* **HBM**（高带宽内存）是一大块快速内存，用于存储张量供 TensorCore 使用。HBM 通常具有数十 GB 的容量（例如，[TPU v5e 有 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

  * 当需要进行计算时，张量从 HBM 通过 VMEM（见下文）流式传输到 MXU，结果从 VMEM 写回 HBM。

  * HBM 和 TensorCore 之间（通过 VMEM）的带宽称为"HBM 带宽"（通常约 1-2TB/秒），限制了在内存受限工作负载中计算的速度。

**通常，所有 TPU 操作都是流水线化和重叠的。** 要执行矩阵乘法 $X \cdot A \to Y$，TPU 首先需要将矩阵 $A$ 和 $X$ 的块从 HBM 复制到 VMEM，然后将它们加载到 MXU 中，MXU 乘以 8x128（对于 $X$）和 128x128（对于 $A$）的块，然后逐块将结果复制回 HBM。为了高效地做到这一点，矩阵乘法是流水线化的，这样到/从 VMEM 的复制与 MXU 工作重叠。这使得 MXU 可以继续工作，而不是等待内存传输，保持矩阵乘法计算受限，而不是内存受限。

这是一个如何从 HBM 执行逐元素乘积的示例：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图示：</b> 一个动画展示了在 TPU 上执行逐点乘积的过程，字节从 HBM 加载。注意字节如何以块的形式从内存流式传输，部分结果在不等待完整数组物化的情况下通过流水线传回。" %}

矩阵乘法看起来几乎相同，只是它会加载到 MXU 而不是 VPU/向量单元，而且加载和存储会以不同的顺序发生，因为相同的权重块用于多个激活块。你可以看到数据块流入 VMEM，然后流入 VREGs（向量寄存器），然后流入向量单元，然后返回 VMEM 和 HBM。正如我们即将看到的，如果从 HBM 到 VMEM 的加载比向量单元（或 MXU）中的 FLOPs 慢，我们就变成"带宽受限"的，因为我们在让 VPU 或 MXU 处于饥饿状态。

<p markdown=1 class="takeaway">**核心要点：** TPU 非常简单。它们将权重从 HBM 加载到 VMEM，然后从 VMEM 加载到脉动阵列，该阵列每秒可以执行约 200 万亿次乘加运算。HBM $\leftrightarrow$ VMEM 和 VMEM $\leftrightarrow$ 脉动阵列的带宽对 TPU 能高效执行哪些计算设定了基本限制。</p>

**VMEM 和算术强度：** VMEM 比 HBM 小得多，但到 MXU 的带宽高得多。正如我们在[第1章](../roofline)中看到的，这意味着如果一个算法可以将其所有输入/输出放入 VMEM，它就不太可能遇到通信瓶颈。这在计算具有较差算术强度时特别有帮助：VMEM 带宽比 HBM 带宽高约 22 倍，这意味着从/向 VMEM 读取/写入的 MXU 操作只需要 10-20 的算术强度就能达到峰值 FLOPs 利用率。这意味着如果我们可以将权重放入 VMEM 而不是 HBM，我们的矩阵乘法可以在更小的批量大小下成为 FLOPs 受限的。这也意味着从根本上具有较低算术强度的算法仍然可以高效。只是 VMEM 太小了，这通常是一个挑战。<d-footnote>我们有时谈到 VMEM 预取，这指的是提前将权重加载到 VMEM 中，这样我们可以掩盖矩阵乘法加载的成本。例如，在正常的 Transformer 中，我们有时可以在注意力期间将大的前馈权重加载到 VMEM 中，如果我们是内存带宽受限的，这可以隐藏权重加载的成本。这要求我们的权重足够小或分片得足够多，以便在 VMEM 中放下单层并留有空间。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一个 TPU 芯片通常（但不总是）由两个共享内存的 TPU 核心组成，可以看作是一个具有两倍 FLOPs 的大型加速器**（称为"megacore"配置）。这从 TPU v4 开始就是这样。较老的 TPU 芯片有独立的内存，被视为两个独立的加速器（TPU v3 及更早版本）。像 TPU v5e 这样的推理优化芯片每个芯片只有一个 TPU 核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片以4个为一组排列在"托盘"上**，通过 PCIe 网络连接到 **CPU 主机**。这是大多数读者熟悉的格式，4 个芯片（8 个核心，尽管通常被视为 4 个逻辑 megacore）通过 Colab 或单个 TPU-VM 暴露。对于像 TPU v5e 这样的推理芯片，每个主机有 2 个托盘，而不是 1 个，但每个芯片也只有 1 个核心，给我们 8 个芯片 = 8 个核心。<d-footnote>在 Cloud TPU VM 上，每个托盘作为单独的 VM 的一部分暴露，所以再次有 4 个可见核心。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽有限：** 像 HBM $\leftrightarrow$ VMEM 链路一样，CPU $\leftrightarrow$ HBM 的 PCIe 连接有特定的带宽，限制了你可以多快地从主机内存加载到 HBM 或反过来。例如，TPU v4 的 PCIe 带宽是每方向 16GB/秒，所以比 HBM 慢近 100 倍。我们*可以*将数据加载/卸载到主机（CPU）RAM，但不是很快。

## TPU 网络

**芯片在 Pod 中通过 ICI 网络相互连接**。在较老的代际（TPU v2 和 TPU v3）、推理芯片（例如 TPU v5e）和 Trillium（TPU v6e）中，ICI（"芯片间互连"）连接 4 个最近的邻居（具有边缘链路以形成 2D 环面）。TPU v4 和 TPU v5p 连接到最近的 6 个邻居（形成 3D 环面）。注意这些连接**不**通过它们的主机，它们是芯片之间的直接链路。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面结构将任意两个节点之间的最大距离从 $N$ 减少到 $N / 2$，使通信快得多。TPU 还有一个"扭曲环面"配置，以类似莫比乌斯带的拓扑包裹环面，进一步减少节点之间的平均距离。

**TPU Pod（通过 ICI 连接）可以变得非常大：** 最大 Pod 大小（称为 **SuperPod**）对于 TPU v4 是 `16x16x16`，对于 TPU v5p 是 `16x20x28`。这些大型 Pod 由可重新配置的 `4x4x4` 芯片立方体组成，通过[光学环绕链路](https://arxiv.org/pdf/2208.10041)连接<d-footnote>光学交换机只是一个具有相同 ICI 带宽的可重新配置连接。它只是让我们在保留环绕链路的同时连接立方体。</d-footnote>，我们可以重新配置以连接非常大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以请求较小的拓扑（例如 `2x2x1`、`2x2x2`），尽管没有环绕链路。这是一个重要的注意事项，因为它通常会使大多数通信的时间翻倍。任何完整立方体的倍数（例如 `4x4x4` 或 `4x4x8`）都将由光学交换机提供环绕链路。<d-footnote>请注意，`2x2x4` 不会有任何环绕链路，因为它们由光学交换机提供，而光学交换机仅在完整立方体上可用。但是，TPU v5e 8x16 _会_在较长轴上有环绕链路，因为它不使用可重新配置的光学网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trillium Pod 由单个 `16x16` 2D 环面组成，任何大小为 16 的轴都有环绕链路（意味着 `8x16` 在长轴上有环绕链路）。TPU v5e 和 v6e（Trillium）无法扩展到 16x16 环面之外，但 Pod 仍然可以通过标准数据中心网络（DCN）相互通信，DCN 将 TPU 主机相互连接。同样，可以请求没有小于 16 的维度上没有环绕链路的较小拓扑。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是 TPU 和 GPU 之间的关键区别**。GPU 通过交换机层次结构连接，近似于每个 GPU 之间的点对点连接，而不是像 TPU 那样使用本地连接。通常，节点内的 GPU（H100 为 8 个 GPU 或 B200 NVL72 为多达 72 个）是直接连接的，而较大的拓扑需要每个 GPU 之间 O(log(N)) 跳。一方面，这意味着 GPU 可以在少量跳内发送任意数据。另一方面，TPU 的成本大大降低（因为 NVLink 交换机很昂贵），连接更简单，并且可以扩展到更大的拓扑，因为每个设备的链路数量和每个设备的带宽是恒定的。更多信息请阅读[这里](../gpus#networking)。

**ICI 相对于 DCN 非常快，但仍然比 HBM 带宽慢。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 具有：

* 每芯片 `2.5e12` 字节/秒（2.5 TB/s）的 HBM 带宽。
* 每轴 `9e10` 字节/秒（90 GB/s）的 ICI 带宽，每芯片 3 个轴。<d-footnote>上面的页面列出了 100 GB/s 的带宽，这与这里列出的略有不同。TPU ICI 链路根据执行的操作具有略微不同的带宽。你通常可以放心使用本文档中的数字。</d-footnote>
* 每 TPU `6.25e9` 字节/秒（6.25 GB/s）的 DCN（出口）带宽（通过每个主机上的 1-2 个 NIC）。<d-footnote>TPU v6e 有 12.5e9 字节/秒，v5e 有 3.125e9 字节/秒。</d-footnote>

这意味着当我们将模型分割到多个芯片时，我们需要小心避免用较慢的跨设备通信来阻塞 MXU。

**多切片训练：** 一组通过 ICI 连接的 TPU 称为**切片（Slice）**。不同的切片可以使用 DCN 相互连接，例如连接不同 Pod 上的切片。由于 DCN 是比 ICI 慢得多的连接，应该尽量限制我们的计算需要等待 DCN 数据的程度。DCN 是主机到主机的，所以要通过 DCN 将缓冲区从 TPU 传输到 TPU，我们首先需要通过 PCIe 传输到主机，然后通过网络出口，然后通过目标主机网络入口，然后通过 PCIe 进入 HBM。

## 核心要点

* TPU 很简单，在大多数情况下可以被看作是一个矩阵乘法单元连接到内存（超级快）、通过 ICI 连接到其他芯片（相当快）以及通过 DCN 连接到数据中心的其余部分（比较快）。

* 通信受我们各种网络带宽的限制，按速度排序：
  * HBM 带宽：TensorCore 与其关联的 HBM 之间。
  * ICI 带宽：TPU 芯片与其最近的 4 或 6 个邻居之间。
  * PCIe 带宽：CPU 主机与其关联的芯片托盘之间。
  * DCN 带宽：多个 CPU 主机之间，通常是不通过 ICI 连接的主机。

* **在切片内，TPU 只通过 ICI 连接到最近的邻居。** 这意味着切片中远距离芯片之间的 ICI 通信需要首先跳过中间的芯片。

* **权重矩阵需要在两个维度上填充到至少 128**（TPU v6 上为 256）以填满 MXU（实际上，较小的轴会被填充到 128）。

* **较低精度的矩阵乘法往往更快。** 对于支持的代际，TPU 可以比 bfloat16 FLOPs 大约快 2x/4x 地做 int8 或 int4 FLOPs。VPU 操作仍然以 fp32 执行。

* 为了避免 TPU 计算单元的瓶颈，我们需要**确保每个通道上的通信量与其速度成比例**。

### TPU 规格

以下是我们芯片的一些具体数字：

| 型号                                       | Pod 大小 | 主机大小  | HBM 容量/芯片 | HBM 带宽/芯片 (字节/秒) | FLOPs/秒/芯片 (bf16) | FLOPs/秒/芯片 (int8) |
| :----------------------------------------- | :------: | :-------: | :-----------: | :---------------------: | :------------------: | :------------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |     32GB      |         9.0e11          |        1.4e14        |        1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |     32GB      |         1.2e12          |       2.75e14        |       2.75e14        |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |     96GB      |         2.8e12          |       4.59e14        |       9.18e14        |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |     16GB      |         8.1e11          |       1.97e14        |       3.94e14        |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |     32GB      |         1.6e12          |       9.20e14        |       1.84e15        |

主机大小指的是连接到单个主机的 TPU 拓扑（例如 TPU v5e 有一个 CPU 主机连接到 4x2 拓扑的 8 个 TPU）。以下是互连数据：

| 型号        | ICI 带宽/链路 (单向, 字节/秒) | ICI 带宽/链路 (双向, 字节/秒) |
| :---------- | :---------------------------: | :--------------------------: |
| **TPU v3**  |             1e11              |            2e11              |
| **TPU v4p** |            4.5e10             |            9e10              |
| **TPU v5p** |             9e10              |           1.8e11             |
| **TPU v5e** |            4.5e10             |            9e10              |
| **TPU v6e** |             9e10              |           1.8e11             |

我们同时包含单向（单向）带宽和双向（双向）带宽，因为单向带宽更接近硬件实际，但双向带宽更常出现在涉及完整环的方程中。<d-footnote>双向带宽是指沿单个链路在两个方向上可以发送的总字节数，或者等价地，假设我们可以高效地使用两个链路，从单个 TPU 沿特定轴的总出口字节数。当我们在特定轴上有一个功能环（即当我们有环绕连接）时，这是正确的。对于推理芯片，当我们有一个完整的 16 轴时会发生这种情况，对于训练芯片（v*p），当我们有一个是 4 的倍数的轴时会发生这种情况。我们更喜欢使用双向带宽，因为它经常出现在涉及双向通信的计算中。</d-footnote>

PCIe 带宽通常约为每 TPU `1.6e10` 字节/秒（TPU v6e 为 `3.2e10`），而 DCN 带宽通常约为每 TPU `6.25e9` 字节/秒（TPU v6e 为 `12.5e9`，TPU v5e 为 `3.125e9`）。

## 练习题

这些数字有点枯燥，但它们让你可以对模型性能进行基本的 Roofline 估计。让我们做几个问题来解释为什么这很有用。你会在第 3 部分看到更多例子。

**问题 1 [LLM 延迟边界]：** 假设你想从一个分布在 32 个 TPU v4p 上的 2000 亿参数 bf16 模型中采样。从 HBM 将所有参数加载到脉动阵列需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 我们在 32 个芯片上加载 `sizeof(bf16) * 200e9 = 400e9` 字节，意味着每芯片 12.5e9 字节，每个芯片的 HBM 带宽为 1.23e12。所以加载大约需要 10ms。

这很酷，因为*这是从模型采样延迟的合理下界*。每个采样步骤需要从 HBM 加载所有参数，所以不可能少于 10 ms。在实践中，在小批量大小下，这接近可实现的。

{% enddetails %}

**问题 2 [TPU 细节]：** 考虑一个完整的 TPU v5e Pod。总共有多少个 CPU 主机？多少个 TPU TensorCore？整个 Pod 的总 FLOPs/s 是多少？总 HBM 是多少？对 TPU v5p Pod 做同样的练习。

{% details 点击这里查看答案。 %}

**答案：** 对于 TPU v5e，每个 Pod 是 `16x16`，每个主机是 4x2 切片，所以我们有 `16*16 / 8 = 32` 个主机。对于 TPU v5e，每个 TPU 只有一个核心，所以我们有 256 个 TensorCore。总 FLOPs/s 是 `16*16*2e14 = 5.1e16`（bfloat16）。每个芯片有 16GB 的 HBM，所以是 `256 * 16 = 4TB` 的内存。

对于完整的 TPU v5p Pod，我们有 `16x20x28` 个芯片，每个主机是 2x2x1，所以我们有 `16*20*28 / 2*2 = 2,240` 个主机。对于 TPU v5p，每个 TPU 有两个 TensorCore，所以我们有 `8960 * 2 = 17,920` 个核心。总 FLOPs/s 是 `8960 * 4.5e14 = 4e18`（bfloat16）。每个芯片有 96GB 的 HBM，所以是 `8960 * 96 = 860TB` 的内存。

{% enddetails %}

**问题 3 [PCIe 运算强度]：** 假设我们被迫在主机 DRAM 中存储一个大的权重矩阵 $A$（类型 $\text{bfloat16}[D, F]$）和一批激活 $x$（类型 $\text{bfloat16}[B, D]$），并想对它们做矩阵乘法。这在单个主机上运行，我们使用连接到它的单个 TPU v6e 芯片。你可以假设 $B \ll D$，且 $F = 4D$（我们将在未来的章节中看到为什么这些是合理的假设）。我们需要保持 FLOPs 受限所需的最小批量大小 $B$ 是多少？假设 PCIe 带宽为 1.5e10 字节/秒。

{% details 点击这里查看答案。 %}

**答案：** 我们必须执行 $2BDF$ 浮点运算，每个芯片可以执行 `9.2e14` 浮点运算/秒。这需要 $2BDF / 9.2e14$ 秒来执行。我们必须从 DRAM 加载 $2DF + 2BD$ 字节，并写回 $2BF$ 字节。我们受 PCIe 传输速度的瓶颈，所以我们需要 $2 \cdot (BD + DF + BF) / 1.5e10$ 秒来传输数据到 TPU 和从 TPU 传输数据。由于我们希望计算比权重加载花费更长时间，假设我们可以将所有权重加载与计算重叠，我们希望 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$。我们可以使用假设 $B \ll D$ 和 $F = 4D$ 来简化，得到

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.5 \times 10^{10}}$$

或

$$B > \frac{9.2 \times 10^{14}}{1.5 \times 10^{10}} \simeq 61{,}000$$

{% enddetails %}

**问题 4 [一般矩阵乘法延迟]：** 假设我们想将 int8[16384, 4096] 的权重矩阵乘以 int8[B, 4096] 的激活矩阵，其中 B 是某个未知的批量大小。假设我们从 1 个 TPU v5e 开始。

1. 这个乘法作为 B 的函数需要多长时间？*提示：计算从 HBM 加载数组需要多长时间以及乘法实际需要多长时间可能会有帮助。哪个在瓶颈你？*
2. 如果我们想从 VMEM 运行这个操作怎么办？作为 B 的函数需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** (1) 我们需要执行的浮点运算数是 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。所以 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从 HBM 加载到 VMEM $16384 \cdot 4096 + 4096 \cdot B$ 字节，并从 VMEM 写回 HBM $16384 \cdot B$ 字节。这意味着 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.1 \times 10^{11}$ 秒。假设通信和计算尽可能多地重叠，整个乘法大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{ \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}}, \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}} \right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$ 时，我们是 FLOPs 受限的，或等价地，$B > 271$。这略大于我们下面推导的 240 数字，因为我们考虑了 $$D$$ 和 $$F$$ 的完整影响。

(2) 如果我们从 VMEM 加载，让我们考虑 VMEM 到 MXU 的带宽是 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这将我们的数据加载分母从 8.1e11 变为 1.78e13，我们得到 $B > 11$。请注意，在实践中，我们不能将所有 VMEM 带宽都用于加载 $W$，所以实际上会接近 20。

{% enddetails %}

**问题 5 [ICI 带宽]：** 假设我们有一个 TPU v5e `4x4` 切片。假设我们想将类型为 `bfloat16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`。假设 TPU v5e 的每跳延迟是 $1\mu s$。

1. 第一个字节多久会到达目的地？
2. 整个传输需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** 在 TPU v5e 中，我们有 2D 连接。因为我们只有一个 `4x4` 切片（没有大小为 16 的轴），所以我们没有环绕连接。因此，目标芯片可以从两个端口接收数据，源芯片也可以从两个端口发送数据。我们要传输的数据量是 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以同时从两个端口传输（即向右发送一半数组，向下发送一半），所以我们得到 `2 * 4.5e10 = 9e10` 字节/秒的传输速度，这意味着传输整个数组大约需要 `1.7e7 / 9e10 = 188us`（假设我们是带宽受限的）。在 `4x4` 切片中，芯片 $(0, 0)$ 和 $(3, 3)$ 之间有六跳，因为对于少于 16 个芯片的轴没有环绕链路。由于每跳的延迟约为 $1\mu s$，第一个字节将在大约 `6us` 到达，整个传输将需要 `188us`。

{% enddetails %}

**问题 6 [综合练习，困难]：** 假设你有一个大矩阵 **A**: `int8[128 * 1024, 128 * 1024]`，均匀分片在 TPU v5e 4x4 切片上，但卸载到每个芯片的主机 DRAM。假设你想将整个数组复制到 TPU{0, 0} 并将其乘以向量 `bf16[8, 128 * 1024]`。这需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 让我们首先概述我们需要执行的操作。我们的数组约为 16GB。从上表可知，TPU v5e 主机有 4x2 拓扑，所以 4x4 有 2 个主机。因此，由于我们的数组是均匀分片的，每个主机实际上包含数组的 1/2，即 8GB。我们需要将这些块全部复制到 TPU{0,0}，这给我们两个选择：

1. 我们可以通过 DCN 复制，然后通过 PCIe 将整个未分片的数组加载到 HBM。
2. 我们可以将分片数组加载到相应的 TPU 上，然后通过 ICI 执行收集，然后在 TPU{0,0} 上执行矩阵乘法。

应该很清楚，选项 (2) 更好。DCN 比 ICI 慢，我们更希望通过多个 PCIe 链路加载大数组，而不是只通过主机 0 上的 8 个。这是系统的部分示意图。如上所述，注意 TPU 通过 ICI 连接到它们的邻居（即使跨主机），所有 TPU 都连接到它们的主机 CPU（通过 PCIe），主机通过 DCN 连接。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每个芯片实际上都有自己到主机的 PCIe 链路，尽管为清晰起见这里只显示一个。" %}

现在让我们计算每个部分需要多长时间：

1. **PCIe 加载**：我们通过 16 个 PCIe 链路加载 16GB 的块，每个链路有 `1.5e10` 字节/秒的带宽。因此这大约需要 66ms。

2. **ICI 复制**：每个 TPU 现在有我们数组的 16GB / 16 = 1GB。我们的 ICI 带宽是每链路 9e10 字节/秒*双向*，你会从上图注意到，在这个拓扑中，TPU{0,0} 只使用了 TPU v5e 4 个 ICI 链路中的 2 个。由于 TPU{0,0} 需要沿 2 个轴以 `4.5e10` 字节/秒/链路接收总共 15GB，我们可以将时间下界设为 `15e9 / (4.5e10 * 2) = 167ms`。实际上这可能无法实现，因为负载非常不均匀，但可能在 2 倍以内。正如你将在第 2 节中看到的，执行完整的 AllGather 也大约需要 `16e9 / (4.5e10 * 2)`，所以这接近最优。

3. **HBM $\rightarrow$ MXU 加载**：要执行我们的最终矩阵乘法，我们需要将这 16e9 字节加上 bf16[8, 128 \* 1024] 数组（另外 2MB，所以可以忽略不计）通过 HBM 带宽加载到 MXU，这将需要 `16e9 / 8.1e11 = 19ms`。

4. **FLOPs**：我们执行总共 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ FLOPs，由于我们可以执行 `1.97e14` bf16 FLOPs/s，我们得到 1.3ms。

总时间的上界是所有这些时间的总和，但由于 TPU 通常可以重叠这些操作，我们可以将其视为一个由最慢部分瓶颈的流水线问题。假设这是正确的，那么答案至少是 167ms，考虑到不完美的重叠，可能接近 200ms。

{% enddetails %}

<h3 markdown=1 class="next-section">第 2 部分到此结束！关于分区和跨 TPU 通信的第 3 部分，[请点击这里](../sharding)。</h3>

## 附录

### 附录 A：TPU 内部细节

这里我们将更深入地了解 TPU 的内部操作。除非另有说明，我们将提供 TPU v5p 的规格。

### VPU

VPU 是 TPU 的向量算术核心。VPU 由一个二维 SIMD 向量机（**VPU**）组成，执行逐元素算术运算，如 vadd（向量加法）或 vmax（逐元素最大值），以及一组称为 **VREGs** 的向量寄存器，用于保存 VPU 和 MXU 的数据。

**VREGs：** 每个 TPU v5p 核心有 64 个 32 位 VREGs（TPU v4 有 32 个），给我们每核约 `64 * 8 * 128 * 4 = 256kB` 的 VREG 内存（或整个芯片的 2 倍，因为我们有两个核心）。TPU v5p 每周期可以从 VMEM 加载 3 个寄存器，向 VMEM 写入 1 个寄存器。

**VPU：** VPU 是一个形状为 `(8, 128)` 的 2D 向量算术单元，其中 128 维称为 lane 轴，8 维称为 sublane 轴。v5 上的每个 (lane, sublane) 对包含 4 个相互独立的标准浮点 ALU。VPU 在其每个 ALU 中以一个周期执行大多数算术指令（如 vadd 或向量加法），延迟为 2 个周期，所以例如在 v5 中，你可以在每个周期从 VREGs 将 4 对 f32 值相加。一个典型的 VPU 指令可能看起来像 `{v2 = vadd.8x128.f32 v0, v1}`，其中 v0 和 v1 是输入 VREGs，v2 是输出 VREG。

所有 lane 和 sublane 每个周期以纯 SIMD 方式执行相同的程序，但每个 ALU 可以执行不同的操作。所以我们可以例如在一个周期内处理 1 个 vadd 和 1 个 vsub，每个操作两个完整的 VREGs 并将输出写入第三个。

**小测验 [计算 VPU 吞吐量]：** 使用上述信息，计算 TPU v5p 可以执行多少向量 FLOPs/s。TPU v5p 的时钟速度约为 1.75GHz。

{% details 点击这里查看答案。 %}

*答案*：每个周期，每个核心可以在 `8 * 128` 个 ALU 上执行 4 条向量指令。这给我们整个芯片每周期 `8 * 128 * 4 * 2` FLOPs，或 `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`。注意这比约 `2e14` 的 MXU FLOPs/s 小得多（大约 10 倍）。

{% enddetails %}

**归约：** 通常，跨 sublane 维度的通信或归约比跨 lane 维度更容易。例如，VPU 支持一个 lane 内的 shuffle 操作，可以在大约一个周期内沿大小为 8 的轴滚动。这可以用于沿 sublane 维度执行高效的归约（只需 shuffle 4、2、1 并做 3 对逐元素求和）。

跨 lane 的归约困难得多，涉及一个称为 XLU 或"跨 lane 单元"的独立硬件单元，它很慢且相当昂贵。

**与 GPU 的比较：** 对于熟悉 NVIDIA GPU 的人，VPU 中的每个 ALU 类似于 CUDA 核心，单个 VPU lane 类似于"Warp 调度器"，即通常执行 SIMD 算术的 32 个 CUDA 核心的集合。lane 内的归约相当容易，但如果我们需要跨 lane，我们至少需要经过 VMEM/XLU/SMEM，这要慢得多。更多细节请参见 [GPU 章节](../gpus)。

### 标量核心

标量核心是 TPU 的控制单元。它获取和分发所有指令，执行从 HBM 到 VMEM 的传输，并可以编程来执行标量元数据工作。因为标量核心是单线程的，一个副作用是 TPU 的每个核心每周期只能创建一个 DMA 请求。

从上下文来看，一个标量核心控制一个 VPU（由 4096 个 ALU 组成）、4 个 MXU、2 个 XLU 和多个 DMA 引擎。这种每单位计算的控制高度倾斜是硬件效率的来源，但也限制了以任何有趣的方式进行数据依赖向量化的能力。

### 附录 B：脉动阵列如何工作？

TPU MXU 的核心是一个 `128x128` 脉动阵列（TPU v6e 上是 `256x256`）。当完全饱和时，脉动阵列每 8 个时钟周期可以执行一次 `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>如果你不熟悉这个符号，它的意思是：将一个 `8x128` 的 bfloat16 元素矩阵乘以一个 `128x128` 的 bfloat16 元素矩阵，并将结果存储在一个 `8x128` 的 float32 元素矩阵中。</d-footnote> 乘法。

* 其核心是一个 2D `128x128`（=16,384）ALU 网格，每个 ALU 能够执行乘加运算。
* 权重（**W**，`128x128` 输入）从上方传入（称为 RHS），而输入（**X**，`8x128` 输入）从左边传入（称为 LHS）。

这是一个将一组权重（蓝色）与一组激活（绿色）相乘的简化动画。你会注意到权重（RHS）首先对角加载，然后激活也对角馈入。在下面的每一帧中，我们将所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的任何残差相加，然后依次将结果向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是一个更通用的动画版本，展示了输出如何从计算中流出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是一个图示，展示了如何跨多个 RHS 和 LHS 数组进行流水线化：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

当权重（RHS）和激活（LHS）被加载时，有一个初始的流水线气泡。在那个初始气泡之后，可以加载新的输入和权重而不产生额外的气泡。

这是一个 bf16[2, 3] x bf16[3, 3] 矩阵乘法的拙劣动画，你可以想象成一个 2x3 权重矩阵与批量大小为 1、大小为 3 的输入激活的矩阵乘法。这与前面的幻灯片相比是旋转的，输入向右流出而不是向下，但你可以大致看到结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以高效地将其流水线化以乘以大矩阵，而不会产生太大的流水线气泡。话虽如此，重要的是我们的矩阵形状要大于 MXU 的边长，通常是 128x128。一些 TPU（从 TPU v3 开始）有多个 MXU，TPU v3 为 2 个，TPU v4/5 为 4 个，所以我们需要确保分块维度大于 128 * MXU 数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的动画。

Trillium（TPU v6e）有一个 `256x256` 脉动阵列，这意味着它每周期可以执行 4 倍的 FLOPs。这也意味着你的张量维度需要是原来的两倍才能充分利用 MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)有另一个关于固定权重矩阵脉动阵列乘法的优秀动画。
