---
layout: distill
title: "如何扩展你的模型"
subtitle: "大语言模型在 TPU 上的系统级视角"
# permalink: /main/
description: "训练大语言模型(LLM)常常让人感觉像是炼金术，但理解和优化模型性能并不必如此神秘。本书旨在揭开语言模型扩展的科学面纱：TPU（和GPU）如何工作、它们如何相互通信、LLM如何在真实硬件上运行，以及如何在训练和推理时并行化你的模型以实现大规模高效运行。如果你曾经想过『训练这个LLM应该花多少钱』、『我需要多少内存来自己部署这个模型』或者『什么是AllGather』，我们希望这本书对你有所帮助。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "第0部分：导论"

next_section_url: roofline
next_section_name: "第1部分：Roofline模型"

bibliography: main.bib

citation: true

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
  - name: 内容概览
  - name: 章节链接

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

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习的大部分内容仍然像某种"黑魔法"，但优化模型性能并不一定如此——即使是在超大规模下也是如此！相对简单的原理适用于所有场景——从单个加速器到数万个加速器——理解这些原理能让你做很多有用的事情：

- 大致估算模型各部分与理论最优值的差距
- 在不同规模下对不同的并行化方案做出明智选择（即如何将计算分配到多个设备上）
- 估算训练和运行大型 Transformer 模型所需的成本和时间
- 设计能够充分利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[优势](https://arxiv.org/abs/2007.00072)的算法
- 基于对当前算法性能限制因素的明确理解来设计硬件

**预期背景知识：** 我们假设你对大语言模型(LLM)和 Transformer 架构有基本了解，但不一定了解它们如何在大规模下运行。你应该知道 LLM 训练的基础知识，最好对 JAX 有一些基本熟悉。一些有用的背景阅读可能包括[这篇关于 Transformer 架构的博客文章](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。另外请查看[这个列表](conclusion#further-reading)获取更多有用的同步和进阶阅读资料。

**目标与反馈：** 读完本书后，你应该能够自如地为给定硬件平台上的 Transformer 模型估算最佳并行化方案，并大致估算训练和推理需要多长时间。如果做不到，请给我们发邮件或留言！我们很想知道如何能让内容更清晰。

<p markdown=1 class="announce">你可能还会喜欢阅读关于 NVIDIA GPU 的新[第12章](gpus)！</p>

### 为什么你应该关心这些？

三四年前，我认为大多数机器学习研究人员不需要理解本书中的任何内容。但今天，即使是"小型"模型的运行也如此接近硬件极限，以至于进行新颖的研究需要你在规模上考虑效率。<d-footnote>历史上，机器学习研究遵循着系统创新和软件改进之间的"钟摆"周期。Alex Krizhevsky 必须编写诡异的 CUDA 代码才能让 CNN 变快，但几年内，Theano 和 TensorFlow 等库意味着你不必这样做了。也许这里也会发生同样的事情，几年后本书中的所有内容都会被抽象掉。但 Scaling Law（规模定律）已经将我们的模型持续推向硬件的最前沿，在可预见的未来，进行前沿研究似乎将与理解如何高效地将模型扩展到大型硬件拓扑紧密相连。</d-footnote> **如果基准测试上 20% 的提升是以 20% 的 Roofline 效率损失为代价的，那就毫无意义。** 有前途的模型架构经常失败，要么是因为它们_无法_在大规模下高效运行，要么是因为没有人花功夫让它们做到这一点。

**"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现吞吐量成比例的线性增长。** 这被称为"*强扩展*（Strong Scaling）"。虽然添加额外的芯片（"并行化"）通常会减少计算时间，但它也带来了芯片间通信增加的代价。当通信时间超过计算时间时，我们就会变成"通信受限"，无法实现强扩展。<d-footnote>随着计算时间的减少，你通常还会在单芯片层面面临瓶颈。你闪亮的新 TPU 或 GPU 可能标称能够每秒执行 500 万亿次运算，但如果你不小心，当它陷入内存中移动参数的困境时，它也可能只能做到十分之一。单芯片计算、内存带宽和总内存的相互作用对扩展至关重要。</d-footnote> 如果我们对硬件足够了解，能够预见这些瓶颈在哪里出现，我们就可以设计或重新配置模型来避免它们。<d-footnote>硬件设计师面临相反的问题：构建为我们的算法提供足够计算、带宽和内存的硬件，同时最小化成本。你可以想象这个"协同设计"问题有多紧张：你必须押注第一批芯片实际可用时算法会是什么样子，通常是 2 到 3 年后的事。TPU 的故事是这场博弈中的巨大成功。矩阵乘法是一种独特的算法，它使用的每字节 FLOPs 远多于几乎任何其他算法（每字节 N 次 FLOPs），早期的 TPU 及其脉动阵列架构实现了比当时构建的 GPU 更好的性价比。TPU 是为机器学习工作负载设计的，GPU 及其 TensorCore 也在迅速改变以填补这一市场。但你可以想象，如果神经网络没有起飞，或者以某种 TPU（本质上比 GPU 灵活性更低）无法处理的根本性方式发生变化，代价会有多大。</d-footnote>

*本书的目标是解释 TPU（和 GPU）硬件如何工作，以及 Transformer 架构如何演进以在当前硬件上表现良好。我们希望这对设计新架构的研究人员和致力于让当前一代 LLM 快速运行的工程师都有用。*

## 内容概览

本书的整体结构如下：

[第1章](roofline)解释 Roofline 分析以及哪些因素可能限制我们的扩展能力（通信、计算和内存）。[第2章](tpus)和[第3章](sharding)详细讨论 TPU 如何工作，既作为单个芯片，也——这一点至关重要——作为具有有限带宽和延迟的芯片间链路的互连系统。我们将回答以下问题：

* 一定大小的矩阵乘法应该需要多长时间？在什么情况下它受计算、内存或通信带宽限制？
* TPU 如何连接在一起形成训练集群？系统的每个部分有多少带宽？
* 跨多个 TPU 收集、分散或重新分配数组需要多长时间？
* 如何高效地乘以在设备间以不同方式分布的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图示：</b> <a href='tpus'>第2章</a>中的图示展示了 TPU 如何执行逐元素乘积。根据数组大小和各种链路的带宽，我们可能处于计算受限状态（充分利用硬件计算能力）或通信受限状态（受内存加载瓶颈限制）。" %}

五年前，机器学习有着丰富多彩的架构景观——卷积网络、LSTM、MLP、Transformer——但现在我们主要只有 Transformer<d-cite key="transformers"></d-cite>。我们坚信理解 Transformer 架构的每一个部分是值得的：每个矩阵的精确大小、归一化发生在哪里、每个部分有多少参数和 FLOPs<d-footnote>浮点运算数（Floating point OPs），基本上是所需的加法和乘法的总数。虽然许多资料将 FLOPs 理解为"每秒运算数"，我们使用 FLOPs/s 来明确表示这一点。</d-footnote>。[第4章](transformers)仔细地讲解了这些"Transformer 数学"，展示了如何计算训练和推理的参数和 FLOPs。这告诉我们模型将使用多少内存、我们在计算或通信上将花费多少时间，以及注意力相对于前馈块何时变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图示：</b> 标准 Transformer 层，每个矩阵乘法（matmul）显示为圆圈内的点。所有参数（不包括归一化）以紫色显示。<a href='transformers'>第4章</a>更详细地讲解了这个图示。" %}

[第5章：训练](training)和[第7章：推理](inference)是本书的核心，我们在这里讨论基本问题：给定某个大小的模型和一定数量的芯片，我如何并行化我的模型以保持在"强扩展"模式下？这是一个简单的问题，但答案出奇地复杂。在高层面上，有 4 种主要的并行化技术用于将模型分割到多个芯片上（**数据并行**、**张量并行**、**流水线并行**和**专家并行**），以及一些其他技术来减少内存需求（**重计算/重物化**、**优化器/模型分片（又称 ZeRO）**、**主机卸载**、**梯度累积**）。我们在这里讨论其中许多技术。

我们希望在读完这些章节后，你应该能够自己为新架构或设置选择合适的并行化方案。[第6章](applied-training)和[第8章](applied-inference)是将这些概念应用于 LLaMA-3（一个流行的开源模型）的实践教程。

最后，[第9章](profiling)和[第10章](jax-stuff)探讨了如何在 JAX 中实现这些想法，以及当出现问题时如何分析和调试你的代码。[第12章](gpus)是一个深入介绍 GPU 的新章节。

在全书中，我们尽量给你提供可以自己练习的问题。请不要有压力非得阅读所有章节或按顺序阅读。请留下反馈。目前，这是一份草稿，将继续修订。谢谢！

*我们要感谢 James Bradbury 和 Blake Hechtman，他们推导出了本书中的许多想法。*

<h3 markdown=1 class="next-section">话不多说，[这是关于 TPU Roofline 模型的第1章](roofline)。</h3>

## 章节链接

*本系列可能比需要的更长，但我们希望这不会阻止你阅读。前三章是预备知识，如果你已经熟悉可以跳过，尽管它们引入了后面使用的符号。最后三个部分可能是最实用的，因为它们解释了如何处理真实模型。*

**第一部分：预备知识**

* [**第1章：Roofline 分析简介**](roofline)。算法受三个因素限制：计算、通信和内存。我们可以使用这些来估算算法的运行速度。

* [**第2章：如何理解 TPU**](tpus)。TPU 如何工作？这如何影响我们可以训练和服务的模型？

* [**第3章：分片矩阵及其乘法**](sharding)。在这里，我们通过我们最喜欢的操作来解释模型分片和多 TPU 并行：（分片）矩阵乘法。

**第二部分：Transformer**

* [**第4章：你需要知道的所有 Transformer 数学**](transformers)。Transformer 在前向和反向传播中使用多少 FLOPs？你能计算参数数量吗？KV 缓存的大小？我们在这里详细推导这些数学。

* [**第5章：如何并行化 Transformer 进行训练**](training)。FSDP、Megatron 分片、流水线并行。给定一定数量的芯片，我如何尽可能高效地训练给定大小和批次大小的模型？

* [**第6章：在 TPU 上训练 LLaMA 3**](applied-training)。我们如何在 TPU 上训练 LLaMA 3？需要多长时间？成本是多少？

* [**第7章：Transformer 推理详解**](inference)。训练完模型后，我们需要部署它。推理增加了一个新的考虑因素——延迟——并改变了内存布局。我们将讨论分离式服务如何工作以及如何考虑 KV 缓存。

* [**第8章：在 TPU 上部署 LLaMA 3**](applied-inference)。在 TPU v5e 上部署 LLaMA 3 需要多少成本？延迟/吞吐量的权衡是什么？

**第三部分：实践教程**

* [**第9章：如何分析 TPU 代码性能**](profiling)。真实的 LLM 从来不像上面的理论那么简单。在这里，我们解释 JAX + XLA 技术栈以及如何使用 JAX/TensorBoard 分析器来调试和修复实际问题。

* [**第10章：在 JAX 中编程 TPU**](jax-stuff)。JAX 提供了一系列神奇的 API 用于并行化计算，但你需要知道如何使用它们。有趣的示例和练习问题。

**第四部分：总结与附加内容**

* [**第11章：总结与进阶阅读**](conclusion)。关于 TPU 和 LLM 的结束语和进阶阅读资料。

* [**第12章：如何理解 GPU**](gpus)。关于 GPU 的附加章节，介绍它们如何工作、如何联网，以及它们的 Roofline 与 TPU 有何不同。
