---
layout: distill
title: "如何扩展你的模型"
subtitle: "大语言模型在 TPU 上的系统级视角"
# permalink: /main/
description: "训练 LLM 常常被说成是『炼丹』，但搞懂模型性能优化其实没那么玄乎。本书想把 LLM 扩展这件事讲明白：TPU 和 GPU 到底怎么干活的、芯片之间怎么通信、LLM 在真实硬件上是怎么跑的，以及怎么把模型拆分到多个芯片上高效运行。如果你曾经想过『训练这个模型要花多少钱』、『部署需要多少显存』、『AllGather 是什么鬼』，希望这本书能帮到你。"
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

toc:
  - name: 这本书讲什么
  - name: 章节导航

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

深度学习里有很多"黑魔法"，但**模型性能优化**不用那么玄——哪怕是超大规模也一样！其实背后的原理挺简单的，而且从单卡到上万张卡都适用。搞明白这些，你就能：

- **估算性能差距**：你的模型离理论最优还差多远？
- **选对并行方案**：不同规模下，怎么把计算合理地分到多张卡上？
- **预估成本和时间**：训练或部署一个大模型要花多少钱、跑多久？
- **设计更好的算法**：针对[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特点](https://arxiv.org/abs/2007.00072)做优化
- **理解硬件设计**：为什么硬件要这样设计？瓶颈在哪里？

**需要什么基础？** 你得大概知道 LLM 和 Transformer 是啥，但不需要了解它们怎么在大规模下运行。最好对 JAX 有点了解（不是必须）。想补课的话，可以看看[这篇图解 Transformer](https://jalammar.github.io/illustrated-transformer/) 和[原始论文](https://arxiv.org/abs/1706.03762)。更多资料见[延伸阅读](conclusion#further-reading)。

**读完能学到什么？** 你应该能为任意模型和硬件组合选出合适的并行策略，并估算训练和推理的耗时。如果做不到，欢迎给我们留言反馈！

<p markdown=1 class="announce">新增了 NVIDIA GPU 专题，见[第12章](gpus)！</p>

### 为什么要学这些？

三四年前，做机器学习研究可能不太需要懂这些底层的东西。但现在不一样了——**即使是"小"模型，也已经逼近硬件极限了**。<d-footnote>历史上，机器学习研究在"硬核系统优化"和"易用框架封装"之间来回摆。当年 Alex Krizhevsky 得手写 CUDA 才能把 CNN 跑快，但几年后 TensorFlow、PyTorch 这些框架出来后，大家就不用管底层了。也许将来这些知识也会被封装掉。但 Scaling Law 让模型越来越大，前沿研究和高效扩展已经分不开了。</d-footnote>

**如果你在 benchmark 上提升了 20%，但效率损失了 20%，那这个提升就是假的。** 很多看起来很酷的新架构最后没火起来，要么是因为跑不快，要么是没人愿意花功夫把它优化好。

**模型扩展的终极目标：加芯片的同时，吞吐量也线性增长。** 这叫"强扩展"（Strong Scaling）。加卡确实能加速计算，但也带来了通信开销。当通信时间超过计算时间，我们就"卡在通信上"了，再加卡也没用。<d-footnote>随着计算时间变短，单卡上也会出现新瓶颈。你的新 TPU/GPU 标称 500 TFLOPS，但如果整天在搬运数据，实际可能只有十分之一。单卡的计算能力、内存带宽、总内存三者的配合，对扩展至关重要。</d-footnote>

如果我们足够了解硬件，就能提前预判瓶颈在哪，从而调整模型避开它们。<d-footnote>硬件设计师面临相反的问题：要在算力、带宽、内存之间找平衡，同时控制成本。这是个"协同设计"的博弈：你得赌两三年后算法会是什么样。TPU 就是这场博弈的一个成功案例。矩阵乘法每搬运一字节数据就能做 N 次运算，非常划算。早期 TPU 的脉动阵列架构比同期 GPU 性价比更高。GPU 后来也加了 TensorCore 来追赶。但你可以想象，如果神经网络没火起来，或者往 TPU 不擅长的方向发展，那赌输了代价有多大。</d-footnote>

*本书的目标：解释 TPU（和 GPU）怎么工作，以及 Transformer 怎么演进来适配当前硬件。希望对设计新架构的研究者和优化现有模型的工程师都有用。*

## 这本书讲什么

整体结构：

[第1章](roofline)讲 **Roofline 分析**——哪些因素在限制你的扩展（通信、计算、内存）。[第2章](tpus)和[第3章](sharding)深入讲 TPU：单芯片怎么工作，多芯片怎么连接，芯片间的带宽和延迟是多少。我们会回答这些问题：

* 一个矩阵乘法应该要多长时间？什么时候受计算限制、什么时候受内存限制、什么时候受通信限制？
* TPU 之间怎么连接成训练集群？每个部分有多少带宽？
* 跨多张 TPU 收集、分散、重新分布数据要多久？
* 怎么高效地乘两个分布在不同设备上的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>动图演示：</b> <a href='tpus'>第2章</a>中会讲 TPU 如何做逐元素乘法。根据数组大小和带宽，我们可能是计算受限（充分利用算力）或通信受限（被数据搬运拖慢）。" %}

五年前，机器学习的模型架构百花齐放——CNN、LSTM、MLP、Transformer 都有市场。但现在基本就剩 Transformer 了<d-cite key="transformers"></d-cite>。我们觉得有必要彻底搞懂 Transformer 的每个细节：每个矩阵多大、归一化在哪里、参数和 FLOPs<d-footnote>浮点运算数（Floating point OPs），就是加法和乘法的总数。很多资料把 FLOPs 说成"每秒运算数"，我们用 FLOPs/s 来明确表示后者。</d-footnote>怎么算。[第4章](transformers)会细细拆解这些"Transformer 数学"，教你算训练和推理的参数量、FLOPs。这能告诉你模型吃多少内存、计算和通信各花多少时间、注意力和 FFN 哪个更重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图示：</b> 标准 Transformer 层。圆圈里的点表示矩阵乘法（matmul），紫色是参数（不含归一化）。<a href='transformers'>第4章</a>会详细解释。" %}

[第5章：训练](training)和[第7章：推理](inference)是本书的重头戏，讨论核心问题：**给定模型大小和芯片数量，怎么并行化才能保持强扩展？** 问题简单，答案却出乎意料地复杂。高层来看，有四种主要的并行策略来把模型拆分到多张卡上：

- **数据并行**（Data Parallel）
- **张量并行**（Tensor Parallel）
- **流水线并行**（Pipeline Parallel）
- **专家并行**（Expert Parallel）

还有一些节省内存的技巧：**重计算**、**ZeRO 优化器分片**、**卸载到主机内存**、**梯度累积**。这些我们都会讲。

读完这些章节，你应该能为新架构或新场景自己选出合适的并行方案。[第6章](applied-training)和[第8章](applied-inference)是实战教程，把这些概念应用到 LLaMA-3（一个流行的开源模型）上。

最后，[第9章](profiling)和[第10章](jax-stuff)讲怎么用 JAX 实现这些想法，以及出问题时怎么调试。[第12章](gpus)是新增的 GPU 专题。

全书穿插了练习题，可以动手试试。不用从头读到尾，挑感兴趣的看就行。欢迎留下反馈！这本书还在持续更新中。

*感谢 James Bradbury 和 Blake Hechtman，书中很多想法源自他们。*

<h3 markdown=1 class="next-section">话不多说，开始看[第1章：Roofline 模型](roofline)吧。</h3>

## 章节导航

*本书可能比需要的更长，但别被吓到。前三章是预备知识，熟悉的话可以跳过（但会引入后面用到的符号）。最后几章最实用，讲怎么处理真实模型。*

**第一部分：基础知识**

* [**第1章：Roofline 分析入门**](roofline)——算法受三个因素限制：计算、通信、内存。用这些可以估算运行速度。

* [**第2章：TPU 是怎么工作的**](tpus)——TPU 的内部原理，以及这对我们能训练什么模型有什么影响。

* [**第3章：分片矩阵与矩阵乘法**](sharding)——通过矩阵乘法这个最重要的操作来讲解模型分片和多卡并行。

**第二部分：Transformer 详解**

* [**第4章：Transformer 数学全解**](transformers)——前向和反向各用多少 FLOPs？参数量怎么算？KV 缓存多大？这里详细推导。

* [**第5章：Transformer 训练并行化**](training)——FSDP、Megatron 分片、流水线并行。给定芯片数量，怎么高效训练指定大小和批次的模型？

* [**第6章：在 TPU 上训练 LLaMA 3**](applied-training)——实战：怎么在 TPU 上训练 LLaMA 3？要多久？花多少钱？

* [**第7章：Transformer 推理详解**](inference)——训练完还得部署。推理多了一个要考虑的因素：延迟。我们会讲分离式服务和 KV 缓存。

* [**第8章：在 TPU 上部署 LLaMA 3**](applied-inference)——用 TPU v5e 部署 LLaMA 3 要多少钱？延迟和吞吐量怎么权衡？

**第三部分：动手实践**

* [**第9章：TPU 代码性能分析**](profiling)——真实的 LLM 没那么理想化。这里讲 JAX + XLA 技术栈，以及怎么用分析器找问题。

* [**第10章：用 JAX 编程 TPU**](jax-stuff)——JAX 提供了一系列 API 用于并行化，这里教你怎么用。含趣味示例和练习题。

**第四部分：总结与扩展**

* [**第11章：总结与延伸阅读**](conclusion)——收尾和更多参考资料。

* [**第12章：GPU 是怎么工作的**](gpus)——GPU 专题：内部原理、网络连接、Roofline 与 TPU 的对比。
