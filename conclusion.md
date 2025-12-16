---
layout: distill
title: "结论与延伸阅读"
# permalink: /main/
description: "感谢阅读！这里我们将包含一些进一步学习的参考资料。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "第10部分：JAX"

next_section_url: "../gpus"
next_section_name: "第12部分：GPU"

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
  - name: "致谢"
  - name: "延伸阅读"
  - name: "反馈"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---
**感谢阅读这组文章，恭喜你一直读到了最后。** 在我们结束之前，有一些致谢：

## 致谢

本文档代表了 Google DeepMind 许多人的重大集体投入，我们想简要致谢！

* James Bradbury、Reiner Pope 和 Blake Hechtman 最初推导了本手稿中的许多想法，并且很早就理解了 Transformer 的系统视角。
* Sholto Douglas 编写了本文档的第一版，并负责启动了这个项目。他比任何人都更负责本文档的整体叙事。
* Jacob Austin 领导了将这个第一版从粗略笔记转变为更精炼和全面的成品的工作。他做了大量的编辑、格式化和发布这份文档的工作，并协调了其他作者的贡献。
* 大部分图形和动画由 Anselm Levskaya 和 Charlie Chen 制作。
* Charlie Chen 编写了推理部分并绘制了许多推理图形。
* Roy Frostig 帮助完成了出版、编辑和旅程中的许多其他步骤。

我们还要感谢许多在整个过程中提供关键反馈的其他人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 等人。感谢 Ruiqi Gao 帮助完成 HTML 格式化。

**感谢大家！**

<p markdown=1 class="announce">在你离开之前，你可能也会喜欢阅读关于 NVIDIA GPU 的新[第12章](../gpus)！</p>

## 延伸阅读

有很多相关的文章，包括以下内容：

* [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html)：一个精彩的深入探讨 TPU 架构的文章，与本书精神相同。
* [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures)：一个硬件和模型深度解析，与本书精神相同。
* [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307)：原始的 TPU 论文之一，其中有很多关于 Google TPU 项目的精彩细节，本书未涵盖。
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：一个更关注 GPU 和 PyTorch 的 LLM roofline 和性能工程教程。
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：越来越多地，TPU 编程涉及用 Pallas 编写自定义内核。本系列讨论如何编写内核以及这里没有提到的许多低级 TPU 细节。
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：虽然是 GPU 和 CUDA 特定的，这是一篇优秀的博客文章，展示了如何在 CUDA 中优化 matmul 内核。这可能是深入了解 TPU 和 GPU 差异的好方式。
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是 JAX 中并行化 API 的一个非常好的指南，是学习如何实际实现我们这里讨论的一些想法的好方法。
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事 Rafi 开设了一个关于 TPU 性能工程的优秀课程，幻灯片都在 GitHub 上。这比我们这里更深入地涵盖了很多内容。
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：一篇关于 Transformer 推理数学的详细论文。这是本文档很多内容的灵感来源。
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：某种程度上是本书的 GPU 对应物，更深入地讨论 PyTorch 如何在训练期间实现并行化技术和内存节省技术。
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一个包含本书许多相同想法和一些优秀插图的博客。
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：一个涵盖 LLM 训练和服务许多细节的精彩斯坦福课程，有一些有用的练习。作业 1 和 2 特别相关。
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：一个高度实用的 ML 基础设施指南，涵盖本书未涉及的主题，如如何与云提供商谈判、集群管理和 GPU 吞吐量的经验测量。

这个领域仍然有很大的空间进行全面的写作，所以
我们希望这份手稿能鼓励更多这样的工作！我们也相信
这是一个值得研究和探索的富有成效的领域。在许多情况下，即使
手头没有很多硬件加速器也可以进行这项工作。

## 反馈

请留下评论或问题，以便我们进一步改进。你可以通过以下方式联系我们的通讯作者 Jacob Austin：
jacobaustin123 [at] gmail [dot] com，或者在 [GitHub](https://github.com/jax-ml/scaling-book) 上发布 issue、pull request 或讨论来建议编辑。
