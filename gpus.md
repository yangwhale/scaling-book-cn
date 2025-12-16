
---
layout: distill
title: "如何理解 GPU"
description: "我们在 Google 很喜欢 TPU，但 GPU 也很棒。本章深入探讨 GPU 的世界——每个芯片如何工作，它们如何联网在一起，以及这对 LLM 意味着什么，特别是与 TPU 相比。虽然有来自 NVIDIA、AMD、Intel 等的众多 GPU 架构，但这里我们将专注于 NVIDIA GPU。本节建立在<a href='https://jax-ml.github.io/scaling-book/tpus/'>第2章</a>和<a href='https://jax-ml.github.io/scaling-book/training'>第5章</a>的基础上，建议你先阅读它们。"
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "第11部分：结论"

next_section_url:
next_section_name: "全文完"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 什么是 GPU？
  - subsections:
    - name: 内存
    - name: "GPU 规格概览"
    - name: GPU 与 TPU 在芯片级别的对比
    - name: "测验 1：GPU 硬件"
  - name: 网络
  - subsections:
    - name: 节点级别
    - name: "测验 2：GPU 节点"
    - name: 超越节点级别
    - name: "测验 3：超越节点级别"
  - name: GPU 上的集合操作如何工作？
  - subsections:
    - name: 节点内集合操作
    - name: 跨节点集合操作
    - name: "测验 4：集合操作"
  - name: "GPU 上 LLM 扩展的 Roofline"
  - subsections:
    - name: "数据并行"
    - name: "张量并行"
    - name: "专家并行"
    - name: "流水线并行"
    - name: "示例"
    - name: "GPU 上 LLM 扩展总结"
    - name: "测验 5：LLM roofline"
  - name: "致谢与延伸阅读"
  - name: "附录"
  - subsections:
    - name: "附录 A：GB200 有什么变化？"
    - name: "附录 B：更多网络细节"

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

## 什么是 GPU？

现代 ML GPU（例如 H100、B200）基本上是一堆专门做矩阵乘法的计算核心（称为**流式多处理器**或 **SM**）连接到一条快速内存（称为 **HBM**）。这是一个图示：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图：</b>显示 H100 或 B200 GPU 抽象布局的图示。H100 有 132 个 SM，而 B200 有 148 个。我们广义地使用"Warp 调度器"一词来描述一组 32 个 CUDA SIMD 核心<i>和</i>向它们分派工作的调度器。注意这看起来多像 TPU！" %}

每个 SM，像 TPU 的 Tensor Core 一样，有一个专用的矩阵乘法核心（不幸的是也叫**Tensor Core**<d-footnote>GPU 的 Tensor Core 是 SM 的矩阵乘法子单元，而 TPU 的 TensorCore 是包含 MXU、VPU 和其他组件的总括单元。</d-footnote>），一个向量算术单元（称为 **Warp 调度器**<d-footnote>NVIDIA 没有一个好名字来称呼这个，所以我们只是在几个糟糕的选项中选择了最好的。Warp 调度器主要是向一组 CUDA 核心分派工作的单元，但我们在这里用它来描述控制单元和它控制的那组核心。</d-footnote>），以及一个快速片上缓存（称为 **SMEM**）。与 TPU 最多有 2 个独立的"Tensor Core"不同，现代 GPU 有超过 100 个 SM（H100 上有 132 个）。这些 SM 中的每一个都比 TPU Tensor Core 弱得多，但整个系统更灵活。每个 SM 或多或少是完全独立的，所以 GPU 可以同时做数百个不同的任务。<d-footnote>尽管 SM 是独立的，但它们通常被迫协调以获得峰值性能，因为它们都共享一个容量有限的 L2 缓存。</d-footnote>

让我们更详细地看一下 H100 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图：</b>H100 SM 的图示（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>）显示了 4 个<i>子分区</i>，每个包含一个 Tensor Core、Warp 调度器、寄存器文件和不同精度的 CUDA 核心组。底部附近的"L1 数据缓存"是 256kB SMEM 单元。B200 看起来类似，但增加了大量的张量内存（TMEM）来馈送笨重的 Tensor Core。" %}

每个 SM 被分成 4 个相同的象限，NVIDIA 称之为 **SM 子分区**，每个包含一个 Tensor Core、16k 个 32 位寄存器，以及一个 SIMD/SIMT 向量算术单元叫做 Warp 调度器，其通道（ALU）NVIDIA 称之为 **CUDA 核心**。每个分区的核心组件可以说是 Tensor Core，它执行矩阵乘法并构成其绝大部分 FLOPs/s，但它不是唯一值得注意的组件。

* **CUDA 核心：** 每个子分区包含一组叫做 CUDA 核心的 ALU，做 SIMD/SIMT 向量算术。每个 ALU 通常每个周期可以做 1 个算术操作，例如 f32.add。<d-footnote>较新的 GPU 支持 FMA（融合乘加）指令，技术上每个周期做两个 FLOPs，NVIDIA 无情地利用这一点来使其报告的规格翻倍。</d-footnote> 每个子分区包含 32 个 fp32 核心（以及较少数量的 int32 和 fp64 核心），它们在每个周期都执行相同的指令。像 TPU 的 VPU 一样，CUDA 核心负责 ReLU、逐点向量操作和归约（求和）。<d-footnote>历史上，在引入 Tensor Core 之前，CUDA 核心是 GPU 的主要组件，用于渲染，包括光线-三角形相交和着色。在今天的游戏 GPU 上，它们仍然做大部分渲染工作，而 TensorCore 用于上采样（DLSS），这允许 GPU 以较低分辨率渲染（更少像素 = 更少工作）并使用 ML 上采样。</d-footnote>

* **Tensor Core (TC)：** 每个子分区有自己的 Tensor Core，这是一个专用矩阵乘法单元，像 TPU MXU。Tensor Core 代表 GPU 的绝大部分 FLOPs/s（例如在 H100 上，我们有 990 bf16 TC TFLOPs/s，而 CUDA 核心只有 66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 132 个 SM 运行在 1.76GHz 意味着每个 H100 TC 每周期可以做 `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs，大约是 8x8x8 matmul。<d-footnote>NVIDIA 不分享很多 TC 硬件细节，所以这更多是猜测而不是确定的事实——当然，它没有说明 TC 是如何实现的。我们知道 V100 每个 TC 每周期可以执行 256 FLOPs。A100 可以做 512，H100 可以做 1024，虽然 B200 的细节没有公布，但看起来很可能是每 TC 每周期约 2048 FLOPs，因为 `2250e12 / (148 * 4 * 1.86e9)` 约是 2048。更多细节在<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>这里</a>确认。</d-footnote>
  * 像 TPU 一样，GPU 可以以更高的吞吐量做更低精度的 matmul（例如 H100 有 2x fp8 FLOPs/s vs. fp16）。低精度训练或服务可以显著更快。
  * 自 Volta 以来每一代 GPU 都增加了 TC 大小（[这里有一篇好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。随着 B200，TC 变得如此大以至于它不能再把输入放在 SMEM 中，所以 B200 引入了一个叫 TMEM 的新内存空间。<d-footnote>在 Ampere 中，Tensor Core 可以从单个 warp 馈送，而在 Hopper 中它需要完整的 SM（warpgroup），在 Blackwell 中它从 2 个 SM 馈送。matmul 在 Blackwell 中也变得如此大以至于参数（特别是累加器）不再适合寄存器内存/SMEM，所以 Blackwell 添加了 TMEM 来解决这个问题。</d-footnote>

**CUDA 核心比 TPU 的 VPU 更灵活：** GPU CUDA 核心（自 V100 以来）使用所谓的 SIMT（*单指令多线程*）编程模型，相比 TPU 的 SIMD（*单指令多数据*）模型。像 TPU VPU 中的 ALU 一样，子分区内的 CUDA 核心必须在每个周期执行相同的操作（例如如果一个核心在加两个浮点数，那么子分区中的每个其他 CUDA 核心也必须这样做）。然而，与 VPU 不同，每个 CUDA 核心（或 CUDA 编程模型中的"线程"）有自己的指令指针，可以_独立编程_。当同一个 warp 中的两个线程被指示执行不同的操作时，你实际上_同时_做两个操作，掩盖不需要执行分歧操作的核心。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图：</b>一组线程中 warp 分歧的示例（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色空间表示至少部分物理 CUDA 核心的停顿" %}

这在线程级别实现了灵活编程，但代价是如果 warp 太频繁分歧会悄悄降低性能。线程在可以访问的内存方面也可以更灵活；虽然 VPU 只能操作连续的内存块，CUDA 核心可以访问共享寄存器中的单个浮点数并维护每线程状态。

**CUDA 核心调度也更灵活：** SM 运行有点像多线程 CPU，因为它们可以并发"调度"许多程序（**warp**）（每个 SM 最多 64 个）但每个 _Warp 调度器_ 在每个时钟周期只执行一个程序。<d-footnote>调度在给定 SM 上的 warp 被称为"常驻"。</d-footnote> Warp 调度器自动在活跃 warp 之间切换以隐藏 I/O 操作如内存加载。相比之下，TPU 通常是单线程的。

### 内存

除了计算单元，GPU 有一个内存层级，最大的是 HBM（主 GPU 内存），然后是一系列更小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器：** 每个子分区有自己的寄存器文件，在 H100/B200 上包含 16,384 个 32 位字（每个 SM `4 * 16384 * 4 = 256kiB`），可由 CUDA 核心访问。
  * 每个 CUDA 核心一次最多只能访问 256 个寄存器，所以尽管我们可以每个 SM 调度最多 64 个"常驻 warp"，如果每个线程使用 256 个寄存器，你一次只能容纳 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM (L1 缓存)：** 每个 SM 有自己的 256kB 片上缓存叫 SMEM，可以由程序员控制为"共享内存"或由硬件用作片上缓存。SMEM 用于存储激活和 TC matmul 的输入。

* **L2 缓存：** 所有 SM 共享<d-footnote>技术上，L2 缓存分成两半，所以一半的 SM 在 H100 上每个可以访问 25MB。有一条链接连接两半，但带宽较低。</d-footnote>一个相对较大的约 50MB L2 缓存，用于减少主内存访问。
  * 这与 TPU 的 VMEM 大小相似，但它**慢得多**并且不由程序员控制。这导致一点"幽灵作用于远处"，程序员需要修改内存访问模式以确保 L2 缓存被良好使用。<d-footnote>L2 缓存在所有 SM 之间共享这一事实有效地迫使程序员以相当协调的方式运行 SM，尽管原则上它们是独立的单元。</d-footnote>
  * NVIDIA 不公布其芯片的 L2 带宽，但它已被[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)为约 5.5TB/s。因此这大约是 HBM 带宽的 1.6 倍，但它是全双工的，所以有效的双向带宽接近 3 倍。相比之下，TPU 的 VMEM 大 2 倍*并且*有更多带宽（约 40TB/s）。

* **HBM：** 主 GPU 内存，用于存储模型权重、梯度、激活等。
  * HBM 大小从 Volta 的 32GB 大幅增加到 Blackwell（B200）的 192GB。
  * 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或内存带宽，在 H100 上约为 3.35TB/s，在 B200 上约为 9TB/s。

### GPU 规格概览

这是近期型号的 GPU 规格概览。SM 数量、时钟速度和 FLOPs 在给定 GPU 的变体之间略有不同。这是内存容量数字：

|  GPU  | 代次 |   时钟速度   | SM/芯片 | SMEM 容量/SM | L2 容量/芯片 | HBM 容量/芯片 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38HGz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

所有代次每 SM 有 256kB 寄存器内存。Blackwell 每 SM 还增加了 256kB TMEM。这是每个芯片的 FLOPs 和带宽数字：

|  GPU  | 代次 | HBM 带宽/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

我们排除 B100，因为它没有量产。<d-footnote>虽然 NVIDIA 做了 B100 代，但它们只是短暂销售和生产，据说是由于设计缺陷阻止它们接近其声称的规格运行。由于热量和功率问题，它们努力在不节流的情况下实现峰值 FLOPs。</d-footnote> 一些规格根据 GPU 的精确版本略有不同，因为 NVIDIA GPU 不如 TPU 标准化。

这是比较 GPU 和 TPU 组件的有用备忘单：

|              GPU              |     TPU     |              这是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器 (SM) | Tensor Core | 包含其他单元的核心"单元" |
|        Warp 调度器         |     VPU     |      SIMD 向量算术单元      |
|           CUDA 核心           |   VPU ALU   |               SIMD ALU                |
|        SMEM (L1 缓存)        |    VMEM     |       快速片上缓存内存       |
|          Tensor Core          |     MXU     |      矩阵乘法单元       |
|        HBM (又名 GMEM)         |     HBM     |  高带宽高容量内存  |

### GPU 与 TPU 在芯片级别的对比

GPU 最初是渲染视频游戏的，但自从深度学习在 2010 年代起飞以来，它们开始越来越像专用的矩阵乘法机器——换句话说，更像 TPU。<d-footnote>在深度学习繁荣之前，GPU（"图形处理单元"）做的是图形——主要是视频游戏。视频游戏用数百万个小三角形表示对象，游戏每秒 30-60 次将这些三角形渲染（或"光栅化"）成显示在屏幕上的 2D 图像（这个频率称为帧率）。光栅化涉及将这些三角形投影到相机的坐标框架中，并计算哪些三角形与哪些像素重叠，每秒数十亿次。正如你可以想象的，这非常昂贵，而且这只是开始。然后你必须通过组合可能与光线相交的几个半透明三角形的颜色来给每个像素着色。GPU 被设计为极快地做这些操作，着眼于多功能性；你需要同时运行许多不同的 GPU 工作负载（称为"着色器"），没有单一操作占主导地位。因此，消费者以图形为重点的 GPU 可以做矩阵乘法，但这不是它们的主要功能。</d-footnote> 在某种程度上，这段历史解释了为什么现代 GPU 看起来是这样。它们不是纯粹为 LLM 或 ML 模型设计的，而是作为通用加速器，硬件旨在达到一种"通用性"水平，这既可能是福也可能是祸。GPU 在应用于新任务时更经常"直接工作"，比 TPU 更不依赖好的编译器。但这也使它们更难推理或获得 roofline 性能，因为太多编译器特性可能导致瓶颈。

**GPU 更模块化。** TPU 有 1-2 个大 Tensor Core，而 GPU 有数百个小 SM。同样，每个 Tensor Core 有 4 个大 VPU，每个有 1024 个 ALU，而 H100 GPU 有 132 * 4 = 528 个小的独立 SIMD 单元。这是突出这一点的 GPU 与 TPU 1:1 对比：

|              GPU              |           TPU            | H100 # | TPU v5p # |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM (流式多处理器) |       Tensor Core        |  132   |     2     |
|        Warp 调度器         |           VPU            |  528   |     8     |
|        SMEM (L1 缓存)        |           VMEM           |  32MB  |   128MB   |
|           寄存器           | 向量寄存器 (VRegs) |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

模块化方面的这种差异一方面使 TPU 构建更便宜、更容易理解，但它也给编译器做正确的事情带来更多负担。因为 TPU 有单线程控制并且只支持向量化 VPU 宽度的指令，编译器需要手动流水线化所有内存加载和 MXU/VPU 工作以避免停顿。GPU 程序员可以只是启动几十个不同的内核，每个在完全独立的 SM 上运行。另一方面，那些内核可能会获得糟糕的性能，因为它们在抖动 L2 缓存或未能合并内存加载；因为硬件控制太多运行时，变得难以推理幕后发生了什么。因此，TPU 通常可以用更少的工作更接近峰值 roofline 性能。

**历史上，单个 GPU 比同类 TPU 更强大（也更昂贵）：** 单个 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 是 1.5 倍。同时，Google Cloud 上的标价约为 H200 每小时 $10，而 TPU v5p 为每小时 $4。TPU 通常更依赖于将多个芯片联网在一起。

**TPU 有更多快速缓存内存。** TPU 的 VMEM 也比 GPU 的 SMEM（+TMEM）多得多，这种内存可以用于存储权重和激活，使它们可以被极快地加载和使用。如果你可以一致地将模型权重存储或预取到 VMEM，这可以使它们在 LLM 推理中更快。

### 测验 1：GPU 硬件

这里有一些问题来测试上面的一些内容。提供了答案，但在查看之前尝试回答问题，手边有纸笔，可能是个好主意。

**问题 1 [CUDA 核心]：** H100 有多少 fp32 CUDA 核心（ALU）？B200 呢？这与 TPU v5p 中独立 ALU 的数量相比如何？

{% details 点击这里查看答案。 %}

**答案：** H100 有 132 个 SM，每个有 4 个子分区，每个包含 32 个 fp32 CUDA 核心，所以我们有 `132 * 4 * 32 = 16896` 个 CUDA 核心。B200 有 `148` 个 SM，所以总共 `18944` 个。TPU v5p 有 2 个 TensorCore（通常通过 Megacore 连接），每个有一个 VPU，有 (8, 128) 个通道，每个通道有 4 个独立 ALU，所以 `2 * 4 * 8 * 128 = 8192` 个 ALU。这大约是 H100 向量通道数量的一半，运行频率大致相同。

{% enddetails %}

**问题 2 [向量 FLOPs 计算]**：单个 H100 有 132 个 SM，运行时钟速度为 1.59GHz（最高 1.98GHz boost）。假设每个 ALU 每周期可以做一个向量操作。每秒可以做多少向量 fp32 FLOPs？使用 boost 呢？这与 matmul FLOPs 相比如何？

{% details 点击这里查看答案。 %}

**答案：** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。使用 boost 是 33.5 TFLOPs/s。这是[规格表](https://www.nvidia.com/en-us/data-center/h100/)中报告的一半，因为技术上我们可以在一个周期内做一个 FMA（融合乘加），算作两个 FLOPs，但这在大多数情况下没用。我们可以做 990 bfloat16 matmul TFLOPs/s，所以忽略 FMA，Tensor Core 做约 30 倍更多的 FLOPs/s。

{% enddetails %}

**问题 3 [GPU matmul 强度]：** H100 上的峰值 fp16 matmul 强度是多少？B200 呢？fp8 呢？*强度指的是 matmul FLOPs/s 与内存带宽的比率。*

{% details 点击这里查看答案。 %}

**答案：** 对于 H100，我们有峰值 990e12 fp16 FLOPs 和 3.35e12 字节/秒的带宽。所以临界强度是 `990e12 / 3.35e12 = 295`，与 TPU 的 240 相当相似。对于 B200 是 `2250e12 / 8e12 = 281`，非常相似。这意味着，与 TPU 类似，我们需要约 280 的批次大小才能在 matmul 中是计算受限的。

对于 H100 和 B200，我们都有正好 2x fp8 FLOPs，所以峰值强度也加倍到 590 和 562，尽管在某种意义上如果我们考虑到我们的权重可能也以 fp8 加载，它保持不变。

{% enddetails %}

**问题 4 [Matmul 运行时间]：** 使用问题 3 的答案，你预期单个 B200 上 `fp16[64, 4096] * fp16[4096, 8192]` matmul 需要多长时间？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

{% details 点击这里查看答案。 %}

从上面，我们知道在批次大小 281 个 token 以下我们将是通信受限的。因此第一个是纯粹带宽受限的。我们读取或写入 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽为 `8e12` 字节/秒，所以大约需要 `69e6 / 8e12 = 8.6us`。实际上我们可能只获得总带宽的一部分，所以可能接近 10-12us。当我们增加批次大小时，我们是完全计算受限的，所以我们预期 `T=2*512*4096*8192/2.3e15=15us`。我们同样只预期获得总 FLOPs 的一部分，所以我们可能看到接近 20us。

{% enddetails %}

**问题 5 [L1 缓存容量]：** H100 的总 L1/SMEM 容量是多少？寄存器内存呢？这与 TPU VMEM 容量相比如何？

{% details 点击这里查看答案。 %}

**答案：** 我们每 SM 有 256kB SMEM 和 256kB 寄存器内存，所以每个约 33MB（`132 * 256kB`）。一起，这给我们总共约 66MB。这大约是现代 TPU 120MB VMEM 的一半，尽管 TPU 总共只有 256kB 寄存器内存！TPU VMEM 延迟比 SMEM 延迟低，这是 TPU 上寄存器内存不那么关键的一个原因（溢出和填充到 VMEM 很便宜）。

{% enddetails %}

**问题 6 [计算 B200 时钟频率]：** NVIDIA [在这里](https://resources.nvidia.com/en-us-blackwell-architecture)报告 B200 可以执行 80TFLOPs/s 的向量 fp32 计算。假设每个 CUDA 核心在 FMA（融合乘加）操作中每周期可以执行 2 个 FLOPs，估计峰值时钟周期。

{% details 点击这里查看答案。 %}

**答案：** 我们知道我们有 148 * 4 * 32 = 18944 个 CUDA 核心，所以我们每周期可以做 `18944 * 2 = 37888 FLOPs`。因此 `80e12 / 37888 = 2.1GHz`，一个高但合理的峰值时钟速度。B200 通常是液冷的，所以更高的时钟周期更合理。

{% enddetails %}

**问题 7 [估计 H100 加法运行时间]：** 使用上面的数字，计算在单个 H100 上将两个 `fp32[N]` 向量加在一起应该需要多长时间。计算 $T_\text{math}$ 和 $T_\text{comms}$。这个操作的算术强度是多少？如果你可以访问，也尝试在 PyTorch 或 JAX 中运行这个操作，用于 `N = 1024` 和 `N=1024 * 1024 * 1024`。这相比如何？

{% details 点击这里查看答案。 %}

**答案：** 首先，将两个 `fp32[N]` 向量相加执行 N 个 FLOPs，需要加载 `4 * N * 2` 字节并写回 4 * N 字节，总共 `3 * 4 * N = 12N`。计算它们的比率，我们有 `总 FLOPs / 总字节 = N / 12N = 1 / 12`，这相当糟糕。

如上所述，忽略 FMA 我们可以做大约 33.5 TFLOPs/s boost。这只有在所有 CUDA 核心都被使用时才成立。对于 `N = 1024`，我们*最多*只能使用 1024 个 CUDA 核心或 8 个 SM，这将需要更长时间（假设我们是计算受限的，大约长 16 倍）。我们还有 3.35e12 字节/秒的内存带宽。因此我们的峰值硬件强度是 `33.5e12 / 3.35e12 = 10`。<d-footnote>值得注意的是，这个强度在最近的 GPU 代次中保持不变。对于 H100 是 33.5 / 3.5，对于 B200 是 80 / 8。为什么不清楚，但这是一个有趣的观察。</d-footnote> 所以我们将是可怕地通信受限的。因此我们的运行时间只是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对于 `N = 65,536`，这大约是 0.23us。实际上我们在 JAX 中看到大约 1.5us 的运行时间，这很好因为我们预期在这里是超级延迟受限的。对于 `N = 1024 * 1024 * 1024`，我们有约 3.84ms 的 roofline，我们看到 4.1ms，这很好！

{% enddetails %}

## 网络

网络是 GPU 和 TPU 差异最大的领域之一。正如我们所见，TPU 以 2D 或 3D 环面连接，每个 TPU 只连接到其邻居。这意味着在两个 TPU 之间发送消息必须通过每个中间的 TPU，并迫使我们只在网格上使用均匀的通信模式。虽然在某些方面不方便，但这也意味着每个 TPU 的链接数量是恒定的，我们可以扩展到任意大的 TPU "pod"而不损失带宽。

另一方面，GPU 使用更传统的分层树状交换网络。称为**节点**的 8 个 GPU 组（GB200 最多 72 个<d-footnote>节点一词是重载的，可以指两件事：NVLink 域，即通过 NVLink 互连完全连接的 GPU 集合，或连接到单个 CPU 主机的 GPU 集合。在 B200 之前，这些通常是相同的，但在 GB200 NVL72 中，我们有一个 72 个 GPU 的 NVLink 域，但仍然只有 8 个 GPU 连接到每个主机。我们在这里使用节点一词指 NVLink 域，但这是有争议的。</d-footnote>）使用称为 NVLink 的高带宽互连在 1 跳内相互连接，这些节点使用附加到每个 GPU 的 NIC 通过较低带宽的 InfiniBand (IB) 或以太网网络连接成更大的单元（称为 **SU** 或可扩展单元）。这些反过来可以通过更高级别的交换机连接成任意大的单元。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b>显示典型 H100 网络的图示。一组 8 个 GPU 通过 NVSwitch（也称为 NVLink 交换机）连接成节点或 NVLink 域，这些节点通过交换的 InfiniBand 结构相互连接。H100 在 NVLink 域中每个有约 450GB/s 的出口带宽，每个节点有 400GB/s 的出口带宽进入 IB 网络。" %}

### 节点级别

GPU 节点是一个小单元，通常是 8 个 GPU（GB200 最多 72 个），通过全对全、全带宽、低延迟的 NVLink 互连连接。<d-footnote>NVLink 被描述为类似于加强版 PCIe 连接，延迟和协议开销低但不是为可扩展性/容错设计的，而 InfiniBand 更像以太网，为更大的有损网络设计。</d-footnote> 每个节点包含几个高带宽 NVSwitch，在所有本地 GPU 之间交换数据包。实际的节点级拓扑随时间变化很大，包括每个节点的交换机数量，但对于 H100，我们每个节点有 4 个 NVSwitch，GPU 以 `5 + 4 + 4 + 5` 链接模式连接到它们，如图所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>从 Pascal (P100) 开始的节点又名 NVLink 域图示。自 Volta (V100) 以来，我们使用一组交换机在节点内具有全对全连接。H100 节点有 4 个 NVSwitch，通过 25GB/s 链接连接到所有 8 个 GPU。" %}

对于 Hopper 代（NVLink 4.0），每个 NVLink 链接有 25GB/s 的全双工<d-footnote>这里全双工意味着每个方向 25GB/s，两个方向彼此独立。你可以在链接上发送总共 50GB/s，但每个方向最多 25GB/s。</d-footnote>带宽（B200 为 50GB/s），给我们从每个 GPU 进入网络的 `18 * 25=450GB/s` 全双工带宽。巨大的 NVSwitch 有多达 64 个 NVLink 端口，意味着带有 4 个交换机的 8xH100 节点可以处理多达 `64 * 25e9 * 4=6.4TB/s` 的带宽。这是这些数字如何随 GPU 代次变化的概述：

| NVLink 代 | NVSwitch 代 | GPU 代次 | NVLink 带宽 (GB/s，全双工) | NVLink 端口 / GPU | 节点 GPU 到 GPU 带宽 (GB/s 全双工) | 节点大小 (NVLink 域) | 每节点 NVSwitch |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell (B200) 有 8 个 GPU 的节点。GB200NVL72 支持更大的 72 个 GPU 的 NVLink 域。我们显示 8 和 72 个 GPU 系统的详细信息。

### 测验 2：GPU 节点

这里有一些关于网络的更多问答问题。我发现这些特别有用，因为它们让你思考实际的通信模式。

**问题 1 [H100 节点的总带宽]：** 带有 4 个交换机的 8xH100 节点每个节点有多少总带宽？*提示：*考虑 NVLink 和 NVSwitch 带宽。

{% details 点击这里查看答案。 %}

**答案：** 我们有 Gen4 4xNVSwitch，每个有 `64 * 25e9=1.6TB/s` 单向带宽。这将给我们在交换机级别 `4 * 1.6e12=6.4e12` 带宽。然而，注意每个 GPU 只能处理 450GB/s 的单向带宽，所以这意味着我们最多有 `450e9 * 8 = 3.6TB/s` 带宽。因为这更小，峰值带宽是 3.6TB/s。

{% enddetails %}

**问题 2 [二分带宽]**：二分带宽定义为网络任何均匀分区之间可用的最小带宽。换句话说，如果将网络分成两半，两半之间有多少带宽？你能计算 8x H100 节点的二分带宽吗？*提示：*二分带宽通常包括两个方向的流量。

{% details 点击这里查看答案。 %}

**答案：** 任何均匀分区将每半有 4 个 GPU，每个可以出口 `4 * 450GB/s` 到另一半。考虑两个方向的流量，这给我们 `8 * 450GB/s` 字节跨越分区，或 3.6TB/s 二分带宽。这是 NVIDIA 报告的，例如[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题 3 [AllGather 成本]**：给定 B 字节的数组，8xH100 节点上（吞吐量受限的）AllGather 需要多长时间？为 bf16[D<sub>X</sub>, F] 做数学，其中 `D=4096`，`F=65,536`。*在回答这个问题之前值得阅读 TPU 集合操作[章节](https://jax-ml.github.io/scaling-book/sharding/)。在这里仔细思考，但我们接下来会更多地讨论集合操作。*

{% details 点击这里查看答案。 %}

**答案：** 每个 GPU 可以出口 450GB/s，每个 GPU 有 $B / N$ 字节（其中 `N=8`，节点大小）。我们可以想象每个节点一个接一个地将其字节发送到其他 $N - 1$ 个节点，导致总共 (N - 1) 轮，每轮 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，或 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这大约是 $B / (N * W_\text{uni})$ 或 $B / \text{3.6e12}$，二分带宽。

对于给定的数组，我们有 `B=4096 * 65536 * 2=512MB`，所以总时间是 `536e6 * (8 - 1) / 3.6e12 = 1.04ms`。这可能是延迟受限的，所以实际上可能比这更长（实际上大约需要 1.5ms）。

{% enddetails %}

## 超越节点级别

超越节点级别，GPU 网络的拓扑不太标准化。NVIDIA 发布了一个[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，使用 InfiniBand 连接比单个节点更大的 GPU 集合，但客户和数据中心提供商可以根据需要自定义。<d-footnote>例如，Meta 在一个与此描述显著不同的数据中心网络上训练了 LLaMA-3，使用以太网、3 层交换结构和顶级过订阅交换机。</d-footnote>

这是参考 1024 GPU H100 系统的图示，底行的每个框是一个单独的 8xH100 节点，有 8 个 GPU、8 个 400Gbps CX7 NIC（每个 GPU 一个）和 4 个 NVSwitch。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b>参考 1024 H100 DGX SuperPod 的图示，有 128 个节点（有时 127 个），每个有 8 个 H100 GPU，连接到 InfiniBand 横向扩展网络。每组 32 个节点（256 个 GPU）称为"可扩展单元"或 SU。叶和脊 IB 交换机提供节点之间全二分带宽。" %}

**可扩展单元：** 每组 32 个节点称为"可扩展单元"（或 SU），在一组 8 个叶 InfiniBand 交换机下。这个 SU 有 256 个 GPU，每个节点有 4 个 NVSwitch，8 个 InfiniBand 叶交换机。所有显示的布线是 InfiniBand NDR（50GB/s 全双工），有 64 端口 NDR IB 交换机（也是每端口 50GB/s）。*注意 IB 交换机的带宽是 NVSwitch 的 2 倍（64 端口 400 Gbps 链接）。*

**SuperPod：** 整个 SuperPod 然后用 16 个顶级"脊"IB 交换机连接 4 个这样的 SU，给我们 1024 个 GPU，有 512 个节点级 NVSwitch、32 个叶 IB 交换机和 16 个脊 IB 交换机，总共 512 + 32 + 16 = 560 个交换机。叶交换机以 32 个节点为一组连接到节点，所以每组 256 个 GPU 有 8 个叶交换机。所有叶交换机都连接到所有脊交换机。

**我们有多少带宽？** InfiniBand 网络（称为"横向扩展网络"）的整体拓扑是一棵**胖树**，电缆和交换机保证节点级别以上的全二分带宽（这里是 400GB/s）。这意味着如果我们将节点分成两半，每个节点可以同时向另一个分区的节点出口 400GB/s。更重要的是，这意味着我们应该在横向扩展网络中有一个大致恒定的 AllReduce 带宽！虽然它可能不是这样实现的，你可以想象在横向扩展网络中的任意多个节点上做环形归约，因为你可以构建一个包括每个节点的环。

| 级别 | GPU | 每单元交换机 | 交换机类型 | 每单元带宽 (TB/s，全双工) | GPU 到 GPU 带宽 (GB/s，全双工) | 胖树带宽 (GB/s，全双工) |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊 |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

相比之下，TPU v5p 每个链接约有 90GB/s 出口带宽，或沿 3D 环面所有轴 540GB/s 出口。这不是点对点的，所以它只能用于受限的均匀通信模式，但它仍然给我们更高的 TPU 到 TPU 带宽，可以扩展到任意大的拓扑（至少到 8960 个 TPU）。

GPU 交换结构理论上可以通过添加额外的交换机或间接层扩展到任意大小，代价是额外的延迟和昂贵的网络交换机。

<p markdown=1 class="takeaway">**要点**：在 H100 节点内，我们从每个 GPU 有 450GB/s 的全胖树带宽，而节点外，这降至 400GB/s 节点到节点。这对通信原语来说将是关键的。</p>

**GB200 NVL72：** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，将 72 个 GPU 组合在一个 NVLink 域中，具有完整的 900GB/s GPU 到 GPU 带宽。这些域然后可以用成比例更高（9 倍）的 IB 胖树带宽链接成更大的 SuperPod。这是该拓扑的图示：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>显示 576 个 GPU 的 GB200 DGX SuperPod 的图示。底层的每个机架包含 72 个 GB200 GPU。" %}

计算单个节点的出口带宽（上面的橙色线），我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 到叶级别的带宽，这比 H100 高 9 倍（正如节点包含 9 倍更多的 GPU）。这意味着关键的节点出口带宽*高得多*，我们的跨节点集合带宽实际上可以比节点内*低*。
有关更多讨论，请参见[附录 A](#appendix-a-how-does-this-change-with-gb200)。

|  节点类型  | 每节点 GPU | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPod 大幅增加了节点大小和给定节点的出口带宽，这显著改变了我们的 roofline。</p>

### 测验 3：超越节点级别

**问题 1 [胖树拓扑]：** 使用上面的 DGX H100 图示，计算整个 1024 GPU pod 在节点级别的二分带宽。证明每个链接的带宽被选择为确保全二分带宽。*提示：确保计算链接带宽和交换机带宽。*

{% details 点击这里查看答案。 %}

**答案：** 让我们逐个组件来做：

* 首先，每个节点有 8x400Gbps NDR IB 电缆连接到叶交换机，给每个节点 `8 * 400 / 8 = 400 GB/s` 到叶的带宽。我们有 8 个叶交换机，每个 3.2TB/s（64 400 GBps 链接），但我们只能使用 64 个端口中的 32 个来入口从 SU，所以是 `32 * 400 / 8 = 12.8TB/s` 对 32 个节点，同样正好是 400GB/s。
* 然后在脊级别我们有 `8 * 16 * 2` 条 400Gbps NDR IB 电缆连接每个 SU 到脊，给每个 SU `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 到叶的带宽。同样，这是每节点 400GB/s。我们有 16 个脊交换机，每个 3.2TB/s，给我们 `16 * 3.2 = 51.2 TB/s`，128 个节点同样是每节点 400GB/s。

因此，如果我们以任何方式二分我们的节点，我们将在它们之间有 400GB/s 每 GPU。每个组件都有确切的必要带宽来确保胖树。

{% enddetails %}

**问题 2 [扩展到更大的 DGX pod]：** 假设我们想在 2048 个 GPU 而不是 1024 个上训练。修改上述 DGX 拓扑以处理这个的最简单/最佳方法是什么？4096 呢？*提示：没有单一正确答案，但尝试控制成本。记住链接容量。[这个](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能有帮助。*

{% details 点击这里查看答案。 %}

**答案：** 一个选项是保持 SU 结构不变（32 个节点在 8 个交换机下），只是用更多的顶级交换机添加更多。我们需要 2 倍更多的脊交换机，所以我们有 8 个 SU 和 32 个脊交换机给我们足够的带宽。

这个问题是我们每个叶交换机只有 64 个端口，上图中我们已经用完了所有。但相反，很容易做每个脊 1x 400 Gbps NDR 电缆而不是 2x，这给出相同的总带宽但节省了一些端口。

对于 4096 个 GPU，我们实际上用完了端口，所以我们需要添加另一个间接层，也就是说，层级中的另一个级别。NVIDIA 称这些为"核心交换机"，用 128 个脊交换机和 64 个核心交换机构建 4096 GPU 集群。你可以做数学来证明这给出足够的带宽。

{% enddetails %}

## GPU 上的集合操作如何工作？

GPU 可以执行与 TPU 相同的所有集合操作：ReduceScatter、AllGather、AllReduce 和 AllToAll。与 TPU 不同，这些工作方式取决于它们是在节点级别执行（通过 NVLink）还是以上（通过 InfiniBand）。这些集合操作由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（发音为"nickel"）库中实现。NCCL 在[这里](https://github.com/NVIDIA/nccl)开源。虽然 NCCL 根据延迟要求/拓扑使用各种实现（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），从这里开始，我们将讨论在交换树结构上的理论最优模型。

### 节点内集合操作

**AllGather 或 ReduceScatter：** 对于节点级别的 AllGather 或 ReduceScatter，你可以像 TPU 一样在环上执行它们，在每跳使用完整的 GPU 到 GPU 带宽。任意排列 GPU，使用完整的 GPU 到 GPU 带宽在环上发送数组的一部分。<d-footnote>你也可以想象每个 GPU 将其大小为 $\text{bytes} / N$ 的块发送到其他 $N - 1$ 个 GPU，总共 $(N - 1) * N * bytes / N$ 字节通信，这给我们</d-footnote> 每跳的成本是 $T_\text{hop} = \text{bytes} / (N * \text{GPU 出口带宽})$，所以总成本是

$$T_\text{AG 或 RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU 出口带宽}} \rightarrow \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

你会注意到这与 TPU 上完全相同。对于 AllReduce，你可以像往常一样组合 RS + AG，成本翻倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b>带宽最优的 1D 环 AllGather 算法。对于 B 字节，这在顶级交换机上发送 V / X 字节 X - 1 次。" %}

如果你关心延迟（例如如果你的数组非常小），你可以做树形归约，在 2 的对内 AllReduce，然后 4，然后 8，总共 $\log(N)$ 跳而不是 $N - 1$，尽管总成本仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内 AllGather 或 ReduceScatter B 字节数组的成本约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU 出口}) \approxeq B / W_\text{GPU 出口}$。这在理论上在 H100 上约为 $B  / \text{450e9}$，在 B200 上为 $B / \text{900e9}$。除非启用了网络内归约，否则 AllReduce 的成本是 2 倍。</p>

<b markdown=1 style="color: #57cf57;">快问快答 1 [AllGather 时间]：</b> 使用带有 450 GB/s 全双工带宽的 8xH100 节点，AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间？设 $B=1024$，$F=16,384$。

{% details 点击这里查看答案。 %}

**答案：** 我们有总共 $2 \cdot B \cdot F$ 字节，450e9 单向带宽。这大约需要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，或更精确地 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。使用提供的值，这给我们大约 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，或更精确地，$\text{65us}$。

{% enddetails %}

**AllToAll：** 节点内的 GPU 有全对全连接，这使得 AllToAll，嗯，相当容易。每个 GPU 只是直接发送到目标节点。在节点内，对于 B 字节，每个 GPU 有 $B / N$ 字节并发送 $(B / N^2)$ 字节到 $N - 1$ 个目标节点，总共

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

与 TPU 相比，成本是 $B / (4W)$。因此，在单个节点内，我们获得 2 倍的理论运行时间加速（$B / 4W$ vs. $B / 8W$）。

对于混合专家（MoE）模型，我们经常想做*稀疏或参差 AllToAll*，我们保证输出维度上最多 $k$ 个 $N$ 分片是非零的，也就是说 $T_\text{AllToAll} \rightarrow K[B, N]$，其中每个轴上最多 $k$ 个 $N$ 条目是非零的。这个成本减少了 $k/N$，总共约 $\min(k/N, 1) \cdot B / (W \cdot N)$。对于 MoE，我们经常独立随机选择非零值，所以有一些机会少于 $k$ 个非零，给我们大约
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>真正的成本实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$，$K$ 次骰子投掷中预期的不同结果数，但它非常接近给出的近似值。更多细节见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">快问快答 2 [AllToAll 时间]：</b> 使用带有 450 GB/s 单向带宽的 8xH100 节点，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间？如果我们知道只有 8 个条目中的 4 个是非零的呢？

{% details 点击这里查看答案。 %}

**答案：** 从上面，我们知道在稠密情况下，成本是 $B \cdot (N-1) / (W \cdot N^2)$，或 $B / (W \cdot N)$。如果我们知道只有 $\frac{1}{2}$ 的条目不是填充，我们可以发送 $B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$，大约是总成本的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 在单个节点内 GPU 上 $B$ 字节数组的 AllToAll 成本约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU 出口}) \approx B / (8 \cdot W_\text{GPU 出口})$。对于参差（top-$k$）AllToAll，这进一步减少到 $(B \cdot k) / (64 \cdot W_\text{GPU 出口})$。</p>

**经验测量：** 这是 8xH100 节点上 AllReduce 带宽的经验测量。Algo BW 是测量的带宽（字节/运行时间），Bus BW 计算为 $2 \cdot W \cdot (8 - 1) / 8$，理论上是实际链接带宽的测量。你会注意到我们确实达到了接近 370GB/s，低于 450GB/s 但相当接近，尽管只在约 10GB/设备时。这意味着虽然这些估计在理论上是正确的，但需要大消息才能实现。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>禁用 SHARP 的 8xH100 节点上的 AllReduce 吞吐量。蓝色曲线是经验链接带宽，计算为 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ 来自经验测量。注意即使使用大量的 10GB 数组，我们也没有特别接近声称的 450GB/s 带宽。" %}

这是一个真正的问题，因为它有意义地复杂化了我们可以做出的任何理论声明，因为例如即使是合理大小数组的 AllReduce，如 LLaMA-3 70B 的 MLP（大小为 `bf16[8192, 28672]`，或使用 8 路模型分片，`bf16[8192, 3584] = 58MB`）只能达到约 150GB/s，而峰值 450GB/s。相比之下，TPU 在更低的消息大小下就能达到峰值带宽（见附录 B）。

<p markdown=1 class="takeaway">**要点：** 虽然 NVIDIA 声称 H100 NVLink 的带宽约为 450GB/s，但实际上很难超过 370 GB/s，所以相应调整上述估计。</p>

**网络内归约：** 自 Hopper 代以来，NVIDIA 交换机已支持 ["SHARP"（可扩展分层聚合和归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，允许"网络内归约"。这意味着*网络交换机本身*可以做归约操作并多路复用或"MultiCast"结果到多个目标 GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b>没有 SHARP 的 AllReduce 理论成本是 2 倍，因为它必须通过每个 GPU 两次。实际上，加速只有约 30%（来自 NCCL 2.27.5）。" %}

理论上，这接近将 AllReduce 的成本减半，因为它意味着每个 GPU 可以将其数据发送到顶级交换机，交换机本身执行归约并将结果广播到每个 GPU，而不必两次出口每个 GPU，同时也减少网络延迟。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

注意这是精确的，而不是差 $1/N$ 的因子，因为每个 GPU 先出口 $B \cdot (N - 1) / N$，然后接收其本地分片的部分归约版本（入口 $B/N$），完成归约，然后再出口 $B/N$，然后入口完全归约的结果（入口 $B \cdot (N - 1) / N$），导致正好 $B$ 字节入口。

然而，实际上我们看到启用 SHARP 时带宽增加约 30%，而不是预测的 75%。这使我们仅达到约 480GB/s 有效集合带宽，远不是 2 倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图：</b>节点内启用和不启用 NVIDIA SHARP 的 AllReduce 算法带宽的经验测量。收益在峰值时约为 30% 吞吐量改进，尽管从算法上讲它应该能够达到接近 75% 的增益。" %}

<p markdown=1 class="takeaway">**要点：** 理论上，NVIDIA SHARP（大多数 NVIDIA 交换机上可用）应该将 $B$ 字节 AllReduce 的成本从约 $2 * B / W$ 降低到 $B / W$。然而，实际上我们只看到约 30% 的带宽改进。由于纯 AllReduce 在 LLM 中相当罕见，这不是特别有用。</p>

### 跨节点集合操作

当我们超越节点级别时，成本稍微微妙一些。在树上做归约时，你可以想象从下往上归约，首先在节点内，然后在叶级别，然后在脊级别，在每个级别使用正常算法。特别是对于 AllReduce，你可以看到这允许我们整体通信更少数据，因为在节点级别 AllReduce 后，我们只需要出口 $B$ 字节到叶，而不是 $B * N$。

**这有多昂贵？** 粗略地说，因为我们有全二分带宽，AllGather 或 ReduceScatter 的成本大约是缓冲区字节大小除以节点出口带宽（H100 上 400GB/s），*不管树形归约的任何细节。*

$$T_\text{AG 或 RS comms} = \frac{\text{bytes}}{W_\text{节点出口}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{节点}$ 出口通常对于上述 H100 网络是 400GB/s（每个节点出口 8x400Gbps IB 链接）。想象这个最干净的方式是想象在*集群中的每个节点*上做环形归约。因为胖树拓扑，我们总是可以构建一个在任意两个节点之间有 $W_\text{节点}$ 出口的环并做正常归约。节点级归约（几乎）永远不会是瓶颈，因为它有更高的整体带宽和更好的延迟，尽管一般成本是

$$T_\text{total} = \max(T_\text{节点通信}, T_\text{横向扩展网络通信}) = \max\left[\frac{\text{bytes}}{W_\text{GPU 出口}}, \frac{\text{bytes}}{W_\text{节点出口}}\right]$$

{% details 你可以在这里看到更精确的推导。 %}

我们可以更精确地注意到我们实际上在网络的每一层做环形归约，我们可以大部分重叠，所以我们有：

$$T_\text{AG 或 RS comms} = \text{bytes} \cdot max_\text{深度 i}\left[\frac{D_i - 1}{D_i \cdot W_\text{链接 i}}\right]$$

其中 $D_i$ 是深度 $i$ 的度（深度 $i$ 的子节点数），$W_\text{link i}$ 是连接每个子节点到节点 $i$ 的链接带宽。

使用这个，我们可以计算给定拓扑的可用 AllGather/AllReduce 带宽为 $min_\text{深度 i}(D_i * W_\text{link i} / (D_i - 1))$。在上述情况下，我们有：

* **节点：** $D_\text{node}$ = 8，因为我们节点中有 8 个 GPU，Wlink i = 450GB/s。因此我们的 AG 带宽是 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **叶：** $D_\text{leaf}$ = 32，因为我们 SU 中有 32 个节点，Wlink i = 400GB/s（8x400Gbps IB 链接）。因此我们的带宽是 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **脊：** $D_\text{spine}$ = 4，因为我们有 4 个 SU，$W_\text{link i}$ = 12.8TB/s（来自上面 `8 * 16 * 2 * 400Gbps` 链接）。我们的带宽是 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此我们的整体 AG 或 RS 带宽是 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s` 在叶级别，所以实际上 $T_\text{AG 或 RS comms} = B / \text{413GB/s}$，即即使在最高级别我们也有约 413GB/s 的 AllReduce 带宽。对于带 SHARP 的 AllReduce，它会稍低一些（约 400GB/s），因为我们没有 $(N - 1) / N$ 因子。尽管如此，450GB/s 和 400GB/s 足够接近可以用作近似值。

{% enddetails %}

**其他集合操作：** 除非启用 SHARP，否则 AllReduce 仍然是上述成本的 2 倍。NVIDIA 也销售支持 SHARP 的 IB 交换机，尽管不是所有提供商都有。AllToAll 在跨节点时确实变化很大，因为它们不像 AllReduce 那样"分层"。如果我们想从每个 GPU 发送数据到每个其他 GPU，我们不能利用节点级别的全二分带宽。这意味着如果我们有一个跨越 $M = N / 8$ 个节点的 N 路 AllToAll，成本是

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{节点出口}} \approxeq \frac{B}{M \cdot W_\text{节点出口}}$$

这实际上有 50GB/s 而不是 400GB/s 的带宽。我们从单个 H100 节点内的 $B / (8 * \text{450e9})$ 变成跨越 2 个节点时的 $B / (2 \cdot \text{400e9})$，超过 4 倍的退化。

这是 1024-GPU DGX H100 SuperPod 架构的摘要：

|   级别   | GPU 数量 | 度（子节点数） | 交换机带宽（全双工，TB/s） | 电缆带宽（全双工，TB/s） | 集合带宽 (GB/s) |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   节点    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| 叶 (SU) |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   脊   |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

我们使用术语"集合带宽"来描述我们可以出口 GPU 或节点的有效带宽。它也是 $\text{二分带宽} * 2 / N$。

<p markdown=1 class="takeaway">**要点：** 超越节点级别，B 字节 AllGather 或 ReduceScatter 的成本大约是 $B / W_\text{节点出口}$，这在 H100 DGX SuperPod 上是 $B / \text{400e9}$，而 AllReduce 成本是两倍，除非启用 SHARP。整体拓扑是一棵胖树，设计为在任意两对节点之间提供恒定带宽。</p>

**数组沿另一个轴分片时的归约：** 考虑如下归约的成本

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

其中我们在一个本身沿另一个轴 $Y$ 分片的数组上 AllReduce。在 TPU 上，这个操作的总成本比未分片版本减少 $1 / Y$ 倍，因为我们每轴发送 $1 / Y$ 的数据。在 GPU 上，成本取决于哪个轴是"内部"轴（节点内与节点间）以及每个分片是否跨越多个节点。假设 $Y$ 是内部轴，数组有 $\text{bytes}$ 总字节，只有当 $Y$ 跨越多个节点时，总成本有效减少 $Y$：

$$T_\text{节点通信} = \frac{\text{bytes}}{W_\text{GPU 出口}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{横向扩展网络通信} = \frac{\text{bytes}}{W_\text{节点出口}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{节点通信}, T_\text{横向扩展网络通信})$$

其中 N 是 GPU 数量，$D_\text{node}$ 是节点中的 GPU 数量（节点的度）。如你所见，如果 $Y < D_\text{node}$，我们在节点级别获得胜利但通常看不到总运行时间的减少，而如果 $Y > D_\text{node}$，我们获得与内部轴跨越的节点数成比例的加速。

如果我们想精确关于环形归约，树 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> })（假设 Y 是内部轴）的一般规则是

$$T_\text{AR 或 RS comms} = \text{bytes} \cdot \max_{\text{深度 } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{链接 } i}}\right]$$

其中 $S_i$ 是 M * N * …，树中级别 i 以下子节点的大小。这大致是说我们跨越的 GPU 或节点越多，我们可用的带宽越大，但仅在该节点内。

**快问快答 3 [沿 2 轴分片]：** 假设我们想在单个 SU（256 个芯片）上执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中 $Y$ 是内部轴。这作为 $D$、$F$ 和 $Y$ 的函数需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** 我们可以分成两种情况，Y <= 8 和 Y > 8。当 $Y <= 8$ 时，我们仍受叶交换机约束，所以答案照常是 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，我们从上面大致有

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于 `D = 8192`，`F = 32,768`，我们有：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图：</b>随着内部轴跨越更多节点，分片 AllGather 的理论成本。" %}

注意，如果我们正好做 8 路模型并行，我们确实将节点级归约的成本减少了 8 倍，但保持总成本不变，所以它是免费的但对改善整体带宽没有帮助。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 当我们有多个分片轴时，外部归约的成本减少了内部轴跨越的节点数的因子。</p>

### 测验 4：集合操作

**问题 1 [SU AllGather]：** 只考虑一个有 M 个节点和每节点 N 个 GPU 的 SU。在 AllGather 期间，节点级交换机精确地入口和出口多少字节？顶级交换机呢？

{% details 点击这里查看答案。 %}

**答案：** 让我们一步一步来，逐步完成归约的组成部分：

1. 每个 GPU 发送 $B / MN$ 字节到交换机，总入口为 $NB / MN = B / M$ 字节入口。
2. 我们出口完整的 $B / M$ 字节到脊交换机。
3. 我们从脊交换机入口 $B * (M - 1) / M$ 字节
4. 我们出口 $B - B / MN$ 字节 $N$ 次，总共 $N * (B - B / MN) = NB - B / M$。

总共是 $B$ 入口和 $BN$ 出口，所以我们应该受出口瓶颈，总时间是 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于脊交换机，数学实际上更简单。我们必须入口 $B / M$ 字节 M 次（总共 $B$ 字节），然后出口 $B (M - 1) / M$ M 次，总共 $B * (M - 1)$ 出去。由于这显著更大，成本是 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题 2 [单节点 SHARP AR]：** 考虑一个每节点有 N 个 GPU 的单个节点。使用 SHARP（网络内归约）的 AllReduce 期间，交换机精确地入口和出口多少字节？

{% details 点击这里查看答案。 %}

**答案：** 和之前一样，让我们一步一步来。

1. 每个 GPU 发送 $B * (N - 1) / N$ 字节，所以我们入口 $N * B * (N - 1) / N = B * (N - 1)$。
2. 我们累积部分和，发回 $B / N$ 字节到每个 GPU，所以 $N * B / N = B$ 字节出口。
3. 我们在残差上本地做部分和，然后发回交换机。这总共是 $N * B / N = B$ 字节入口。
4. 我们捕获所有分片并多播它们，发送 $B * (N - 1) / N$ 到 $N$ 个目的地，总共 $B * (N - 1) / N * N = B * (N - 1)$ 出口。

因此总共是 $B * (N - 1) + B = BN$ 字节入口和出口。这支持整体吞吐量正好是 $B / W_\text{egress}$。

{% enddetails %}

## GPU 上 LLM 扩展的 Roofline

现在让我们看看这一切构建的目标：理解 GPU 上 LLM 扩展的 roofline。这是为了补充[这里](../training)的 TPU 训练章节。正如我们在那里所做的，这里的目标是查看不同并行策略的总 $T_\text{math}$ 和 $T_\text{comms}$ 并理解在什么点 $T_\text{comms} > T_\text{math}$。和以前一样，我们只考虑 MLP 块的操作

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是**以 token 计的**全局批次大小（即 $B = \text{批次大小} \cdot \text{序列长度}$）。

这里我们将重现上面显示 GPU 和节点级别有效带宽的表：

|  节点类型  | 每节点 GPU | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU 和节点出口带宽都决定了我们 LLM 的 roofline。我们将使用术语 $W_\text{collective}$ 来描述取决于我们是在节点级别内还是以上操作的 GPU 或节点带宽。

让我们像为 TPU 做的那样查看**数据并行、张量并行、流水线并行、专家并行**及其组合的计算通信 roofline。在本节的其余部分，我们将专注于 H100 roofline 进行具体计算。GB200-NVL72 有相同的一般 roofline，但因为我们有更大的节点出口带宽，我们有时可能受节点级别瓶颈。

### 数据并行

如前所述，DP 和 ZeRO 分片涉及反向传播中的权重 AllReduce 或 ReduceScatter + AllGather。由于这两者成本相同，对于纯数据并行或 FSDP *没有网络内归约*要成为计算受限，我们每层在反向传播中有，大小为 X 的轴：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，对于 $T_\text{math} > T_\text{comms}$，我们需要 $B / (XC) > 1 / W_\text{collective}$ 或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 或节点级别出口带宽，取决于我们是在节点内还是跨节点分片。因此：

* **在节点内**，我们只需要每 GPU **token** 批次大小 > $\text{990e12} / \text{450e9} = 2200$。
* **在 SU 内或脊级别**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 高得多，TPU 在所有三个轴上数字是 850。例如，在 16000 H100 上训练的 LLaMA-3 需要至少 4000 万 token 的批次大小（作为参考，他们使用了 1600 万）。在 2048 H800 GPU 上训练的 DeepSeek v3 带宽较低 300GB/s（而不是 H100 的 450GB/s）需要 $\text{990e12} / \text{300e9} = 3300$ 每 GPU token，或约 670 万（实际上，他们使用了 400 万）。

启用网络内归约并使用纯数据并行，理论上我们有 2 倍的 AllReduce 带宽，这将使这两个数字减半。然而，实际上好处接近 30%，这只是弥补了我们通常难以达到报告数字的事实。此外，因为纯数据并行很少有用，这在实践中基本不重要。

**MoE 模型：** 对于混合专家（MoE）模型，其中我们有 E 个专家，每个 token k 个专家，这增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这将每 GPU token 批次大小膨胀了 $E/k$ 倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的 OpenAI OSS 模型有 $k=4$ 和 $E=128$，这增加到跨节点 `32 * 2475  = 79,200`，一个相当荒谬的高数字。

**当 X 很小时会发生什么？** 当我们只做例如 2 节点数据并行时，我们受益于 $(X - 1) / X$ 缩放，这给我们

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数，$N = 8 \cdot X$。然后对于稠密模型我们有 $B / N > \alpha \cdot (X - 1) / X$，或例如 $B / N > \text{1237}$，上述值的一半。你会经常看到 2 路数据并行就是这个原因。

<p markdown=1 class="takeaway">**要点：** 数据并行和 ZeRO 分片需要每 GPU 约 2500 token 的批次大小才能在 H100 或 B200 上计算受限，假设完美重叠和 FLOPs 利用率。对于 MoE 模型，这增加了 $E / k$ 倍，即总参数与激活参数的比率。当做少量数据并行时，临界批次大小减少。</p>

### 张量并行

张量并行需要在激活上 AllGather 和 ReduceScatter，我们需要与 MLP FLOPs 重叠。换句话说，在前向传播中，我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要计算受限给我们规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内，这给我们约 $F / 2200$ 或节点外 $F / 2475$。对于像 LLaMA-3 的 $F=\text{28000}$，这约是 11 路 TP（或四舍五入，约 8 路，这是节点的大小）。如上所述，当我们正好跨越 2 个节点时，我们获得额外 2X 带宽，所以我们通常可以做 16 路数据并行（$F > 2475 \cdot (Y - 8)$），这理论上给我们最多 19 路模型并行。

<p markdown=1 class="takeaway">**要点：** 大小为 Y、前馈维度为 F 的轴上的张量并行当 $Y > F / 2475$ 时变成通信受限，这通常将我们限制为仅节点内 TP 或最多 2 节点 TP。</p>

### 专家并行

如上所述，混合专家（MoE）模型有 E 倍更多的模型权重，只有 k 倍更多的 FLOPs，这使数据并行显著更难。我们可以通过沿专家维度分片我们的权重来在一定程度上缓解这一点，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]。要做 MLP 块，我们需要引入 2x AllToAll 来将我们的激活发送到相应的专家。

如上所述，如果它跨越多个节点，AllToAll<sub>Z->k</sub>([B, D, k]) 的成本大约是 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，所以对于纯专家并行我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们要么需要 $K > Z/8$ 和 $F > \alpha \cdot (Z - 8)/k$，要么 $Z \gg K$ 和 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给你两个专家并行可能的域，一个是少量专家并行（大约 2 节点）和小 $F$，或一个是大 $F$ 和 $Z$ 任意大（最多 E 路专家并行）。

你会在实践中看到这两种情况，要么是少量专家并行（如 DeepSeek v3，它有非常小的 F 和相对较小、受限的跨节点专家并行），要么是大 F 的模型，在这种情况下我们可以做显著的跨节点 EP 以及 TP。

<p markdown=1 class="takeaway">**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以跨越 1-2 个节点，成本与 TP 相似（稍低），或者如果 $F > 8 * C / W_\text{node}$，我们可以做显著数量的专家并行（最多 $E$ 个节点），成本相对较低。</p>

### 流水线并行

流水线并行将层跨节点分割，通信成本极低，因为我们只是每隔几层发送小的微批次激活。历史上流水线受"流水线气泡"困扰，但有了新的零气泡流水线方法，通常可以避免。

流水线的总体通信成本很小：有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段，我们有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ 和 $N_\text{MB} + N_\text{stages} - 2$ 跳，所以大约

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们除以 $N_\text{layers}$，这比任何其他成本都小得多。换句话说，从通信角度来看，流水线基本上是免费的。那么为什么我们不只做流水线呢？有几个原因：

(1) **代码复杂性：** 流水线不像其他方法那样很好地适合自动并行框架（如 XLA 的 GSPMD）。因为它引入微批次来隐藏流水线气泡，它改变了程序的结构，自定义零气泡流水线调度通过要求前向和反向传播的复杂交错来加剧这个问题。

(2) **流水线使数据并行和 FSDP 变难：** 可能不做流水线的最大原因是它与 FSDP 和数据并行配合不好。特别是 ZeRO-3 分片效果不好，因为它要求我们在每个微批次上 AllGather 权重，当我们只有 $B / N_\text{microbatches}$ 个 token 来摊销 AllGather 成本时这不起作用。此外，在反向传播期间，*我们不能 AllReduce 或 ReduceScatter 梯度，直到最后一个微批次通过了给定阶段，这意味着我们有显著的非重叠通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b>一个 2 阶段、2 微批次流水线的示例。F 表示阶段前向传播，B 是阶段反向传播（成本 2 倍）。G 表示数据并行 AllReduce，可以比单个微批次的时间长得多。" %}

(3) **流水线气泡和步骤不平衡：** 如你在上面（糟糕的）流水线调度中看到的，很容易在朴素流水线调度期间有显著的气泡（意味着浪费计算）。上面，第二阶段在步骤 0 空闲，第一阶段从步骤 2 到 3 空闲，第二阶段在最后一步再次空闲。虽然我们可以通过仔细调度在一定程度上避免这些，但我们通常仍然有一些气泡。我们还必须在关键路径上将激活从一个阶段传递到下一个阶段，这可能增加开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b>一个流水线示例，红色显示传输成本。这使阶段相对彼此移位并增加流水线气泡开销。" %}

这些问题都有变通方法，但它们往往实现复杂、维护困难，但流水线仍然是相对于其他方法通信成本低的技术。

**关于延迟的注意事项：** 如前所述，GPU 即使使用相当大的消息也难以实现完整的 AllReduce 带宽。这意味着即使我们理论上可以跨多个节点扩展例如专家并行 AllToAll，我们可能难以达到总带宽的 50%。这意味着我们确实尝试将 TP 或 EP 保持在更少数量的节点内以最小化延迟开销。

### 示例

**DeepSeek 做什么？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 用 2048 H800 GPU 训练：

* 64 路专家并行（EP）跨越 8 个节点
* 16 路流水线并行（PP）
* 2 路 ZeRO-1 数据并行（DP）

他们的稳态批次大小是 `4096 * 15360 = 62,914,560` 个 token，或每 GPU 30k 个 token。你可以看到这已经相当大，但他们的模型也非常稀疏（k=8, E=256）所以你需要相当大的批次大小。你可以看到 64 路 EP 和 16 路 PP，我们最终得到总共 1024 路模型并行，这意味着 AllReduce 在脊级别完成，而且因为它只是 2 路，我们实际上得到 $2 / (2 - 1) = 2$ 倍更多的带宽。这也有助于减少与最终流水线阶段重叠的最终数据并行 AllReduce 的成本。

**LLaMA-3 做什么？** LLaMA-3 用 16M token 的 BS 在 16k GPU 上训练，或每 GPU 约 1k token。他们做：

* 8 路张量并行在节点内（TP）
* 16 路流水线并行（PP）
* 128 路 ZeRO-1 数据并行

这也是一个稠密模型，所以总的来说这些事情相当简单。16 路 PP 将数据并行 AllReduce 的成本减少了 16 倍，这有助于我们降低临界批次大小。

### GPU 上 LLM 扩展总结

让我们退一步，提出我们到目前为止学到的东西的总结：

* **数据并行或 FSDP（ZeRO-1/3）需要每 GPU 约 2500 token 的本地批次大小**，尽管理论上网络内归约 + 纯 DP 可以稍微减少这一点。
* **张量并行在最多约 8 路时是计算受限的**，但我们缺乏带宽来扩展超过这个范围而不变成通信受限。这主要将我们限制为单个 NVLink 域（即单节点或需要使用 GB200NVL72 达到 72 个 GPU）。
* **任何跨越多个节点的模型并行形式都可以进一步降低 FSDP 的成本**，所以我们经常想混合 PP + EP + TP 来跨越许多节点并降低 FSDP 成本。
* **如果你能处理零气泡流水线的代码复杂性并保持批次大小相当大以避免数据并行瓶颈，流水线并行效果很好。** 流水线通常使 ZeRO-3 不可能（因为你需要在每个流水线阶段 AllGather），但你可以做 ZeRO-1 代替。

**在高层次，这给我们一个在 GPU 上分片大模型的配方：**

* 对于相对较小的稠密模型，如果你有批次大小，激进的 FSDP 效果很好，如果需要可能有一些流水线或张量并行。
* 对于较大的稠密模型，1-2 节点 TP + 多节点 PP + 纯 DP 的某种组合效果很好。
* 对于 MoE，上述规则适用，但我们也可以做专家并行，我们通常更喜欢它而不是 TP。如果 $F > 8 * C / W_\text{node}$，我们可以做大量的多节点专家并行，否则我们被限制为大约 2 节点 EP。

### 测验 5：LLM roofline

**问题 1 [B200 roofline]：** B200 DGX SuperPod（**不是 GB200 NVL72**）节点内带宽是 2 倍（900GB/s 出口），但横向扩展网络的带宽相同（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总 FLOPs 如上所报。这如何改变模型和数据并行 roofline？

{% details 点击这里查看答案。 %}

**答案：** 我们的 bfloat16 FLOPs/s 从 990 增加到 2250 TFLOPs，增加 2.25 倍。节点内带宽 2 倍，我们的 roofline 大致保持不变。例如对于 TP，临界强度上升到 `2250e12 / 900e9 = 2500`，所以我们有 $Y < F / 2500$ 的限制，只是稍高（除非节点大小增加，否则这对我们没有帮助）。

然而，节点外，缺乏额外带宽实际上使我们更难计算受限！例如，对于数据并行，我们的临界批次大小增加到 `2250e12 / 400e9 = 5625`，因为我们的 GPU 可以用相同的带宽做显著更多的 FLOPs。

带有 72-GPU 节点的 GB200 SuperPod 通过添加更多出口带宽来改变这一点（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题 2 [如何分片 LLaMA-3 70B]：** 考虑 LLaMA-3 70B，在 bfloat16 中使用 fp32 优化器状态和 Adam 训练。

1. 最少需要多少 H100 才能简单地存储权重和优化器？
2. 假设我们想在 4096 H100 GPU 上训练 15T token。假设我们达到 45% MFU（模型 FLOPs 利用率）。需要多长时间训练？
3. LLaMA-3 70B 有 `F = 28,672`，训练批次大小约 400 万 token。我们可以做多少模型并行而不变成通信受限？加上纯 DP，我们能在 4k 芯片上训练 LLaMA-3 同时保持计算受限吗？ZeRO-3 呢？8 路流水线呢？*注意：考虑通信成本和 GPU 内存使用。*

{% details 点击这里查看答案。 %}

1. 我们需要权重 2 字节，优化器状态 8 字节，所以至少 700GB。80GB DRAM，我们最少需要 9 个 GPU，或（四舍五入）至少 2 个 8xH100 节点。这训练起来会永远，也装不下梯度检查点，但这是下界。
2. 这将需要总共 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`。每个 GPU 可以做 `990e12` FLOPs，所以 45% MFU 我们可以做 1.8e18 FLOPs/s。因此整个事情需要 3.5e6 秒，或 40 天。
3. 节点内，我们有 450GB/s 带宽，所以限制大约是 `F / 1995 = 28672 / 1995 = 14.372`。因为这没给我们很多空间，我们可以做节点内 8 路 TP，但仅此而已。对于纯 DP，我们每 GPU 有 `4M / 512 = 7812` 个 token 每 GPU，使用 `512 = 4096 / 8` 是纯 DP GPU 数量。这足够在 roofline 上，但只是勉强。ZeRO-3 是相同的成本，2475，所以我们很好。8 路流水线意味着我们做 `4096 / 64 = 64` 路 DP。这是每 GPU `4M / 64 = 62.5k` token，绰绰有余。

{% enddetails %}

**问题 3 [MoE rooflines]：** 考虑一个假设的 400B 参数 MoE 模型，有 256 个专家，每 token 4 个专家，前馈大小 `F = 2048`，隐藏大小 `D = 8192`。给定这些选择，如果可能的话，我们应该如何分片？考虑一个通用数据中心有 512 个 GPU。假设每个专家的大小是 `D * F * 2 = 32MB`。假设我们有 8% 激活权重，所以总共约 40B 参数。

{% details 点击这里查看答案。 %}

因为 $F$ 相当小，专家并行可能不值得。如果 8 路 TP 在节点内，通信受限约束是 `Y < F * 450e9 / 990e12 = 0.9`，这意味着任何 TP 都是通信受限的！$F$ 只是太小了。如果我们可以让 2 路 TP 工作，我们也许可以做约 2 节点 EP（2x 带宽），这会给我们足够的模型并行来至少存储优化器和权重。

然而，如果 EP 跨越多个节点，这变得有意义。AllToAll 的成本大约是 `2 * B * D * (Z - 8) / Z * min(8 * k / Z, 1) / W`，约等于 B D / (50 * Z) 或更少。因此成本变成 `4 * B * D / (Z * 50e9) < 4 * B * k * D * F / (Z * 990e12)`，简化为 `990e12 < 50e9 * k * F`，或 `k * F > 19800`。对于 k = 4，只有当 `F > 4950` 时我们才能计算受限，这不是我们的情况！所以我们也不能做这个。

因此，我们必须做纯流水线 + 数据并行，或者放弃使用如此小的 $F$。PP 和 DP 是完全独立于 $F$ 的，所以我们应该能够做那个。

{% enddetails %}

**问题 4 [DeepSeek 分析]：** DeepSeek v3 选择做什么？具体来说，考虑 DeepSeek v3 671B 架构，有 256 个专家，每 token 激活 8 个，前馈大小 `F = 2048`，隐藏大小 `D = 7168`。他们使用 `32 + 1` 层，批次大小 `4096 * 15360`。他们在 2048 H800 上训练，每个有 300GB/s 节点内带宽和 300GB/s 节点外带宽。他们做什么规模的张量/专家/流水线并行，为什么？

{% details 点击这里查看答案。 %}

首先，让我们计算 TP 临界点：`F * 300e9 / 990e12 = 0.6`，甚至比上面更低。这意味着 TP 不可能：任何 TP 都是通信受限的。EP 的成本大约是 `2 * B * D * (Z - 8) / Z * min(8 * k / Z, 1) / W`。当 Z = 64 时，这大约是 `2 * B * D * 56/64 * 1 / 300e9 = 5.8e-12 * B * D`。MLP FLOPs 成本是 `2 * 2 * B * k * D * F / (Z * 990e12) = 2.3e-12 * B * D * F`。因此，等式变成 `5.8e-12 < 2.3e-12 * F`，即 `F > 2.5`。

这意味着 64 路 EP 应该工作，但即使这样，EP 也有相当大的固定开销（64 / 2048 = 3% 的芯片），所以 EP 通信成本意味着你不能更高。DeepSeek 确实这样做，用 16 路 PP 来获得 `256 / 16 = 16` 专家每设备。注意他们有多近乎刚好满足 EP 限制！

{% enddetails %}

## 致谢与延伸阅读

衷心感谢 Vedant Sarkar、Jared Davis 和 Karan Desai 对本章草稿的全面反馈。还要感谢 Tian Zheng、Yifeng Lu、Zixuan Jiang、Yunpeng Liu 指出 H100 NCCL 性能基准测试中的一些错误。

**延伸阅读：**

* NVIDIA 针对最新 GPU 架构发布了[架构白皮书](https://resources.nvidia.com/en-us-blackwell-architecture)（每代一个）。
* NVIDIA 还发布了 [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)，这是了解 CUDA 语义的有用资源。
* NVIDIA 的 [SuperPod 网络指南](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)提供了关于节点级别及以上标准网络配置的背景信息。
* NVIDIA 的 [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) 提供了关于集合操作的信息。
* [Chips and Cheese](https://chipsandcheese.com/) 有关于 GPU 微架构的很好信息和分析。
* [GPU Mode](https://www.youtube.com/@GPUMODE) 是一个关于 GPU 特定编程的很好的 YouTube 视频来源。

## 附录

### 附录 A：GB200 有什么变化？

正如上面简要讨论的，GB200 NVL72 大幅增加了节点大小（8 → 72）和给定节点的出口带宽（400GB/s → 3600GB/s）。这显著改变了我们的 roofline，因为我们现在可以做更多的节点内 TP，并且即使跨节点 TP 或 EP，我们的通信成本也可能类似地受节点级别而不是跨节点约束。我们来做数学。

对于 TP，临界点是 `Y < F * 900e9 / 2250e12 = F / 2500`。如果 F = 28672，这意味着 `Y < 11.4`，所以我们可以做约 8 路 TP（唯一正确划分 72 的数）而保持计算受限，但在 72 路时将是通信受限。对于 EP，临界点大约是 `F > 2500`，DeepSeek v3 满足。但是，EP 跨越多个节点实际上可能更难，因为节点出口带宽会阻止一些通信。

对于 DP，临界点变成节点内 `2250e12 / 900e9 = 2500` 或节点外 `2250e12 / 3600e9 = 625`，所以实际上我们在节点级别瓶颈！

### 附录 B：更多网络细节

如上所述，达到 GPU 的峰值带宽需要非常大的消息。与 TPU（相对较快达到峰值带宽）相比，这意味着 GPU 有显著的延迟开销，这在实践中很重要。

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5e 上的 AllReduce 吞吐量。相比 H100 在 1GB 消息时达到峰值，TPU 在约 1MB 消息时达到峰值。" %}

另一个重要的考虑是 NCCL 提供的额外调优旋钮。NCCL 有许多环境变量可以改善延迟或吞吐量，在某些情况下也可以减少。在我们的经验中，NCCL 应该正确"开箱即用"，但在某些情况下你可能想要手动调优。

**参差 AllToAll 理论：** 如上所述，参差 AllToAll 的成本大约是 `(1 - ((Z - 1) / Z)^K) * (Z - 1) / Z * B / (W * Z)`，其中 K 是每 token 选择的专家数，Z 是专家数。这来自于我们有效地做 K 轮骰子投掷并计算得到的不同结果数的期望值。这非常接近 `min(k/Z, 1)`。

**InfiniBand 链接容量如何工作：** InfiniBand 链接使用 64B/66B 编码，这意味着每 66 位传输包含 64 位数据。这意味着 400Gbps 链接实际上传输 `400 * 64 / 66 = 388Gbps` 的有效数据。然而，NVIDIA 报告的带宽数字通常是原始数字而不是有效数字，所以我们在这里使用原始数字。

**问题 3 [跨节点 SHARP AR]：** 考虑一个数组 bf16[D<sub>X</sub>, F<sub>Y</sub>] 在单个节点的 N 个 GPU 上分片。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 需要多长时间？你可以假设我们做网络内归约。解释如果我们有多于一个节点这有什么不同？

{% details 点击这里查看答案。 %}

**答案：** 我们可以尝试修改上面前一个问题的答案。基本上，我们首先从每个 GPU 出口 $B * (X - 1) / XY$ 字节，然后发回 $B / XY$ 到每个 GPU，然后发送相同数量回交换机，然后发送 $B * (X - 1) / XY$ 回每个 GPU。总共是 $NB / Y$ 入口和出口，所以总时间是 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，所以总时间确实随 $Y$ 减少。

如果我们超越单个节点，我们可以做与上面大致相同的归约，但当我们出口节点级交换机时，我们需要发送所有 B 字节，而不仅仅是 $B / Y$。这是因为我们需要保持每个分片分开。

{% enddetails %}

**问题 4 [脊级别 AR 成本]：** 考虑与上面相同的设置，但 $Y = 256$（所以 AR 发生在脊级别）。AllReduce 需要多长时间？同样，随时假设网络内归约。

{% details 点击这里查看答案。 %}

**答案：** 这让我们利用脊级别相当荒谬的带宽量。我们在 4 个节点上有 25.6TB/s 带宽，所以 AllReduce 带宽是 6.4TB/s。使用 SHARP，这可能只需要 `2 * D * F / 6.4e12` 秒。

{% enddetails %}

**问题 5 [2 路 AllGather 成本]：** 计算正好 2 个节点上 $B$ 字节 AllGather 的精确成本。*确保计算精确成本而不是近似值，并考虑节点内和跨节点成本。*

{% details 点击这里查看答案。 %}

**答案：** 在节点级别，我们有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$，而超越我们实际上有 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，我们实际上受节点级归约约束，而不是叶级别！这激发了例如做 2 路数据并行的 DeepSeek v3。

{% enddetails %}
