---
layout: distill
title: "在 TPU 上服务 LLaMA 3-70B"
# permalink: /main/
description: "让我们仔细看看如何在 TPU v5e 上服务 LLaMA 3-70B 模型。在 roofline 下服务不同模型的成本是多少？它们的 KV 缓存有多大？我们应该使用什么批次大小？推理期间参数和激活是如何分片的？让我们通过一些粗略估算来计算生产环境中的延迟和吞吐量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "第7部分：推理"

next_section_url: ../profiling
next_section_name: "第9部分：性能分析"

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
  - name: "LLaMA 服务概述"
  - subsections:
    - name: "思考吞吐量"
    - name: "预填充呢？"
  - name: "可视化延迟吞吐量权衡"
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

*本节将探讨服务 LLaMA-3 需要什么以及如何高效完成。与之前的"实战"章节一样，在查看答案之前，尝试用纸笔自己算出答案！*

## LLaMA 服务概述

让我们回顾一下 LLaMA 3-70B 的样子（参见[第6章](../applied-training)）：

| **超参数**                  | **值**    |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

让我们从一个简单的问题开始：**我们应该在什么硬件上服务？** 答案基本上是，FLOPs / 美元最便宜的那个。<d-footnote>这并不总是正确的，有时更多的 HBM 或 ICI 带宽比 FLOPs 更关键，但这是一个好的启发式方法。</d-footnote> 因此，我们通常希望在 TPU v5e 上服务，这是我们当前专用的推理芯片（成本来自 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元 / 小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每个 TPU v5e 有 16GB 的 HBM，这要求我们相当积极地分片模型。让我们首先思考一些对我们可能重要的基本量：

**问题：** LLaMA 3-70B 的 KV 缓存每 token 有多大？*你可以假设我们用 int8 存储它们。这决定了在给定拓扑上我们的批次大小可以有多大。*

{% details 点击这里查看你思考后的答案！ %}

LLaMA 3-70B 有 8 个 KV 头，所以每 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意这有多大！** 如果我们有 32k token 的序列长度（这很常见），这使用 `162e3 * 32,768 = 5.3GB / 序列`。对于 BS=240，这是 1.3TB！由于 TPU v5e 每个只有 16GB，我们需要大约 `(70e9 + 1.3e12) / 16e9 = 86` 个 TPU v5e 芯片才能容纳这么多内存。还要注意这与 70GB 的模型参数相比有多大。

{% enddetails %}

**问题：** 假设我们想用批次大小 32 和 8192 序列长度服务 L3 70B，所有东西（参数和 KV）都用 int8。这将使用多少总内存？我们能服务的最小切片是什么？

{% details 答案 %}

由于我们的 KV 在 int8 中是 `160e3` 字节，我们的总 KV 内存是 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数是 `70e9` 字节，因为每个参数 1 字节。因此，我们的总内存使用量是 `41.9e9 + 70e9 = 112GB`。

我们能使用的最小切片将有 `112e9 / 16e9 = 7` 个 TPU，或（四舍五入到偶数大小），TPU v5e `4x2`。考虑到其他开销，这将是一个紧凑的配置，我们可能实际上无法完全适应，所以我们可能最少需要 `4x4`（或减少批次大小）。

{% enddetails %}

**问题：** 在 TPU v5e `4x2` 上使用这个批次大小和量化，我们预期每个解码步骤大约需要多少延迟？吞吐量是多少（tokens / sec / chip）？`4x4` 呢？*假设我们用 bfloat16 执行 FLOPs，一切都完全分片。*

{% details 答案 %}

我们可以调用上一节的公式

$$\begin{align*}
\tiny \text{理论步骤时间（通用）} = \underbrace{\frac{\text{批次大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批次大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能是计算受限）}}
\end{align*}$$

这里我们的临界批次大小将约为 120，因为我们的参数是 int8 但 FLOPs 是 bfloat16。我们也可以手动计算右边的最大值，但这基本上是我们已经做过几次的计算。**所以我们对 matmul 和 FLOPs 都处于内存受限区域。**

严格看内存带宽，我们的步骤时间基本上是 `(KV 大小 + 参数大小) / (8 * HBM 带宽) = 112e9 / (8 * 8.1e11) = 17ms`。**所以理论上我们的步骤时间约为 17ms。** 我们的吞吐量将是 `32 / .017 = 1882 tokens / sec`，或 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要检查的注意事项是我们是否可能在 matmul 上是 ICI 受限的。我们可以在这里为它分配 2 个轴，所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时我们是 ICI 受限的，所以我们没问题！

如果我们在 `4x4` 上运行，ICI 方面我们仍然没问题，所以我们的延迟将降至 `17 / 2 = 8.5ms`，但我们每芯片的吞吐量将保持不变。

{% enddetails %}

### 思考吞吐量

让我们花一点时间纯粹思考吞吐量。当我们优化吞吐量时，我们希望是计算受限的，这意味着我们接近利用所有的 TPU MXU 容量。通常这意味着我们希望批次大小尽可能大，这样我们做尽可能多的工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活，我们的批次大小需要多大才能在 matmul 中是计算受限的？如果我们用 int8 权重但在 bfloat16 中执行 FLOPs 呢？int8 权重和 int8 FLOPs 呢？

{% details 答案 %}

如第 7 章所讨论的，对于任何 $B \ll D, F$ 的 bfloat16 matmul，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM 带宽}} = 240
\end{equation*}$$

当我们的权重是 int8 时，我们在分母上失去了 2 倍，所以我们有 $2BDF / DF = 2B > 240$，同样 $B > 120$，是之前临界批次大小的一半。这对我们真的很有帮助！当我们用 int8 权重和 int8 FLOPs 时，我们必须使用 int8 值的 TPU FLOPs/s，从 bfloat16 的 1.97e14 变为 3.94e14，几乎翻倍。这意味着我们回到原点，大约 $B > 240$。

int8 权重和 bfloat16 FLOPs 的情况相当常见，因为无损量化参数通常比做低精度算术更容易。

{% enddetails %}

**问题：** 使用 bfloat16、int8 和 int4（KV 和参数都是）以及 8k 上下文，我们能服务 LLaMA 3-70B 的最小 TPU v5e 拓扑是什么？*对于这个问题，你可以认为 KV 缓存可以忽略不计。*

{% details 答案 %}

这很简单！如果我们接受小批次大小，那么唯一的限制是将参数内存装入 HBM，即只是 `ceil(num_params * sizeof(dtype) / 每个 TPU 的 HBM`，或 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最近的合理拓扑（2 的某个倍数）：

| dtype | 参数大小 | KV 大小 / token (字节) | 最小 TPU v5e | 实际最小切片    | 剩余 HBM 用于 KV 缓存 | 8k 时 KV 缓存数量 |
| :---: | :------: | :--------------------: | :----------: | :-------------: | :-------------------: | :---------------: |
| bf16  |   140GB  |          324kB         |     8.75     |  4x4 = 16 芯片  |          116          |         43        |
| int8  |    70GB  |          162kB         |     4.38     |  4x2 = 8 芯片   |           58          |         43        |
| int4  |    35GB  |          81kB          |     2.81     |  2x2 = 4 芯片   |           29          |         43        |

这很酷！它告诉我们如果我们想的话可以将 LLaMA 70B 放在 TPU v5e 2x2 上。但你会注意到 KV 缓存的数量非常少。那就是我们的批次大小！这意味着我们将获得糟糕的 FLOPs 利用率。我们会很乐意使用更大的拓扑来将批次大小推到 240。

{% enddetails %}

**问题：** 假设我们使用这些拓扑上能容纳的最大批次大小，我们预期每个生成步骤的延迟是多少？

{% details 答案 %}

这也很简单，因为我们选择批次大小来填满所有 HBM！这只是一个问题：将完整的 TPU v5e 的字节加载到 MXU 需要多长时间。这只是 `v5e HBM / v5e HBM 内存带宽 = 16GB / 8.2e11 = 19ms`，所以这是**每步 19ms**。假设我们的生成中位长度是 512 个 token，每次解码大约需要 9s。注意，使用较小的批次大小我们可以获得略微更好的延迟，例如如果我们只看 int4 中的模型参数，我们的最小延迟约为每步 10ms，因为 HBM 不再满了。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们总是可以通过询问从 HBM 将所有模型参数加载到 MXU 需要多长时间来下界解码延迟。当我们的 KV 缓存很小时，你可以将每一层想象成只是逐块加载权重然后丢弃它们。除非我们使用大批次大小或大量设备间通信，这通常是一个合理的界限（在 1.5 倍以内）。当我们的批次大小更大时，我们还需要建模 KV 缓存加载，因为它主导了参数。</p>

同样，在 FLOPs 受限区域（例如训练或大批次推理），我们可以使用 $$\text{总 FLOPs} / (N \cdot C) = 2 \cdot \text{参数数量} \cdot B / (N \cdot C)$$ 下界，这假设没有通信。

**问题：** 对于上面每种情况，这给我们每芯片的吞吐量是多少（以 queries / chip 计）？*你可以假设我们的中位解码长度是 512 个 token。*

{% details 答案 %}

这是一个重要的问题，因为它与每 token 成本完全相关。

根据我们关于中位解码长度的假设，我们的吞吐量只是 $$B / (\text{每步延迟} \cdot \text{中位步数} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$。这给我们大约 $$(4.42 / N)$$ QPS，所以代入 $$N$$ 我们得到：

|  dtype   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

注意这相当乐观，因为它完全忽略了前向传播的工作内存（分配给激活和注意力的内存）。这在 Flash Attention 下不是荒谬的，但也不现实。真实数字可能是这个的约 1/2。为了绝对最大吞吐量，我们可能希望将芯片数量增加一倍以上并显著增加批次大小。

{% enddetails %}

**问题：** 如果我们将上面每个例子的拓扑翻倍，我们的峰值吞吐量会如何变化？

{% details 答案 %}

如果我们在 bfloat16 中使用 4x8 切片，我们将有 372GB 剩余用于 KV 缓存，这将让我们将批次大小提高到 140。然后由于我们的步骤时间保持不变，我们将有 `14.39 / num_chips` 的吞吐量，或

|       dtype       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16（在 4x8 上） |    0.44    |
|   int8（在 4x4 上）   |    0.90    |
|   int4（在 2x4 上）   |    1.80    |

进一步增加会带来更大的收益！重要的要点是**最小拓扑并不总是性能最高的拓扑**，如果我们受 KV 缓存大小限制的话。

{% enddetails %}

**问题：** 现在让我们深入研究分片问题。假设我们想在 TPU v5e 4x8 上用 bfloat16 服务。在生成期间我们会在 TPU v5e 4x8 上使用什么分片？我们能避免成为通信受限吗？

{% details 答案 %}

如上一节所讨论的，生成期间我们实际上只有一个分片选项：模型并行。在我们变成通信受限之前能做多少？正如我们在上一节讨论的，我们的模型大致在以下情况变成通信受限

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B 我们有 `F = 28,672`，所以如果我们做 2 轴模型分片，这给我们大约 $$Y = 28672 \cdot 2 / 2200 = 26$$，所以一般来说我们可以扩展到大约 16 个芯片而不会通信受限，这让我们可以使用 `4x4` 但不能是 `4x8`。一般来说，由于我们不能完美重叠计算，即使这个估计也过于乐观。

**要点：我们实际上不能在纯模型并行下在 4x8 上服务。** 我们能做的最好的是 4x2 或*也许* 4x4。

然而，正如我们讨论过的，当我们的批次大小很小时，我们通常可以做更多模型并行而不会显著损害吞吐量，因为我们的模型是内存带宽受限而不是 FLOPs 受限。我们之前说过这个值大约是 $Y=F / (8\cdot B)$，所以如果我们用批次大小 64，理论上我们可以达到 `Y = 28,672 / (8 * 64) = 56` 路模型并行才会 ICI 受限。为了验证这一点，我们可以看单个 matmul 的 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$。我们显然有：

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`，这将给我们 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`，和 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，所以理论上我们仍然是 HBM 带宽受限的，这很好！*注意从 `4x4` 扩展到 `4x8` 从吞吐量角度可能没有帮助，但它会减少我们的延迟！*

如果我们看 int8 和 int4 配置，我们*可以*用纯模型并行做那些。所以我们已经到了一个点，量化实际上给了我们一个超越更快 FLOPs 的有意义优势：它让我们在变成通信受限之前使用更大的批次大小。**所以这个故事的结局是我们不能在 4x8 上实现峰值吞吐量，但对于 int8 和 int4 配置我们可以做纯模型并行*。

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用的模型并行的最大量取决于 $$d_{ff}$$ 和你分片模型的轴数。最大值通常在 8 到 32 之间，取决于模型大小。你可以扩展超过这个限制以改善延迟，但会有一些吞吐量成本。</p>

### 预填充呢？

我们这里大部分忽略了预填充，因为它简单得多。让我们把几个概念放在一起思考端到端的画面。

**问题：** 假设我们在预填充期间达到 40% 的 FLOPs 利用率。在 16 个 TPU v5e 芯片上，长度为 8192 的预填充需要多长时间？

{% details 答案 %}

在 8k token 时，我们是坚实的计算受限，所以我们只需要推理 FLOPs。我们知道我们的模型有 `70e9` 个参数，所以每次前向传播使用 `2 * 70e9 * B` FLOPs。假设 40% MFU（FLOPs 利用率），这给我们大约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s` 的运行时间。与我们之前看的数字相比，这实际上相当长！

{% enddetails %}

**问题：** 假设我们有 8192 token 的中位预填充长度和 4096 token 的中位解码长度。假设我们有 32 的生成批次大小。平均每步完成多少序列解码？平均每步从我们的 KV 缓存中驱逐多少 token？

{% details 答案 %}

这相当直接。由于我们有 4096 token 的中位解码长度，一个序列大约每 1 / 4096 个 token 完成。给定批次大小 32，这意味着我们每步有 `32 / 4096` 个序列被驱逐。由于我们的 KV 缓存长度大约是 `8192 + 4096`，这是每步 `32 * (8192 + 4096) / 4096 = 96` 个 token 被驱逐。通用公式是 $B * (P + G) / G$，其中 $P$ 和 $G$ 是预填充和生成长度。

{% enddetails %}

**问题：** 假设我们做解聚服务，中位预填充长度 8192，中位解码长度 512。假设上面在 bfloat16 中计算的预填充和生成延迟。你需要什么比例的预填充:生成服务器来保持两者都完全饱和。

{% details 答案 %}

这是一个有趣的问题。设 $P$ 为预填充服务器数量，$G$ 为生成服务器数量。所以一般来说，这是一个流水线问题，我们以 `P / 预填充延迟` 的速率输入序列，以 `B * G / (生成延迟 * 中位解码长度)` 的速率消费它们。我们计算了批次大小 43（我们称之为 32）时预填充每步 `910ms`，解码每步 `19ms`。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`，即我们需要大约 3 倍于生成服务器的预填充服务器！

{% enddetails %}

## 可视化延迟吞吐量权衡

继续以 LLaMA 70B 为例，让我们实际看看生成期间不同批次大小的延迟和吞吐量。正如我们在上一节为 PaLM 模型展示的，这给我们一个吞吐量/延迟的帕累托前沿。让我们假设 16 路张量并行，因为这是我们在 MLP 块中保持计算受限时能使用的合理上限。我们将在这里使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，这样你可以看到更大 KV 缓存的效果。**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **看看成本和延迟之间的权衡有多戏剧性。** 以每 token 延迟翻倍为代价，我们可以实现大约 100 倍的每 token 成本降低。此外，我们的延迟可以从低批次大小时的 5.5ms 到非常大批次时的 20ms 不等。
* 注意在 2k 上下文时，吞吐量在达到 BS 120 roofline 时有效地在大约 1 token / ms / chip 处达到平台期（这里是 120 因为我们用 int8 权重但 bf16 FLOPs）。然而随着序列长度增加，我们不再能在内存中容纳这个批次大小，所以我们永远不会达到完全饱和点。
* 注意在相同吞吐量下，大批次大小时延迟有多高，因为 KV 加载变得主导（而不是参数加载）。

我们可以通过将成本和延迟来源分解为参数加载时间、KV 加载时间和 FLOPs 时间来更好地理解这一点。红色区域是我们预期在 MLP 块中计算受限的区域。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这讲述了一个相当有趣的故事。你可以看到最初，参数加载代表了绝大部分延迟，直到批次大小变得足够大使 FLOPs 和 KV 加载变得更显著。值得注意的是，在所有大于 2048 的序列长度下，我们花在 KV 缓存加载上的时间比花在 FLOPs 上的更多！**所以虽然我们可以通过增加批次大小来提高硬件利用率，但在长上下文长度下 KV 加载始终主导总步骤时间。**

<p markdown=1 class="takeaway">**要点：** 对于 LLaMA 3-70B，我们在几乎所有这些配置中都是强烈的 KV 缓存内存带宽受限（和 HBM 受限），这突出了减少 KV 缓存大小对生成吞吐量的重要性。还要注意延迟/吞吐量权衡在这里仍然有多戏剧性。</p>

{% details 这段代码相当简单。 %}

这是计算这些 roofline 的代码：

```py
import numpy as np

num_chips = 16  # 我们固定 16 作为总模型并行量
param_size = 70e9  # int8 意味着每参数 1 字节
sequence_length = 8192  # 可以变化

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # 获取能容纳的最大批次大小
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # 在 2ND 意义上大致正确

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # 生成时总是带宽受限

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们如何非常明确地将延迟分解为两个来源：KV 加载和参数加载，以及延迟是由 FLOPs 或通信受限，取较大者。

{% enddetails %}

## 练习题

这里有一些练习题。其中一些重复了上面解决的问题，但可能在教学上有用。

**问题 1：** LLaMA 3-405B 每次前向传播每 token 使用多少 FLOPs？假设我们是 FLOPs 受限的，在 TPU v5e 上 N 个芯片上单次前向传播的下界是多少？如果我们是通信受限呢？*忽略模型不适合单个芯片的事实。*

**问题 2：** 假设我们想用 BS240 使用 int8 权重和 int8 KV 缓存服务 LLaMA 3-8B。(a) 模型参数 (b) KV 缓存和 (c) 峰值工作激活（大约）使用多少字节？我们能运行的最小拓扑是什么？

**问题 3：** 你会如何在 TPU v5e 上服务 LLaMA 3-405B？假设 int8 权重和 bfloat16 FLOPs。假设我们有 15ms / token 的硬性限制，我们能实现的最高吞吐量配置是什么？理论最小步骤时间是多少？

<h3 markdown=1 class="next-section">第 8 部分到此结束！第 9 部分将深入探讨 XLA 和 TPU 性能分析，请点击[这里](../profiling)。</h3>
