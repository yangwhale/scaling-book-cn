
---
layout: distill
title: "如何并行化 Transformer 训练"
# permalink: /main/
description: "训练大模型，一张卡肯定不够。这章我们聊聊四种『分而治之』的方法：数据并行、FSDP、张量并行、流水线并行。每种方法各有利弊，关键是搞清楚什么时候通信会拖后腿。"
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
  - name: "什么是『扩展』"
  - subsections:
    - name: "数据并行"
    - name: "全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "FSDP + 张量并行混合用"
    - name: "流水线并行"
    - name: "跨 Pod 扩展"
  - name: "TPU 训练要点速查"
  - name: "练习题"
  - name: "附录"
  - subsections:
    - name: "附录 A：反向传播的通信推导"

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

## 什么是『扩展』

**核心问题**：我有一堆芯片，怎么让它们高效协作？

理想情况下，芯片数量翻倍，训练速度也翻倍——这叫**强扩展**。但现实没这么美好。芯片越多，它们之间的"沟通成本"也越高。如果沟通时间超过了干活时间，加再多芯片也是浪费。

> 打个比方：一个人搬砖很慢，两个人可以快一倍。但如果 100 个人挤在一起搬，光是"你往左我往右"的协调就够呛，效率反而可能下降。

**本章目标**：搞清楚四种"分活儿"的方法，以及每种方法什么时候会被"沟通成本"拖累。

### 四种并行策略一览

| 策略 | 一句话解释 |
|:-----|:----------|
| **数据并行** | 每张卡都有完整模型，各自算不同的数据，最后把梯度汇总 |
| **FSDP（ZeRO）** | 模型参数切成碎片分给各卡，用的时候再拼起来 |
| **张量并行** | 每个矩阵乘法都分给多张卡一起算，算完再合并 |
| **流水线并行** | 模型按层切开，数据像流水线一样一层层往下传 |

### 符号约定

为了后面计算方便，我们统一用这些符号：

**模型参数**：

| 符号 | 含义 |
|:-----|:-----|
| D | 隐藏维度（d_model） |
| F | FFN 中间维度（d_ff，通常是 4D 或 8D） |
| B | 批次大小（总 token 数，不是每卡的） |
| T | 序列长度 |
| L | 层数 |

**硬件参数**：

| 符号 | 含义 |
|:-----|:-----|
| C | 每芯片的 FLOPs/s |
| W | 网络带宽（双向），比如 $W_{ici}$ 表示 ICI 带宽 |
| X, Y, Z | 网格各轴的芯片数 |

### 简化模型

为了聚焦核心问题，我们做两个简化：

1. **把 Transformer 简化成一堆 MLP**：注意力只占小头，FFN 才是大头
2. **每层就两个矩阵**：上投影 W_in 和下投影 W_out

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图：</b>简化后的 Transformer 层。每层就两个矩阵：W<sub>in</sub>（D→F）负责升维，W<sub>out</sub>（F→D）负责降维。" %}

{% details 未并行化的基础算法（供参考）%}

<div markdown=1 class="algorithm">

**前向传播**：

1. Tmp[B, F] = In[B, D] × W_in[D, F]
2. Out[B, D] = Tmp[B, F] × W_out[F, D]
3. Loss[B] = ...

**反向传播**：

1. dOut[B, D] = ...
2. dW_out[F, D] = Tmp^T × dOut
3. dTmp[B, F] = dOut × W_out^T
4. dW_in[D, F] = In^T × dTmp
5. dIn[B, D] = dTmp × W_in^T

</div>

{% enddetails %}

---

### 数据并行

> **一句话**：每张卡都有完整模型，各自算不同的数据批次，最后平均梯度。

**分片公式**：$$\text{In}[B_X, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

B_X 表示把 B 切成 X 份，每卡只处理 B/X 的数据。

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b>数据并行示意。输入按批次切分（左边），权重完全复制（右边）。前向传播不需要通信，反向传播时做一次 AllReduce 汇总梯度。" %}

**工作流程**：
1. 前向传播：各卡独立算，互不通信 ✅
2. 反向传播：算完本地梯度后，做 AllReduce 平均
3. 更新权重：每卡独立更新（因为梯度一样，更新后权重还是一样）

{% details 完整算法 %}

<div markdown=1 class="algorithm">

**前向传播**（无通信）：

1. Tmp[B_X, F] = In[B_X, D] × W_in[D, F]
2. Out[B_X, D] = Tmp[B_X, F] × W_out[F, D]
3. Loss[B_X] = ...

**反向传播**：

1. dOut[B_X, D] = ...
2. dW_out_local = Tmp^T × dOut
3. dW_out = **AllReduce**(dW_out_local)  ← 可以异步
4. dTmp = dOut × W_out^T
5. dW_in_local = In^T × dTmp
6. dW_in = **AllReduce**(dW_in_local)  ← 可以异步
7. dIn = dTmp × W_in^T

</div>

{% enddetails %}

**优点**：
- 实现简单，前向传播零通信
- AllReduce 可以异步执行，不阻塞后续计算

**缺点**：
- 每张卡都要存完整模型 + 优化器状态
- 内存占用 = 参数数 × 10 字节（bf16 参数 + fp32 优化器）
- TPU v5p（96GB HBM）最多只能放 90 亿参数的模型

<p markdown=1 class="takeaway">**要点**：数据并行最大能训练的模型 ≈ HBM 容量 ÷ 10。对于 TPU v5p，约 90 亿参数。</p>

#### 什么时候会被通信拖累？

计算时间：$$T_{计算} = \frac{8 \cdot B \cdot D \cdot F}{X \cdot C}$$

通信时间：$$T_{通信} = \frac{8 \cdot D \cdot F}{W_{ici}}$$

（8 = 2 个矩阵 × 2 次 AllReduce × 2 字节）

要想计算受限（通信能被计算掩盖），需要：

$$\frac{B}{X} > \frac{C}{W_{ici}}$$

翻译成人话：**每卡的批次大小，要超过"ICI 算术强度"**。

对于 TPU v5p：
- C = 4.6×10¹⁴ FLOPs/s
- W_ici = 1.8×10¹¹ bytes/s
- C/W = 2550

也就是说，**每卡至少要处理 2550 个 token**，否则就会被通信拖累。

如果用三个轴都做数据并行，带宽变成 3 倍，阈值降到 850。但即使这样，一个 pod（8960 芯片）也需要 760 万 token 的批次才能跑满。

**结论**：纯数据并行被通信卡住的情况其实不多见！

<p markdown=1 class="takeaway">**上下文并行**：这里的 B 是"总 token 数"。MLP 不在乎 token 是来自同一个序列还是不同序列，所以可以沿序列维度做数据并行（叫"上下文并行"）。注意力需要特殊处理（环形注意力），但 MLP 完全不用管。</p>

---

### 全分片数据并行（FSDP）

> **一句话**：不光切数据，连模型参数和优化器状态也切了。用的时候再临时拼起来。

**分片公式**：$$\text{In}[B_X, D] \cdot W_\text{in}[D_X, F] \cdot W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

注意：权重的收缩维度（D）也被切了！

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b>FSDP 分片示意。权重沿收缩维度切分，使用前需要 AllGather 拼起来。这样内存省了，但通信多了。" %}

**核心思想**：

还记得吗？AllReduce = AllGather + ReduceScatter。

既然反向传播要做 AllReduce，不如这样：
- 把模型切成碎片存着（省内存）
- 前向传播时 AllGather 拼起来用
- 反向传播时 ReduceScatter 分摊梯度

**通信量完全一样**，但内存省了 X 倍！这就是为什么叫"ZeRO"（Zero Redundancy Optimizer）。

{% details 完整算法 %}

<div markdown=1 class="algorithm">

**前向传播**：

1. W_in[D, F] = **AllGather**(W_in[D_X, F])  ← 可提前异步
2. Tmp[B_X, F] = In[B_X, D] × W_in[D, F]
3. W_out[F, D] = **AllGather**(W_out[F, D_X])  ← 可提前异步
4. Out[B_X, D] = Tmp[B_X, F] × W_out[F, D]

**反向传播**：

1. dW_out_full = Tmp^T × dOut
2. dW_out[F, D_X] = **ReduceScatter**(dW_out_full)  ← 可异步
3. W_out[F, D] = **AllGather**(W_out[F, D_X])  ← 反向需要权重
4. dTmp = dOut × W_out^T
5. dW_in_full = In^T × dTmp
6. dW_in[D_X, F] = **ReduceScatter**(dW_in_full)  ← 可异步
7. W_in[D, F] = **AllGather**(W_in[D_X, F])
8. dIn = dTmp × W_in^T

</div>

{% enddetails %}

**ZeRO-1/2/3 是什么？**
- ZeRO-1：只切优化器状态
- ZeRO-2：切优化器 + 梯度
- ZeRO-3：切优化器 + 梯度 + 权重（最省内存）

通信量都一样，所以一般直接用 ZeRO-3。

#### 什么时候会被通信拖累？

和数据并行**完全一样**！因为 AllReduce = AllGather + ReduceScatter，通信总量没变。

$$\frac{B}{X} > \frac{C}{W_{ici}} = 2550$$

<p markdown=1 class="takeaway">**要点**：FSDP 和数据并行的通信门槛一样，但 FSDP 省内存。如果你的数据并行能跑，换成 FSDP 只有好处没坏处！</p>

**实际例子**：

DeepSeek-V2 用了 4000 万 token 的批次。这意味着可以扩展到约 47000 芯片（~5 个 TPU v5p pod）而不被通信限制。

LLaMA-3 70B 用 1600 万 token 批次，可以分到约 18000 芯片（~2 个 pod）。

<p markdown=1 class="takeaway">**临界批次大小**：有个反直觉的事实——批次越小，越容易被通信卡住。因为通信量是固定的（和模型大小相关），但计算量随批次变小。这就是为什么 DeepSeek 等模型用超大批次训练。</p>

---

### 张量并行

> **一句话**：不切数据，切模型。每个矩阵乘法都让多张卡一起算。

**分片公式**：$$\text{In}[B, D_Y] \cdot W_\text{in}[D, F_Y] \cdot W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

用 Y 表示张量并行轴（后面会和 FSDP 的 X 轴组合）。

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图：</b>张量并行示意。权重沿 F 维度切分，激活沿 D 维度切分。每次矩阵乘法前要 AllGather 激活。" %}

**工作流程**：
1. AllGather 拼出完整的 In[B, D]
2. 每卡算 In × W_in_local → Tmp_local
3. 每卡算 Tmp_local × W_out_local → Out_local
4. ReduceScatter 把 Out 切回去

**关键区别**：FSDP 移动的是**权重**，张量并行移动的是**激活**。

{% details 完整算法 %}

<div markdown=1 class="algorithm">

**前向传播**：

1. In[B, D] = **AllGather**(In[B, D_Y])  ← 关键路径
2. Tmp[B, F_Y] = In[B, D] × W_in[D, F_Y]
3. Out_partial = Tmp × W_out[F_Y, D]
4. Out[B, D_Y] = **ReduceScatter**(Out_partial)  ← 关键路径

**反向传播**：同理，也需要 AllGather 和 ReduceScatter

</div>

{% enddetails %}

**巧妙之处**：

两个矩阵配合得刚刚好！
- W_in 把 D 变成 F_Y（每卡算一部分 F）
- W_out 把 F_Y 变回 D（部分结果，需要加起来）

这样，一进一出正好配对：进的时候 AllGather D，出的时候 ReduceScatter D。

#### 什么时候会被通信拖累？

$$T_{计算} = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}$$

$$T_{通信} = \frac{4 \cdot B \cdot D}{W_{ici}}$$

要计算受限：$$F > Y \cdot \frac{C}{W_{ici}} = Y \times 2550$$

也就是说，**张量并行的路数不能超过 F / 2550**。

<p markdown=1 class="takeaway">**要点**：张量并行最多做到 F / 2550 路。对于大多数模型（F≈30000），就是 8-16 路。再多就会被通信卡住。</p>

**实际例子**：
- LLaMA-3 70B：F≈30000 → 最多 11 路，实际用 8 路
- Gemma 7B：F≈50000 → 最多 19 路，可以用 16 路

**有趣的是**：这个门槛和批次大小无关！因为通信量和计算量都与 B 成正比，抵消了。

---

### FSDP + 张量并行混合用

> **一句话**：两个维度一起切，既省内存又能用小批次。

**分片公式**：$$\text{In}[B_X, D_Y] \cdot W_\text{in}[D_X, F_Y] \cdot W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

X 轴做 FSDP，Y 轴做张量并行。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b>混合分片示意。模型参数在两个轴上都切分了，没有任何冗余。" %}

**为什么要混合？**

- 张量并行移动激活，激活大小 ∝ B
- FSDP 移动权重，权重大小与 B 无关

当 B 变小时：
- 张量并行的通信变少（激活变小了）
- FSDP 的通信不变（权重还是那么大）

所以，**批次小的时候多用张量并行，批次大的时候多用 FSDP**。

{% details 完整算法 %}

<div markdown=1 class="algorithm">

**前向传播**：

1. In[B_X, D] = **AllGather_Y**(In[B_X, D_Y])  ← 关键路径
2. W_in[D, F_Y] = **AllGather_X**(W_in[D_X, F_Y])  ← 可提前
3. Tmp[B_X, F_Y] = In × W_in
4. W_out[F_Y, D] = **AllGather_X**(W_out[F_Y, D_X])  ← 可提前
5. Out_partial = Tmp × W_out
6. Out[B_X, D_Y] = **ReduceScatter_Y**(Out_partial)  ← 关键路径

</div>

{% enddetails %}

#### 最优比例是多少？

设 N = X × Y 是总芯片数，最优 FSDP 分片数是：

$$X_{opt} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}$$

其中 M_X、M_Y 是各方向的网格轴数（大约各占一半，乘积约为 2）。

**实际例子**：
- N = 64 芯片（4×4×4）
- B = 48000 token
- F = 32768

代入公式：X ≈ 14，所以用 X=16 做 FSDP，Y=4 做张量并行。

#### 什么时候会被通信拖累？

$$\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

其中 α = C/W ≈ 2550。

代入 F=32000, M_X M_Y=2：

$$\frac{B}{N} > \frac{2550^2}{2 \times 32000} \approx 100$$

<p markdown=1 class="takeaway">**要点**：混合 FSDP+TP 可以把每芯片批次降到约 100 token！这比纯 FSDP 的 850 小了 8 倍多。</p>

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b>不同策略的计算/通信比。混合策略在中等批次大小时表现最好。纯 FSDP 在大批次时最好，纯 TP 有固定上限。" %}

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图：</b>不同策略的通信时间对比。黑色虚线是计算时间，高于这条线就是通信受限。混合策略（绿线）在最大范围内保持计算受限。" %}

下面是交互式演示，可以拖动滑块调整批次大小：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

---

### 流水线并行

> **一句话**：按层切模型，数据像流水线一样一层层传下去。

GPU 世界用得很多，TPU 上不太必要（因为 ICI 带宽够大）。

**基本流程**：
1. TPU 0 算第 0~3 层，结果传给 TPU 1
2. TPU 1 算第 4~7 层，结果传给 TPU 2
3. ...以此类推
4. 反向传播时倒着来

{% details Python 伪代码 %}

```python
batch_size = 32
d_model = 128
num_layers = len(jax.devices())

x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

# 前向传播
for i in range(num_layers):
    x = x @ weights[i]
    if i != num_layers - 1:
        x = jax.device_put(x, jax.devices()[i+1])

# 反向传播
loss, dx = jax.value_and_grad(loss_fn)(x)
for i in range(num_layers-1, -1, -1):
    _, f_vjp = jax.vjp(layer_fn, intermediates[i+1], weights[i])
    dx, dw = f_vjp(dx)
    if i != 0:
        dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**优点**：
- 阶段间只需传激活，通信量小
- 对低带宽互联（如 GPU 间）很友好

**缺点**：
- **流水线气泡**：TPU 0 大部分时间在等着！第一层很快算完，然后就闲着等反向传播

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b>DeepSeek v3 的流水线调度。橙色=前向，绿色=dL/dx，蓝色=dL/dW。通过精心排列可以消除气泡。" %}

**解决气泡的方法**：
1. **微批处理**：把批次切成小块，流水线送入，让各 TPU 都忙起来
2. **操作重叠**：把 dx 计算和 dW 计算巧妙交错，填满空闲时间

因为 TPU 有很强的 ICI，流水线并行不是必需品。我们一般用 FSDP + TP 就够了。

---

### 跨 Pod 扩展

一个 TPU v5p Pod 最大 8960 芯片。想要更多？得走 DCN（数据中心网络）。

**DCN 带宽**：每 TPU 约 6.25GB/s（比 ICI 慢 30 倍）

**常见策略**：
- Pod 内：FSDP + 张量并行
- Pod 间：纯数据并行（跨 DCN 做 AllReduce）

#### 什么时候被 DCN 卡住？

$$\frac{B}{切片数} > \frac{C}{W_{dcn}} = \frac{4.6 \times 10^{14}}{6.25 \times 10^9} \approx 71000$$

也就是说，**每个 Pod 至少要处理 7 万多 token**，否则 DCN 带宽不够用。

<p markdown=1 class="takeaway">**要点**：跨 Pod 数据并行，每 Pod 需要至少 7 万 token 的批次。</p>

**实际例子**：

训练 LLaMA-3 70B，批次 200 万 token：
- 单 Pod（8k 芯片）：可以用 FSDP + TP，每芯片 ~250 token，刚好够
- 两个 Pod：每 Pod 100 万 token，远超 7 万，DCN 不会是瓶颈

---

## TPU 训练要点速查

### 核心原则

1. **芯片越多 or 批次越小 → 越容易被通信卡住**
2. 对于合理的序列长度（~32k），可以把 Transformer 简化成一堆 MLP 分析
3. 四种并行策略各有适用场景

### 策略公式速查

| 策略 | 分片公式 |
|:-----|:---------|
| 数据并行 | In[B_X, D] · W_in[D, F] · W_out[F, D] → Out[B_X, D] |
| FSDP | In[B_X, D] · W_in[D_X, F] · W_out[F, D_X] → Out[B_X, D] |
| 张量并行 | In[B, D_Y] · W_in[D, F_Y] · W_out[F_Y, D] → Out[B, D_Y] |
| FSDP + TP | In[B_X, D_Y] · W_in[D_X, F_Y] · W_out[F_Y, D_X] → Out[B_X, D_Y] |

### 计算量和通信量

| 策略 | 每层计算 | 每层通信（字节，前向+反向） |
|:-----|:---------|:---------------------------|
| DP | 12BDF/X | 0 + 8DF |
| FSDP | 12BDF/X | 4DF + 8DF |
| TP | 12BDF/Y | 4BD + 4BD |
| FSDP+TP | 12BDF/(XY) | (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) |

### 门槛速查

| 策略 | 通信受限条件 | TPU v5p 数值 |
|:-----|:------------|:-------------|
| DP/FSDP | B/X < C/W_ici | 每卡 < 2550 token（单轴）<br>每卡 < 850 token（三轴） |
| 张量并行 | Y > F/2550 | 超过 8-16 路 |
| FSDP+TP | B/N < α²/(2F) | 每卡 < 100 token |
| 跨 Pod | B/Pod < C/W_dcn | 每 Pod < 71000 token |

### 内存估算

- 参数（bf16）：2 字节/参数
- 优化器（Adam, fp32）：8 字节/参数
- **总计**：10 字节/参数

TPU v5p（96GB）最多放 90 亿参数（纯数据并行）

### 实用建议

1. **模型小、批次大**：FSDP 就够了
2. **模型中等、批次中等**：FSDP + 8~16 路张量并行
3. **超大规模**：多 Pod + DCN 数据并行

---

## 练习题

用 LLaMA-2 13B 作为例子：

| 参数 | 值 |
|:-----|:---|
| L（层数） | 40 |
| D（隐藏维度） | 5120 |
| F（FFN 维度） | 13824 |
| H（头维度） | 128 |
| V（词表大小） | 32000 |

**问题 1**：验证一下参数量确实是 130 亿。

{% details 答案 %}

- FFN 参数：3LDF = 3 × 40 × 5120 × 13824 = 85 亿
- 注意力参数：4DNHL = 4 × 5120 × 40 × 40 × 128 = 42 亿
- 词表参数：2VD = 2 × 32000 × 5120 = 3.3 亿
- **总计**：85 + 42 + 3.3 ≈ 130 亿 ✓

{% enddetails %}

**问题 2**：用 BS=1600 万 token 和 Adam 训练，总内存需求是多少？

{% details 答案 %}

参数 + 优化器：(2 + 4 + 4) × 13×10⁹ = 130GB

激活（每层存 3 个检查点）：
- 每层：2 × (B×D + 2×B×F) = 2 × 16×10⁶ × (5120 + 2×13824) ≈ 1TB
- 40 层：40TB

**总计**：约 42TB

{% enddetails %}

**问题 3**：在 TPU v5p 16×16×16（4096 芯片）上，用 300 万 token 批次训练：

a) 能用纯数据并行吗？
b) 能用纯 FSDP 吗？
c) 应该怎么配置 FSDP + TP？

{% details 答案 %}

a) **不能**。纯数据并行需要每卡存完整模型（130GB），但 TPU v5p 只有 96GB。

b) **勉强不行**。内存没问题（300万 token 只需要 ~8TB 激活），但：
   - 每卡批次 = 300万 / 4096 = 732
   - 门槛 = 2550 / 3 = 850（三轴）
   - 732 < 850，会被通信卡住

c) **可以用混合策略**：
   - 门槛 = 2550² / (2×13824) ≈ 235
   - 每卡批次 = 732 > 235 ✓
   - 最优 X = √(300万 × 2 × 4096 / 13824) ≈ 1333
   - 实际配置：X=1024（FSDP），Y=4（TP）
   - 预计步骤时间：6 × 300万 × 13×10⁹ / (4096 × 4.6×10¹⁴ × 0.4) ≈ 300ms

{% enddetails %}

<h3 markdown=1 class="next-section">下一章我们用这些知识来[实际训练 LLaMA](../applied-training)！</h3>

---

## 附录

### 附录 A：反向传播的通信推导

前向传播：Out = In × W_in × W_out

反向传播需要计算四个量：

1. dW_out = Tmp^T × dOut（Tmp = In × W_in）
2. dTmp = dOut × W_out^T
3. dW_in = In^T × dTmp
4. dIn = dTmp × W_in^T

**如何确定通信**：
1. 写出每个矩阵乘法需要的操作数
2. 根据并行策略确定每个操作数的分片
3. 应用分片矩阵乘法的规则

关键洞察：dOut 的分片方式和 Out 相同（都是输出），所以反向传播的通信模式和前向传播对称。