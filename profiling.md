---
layout: distill
title: "如何分析 TPU 程序性能"
# permalink: /main/
description: "前面几章全是理论推导。理论能让你走很远，但真到了优化的时候，还得看实际情况：XLA 编译器干了什么？哪里慢了？这章教你用 Profiler 工具找出问题所在。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "第8部分：服务 LLaMA"

next_section_url: ../jax-stuff
next_section_name: "第10部分：JAX"

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
  - name: "TPU 软件栈全景"
  - name: "TensorBoard Profiler 使用指南"
  - subsections:
    - name: "Trace Viewer：时间线视图"
    - name: "怎么读 HLO 代码"
    - name: "Graph Viewer：计算图视图"
    - name: "实战：看一个真实 Profile"
    - name: "Memory Profile：内存视图"
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

## 为什么需要 Profiling？

前面几章我们一直在做"纸上谈兵"：用 roofline 模型估算性能上限。

但实际优化时，你需要知道：
- XLA 编译器到底生成了什么代码？
- 时间花在了哪里？
- 内存用了多少？

这就需要 **Profiler**（性能分析器）。

---

## TPU 软件栈全景

写 TPU 程序有几个层次，从高到低：

| 层次 | 是什么 | 谁用 |
|:-----|:-------|:-----|
| **JAX** | NumPy 风格的高级 API | 大多数程序员 |
| **StableHLO** | 平台无关的中间表示 | XLA 编译器 |
| **HLO** | 硬件相关的中间表示 | Profiler 显示的 |
| **LLO** | 低级优化器，直接操作 TPU | 内部 |
| **机器码** | TPU 执行的二进制 | 内部 |

我们写的是 JAX，看的是 HLO。

### 一个简单例子

```python
import jax
import jax.numpy as jnp

def multiply(x, y):
    return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

`jax.jit` 告诉 JAX：追踪这个函数，编译成高效代码。

编译后的 HLO 大概长这样：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1),
       lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

别慌，这其实很好读。`dot.4` 就是那个矩阵乘法，沿着维度 0 和 1 收缩。

**当程序跑得慢时**，我们用 Profiler 看 HLO 层面发生了什么。如果 HLO 层面都解决不了，就用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 写自定义内核。

---

## TensorBoard Profiler 使用指南

### 怎么生成 Profile

```python
import jax

with jax.profiler.trace("/tmp/tensorboard"):
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 1024))
    y = x @ x
    y.block_until_ready()  # 等待计算完成

# 然后在终端运行：
# tensorboard --logdir=/tmp/tensorboard

# 或者在 Colab 里：
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
```

### Profiler 能看什么？

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

三个最有用的标签：

| 标签 | 看什么 |
|:-----|:-------|
| **Trace Viewer** | 时间线，看每个操作花了多久 |
| **Graph Viewer** | 计算图，看操作之间怎么连接 |
| **Memory Profile** | 内存使用随时间的变化 |

想先体验一下？这里有个在线 Perfetto 链接：[简单 Transformer 的 Trace](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)

或者用这个 [Colab](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8) 生成完整 Profile 自己玩。

---

### Trace Viewer：时间线视图

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

这是一个 Transformer 的 Trace。可以看到：

1. **顶行（XLA Ops）**：实际的 TPU 操作，名字是 HLO 名字
2. **重复的块**：每个重复就是一层
3. **点击操作**：可以看到对应的代码位置

<p markdown=1 class="takeaway">**导航技巧**：W/S 放大缩小，A/D 左右移动。像游戏一样！</p>

---

### 怎么读 HLO 代码

看到这种东西不要怕：

```
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{...} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

拆开来看：

| 部分 | 含义 |
|:-----|:-----|
| `fusion.3` | 操作名 |
| `bf16[32,32,4096]` | 输出类型和形状 |
| `{2,1,0:T(8,128)(2,1)}` | 内存布局和 tiling |
| `S(1)` | 存储位置：S(0)=HBM, S(1)=VMEM |
| `fusion(...)` | 输入参数 |
| `kind=kCustom` | 操作类型 |

**关于 Tiling**

`{1,0:T(2,2)}` 是什么意思？

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

- `1,0`：维度在内存中的顺序（从右往左读）
- `T(2,2)`：以 2×2 块 tiling
- 数组会被填充到能被 tiling 整除

更复杂的例子：`bf16[4,8]{1,0,T(2,4)(2,1)}`

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

两层 tiling：外层 2×4，内层 2×1（bf16 需要 4 字节对齐）。

**为什么 Tiling 重要？**

有时候 XLA 会插入"重新布局"操作来调整 tiling，这会带来开销。如果你在 profile 里看到很多 `copy` 操作，可能就是这个问题。

---

### Graph Viewer：计算图视图

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

HLO 操作太复杂？Graph Viewer 把它可视化了。

鼠标悬停在节点上，可以看到对应的代码行。

多盯着看几遍，试着把 HLO 操作和你的代码对应起来。

---

### 实战：看一个真实 Profile

这是一个假 Transformer 的 Profile：

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

用 [这个 Colab](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8) 自己生成一个来玩。

#### FFN 块分析

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

放大 FFN 块。up-projection 操作：

- 输入：`bf16[8, 1024, 8192]` × `bf16[8192, 16384]`
- 输出：`bf16[32, 1024, 16384]`

这是 4 路 DP + 2 路 TP 分片后的本地视图。全局形状：

**X**: bf16[32, 1024, 8192] × **W_in**: bf16[8192, 32768] → **Tmp**: bf16[32, 1024, 32768]

**验算一下时间**：

每分片批次 = 8 × 1024 = 8192 token → 计算受限 ✓

理论时间 = 2 × 32 × 1024 × 8192 × 32768 / (23e12 × 8) = **95.6ms**

实际时间 = **96ms**

几乎完美命中 roofline！

#### 通信分析

第二个 matmul 末尾有个小操作：

```
%fusion.1 = bf16[8,1024,4096]{...} fusion(...), kind=kCustom, calls=%all-reduce-scatter.1
```

这是个 ReduceScatter。

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

**验算**：

数组大小 = 2 × 32 × 1024 × 8192 = 537MB（全局）
每分片 = 537 / 4 = 134MB

单跳 ICI 带宽 = 1.2e11 B/s

理论时间 = 134e6 / 1.2e11 = **1.1ms**

实际时间 = **1.13ms**

又命中了！

#### 注意力块分析

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

Q 投影用的矩阵：[d_model=8192, n_heads=32, d_qkv=256]

沿头维度做 Megatron 分片。

试试自己验算这些操作应该花多久？

---

### Memory Profile：内存视图

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

这个视图显示内存使用随时间的变化。

例子里可以看到：
- 模型参数约 7.5GB
- 空闲约 10GB

对调试 OOM 很有帮助：找到峰值是在哪里，是什么操作导致的。

---

## 练习题

### 问题 1：找 Bug

看看[这个 Colab/Profile](https://colab.sandbox.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli)：

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

**任务**（先不要看代码！只看 Profile）：
- 这是什么计算？
- 每个矩阵的真实形状是什么？
- 怎么分片的？

{% details 答案 %}

这是两个矩阵乘法：

```python
def matmul(w1, w2, x):
    return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

Profile 里可以看到：reduce → 两个大 fusion → all-reduce

第一个 fusion：
```
%fusion.1 = bf16[4096]{...} fusion(bf16[4096,8192]{...} %param.1, bf16[8192]{...} %reduce.6)
```

每个分片：`bf16[8192] × bf16[4096, 8192] → bf16[4096]`

AllReduce 的 replica_groups 显示 8 组 → 8 路张量并行

全局形状：`bf16[8, 8192] × bf16[32768, 8192] → bf16[8, 32768]`

{% enddetails %}

---

### 问题 2：优化 Transformer

用[这个 Colab](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8) 里的简单 Transformer：

1. 生成基准 Profile
2. 每个部分花了多久？应该花多久？
3. 用了什么分片策略？
4. 用 `jax.lax.with_sharding_constraint` 试着优化

**参考数据**：
- 初始版本：约 184ms/层（现在 XLA 更好了，约 90ms/层）
- 优化后：约 67ms/层（现在约 80ms/层）

完成后，纯从 Profile 回答：
- 什么分片策略？
- 批次大小、d_model、d_ff 是多少？
- 注意力 vs MLP 各占多少时间？
- 按 roofline，应该各占多少？

---

<h3 markdown=1 class="next-section">下一章我们深入看 [JAX 并行化](../jax-stuff)！</h3>
