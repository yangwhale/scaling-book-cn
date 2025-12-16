---
layout: distill
title: "用 JAX 编程 TPU"
# permalink: /main/
description: "手把手教你用 JAX 操控 TPU！本节大部分内容参考自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>官方文档</a>。你可以在 <a href='https://colab.sandbox.google.com/'>Google Colab</a> 上白嫖免费 TPU 来跑这些代码。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "第9部分：性能分析"

next_section_url: ../conclusion
next_section_name: "第11部分：结论"

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "JAX 并行编程三板斧"
  - subsections:
    - name: "自动挡：让编译器帮你搞定"
    - name: "半自动挡：JAX 控制分片传播"
    - name: "手动挡：shard_map 全手写"
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

## JAX 并行编程三板斧

> **一句话总结**：JAX 提供三种并行编程模式——你可以完全信任编译器、半信任让 JAX 管分片、或者全手动自己写通信。选哪个取决于你想省心还是想精细控制。

**先打个比方**：假设你要指挥一个 8 人乐队演奏交响乐。

| 模式 | 比喻 | 你的工作量 |
|:---:|:---|:---:|
| **自动模式** | 请一个指挥家，你只管写谱子 | 最少 |
| **显式模式** | 你是指挥，但乐手自己看谱协调 | 中等 |
| **手动模式** | 你亲自指挥每个乐手的每个动作 | 最多 |

更技术一点说，JAX 支持三种思想流派：

**1. 🤖 自动挡："编译器，你来掌舵！"**
- 把单机代码直接扔给 XLA 编译器
- 编译器自动决定怎么切分数据、加什么通信
- 好处：代码零改动就能跑在 1000 张卡上
- 坏处：编译器有时候会"抽风"，加一些莫名其妙的通信

**2. 🚗 半自动挡："JAX，帮我盯着！"**
- 你写单机代码，JAX 负责传播分片信息
- 分片信息变成类型系统的一部分
- 遇到模糊情况，JAX 会报错让你明确
- 比自动挡可控，比手动挡省事

**3. 🏎️ 手动挡："老子自己来！"**
- 你拿到的是每个设备的本地视图
- 所有通信（AllGather、AllReduce 等）自己写
- 完全掌控，但写起来费劲
- 适合性能极致优化场景

| 模式 | 你看到的视图 | 需要指定分片？ | 需要写通信？ |
|:---:|:---:|:---:|:---:|
| 自动 | 全局（整个数组） | ❌ | ❌ |
| 显式 | 全局（整个数组） | ✅ | ❌ |
| 手动 | 本地（当前设备的那块） | ✅ | ✅ |

对应的 JAX API：

| 模式 | API | 特点 |
|:---|:---|:---|
| 自动 | `jax.jit` + Auto mesh | XLA [Shardy](https://openxla.org/shardy) 自动加通信 |
| 显式 | `jax.jit` + Explicit mesh | JAX 追踪分片，遇到歧义报错 |
| 手动 | `jax.shard_map` | 本地视图，手写 `lax.all_gather`/`lax.psum` 等 |

<h3 id="auto-sharding-mode">自动挡：让编译器帮你搞定</h3>

> **核心思想**：你写正常的 JAX 代码，告诉 JAX 输入输出怎么分片，剩下的交给 XLA 编译器。

`jax.jit` 在 JAX 里其实干两件事：
1. **JIT 编译**：把 Python 函数编译成高效的机器码
2. **自动并行**：如果输入是分片的，自动在多设备间分发计算

**来看个例子**——分片矩阵乘法：

```py
import jax
import jax.numpy as jnp

# 假设在 TPU v5e 4x2 上跑，8 个芯片排成 4 行 2 列
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 告诉 JAX 后面都用这个 mesh
jax.set_mesh(mesh)

# 创建分片的输入和权重
# In: [8, 2048] 沿 X 切 4 份（行），沿 Y 切 2 份（列）
# W: [2048, 8192] 只沿 Y 切 2 份（行），列方向不切
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 编译！指定输出分片为 P('X', None) 表示行切 X 份，列不切（复制）
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

**底层发生了什么？**

让我们用之前学过的符号来理解：

| 张量 | 全局形状 | 分片 | 每个设备上的形状 |
|:---|:---|:---|:---|
| In | [8, 2048] | [B<sub>X</sub>, D<sub>Y</sub>] | [2, 1024] |
| W | [2048, 8192] | [D<sub>Y</sub>, F] | [1024, 8192] |
| Out | [8, 8192] | [B<sub>X</sub>, F] 复制 | [2, 8192] |

因为 In 和 W 在收缩维度 D 上都被 Y 轴切分了，本地 matmul 得到的是**部分和**，需要 AllReduce 才能得到完整结果：

```
1. Out[B_X, F] { 部分和 } = In[B_X, D_Y] × W[D_Y, F]   # 本地 matmul
2. Out[B_X, F] = AllReduce(Out[B_X, F])                # 跨 Y 轴求和
```

用 `jit_matmul.as_text()` 可以看到生成的 HLO：

```py
# matmul 融合操作
%fusion = bf16[2,8192] fusion(bf16[2,1024] %param, bf16[8192,1024] %copy-done)

# AllReduce 求和
ROOT %AllReduce = bf16[2,8192] AllReduce(bf16[2,8192] %fusion)
```

注意形状：`bf16[2, 1024]` 是本地激活（全局 8 被 4 切分成 2，全局 2048 被 2 切分成 1024）。

**这就是 magic！** 不管你的程序多复杂，[Shardy](https://openxla.org/shardy) 都会尝试：
- 为所有中间激活找到合适的分片
- 自动插入必要的通信操作

**但编译器有时会"抽风"**

Shardy 不是完美的。有时你打开 profile 一看——我擦，一个巨大的 AllGather 占了 80% 的时间，但其实根本不需要！

这时候可以用 `jax.lax.with_sharding_constraint` 来"纠正"编译器：

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  # 强制 hidden 沿 y 维度分片（编译器本来可能选别的分片）
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

**自动模式的痛点**："调教编译器"是个玄学活儿。你可以标注每个中间变量的分片，但还是不确定最终会不会得到想要的结果。能不能让 JAX 自己管分片传播呢？

<h3 id="explicit-sharding-mode">半自动挡：JAX 控制分片传播</h3>

> **核心思想**：分片信息变成类型系统的一部分。JAX 会追踪每个操作的分片，遇到歧义就报错让你明确。

显式分片（Explicit Sharding）又叫"类型中的分片"——分片传播在 JAX 层面完成，而不是交给 XLA。

**看个例子**：

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 创建 2x2 mesh，注意 axis_types 是 Explicit
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y] ← 分片信息直接在类型里！
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y] ← 逐元素操作保持分片
  return out

f(x)
```

**JAX 怎么传播分片？**

每个 JAX 操作都有分片规则：
- **逐元素操作**（加减乘除）：输出分片 = 输入分片（很显然）
- **规约操作**（sum、mean）：可能改变分片
- **矩阵乘法**：可能有歧义，需要你明确

**歧义情况——JAX 会报错**：

```py
# 创建分片的输入和权重
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))   # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 💥 报错！
```

报错信息很清楚：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
收缩维度被分片了，输出分片不明确。请用 out_sharding 参数指定。
```

**为什么有歧义？** 因为输出可以是：
- `P('X', 'Y')` → 触发 ReduceScatter
- `P('X', None)` → 触发 AllReduce

自动模式会随便选一个（可能选错），显式模式让你自己决定：

```py
@jax.jit
def matmul_square(In, W):
  # 明确告诉 JAX：我要输出沿 X 和 Y 都分片
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

**自动 vs 显式可以混用**

通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` 可以在同一程序里混合使用。详见[官方文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)。

<h3 id="manual-sharding-mode-via-shard_map">手动挡：shard_map 全手写</h3>

> **核心思想**：你拿到每个设备的本地视图，所有通信自己用 `lax` 原语写。完全掌控，但也完全负责。

**jax.jit vs jax.shard_map**：

| 对比 | jax.jit | jax.shard_map |
|:---|:---|:---|
| 你看到的 | 全局数组 | 本地分片 |
| 通信 | 编译器自动加 | 你手动写 |
| 控制力 | 低 | 高 |
| 难度 | 简单 | 困难 |

**看个例子**——在每个设备上取前 4 个元素，然后全局平均：<d-footnote>没有 TPU？用这行模拟：`import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# 这个函数在每个设备上只看到 1/8 的数据！
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)  # 每个设备只有 64 个元素
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))  # 手动写通信！

out = slice_and_average(x)
assert out.shape == (4,)
```

**这代码干了啥？**

1. 全局数组 `x` 有 512 个元素
2. 被 8 个设备（2×4）分片，每个设备只看到 64 个元素
3. 每个设备取自己那份的前 4 个元素
4. `pmean` 是手动的 AllReduce，对所有设备的这 4 个元素求平均

实际效果：`mean(x[:4], x[64:68], x[128:132], ...)`

**为什么不用 jax.jit？**

用 jax.jit，你看到的是全局的 `[512]` 数组，要切出这种"每 64 个取前 4 个"的模式很别扭，而且编译器可能加错通信。用 shard_map，你直接操作本地数据，需要什么通信自己加。

---

### 实战：Collective Matmul（通信计算重叠）

这是 shard_map 最经典的应用场景。

**问题背景**：模型并行时，激活是分片的：

```
A[B_X, D_Y] × W[D, F_Y] → Out[B_X, F_Y]
```

**朴素做法**——先 AllGather，再 matmul：

```
1. A[B_X, D] = AllGather_Y(A[B_X, D_Y])   # 先收集完整激活
2. Out[B_X, F_Y] = A[B_X, D] × W[D, F_Y]  # 再做 matmul
```

**问题**：通信和计算完全串行，效率低下！

**Collective Matmul**——边通信边计算（参考 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)）：

核心思想：
1. 把 W 按 Y 轴分成若干块
2. 每一步：做一块 matmul + 把 A 往下一个设备传
3. 通信和计算完美重叠！

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

# TPU v5e-8 或用 jax.config.update('jax_num_cpu_devices', 8) 模拟
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  """边传数据边算，通信计算重叠"""
  axis_size = jax.lax.axis_size('Y')  # Y 轴有 4 个设备
  idx = jax.lax.axis_index('Y')        # 当前设备在 Y 轴的位置

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    # 从 W 中取出对应块
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 本地 matmul
    update = lhs @ rhs_chunk
    # 把 lhs 向左循环移位（传给相邻设备）
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # 处理最后一块
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

**性能对比**：

| 版本 | 耗时 | Profile 特征 |
|:---|:---:|:---|
| 朴素 jit matmul | 311us | 开头有大块阻塞 AllGather |
| Collective matmul | 244us | 没有独立通信，全是有用计算 |
| 无分片基线 | 224us | 纯 matmul |

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}
<center><i>朴素版本：开头的大块蓝色是 AllGather，计算在等通信</i></center>

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}
<center><i>Collective 版本：没有独立通信，FLOPs 利用率暴涨</i></center>

我们达到了接近无分片基线的性能！这就是性能优化的威力。更多 shard_map 例子见[官方笔记](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

---

## 练习题

> **准备工作**：这些题需要多个 TPU。可以用免费的 Colab TPUv2-8，或者用 `jax.config.update('jax_num_cpu_devices', 8)` 模拟。

---

### 问题 1：分片内平均 & 分片内 Roll

设 **A** 是形状为 `float32[S_X, D_Y]` 的数组，被分片到 N = X × Y 个设备上。

**(a) 分片内平均**

写一个函数：返回形状 `[X, Y]` 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 上数据的平均值。

要求：
- 分别用 `jax.jit` 和 `shard_map` 实现
- 用 profiler 看看耗时
- 检查有没有不必要的通信（理论上不应该有！）

**(b) 分片内 Roll**

写一个函数：在每个 X 分片内做 `roll(x, shift, axis=0) - x`。

只需要用 `shard_map` 实现（jit 版本太折磨人了）。

{% details 点击查看答案 %}

**(a) 分片内平均**

注意 jit 版本需要做复杂的 reshape：

```py
import numpy as np
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X','Y'))

# shard_map 版本：直接对本地数据求平均
average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

# jit 版本：需要手动 reshape 来模拟分片
def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

# 测试
x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

**(b) 分片内 Roll**

```py
import numpy as np
import jax
import jax.numpy as jnp
import functools

P = jax.sharding.PartitionSpec
mesh = jax.make_mesh((4, 2), ('X','Y'))

# shard_map 版本：直接对本地数据 roll
def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

# jit 版本：reshape 后在正确维度上 roll
@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

# 测试
x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

---

### 问题 2：混合专家（MoE）实现

这题一起来实现一个基础的 MoE 层。

**设定**：
- **W**: `float32[E_X, D, F]` — E 个专家矩阵
- **A**: `float32[S_X, D]` — 输入激活
- **B**: `int32[S_X]` — 路由分配，`B[i]` 告诉我们第 i 个 token 该用哪个专家

**目标**：返回 `Out[i] = W[B[i]] @ A[i]`

**(a) 本地实现**

先忽略分片，在单设备上实现。

⚠️ **不要**具体化 `[S, D, F]` 形状的数组！

提示：把 token 排序到 `[E, S, D]` 缓冲区，用 mask 处理。

**(b) 直接 jit**

用 `jax.jit` 包装你的实现，profile 看看编译器加了什么通信，耗时多少？

**(c) shard_map 实现**

你会发现 jit 版本可能在本地 AllGather 完整激活 A，通信和内存都很贵。用 `shard_map` 重写：

1. 第一步：用 `jax.lax.all_gather` 收集后重排序
2. 进阶：避免具体化 `[E, S, D]` 数组，用 `jax.lax.while_loop` + `jax.lax.all_to_all` 做不规则计算

比原始版本快多少？

**(d) Top-K 路由**

大多数 MoE 路由到多个专家再平均。让 **B**: `int32[S, k]`，实现 top-k 路由。

{% details 点击查看部分答案 %}

**(a/b) 本地实现**

有很多方法，这是用 mask 迭代专家的版本：

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - 这个专家对所有 token 的变换
        output = output + expert_result * mask  # 只保留分配的 token
        return output, None

    output = jnp.zeros((S, F))
    output, _ = lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

你也可以用 `jax.lax.ragged_dot`，更高效。

**(c) shard_map 伪代码**

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # 根据分配排序 x
  while (chunk := x[i:i+chunk_size].any()):
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
  return concat(out)
```

核心思想：迭代数组的块，排序 + all_to_all，然后做本地 FLOPs。

{% enddetails %}

---

### 问题 3：Collective Matmul 进阶

上面的 collective matmul 例子对真实 LLM 非常有用。来扩展一下：

**(a) AllReduce Collective Matmul**

实现：`A[B_X, D_Y] ×_D W[D_Y, F] → Out[B_X, F]`

朴素版本是本地 matmul + AllReduce。实现通信重叠版本。

提示：在输出维度上分块，边算边 `jax.lax.psum`。

注意：由于 XLA 优化，可能不比基线快。

**(b) ReduceScatter Collective Matmul**

这是 AllReduce 的补充，发生在 Transformer 的 down-projection：

`Tmp[B_X, F_Y] ×_F W2[F_Y, D] → Out[B_X, D_Y]`

实现通信重叠版本，只传必要的数据量。

提示：累积时置换结果。

**(c) 端到端 Transformer 块**

把 (a) 和 (b) 组合起来：

`In[B_X, D_Y] ×_D W_in[D, F_Y] ×_F W_out[F_Y, D] → Out[B_X, D_Y]`

<d-footnote>记住：不能先算 W_in · W_out，因为中间有非线性！</d-footnote>

比 jit 版本快多少？

---

### 问题 4：双向通信优化

上面的 collective matmul 都是单向置换。改成双向通信。

- 重写 AllReduce collective matmul
- 重写 ReduceScatter collective matmul

快了多少？

---

### 🎉 第 10 部分完结！

基本内容就是这些了。要看最终总结和进一步阅读材料，请点击[这里](../conclusion)。
