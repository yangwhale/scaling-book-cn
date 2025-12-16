---
layout: distill
title: "用 JAX 编程 TPU"
# permalink: /main/
description: "如何使用 JAX 高效地编程 TPU！本节的大部分内容取自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>。你可以在 <a href='https://colab.sandbox.google.com/'>Google Colab</a> 上使用免费 TPU 运行本节中的代码示例。"
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
  - name: "JAX 中的并行化如何工作？"
  - subsections:
    - name: "自动分片模式"
    - name: "显式分片模式"
    - name: "通过 shard_map 的手动分片模式"
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

## JAX 中的并行化如何工作？

JAX 支持三种多设备编程的思想流派：

1. **编译器，你来掌舵！** 让 XLA 编译器自动分区数组并决定添加什么通信来促进给定程序。这让你可以将在单个设备上运行的程序自动运行在数千个设备上而不需要改变任何东西。
2. **JAX，你来掌舵！** 自动并行化很棒，但有时编译器会做一些疯狂的事情。显式分片让你像往常一样编写单设备代码，但让 JAX 处理分片传播（而不是编译器）。这意味着当不清楚你想要什么时，JAX 可以请求你澄清。
3. **让我写我想写的，该死的！** 虽然编译器很好，但它们有时会做错事并添加你不想要的通信。有时我们想明确地说明你打算运行什么通信。

| 模式 | 视图？ | 显式分片？ | 显式集合操作？ |
|:---:|:---:|:---:|:---:|
| 自动 | 全局 | ❌ | ❌ |
| 显式 | 全局 | ✅ | ❌ |
| 手动 | 每设备 | ✅ | ✅ |

相应地，JAX 为每种模式提供了 API：

1. `jax.jit`（带 `Auto` mesh 轴）让你可以取任何现有的 JAX 函数并用分片输入调用它。然后 JAX 使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器自动并行化程序。XLA 会在需要时为你添加通信（AllGather、ReduceScatter、AllReduce 等）以促进现有操作。虽然它不完美，但它通常在不改变代码的情况下自动将程序扩展到任意数量的芯片方面做得相当不错。
2. 带 `Explicit` mesh 轴的 `jax.jit` 看起来类似于 (1)，但让 JAX 而不是 XLA 处理分片传播。这意味着数组的分片实际上是 JAX 类型系统的一部分，当 JAX 检测到模糊的通信时可以报错并让用户解决。
3. `jax.shard_map` 是更手动的对应物。你获得程序的设备本地视图，必须显式编写任何你想要的通信。有一个分片数组想要在每个设备上获得整个数组？添加一个 `jax.lax.all_gather`。想要在设备间求和数组？添加一个 `jax.lax.psum`（即 AllReduce）。编程更难但更不可能做你不想要的事情。

<h3 id="auto-sharding-mode">自动分片模式</h3>

jax.jit 在 JAX 中扮演两个角色。顾名思义，它"即时"编译一个函数从 Python 到字节码（通过 XLA/HLO/LLO）以便运行得更快。但如果输入是分片的或用户指定了 `in_sharding` 或 `out_sharding`，它还让 XLA 在多个设备间分发计算并根据需要添加通信。例如，以下是如何使用 jax.jit 编写分片 matmul：

```py
import jax
import jax.numpy as jnp

# 在 TPU v5e 4x2 上运行。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 这告诉 JAX 对所有操作使用这个 mesh，所以你只需指定 PartitionSpec P。
jax.set_mesh(mesh)

# 我们创建矩阵 W 和输入激活 In，在我们的设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在这里显式编译分片的 matmul 函数。这添加了所有
# 必要的通信（例如 matmul 后的 AllReduce）。
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将自动使用任何分片运行并在我们的设备间分区计算。**但在硬件级别实际发生了什么？**

1. 首先我们创建在我们设备间分片的 In 和 W<d-footnote>注意我们是如何做的。这是创建具有特定分片的数组的一种方式（即通过向创建函数添加 device 参数）。另一种是正常用 `jnp.array(....)` 创建数组，然后做例如 `jax.device_put(..., jax.P('x', 'y'))`。还有一种是编写创建你想要的数组的函数，并用你想要的 `out_shardings` jit 编译它。</d-footnote>。W 沿收缩维度 2 路分片，而 In 4 路分片（沿收缩和输出维度）。这对应于分片 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>]，即一种模型和数据并行。
2. 如果我们在本地运行这个（即在一个设备上），`matmul_square` 将简单地对输入求平方并执行简单的 matmul。但因为我们将 `out_shardings` 指定为 `P('X', None)`，输出将沿批次分片但在模型维度上复制，需要 AllReduce 来计算。

使用我们前面章节的符号，这可能会做类似于

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会自动为我们添加这个！我们实际上可以用 `jit_matmul.as_text()` 打印 HLO 并看到以下 HLO（大幅简化）：

```py
# 这个融合是分片输入和矩阵的实际 matmul
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们在设备间归约部分求和的结果
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以在上面看到 matmul（融合）和 AllReduce。特别注意形状。`bf16[2, 1024]` 是激活的本地视图，因为我们的 `batch_size=8` 在 4 个设备间分割，我们的 `d_model=2048` 同样分割 2 路。

**这相当神奇！** 无论我们的程序多复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试为所有中间激活找到分片并根据需要添加通信。话虽如此，Shardy 有其缺陷。它可能会犯错。有时你会查看 profile 并注意到某些事情出了问题。一个巨大的 AllGather 占用了 profile 的 80%，而它本不需要。当这种情况发生时，我们可以尝试通过使用 `jax.lax.with_sharding_constraint` 显式标注中间张量来纠正编译器。例如，对于两个 matmul，我可以用以下方式强制中间激活沿 `y` 维度分片（不是说这是个好主意）：

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

这大约占自动分区世界中 JAX 并行编程的 60%，你通过 `jax.lax.with_sharding_constraint` 控制中间分片。但"编译器调教"出了名的不是一个有趣的编程模型。你可以标注每个中间变量，仍然不知道是否会得到正确的结果。相反，如果 JAX 本身可以处理和控制分片传播呢？

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片（或"类型中的分片"）看起来很像自动分片，但分片传播发生在 JAX 级别！每个 JAX 操作都有一个分片规则，它接收操作参数的分片并为操作结果产生分片。你可以使用 `jax.typeof` 查看结果分片：

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 在 TPU v5e 2x2 上运行。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# 这告诉 JAX 对所有操作使用这个 mesh，所以你只需指定 PartitionSpec P。
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)
```

如你所见，JAX 将分片从输入（`x`）传播到输出（`x`），这些在追踪时可以通过 `jax.typeof` 检查。对于大多数操作，这些规则简单明了，因为只有一个合理的选择（例如逐元素操作保留相同的分片）。但对于某些操作，如何分片结果是模糊的，在这种情况下 JAX 抛出追踪时错误，我们要求程序员显式提供 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。让我们看另一个有冲突的例子：

```py
# 我们创建矩阵 W 和输入激活 In，在我们的设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 这将报错
```

这段代码报错 `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)`

这很棒，因为 einsum 输出应该如何分片是模糊的。输出分片可以是：
* P('X', 'Y') 这将引发 reduce-scatter 或
* P('X', None) 这将引发 all-reduce

与自动模式不同，显式模式在检测到模糊通信时报错并要求用户解决。所以这里你可以做：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

自动模式和显式模式可以通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` API 组合。[这是一个很好的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)可以阅读更多信息。

<h3 id="manual-sharding-mode-via-shard_map">shard_map：对程序的显式并行控制</h3>

虽然 Shardy 是"编译器掌舵"模式，jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 将一切都放在你手中。你像在 jax.jit 中一样指定输入的分片，但然后你显式编写所有通信。而 `jax.jit` 给你程序的全局跨设备视图，`shard_map` 给你本地每设备视图。

这是一个例子。试着推理这个函数做了什么：<d-footnote>如果你想在 colab 中通过模拟 mesh 自己玩这个，你可以使用以下单元格 `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# 这个函数将操作数组的 1/8。
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这做了什么？** `slice_and_average` 在每个 TPU 上用数组的 1/8 运行，从中我们切片前 4 个元素并在整个 mesh 上平均它们。这意味着我们实际上在做 `mean(x[:4], x[64:68], x[128:132], …)`。这很酷，因为在 JAX 中用其他方式表达这个不是一个容易的操作。

**为什么用这个而不是 jax.jit？** 如果我们用 `jax.jit`，`slice_and_average` 会看到数组的全局视图（完整的 `[512,]` 数组）。我们必须切出这个不均匀的切片然后执行平均，XLA 必须正确解释。XLA 可能添加了错误的通信或搞混了。这里我们看到本地视图并只编写我们需要的通信。

**示例 [Collective Matmul]：** 举一个更现实的例子，假设我们要实现模型并行，其中激活最初是模型分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。朴素地，我们会先 AllGather A 然后做本地矩阵乘法：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是，这很糟糕因为它不允许我们重叠通信和计算。重叠它们可以用"collective matmul"完成，如 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 所述。算法基本如下：

* 对于每个 Y 分片，执行 A 的本地块与 W 的本地块的 matmul，产生形状为 `[B / X, F / Y]` 的结果。同时，置换 A 以便你在本地获得下一个块，执行 matmul，并对结果求和。

我们可以用 `jax.shard_map` 相当容易地实现：

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

# 这是为了在 TPU v5e-8 运行时上运行。如果你无法获得这个，
# 尝试设置 jax.config.update('jax_num_cpu_devices', 8)。
#
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
  # lhs 是循环操作数；rhs 是本地操作数
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 对于这个例子
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 一个块的 Matmul
    update = lhs @ rhs_chunk
    # 向左循环移位
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # 在最终置换后计算最后一个块以保持 lhs 处于我们发现它的状态
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

这相当整洁！我们可以对其进行基准测试并看到它也快得多！[这是](https://imgur.com/a/e9I6SrM)默认 jit matmul 的 profile，需要 311us，开头有一个大的阻塞 AllGather：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

[这是](https://imgur.com/a/21iy0Sv)上面的版本，需要 244 us。你可以看到 profile 没有 AllGather。全是有用的工作！我们的 FLOPs 利用率也高得多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还值得注意的是，没有收缩维度分片的 matmul 时间是 [224us](https://imgur.com/a/i3gNKfq)，所以我们这里非常接近未分片的基线。这是你可能最终做的性能工程的一个好例子，以提高 TPU 利用率。更多 `shard_map` 示例，[这个笔记很棒](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

现在这里有几个有用的练习题，尝试使用 `jax.jit` 或 `shard_map` 实现！

## 练习题

这里有一些随机的 JAX 相关问题。我稍后会添加更多。对于所有这些，你需要 Colab 中一定数量的 TPU。你可以使用带有 TPUv2-8 的公共 Colab。从现在开始，我们假设你有 N 个可用设备。

**问题 1：** 设 **A** 是形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活数组，其中 `X * Y = N`。做以下事情：

1. 用 JAX 编写一个函数，计算每个 `(X, Y)` 分片内的平均值，即它返回一个大小为 [X, Y] 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 上的平均值。用 `jax.jit` 和 `shard_map` 都做一遍。分析每个并看看它们花了多长时间。有没有添加任何通信？*提示：不应该有，但有时 XLA 还是会添加。*

2. 用 JAX 编写一个函数，对某个 shift **在每个分片 X 内**返回 roll(x, shift, axis=0) - x。我不够受虐狂让你用 jax.jit 做这个，所以只用 `shard_map` 做。

{% details 点击这里查看答案。 %}

第 1 部分：这是第 1 部分的解决方案。注意我们必须为 `jax.jit` 解决方案做相当复杂的 reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第 2 部分：这是第 2 部分的类似解决方案。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2：** 这里我们一起做一个基本的"混合专家"模型。设 **W**: float32[E<sub>X</sub>, D, F] 是一组 E 个"专家"矩阵。设 **A**: float32[S<sub>X</sub>, D]（我们的激活）和设 **B**: int32[S<sub>X</sub>] 是一组"路由分配"，其中 B[i] 是范围 `[0, E)` 内的整数，告诉我们想用哪个矩阵处理那个激活。我们想在 JAX 中编写一个函数返回 `Out[i] = W[B[i]] @ A[i]`。

1. 让我们先完全忽略分片。让所有这些张量足够小以便它们适合一个设备。编写这个函数的本地实现。*确保你不会具体化一个形状为 `[S, D, F]` 的数组！提示：尝试将 token 排序到形状为 `[E, S, D]` 的新缓冲区中，注意掩码（为什么我们需要第二个维度的大小是 S？）。*

2. 如果你只是 `jax.jit` 上述方法，会发生一些事情。分析这个并看看它决定做什么通信。需要多长时间？

3. 你会注意到上述的一个问题是它可能在本地 gather 完整的激活集 **A**，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不仅在通信上很昂贵，如果我们不能在本地容纳完整的激活集，在内存上也非常昂贵。使用 `shard_map` 和显式通信实现上述功能。

      1. 作为第一步，可能最容易使用 `jax.lax.all_gather` 并像 (a) 中那样重新排序。

      2. 作为第二步，尝试避免具体化任何大小为 `[E, S, D]` 的数组，即尝试使用 `jax.lax.while_loop` 内的 `jax.lax.all_to_all` 以不规则方式执行计算。这样，你可以避免具体化完整激活并在填充上浪费计算。这比你原始实现快多少？

4. 大多数 MoE 路由到多个 (k) 专家然后平均结果。重构上述以实现这个。在这种情况下设 **B**: int32[S, k] 为要路由到的 k 个专家。

{% details 点击这里查看（部分）答案。 %}

1/2. 对于第 (1) 部分，你有很多选择。这是一个只是用掩码迭代专家的选项。

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - 这个专家对所有 token 的变换
        output = output + expert_result * mask  # 只保留分配的 token 的结果
        return output, None

    output = jnp.zeros((S, F))
    output, _ = lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

你也可以使用 `jax.lax.ragged_dot`，它会做类似的事情但更高效。

3. 我只会在这里勾勒伪代码（如果你有一个干净的解决方案，随时可以添加）：

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

基本思想是迭代数组的块，排序它们并做 all_to_all，然后做本地 FLOPs。

{% enddetails %}

**问题 3：** 上面的 collective matmul 示例实际上对真实 LLM 非常相关。让我们调整这个例子来做完整的 Transformer 栈。

1. 作为练习，让我们首先实现一个 AllReduce collective matmul，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出不是复制的。朴素算法在上面讨论过，基本上只是本地 matmul 后跟 AllReduce。尝试制作这个操作的通信重叠"collective"版本。*提示：在输出维度上分块，随时使用 `jax.lax.psum`（即 AllReduce）。* *注意：由于 XLA 处理这个的方式，它实际上可能不比基线快。*

2. 上面 AllReduce collective matmul 的补充是 ReduceScatter collective matmul，如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。这发生在 Transformer 的 down-projection 矩阵中。在 JAX 中实现这个的 collective、重叠版本。注意只传递你需要的最少数据量。*提示：尝试在累积时置换结果。*

3. 将这两个放在一起，形成一个端到端的 Transformer 块，执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，带有重叠通信。<d-footnote>和以前一样，我们不能先做 $W_{in} \cdot W_{out}$ 因为这里省略了非线性。</d-footnote> 这比 `jax.jit` 实现快多少？

**问题 4：** 上面实现的所有 collective matmul 都是单向的：它们只在一个方向置换。重写 collective AllReduce matmul 和 collective ReduceScatter matmul 以使用双向通信。这些快多少？

### 第 10 部分到此结束。基本上就是这样！要阅读最终结论和进一步阅读，请点击[这里](../conclusion)。
