### Edit Flows 是什么

Edit Flows 把序列生成建模为序列空间上的**连续时间马尔可夫链**（CTMC）。状态是序列，跃迁是编辑操作：

- insert
- delete
- substitute

模型在每个位置预测：

- 三类编辑的总速率 `λ_ins / λ_sub / λ_del`
- 插入 token 分布 `Q_ins`
- 替换 token 分布 `Q_sub`

从而定义所有可能编辑的瞬时速率。

### Edit Flows 中的训练步骤

引入增广空间 $(\mathcal{Z} = (\mathcal{T} \cup \{\varepsilon\})^N)$，其中 $\varepsilon$ 是特殊的 **gap token**（不在真实词表中，代码中为 `GAP_TOKEN = 130`）。

#### 对齐 (Alignment)

将 $x_0, x_1$ 对齐到等长的 $z_0, z_1$。对齐方式决定了编辑操作的解释：

- $z_0[i] = \varepsilon, z_1[i] = c$ → 需要**插入**
- $z_0[i] = c, z_1[i] = \varepsilon$ → 需要**删除**
- $z_0[i] = c_1, z_1[i] = c_2$ → 需要**替换**

在 Z 空间中，条件路径定义为逐 token 独立的混合：

$$
p_t(z_t^i \mid z_0^i, z_1^i) = (1 - \kappa_t) \cdot \delta_{z_0^i} + \kappa_t \cdot \delta_{z_1^i}
$$

简单来说 $z_t$ 就是按 $\kappa_t$ 的概率选择是 $z_0$ 还是 $z_1$，而 $x_t$ 就是 $z_t$ 移除掉所有的 gap tokens。其中 $\kappa_t$ 为时间的调整函数（下文称之为scheduler），原论文中建议为 $\kappa_t = t^3$。

#### 训练loss

$$
\mathcal{L}(\theta) = \mathbb{E} \left[ \sum_{x \neq x_t} u_t^\theta(x|x_t) -
\sum_{i=1}^N \mathbf{1}[z_1^i \neq z_t^i] \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \log u_t^\theta(x(z_t, i, z_1^i) \mid x_t) \right]
$$

简单来说，网络需要对 $x_t$ 中的每一个token预测每一种编辑的速率。假设v是真实的编辑速率，例如对于 $x_t$ 来说这个token后面需要再添加三个token `C`才能和 $x_1$ 一致，则"插入token `C`"这个操作的速率即为3；若需要删除，则"删去此token"的速率即为1。假设 $u^\theta$ 为网络预测的速率，则训练loss的不严谨的简单版本为：

$$
\mathcal{L}(\theta) = u^\theta - \frac{\dot{\kappa}_t}{1-\kappa_t} v \cdot \log u^\theta
$$

求导计算后可知最优 $u^\theta$ 为 $ \frac{\dot{\kappa}_t}{1-\kappa_t} v $。因此可以说我们希望网络预测的就是真实的编辑速率（乘一个系数）。

### Edit Flows 中的采样步骤

从 $t=0$ 开始，用一阶 Euler 步近似求解 CTMC，直到 $t=1$：

```
1. 初始化: x_0 ~ p(x_0), t = 0
2. While t < 1:
   a. 模型前向: λ_ins, λ_sub, λ_del, Q_ins, Q_sub = model(x_t, t)
   b. 自适应步长: h_adapt = min(h, (1-κ_t)/κ̇_t)
   c. 每种编辑独立采样:
      P(ins)  = 1 - exp(-h_adapt · λ_ins)
      P(del/sub) = 1 - exp(-h_adapt · (λ_sub + λ_del))
      P(del) = P(del/sub) · λ_del/(λ_sub + λ_del)
      P(sub) = P(del/sub) · λ_sub/(λ_sub + λ_del)
   d. 插入/替换从 Q_ins/Q_sub 采样具体 token
   e. 并行应用所有编辑操作
   f. t += h_adapt
```

简单来说，由于预测的速率与真实速率相似，我们就可以相应地按照速率的大小来选择进行编辑操作的概率，然后就可以进行采样了。

### 初始实验

在之前r-smiles处理的两种数据集上，直接进行训练与采样。结果为：standard Top-1 22.2%，#global# Top-1 40.6%。之后修复了模型的一个实现错误，#global# Top-1 提升到了 43.519%。（接下来反馈的数据都为#global# Top-1的结果）。

### Oracle 分析

上文已经说明了，模型预测的速率的最优解即为真实的速率，那我们直接用真实速率进行采样结果会怎么样呢？这样可以先排除模型训练的问题，来分析是否采样的相关代码有没有写错，以及采样时使用的kappa函数该如何选择。

#### Standard 数据集 (USPTO_50K_PtoR_aug20)

| Metric | Cubic | Linear | 变化 |
|--------|-------|--------|------|
| Top-1 Acc | 93.2% | **97.4%** | **+4.2pp** |
| Top-2 Acc | 99.4% | 99.9% | +0.5pp |
| Top-3 Acc | 99.8% | 100.0% | +0.2pp |
| Invalid SMILES | 22.6% | **9.2%** | **-59%** |
| Unique Rates | 14.8% | 12.2% | — |

#### #global# 数据集 (USPTO_50K_PtoR_aug20_#global#)

| Metric | Cubic | Linear | 变化 |
|--------|-------|--------|------|
| Top-1 Acc | 93.4% | **98.6%** | **+5.2pp** |
| Top-2 Acc | 99.5% | 100.0% | +0.5pp |
| Top-3 Acc | 100.0% | 100.0% | — |
| Invalid SMILES | 9.0% | **2.6%** | **-71%** |
| Unique Rates | 15.1% | 11.9% | — |

发现oracle的整体表现很好，说明采样的相关逻辑没有问题。此外，发现linear scheduler（即 $\kappa_t = t$）整体结果更好，这很可能是由于相比于cubic scheduler，linear scheduler让采样步骤更加平均，而不是集中在一开始。

### scheduler 无关的训练过程与模型

上文有说原论文中的训练loss为（省去 $\theta$，并设 $k_t = \frac{\dot{\kappa}_t}{1-\kappa_t}$）：

$$
\mathcal{L} = u - k_t v \cdot \log u
$$

设 $k_t u' = u$, 则：

$$
\mathcal{L} = k_t u' - k_t v \cdot \log(k_t u') = k_t (u' - v \cdot \log u') - k_t v \cdot \log k_t
$$

其中 $ k_t v \cdot \log k_t $ 为一个模型无关的常数，可以丢弃。此时经过求导推导后最优的 $u$ 为 $v$。

接下来设 $l_t = u' - v \cdot \log u'$，为时间刻 $t$ 时的损失（这里省去了对于不同token位置、不同操作、不同样本的平均）：
$$
\mathcal{L'} = \int_0^1 k_t l_t \ dt = \int_0^1 \frac{\dot{\kappa}_t}{1-\kappa_t} l_t \ dt = \int_0^1 \frac{1}{1-\kappa_t} l_t \ d\kappa_t 
$$

因此我们发现其实$\kappa_t$只是作为一个采样密度调整的作用（例如cubic就会在接近0的部分采样的更密集），但是在数学上来说在概率期望意义上是没有区别的。

因此这里有几点可以变动的地方：

1. 让模型预测 $u'$ 而不是 $u$。因为 $u'$ 没有乘 $k_t = \frac{\dot{\kappa}_t}{1-\kappa_t}$ 这个系数，所以有两点好处：第一点是当t接近为0时，$k_t$ 对于cubic scheduler会接近0，从而$u$也接近0，我们就不能对模型$t=0$附近的初始预测进行分析了，换为$u'$就不会有这个问题；第二点是t接近1时，$k_t$会发散为无穷，让网络的预测输出量级有很大的变化，而使用$u'$就不会有这个问题。

2. 让模型输出scheduler无关的$\kappa_t$，而不是$t$。

3. 生成采样时可以使用与训练时不同的scheduler，因为已经证明了训练过程本质是scheduler无关的。

于是尝试让模型预测$u'$ 而不是 $u$的实验，结果为`45.057%`。换为linear scheduler进行采样，结果为`46.814%`，这稍有符合oracle相关的猜测，但是提升并不明显。

### 让模型知道哪些token是来自于$x_0$的尝试

可以注意到，无论是训练时还是采样时，我们都可以明确知道哪些token是完全来自于 $x_0$ 而不是后续添加与更改的。尝试将这个信息作为模型的输入进行训练，结果为`50.689%`，说明这个信息还是比较有用的。

### 去除dropout

在尝试调整参数的时候，发现dropout变小会显著地提升性能。当去掉dropout的时候，正确率为`54.803%`。

### 增加正则entropy loss

之前在`first-step-analysis`的分支中的实验发现，目前结果在面对一些不确信的数据时，输出速率分布得有些不集中，直观来讲就是不太“干净”。这和ground truth数据是不相符得，它一般只有两三个位置有正的编辑速率。因此，此次尝试（在main分支的提交中）使用熵正则损失旨在让模型输出速率分布更加集中一些：假设输出速率为 $u_i$ 对于 $i = 1 \sim N$，其中 $N$ 为序列长度，则损失为：
$$
\mathcal{L} = \sum_{i=1}^N - p_i\log p_i,\ p_i = {u_i \over \sum_{i=1}^N u_i}
$$

### beam search研究

目前问题：目前edit flow模型是按照概率进行直接采样，这导致目前仍然无法避免之前flow模型的一些遗留问题，例如采样效率低、生成结果没有确信度（即模型最相信的结果是哪一个）、top10准确率增长不高。

因此目前的目标是：能不能参考自回归模型的beam search，将每一步编辑认为是走一步，然后按照某种score对路径进行赋值，然后最终按照score排序来选出top k的路径。

目前的定下的框架如下：

假设目前绝对时间为kappa，序列为seq，模型输入kappa与seq后，得出的速率预测u_e（指的是没有乘k(t)
  的，即直接表示kappa~1中应该进行几次编辑）目前我觉得遗留的问题是：

1. 如何通过(kappa, seq, u_e)计算出$p_e, p_{stop}$。然后我们就可以选出动作e了，并计算出下一步的序列seq'（若不stop）。 其中 $\sum_e p_e + p_{stop} = 1$，然后我们可以将$\sum_i \log p_e^i $ （即概率的乘积）作为最终分数。
2. 如何通过(kappa, seq, u_e, e)来确定下一步时间kappa'，然后就可以继续进行(kappa', seq')的迭代。

目前采用的一个方案如下：

对当前 state：

$$
U = \sum_e u_e \\
p_{\text{stop}} = e^{-U} \\
p_e = (1-e^{-U}) \cdot \frac{u_e}{U} \\
$$
- `STOP` child：`log_prob = log(p_stop)`

- edit child：`log_prob = log(p_e)`

下一时间

对所有 edit child，共用：
$$
kappa' = kappa + (1-kappa)\left(\frac{1}{U} - \frac{e^{-U}}{1-e^{-U}}\right)
$$
再做：

- `clip(kappa', kappa + eps, 1 - eps)`

### 目前正确率演进

```
45.057%
|
| 采样侧scheduler: cubic -> linear
V
46.814%
|
| + 使用是否编辑的输入（use_origin_mask）
V
50.689%
| |
| | dropout 0.3 -> 0.1
| V
| 52.826%
| |
| | dropout 0.1 -> 0.0
| V
| 54.803% 
|
| + entropy正则损失：entropy_alpha = 1.0
V
51.548%
|
| entropy_alpha 1.0 -> 0.1 
V
52.367%
```