﻿@[toc]

## 一、神经网络参数优化器

参考曹健[《人工智能实践：Tensorflow2.0 》](https://blog.csdn.net/weixin_45558569/article/details/110728137?spm=1001.2014.3001.5501)
深度学习优化算法经历了SGD -> SGDM -> NAG ->AdaGrad -> AdaDelta -> Adam -> Nadam
这样的发展历程。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/504d596021a492b07512552e88ce5b0e.png)

- 上图中f一般指loss
- 一阶动量：与梯度相关的函数
- 二阶动量：与梯度平方相关的函数
- <font color='red'>不同的优化器，实质上只是定义了不同的一阶动量和二阶动量公式</font>

### 1.2 SGD（无动量）随机梯度下降

- 最常用的梯度下降方法是随机梯度下降，即随机采集样本来计算梯度。根据统计学知识有：采样数据的平均值为全量数据平均值的无偏估计。
- 实际计算出来的SGD在全量梯度附近以一定概率出现，batch_size越大，概率分布的方差越小，SGD梯度就越确定，相当于在全量梯度上注入了噪声。适当的噪声是有益的。
- batch_size越小，计算越快，w更新越频繁，学习越快。但是太小的话，SGD梯度和真实梯度差异过大，而且震荡厉害，不利于学习
- <font color='red'>SGD梯度有一定随机性，所以可以逃离鞍点、平坦极小值或尖锐极小值区域</font>
- 最初版本是vanilla SGD，没有动量。$m_{t}=g_{t}，V_{t}=1$（p32—sgd.py）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/92d47cbbda07143b0adfa3ec4a2b195a.png)

- 每个bacth_size样本的梯度都不一样，甚至前后差异很大，造成梯度震荡。（比如一个很大的正值接一个很大的负值）
- 神经网络中，输入归一化。但是多层非线性变化后，中间层输入各维度数值差别可能很大。不同方向的敏感度不一样。有的方向更陡，容易震荡。

- vanilla SGD最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优
点

```python
# sgd
w1.assign_sub(learning_rate * grads[0])
b1.assign_sub(learning_rate * grads[1])
```

### 1.3 SGDM——引入动量减少震荡

- 动量法是一种使梯度向量向相关方向加速变化，抑制震荡，最终实现加速收敛的方法。
- 为了抑制SGD的震荡，SGDM认为 <font color='red'>梯度下降过程可以加入惯性。如果前后梯度方向相反，动量对冲，减少震荡。如果前后方向相同，步伐加大，加快学习速度。</font>

- SGDM全称是SGD with Momentum，在SGD基础上引入了一阶动量：
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b208f001ee7531adfe53c54037e1b27e.png)
- 一阶动量是各个时刻梯度方向的指数移动平均值，约等于最近$1/(1-\beta _{1})$个时刻的梯度向量和的平均值。(指数移动平均值大约是过去一段时间的平均值，反映“局部的”参数信息$)。\beta _{1}$的经验值为0.9。所以t 时刻的下降方向主要偏向此前累积的下降方向，并略微偏向当前时刻的下降方向。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f05c10966d5bae89926500569eaacc1e.png)
老师书上写$$W_{t+1}=W_{t}-v_{t}=W_{t}-(\alpha v_{t-1}+\varepsilon g_{t})$$
$\alpha、\varepsilon$是超参数。
- SGDM问题1：时刻t的主要下降方向是由累积动量决定的，自己的梯度方向说了也不算。
- SGD/SGDM 问题2:会被困在一个局部最优点里
- SGDM 问题3：如果梯度连续多次迭代都是一个方向，剃度一直增大，最后造成梯度爆炸

```python
# sgd-momentun
beta = 0.9
m_w = beta * m_w + (1 - beta) * grads[0]
m_b = beta * m_b + (1 - beta) * grads[1]
w1.assign_sub(learning_rate * m_w)
b1.assign_sub(learning_rate * m_b)
```

### 1.4 SGD with Nesterov Acceleration

- SGDM：主要看当前梯度方向。计算当前loss对w梯度，再根据历史梯度计算一阶动量$m_{t}$
- NAG：主要看动量累积之后的梯度方向。 <font color='red'>计算参数在累积动量之后的更新值，并计算此时梯度</font>，再将这个梯度带入计算一阶动量$m_{t}$
- <font color='red'>主要差别在于是否先对w减去累积动量</font>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ecdb9c9a36300fe9f8c7d504eb885311.png)
思路如下图：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/90f223da6c739ad07615052b5f828b17.png)
首先，按照原来的更新方向更新一步（棕色线），然后计算该新位置的梯度方向（红色线），然后
用这个梯度方向修正最终的更新方向（绿色线）。上图中描述了两步的更新示意图，其中蓝色线是标准momentum更新路径。

### 1.5 AdaGrad——累积全部梯度，自适应学习率

- SGD:对所有的参数使用统一的、固定的学习率
- AdaGrad:<font color='red'>自适应学习率。对于频繁更新的参数，不希望被单个样本影响太大，我们给它们很小的学习率；对于偶尔出现的参数，希望能多得到一些信息，我们给它较大的学习率</font>。
- 另一个解释是：初始时刻W离最优点远，学习率需要设置的大一些。随着学习你的进行，离最优点越近，学习率需要不断减小。
- 引入二阶动量——该维度上，所有梯度值的平方和(梯度按位相乘后求和），来度量参数更新频率，用以对学习率进行缩放。（<font color='red'>频繁更新的参数、越到学习后期参数也被更新的越多，二阶动量都越大，学习率越小）</font>
$$V_{t}=\sum_{\tau =1}^{t}g_{\tau }^{2}$$
$$\eta _{t}=lr\cdot m_{t}/\sqrt{V_{t}}=lr\cdot g_{t}/\sqrt{\sum_{\tau =1}^{t}g_{\tau }^{2}}$$
$$W_{t+1}=W_{t}-\eta _{t}$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/235d13b85aeadc0577391077e3f8486e.png)
- 优点：AdaGrad 在稀疏数据场景下表现最好。因为对于频繁出现的参数，学习率衰减得快；对于稀疏的参数，学习率衰减得更慢。
- 缺点：<font color='red'>在实际很多情况下，频繁更新的参数，学习率会很快减至 0 </font>，导致参数不再更新，训练过程提前结束。（二阶动量呈单调递增，累计从训练开始的梯度)

```python
# adagrad
v_w += tf.square(grads[0])
v_b += tf.square(grads[1])
w1.assign_sub(learning_rate * grads[0] / tf.sqrt(v_w))
b1.assign_sub(learning_rate * grads[1] / tf.sqrt(v_b))
```

### 1.6 RMSProp——累积最近时刻梯度

- RMSProp算法的全称叫 Root Mean Square Prop。AdaGrad 的学习率衰减太过激进，考虑改变二阶动量的计算策略：<font color='red'>不累计全部梯度，只关注过去某一窗口内的梯度</font>。
- 指数移动平均值大约是过去一段时间的平均值，反映“局部的”参数信息，因此我们用这个方法来计算二阶累积动量：（分母会再加一个平滑项，防止为0）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/42bba87fd6bece6b3cc960deb6ff54da.png)
对照SGDM的一阶动量公式：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3a348499a3ecfd3bc24a37d63c554841.png)

```python
# RMSProp
beta = 0.9
v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
w1.assign_sub(learning_rate * grads[0] / tf.sqrt(v_w))
b1.assign_sub(learning_rate * grads[1] / tf.sqrt(v_b))
```

### 1.7 Adam

将SGDM一阶动量和RMSProp二阶动量结合起来，再修正偏差，就是Adam。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/77ccd088d3378778b95407d372fe4ada.png)
一阶动量和二阶动量都是按照指数移动平均值进行计算的。初始化 $m_{0}=0,V_{0}=0$，在初期，迭
代得到的$m_{t}、V_{t}$会接近于0。我们可以通过偏差修正来解决这一问题：
$$\widehat{m_{t}}=\frac{m_{t}}{1-\beta _{1}^{t}}$$
$$\widehat{V_{t}}=\frac{V_{t}}{1-\beta _{2}^{t}}$$
$$\eta _{t}=lr\cdot \widehat{m_{t}}/(\sqrt{\widehat{V_{t}}}+\varepsilon )$$
$$W_{t+1}=W_{t}-\eta _{t}$$

- ${1-\beta _{1}^{t}}、1-\beta _{2}^{t}$的取值范围为（0,1）,可以将开始阶段$m_{t}、V_{t}$放大至$\widehat{m_{t}}、\widehat{V_{t}}$。
- 随着迭代次数t的增加，$\beta _{1}^{t}、\beta _{2}^{t}$趋近于0，放大倍数趋近于1，即不再放大$m_{t}、V_{t}$。

```python
# adam
m_w = beta1 * m_w + (1 - beta1) * grads[0]
m_b = beta1 * m_b + (1 - beta1) * grads[1]
v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])
m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))
w1.assign_sub(learning_rate * m_w_correction / tf.sqrt(v_w_correction))
b1.assign_sub(learning_rate * m_b_correction / tf.sqrt(v_b_correction))
```

### 1.8 悬崖、鞍点问题

- 高维空间中，一个点各个方向导数为0，只要有一个方向对应的极值和其它方向不一样，该点就是鞍点。鞍点是一个不稳定的点，梯度轻微扰动就可以逃离。
- SGD梯度因为具有一定的随机性，反而可以逃离鞍点
- 例如n维空间中，某点各方向导数都为0。假设其中极大或者极小的概率为0.5，则该点为极大或极小值概率均为$0.5^{n}$，反之为鞍点的概率为$1-2*0.5^{n}=1-0.5^{n-1}$。高维空间中鞍点概率几乎为1。因此不必非要找到极小值，只需要loss降到比较低就行了。
- 梯度裁剪：悬崖部位loss会突然下降，梯度太大W容易走过头，所以可以梯度裁剪，限制梯度最大值。

## 二、过拟合解决方案

### 2.1 正则化

- 逻辑回归中，损失函数为$Loss+\lambda \left | W \right |or Loss+\lambda \left \| W \right \|$。在神经网络中则更加灵活。可以为不同层分别设置L1或者L2正则，且系数$\lambda$也可以不同。即各层正则化项可以完全独立。
- <font color='red'>L1正则可以对神经网络剪枝</font >（很多权重趋近于0），网络运行速度可以大大加快。所以对实时要求高的场景可以用L1正则。
- 神经网络中可以设置前层的正则化项系数$\lambda$更大，后层小一些。因为前层面对真实物理信号，噪声较大。为了过滤噪声，W应当较小，$\lambda$更大。且后层面对输出，正则化太厉害，影响分类效果。

### 2.2 dropout

1. <font color='red'>Dropout：每次训练时直接随机去除部分神经元。可以达到剪枝的目的，生成新的子网络。所以本质也是一种正则化，类比于L1正则。各层可以独立设置Dropout</font>
2. dropout相当于训练大量子网络，预测时使用全部神经元，等同于集成学习，增加泛化能力。
3. 降低模型复杂度和各神经元之间的依赖程度，降低模型对某一特征的过度依赖，从而降低过拟合

- 子网络可以是海量的，大量子网络可能没有训练或者就训练一次，但是由于大部分神经元是一样的，少部分才被去除，所以大部分自网络高度相关。而且训练一个子网络，等于主网络大部分参数也得到了更新
- dropout是随机去除神经元，所以发生在$d^{k}=W^{k}\cdot a^{k-1}$加权求和的时候，而不是$a^{k-1}=f(d^{l-1}+w_{0}^{k-1})$激活之后。
- 训练时dropout概率为p，即只有（1-p)个神经元进行计算。则预测时结果要乘以（1-p）
- <font color='red'>跟正则化系数一样，dropout应该是前层设置的高。前层噪声多，需要过滤。后层主要用来精确拟合，dropout过高，影响预测结果。</font>
- dropout可以取代正则，例如训练100个神经元，dropout=0.2；效果好于训练80个神经元。

### 2.3 Batch Normalization

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5fe470c7c42b59ac2337b433abbd3407.png)
&#8195;&#8195;一般来说，顶层梯度较大，学习更快。而底层接近原始数据，但是因为梯度衰减，造成底层梯度较小，学习慢。底层参数更新后，根据前向传播，顶层输入也跟着变化，等于顶层白学了。造成整个网络不稳定，收敛很慢。学习第一层时，我们可以避免更改最后一层吗？
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5f87e829c754d675489dd1ae4c01bb48.png)
&#8195;&#8195;神经网络学习隐含条件：<font color='red'>输入数据有一定的规律，符合一定的概率分布，模型就是学习输入分布和类别之间的映射关系。</font>但是神经网络中，各层输入分布并不稳定：

1. SGD训练时，随机选取样本，造成分布的差异，即$\mu /\sigma$都不一样
2. 前层权重W的变化造成后层输入的变化，进而导致后层最优的参数W变化。即使后层参数已接近最优，也要重新学习，造成网络不稳定。后层分布累积了前层所有的W，所以不同轮次输入分布差异很大，学习速度会很慢

&#8195;&#8195;即训练样本数据本身是符合一定的概率分布的，但是因为以上两个原因，造成每个batch训练时概率分布差异很大。对于一个batch的数据，第l层第m个神经元来说：

- 计算神经元在一个batch样本的输入$d_{i,m}^{l}$的均值和方差，将数据分布转为标准正态分布
$$\mu _{m}^{l}=\frac{1}{batch-size}\sum_{i\epsilon batch }d_{i,m}^{l}$$
- 通过两个待学习参数$\beta_{1} ,\gamma _{1}$，将标准正态分布调整至$N(\beta_{1} ,\gamma _{1})$。

&#8195;&#8195;所以BN就是就是在每个mini-batch里面，将所有样本在某一个特征维度做标准化（减去均值，除以方差）得到$N(\beta_{1} ,\gamma _{1})$数据分布。均值和方差是在训练时计算每个mini-batch里面的数据；预测时使用全局的均值和方差（之前训练时所有batch的平均均值和方差存起来，在预测时使用）。

BN的作用;

- 使输入数据对应的概率分布保持不变（不是数值不变）。<font color='red'>无论训练集数据如何选择，前层网络参数如何变化，各层接受的输入分布在不同的训练阶段都是一致的，各层最优参数稳定，提高了收敛速度</font>
- 不容易受极端数据影响，所以可以提高学习率，使收敛速度进一步提高。
- 降低对参数初始化的依赖程度
- 对参数进行正则化，提高了模型泛化能力

其它注意点：

- batch_size不宜过低，否则均值方差估计不准
- <font color='red'>预测时没有批量的概念，都是对单个样本进行预测，无法进行BN操作计算$\mu /\sigma$ 。所以要保存训练中计算的$\mu /\sigma$ ，预测时使用它们进行无偏估计</font>（公式中t表示当前迭代次数，T为总迭代次数)
$$\mu =\frac{1}{T}\sum_{t=1}^{T}\mu_{t} $$
$$\sigma  =\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}\sigma _{t}^{2} }$$
- BN的位置在激活函数之前。因为将输入转换为正态分布$N(\beta_{1} ,\gamma _{1})$的前提是输入数据本身符合正态分布。而神经网络中激活函数的值域往往有限，造成激活后的输出不会符合正态分布。这样BN之后也不会符合正态分布。一般在卷积层、全连接层输出后，激活函数之前；或者卷积层、全连接层输入上：$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) )$
- 对全连接层，作用在特征维度上（对所有神经元进行批量归一化）；对卷积层，作用在通道维（每个通道一次批量归一化。每个通道对应一个卷积核，一种特征。）

```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

- 不需要和dropout一起用，浅层网络也可以不用。深层网络才有上下层梯度不一致学习快慢、网络不稳定的问题。

代码实现：
（这里只考虑2维和4维输入。如果是别的维数，也都是对第二维做BatchNorm）

```python
import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #gamma, beta就是待学习的参数γ ,β。 
    #moving_mean, moving_var是全局而非小批量的均值方差。momentum存储全局均值方差用于后面推理
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)#eps防止除0
    else:
        assert len(X.shape) in (2, 4)#2维是全连接，4维是2D卷积。这里只考虑简单情况
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维（列）上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```

```python
X=torch.randn(1,2,3,3)
mean = X.mean(dim=(0, 2, 3), keepdim=True)
X,mean

(tensor([[[[ 0.3566, -0.3354, -0.4800],
           [ 1.8908,  0.4621, -0.2268],
           [-0.3083, -0.7575, -0.7016]],
 
          [[-0.1981, -0.1360, -1.6303],
           [-0.8117, -0.9460, -0.5748],
           [ 1.3609,  0.7297, -0.2661]]]]),
 tensor([[[[-0.0111]],
 
          [[-0.2747]]]]))
```

```python
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

### 2.4 Layer Normalization

LN是对每个样本进行归一化，对于l层第i个样本有：（l层共有m=1、2、......$M^{l}$个神经元）
$$\mu _{i}^{l}=\frac{1}{M^{l}}\sum_{m=1 }^{M^{l}}d_{i,m}^{l}$$
计算出$\mu /\sigma$后，和BN一样将数据归一化为$N(\beta_{1} ,\gamma _{1})$分布。

### 2.5 BN和LN的对比

- Batch Normalization：在特征d/通道维度做归一化，即<font color='red'>归一化不同样本的同一特征</font>。缺点是
 	- 计算变长序列时，变长序列后面会pad 0，这些pad部分是没有意义的，这样进行特征维度做归一化缺少实际意义。
 	- 序列长度变化大时，计算出来的均值和方差抖动很大。
 	- 预测时使用训练时记录下来的全局均值和方差。如果预测时新样本特别长，超过训练时的长度，那么超过部分是没有记录的均值和方差的，预测会出现问题。
- Layer Normalization：在样本b维度进行归一化，即<font color='red'>归一化一个样本所有特征</font>。
 	- NLP任务中一个序列的所有token都是同一语义空间，进行LN归一化有实际意义
 	- 因为实是在每个样本内做的，序列变长时相比BN，计算的数值更稳定。
 	- 不需要存一个全局的均值和方差，预测样本长度不影响最终结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/89817df58917eeadcbd4388e43d78216.png)

- BN和LN都可以用的时候BN一般更好，因为不同数据，同一特征得到的归一化结果更不容易造成信息损失。LN会造成神经元耦合。
- batch_size过小的场合，或者RNN、LSTM、Attention等变长神经网络一般使用LN。

综合之前的讲解：
<font color='red'>model.train()的作用是启用 Batch Normalization 和 Dropout。model.eval()的作用是不启用 Batch Normalization 和 Dropout。</font>
