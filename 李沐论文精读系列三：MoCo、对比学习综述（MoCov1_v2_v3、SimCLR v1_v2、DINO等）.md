﻿@[toc]
传送门：
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT](https://blog.csdn.net/qq_56591814/article/details/127313216?spm=1001.2014.3001.5501)
- [李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
- [李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5502)

## 一、MoCo
### 1.1 导言
>参考：
>- 论文：[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722v3)（用动量对比的方法去做无监督的表征学习）
>- 李沐论文精度系列之[《MoCo 论文逐段精读》](https://www.bilibili.com/video/BV1C3411s7t9/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[精读笔记](https://www.bilibili.com/read/cv14623549?spm_id_from=333.999.0.0)

#### 1.1.1 前言
&#8195;&#8195;`MoCo`于2019年11月13在 CVPR发表，并获得 CVPR2020最佳论文提名，它是用一种**对比学习的方式进行无监督训练**的模型。`MoCo`是第一个在很多主流的机器视觉领域上（比如分类、检测、分割、人体关键点检测等），都超越了有监督预训练模型的无监督模型，从某种程度上证明了无监督学习在机器视觉领域，也能达到很好的效果。

`MoCo`这个词，来自于论文标题的前两个单词`动量对比Momentum Contrast`。
1. 首先介绍一下什么是动量。
动量从数学上可以理解成一种加权移动平均：
$$y_{t}=m\cdot y_{t-1}+(1-m)\cdot x_{t}$$
- 上式中，$y_{t}$和$y_{t-1}$是当前时刻和前一时刻的输出，$x_{t}$是当前时刻的输入，m就是动量。
- 这个式子表示：当前时刻的输出，不仅依赖于当前时刻的输入，还依赖于前一时刻的输出。m越大，当前时刻的输入$x_{t}$对结果$y_{t}$就越小。
- MoCo利用了动量的这种特性，从而缓慢地更新一个编码器，让中间学习的字典中的特征尽可能地保持一致。（下文会详细讲到）

2. 然后介绍一下什么是对比学习。
- 原理 ：对比学习是无监督学习的一种，着重于学习同类实例之间的共同特征，区分非同类实例之间的不同之处。
>&#8195;&#8195;举个例子，从imagenet中抽出猫、猫、狗、飞机四张图，那么猫和猫的图片肯定是相似的，和狗不相似。但是和飞机比起来，猫和狗是相似的。所以**对比学习就是对比着差异去学习，模型并不需要真的知道图片中代表的是什么，而只需要知道哪些图片是类似的，哪些图片是不一样的就可以了**。
- 训练目的：对比学习，希望相似数据（图片）最终学到的特征是相似的，在特征空间（`embedding space` ）中，特征向量尽量靠近；反之还希望不同的数据学到的特征向量，尽量远离。
- `pretext task`（代理任务）：对比学习是不需要标签的（比如不需要知道图片是哪一类），但模型还是需要知道哪些图片是类似的，哪些是不相似的，才能训练。这就需要通过通过设计一些巧妙的代理任务，人为指定一些任务来实现。
- 应用最广的代理任务：[instance discrimination](https://paperswithcode.com/paper/unsupervised-feature-learning-via-non-1) 。
	- 简单说就是，从一堆图片中调出任意一张图片$x_i$，将其做一次转换（`transformation` ，比如随机裁剪等数据增广），得到新的图片$x_{i1}$、$x_{i2}$。那么**样本$x_{i1}$叫做基准点（锚点），$x_{i2}$被认为是正样本**（两者都是从$x_i$变化得到的，虽然看起来有差异，但语义信息不应该发生变化），数据集中其它所有图片都是负样本。
	- 有了正负样本的划分，就可以将数据都输入编码器进行编码提取特征了。因为所有的正负样本都是基于锚点来说的，所以$x_{i1}$会单独使用一个编码器$E_{11}$，$x_{i2}$和其它所有负样本使用另外的编码器（可以是同一个编码器，也可以也可以使用不同的编码器。但是不同的编码器之间必须相似，这样编码的特征才有一致性，才有比较的意义）。
	- 对比学习就是要让正样本的编码特征和锚点的编码特征尽可能靠近（相似），让负样本的特征和锚点特征尽量远离。
	- `instance discrimination`直译过来就是个体判别，在这个任务中，只有经过这张图片转换的样本才是正样本，其它图片都是负样本，所以每张图都自成一类。对于ImageNet来说，就不是1000类，而是128万个类别。
- 目标函数：确定了代理任务，知道如何定义正负样本之后，就需要用一个目标函数，来告诉模型该如何学习，比如常见的对比学习目标函数`NCE loss`等。
- 特性：对比学习最大的特性，是这种方法非常的灵活，可以设置各种不同的代理任务。只要找到一种方式去定义正负样本，剩下的都是一些比较标准化的流程。

#### 1.1.2 摘要
&#8195;&#8195;我们在机器视觉领域提出了一种新的无监督学习方法——`MoCo`。`MoCo`虽然是基于对比学习的，但是本文是从另外一个角度来看对比学习，即<font color='deeppink'>把对比学习看作是一个字典查询任务</font>。
>&#8195;&#8195;比如将上面提到的$x_i$当做是`query`，其它包括$x_{i1}$、$x_{i2}$这些图片都是字典中的`key`。我们每次判断正负样本，就是看字典中的这些key和query是否相似，而这些`key`都是通过`encoder`来更新的。

&#8195;&#8195;具体来说，我们构建了一个动态的字典，这个字典有两个特性：队列特性和`moving-averaged encoder`（这两点在下文模型结构中会具体说明，现在记住就行）。因为这样，我们的字典非常大，且特征一致性非常好，从而便于进行对比学习。

&#8195;&#8195;最终，`MoCo`作为一个无监督的预训练模型，能够在7个下游任务（分割、检测等）上 ，超越之前的有监督的预训练模型 ，填平了CV领域中，无监督训练和有监督训练之间的坑。

### 1.1.3 导言
>&#8195;&#8195;GPT和BERT已经证明了无监督的表征学习在NLP领域是非常成功的，但是在视觉领域，无监督学习效果差很多，作者认为可能是二者的原始信号空间不同。
>- 在NLP任务中，原始信号空间是离散的（都是一些含有不同语义的单词或者词根），信号本来就拉得比较开，容易建立tokenize（将单词映射成向量）的字典。这样无监督学习容易建模，且模型容易优化。
>- CV中，视觉信号都是在一个连续且高维的空间里，不想单词那样信息和语义浓缩的那么好，不够简洁，这样就不容易建立一个这样的字典，也就不容易进行无监督学习。

&#8195;&#8195;最近有一些无监督学习方法表现不错，但是都可以归结为**建立动态字典**。
&#8195;&#8195;如果将上一节讲到的所有样本都构建到一个字典中，字典的key就是各个样本，字典的value就是编码之后的特征（后面直接以$k_0$表示第一个样本的编码特征）。我们先编码好锚点的特征，当做query；其它所有样本特征当做字典中不同的key，那么那对比学习就转化成为了一个字典查询的问题了。
如下图所示，我们训练一些编码器，再根据q去字典中查找key。查找的目的，就是让已经编码好的特征q，和与它匹配的特征key（其实就是正样本$x_{i2}$的特征）最相似；与其它不匹配的特征不相似。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d84e9fcf36e3d852be13792afa01eda.png)
>&#8195;&#8195;在MoCo这篇论文当中，因为作者已经把所有的对比学习的方法归纳成为了一个动态字典的问题，所以很少使用anchor或者正负样本这些词，用的都是query和key。所以<font color='deeppink'>锚点$x_{i1}$用$x^{query}$表示，其编码特征用q表示。其它样本和对应特征分别用$x_{i}^{key}$和$k_i$表示。</font>

作者认为，一个好的字典应该有两个特性：
- 字典足够大
	- 字典越大，`key`越多，所能表示的视觉信息、视觉特征就越丰富 ，这样拿`query`去做对比学习的时候，才越能学到图片的特征。
	- 反之，如果字典很小，模型很容易通过学习一些捷径来区分正负样本，这样在碰到大量的真实数据时，泛化就会特别差（我的理解是，字典中只有猫和狗，狗都是黑色，猫都是黄色。模型简单的判断图片中物体是否是黄色，来区分猫和狗，而不是真的学到了猫和狗的特征）
- 编码的特征尽量保持一致性
字典里的`key`都应该用相同或者说相似的编码器去编码得到，否则模型在查找`query`时，可以简单的通过找到和它使用相同或者相似编码器的`key`，而不是真的和它含有相同语义信息的key（变相引入两一个捷径）。

&#8195;&#8195;以前的对比学习，都至少被上述所说的两个方面中的一个所限制（要么一致性不好，要么字典不够大）。本文最大的贡献，就是使用队列以及动量编码器来进行对比学习，解决了这个问题。具体来说：
- `key`（编码特征）并不需要梯度更新，而是通过更新编码器，新的编码器使输出的`key`更新。
- `queue` ：整个队列里面的元素都是字典，队首输入当前batch的编码特征，队尾弹出最旧的batch特征。每次移除的是最老的那些key，从一致性的角度来说 ，有利于对比学习。
	- 用队列的好处是可以重复使用那些已经编码好的key，而这些key是从之前的那些mini-batch中得到的。
	- 用队列结构，就可以把的mini_batch的大小和队列的大小直接分开了，所以最后这个队列的大小，也就是字典的大小可以设的非常大，因为它大部分的元素都不是每个iteration都需要更新的。
	- 在字典里计算loss而不是整个数据集上计算loss，使用队列的数据结构，可以让维护这个字典的计算开销非常小。
- `momentum encoder`：
	- 如果只有当前batch的key是从当前的编码器得到特征，其它的key都是另外时刻的编码器输出的特征，这样就无法保证字典中key的一致性。所以作者又提出了动量编码器 
	- 动量编码器，即编码器参数的更新方式就是$y_{t}=m\cdot y_{t-1}+(1-m)\cdot x_{t}$（`MoCo`中`m=0.999`）。
	- 初始化的编码器来自于`query`的编码器，之后每次更新，只有`1‰`的参数会从`query`的编码器参数里拿过来更新，所以这个编码器参数更新的非常缓慢。从而保证了字典中所有的key都是由相似的编码器抽取得到的，尽最大可能地保持了他们的一致性。（直接更新编码器k的所有参数，会导致编码器更新过快，降低了这个队列中所有key的特征的一致性）
- 动态字典：字典中的key都是随机取样的，而且key的编码器在训练的过程中也是在不停的改变。

### 1.2 相关工作
#### 1.2.1 SimCLR：端到端的学习方式（Inva Spread也是）
端到端学习，顾名思义就是编码器都是可以通过梯度回传来更新模型参数的，优缺点都很明显：
- 缺点：字典大小和mini_batch大小一致，但是现在一般是存不了太大的batch的，而且太大的batch难以优化，处理不好的话，不容易收敛，所以最终模型效果没那么好。
- 优点：因为进行梯度回传，所以编码器可以实时更新，字典中的key的特征一致性非常高
- SimCLR最终使用batch_size=8192来做训练（google有TPU，内存大，可以无脑上batch-size），可以支持模型做对比学习
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/808ceafceb5d2b2d42a64c0c87d39214.png)
#### 1.2.2 memory bank （InstDisc模型）
- 在`memory bank`中，q的编码器是梯度更新的，但是字典中的k，是没有单独的编码器。
-  `memory bank`把整个数据集的特征都存到了一起。每次训练时，只需要从`memory bank`中采样一些key来作为字典（比如$k_1$、$k_2$、$k_3$），然后正常计算q和k的loss，进行梯度回传更新编码器。
- 编码器更新后，重新编码$k_1$、$k_2$、$k_3$得到新的值，替换原来对应的值，这样就完成了一次`memory bank`的更新，依此类推。
>&#8195;&#8195;ImageNet虽然有128万张图片，即128w的key，但是特征维度为`dim=128`，用`memory bank`存下来只需要600M，所以这样做是没问题的。但是对于一个拥有亿级图片规模的数据，存储所有的特征就需要几十G甚至上百G的内存了，所以memory bank的扩展性不如MoCo好。
>

但是这样做有一个明显的问题，就是特征的一致性非常差。表现在：
- 编码器q是梯度回传更新的，所以更新的很快，这样key都是在不同时刻编码器编码的特征，所以特征一致性很差
- `memory bank`存储了所有的图片，也就意味着模型训练了整整一个epoch才能把整个memory bank更新一遍，那也就意味着，当开始下一个epoch训练的时候，假如选了三个key，那这三个key的特征都是上一个epoch不知道哪个时间点算出来的特征了，这也就导致query的特征和key的特征差的特别远。
- `memory bank` 的作者也意识到了这一点，所以使用另外一个loss（proximal optimization），目的就是为了让训练变得更加平滑，而且也提到了动量更新，只不过它的动量更新的是特征。

&#8195;&#8195;由此，作者才提出了`MoCo`，采用队列的形式去实现字典，使其不必受限于字典的大小；使用动量编码器进行缓慢更新，使特征保持一致性。

### 1.3 算法
#### 1.3.1 损失函数
在本文中，采取了一个叫做`InfoNCE`的对比学习函数来训练整个模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5c6bf1c4c88dc6cc7489c02267454bbc.png)
&#8195;&#8195;式子中，`τ`是一个超参数。如果去掉`τ`，整个式子其实就是交叉熵损失函数（`cross entropy loss` ），在后面的伪代码中，也是基于cross entropy loss实现。
- 分子表示q和正样本做计算，分母其实是k个负样本上做累加和，因为是从0到k，所以是k+1个样本，也就指的是字典里所有的key。
- 直接计算复杂度太大： `MoCo`使用 `instance discrimination`作为代理任务，那么光是ImageNet数据集，就有128万个类别，直接计算，复杂度会非常高，难以训练（128万类的softmax）。
- `NCE loss`（`noise contrastive estimation` ）：将超级多分类转为二分类——数据类别data sample和噪声类别noisy sample。这样解决了类别多的问题。
>&#8195;&#8195;`estimation`：近似的意思。为了降低计算复杂度，不是在每次迭代时遍历整个数据集128万张负样本，而是只从数据集中选一些负样本来计算loss（也就是选队列字典中的6万多个负样本），相当于一种近似。所以这也是`MoCo`一直强调的希望字典足够大，因为越大的字典，越能够提供更好的近似。

`InfoNCE`:NCE的一个简单的变体.
- 作者认为如果只把问题看作是一个二分类（只有数据样本和噪声样本）的话，可能对模型学习不是很友好，毕竟在那么多的噪声样本中，大家很有可能不是一个类，所以还是把它看成一个多分类的问题比较合理。
- 公式中的q * k，其实就相当于是logit，也可以类比为softmax中的z。
- `τ`：一个超参数，用来控制分布的形状 。τ越大，分布中的数值越小，经过exp之后就更小了，分布就会变得更平滑，相当于对比损失对所有的负样本都一视同仁，导致学习的模型没有轻重 
- τ越小，分布更集中，模型只关注那些特别困难的样本，其实那些负样本很有可能是潜在的正样本，如果模型过度地关注这些特别困难的负样本，会导致模型很难收敛，或者学好的特征不好去泛化。

#### 1.3.2 伪代码
&#8195;&#8195;对于整个模型来说，在代理任务不一样的时候，输入$x^q$和$x^k$既可以是图片，也可以是图片块（CPC），或者是含有上下文的一系列的图片块。
&#8195;&#8195;`query`的编码器和`key`的编码器既可以是相同的（模型的架构一样，参数完全共享，比如Inva Spread），或者说它们的参数是部分共享的，也可以是彻底不一样的两个网络（CMC，多视角多编码器）。
>上面提到的`CPC、CMC、Inva Spread、SimCLR、InstDisc`在后面对比学习综述中都会简单介绍。
>
下面是论文中作者给出的伪代码，其中：
- fq、fk分别是query和key的编码器
- queue这个队列指的是字典，里面一共有k个key，所以它的维度是`c*k`，c指的是每个特征的维度（`c=128`）
- `m`是动量，`t`是`InfoNCE`里面的超参数`τ`
- aug表示数据增强

1. 初始化编码器`fq`，并将其参数赋值给编码器`f_k`
2. 从data loader里拿一个batch的数据（n=bacth_size=256，n是采样数）
3. 通过数据增强得到正样本对`x_q`和`x_k`，然后通过各自的编码器得到特征`q`和特征`k`（大小都是`N*C`）。key不需要梯度回传，所以用.detach() 去掉梯度信息。
4. 计算N张图片的自己与自己的增强图的特征的匹配度
`q 、k`之间计算`logit`（正样本），也就是之前公式1中算InfoNCE loss的时候的分子$q * k+$，其特征维度就变成了`n * 1`（`256，1`）。
5. 计算N张图片与队列中的K张图的特征的匹配度
 `q、queue`拿出来计算，得到InfoNCE的分母，也就得到了负样本的logit，维度是`n*k`（`256*65536`，MoCo中，字典大小为65536）
6. 将正负样本logit进行`cat`拼接
7. 通过交叉熵损失函数实现loss计算。具体的，设置一个全0向量作为`ground truth`来进行计算。
因为按照作者的这种实现方式，所有的正样本永远都是在logit的第一个位置上，也就是位置0，所以对于正样本来说，如果找对了那个key，在分类任务中得到的正确的类别就是类别0，所以巧妙地使用了这种方式创建了一个ground truth，从而计算出了对比学习的loss
8. 根据loss进行梯度回传，更新编码器`fq`
9. 动量更新编码器`f_k`
10. 更新队列（队首压入新的batch编码的key，队尾弹出最旧的key）
```python
f_k.params = f_q.params # 初始化
for x in loader: # 输入一个图像序列x，包含N张图，没有标签
    x_q = aug(x) # 用于查询的图（数据增强得到）
    x_k = aug(x) # 模板图（数据增强得到），自监督就体现在这里，只有图x和x的数据增强才被归为一类
    q = f_q.forward(x_q) # 提取查询特征，输出NxC
    k = f_k.forward(x_k) # 提取模板特征，输出NxC
    # 不使用梯度更新f_k的参数，这是因为文章假设用于提取模板的表示应该是稳定的，不应立即更新
    k = k.detach() 
    # 这里bmm是分批矩阵乘法
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1)) # 输出Nx1，也就是自己与自己的增强图的特征的匹配度
    l_neg = mm(q.view(N,C), queue.view(C,K)) # 输出Nxk，自己与上一批次所有图的匹配度（全不匹配）
    logits = cat([l_pos, l_neg], dim=1) # 输出Nx(1+k)
    labels = zeros(N)
    # NCE损失函数，就是为了保证自己与自己衍生的匹配度输出越大越好，否则越小越好
    loss = CrossEntropyLoss(logits/t, labels) 
    loss.backward()
    update(f_q.params) # f_q使用梯度立即更新
    # 由于假设模板特征的表示方法是稳定的，因此它更新得更慢，这里使用动量法更新，相当于做了个滤波。
    f_k.params = m*f_k.params+(1-m)*f_q.params 
    enqueue(queue, k) # 为了生成反例，所以引入了队列
    dequeue(queue)
```
### 1.4 实验
#### 1.4.1 对比其他模型
下图是端到端学习、memory bank和MoCo三种流派的模型，在只做特征提取时候的精度对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8f8b140a4f2a015c66d930232fd275b7.png)
- 横坐标用k表示，指的是用了多少个负样本，也可以粗略地理解为字典的大小
- 纵坐标指的是在ImageNet数据集上的top 1的准确率
- 端到端学习，受限于显卡内存，实验结果只有三个点（字典最大1024）
- `MoCo`性能最好，对硬件要求最低，而且扩展性也比较好
#### 1.4.2  imagenet数据集结果对比
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b3e5a518463dbd6c1218f597d3c434e5.png)
- 表格中上半部分都不是使用的对比学习，下半部分都是使用的对比学习，可以看到对比学习效果明显更好
- 精度结果后面的特殊标记表示用fast auto augment做了数据增强（ImageNet有监督训练的数据增强策略）

#### 1.4.3 迁移学习效果
**归一化**
&#8195;&#8195;预训练好的`MoCo`做微调，其学习率需要设为30，远大于以前模型的微调时的一些学习率（比如lr=0.03）说明MoCo学到的特征跟有监督学到的特征的分布是非常不一样的，但是不能每次微调时都去grid search找一下它最佳的学习率是多少，这样失去了微调的意义。
&#8195;&#8195;当分布不一致的时候，最常想到的方法就是归一化，所以作者这里使用了特征归一化的方法（整个模型都做BN，包括检测时用到的FPN结构，也使用BN）。做完归一化之后，就可以拿这些有监督训练用的超参数来做微调了。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b94aa8384c4abbc39b5bf96bdfd88758.png)
在keypoint detection人体关键点检测、pose estimation姿态检测、实例分割、语义分割四个任务中做测试：
- 第一行使用的是随机初始化的模型再做微调，所以它是一个基线网络，分数比较低
- 第二行使用的是有监督的ImageNet的预训练的模型做初始化然后再做微调，也就是一个比较强的极限结果
- 最后两行分别是MoCo在ImageNet上和在Instagram 1Billion上做无监督预训练当作模型的初始化，然后再做微调 

&#8195;&#8195;结论：`MoCo`预训练的模型在大部分时候都比ImageNet的有监督预训练模型要好，在实例分割和语义分割的任务上有时候会稍差一些。
>&#8195;&#8195;所以大家怀疑对比学习可能不太适合做这种每个像素的都要预测的任务，基于这一点，后续发展出dence contrast或者是pixel contrast。

### 1.5 总结
&#8195;&#8195;`MoCo` 的主要贡献就是把之前对比学习的一些方法都归纳总结成了一个字典查询的问题，并提出了队列存储和动量编码器。前者解决字典太大不好存储和训练的问题，后者解决了字典特征 不一致的问题；从而形成一个又大又一致的字典，能帮助模型更好的进行对比学习。

&#8195;&#8195;`MoCo`跟`Inst Disc`是非常相似的，比如它用队列取代了原来的memory bank作为一个额外的数据结构去存储负样本，用动量编码器去取代了原来loss里的约束项，这样就可以动量的更新编码器，而不是动量的去更新特征，从而能得到更好的结果。其整体的出发点以及一些实现的细节（比如backbone和lr、batch_size，dim、τ等等超参数都是一样的）和`Inst Disc`都是非常类似的，所以可以说`MoCo`是`Inst Disc`的改进工作。但是MoCo真正出色的地方其实有两点 ：
- 使用动量编码器。这个改进简单有效，并在后面一系列工作中被一直沿用（比如SimCLR、BYOL），所以也非常深刻。
- 写作高人一等。直接把之前所有的方法都总结成了一个字典查找的问题，所以直接把问题给归纳升华了。而且提出CV和NLP的对比学习大一统框架，论文的泛用性彻底扩大了。

&#8195;&#8195;`MoCo`还有一个优点，就是训练比较便宜。在一张`8卡V100 16G GPUs`上，训练200个epoch只需要`53`小时（batch_size=256，GPU memory=5.3G），完全就是大佬给我们送福利。`MoCo`这篇论文以及它高效的实现，能让大多数人有机会用普通的GPU就能跑对比学习实验，做自己的研究。

&#8195;&#8195;最后，因为MoCo在各个视觉任务上取得了更好的性能，也激发了很多后续分析性的工作，去研究MoCo学出来的特征到底和有监督学出来的特征有什么不同，还能从别的什么方向去提高对比学习。

## 二、对比学习论文综述
>参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14700928?spm_id_from=333.999.0.0)

&#8195;&#8195;如果把 近几年对比学习在视觉领域有代表性的工作做一下总结，那么对比学习的发展历程大概可以分为四个阶段：
1. 百花齐放
这个阶段代表性工作有InstDisc（instance discrimination，）、CPC、CMC等。在这个阶段中，方法、模型、目标函数、代理任务都还没有统一，所以说是一个百花齐放的时代
2. CV双雄
代表作有MoCo v1、SimCLR v1、MoCo v2、SimCLR v2；CPC、CMC的延伸工作、SwAV等。这个阶段发展非常迅速，有的工作间隔甚至不到一个月，ImageNet上的成绩基本上每个月都在被刷新。
3. 不用负样本
BYOL及其改进工作、SimSiam（CNN在对比学习中的总结性工作）
4. transformer
MoCo v3、DINO。这个阶段，无论是对比学习还是最新的掩码学习，都是用Vision Transformer做的。

下面就简单介绍一下这14篇工作，重点是其研究动机。

## 三、 第一阶段：百花齐放（2018-2019Mid）
###   3.1 InstDisc（instance discrimination） 
>- [《Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination》](https://paperswithcode.com/paper/unsupervised-feature-learning-via-non)
>- 参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14700928?spm_id_from=333.999.0.0)
>- 参考：[《对比学习一 |Instance Discrimination》](https://zhuanlan.zhihu.com/p/457986773)、[《Instance Discrimination论文阅读笔记》](https://blog.csdn.net/Nin7a/article/details/103020861?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166720909816782427479081%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166720909816782427479081&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-4-103020861-null-null.142%5Ev62%5Econtrol,201%5Ev3%5Econtrol_1,213%5Ev1%5Econtrol&utm_term=instance%20discrimination&spm=1018.2226.3001.4187)、

&#8195;&#8195;这篇文章提出了个体判别任务（代理任务）以及`memory bank` ，非常经典，后人给它的方法起名为InstDisc。

#### 3.1.1 研究动机

&#8195;&#8195;在有监督学习的分类模型中，如果给一张豹子图片进行分类，会发现排前几名的都是跟这张图很像的图片，而排名靠后的那些往往是跟豹子一点关系都没有的类别。
&#8195;&#8195;作者研究发现，让这些图片聚集在一起的原因并不是因为它们有相似的语义标签，而是因为这些照片里的物体都很相似。最后作者由此提出了个体判别任务：把每一个instance（实例，这里就是指每一张图）都看成是一个类别，目标是学一种特征，把每张图片都区分开来。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9f31d8cc2446edd28c1ab54f91b62901.png)
#### 3.1.2算法
**1. 模型结构**
&#8195;&#8195;将图片经过CNN网络编码后得到的图片特征，使用对比学习的方式将其在特征空间中尽可能的区分开来（因为每张图都是自己的类）。
&#8195;&#8195;既然是对比学习，就需要正负样本。InstDisc中正样本就是就是这个图片本身（可能经过一些数据增强），负样本就是数据集里所有其它的图片，这些负样本都存储在 memory bank里。对于ImageNet有128万张图片，那么memory bank就要存储128万行，所以最后每张图都用128维特征表示（维度太高存储不了）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5aea73f72d154e16dcad50cad556fbf9.png)
**2. 前向过程**
-  $image\overset{ResNet50}{\rightarrow}2048Dim\rightarrow 128Dim$，即经过ResNet50编码得到128维的图片特征
- 论文的softmax不设置参数w。而是和Word2vec一样把特征当作参数，并创建一个叫做memory bank的堆进行存储所有单词的128维特征，每次通过loss更新。这样训练和测试通过存储的memory bank同使用一个度量空间。
- 论文取batch_size=256，则每个batch有256个正样本，然后从 memory bank 里随机地抽取4096个负样本。根据正负样本计算对比学习目标函数`NCELoss`。然后根据loss更新backbone和memory bank（把 mini batch里的数据样本所对应的那些特征，在 memory bank 里更换掉，这样无论是训练还是测试就都来自于一个度量空间了）。
- 测试时，使用KNN进行分类
我们获得了训练好的模型后，对于一张图片提取他的特征，将他和memorybank中所有的存储图片特征计算相似度，然后采用k近邻算法，返回最相似的k张图片。最后根据相似度权重投票，得到其类别c。

**3. 训练细节**

&#8195;&#8195;本文的一些超参数设定，比如backbone选择ResNet50，batch_size=256，负样本采样数为4096，特征维度dim=128，epoch=200，初始lr=0.03，计算NCELoss时τ=0.07；这些超参数在在MoCo 中也是沿用的，没有进行更改。
#### 3.1.3  NCELoss损失函数
- Parametric Classifier参数分类器
在传统的参数softmax函数中，对图片x及特征 $v=f_\theta (x)$，被识别为第i类样例的概率为：

$$P(i|v)=\frac {exp(w_i^{T}v)} {\sum_{j=1}^n exp(w_j^Tv)}$$
其中 v是卷积网络输出的特征表示,i 是预测类别(实例级)，w是需要优化的权重向量。
- Non-Parametric Softmax Classifier
作者认为纯粹的参数w阻碍了个体之间的对比，于是文章采用的无参softmax：使用L2正则化的$v_i^{T}$来替换$w_i^{T}$，$\tau$用来调整类别分布的集中程度：

$$P(i|v)=\frac {exp(v_i^{T}v/\tau)} {\sum_{j=1}^n exp(v_j^Tv/\tau)}$$
使用Mermory Bank V 来存储上述的$v_j$，在每个iteration对应修改其值$f_i\to v_i$,在初始化时通过单位随机向量对V进行初始化。
- Noise-Contrastive Estimation：多分类问题转化为一组二分类问题，其中二分类任务是区分数据样本和噪声样本。
&#8195;&#8195;由上式可知，计算瓶颈在于分母，需要枚举所有图片，这样的计算复杂度是无法接受的。为了解决这一问题，我们不再采用原先的采样方式，而是用随机负采样，即从噪音分布当中进行随机采样，真实样本和噪音分布的数据比为 m。
&#8195;&#8195;如果噪音分布当中采样n个数据，那么真实样本就采样n/m个数据（一般就为1个）。这样原先的多元问题就转化为了二元问题，则Memory bank中特征表示 v 对应于第i 个样例的概率为：

 $$P(i|v)=\frac {exp(v^{T}f_i/\tau)} {Z_i}$$

$$Z_i=\sum_{j=1}^n exp(v^{T}f_i/\tau)$$


我们设定噪声分布为一个均匀分布$P_n=1/n$，则v属于第i个个体的后验概率为：

$$h(i,v)=P(D=1|i,v)=\frac {P(i|v)}{P(i|v)+mP_n(i)}$$

 

训练目标为最小化似然函数 $$J_{NCE}(\theta)=-E_{P_d}[\log h(i,v)]-m \cdot E_{P_n}[\log(1-h(i,v^\prime)]$$
&#8195;&#8195;其中$P_d$指代真实数据分布，对$P_d$而言v 是$x_i$ 的特征；v ′  是来自另一幅图片，从噪声分布$P_n$中随机采样得到,v 和v ′  都是从Memory Bank中采样得到的。
&#8195;&#8195;在正向计算时, 分母项$\sum_{j=1}^{n} \exp \left(\mathbf{v}_{j}^{T} \mathbf{v} / \tau\right)$的计算是无法避免的, 直接计算的计算量同样很大, 于是本文使用蒙特卡罗方法来估计这一项:
$$Z \simeq Z_{i} \simeq n E_{j}\left[\exp \left(\mathbf{v}_{j}^{T} \mathbf{f}_{i} / \tau\right)\right]=\frac{n}{m} \sum_{k=1}^{m} \exp \left(\mathbf{v}_{j k}^{T} \mathbf{f}_{i} / \tau\right).$$


#### 3.1.4 Proximal Regularization

由于每个“类”只有1个样例，在每个epoch中，一个“类”只被访问一次，训练的过程比较不稳定。为了使训练更加平滑，在损失函数上增加一项针对v 的惩罚, 来稳定训练过程:
 $$-\log h\left(i, \mathbf{v}_{i}^{(t-1)}\right)+\lambda\left\|\mathbf{v}_{i}^{(t)}-\mathbf{v}_{i}^{(t-1)}\right\|_{2}^{2}$$
&#8195;&#8195;其中，$v_i^{(t)}=f_\theta(x_i)$（第t次迭代时backbone的输出特征），$V={v_i^{(t-1)}}$来自于memory bank。这样随着多次迭代，由于${v_i^{(t)}}-{v_i^{(t-1)}}$的加入，backbone和memory bank存储的特征就逐渐相同了，回到了原始的损失，加速了收敛。
&#8195;&#8195;所以Proximal Regularization相当于模型的训练加了一个约束，从而能让 memory bank 里的那些特征进行动量式的更新（当前时刻的输出和上一时刻的输入有关），跟 MoCo 的想法是非常一致的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/edc4a65b1b419efe6f3b77c0493a4acb.png)





#### 3.1.5 结论
&#8195;&#8195;`Inst Disc` 这篇论文也是一个里程碑式的工作：它不仅提出了个体判别这个代理任务，而且用这个代理任务和 NCE loss做对比学习，从而取得了不错的无监督表征学习的结果。同时它还提出了用别的数据结构存储这种大量的负样本，以及如何对特征进行动量的更新，所以真的是对后来对比学习的工作起到了至关重要的推进作用。
### 3.2  Inva Spread
>- 论文：[《Unsupervised Embedding Learning via Invariant and Spreading Instance Feature》](https://paperswithcode.com/paper/unsupervised-embedding-learning-via-invariant)
>- 知乎[《对比学习二 | Unsupervised Embedding Learning via Invariant and Spreading Instance Feature》](https://zhuanlan.zhihu.com/p/459345219)

#### 3.2.1 前言
&#8195;&#8195;这篇文章作者同样没有为自己的方法起名字，所以后面一般将其简称为`Inva Spread`。`Inva Spread`是一种端到端的训练方式，直接训练特征本身，无需额外的数据结构（比如上文的memory bank），提升了效率和准确度。作者还使用了新的采样方式，降低了计算复杂度。

&#8195;&#8195;简单来说，本文中的正负样本都来自同一个mini_batch。比如对于图片$x_i$，其正样本就是数据增强后的图片${x_{i}}'$，而负样本就是这个mini_batch中除了$(x_i,{x_{i}}')$之外的所有样本，而不是整个数据集中的所有其它样本。这样负样本数大大减少，可以不需要额外的数据结构来存储，就可以用一个编码器做端到端的训练了。

&#8195;&#8195;`Inva Spread`可以看做是`SimCLR`的前身，但由于数据增强策略不足以及负样本数量太少，也没有`SimCLR`提出的mlp projector ，使得最终的训练效果不好，没有太大的影响力。
>&#8195;&#8195;`Inva Spread`的作者太穷，没有TPU，只能选择`batch_size=256`来训练。这样每次迭代的负样本只有255*2个，数量太少，对比学习的效果不够好（也就是在MOCO中说过的字典太小）。而`SimCLR`的作者来自谷歌，可以使用大量的TPU，最终训练的`batch_size=8192`，足以达到不错的训练效果。

#### 3.2.1 算法
&#8195;&#8195;作者认为提升效率的方法就是直接优化特征本身，拒绝额外的数据结构，也就是用端到端的方式。但这样做会有两种阻碍：一是如果抛弃通过参数w来学习，也不采用memory bank利用时间差更新而让特征自己乘自己，就会使得网络得不到训练。二是不采用NCE等方式，训练的复杂度就太大了。

&#8195;&#8195;作者认为，相似图片通过编码器以后，它的特征应该很类似，不同的图片，它的特征出来就应该不类似，这就是题目中说的invariant和 spreading 。于是作者提出的孪生神经网络结构，有效地解决了这两个问题：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa13efa312d8b417c25b7a0af5861162.png)
- 设batch_size=256，即输入256张图片。经过数据增强，又得到了256张增强后的图片。这样每个batch有256个正样本和（256-1）*2个负样本。
- 根据正负样本计算loss（NCE loss 的一个变体），然后更新网络参数。
- 训练结果表示在最后特征空间中，就是绿色的两个球靠近，和所有别的球远离；其余类似。

### 3.3 CPC
>论文：[《Representation Learning with Contrastive Predictive Coding》](https://paperswithcode.com/paper/representation-learning-with-contrastive)

&#8195;&#8195;之前的几篇代理任务都是个体判别任务，那么自然也有生成式的代理任务，CPC就是其中之一，它使用预测的代理任务去做对比学习。CPC是一个通用结构，其输入是一个序列，可以是图片（不同patch）、文字或者音频、视频等等。本文使用音频为输入，如下图所示：
- 对于一个输入序列x，当前时刻为t。t时刻输入经过编码器$g_{enc}$得到编码特征${z_t}$。
- ${z_t}$经过自回归模型$g_{ar}$（比如RNN/LSTM）得到输出$c_t$（context representation，上下文特征，因为含有之前时刻的信息）。如果$c_t$表示的足够好，包含之前所有时刻的信息，那么应该可以用来预测未来时刻的输出特征$z_{t+i}$。
- 对比学习的正样本就是未来的输入通过编码器以后得到的未来时刻的特征输出，负样本就是任意输入通过这个编码器得到输出。（感觉这里负样本都没说明白，老师一句话带过。别的博文说负样本是其它的输入序列，比如另一段音频的编码输出，但如果这样的话，前面一大段讲$c_t$预测$z_{t+i}$有啥意义。还有很重要的互信息也没讲，这里先放着了）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7f66e3114cd024ae47d11118d826d829.png)

### 3.4 CMC
>论文：[《Contrastive Multiview Coding》](https://paperswithcode.com/method/contrastive-multiview-coding)、[《对比学习四 | Contrastive Multiview Coding》](https://zhuanlan.zhihu.com/p/464393607)

#### 3.4.1 前言
&#8195;&#8195;`CMC`使用一个物体的多个视角来作为正样本。这个思想来自于人类对世界的感受、观察。
&#8195;&#8195;在摘要中，作者说人类观察这个世界是通过很多个不同视角的传感器，比如说眼睛或者耳朵，来给大脑提供不同的信号。每一个视角都是带有噪声的，而且有可能是不完整的。但是最重要的那些信息，比如物理性质，几何形状以及语义信息，在所有的这些视角中间共享。例如一只狗可以被看到、听到、感受到。
&#8195;&#8195;基于此，作者认为一个强大的特征，应该具有视觉不变性（不论是看到还是听到，都应该能判断出那是一只狗）。所以CMC目的，就是最大化同一个场景不同视角的互信息，并且可以扩展到任意数量的未知视角，且视角越多效果越好。

#### 3.4.2 算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e2836d13ea38b56d6a3a6d85c1a6a056.png)
&#8195;&#8195;如上图所示，`CMC`选用 `NYU RGBD` 数据集进行 训练。数据集中每张图有4个视角（view）：原始的图像、原图对应的深度信息（每个物体离观察者到底有多远）、SwAV ace normal以及原图的分割图像。
&#8195;&#8195;在CMC中，一张图的四个视角就是互为正样本，因为其代表的是同一个东西；其它的图片就是负样本。在上图表示，就是特征空间中四个绿色的点互相靠近，而都和红色的点远离。

#### 3.4.3 目标函数
&#8195;&#8195;CPC可以看做是学习过去和未来两个视角，个体判别是学习一张图片的不同crops，但使用的却都是一种目标函数。本文使用的也是普通的NCELoss目标函数，但作者将其进行扩展以适应不同视角的需求，对比学习也扩展到了很多其他领域。

1. 两个视角目标函数（两视角对比着学）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/66daaee42c9e3d363f5744cfd55abef4.png)
其中：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/672ba243642f23f4f124885510adb403.png)
即固定$v_{1}^{1}$  ，列举$v_{2}^{j}$  ，同样的也可以反过来固定$v_{2}^{1}$  于是：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c6ff4ce4499c693250edd13f62f99f66.png)
&#8195;&#8195;这里的$f_{\theta }^{1}$  和$f_{\theta }^{2}$  是两种backbone，不共享参数，这个和Spreading Instance是有区别的。
2. 多个视角目标函数，有两种范式：

- 仅将一个视角和其他所有视角对比：$L_{c}=\sum_{j=2}^{M}L(V_{1},V_{j})$
- 每个视角相互对比：$L_{c}=\sum_{1\leqslant i<j\leq M}L(V_{1},V_{j})$

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/affb0e9d7872641fa8c60f9dc6e6d67c.png)
>&#8195;&#8195;cmc原班作者人马还用对比学习的思想做了一篇蒸馏的工作。
>&#8195;&#8195;对于`teacher` 模型和`student` 模型，不论用什么网络，不论这个网络是好是坏是大是小，只要你的输入是同一张图片，那得到的这个特征就应该尽可能的类似，即二者的输出尽可能的相似。通过这种方式把 `teacher`和`student`做成了一个正样本对，从而可以做对比学习。
#### 3.4.4 结论
&#8195;&#8195;`CMC`正负样本确定的方式由个体升级成了个体的不同的视角（如色彩模型）。它同样使用了NCE，但将其扩展以适应不同的视角。`CMC`采用多视角对比学习，证明了对比学习的灵活性，也同时证明了多视角多模态的可行性，为之后的CLIP工作（图文配对的多模态对比学习）打下了基础。
&#8195;&#8195;但是本文也有一个局限，即处理不同的视角（模态）时，可能需要不同的编码器，因为不同的输入特点不一样。 如果每个视角都有一个编码器，那么训练的成本就有点高（比如在CLIP里，文本编码器是BERT，图片编码器是ResNet或者ViT）。
&#8195;&#8195;所以这也是现在`Transformer`最吸引人的地方，这个结构可以同时处理文本和图片，那么就可以用一个解码器处理两种模态，而不用做针对每种数据去做特有的改进。今年在ICLR上发表的[MA-CLIP](https://paperswithcode.com/paper/ma-clip-towards-modality-agnostic-contrastive)，就是用一个`Transformer`去同时处理两个输入模态，效果反而更好。
>&#8195;&#8195;关于CLIP及其7篇拓展工作（目标检测。实例分割。图片生成），可以参考我另一篇博文[《李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）》](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5501)
### 3.5 小结
- InstDisc：一个编码器+memory bank，特征一致性比较差
- Inva Spread：只使用一个编码器进行端到端训练，但是字典太小，负样本不够
- CPC：一个编码器+一个自回归模型
- CMC：有两个甚至多个编码器

## 四、 第二阶段：CV双雄（MoCo和SimCLR）
>参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14700928?spm_id_from=333.999.0.0)

这段时间（2019Mid-2020Mid）代表作就是MoCo和SimCLR，最后还有个SwAV没用负样本，承上启下。
### 4.1 MoCo
见上文

### 4.2 SimCLR(simple contrastive learning) 
>[《A Simple Framework for Contrastive Learning of Visual Representations》](https://paperswithcode.com/paper/a-simple-framework-for-contrastive-learning)、[代码（TF）](https://github.com/google-research/simclr)
#### 4.2.1 算法
&#8195;&#8195; `SimCLR`是2020年2月13上传到arxiv的，其想法非常简单，在很多博客里介绍对比学习都会将其作为例子，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/91d5009873e2eb47517f8f3368f9d2db.png)
- 图片x经过不同的数据增强得到不同的图片$\tilde{x_{i}}$和$\tilde{x_{j}}$，这两个就是互为正样本；同一个mini_batch里面的其它图片都是负样本，这点和`inva spread`一样。
- 正负样本经过同一个编码器$f(\cdot )$（权重共享）得到编码特征$h_i,h_j$。比如encoder选ResNet50，就是输出2048维特征。
- $h_i,h_j$经过同一个projector即图中的$g(\cdot )$（其实就是一个mlp层，全连接层+relu激活）得到最终的对比学习特征$z_i,z_j$（128维）。
- 对比学习的训练目标就是使正样本特征更相似（同一张图片得到的 $z_i,z_j$），而负样本的特征不相似。
- 选用的损失函数是 `NT-Xent loss`（the normalized temperature-scaled cross entropy loss）。normalized是指在特征后面进行了 L2 归一化，temperature-scaled  就是说在 loss 里加了个τ，所以和`infoNCE loss`也是非常接近的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/12a4497d3ed47249ad582d66eb4dbcd9.png)

- projector在训练时才使用，推理时直接去掉，只用特征h特征。
#### 4.2.2 对比`inva spread`
- `SimCLR`可以被认为是`inva spread`的改进工作。其最大创新点就是在图片编码特征之后加了一个`projector`，但就这么简简单单的一层mlp，能让模型在ImageNet 分类任务上直接涨了近10个点。
- 使用更优的数据增强技术
- 使用更大的batch_size（256→8192）

&#8195;&#8195;`SimCLR`的前两点贡献，添加`projector`和使用的数据增强，在之后的对比学习模型（MoCov2、BYOL）中也一直被沿用。
#### 4.2.3 实验
**1. 模型效果**
&#8195;&#8195;`SimCLR (4×)` 这个模型可以在 ImageNet 上面达到 76.5% 的 Top 1 Accuracy，比当时的 SOTA 模型高了7个点。如果把这个预训练模型用 1%的ImageNet的标签给 Fine-tune 一下，借助这一点点的有监督信息，SimCLR 就可以再达到 85.5% 的 Top 5 Accuracy，也就是再涨10个点。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2df52aae6538e625dbc11e96a81872c4.png)

**2. 数据增强**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4f54a62ed7a606e6ba3b057e74fc6e74.png)
&#8195;&#8195;作者试验了以上10种数据增强，比如随机裁剪、变换色彩、翻转、Cutout、高斯噪声、blur噪声等等；并做了如下的消融试验（除了最后一列，余下是两两组合）。最后发现随机的裁剪和随机色彩变换组合效果最好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/021b792fa78a5983090d502a1375df44.png)
**3. Projection head 及特征维度**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0c992ee0eb8088ce26ef1f029cc2db07.png)
- linear ：只有全连接层，不接Relu激活函数
- None：没有Projection head，直接做对比训练
- non-linear ：本文的方法，加了Projection head。（后面跟Relu，所以是非线性）
- 可以发现使用Projection head，结果提了近10个点
- 最后z的维度不论是32、64还是2048其实都没太大区别，这就是为什么对比学习现在一般都选一个比较低的特征维度，因为128就够了。

### 4.3 MOCOv2
>论文：[《Improved Baselines with Momentum Contrastive Learning》](https://paperswithcode.com/paper/improved-baselines-with-momentum-contrastive)、[代码](https://github.com/facebookresearch/moco)
#### 4.3.1 改进策略
&#8195;&#8195;`MoCov2`主要是借鉴了`SimCLR`而做的优化，比如引入了mlp projection head以及使用更多的数据增强。`MoCov2`刷新了ImageNet 上的最好成绩，比之前的`MoCo`以及最新的`SimCLR`都高很多 。其上传的日期是3月9日，离`SimCLR`的发布还不到一个月。

&#8195;&#8195;`MoCov2`对比`MoCo`主要有4个改动：
- 添加 projection head
- 使用更多的数据增强
- 训练时使用cosine的learning rate schedule 
- 训练的epoch，从200增加到800 
 
####  4.3.2 实验
**1. 改进策略效果对比**
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bafe3e6826ae5ae5528b04b04f4e901d.png)

上图列出了模型效果对比图。
- MLP表示增加projection head，可以看到只增加这一点，就提了近6个点
- aug+和cos分别表示上面提到的数据增强和cosine schedule 
- 灰色行是有监督baseline模型

2. 与SOTA模型分类效果对比
下面是和`MoCov1`以及 `SimCLR` 在ImageNet数据集上分类效果对比。
- 在都只训练200epochs的情况下，`MoCov2`比`SimCLR`高了大概一个点
- 训练800个epochs时，`MoCo v2`能到71.1，比`SimCLR`训练了1,000个epochs还要好将近2个点。这就意味着`MoCov2`能更好的利用数据，能在更短的时间内取得更好的结果 
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1576d6e851402381b343f3665501f902.png)
**3. 训练资源对比**
`MoCov2`相比`SimCLR`，其训练时消耗的内存以及训练时长都更少。下表的end-to-end其实就是指`SimCLR`。
- 在一张`8卡V100 16G GPUs`上，训练200个epoch只需要`53`小时（batch_size=256，GPU memory=5.3G），大佬又给我们送福利啦
-  `†` 这个符号表示是作者的推测，但是因为没有这么大的GPU，所以最终没有训练时长这一项。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6c496116820a56dbe18cb0fc061f57f4.png)
### 4.4 SimCLRv2 
>- 论文： [《Big Self-Supervised Models are Strong Semi-Supervised Learners》](https://paperswithcode.com/paper/big-self-supervised-models-are-strong-semi)
>- 知乎[《自监督黑马SimCLRv2来了》](https://zhuanlan.zhihu.com/p/150358540)、 [《半监督学习之Noisy Student》](https://blog.csdn.net/weixin_42764932/article/details/112980737)
#### 4.4.1简介
&#8195;&#8195;`SimCLRv2`的主要思想体现在其标题里，即大的自监督模型很适合做半监督学习。在摘要中，作者提出：一种从少量带标签数据+大量无标签数据中进行学习的方案是：无监督预训练（必须是大模型）+有监督微调，这种半监督学习的方案在ImageNet上极为有效，具体的可以总结为三步：
1. `pretrain`：在无标签数据上无监督训练（SimCLR对比学习）一个Big ResNet模型（模型大小至关重要）以学习广义视觉特征表达。
2. `fine-tune`：在少量有标签数据上通过进行有监督的微调
3. `distill`：用微调后的模型作为`teacher`模型，在之前的无标签数据集上生成伪标签，然后训练一个`student`模型进行自监督训练（蒸馏阶段采用KL散度）。
>&#8195;&#8195;微调后，作者发现：模型的任务已知预测属性可以进一步改善并蒸馏到一个更小的网络中。为此，作者对无标签数据进行了二次利用以促使学生网络尽可能的模拟老师网络的标签预测性能，且蒸馏阶段采用伪标签方式且不会造成额外的更多复杂度。
>&#8195;&#8195;整个框架其实也是受启发于google的另外一篇工作 [Noisy Student](https://paperswithcode.com/method/noisy-student)。`noisy student`就是在`ImageNet`数据集上先训练了一个 teacher 模型，然后在`JFT 300M`那个数据集上生成了很多的伪标签，最后一起训练了一个student模型，其精度为88，霸榜ImageNet快一年。

&#8195;&#8195;`SimCLRv2`在仅仅采用`1%/10%`有标签数据时，backbone使用ResNet50就取得了`73.9%/77.5%`的top-1精度。
####   4.4.2 算法
模型结构如下图所示，训练过程就是上面提的三步：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/308c9d1bc77357adf2e2105959a6987d.png)

`SimCLRv2`相比`SimCLRv1`有三处改进：
- 大模型：backbone从`ResNet50`替换为`ResNet152+SK net` （selective kernels）
- 加深`protection head` ：从一层加到两层。
protection head在SimCLRv1和MOCOv2中都被证明很有用，所以作者考虑多家几层。最后发现加到两层效果就够了
- 引入了动量编码器：使用了类似`MOCO`的动量编码器，效果提升了一个点。
作者解释是，`SimCLR`模型的 batch_size已经够大了，也就是字典的大小和字典里特征一致性，SimCLR v2 都已经做的很好了。换成`MOCO`这种队列结构的动量编码器，虽然可训练的负样本更多，但是提升没有那么明显了。    

**微调**
- `SimCLRv1`在微调时，是去掉$g(\cdot )$（projector层），只保留编码器$f(\cdot )$进行微调，即$f^{task}(x_{i})=W^{task}f(x_{i})$；
- `SimCLRv2`在微调时，是保留$g(\cdot )$的第一层 ，即$f^{task}(x_{i})=W^{task}\cdot \sigma (W^{MLP}\cdot f(x_{i}))$

#### 4.4.3  实验 
&#8195;&#8195;下表给出了微调的`SimCLRv2`、编码器+线性分类器（`SimCLRv2`不微调）和有监督基线模型，在ImageNet上的Top-1精度。可以看到：提升模型宽度、深度以及添加SK可以取得更好的性能。   
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/faa2f0f811cae5ce364d3783aee50d1c.png)
### 4.5 SwAV 
>- 论文[《Unsupervised Learning of Visual Features by Contrasting Cluster Assignments》](https://paperswithcode.com/paper/unsupervised-learning-of-visual-features-by)
>- [《无监督对比学习之假装自己有监督的SwAV》](https://blog.csdn.net/weixin_42764932/article/details/112845236)、

#### 4.5.1 研究动机
&#8195;&#8195;`SwAV`即swap assignment view的缩写，意思就是一张图片不同视角的特征可以互相预测，因为来自同一张图片的不同视角特征按道理来说都是相似的。具体的做法，就是将聚类加入到了对比学习中。（将匹配问题转为预测问题，预测时借助簇类中心）
&#8195;&#8195;作者认为之前的对比学习，直接拿所有图片的编码特征去做对比有点原始而且计算量太大，因为所有的图片都是自己的类。作者考虑，能不能不做近似，能不能借助一些先验信息，一些更简洁的东西比进行对比，而不是和所有负样本直接进行对比。由此作者提出了可以和聚类中心特征进行对比（128万张图片被聚成3000个簇类中心`cluster center`）。
>&#8195;&#8195;比如MoCo在ImageNet上训练那就有128万类，即使在计算loss时取近似，只是取队列编码器里的作为负样本，那负样本也有6万多个。
>&#8195;&#8195;之前的一些聚类方法常常将`ImageNet`数据集聚成`3000`个簇类中心。

&#8195;&#8195;作者选择聚类这个想法有两个原因。首先，聚类方法也是一种无监督的特征表示学习方式，其目标也是希望相似的物体聚在一起，不相似的物体尽量互相远离，这个思想与做法和对比学习都比较接近；第二就是论文一作之前是做聚类的，比如deep cluster，也是一篇很好的无监督学习论文。

#### 4.5.2 算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f50c12d33e49b7f2bfff24930aa09d8f.png)
- 左图：普通的对比学习方法。
同一张图片，做两次数据增强得到$x_1,x_2$（正样本），然后所有的样本通过一个图片编码器（比如ResNet50等等，也可以加几层mlp之类的，这里没有具体说明）输出编码特征$z_1,z_2$，然后在编码特征z上去做对比学习。
- 右图：`SwAV`的做法
	- 每个batch输入数据为  $x\in R^{N*C*H*W}$， 分别经过不同的Aug， 得到 $x_1, x_2$
	- 将$x_1, x_2$ 输入编码器中，得到编码特征$z_1, z_2 \in R^{N*d}$
	- 已知K个聚类中心prototypes $\left \{ c_1,...,c_K \right \}$，表示为$C\in R^{K*d}$。将编码特征与聚类中心计算相似度，得到相似度矩阵$Q \in R^{K*N}$，这样算完又获得了一个新的表示 $q_1, q_2$（Codes）
理想情况下，样本与自己的类簇中心相似度为1，与其他的为0，类似于有监督任务中的one-hot label。不过作者发现soft label效果会好一些。
	- 理论上同一张图片不同view（比如Augment）所产生的 z 和 q 可以相互预测。也就是说，如果拿$z_1$这个特征去跟c去做点乘，按道理来说也是可以去预测$q_2$；反之亦然。所以说点乘之后的结果就是预测，而ground truth就是之前按照clustering分类而得到的q1和q2。作者由此定义了新的loss：
$$L(z_{t},z_{s})=l(z_{t},q_{s})+l(z_{s},q_{t})$$
其中
$$l(z_{t},q_{s})=- \sum _{k}q_{s}^{(k)}\log gp_{t}^{(k)}$$
 $$p_{t}= \frac{exp(z_{t}^{T}c_{k}/ \tau)}{\sum _{k^{\prime}}exp(z_{t}^{T}c_{k}// \tau)}$$
- 通过这种Swapped prediction，也就是换位预测的方法，SwAV可以对模型进行训练 。

**用聚类做对比学习的好处到底有哪些？**
1. 减少计算量
聚类可以将需要对比的样本数大大减少。比如之前的对比学习，需要去和成千上万的负样本进行对比，即便如此也只是算一个近似。而如果只是跟聚类中心做对比，则只需要最多3,000个聚类中心。因为ImageNet也就1,000类，COCO才80类，所以说 3,000个聚类中心就足够用了。
2. 聚类对比更加合理
随机抽取的负样本中，有的可能还是正样本（本身就相似的图片），而且有的时候抽出来的负样本类别也不均衡。但是聚类中心是有明确的语意含义的，自然更加有效，这也是SwAV的基本思想。

#### 4.5.3 Multi-crop增强
`SwAV`也提出了一种新的数据增强方法`Multi-crop`（多次裁剪）。
- 之前的那些对比的学习方法都是用的两个crop。比如输入一张图片，先把它resize 到256×256，然后随机crop两个224×224的图片当成 正样本对$x_1, x_2$ 
- `Multi-crop`：一张图片经过两个160×160的crop，和四个96×96的crop得到6个正样本。
前两个crop争取学习到全局特征，后四个crop争取学到局部特征。因为之前crop尺寸是224，明显非常大，学习到的基本都是全局特征。如果可以学习局部特征，就更容易关注到局部的物体了。但是为了保持和原来计算量差不多，所以原先的尺寸从224降到了160。

这个想法非常简单但是确实有用，在后面的很多对比学习中也被一直沿用。
#### 4.5.3 实验
**1. 不同模型在ImageNet上的Top-1精度对比**
- 左侧图：冻结模型只将其作为特征提取器，然后接一个线性分类器。此时`SwAV`（batch_size=4096，且训练了800epoch）比目前最好的无监督模型`MoCov2`还要高4.2个点，和自监督的baseline模型只差了1.2%（之后要讲的BYOL和SimSiam都是74点几）。
- 右侧图：backbone使用不同宽度的ResNet50后的对比结果。可以看到随着模型宽度的增加，其效果也更好。最大的模型（`SwAV5×`）和有监督baseline模型精度差缩小到了0.6%。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3c5963f546fb36e0670884347b5187c9.png)
下图是使用ResNet50作为backbone时，在少量有标签数据上进行微调后的top-1和top-5精度。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c7f9620fac7dd2fd9dd8f295cfb37f0e.png)
**2. Multi-crop消融试验**

下面是将模型作为特征提取器后在ImageNet上训练不同epoch时的的top-1精度对比（backbone都是ResNet50）
- 左图：有监督模型、对比学习模型、聚类模型是否使用Multi-crop的效果对比。其中自监督模型训练400个epoch，有监督baseline训练200个epoch
- 右图：`SwAV`训练不同的epoch时，训练时长和精度对比
- 可以看到，如果没有这个multi crop的这个技术，把这四个点拿掉，其实SwAV的性能也就跟MoCo v2是差不多的，也就是说一个纯聚类的方法，或者说聚类和对比学习结合的方法其实也并没有什么优势，真正提点的是multi crop的技术 （老师在精读视频里说的啊）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5bfe43efb57e33153de7f64c98bc4e5c.png)
下图是batch_size=256时，训练不同epoch的效果对比（推理方式和上面一样）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a3a9af34adf5ef7db6974ca4ae0d0519.png)
### 4.6 CPCv2
&#8195;&#8195;简单提一下。CPCv2其实也是融合了很多的技巧，它用了更大的模型、用了更大的图像块、做了更多方向上的预测任务，把batch norm 换成了 layer norm，而使用了更多的数据增强，所以这一系列操作下来，CPC v2直接就把CPC v1之前在 ImageNet 上40多的准确率一下就拔到70多。
## 五、第三阶段----不用负样本（BYOL和SimSiam）
>参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14700928?spm_id_from=333.999.0.0)
>
&#8195;&#8195;其实在上一阶段已经有不用负样本的趋势了，比如`SwAV`就是用的聚类中心进行对比。接下来要讲的BYOL和SimSiam其实就是正样本自己在玩，已经没有负样本或者聚类中心这样明确的一个对比的东西去做对比了。
### 5.1 BYOL
>- 论文 [《Boostrap Your Own Latent：A New approach to Self-Supervised Learning》](https://paperswithcode.com/paper/bootstrap-your-own-latent-a-new-approach-to-1)
>- BYOL分析博客[《Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)》](https://generallyintelligent.ai/blog/2020-08-24-understanding-self-supervised-contrastive-learning/)

#### 5.1.1 前言
&#8195;&#8195;`BYOL`就是论文标题`Boostrap Your Own Latent`的缩写。Latent、Hidden、Feature、Embedding其实都是特征的意思，就是各种花里胡哨的用法而已；Boostrap就是类似自我改造的意思。
&#8195;&#8195;`BYOL`使用了一种新的对比学习方法（A New approach），即没有引入任何形式的负样本，而是用图片的编码特征（梯度更新）去预测自己的编码特征（动量更新），模型就这样训练起来了。（相当于用一个视角的特征取预测另一个视角的特征，将匹配转为预测问题）
&#8195;&#8195;这种训练方式类似`SwAV`，但是这次连簇类中心都没了，所以听起来有点不可思议。后来还有一篇博文分析了`BYOL`，认为其实是在使用BacthNorm时引入了隐式的负样本进行对比学习。BYOL作者一听不高兴了，这样不是说明我的工作大大折扣了吗，所以立马写了一篇技术实验论文驳斥了这个说法，证明了对比学习完全不使用负样本是可行的（后面会详细介绍）。
#### 5.1.2 算法
下面是`BYOL`模型总览图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a5d548a9b46092668fbe47be9264cf0a.png)
前向过程：
- 输入x经过两次不同的Aug得到$v,{v}'$。
- 编码特征
	- 上面的online分支$v$经过编码器$f_{\theta }$得到编码特征$y_{\theta }$，$f_{\theta }$是梯度更新
	- 下面的target分支$v'$经过编码器$f_{\xi }$得到编码特征${y_{\xi }}'$，$f_{\xi }$和$f_{\theta }$模型结构一样，但用的是动量更新的方式。也就是说，  $f_{\xi }$引入了MoCo中的动量编码器，其参数和$f_{\theta }$不同，但是结构一样。
	- 如果这两个编码器都是ResNet50，则输出特征是2048维
- projection head
	- 使用类似`SimCLR`中一样的projection head $g_{\xi }$和$g_{\theta }$（也是一个MLP，`BYOL`中也把这个结构叫`predictor`），将特征降到256维，得到特征$z_{\theta },{z_{\xi }}'$。
	- $g_{\xi }$和$g_{\theta }$分别是梯度更新和动量更新，但二者结构一样。
- 对比预测
	- 在 SimCLR中，是在$z_{\theta },{z_{\xi }}'$之间做maximum agreement，即使不同增强后再编码和MLP映射后的特征尽可能的接近
	- 在SwAV中，是将$y_{\theta },{y_{\xi }}'$分别和K个簇类中心c计算相似度得到$q_\theta, q_\xi$，然后互相预测作对比学习（$y_{\theta }$和相似度矩阵点乘的结果去预测$q_\xi$，反之亦然）
	- `BYOL`中，上分支使用`prediction head`（也是`predictor`结构）将$z_{\theta }$映射为$q_{\theta }(z_{\theta })$，然后用$q_{\theta }(z_{\theta })$去预测$sg({z_{\xi }})'$来进行对比学习，其中sg表示`stop-gradient`，因为下分支编码器是动量更新。
	- 损失函数是`MSELoss`，即直接计算预测特征$q_{\theta }(z_{\theta })$和标签$sg({z_{\xi }})'$这两个向量之间的mse。

推理：
&#8195;&#8195;当训练完成只留下编码器$y_{\theta }$，剩下所有的东西都被拿掉了。然后用这个编码器编码图片，输出维特征去做下游任务的推理。

对比：
- 按过程来看，`BYOL`就是将上分支输入经过一个梯度更新的编码器和两个`predictor`得到的$q_{\theta }(z_{\theta })$，去预测下分输入经过一个动量更新的编码器和一个`predictor`得到的$sg({z_{\xi }})'$。
- 所以可以看出`BYOL`使用了MoCo的动量编码器、SimCLR的`projection head`以及预测任务，但是没有负样本，目标函数也不一样。通过自己预测自己就学起来了。
- `BYOL`的两个分支叫online和target，其实就相当于`MoCo`中的query和key分支。

#### 5.1.3 学习机制分析
##### 5.1.3.1 为何不使用负样本这么重要
- 在对比学习中，负样本是一个约束。如果在算目标函数的时候只有正样本，也就是让所有相似的物体的特征也尽可能的相似，此时就有一个很明显的捷径：模型输出恒等于输入，对比学习的oss永远都是0，模型直接就躺平（也叫模型坍塌`model collapse`，表示模型根本就没有在学习）。
- 只有加上负样本这个约束，即不相似的物体也要有不相似的特征，这样模型才会继续学习，否则负样本的loss就无穷大了。所以加入负样本能防止模型学到捷径，是必须的。
- `BYOL`之所以神奇就是它没有用负样本，正样本自己跟自己学最后在ImageNet上也达到了74.3的top-1准确率，也是相当高了。

##### 5.1.3.2 不同对比学习模型projection head结构对比
1. `SimCLR` 
	- 编码特征y经过projection head映射为z，z之间进行对比学习。
	- projection head结构如右图所示：`Linear（2048×2048）+BN+ReLU +Linear（2048×128）+BN`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/abcfcd833a66490e2bc4926c8832e6fe.png)
2. `MoCov2`（MoCo v1没有用projection head）
	MoCov2确实是用了projection head，就是$g_θ$，但是$g_θ$ 里面是没有batch norm的，其结构是`Linear（2048×2048）+ReLU +Linear（2048×128）`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5f4e272ff273e931e66695c0546a0939.png)
3. `BYOL`
.	 - $g_{\xi },g_{\theta },q_{\theta }$都是projection head，其结构为`Linear+BN+ReLU +Linear`
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ee7e87ac1ee3d89df70109ddfd7284b0.png)

##### 5.1.3.3 `BYOL`被认为是使用了隐式负样本
&#8195;&#8195;`BYOL`发布到arxiv之后，在reddit、twitter、知乎全都引起了剧烈的讨论，因为大家都觉得很不可思议；不用负样本，只是自己预测自己，模型的学习怎么能不坍塌。由此引出了一篇博文[《Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)》](https://generallyintelligent.ai/blog/2020-08-24-understanding-self-supervised-contrastive-learning/)。
&#8195;&#8195;这篇博文的作者在复现`BYOL`时遗漏了一个小细节，即借用了 `MoCov2`的`projection head`导致`projection head`中没有加`batch norm`，最终模型坍塌。作者就觉得实在是太奇怪了，所以赶紧又做了一些额外的实验，如下表所示 ：

Name|	Projection MLP Norm|	Prediction MLP Norm|	Loss Function|	Contrastive|	Performance 5
|--|--|--|--|--|--|
Contrastive Loss|	None|	None|	Cross Entropy|	Explicit	|44.1
BYOL|	Batch Norm	|Batch Norm|	L2|	Implicit	|57.7
Projection BN Only|	Batch Norm|	None|	L2|	Implicit|	55.3
Prediction BN Only|	None|	Batch Norm	|L2|	Implicit	|48
No Normalization|	None|	None|	L2|	None|	28.3
Layer Norm|	Layer Norm|	Layer Norm|	L2|	None|	29.4
Random|	—|	—|	—|	None|	28.8
- `Projection MLP Norm`：第二第三列这里指的是两层`Projector`有没有用归一化
- `Loss Function`：普通对比学习loss是交叉熵损失函数，而`BYOL`用的是L2 loss，即mse 损失函数。
- `performance`：测试模型性能是在一个STL-10的数据集上做的，不是 ImageNet，但衡量标准还是准确度。

实验结果：
-  `random`：使用一个随机初始化的残差网络，没有经过任何训练，直接去抽特征。然后在这个特征上训练一个全连接层，最后的结果是28.8。所以这个结果是一个完全随机的结果。
- 正常的`BYOL`：两层`Projector`都使用BN，效果最好
- `BYOL`变体：只使用一层BN，模型起码 也有学到东西。如果是都用LN或者干脆都不用BN，模型坍塌，什么都没学到。

最终分析：

&#8195;&#8195;作者认为在`Projector`层使用BN之后，是计算了整个batch的均值和方差，这意味着是有信息泄露的（MoCo使用了 Shuffling BN ，就是为了防止这种信息泄露）。模型不光是正样本自己和自己学，还和batch norm产生的平均图片（mode，中值）对比着学，这种平均图片就类似 `SwAV`的聚类中心了。

&#8195;&#8195;所以说，这篇博客的作者认为batch norm是`BYOL`能够成功的关键，其实是做了一种隐式的对比学习，这个观点很快就被大家所接受了，因为听起来确实很合理，而且后续试验也都验证了这一点。batch norm确实至关重要，拿掉batch norm以后模型就是不好训练，对超参数的设置非常的敏感，稍有不慎它就啥也不学了。

##### 5.1.3.4  BN只是加强模型的稳定性，`BYOL`不使用负样本的思想是没问题的
&#8195;&#8195;BYOL的作者看到博客就急了，如果真是这样的话，BYOL就还是没有逃脱出对比学习的范畴，它还是找了一个东西去做对比，其创新性就大大降低了。所以作者赶紧做实验，看看能不能找到BYOL 模型不坍塌的另外一种解释。最终又写了一篇论文进行回应。
&#8195;&#8195;这篇论文叫BYOL works even without batch statistics，即在没有batch norm的时候`BYOL`照样能工作，详细的消融实验结果如下表所示 ：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1684bbdcd4f2231a0c804a4f15806966.png)


作者是在encoder（比如ResNet50）和两层`Projector`里分布使用BN/LN和什么都不用去做对比实验，最后发现：
- BN非常关键：只要是projector中没有BN的地方，`SimCLR`性稍微下降；但是`BYOL`全都模型坍塌了 
- 有BN也会坍塌：作者找到了特例（红色框），即使当projector有BN的时候，`BYOL` 还是训练失败了 。如果BN真的很关键，它真的提供了隐式负样本的对比学习的话，训练就不应该失败 
- 完全没有BN，`SimCLR`也坍塌（最后三列的结果。要注意`SimCLR`只有一层projector）。这表明完全不用归一化，`SimCLR`这种使用负样本进行对比学习的方式也无法训练。

&#8195;&#8195;最终结论：BN跟它原来的设计初衷一样，主要作用就是提高模型训练时的稳定性，从而不会导致模型坍塌 。作者进一步延伸，如果一开始就能让模型初始化的比较好，后面的训练即使离开了BN也没有问题。
&#8195;&#8195;作者为此又设计了一个实验，借鉴`BEiT`中的`group norm+weight standardization` （前者也是一种归一化方式，后者是一种模型初始化的方式，但都没有进行批量统计操作），BYOL的top-准确率可以达到74.1%，和原来精度可以认为是一样了（74.3%）。
### 5.2 SimSiam
>- 论文：[《Exploring Simple Siamese Representation Learning 》](https://paperswithcode.com/paper/exploring-simple-siamese-representation)
>- [《CVPR 2021 Oral | 何恺明团队提出SimSiam：探索简单的孪生表示学习》](https://blog.csdn.net/weixin_42111770/article/details/123723652)

#### 5.2.1 `SimSiam`：化繁为简
&#8195;&#8195;`SimSiam`即simple Siamese network（简单孪生网络）。在BYOL发布时，就已经有很多对比学习的分析性工作了。大家发现，对比学习的成功好像是被很多trick一点点堆起来的性能，比如projection head、更多的数据增强、使用用动量编码器、更大的 batch size等等，好像都缺一不可。
&#8195;&#8195;这样因素太多就不方便分析，也不知道每个点到底带来了哪些贡献，所以凯明团队又再次出手，把整个过程化繁为简了一下，最后提出了SimSiam。
&#8195;&#8195;SimSiam结构非常简单，不需要用负样本（结构类似 `BYOL`）、大的batch size，也不需要动量编码器。然而即使在这种情况下，模型效果也很好。
#### 5.2.2 算法
**1. 模型结构**
具体的模型总览图如下图所示，整体结构非常类似 `BYOL`：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7ab44b060bf8fc637864232bbcca15e1.png)
**2. 伪代码**
```python
Algorithm 1 SimSiam Pseudocode, PyTorch-like
# f: backbone + projection mlp
# h: prediction mlp

for x in loader: # load a minibatch x with n samples
	x1, x2 = aug(x), aug(x) # random augmentation
	z1, z2 = f(x1), f(x2) # projections, n-by-d
	p1, p2 = h(z1), h(z2) # predictions, n-by-d
	L = D(p1, z2)/2 + D(p2, z1)/2 # loss
	L.backward() # back-propagate
	update(f, h) # SGD update
def D(p, z): # negative cosine similarity
	z = z.detach() # stop gradient
	p = normalize(p, dim=1) # l2-normalize
	z = normalize(z, dim=1) # l2-normalize
	return -(p * z).sum(dim=1).mean()
```
&#8195;&#8195;D函数就是定义怎么计算loss，这里使用的也是mse损失函数。结合模型结构和伪代码，其前向过程如下：
- image x经过两次数据增强得到$x_1,x_2$
- 经过两个编码器`encoder f`（结构一样参数共享，所以叫孪生网络）得到编码特征$z_1,z_2$。
- $z_1,z_2$的经过`Projector`得到预测 $p_1,p_2$，然后计算对称性loss（$p_1$预测$z_2$，同时$p_2$预测$z_1$，单次结果除以2）。
- 和`BYOL`不同的是，这里没有使用动量编码器，两个encoder完全一样。


#### 5.2.3 实验
##### 5.2.3.1 `stop-gradient`避免了模型坍塌
作者做了一系列实验分析，发现SimSiam能够成功训练，而没有模型坍塌，主要是因为有`stop gradient`这个操作。
1. `stop gradient`操作的影响
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9d5eb9cfc650f14a1ca7dc405c26c536.png)
	- 蓝色线表示使用`stop gradient`操作，红色线表示witdout stop-gradient
	- 左图对比训练损失：不使用`stop gradient`时，优化器快速找到一个退化解，并且达到最小损失值− 1 
	- 中间图：验证退化解是由模型坍塌导致的。作者研究了$l_{2}$正则化输出$z / \| z \|_{2}$ 的标准差std。如果输出坍塌为一个常数向量，那么它们在所有例子上的std对于每一个通道应当是0，中间图的红色曲线验证了这一点。如果输出z 具有零均值各向同性高斯分布，那么的标准差为 $\frac{1}{\sqrt{d}}$ ，中间图的蓝色曲线显示在带有stop-gradient的情况下，它的标准差接近于 $\frac{1}{\sqrt{d}}$。
	- 右图训练集acc：接KNN分类器时的验证集精度，没有`stop gradient`时精度为0
	-  右表：ImageNet linear evaluation，w/ stop-grad操作试验超过了五次

结论：上述实验表明，“坍塌”确实存在，但不足以证明是`stop gradient`避免了坍塌

2.  batch sizes和BN的影响
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/95b60245dc63f58e9ae882427458ef5b.png)
	- batch sizes=256时效果就够了
	- 去掉所有BN后`SimSiam`依旧有学习，尽管精度只有34.6%。对backbone隐含层添加BN后精度则提升到了67.4%；
	- `encoder f`最后的线性层输出（从2048降维）添加BN，精度可以进一步提升到68.1%；
	- `Projector h`添加BN ，训练反而不稳定，loss波动太大

结论：BN有助于训练优化，但主要是提高模型训练的稳定性，而非避免模型坍塌（见第一行结果）。

3. Similarity Function和Symmetrization（对称性损失）
作者还试验了Similarity Function和Symmetrization，发现不使用余弦相似性，使用非对称损失模型都训练的还可以，只是性能会下降一点。所以这两点也和模型坍塌无关。

&#8195;&#8195;通过上面的一些列消融实验对比分析可知，优化器、BN、相似性函数、对称损失可能会影响精度，但与“坍塌”避免无关；对于避免“坍塌”起关键作用的是`stop-gradient`操作。
##### 5.2.3.2 交替优化假设
&#8195;&#8195;`SimSiam`到底在隐式的优化什么？作者认为可以将SimSiam当做是一个[EM算法](https://blog.csdn.net/v_JULY_v/article/details/81708386)。因为stop gradient操作将一套模型参数被人为劈成了两份，即需要解决两个子问题，模型的更新其实也是在交替进行的。
&#8195;&#8195;作者假设SimSiam是一种类似交替优化的方案后（其SGD更新间隔为1），基于该假设，此方案在多步SGD更新下应该同样有效。为此，作者设计了一组实验验证上述假设，结果见下表：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/88e157b36fb5e31d1c2af1d71691ca1d.png)
- 这里1-step就代表`SimSiam`
- 更多步的SGD更新甚至可以取得比`SimSiam`更优的结果

结论：交替优化是一种可行的方案，而`SimSiam`是其特例。

&#8195;&#8195;作者接下来又做了一些推导，到最后可以把`SimSiam`理解成是一个k-means聚类问题。在k-means中，也是分两步走的。每次先要把所有的点分配给一些聚类中心；分配完后再更新这些聚类中心。后面就是不断迭代这两个过程。

##### 5.2.3.3 模型总结及效果对比
作者对比了所提方法与其他对比学习SOTA方法的区别&联系所在，见下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3107525a4f1a1232e742613314198168.png)
- 对比`SimCLR`：SimSiam可以是作为“SimCLR without negative”（SimCLR依赖于负采样以避免“坍塌”）；
- 对比 `SwAV`：SimSiam可以视作“SwAV without online clustering”；
- 对比`BYOL`: SimSiam可以视作“没有动量编码器的BYOL”。

**`SimSiam`与其他SOTA无监督学习方法效果对比：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eedb212e7cb3e0d0992f5110f28c7410.png)
- 上图给出了在ImageNet上的精度对比（backbone都是ResNet50）。
	- SwAV精度只有71.8，这个应该是没有用multi crop的技术。
	- 在训练100个epoch时，`SimSiam`具有最高的精度；但更长的训练时长所得收益反而变小。
	- 实验证明去掉负样本、动量编码器、大的batch-size这些trick，模型也能训练的很好。
- 下图是迁移学习性能对比。
	- 前三个是检测任务，最后一个是分割任务
	- SimSiam和MoCov2的特征迁移性最好。
	- 直到现在，做一些一些对比学习的尝试工作时，还是会用`MoCov2`当基线模型，因为训练快、效果稳，而且下游任务迁移的好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8832669bb2af82a945b0c70a986965f2.png)
## 六、第四阶段：融入transformer（MoCov3、DINO ）
>参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14764424)

### 6.1 MoCov3
>论文：[《An Empirical Study of Training Self-Supervised Vision Transformers》](https://paperswithcode.com/paper/an-empirical-study-of-training-self)、[官方代码Code](https://github.com/facebookresearch/moco-v3)
####  6.1.1 前言
&#8195;&#8195;无监督的预训练（BERT/GPT等）已经彻底改变了NLP，自从Vision Transformer成功之后，将ViT引入CV领域的自监督训练已经是大势所趋了。但是使用ViT作为backbone会导致训练很不稳定，这种不稳定性是造成模型准确率降低的一个主要问题。
&#8195;&#8195;本文作者发现只需要做一点小小的改动（冻结ViT的`patch projection`层），就能让这个训练变得更稳定、效果也更好。所以作者不得不写一篇论文来把这个发现告诉大家，也就是标题说的An Empirical Study （一个实验性的study ）。这篇论文是ICCV 21的一篇口头报告论文，但它的的影响力依旧很大。
#### 6.1.2 伪代码
MoCo v3的架构，其实就相当于是MoCo v2和SimSiam 的一个合体。因为没有模型总览图，所以直接看伪代码：
```python
# f_q: query encoder: backbone + proj mlp + pred mlp
# f_k: key momentum encoder: backbone + proj mlp
# m: momentum coefficient
# tau: temperature，也就是τ
for x in loader: # load a minibatch x with N samples
	x1, x2 = aug(x), aug(x) # augmentation
	q1, q2 = f_q(x1), f_q(x2) # queries: [N, C] each
	k1, k2 = f_k(x1), f_k(x2) # keys: [N, C] each
	loss = ctr(q1, k2) + ctr(q2, k1) # symmetrized
	loss.backward()
	update(f_q) # optimizer update: f_q
	f_k = m * f_k + (1-m) * f_q # momentum update: f_k
	
# 对比 loss
def ctr(q, k):
	logits = mm(q, k.t()) # [N, N] pairs
	labels = range(N) # positives are in diagonal
	loss = CrossEntropyLoss(logits/tau, labels)
	return 2 * tau * loss
```

- 整体的框架来说，它还是有两个网络：query编码器和key编码器（动量编码器），目标函数是对比学习loss，所以说从这个角度讲，它是个`MoCov2`
- query编码器除了backbone之外，还有projection head和predictor，而是还算了对称性loss，即`loss = ctr(q1, k2) + ctr(q2, k1)`。所以从这个角度讲，它又是`SimSiam`。
- 所以说，从整体结构上来看，`MoCov3`就是`MoCov2`和`SimSiam`一个延伸工作。
#### 6.1.3 实验
1. 冻结`patch projection`可以解决训练不稳定的问题
下图是作者将backbone从一个残差网络换成了ViT之后，模型自监督训练曲线：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/52788b8035bbf6917e8ce0612c27cb47.png)
&#8195;&#8195;从实验中可以看出随着batch的增大或者lr的增大，kNN accuracy都逐渐出现了突然掉点的情况，并且掉点的程度逐渐增加，呈现周期性出现。

&#8195;&#8195;作者后来观察了一下模型训练时每一层梯度回传的情况。作者发现，每次准确度大幅下降时，模型第一层梯度也会有一个波峰。于是作者尝试将这一层的权重全部冻住，结果发现问题就解决了。而且很神奇的是这个trick不光是对`MoCov3`有用，它对`BYOL`和 `SimCLR`也有用。
>&#8195;&#8195;第一层就是`ViT`的`patch projection`层，会将图片分割成一个个patch，然后经过线性层映射为Pacth embedding。
2. 模型性能对比
下图对比了 `MoCov3`和 `SimCLRv2`以及BYOL的性能， `MoCov3`更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bf013469b53fb211a429415ab9d89af0.png)
- ViT-BN：ViT模型中使用的是LN，而 ResNet默认是使用BN。为了避免这一点区别对实验的影响，将ViT中MLP层的LN替换为BN，acc提高了一个点
-  ViT-BN/7：将patch的尺寸降为7×7，也就是图片被切分为更小的patches，模型精度提高了2到3个点，但是计算量增加了6倍。
- `MoCov3 ViT-BN-L/7` 精度最高，达到了81%。之前最好结果是`SimCLR v2 (SK-ResNet152-3×)`的79.8%  , 以及`BYOL (ResNet200-2×)`的79.6%.

3. 不同配置模型的训练时长
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/724eddf970f80f3158985786a1862f00.png)
### 6.2 DINO
>- 论文[《Emerging Properties in Self-Supervised Vision Transformers 》](https://paperswithcode.com/paper/emerging-properties-in-self-supervised-vision)
>- [《论文笔记 ：DINO - Emerging Properties in Self-Supervised Vision Transformers》](https://zhuanlan.zhihu.com/p/370199613)、[《DINO》](https://blog.csdn.net/MengYa_Dream/article/details/120797395?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166745465316782425157315%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166745465316782425157315&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-120797395-null-null.142%5Ev62%5Epc_rank_34_queryrelevant25,201%5Ev3%5Econtrol_2,213%5Ev1%5Et3_esquery_v1&utm_term=Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers&spm=1018.2226.3001.4187)

#### 6.2.1 前言
&#8195;&#8195;`DINO`这个名字，来自于它的题目self distillation with no labels，也就是无标签的自蒸馏方法（学生网络预测教师网络的输出）。本文和`MoCov3`一样，也是一种自监督训练Vision Transformer的方式，但作者使用另一种操作——centering，使ViT可以稳定训练。另外本文发现自监督训练为 Vision Transformer features提供了一些新的特性。

**1. 研究动机**
&#8195;&#8195;CV领域，Vision Transfomer（ViT）虽然可以取得和convnets（卷积网络）相比拟的结果，但是还没有展现出足够的优势。比如，相比于convnets，ViT需要更多的计算资源和数据，但是他们的features并没有展现出独特的特性。

&#8195;&#8195;transformer在NLP中的成功的一个主要方面来自自监督预训练的应用（BERT，GPT)，因为自监督训练出来的特征会包含更丰富的语义和信息。另一方面，卷积网络的自监督学习在CV领域也表现出很大的潜力。受此启发，本文将ViT和自监督学习结合，并研究**自监督预训练对ViT feature的影响**。

>&#8195;&#8195;自监督学习通过利用句子中的词创建`pretext tasks` ，相比于有监督学习中每个句子对应一个label， `pretext task`提供了更加丰富的学习信号。类似的，图像层面的有监督学习将丰富的图片信息减少到单一的分类概念。
>
**2. 发现**

通过研究，本文发现自监督ViT features具有一些独有的特性
1. 自监督ViT features 中包含清晰的图像语义分割信息，而这在有监督ViT和convnets中都没有类似的表现。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98dd5fe4f488ba635efbcc797c5fa69b.png)
>&#8195;&#8195;一个完全不用任何标签信息训练出来的`Vision Transformer` ，将它的自注意力图进行可视化，会发现能非常准确的抓住每个物体的轮廓，效果甚至可以媲美对这个物体做分割。

2.  只使用一个比较小的ViT backbone（`ViT-S/8`），自监督训练出来的ViT features 就能在KNN分类器中表现的很好，ImageNet数据集的 top-1精度达到78.3%，超过之前的自监督方法。（也就是ViT features直接去做最近邻分类，连线性分类头或微调都不需要）。

&#8195;&#8195;另外在消融实验中证明，动量编码器、multi-crop 数据增强和更小的 ViT patches（计算量更高）都有重要的作用。
#### 6.2.2 算法
模型结构图如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aeb8c240b78f3b0761e76a2ec08fa495.png)
伪代码：

```python
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
	x1, x2 = augment(x), augment(x) # random views
	s1, s2 = gs(x1), gs(x2) # student output n-by-K
	t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
	loss = H(t1, s2)/2 + H(t2, s1)/2
	loss.backward() # back-propagate
	# student, teacher and center updates
	update(gs) # SGD
	gt.params = l * gt.params + (1-l) * gs.params
	C = m * C + (1-m) * cat([t1, t2]).mean(dim=0)
def H(t, s):
	t = t.detach() # 教师网络stop gradient
	s = softmax(s / tps, dim=1)
	t = softmax((t - C) / tpt, dim=1) # center + sharpen
	return - (t * log(s)).sum(dim=1).mean() # 
```


前向过程：
- 一张图片经过不同的视角得到$x_1,x_2$
- $x_1,x_2$分别经过两个编码器$g_{\theta _{s}},g_{\theta _{t}}$（结构相同参数不同，包含projection head和prediction head）得到编码特征。
- teacher网络的编码器$g_{\theta _{t}}$是动量更新；且为了避免模型坍塌，其编码特征会额外进行一个centering的操作
- 这样学生分支和教师分支经过softmax分别得到K维概率分布$p_1,p_2$，然后用$p_1$去预测$p_2$ （$-p_2logp_1$）

>- `DINO`的知识蒸馏是一种范式，是通过训练一个学生网络 $g_{\theta _{s}}$ 去match一个教师网络$g_{\theta _{t}}$的输出。
>- 两个网络分支最后分别输出概率分布$p_s,p_t$，这里概率P是对网络输出进行softmax归一化的结果：
$$P_{s}(x)^{i}=\frac{   exp(g_{\theta _{s}}(x)^{i})  /\tau_{s}  }{\sum_{k=1}^{K}exp(g_{\theta _{s}}(x)^{k})  /\tau_{s} }$$
其中温度参数$\tau_{s} >0$控制分布的sharp程度。$P_{t}$结果也是这样的公式算出。然后通过固定教师网络，训练学生网络使其参数$\theta _{s}$最小化交叉熵损失函数来匹配分布：
$$min_{\theta _{s} }H(P_{t}(x),P_{s}(x)),whereH(a,b)=-alogb$$
>- Teacher Network：和知识蒸馏不同，这里没有一个预先已知的teacher网络。teacher网络来自过去几轮的student网络，因为作者实验发现经过一个epoch训练后冻结teacher网络的训练方式表现不错。（应该是理解为教师网络使用动量编码器，如果参数全部从student网络复制，模型坍塌）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/abe8564e5c307cce7342528f7ce6b883.png)


&#8195;&#8195;前向过程可以看出，DINO也是自己预测自己（student要预测teacher，teacher的输出当成是ground truth ），所以叫自蒸馏。DINO其实就是延续的BYOL，只不过是换了个名字。


|  模型| 左分支 |右分支|
|--|--|---|
| MoCo |query 编码器  |key编码器|
| BYOL |online network |target network|
| BYOL |student network |teacher network|

>- centering：可以看作在teacher分支上加一个偏置项c：$g_{t}(x)\leftarrow g_{t}(x)+c$，其中c通过EMA更新：$c\leftarrow mc+(1-m)\frac{1}{B}\sum_{i=1}^{B}g_{\theta _{t}}(x_{i})$，m是一个大于0的参数。
>
>-  centering可以看做是计算整个batch样本的均值，然后减掉这个均值。centering类似BYOL对于 batch norm 的讨论，因为batch norm也是对整个batch里的样本做了一个均值和方差 。
>- 感觉只是看了个大概，还有很多细节，有空再补把
#### 6.2.3 实验
1. 对比其它自监督模型，DINO效果更优
	- 以下是在ImageNet验证集上的top-1精度，backbone选择ResNet50、ViT-S和各自最优模型backbone这三种结构
	- im/s是在V100 GPU上每次前向128个样本时的吞吐量（throughput）
	- `ViT-S/8`+`KNN  classifiers`，在ImageNet数据集的 top-1精度达到78.3%；`ViT-B/8`接线性分类器，在ImageNet数据集的 top-1精度达到80.1%。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5eea2c7795415152462ec0b580ea2ff7.png)
2. 模型结构消融试验
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/449e7044033a0921d77e22424f819b97.png)
- 模型都是使用 `ViT-S/16`网络预训练300个epoch，然后分别使用线性分类器（Lin.）和KNN测试精度
- 第一行是`DINO`的默认结构：使用动量编码器、 multi-crop 数据增强和交叉熵损失函数。上图标红色和和默认模型不一样的地方
- 结果：
	- 去掉教师网络不使用动量编码器，模型不能训练
	- 去掉 multi-crop或不使用交叉熵损失，模型掉点比较多
	- 加入Predictor略有提点
	- 比其它几个对比学习模型效果更好

## 七、对比学习总结

| 模型 |创新点  |优势|局限性|
|--|--|--|--
|阶段一|百花齐放
| Inst Disc |提出了个体判别的任务，对比学习loss，使用一个 memory bank的外部数据结构去存储负样本来做对比学习  ||特征一致性差
|  Inva Spread | 只使用一个编码器而不需要额外的数据结构去存储负样本  |   可以进行端到端的对比学习 |字典太小，对比学习效果不好
|  CPC v1  | 提出了infoNCE Loss，以及预测型的代理任务，其输入可以是图像、音频、视频、文字或加强学习  |  是一个非常全能的结构  |
|   CMC| 把两个视角的任务扩展到了多个视角，为以后的多视角多模态对比学习打下了基础 。  |    |
| 阶段二  |   |    |
|   MoCov1| Inst Disc的延伸工作，使用队列结构代替 memory bank来存储负样本，使用动量更新编码器代替动量更新特征；把之前对比学习方法都归纳成字典查询问题；第一个让无监督预训练媲美有监督预训练的方法 |  字典大且特征一致性好，训练便宜  |
SimCLR v1|  Inva Spread延伸工作。batch-size加大到8192，引入`projection head` ，使用更优的数据增强（随机裁剪和随机色彩变换）|  端到端训练 |    |
CPC v2|   引入SimCLR v1的几个技术，ImageNet精度直接从40多提到70多|   |    |
 MoCov2|  相比 MoCov1，引入了`projection head`；使用更多数据增强、cosi调度器和更长的训练epoch |   |    |
SimCLR v2| 受noisy student影响，使用伪标签进行半监督训练。相比SimCLRv1使用了更大的backbone，动量编码器和两层的 `projection head` |   |    |
SwAV| 结合聚类和对比学习，使得对比学习不再需要负样本（跟聚类中心对比）；使用`multi crop`技术  |   |    |
阶段三|  不用负样本 |   |    |
BYOL|  处理负样本实在是太过麻烦，所以完全舍弃负样本，自己预测自己（mse loss），也可以训练 |   |    |
SimSiam |  化繁为简，使用孪生网络，不需要动量编码器、负样本、大的batch-size就可以训练。不过一个分支必须是stop gradient，这样交替 优化，类似K-means|   |    |
阶段四| 引入Vision Transformer  |   |    |
 MoCov3| 冻住`ViT`结结构中的patch projection layer就可以稳定训练  |   |    |
DINO| teacher网络的输出先做centering归一化也可以稳定训练  |   |    |

`MAE`火爆了以后，大家都去尝试掩码学习，对比学习又从一个火爆发展期变成了一个发展潜伏期。







