@[toc]
传送门：
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT](https://blog.csdn.net/qq_56591814/article/details/127313216?spm=1001.2014.3001.5501)
- [李沐论文精读系列三：MoCo、对比学习综述（MoCov1/v2/v3、SimCLR v1/v2、DINO等）](https://blog.csdn.net/qq_56591814/article/details/127564330)
- [李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5502)
## 一、Vision Transformer论文精读
>- 论文名称： [An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929)、[论文源码](https://github.com/google-research/vision_transformer)、[Pytorch官方源码](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)、 [ViT PytorchAPI](https://pytorch.org/vision/stable/models/vision_transformer.html)、
>- 参考李沐[《ViT论文逐段精读》](https://www.bilibili.com/video/BV15P4y137jb/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、笔记[《ViT全文精读》](https://blog.csdn.net/SeptemberH/article/details/123730336)
>- 小绿豆[《Vision Transformer详解》](https://blog.csdn.net/qq_37541097/article/details/118242600)及[bilibili视频讲解](https://www.bilibili.com/video/BV1Jh411Y7WQ)
>- transformer原理可参考我的博文[《多图详解attention和mask。从循环神经网络、transformer到GPT2》](https://blog.csdn.net/qq_56591814/article/details/119759105?spm=1001.2014.3001.5502)

### 1.1 引言

#### 1.1.1 前言
&#8195;&#8195;作者介绍的另一篇论文：[《Intriguing Properties of Vision Transformer》](https://arxiv.org/abs/2105.10497)（Vision Transformer的一些有趣特性）如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/032528dfc978bf1dd59e8b5343638c19.png)
- 图a表示的是遮挡，在这么严重的遮挡情况下，不管是卷积神经网络，人眼也很难观察出图中所示的是一只鸟
- 图b表示数据分布上有所偏移，这里对图片做了一次纹理去除的操作，所以图片看起来比较魔幻
- 图c表示在鸟头的位置加了一个对抗性的patch
- 图d表示将图片打散了之后做排列组合

&#8195;&#8195;上述例子中，卷积神经网络很难判断到底是一个什么物体，但是对于所有的这些例子Vision Transformer都能够处理的很好。

#### 1.1.2 摘要
&#8195;&#8195;在VIT之前，`self-attention`在CV领域的应用很有限，要么和卷积一起使用，要么就是把CNN里面的某些模块替换成self-attention，但是整体架构不变。
&#8195;&#8195;VIT的出现，打破了AlexNet出现以来CNN网络在CV领域的统治地位。VIT表明，在图片分类任务中，只使用纯的`Vision Transformer`结构也可以取的很好的效果（最佳模型在`ImageNet1K`上能够达到88.55%的准确率）开启CV新时代。而且<font color='red'> VIT将CV直接当做NLP来做，还打破了CV和NLP的模型壁垒，推进了多模态领域的发展。</font>


>&#8195;&#8195;下图`Ours-JFT`表示在在Google自家的JFT数据集上进行了预训练，`Ours-I21K`表示在ImageNet 21K上预训练，再在ImageNet 1K上测试。实验分数也可以在`paperwithcode`的[Dataset排行榜](https://paperswithcode.com/sota/image-classification-on-imagenet)上看到。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e48d80b5a4dac11b874adf8d86f67c5e.png)
&#8195;&#8195;在这篇文章中，作者主要拿`ResNet`、`ViT`（纯Transformer模型）以及`Hybrid`（卷积和Transformer混合模型）三个模型进行比较，所以本博文除了讲ViT模型外还会简单聊聊Hybrid模型。


#### 1.1.3 引言
&#8195;&#8195;基于self-attention的模型架构，特别是Transformer，在NLP领域几乎成了必选架构。现在的主流方式就是：在大型语料库上训练一个大模型，然后迁移到小的数据集上进行微调。多亏了Transformer的高效性和可扩展性，现在已经可以训练超过1000亿参数的大模型，<font color='deeppink'>随着模型和数据集的增长，还没有看到性能饱和的现象。 </font>（英伟达和微软联合推出的大型语言生成模型[Megatron-Turing](https://paperswithcode.com/paper/using-deepspeed-and-megatron-to-train)，有5300亿参数，还能在各种任务上大幅度提升性能，而没有达到性能饱和）
>&#8195;&#8195;Transformer是对输入序列做self-attention，其复杂度是$O(n^2)$，现在支持的最大序列长度一般就是几百几千。如果直接把图片每个像素值拉平当做输入序列，那么序列就太长了。所以在CV领域CNN一直占主导地位。
>
&#8195;&#8195;受NLP启发，很多工作尝试将CNN和self-attention结合起来。那怎么降低输入序列把它用到CV领域呢？
1. [Non-local Neural Networks](https://paperswithcode.com/paper/non-local-neural-networks)（CVRP，2018）：将原始图片中间层输出的特征图作为输入序列，来传入Transformer（比如ResNet50在最后一个Stage的特征图size=14×14）。
2. [《Stand-Alone Self-Attention in Vision Models》](https://paperswithcode.com/paper/stand-alone-self-attention-in-vision-models)（NeurIPS，2019）：使用孤立注意力`Stand-Alone Axial-Attention`来处理。具体的说，不是输入整张图，而是在一个local window（局部的小窗口）中计算attention。窗口的大小可以控制，复杂度也就大大降低。（类似卷积的操作）
3.  [《Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation》](https://paperswithcode.com/paper/axial-deeplab-stand-alone-axial-attention-for)（ECCV，2020a）：将图片的2D的矩阵想办法拆成2个1D的向量。等于是先在高度的维度上做一次self-attention，然后再在宽度的维度上再去做一次self-attention，大幅度降低了计算的复杂度。
4.  [《On the Relationship between Self-Attention and Convolutional Layers》](https://paperswithcode.com/paper/on-the-relationship-between-self-attention-1)（ ICLR, 2020.）：在相关工作中，作者介绍了，和本文最相似的工作是这一篇文章。其作者从输入图片中抽取2×2的图片patch，然后一样的做self-attention（只在CIFAR-10数据集上做了实验，其数据集大小为32×32，所以patch size=2×2就行）
5. image GPT：在相关工作中，作者也说到image GPT和本文很相近。但是image GPT是一个生成模型，虽然用了Transformer，但最终结果差很多（ImageNet上的最高的分类准确率也只能到72）。不过受此启发，后面出现了[MAE](https://arxiv.org/pdf/2111.06377.pdf)，用一个**自监督模式训练的生成式的模型**，做出了比之前判别式的模型更好的效果（分类和检测）。

&#8195;&#8195;所以，自注意力早已经在计算机视觉里有所应用，而且已经有完全用自注意力去取代卷积操作的工作了。
&#8195;&#8195;这些模型虽然理论上是非常高效的，但事实上因为这个自注意力操作都是一些比较特殊的自注意力操作（除了上面举例的最后一篇），要很复杂的工程去加速算子，所以就导致很难训练出一个大模型。因此在大规模的图像识别上，传统的残差网络还是效果最好的。

&#8195;&#8195;本文是被transformer在NLP领域的可扩展性所启发，本文想要做的就是<font color='deeppink'>直接应用一个标准的transformer作用于图片，尽量做少的修改。好处是可以直接使用NLP中成熟的Transformer架构，不需要再魔改模型，而且Transformer这么多年有很多高效的实现。</font>具体的处理方式见下一节。
>&#8195;&#8195;上面举例的例4模型，从技术上讲就是`Vision Transformer`。但是作者认为二者的区别，是<font color='red'>本文证明了，使用一个标准的`Transformer endoder`（类似BERT，不需要任何特殊改动）在大规模的数据集上做预训练的话，就能取得比现在最好的卷积神经网络差不多或者还好的结果。</font>（这是本文的主要论证工作，在1.4.2和1.4.3章节都有体现 ）另外就是二者可处理图片的分辨率不同（32×32对比224×224）。


引言的最后部分放出了结论：
- 在中型大小的数据集上（比如说ImageNet）上训练的时候，如果不加比较强的约束，Vit的模型其实跟同等大小的残差网络相比要弱一点。
作者对此的解释是：transformer跟CNN相比，缺少了一些CNN所带有的归纳偏置（`inductive bias`，是指一种先验知识或者说是一种提前做好的假设）。CNN的归纳偏置一般来说有两种：
	- `locality`：CNN是以滑动窗口的形式一点一点地在图片上进行卷积的，所以假设图片上相邻的区域会有相邻的特征，靠得越近的东西相关性越强；
	- `translation equivariance`（平移等变性或平移同变性）：写成公式就是`f(g(x))=g(f(x))`，不论是先做 g 这个函数，还是先做 f 这个函数，最后的结果是不变的；其中f代表卷积操作，g代表平移操作。（因为在卷积神经网络中，卷积核就相当于是一个模板，不论图片中同样的物体移动到哪里，只要是同样的输入进来，然后遇到同样的卷积核，那么输出永远是一样的）
	- 一旦神经网络有了这两个归纳偏置之后，他就拥有了很多的先验信息，所以只需要相对较少的数据就可以学习一个相对比较好的模型。但是对于transformer来说，它没有这些先验信息，所以它对视觉的感知全部需要从这些数据中自己学习。
- 为了验证这个假设，<font color='red'> 作者在更大的数据集（ImageNet 22k数据集&JFT 300M数据集）上做了预训练，然后发现大规模的预训练效果要比归纳偏置好。 </font>如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6ff276e10cd3da5e117df7dfdc61b2bb.png)
- 上图中VTAB也是作者团队所提出来的一个数据集，融合了19个数据集，主要是用来检测模型的稳健性，从侧面也反映出了VisionTransformer的稳健性也是相当不错的。
### 1.2 相关工作
&#8195;&#8195;简单介绍了一下Transformer在NLP领域应用最广的两大分支BERT和GPT，都是基于自监督的训练方式（MLM任务和Next word prediction）。
&#8195;&#8195;直接将图片的像素作为序列输入Transformer是不可行的，所以作者介绍了一下之前的相关处理方式（类似上面引言提到的几个模型）。
&#8195;&#8195;第三段介绍了，有一些工作研究了用比ImageNet更大的数据集去做预训练，效果会更好，比如说ImageNet-21k和JFT300M。最终作者也是在这两个数据集上预训练模型。

### 1.3 ViT
#### 1.3.1 整体结构
下图是原论文中给出的关于Vision Transformer(ViT)的模型框架。简单而言，模型由三个模块组成：

- `Embedding`层（线性投射层Linear Projection of Flattened Patches）
- `Transformer Encoder`(图右侧有给出更加详细的结构)
- `MLP Head`（最终用于分类的层结构）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/346678198203ee1678edef5a89814594.png)
如上图所示：
1. `embedding`层（图片输入）
	- 一张图片先分割成n个`patchs`，然后这些patchs一个个输入线性投射层得到`Pacth embedding`。比如`ViT-L/16`表示每个`patchs`大小是16×16。
	-  在所有tokens前面加一个新的`class token`作为这些patchs全局输出，相当于transformer中的CLS（这里的加是concat拼接）。
	-  和transformer一样，引入位置信息。VIT的做法是在每个token前面加上位置向量，即`position embedding`（这里的加是直接向量相加，不是concat）。
self-attention本身没有考虑输入的位置信息，无法对序列建模。而图片切成的patches也是有顺序的，打乱之后就不是原来的图片了。
2. `Pacth embedding`+`position embedding`+`class token`一起输入`Transformer Encoder`层得到其输出。
3. class token的输出当做整个图片的特征，经过`MLP Head`得到分类结果。（VIT只做分类任务）
4. 整个模型是使用有监督的训练（相对于NLP中，transformer类模型很多都是无监督训练）

整体上看，VIT的模型结构还是很简洁的，难点就是如何将图片转为token输入网络。

#### 1.3.2 Embedding层结构详解
&#8195;&#8195;对于标准的Transformer模块，要求输入的是token（向量）序列，即二维矩阵[num_token, token_dim]。对于图像数据而言，其数据为[H, W, C]格式的三维矩阵，明显不是Transformer想要的。所以需要先通过一个Embedding层来对数据做个变换。
1. 如下图所示，首先将一张图片按给定大小分成一堆Patches。以`ViT-B/16`为例（后面都是以此模型举例），将输入图片(224x224)按照16x16大小的Patch尺寸进行划分，划分后会得到$(224/16)^2=196$个Patches。
2. 接着通过线性映射将每个Patch映射到一维向量中。具体的，每个Patche数据shape为[16, 16, 3]，通过映射得到一个长度为768的向量（token）。`[16, 16, 3] -> [768]`
>&#8195;&#8195;在代码实现中，直接通过一个卷积层来实现。卷积核大小为16x16，步距为16，卷积核个数为768。通过卷积`[224, 224, 3] -> [14, 14, 768]`，然后把H以及W两个维度展平即可`[14, 14, 768] -> [196, 768]`，此时正好变成了一个二维矩阵，正是Transformer想要的。
>&#8195;&#8195;如果模型更大的话，`Pacth embedding`可以映射到更大的维度，也就是论文中提到的参数`D`。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/14a88521fbcb372c1b6b02e31c9c8d25.png)
3. 加上[class]token以及Position Embedding。
&#8195;&#8195;在原论文中，作者说参考BERT，添加一个[class]token（用来表示序列特征），然后与之前从图片中生成的tokens拼接在一起，`Cat([1, 768], [196, 768]) -> [197, 768]`
&#8195;&#8195;这里的Position Embedding采用的是一个可训练的参数（1D Pos. Emb.），是直接叠加在tokens上的（add），所以shape要一样，也是`[197, 768]`。
&#8195;&#8195;关于为什么使用`[class]token`和`1D Pos. Emb`，在本文1.5.1消融试验部分会讲到。
#### 1.3.3 Transformer Encoder详解
Transformer Encoder其实就是重复堆叠Encoder Block `L`次，主要由以下几部分组成：
- Layer Norm层标准化：这种Normalization方法主要是针对NLP领域提出的，对每个token进行Norm处理。
- Multi-Head Attention：多头注意力
- Dropout/DropPath：在原论文的代码中是直接使用的Dropout层，在但rwightman实现的代码中使用的是DropPath（stochastic depth），可能后者会更好一点。
- MLP Block，如图右侧所示，就是全连接+GELU激活函数+Dropout组成。需要注意的是第一个全连接层会把输入节点个数翻4倍`[197, 768] -> [197, 3072]`，第二个全连接层会还原回原节点个数`[197, 3072] -> [197, 768]`，原来跟transformer中做法一样。
- 详细结构见下一节结构图右侧部分 `Encoder Block`。

#### 1.3.4 MLP Head和`ViT-B/16`模型结构图
&#8195;&#8195;上面通过`Transformer Encoder`后输出的shape和输入的shape是保持不变的，以ViT-B/16为例，输入的是[197, 768]输出的还是[197, 768]。对于分类，我们只需要提取出[class]token生成的对应结果就行，即`[197, 768]`中抽取出`[class]token`对应的`[1, 768]`。
&#8195;&#8195;接着我们通过MLP Head得到我们最终的分类结果。MLP Head原论文中说在训练ImageNet21K时是由Linear+tanh激活函数+Linear组成。但是迁移到ImageNet1K上或者你自己的数据上时，只定义一个Linear即可。
&#8195;&#8195;下面是小绿豆绘制的`ViT-B/16`模型结构图：（Pre-Logits就是Linear+tanh，一般迁移学习是可以不用的。）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/43f724679372ca93b7773b533b081428.png)

作者在论文`3.1 VIT`部分，对整个过程用数学公式描述了一次：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/db334d112a4e7dce4b4fa51715a4b25c.png)
- $X_p$表示图像块的patch，一共有n个patch，所以有$X_{p}^{1}$到$X_{p}^{N}$，E表示线性投影的全连接层，得到一些patch embedding。
- 第一层transformer endoder的输入是$z_0$；MSA表示多头自注意力（Multi head self attention）。
- $z_{L}^{0}$表示经过L层transformer endoder后输出的第一个位置的token（CLS）的结果

#### 1.3.5 归纳偏置
&#8195;&#8195;在CNN中，locality（局部性）和translate equivariance（平移等变性）是在模型的每一层中都有体现的，这个先验知识相当于贯穿整个模型的始终。
&#8195;&#8195;但是对于ViT来说，只有MLP层是局部而且平移等变性的，其它的自注意力层是全局的，这种图片的2d信息ViT基本上没怎么使用。
>&#8195;&#8195;就是只有刚开始将图片切成patch的时候和加位置编码的时候用到了，除此之外，就再也没有用任何针对视觉问题的归纳偏置了，而且位置编码也是随机初始化的1-D信息。所以在中小数据集上ViT不如CNN是可以理解的。

#### 1.3.6 Hybrid混合模型试验
&#8195;&#8195;既然transformer全局建模的能力比较强，卷积神经网络又比较data efficient（不需要太多的训练数据），那么自然想到去搞一个前面层是CNN后面层是transformer的混合网络，也就是`Hybrid`混合模型。`Hybrid`不再直接将图片打成一个个patch，而是直接送入CNN得到embedding，比如经过Resnet50，最后一个stage输出特征图是14×14，拉直了也是196维向量。这部分细节参考文本1.4实验部分。
#### 1.3.7 更大尺寸上的微调
&#8195;&#8195;之前的工作有表明，使用更大的图片输入尺寸往往模型效果会更好。但是**使用一个预训练好的vision transformer，其实是不太好去调整输入尺寸的**。如果还是将patch size保持一致，但是图片扩大了，那么序列长度就增加了，提前预训练好的位置编码有可能就没用了。
&#8195;&#8195;这个时候位置编码该如何使用？作者发现其实做一个简单的2d的插值就可以了（使用torch官方自带的interpolate函数就完成）。但这只是一个临时的解决方案，如果需要从一个很短的序列变成一个很长的序列时，简单的插值操作会导致最终的效果下降（比如256→512）。这也算是vision transformer在微调的时候的一个局限性。
&#8195;&#8195;因为使用了图片的位置信息进行插值，所以这块的尺寸改变和抽图像块是vision transformer里唯一用到2d信息的归纳偏置的地方。

### 1.4 实验部分
#### 1.4.1 ViT三个尺寸模型参数对比
&#8195;&#8195;在论文的4.1章节的Table1中有给出三个模型（`Base/ Large/ Huge`，对应BERT）的参数，在源码中除了有`Patch Size`为16x16的外还有32x32的。其中：
- `Layers`：Transformer Encoder中重复堆叠Encoder Block的次数
- `Hidden Size`：对应通过Embedding层后每个token的dim（向量的长度）
- `MLP size`：Transformer Encoder中MLP Block第一个全连接的节点个数（是Hidden Size的四倍）
- `Heads`：代表Transformer中Multi-Head Attention的heads数（多头注意力有几个头）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4cd01b4b78f0db312cc7776150f56c45.png)
#### 1.4.2 对比其它最新模型
&#8195;&#8195;对比VIT的几个不同配置模型和`BiT-L`，以及`Noisy`模型。最终效果提升不大，但是`ViT`训练时间短很多。（的后面还有其它的对比试验支持作者的这个观点）
>`TPU-v3-core-days 2.5k`：表示最大的`ViT-H/14`模型在TPU-v3上只需要训练2500天。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/72b5406d063daa8e29a850e4109cf775.png)
#### 1.4.3 `vision trasformer`预训练需要多大的数据规模？（重要论证）
&#8195;&#8195;下图展示了在不同大小的数据集上预训练后，`BiTt`和`VIT`到底在ImageNet的微调效果如何。
- 下图中灰色$\blacksquare$表示`BiT`，上下两条线分别表示使用ResNet50和ResNet152结构；其它五个$\bullet$就是不同配置的`ViT`。 
- 左图是在`ImageNet`、`ImageNet 21K`、`JFT-300M`三种规模的数据集上训练的结果，三种数据集规模分别是1.2Million，14M和300M
- 右图是在`JFT`数据集上分别采样不同规模子集的实验结果，这里`ViT`用作特征提取器而不是微调。（去除了训练时的强约束，比如说dropout、weight decay、label smoothing；而且不同规模的数据集来自同一分布，更能看出模型本身的特性）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/00449859f4e99cb2716e26df097bbbbb.png)
&#8195;&#8195;结论：如果想用`ViT`，那么得至少是在`ImageNet-21k`这种规模的数据集上预训练（数据量14M），否则还不如用CNN。

下图再次论证作者观点（VIT训练更便宜）：
- 左图的`Average-5`就是他在五个数据集（ImageNet real、pets、flowers、CIFAR-10、CIFAR-100）上做了验证后的平均结果
-  右图是`ImageNet` 上的验证结果
- ViT和Hybrid混合模型都是在`JFT-300M`上预训练号的模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eb974d76933996f7b4e7cb01cceb1f83.png)
结论：
- 同等复杂度下， `ViT`比`BiT`效果要好，这也说明`ViT`训练比`CNN`更便宜
- 小规模的模型上，`Hybrid`混合模型比其它两者效果都要好
- 随着模型规模增大，`Hybrid`精度慢慢的跟`ViT`差不多了，甚至还不如在同等计算条件下的`ViT`。为什么卷积神经网络抽出来的特征没有帮助`ViT`更好的去学习？这里作者对此也没有做过多的解释。
- 随着规模的增大，`ViT`和`BiT`都没有性能饱和的迹象。

#### 1.4.5 ViT可视化
>统一见本文1.6 可视化部分。
#### 1.4.6 自监督训练
&#8195;&#8195;这部分放在正文而不是负类中，是因为作者认为：在NLP领域，`Transformer`这个模型确实起到了很大的推动作用，但另外一个真正让`Transformer`火起来的原因其实是大规模的自监督训练，二者缺一不可。
&#8195;&#8195;NLP中自监督方式是MLM任务或者Next word prediction，本文模仿的是BERT，所以作者考虑构造专属于`ViT`的目标函数`Mask patch prediction`。具体来说，给定一张图片，将它打成很多patch，然后将某些patch随机抹掉，然后通过这个模型将这些patch重建出来。
&#8195;&#8195;但是最后`ViT-Base/16`在ImageNet只能达到80的左右的准确率，比最好的有监督方式训练，还是差了四个点，所以作者后面的工作是考虑引入对比学习。
>&#8195;&#8195;对比学习是所有自监督学习中表现最好的，紧接着出现的`ViT MoCo v3`和`DINO`就是在`ViT`的基础上使用了对比学习。
### 1.5 附录
#### 1.5.1 [CLS]token和1D-Position Embedding消融实验
>这部分内容在论文附录的D Additional Analyses部分
>
&#8195;&#8195;虽然最终VIT采用的`[CLS]token`和`1D-Position Embedding`都和`BERT`是一样的，但是作者在论文实验部分也都对此作了很多消融试验。比如：
1. `[CLS]token`是从NLP领域借鉴的，但是之前的CNN网络用于图片分类的特征并非这么做。
	 - 比如ResNet50模型，是在最后一个stage5上输出的特征图（size=14×14）上做一个全局平均池化（GAP），拉直为一个一维向量，来作为整个图片的全局特征。最后用这个特征去做分类。
	 - 作者也试验了在`Pacth embedding`中不加入`[CLS]token`，而是在最后`Transformer Encoder`输出序列向量时，用一个`GAP`来得到最终全局表征向量，最终结果和使用`[CLS]token`的效果几乎差不多 </font>。
	 - 下图表示CLS和GAP最后收敛结果差不多，只是学习率会不一样。最终作者还是使用了`[CLS]token`，因为本文目的是跟原始的Transformer尽可能地保持一致，不想大家觉得效果好可能是因为某些trick或者某些针对cv的改动而带来的，作者就是想证明，<font color='red'> 一个标准的Transformer照样可以做视觉。
3.  对于`Position Embedding`作者也有做一系列对比试验，在源码中默认使用的是`1-D Pos. Emb`.，对比不使用Position Embedding准确率提升了大概3个点，和`2-D Pos. Emb.`以及`Rel. Pos. Emb.`比起来没太大差别。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ea58096adbd21f01d539b58825db136e.png)
>&#8195;&#8195;`2-D Pos. Emb.`（2D位置编码）就是用patch的横纵坐标表示其位置，而不是一个一维的序号。其中横坐标有D/2的维度，纵坐标也有D/2的维度，两者拼接到得到了一个D维的向量。
>&#8195;&#8195;`Rel. Pos. Emb.`：相对位置编码。两个patch之间的距离可以使用绝对位置编码，也可以使用相对位置编码，最终结果差不多。
>&#8195;&#8195;作者对此解释是：ViT是直接在图像块上做的，而不是在原来全局尺度的像素块上做的，所以在排列组合这种小块或者想要知道这些小块之间相对位置信息的时候，还是相对比较容易的，所以使用任意的位置编码都无所谓。

#### 1.5.2  Hybrid混合模型试验（不感兴趣可跳过）

&#8195;&#8195;在论文4.1章节的Model Variants中有比较详细的讲到，就是将传统CNN特征提取和Transformer进行结合。下图绘制的是以`ResNet50`作为特征提取器的混合模型`Hybrid`，但这里的Resnet与之前讲的Resnet有些不同。
1. 首先这里的ResNet50的卷积层采用的`StdConv2d`不是传统的`Conv2d`，然后将所有的`BatchNorm`层替换成`GroupNorm`层。
2. 在原Resnet50网络中，stage1重复堆叠3次，stage2重复堆叠4次，stage3重复堆叠6次，stage4重复堆叠3次，但在这里的ResNet50中，把stage4中的3个Block移至stage3中，所以stage3中共重复堆叠9次。
>&#8195;&#8195;如果有stage4就是下采样32倍，改为只有stage3就是下采样16倍，这样224×224的图片输出的就是14×14大小。

&#8195;&#8195;通过`ResNet50 Backbone`进行特征提取后，得到的特征矩阵shape是[14, 14, 1024]，接着再输入Patch Embedding层，注意Patch Embedding中卷积层Conv2d的kernel_size和stride都变成了1，只是用来调整channel。
&#8195;&#8195;后面的部分和前面ViT中讲的完全一样，就不在赘述。
&#8195;&#8195;<font color='red'>简单说，ViT是图片经过`Conv2d`卷积层得到Patchs，而Hybrid是多加了一步，图片经过`ResNet50 Backbone`进行特征提取后，经过卷积得到Patchs。</font>然后都是加上class token和位置向量输入Transformer Encoder，得到class token都是输出。再经过MLP得到分类结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f214dc10381a12b19ad58d83a1788694.png)
&#8195;&#8195;下表是论文用来对比ViT，Resnet（和刚刚讲的一样，使用的卷积层和Norm层都进行了修改）以及Hybrid模型的效果。通过对比发现，在训练epoch较少时Hybrid优于ViT，但当epoch增大后ViT优于Hybrid。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0807b730745d60d0476f4294cfffd75b.png)

### 1.6 可视化
&#8195;&#8195;分析完训练成本以后，作者也做了一些可视化，希望通过这些可视化能够分析一下vit内部的表征。
1. patch embedding可视化
	- 下图左侧展示了E（第一层linear projection layer）是如何对图片像素进行embedding。
	-  只展示了开头28个token的可视化结果，可以看到结果类似CNN，都是提取的一些颜色和纹理这样的底层特征，作者说它们可以用来描述每一个图像块的底层的结构。
2. position embedding可视化
	- 下图中间部分展示了不同patch之间，位置编码的cos相似性，越接近1 表示越相似。
	- 可以看到黄色（最相似）出现位置和patch本身所处位置对应，说明1D编码已经学习到2D的位置信息，所以换成2D位置编码效果几乎不变。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/96b049fe16c6f2dc83c901640094b6fa.png)
3. 自注意力是否起作用了？
为了了解ViT如何利用self-attention来整合图像中的信息，我们分析了不同层的`Mean attention distance`
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4275cd1723c2fa6e98e7466ec2be1c90.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9b64ea17e703669ae77d91442f612ea9.png)
- `ViT-L/16`有24层，所以横坐标网络深度是从0到24 。
- 图中五颜六色的点就是每一层的transformer block中多头自注意力的头。`ViT-L/16`有16个头，每一列其实就是16个点。
- 纵轴所表示的是`Mean attention distance`（平均注意力的距离：图上两个像素点的真实距离乘以他们之间的attention weights，因为自注意力是全局都在做，所以**平均注意力的距离就能反映模型到底能不能注意到两个很远的像素**）
- 结论;
	- 网络的开始几层，`Mean attention distance`从10几到100多，有的近有的远，这也就证明了**自注意力能够在网络最底层，也就是刚开始的时候就已经能够注意到全局上的信息了**，而不是像卷神经网络一样，逐步提升感受野。
	- 网络的后半部分，模型的自注意力的距离已经非常远了（most heads attend widely across tokens.），也就是说它已经学到了带有语义性的概念，而不是靠邻近的像素点去进行判断。即模型学到的特征越来越高级，越来越具有语义信息。
4. out token输出
	- 在本文的4.5实验部分，作者对上面attention的结果又进行了一次验证。
	- 作者用`output token`的自注意力折射回原来的输入图片，可以发现模型确实是学习到了这些概念。
	- 因为输出的token是融合了所有的信息（全局的特征），模型已经可以关注到与最后分类有关的图像区域
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dced9e6f264330cf98d97ea216c5fb35.png)


### 1.7 论文结论

&#8195;&#8195;`Vision Transformer`和NLP领域标准的`Transforme`r的自注意力机制区别在于：除了在刚开始抽图像块的时候，还有位置编码用了一些图像特有的归纳偏置，除此之外就再也没有引入任何图像特有的归纳偏置了。这样的好处就是可以直接把图片当做NLP中的token，拿NLP中一个标准的Transformer就可以做图像分类了。
&#8195;&#8195;这个策略扩展性非常好，和大规模预训练结合起来的时候效果也出奇的好，而且训练起来还相对便宜。
&#8195;&#8195;另外作者在这里对`Vision Transformer`还做了一些展望：
1. 将VIT从分类扩展到目标检测和分割，所以后面出现了一系列将Transformer应用在在视觉领域的模型：
	- [ViT-FRCNN](https://arxiv.org/abs/2012.09958)
	- SETR，即[Segmentation Transformer](https://paperswithcode.com/method/setr)
	- DETR，即[Detection Transformer](https://paperswithcode.com/method/detr)
	- 2021年3月[Swin-Transformer](https://arxiv.org/abs/2103.14030)横空出世，将多尺度的设计融合到了Transformer中，更加适合做视觉的问题了。
2. 探索一下自监督的预训练方案（NLP领域常用做法）。
本文也做了一些初始实验，证明了用这种自监督的训练方式也是可行的，但是跟有监督的训练比起来还是有不小的差距。
3. 将Vision Transformer做得更大，效果应该会更好。
半年后，本文作者团队又出了一篇[Scaling Vision Transformer](https://arxiv.org/pdf/2106.04560.pdf)，训练出更大的VIT网络，即ViT-G，将ImageNet图像分类的准确率提高到了90以上。
>&#8195;&#8195;除以上几点之外，作者还在`多模态领域`挖了一个大坑。CV和NLP大一统之后，是不是这些任务都可以用一个Transformer取解决呢？后续这一块也是推进的很快。
### 1.8 PyTorch代码实现
>[项目代码github地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)、B站代码讲解[《使用pytorch搭建Vision Transformer(vit)模型》](https://www.bilibili.com/video/BV1AL411W7dT/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>
#### 1.8.1 PatchEmbed层
```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 在VIT-B/16中就是16*16

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 默认不传入norm_layer， nn.Identity()表示不做任何操作
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape # x就表示传入的图片
        # 需要注意的是，VIT模型不像传统的CNN模型那样，可以更改图片的入网尺寸。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]，最后两个维度拉平
        # transpose: [B, C, HW] -> [B, HW, C]，后两个维度交换
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x) # 得到 Patch Embedding
        return x
```
#### 1.8.2 Attention层（实现多头注意力Multi-Head Attention）
```python
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8, # 多头注意力使用几个head
                 qkv_bias=False, # 生成qkv时是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0., # dropout概率，下同
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 每个head的维度，一般要求dim能整除num_heads
        self.scale = qk_scale or head_dim ** -0.5 # 不传入qk_scale时，self.scale=根号dk
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 使用一个全连接层得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim) # 多头拼接后用Wo映射，Wo就是全连接层实现
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # num_patches + 1就是196+1（class token），total_embed_dim=768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```
#### 1.8.3 待续
## 二、MAE论文精读
>参考：
>- 论文[《Masked Autoencoders Are Scalable Vision Learners》](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)
>- 李沐[《MAE 论文逐段精读【论文精读】》](https://www.bilibili.com/video/BV1sq4y1q77t/?spm_id_from=333.999.0.0&vd_source=21011151235423b801d3f3ae98b91e94)、笔记[《MAE 论文逐段精读》](https://www.bilibili.com/read/cv14591955)
>- 知乎[《别再无聊地吹捧了，一起来动手实现 MAE(Masked Autoencoders Are Scalable Vision Learners) 玩玩吧！》](https://zhuanlan.zhihu.com/p/439554945)

<font color='red'>**本文算法在2.3章节！！！**</font >
### 2.1 导言
#### 2.1.1 前言
&#8195;&#8195;自从`Vision Transformer`将标准的transformer成功的应用到CV上面以来，就出现了很多相关性的工作，`MAE`就是其中的一篇。本文是2021.11.11发布在Arxiv上的文章，主要工作是在`Vision Transformer`基础上，引入自监督训练，相当于将`BERT`应用到CV领域。通过完形填空来获取对于图片的理解，把整个训练拓展到没有标号的数据上面 ，使得transformer在CV上的应用更加普及。
&#8195;&#8195;最终MAE只需要`Vision Transformer`百分之一规模的数据集上预训练，就能达到同样的效果。而且在目标检测、实例分割、语义分割等任务上，效果都很好。
>&#8195;&#8195;本文标题中的`Autoencoders`，是‘`自`’而非自动的意思，表示类似自回归这一类模型的特点， 即**标号和样本（y和x）来自于同一个东西**。比如说在语言模型中，每一次用前面的次去预测下一个词。而对于MAE，则表明`模型的标号也是图片本身`。
#### 2.1.2   摘要
&#8195;&#8195;MAE的自监督训练实现途径非常简单，随机地mask图片中地一些`patches`（块），然后再去重构`the missing pixels`.，这个思想来自于BERT中的带掩码的语言模型。

在MAE中，自监督训练还有两个核心的设计：
- ==非对称的编-解码器架构==，非对称体现在两个方面：
	-  两者输入不一样：MAE的编码器只编码可见的patches，被masked的块不编码，而解码器需要重构所有块。
	- Decoder更加轻量的：比如 Encoder 通常是多层堆叠的 Transformer，而 Decoder 仅需较少层甚至1层就 ok。这也表明 Encoder 与 Decoder 之间是解耦的。
	
- ==高掩码率==：比如`mask 75%`，才能得到一个比较好的自监督训练效果（否则只需要插值还原）。这等于只编码1/4的图片，训练速度加快了四倍，所以MAE可以扩展到更大的模型。

&#8195;&#8195;最终作者只使用一个最简单的`ViT-Huge`模型，在`ImageNet-1K`上预训练，精度也能达到`87.8%`。

下面是作者展示的一些模型重构效果图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7a1ef366b9fdeb2992bc03d599c31b75.png)

- 图2：MAE在ImageNet验证集上构造出来的图片，验证集没有参与训练，所以相当于是测试结果。
	- 红色框选的这一块，左侧列表示80%的patches被mask，中间列表示MAE重构的结果，右侧列表示原始图片。
	- 可以看到被mask的地方非常多，很难看出原图是什么，但是重构之后还原度非常高，特别神奇。
- 图3：MAE在COCO验证集上的结果，MAE重构效果也很好。
- 图4：MAE取不同的遮盖比例时，模型重构效果对比。可以看到即使mask比例达到95%，还是能还原出大致图像，很玄乎。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/89ec3881d47481b5d82cf94be0cbdaeb.png)

#### 2.1.3 导言： 自编码模型为何在CV领域应用不及NLP？
&#8195;&#8195;在导言一开始，作者大意就说自监督学习很香，使得Transformer在NLP领域的应用非常火。尽管也有一些工作将BERT应用到CV领域，比如[Denoising Autoencoder](https://paperswithcode.com/method/denoising-autoencoder)是在一个图片中加入很多噪音，然后通过去噪来学习对这个图片的理解，但最终这些效果都不及NLP。
>MAE也是一种类似去噪 的模型，mask一些patches，也相当于往模型中加入很多噪音

&#8195;&#8195;为何masked自编码模型在CV和NLP领域应用不一样呢？
1. 架构差异
之前的CV领域都是用的CNN网络，而CNN中的卷积，是不好做掩码操作的。
&#8195;&#8195;因为卷积核滑动时，不好将`msaked patches`单独剔除，导致后面不好将图片还原（Transformer中可以将masked token直接去掉，和其他词区分开来）。但是`ViT`成功将`Transformer`应用到图片分类，所以这个问题不再有了。
2. 信息密度(information density)不同。
	- 在自然语言中，一个词就是一个语义的实体，比如说字典中对一个词的解释就是很长的一段话，所以一句话中很难去去掉几个词（这样做完形填空才有意义）。
	- 在图片中，像素是比较冗余的，取决于相机的分辨率有多大。所以如果只是简单去掉一些块的话，很容易进行插值还原。
	- 作者的做法是，mask很高比例的像素块（比如`75%`），就可以大大降低图片的冗余性。这样就压迫模型必须学习全局信息，也就提高了模型学习的难度。
>&#8195;&#8195;将图片中一大片都去掉， 剩下的块离得比较远，就没那么冗余了。否则模型仅仅学一个局部模型就可以进行插值还原。
>&#8195;&#8195;关于此点论证，看上面图2.3.4就可以发现：仅仅一些局部的，很稀疏的块，就可以重构全局图片。


3. 解码差异

	- NLP中需要预测`masked tokens`，而`token`本身就是一种比较高级一些的语义表示。而来自编码器的特征也是高度语义的，与需要解码的目标之间的 gap 较小，所以只需要一个简单的全连接层就可以解码这些`tokens`。（ViT最后需要的是图片特征，所以也只需要MLP就可以解码）
	- MAE中需要还原`the missing pixels`，这些像素是一个很基础的特征。要将自编码器的高级语义特征解码至低级语义层级，所以可能需要一个比较复杂的转置卷积网络才能完成解码。

&#8195;&#8195;正是基于以上分析，作者才提出本文的两个核心设计：==高掩码率以及非对称编-解码器==。


&#8195;&#8195;在导言最后一部分，作者说MAE可以只在`ImageNet 1K`上就能预训练`ViT-Large/-Huge`模型（得到很好的效果）。类比`ViT`，相同效果后者需要在近`100`倍规模的数据集上训练才能达到。
&#8195;&#8195;另外，在目标检测、语义分割和实例分割上，MAE比之前所有模型（包括有监督训练模型）效果都要好，而且加大模型会有显著的收益。

### 2.2 相关工作（都是老故事，略）
### 2.3 算法
#### 2.3.1 MAE主要结构
&#8195;&#8195;MAE使用编码器将观察到的信号映射到潜在表示，再使用一个解码器从潜在表示重构原始信号。与经典的自动编码器不同，我们采用了一种非对称设计，允许编码器仅对部分观察信号（无掩码标记）进行操作，模型设计图如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3550aabb5b54c94e1beb5a88e8d1dc69.png)
- 预训练时：
	-  切片：首先将图像切成一个个的patch（图像块） 
	- encoder：将没有被mask的块输入到encoder层进行处理。（类似`ViT`：）
		- 通过线性映射（linear projection）将每个Patch映射到一维向量中，得到Pacth embeddings
		- `Pacth embedding`+`position embedding`一起输入Transformer Encoder层得到其输出。
	- decoder：
		- 将序列拉长，填入灰色块像素对应的embedding（主要就是`position embeddings`信息）。
		- 将上面得到的长向量输入`decoder`，解码器会尝试将里面的像素信息全部重构回来，使之成为原始图片。
- 微调时：
	- 只需用到`endoder`，将图片切成`Patches`后编码得到特征，然后用来处理自己的任务就行。（此时不需要掩码，也就不需要解码器）

>&#8195;&#8195;说明：上图中编码器比解码器更宽，表示模型的主要计算部分在编码器（最重要的就是编码像素得到特征）

#### 2.3.2 具体实现
- **Masking**（随机均匀采样）
	- ==随机均匀采样==：在不替换的情况下，按照均匀分布对patches进行随机采样，采到的样本保留，剩下的全部mask掉。
	- 具体实现：将patches经过线性映射成embedding，加上位置编码之后得到的token embeddings序列。再将这个序列随机打乱（`shuffle`），然后取前25%完成采样，输入encoder层。（后面75%就是做mask处理）
	- 为何采用随机均匀抽样策略：一是避免潜在的“中心归纳偏好”(也就是避免 patch 的位置大多都分布在靠近图像中心的区域)，二是这种策略还造就了稀疏的编码器输入（只处理可见块），能够以更低的代价训练较大规模的 Encoder。
	- 这种掩码策略虽然简单，但其实很重要。因为其决定了预训练任务是否具有足够的挑战性，从而影响着 Encoder 学到的潜在特征表示 以及 Decoder 重建效果的质量。（后面有相关消融实验验证）
- **Encoder**
这部分也和`ViT`一样，不再赘述，唯一不同的是，被mask的块不进行编码。
- **解码器**
	- masked token：
		- 需要注意的是，==`masked tokens`不是通过mask 掉的 patch 经过 embedding 转换而来，而是通过一个共享的可学习向量来表示，也就是说每一个被mask的块都表示成同样的向量。==
		- 这个向量简单粗暴的复制n次之后每人一份。然后再加上对应的`position emebdding`进行区分。
	- 解码器仅在预训练期间用于重构图像，下游任务只需要编码器对图像进行编码就行，这样解码器可以独立于编码器，就可以设计的比编码器更简单（更窄、更浅）。最终相比编码器，其计算量只有其10%。
	- 通过结构图可以看到，encoder结构重载但是只处理`unmask tokens`；decoder处理所有tokens，但是结构轻量；通过这种非对称设计，使得整体结构十分高效，大大减少了预训练时间。
- **重构图像**
	- 解码后的所有 tokens 中取出 `masked tokens`(在最开始 mask 掉 patches 的时候可以先记录下这些 masked 部分的索引)
	- 用一个MLP，将其从向量维度映射到一个patch的像素。
	- 损失计算时，只对被mask的patches做计算（类似BERT），且使用MSE函数，也就是预测像素值和真实像素值的MSE。
	- 对要预测的像素，可以在每个patch内部做一次`normalization`（减均值除方差），这样数据更稳定。

#### 2.3.3 简单实现
将以上讲解的全部串联起来就是：
1. 将图像划分成 `patches`：(B,C,H,W)->(B,N,PxPxC)；
2. 对各个 patch 进行 `embedding`(实质是通过全连接层)，生成 tokens，并加入位置信息(position embeddings)：(B,N,PxPxC)->(B,N,dim)；
3. 根据预设的掩码比例，进行随机均匀采样。`unmask tokens` 输入 Encoder，另一部分“扔掉”(mask 掉)；
4.  编码后的 `tokens` 与 `masked tokens`（ 加入位置信息） 按照原先在 patch 形态时对应的次序拼在一起，然后喂给 Decoder 玩。
	- 整个序列是通过`unshuffle`操作，还原到原来的顺序。
	- 如果 Encoder 编码后的 token 的维度与 Decoder 要求的输入维度不一致，则需要先经过 linear projection 将维度映射到符合 Decoder 的要求
5. Decoder 解码后取出 masked tokens 对应的部分送入到全连接层，对 masked patches 的像素值进行预测，最后将预测结果(B,N',PxPxC)与 masked patches 进行比较，计算 MSE loss。

### 2.4 实验部分
**微调层数对比**
&#8195;&#8195;我们知道，预训练模型做微调的效果是比只做特征提取的效果更好的，但是训练时间会更长。所以作者试验了一下，微调不同的层数，模型的效果如何（见下图）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4101d1a56bb322caff631f42561bfbbd.png)
- 横坐标表示被微调的层数（这些层参数可以训练，剩下层被冻结）
- 可以看到，MAE基本只需要微调最后4层就可以了；因为最后面的层，和任务最相关。而前面的层是更底层一些的特征，在各种任务上都更适用。
#### 2.4.1 MAE超参数实验
**1. ImageNet实验**

下图是在 `ImageNet-1K`验证集上的top-1结果。（详见附录A1）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b939e5e5a040a2af38ca87cdfdc02cd5.png)
- `scratch，original`：`ViT-L/16`模型正常地从头训练，效果其实不是很稳定。（200epoch）
- `scratch，our impl.`：`ViT-L/16`加上比较强的正则（详见附录A2）
VIT原文中一直说需要很大的数据集才预训练出好的模型，但是后来大家发现，如果加入合适的正则项，在小一点的数据集上（ImageNet-1k）也能够训练出好的效果。 
- `baseline MAE`：在ImageNet上预训练，然后在`ImageNet-1K`上微调50epoch。

**2. decoder/mask策略/数据增强**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9dea96842bf6ab2a9feabaf41624c25c.png)

上图展示的是`MAE`使用`ViT-L/16`结构，在`ImageNet-1K`数据集上的acc精度。下面一个个来看：
>- `ft`： fine-tuning微调
>- `lin`：linear probing，只训练最后一个线性层的参数，相当于MAE只做特征提取
>- 最终`decoder`的默认深度是`8 blocks`，宽度是`512 dim`，与`VIT-L`相比，计算量只有前者的`9%`.。
- 表a：展示不同的decoder深度，可见堆叠8个transformer block效果最好。（其实对微调来说，都差不多）
- 表b：decoder宽度，每个token表示成一个512维的向量，效果最好。
- 表c：是否编码masked tokens。结果是不编码精度更高，且计算量更少（编码的Flops是不编码的3.3倍）
- 表d：重建目标对比.。
	- 第一行：`MAE`现行做法
	- 第二行：预测时对每个patch内部做`normalization`，效果最好。
	- 第三行：`PCA`降维
	- 第四行：`BEiT`的做法，通过vit把每一块映射到一个离散的token上面再做预测。
- 表e：数据增强。结果显示，做简单的随机大小裁剪，效果就不错
`fixed size`和` rand size`分别表示裁剪成固定大小，和裁剪成随机大小。`color jit` 是加入颜色变换，`none`就是不使用数据增强了。

- 表f：采样策略。随机采样方式最简单，效果也最好。
`random/block/grid`分别表示随机采样、按块采样（mask大块）、按网格采样。下面有进一步的说明图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/432a49cd4a3611b592f0f875758f0759.png)

**3. mask比例**
综合ft和lin考虑，`masking ratio=75%`时效果最好。（图就不放了）

#### 1.4.2 对比其它模型
**1. 自监督模型效果对比**

下表是在ImageNet-1K训练集上预训练后的微调结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c98169a0d4b88038ceabc80c912df475.png)
在COCO数据集上的效果对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/10067e54d7ef69b27ddd9e01299f6019.png)

over

### 2.5 结论&讨论
简单且拓展性好的算法，是整个深度学习的核心。
>- 简单：是指`MAE`在`ViT`模型上加了一些简单的扩展，但是模型本身是不简单的。
>- 拓展性好：是指你有钱就可以无限的加数据了，毕竟MAE不需要标号，你可以训练更大的模型。

&#8195;&#8195;自监督学习在最近几年是比较火的，但是在计算机视觉中，还是主要用有标号的数据来训练 。本文在`ImageNet`数据集上通过自编码器学习到可以媲美有标号训练的效果，使得CV领域的自监督学习可能走上与NLP类似的轨迹。
&#8195;&#8195;另一方面，**图像和语言是不同类型的信息，要谨慎处理这种差异**。
- 对于语言来讲，一个token是一个语义单元，它含有的语义信息比较多（这种信息在ViT里提取出来了）。
- 在图片中，虽然一个patch也含有一定的语义信息，但它不是语义的分割（即某个patch中并不含有特定的物体，可能含有多个物体的一小块，或者是某一个物体重叠的一块）。但即使是在这样的情况下，MAE也能做很复杂的一些任务。作者认为MAE（transformer）确实能够学到隐藏的比较好的语义表达。

Broader impacts：**MAE是个生成模型**，可以生成不存在的内容，类似GAN，所以使用要注意。

## 三、Swin-Transformer论文精读
>- 论文名称：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)、[官方开源代码](https://github.com/microsoft/Swin-Transformer)
>- 李沐[《Swin Transformer论文精读视频》](https://www.bilibili.com/video/BV13L4y1475U/?vd_source=21011151235423b801d3f3ae98b91e94)、[视频笔记](https://www.bilibili.com/read/cv14877004?spm_id_from=333.999.0.0)
>- 参考：太阳花的小绿豆帖子[《Swin-Transformer网络结构详解》](https://blog.csdn.net/qq_37541097/article/details/121119988)、[讲解视频](https://www.bilibili.com/video/BV1Jh411Y7WQ/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>- [Pytorch实现代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer)及[Swin Transformer API](https://pytorch.org/vision/stable/models/swin_transformer.html)、[Tensorflow2实现代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/swin_transformer)

### 3.1 导言
#### 3.1.1前言
&#8195;&#8195;`Swin transformer`是微软研究院于2021年3月25日发表在ICCV的一篇文章（ICCV 2021最佳论文），利用transformer架构处理计算机视觉任务，在图像分割，目标检测各个领域已经霸榜。比如打开其官网代码可以看到：（这排名不是卷死？）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/51846cd4f3bfbd86adbf15ea4ed30c52.png)
&#8195;&#8195;首先看论文题目。`Swin Transformer： Hierarchical Vision Transformer using Shifted Windows`。即：Swin Transformer是一个用了移动窗口的层级式Vision Transformer（Hierarchical是层级式的意思）。
&#8195;&#8195;所以Swin来自于 `Shifted Windows` ， 它能够使Vision Transformer像卷积神经网络一样，做层级式的特征提取，这样提取出来的特征具有多尺度的概念 ，这也是 Swin Transformer这篇论文的主要贡献。

#### 3.1.2 摘要
&#8195;&#8195;本文提出了一个新的 Vision Transformer 叫做 `Swin Transformer`，它可以被用来作为计算机视觉领域一个通用的骨干网络 。

标准的Transformer直接用到视觉领域有一些挑战，即：
1. 多尺度问题：比如一张图片里的各种物体尺度不统一，NLP中没有这个问题；
2. 分辨率太大：如果将图片的每一个像素值当作一个token直接输入Transformer，计算量太大，不利于在多种机器视觉任务中的应用。

基于这两点，本文提出了 hierarchical Transformer，通过移动窗口来学习特征。

- 移动窗口学习，即只在滑动窗口内部计算自注意力，所以称为`W-MSA`（Window Multi-Self-Attention）。
- `W-MSA`大大降低了降低了计算复杂度。同时通过`Shiting`（移动）的操作可以使相邻的两个窗口之间进行交互，也因此上下层之间有了cross-window connection，从而变相达到了全局建模的能力。 
- 分层结构使得模型能够灵活处理不同尺度的图片，并且计算复杂度与图像大小呈线性关系，这样模型就可以处理更大分辨率的图片（为作者后面提出的[Swin V2](https://paperswithcode.com/paper/swin-transformer-v2-scaling-up-capacity-and)铺平了道路）。

>- `Vision Transformer`：进行`MSA`（多头注意力）计算时，任何一个patch都要与其他所有的patch都进行attention计算，计算量与图片的大小成平方增长。
>- `Swin Transformer`：采用了`W-MSA`，只对window内部计算MSA，当图片大小增大时，计算量仅仅是呈线性增加。

&#8195;&#8195;在摘要后部分作者也说了，`Swin Transformer`能够提取多尺度特征之后，在多种视觉处理任务上都有很好的效果。比如`ImageNet-1K` 上准确度达到87.3%；在 COCO  mAP刷到58.7%（比之前最好的模型提高2.7）；在ADE上语义分割任务也刷到了53.5（提高了3.2个点 ）
>另外对于 MLP 的架构用 shift window 的方法也能提升，这一点在[MLP Mixer](https://paperswithcode.com/paper/mlp-mixer-an-all-mlp-architecture-for-vision)中就有体现。
#### 3.1.3 导言

在这部分作者对比了Vision Transformer和Swin Transformer的结构区别，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2ab4bfdd858fb8a8d816f83ed17fa80d.png)
可以看出主要区别有两个：
1.  <font color='deeppink'>层次化构建方法（Hierarchical feature maps） </font>：Swin Transformer使用了类似卷积神经网络中的层次化构建方法。
	- 对于计算机视觉的下游任务，尤其是密集预测型的任务（检测、分割），有多尺寸的特征至关重要的。（比如目标检测里的FPN、分割里面的UNet等等）
	- `Vision Transformer`中是一开始就直接下采样16倍，这样模型自始至终都是处理的16倍下采样率过后的特征，这样在处理需要多尺寸特征的任务时，效果不够好。
	- `Swin Transformer` 使用`patch merging`，可以把相邻的四个小的patch合成一个大的patch，提高了感受野，这样就能获取多尺度的特征（类似CNN中的池化效果）。这些特征通过FPN结构就可以做检测，通过UNet结构就可以做分割了。
2. <font color='deeppink'>使用W-MSA </font>，好处有两点：
	- `Swin Transformer`使用窗口（Window）的形式将特征图划分成了多个不相交的区域，并且只在每个窗口内进行多头注意力计算，大大减少计算量。
	- 获得了和CNN一样的归纳偏置特性——`locality`。（在本帖1.1.3中讲过）。
	归纳偏置：一种先验知识或者说提前的假设
`locality`：CNN是以滑动窗口的形式一点一点地在图片上进行卷积的，所以假设图片上相邻的区域会有相邻的特征，靠得越近的东西相关性越强
>- 在 `Swin Transformer`里，默认每个窗口有49个patch，第一层每个patch尺寸是4*4。
>- `locality`进一步说明：对于图片来说，语义相近的不同物体还是大概率会出现在相连的地方，所以即使是在一个小范围的窗口内计算自注意力也是差不多够用的，全局计算自注意力对于视觉任务来说，其实是有点浪费资源的。
>- `W-MSA`虽然减少了计算量，但也会隔绝不同窗口之间的信息传递。所以在论文中作者又提出了 `SW-MSA`的概念，通过此方法能够让信息在相邻的窗口中进行传递，后面会细讲。
### 3.2 相关工作（略）
### 3.3 算法
#### 3.3.1 模型结构

&#8195;&#8195;原论文中给出的关于Swin Transformer（Swin-T）网络的架构图如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d9e7f639805a69f503327f3b9d9cbd18.png)
1.  Patch Partition层：类似`ViT`一样将图片分割成一个个4*4大小的patch（`[224,224,3]—>[56,56,48]`）
2. `Linear Embeding`层：将每个像素的channel数调整为C，并对每个channel做一次Layer Norm。（`[56,56,48]—>[56,56,96]`）
>- 假设输入的是RGB三通道图片，那么每个patch就有4x4=16个像素，然后每个像素有R、G、B三个值，所以展平后是16x3=48。
>- swin-transformer有T、S、B、L等不同大小，其C的值也不同，比如`Swin-Tiny`中，`C=96`。
>- 在源码中`Patch Partition`和`Linear Embeding`就是直接通过一个卷积层实现的，和之前Vision Transformer中讲的 Embedding层结构一模一样。（kernel size=4×4，stride=4，num_kernel=48）

3. 将每`49`个patch划分为一个窗口，后续只在窗口内进行计算。
4. 通过四个`Stage`构建不同大小的特征图。其中后三个stage都是先通过一个`Patch Merging`层进行2倍的下采样。（`[56,56,96]—>[28,28,192]—>[14,14,384]—>[7,7,768]`）
5. 每个stage中，重复堆叠`Swin Transformer Block`偶数次（结构见上图右侧，分别使用`W-MSA`和`SW-MSA`，两个结构成对出现）。

6. 如果是分类任务，后面还会接上一个Layer Norm层、全局池化层以及全连接层得到最终输出。`[7,7,768]—>[1,768]—>[1,num_class]`（也就是做序列的全局平均，类似CNN的做法，而不是加上CLS做分类）
>- 如果不划分窗口，以`Swin-Tiny`举例，Linear Embeding层输出矩阵为`[56,56,96]`。如果计算全局注意力的话，输入序列长度为`56*56=3136`，每个元素是96维，这个序列就太长了。
>- 引入 `Shifted Windows`后，每个序列长度固定为`49`。
>- 与ViT还有一点不同的是：`ViT`在输入时会给embedding加上`1D-位置编码`。而Swin-T这里则是作为一个可选项（self.ape）。另外Swin-T在计算Attention时用的是`相对位置编码`。

&#8195;&#8195;看完整个前向过程之后，就会发现 `Swin Transformer` 有四个 `stage`，还有类似于池化的 `patch merging` 操作，自注意力还是在小窗口之内做的，以及最后还用的是全局平均池化 。所以可以说 `Swin Transformer`是披着Transformer皮的卷积神经网络，将二者进行了完美的结合。

&#8195;&#8195;接下来，在分别对`Patch Merging`、`W-MSA`、`SW-MSA`以及使用到的相对位置偏置（`relative position bias`）进行详解。

#### 3.3.2 Patch Merging
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2b724fe222fe7298742b8277e9ad0f5e.png)

&#8195;&#8195;上面讲了，在每个Stage中首先要通过一个`Patch Merging`层进行下采样（Stage1除外）。
如上图所示，假设输入的是一个**4x4大小的单通道特征图**（feature map），分割窗口大小为2×2，`Patch Merging`操作如下：
1. **分割**：Patch Merging会将每个2x2的相邻像素划分为一个patch，然后将每个patch中相同位置（同一颜色）像素给拼在一起，得到了4个feature map。
2. **拼接**：将这四个feature map在深度方向进行concat拼接
3. **归一化**：进行LayerNorm处理
4. **改变通道数**：最后通过一个全连接层对每个像素的channel进行改变，将feature map的深度由C变成C/2。

可以看出，通过`Patch Merging`层后，feature map的高、宽会减半，深度翻倍。

#### 3.3.3  W-MSA
&#8195;&#8195;W-MSA全称是`window Multi-heads Self-attention`。普通的Multi-heads Self-attention会对`feature map`中的每个像素（或称作token）都要计算`Self-Attention`。而W-MSA模块，首先将feature map按照MxM大小划分成一个个Windows，然后单独对每个Windows内部进行Self-Attention。

目的：减少计算量（下图可以看出两种方式分别是多少计算量）
缺点：窗口之间无法进行信息交互，等于减少了感受野，无法看到全局信息
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5841f46958b2a2123f6569ae86590c47.png)
>具体的计算过程可以参考：太阳花的小绿豆帖子[《Swin-Transformer网络结构详解》](https://blog.csdn.net/qq_37541097/article/details/121119988)第三章
#### 3.3.4 SW-MSA（Shifted Windows Multi-Head Self-Attention）
&#8195;&#8195;与W-MSA不同的地方在于这个模块存在窗口滑动，所以叫做shifted window。滑动距离是window_size//2，方向是向右和向下。
&#8195;&#8195;<font color='deeppink'>**滑动窗口是为了解决W-MSA计算attention时，窗口与窗口之间无法进行信息传递的问题**</font>。如下图所示，左侧是网络第L层使用的W-MSA模块，右侧是第L+1层使用SW-MSA模块。对比可以发现，窗口（Windows）发生了偏移。
>&#8195;&#8195;比如在L+1层特征图上，对于第一行第2列的2x4的窗口，它能够使第L层的第一排的两个窗口信息进行交流。再比如，第二行第二列的4x4的窗口，他能够使第L层的四个窗口信息进行交流，其他同理。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/651d40b67d6369bb1824eb4a824c5ffb.png)
&#8195;&#8195;但是这样也有一个问题：偏移后，窗口数由原来的4个变成了9个，后面又要对每个窗口内部进行MSA，非常麻烦。为此，作者又提出了一种更加高效的计算方法——`masked MSA`（后面会讲）。

#### 3.3.5 Masked MSA（技术细节，选看）
&#8195;&#8195;下图左侧是刚刚通过偏移窗口后得到的新窗口，右侧是为了方便大家理解，对每个窗口加上了一个标识。
- 0对应的窗口标记为区域A
- 3和6对应的窗口标记为区域B
- 1和2对应的窗口标记为区域C
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/46c987cac3f4322b4e15f39a257867a3.png)
1. 将区域A和C移到最下方
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9316dbd4ebacd92b5134711d17335297.png)
2. 再将区域A和B移至最右侧
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9408358688e7b5e5bb720f3ed1c9b0a7.png)
3. 分成4个4x4的窗口
如下图所示，移动完后，4是一个单独的窗口；5和3合并成一个窗口；7和1合并成一个窗口；8、6、2和0合并成一个窗口。这样又和原来一样是4个4x4的窗口了，所以能够保证计算量是一样的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/737135e6380c6fe8e322ef3504ce6c00.png)
4. Masked MSA计算。
上图这样划分之后，还有个问题，就是合并的窗口本身是不相邻的，比如5和3，如果直接进行MSA计算，是有问题的。我们希望是能够单独计算区域3和区域5各自区域内的MSA，论文中给的方法是Masked MSA，来隔绝不同区域的信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6e527395c845f4afbde74f946cdf3595.png)
&#8195;&#8195;如上图，区域5和区域3为例。对于该窗口内的每一个像素，都要先生成对应的query，key和value，然后与每一个像素进行attention计算。设$\alpha _{0,0}$代表 $q^0$与像素0对应的$k^0$进行匹配得到的attention score，同理可以得到$\alpha _{0,1}$至$\alpha _{0,15}$。
&#8195;&#8195;接下来就是SoftMax操作，但是对于Masked MSA，不同区域之间的像素计算的attention score会减去100（attention score∈[0,1]，减去100是一个很大的负数，经过softmax之后基本等于0了，后续加权求和结果也是0），这样等于不同区域之间的像素计算的attention被置为0，被屏蔽了。最终像素0计算的attention结果，还是和本区域5各个像素计算之后的结果。
&#8195;&#8195;**注意：在计算完后还要把数据给挪回到原来的位置上（例如上述的A，B，C区域）**

&#8195;&#8195;`Efficient batch computation for shifted configuration`简单说，就是通过移动合并，将9个窗口还是变为原来的4个窗口，再进行Masked MSA计算。计算量和普通的MSA一样，只是多了个mask。
&#8195;&#8195;Masked MSA是为了解决合并窗口中各个窗口应该单独计算的问题。使用Masked MSA，将不同窗口元素计算的attention score-100，等价于屏蔽掉不同窗口元素的attention结果。最终达到了4个窗口同时进行MSA计算，又保证得到只在在窗口内进行计算的效果。
### 3.4 实验
#### 3.4.1 ImageNet
结果见下图左侧：
- 表A：在`ImageNet-1K`上预训练然后在`ImageNet-1K`上微调的结果（测试集）
- 表B：在`ImageNet-22K`上预训练然后在`ImageNet-1K`上微调的结果（测试集）
- `ViT`在小规模数据集上效果并不好，在足够大的数据集上效果明显提升，和之前论文中的结论是一样的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8dc9bb7bc721a1ba337fb11339068d07.png)
#### 3.4.2 COCO数据集
结果见上图右侧：
- 表A：不同的目标检测模型，分别使用`ResNet-50` 和`Swin-Tiny`作为backbone的结果对比。（二者参数量和计算量相近）
- 表B：使用 `Mask R-CNN`做检测模型，然后替换不同的backbone。
- 表C：不做模型限制进行比较。

#### 3.4.3 语义分割数据集ADE20K
效果好

#### 3.4.4移动窗口和相对位置编码
&#8195;&#8195;论文中还提到，使用相对位置偏置后给够带来明显的提升。（下图公式中的B就是偏置，ADE20k是图片分割数据集）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/be1875fac507e2a8926d7469a2b7be2a.png)
- 上图蓝色框表示是否使用移动窗口（SW-MSA）的效果对比，可见比起分类任务，检测和分割任务差别更明显。
- 红色框表示使用相对位置偏置之后效果更好。
- 可见密集型预测任务，需要特征对位置信息更敏感，所以更需要周围的上下文关系。用了移动窗口和相对位置编码以后 ，效果提升更多。

>&#8195;&#8195;下面具体讲解什么是`Relative Position Bias`（假设特征图大小为2×2）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/43e5869948120efe5666787d0a5422bb.png)
>1. **计算相对位置索引**：比如蓝色像素，在蓝色像素使用q与所有像素k进行匹配过程中，是以蓝色像素为参考点。然后用蓝色像素的绝对位置索引与其他位置索引进行相减，就得到其他位置相对蓝色像素的相对位置索引，同理可以得到其他位置相对蓝色像素的相对位置索引矩阵（第一排四个位置矩阵）。
>2. **展平拼接**：将每个相对位置索引矩阵按行展平，并拼接在一起可以得到第二排的这个4x4矩阵。
>3. **索引转换为一维**：在源码中作者为了方便把二维索引给转成了一维索引。
	- 首先在原始的相对位置索引上加上M-1(M为窗口的大小，在本示例中M=2)，加上之后索引中就不会有负数了。
	- 接着将所有的行标都乘上2M-1
	- 最后将行标和列标进行相加。这样即保证了相对位置关系，而且不会出现直接相加后位置重叠的问题。（0+-1和-1+0结果都一样，但其实其位置不一样）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1a0eef73ca163da2ae229b1f6002d0ff.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27d1a7a7bf68cf78c1245b9db0810ff1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ff9651b75c5071bf58b82a5416b28c59.png)
>4. **取出相对位置偏置参数**。真正使用到的可训练参数B 是保存在relative position bias table表里的，其长度是等于 $(2M-1) \times (2M-1)$。相对位置偏置参数B，是根据相对位置索引来查relative position bias table表得到的，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/38fa029a28aaae988c7ef26c51ca2d76.png)
>&#8195;&#8195;为啥表长是 $(2M-1) \times (2M-1)$？考虑两个极端位置，（0,0）能取到的相对位置极值为（-1，-1），（-1，-1）能取到的极值是（1，1），即行和列都能取到（2M-1）个数。考虑到所有的排列组合，表的长度就是$(2M-1) \times (2M-1)$
#### 3.4.5  模型详细配置参数
模型结构和参数如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/557bb65925d269fcc3922634b0b5535f.png)

以`Swin-T`举例:
- 模型一开始的Patch Partition和Linear Embeding层，其效果和Patch Merging层一样，都是进行下采样和调整channel。 `concat4×4，96d，LN`表示宽高都下采样4倍，调整后channel数为96，再经过一个Layer Norm层。
- 堆叠两个`Swin Transformer Block`，其中：
	- `win. sz. 7x7`表示使用的窗口的大小（window size）为7×7
	- `dim`表示这个层输出的feature map的channel深度（或者说token的向量长度）
	- `head`表示多头注意力模块中head的个数
- 后面以此类推，堆叠2、6、2个block。堆叠时是交替使用`W-MSA`和`SW-MSA`。

### 3.5 结论
&#8195;&#8195;本文最关键的一个贡献就是基于 `Shifted Window` 的自注意力，它对很多视觉的任务，尤其是对下游密集预测型的任务是非常有帮助的 。所以未来工作的重点就是把`Shifted Window` 用到NLP任务中，推进模型的大一统（多模态）。
### 3.6 代码讲解
>- 太阳花的小绿豆：[代码地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)、[代码讲解视频](https://www.bilibili.com/video/BV1yg411K7Yc/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>- [《Swin Transformer 论文详解及程序解读》](https://zhuanlan.zhihu.com/p/401661320)


