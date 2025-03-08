@[toc]

## 一、Vision Transformer
参考我的另一篇博文[《李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer》](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
## 二、Swin-Transformer
参考我的另一篇博文[《李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer》](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
## 三、MobileViT
>论文名称：[MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
>参考小绿豆的博文[《MobileViT模型简介》](https://blog.csdn.net/qq_37541097/article/details/126715733)
>[官方源码（Pytorch实现）](https://github.com/apple/ml-cvnets)、[小绿豆的项目代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/MobileViT)

### 3.1 为什么引入CNN与Transformer的混合架构
&#8195;&#8195;自从2020年ViT(Vision Transformer)模型的横空出世，人们发现了Transformer架构在视觉领域的巨大潜力，视觉领域的各项任务也不断被Transformer架构模型刷新。同时其缺点也很明显，模型参数太大（比如ViT Large Patch16模型光权重就有1个多G），算力要求太高，这基本就给移动端部署Transformer模型判了死刑。
&#8195;&#8195;Apple公司在2021年发表的一篇CNN与Transfomrer的混合架构模型——MobileViT：CNN的轻量和高效+Transformer的自注意力机制和全局视野。
&#8195;&#8195;**纯Transformer架构除了模型太重，还有一些其他的问题**，比如：
1. Transformer缺少空间归纳偏置(spatial inductive biases)：
	- self-attention本身计算时是不考虑次序的，计算某个token的attention如果将其他token的顺序打乱对最终结果没有任何影响。
	- 为了解决这个问题，常见的方法是加上位置偏置(position bias)/位置编码，比如Vision Transformer中使用的绝对位置偏置，Swin Transformer中的相对位置偏置。
2. <font color='deeppink'>Transformer模型迁移到其他任务(输入图像分辨率发生改变)时比较繁琐（相对CNN而言），主要原因是引入位置偏置导致的。
	- Vision Transformer的绝对位置偏置的序列长度是固定的，等于$\frac{H \times W}{16 \times 16}$。其中H、W代表输入图片的高和宽，所以只要改变输入图像的尺度就无法直接复用了。
	- 最常见的处理方法是通过插值的方式将位置偏置插值到对应图像的序列长度。但如果不对网络进行微调直接使用实际效果可能会掉点，如果每次改变输入图像尺度都要重新对位置偏置进行插值和微调，那也太麻烦了。
	>比如在Imagenet上预训练好的网络（224x224），直接对位置偏置进行插值不去微调，在384x384的尺度上进行验证可能会掉点(CNN一般会涨点)。
	- Swin Transformer相对位置偏置的序列长度只和Windows大小有关，与输入图像尺度无关。但在实际使用中，一般输入图像尺度越大，Windows的尺度也会设置的大些。只要Windows尺度发生变化，相对位置偏置也要进行插值了。所以在Swin Transformer v2中就对v1的相对位置偏置进行了优化。
3. <font color='deeppink'>Transformer相比CNN要更难训练
&#8195;&#8195;比如Transformer需要更多的训练数据，需要迭代更多的epoch，需要更大的正则项(L2正则)，需要更多的数据增强，且;对数据增强很敏感 。</font>
&#8195;&#8195;比如在MobileViT论文的引言中提到，如果将CutMix以及DeIT-style的数据增强移除，模型在Imagenet上的Acc直接掉6个点。

针对以上问题，现有的、最简单的方式就是采用CNN与Transformer的混合架构。
- CNN能够提供空间归纳偏置所以可以摆脱位置偏置
- 加入CNN后能够加速网络的收敛，使网络训练过程更加的稳定。
### 3.2 性能对比
1. MobileViT与主流的一些Transformer模型对比
通过下图可以看出，即使使用普通的数据增强方式，在ImageNet 1K验证集上，MobileViT也能达到更高的Acc并且参数数量更少。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0759d950d84fff7f2eb5da4a74535b7c.png)
- basic：ResNet -style数据增强
- advanced：basic数据增强+RandAugmentation+CutMix
2. MobileViT与一些传统的轻量级CNN进行了对比
如下图所示，在近似的参数数量下MobileViT的Acc要更高。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/de9af08ceb2ac8ea01bf1446d77b4253.png)
3. 推理速度对比
参数量不等于推理书速度。下表中，通过对比能够看到基于`Tranaformer`的模型(无论是否为混合架构)推理速度比纯CNN的模型还是要慢很多的(移动端)。作者在论文中给出解释主要还是说当前移动端对Transformer架构优化的还太少。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/91b10b228072612324461d69fd2db8ee.png)
### 3.3 模型结构
**Vision Transformer结构**
&#8195;&#8195;在讲MobileViT网络之前先简单回顾下Vision Transformer的网络结构。下图是`MobileViT`论文中绘制的`Standard visual Transformer`。
- 首先将输入的图片划分成一个个Patch，然后通过线性变化将每个Patch映射到一个一维向量中（视为一个个Token）
- 接着加上位置偏置信息（可学习参数），通过一系列`Transformer Block`得到输出
- 最后通过一个全连接层得到最终预测输出。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f072e0f6514ff5e236b9727c5b3e046a.png)

**MobileViT结构**
&#8195;&#8195;下图对应的是论文中的图1(b)，可以看到`MobileViT`主要由普通卷积，`MV2`，`MobileViT block`，全局池化以及全连接层共同组成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ccd1643c6eaf0cff1107b1b3dcffe552.png)
- `MV2`即`MobiletNetV2`中的`Inverted Residual block`，在本文4.3.1 `Inverted residual block`中有详细讲解。
- 上图中标有向下箭头的`MV2`结构代表stride=2的情况，即需要进行下采样。
- 下图是当stride=1时的`MV2`结构，有shortcut连接（输入输出size相同）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d17218a897b2c00bee87f97e845de2e3.png)
###  3.4 MobileViT block
MobileViT block的大致结构如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5176b65a1071c9b192df2fea4507a728.png)
- **局部的特征建模**：将特征图通过一个卷积核大小为`nxn`（代码中是3x3）的卷积层来完成
- **调整通道数**：通过1×1的卷积完成 
- **全局的特征建模**：通过`Unfold` -> `Transformer` -> `Fold`结构来完成
- **通道数调整回原始大小**：通过1×1的卷积完成 
- **`shortcut`**：捷径分支与原始输入特征图沿通道方向拼接进行Concat拼接
- **最终输出**：最后再通过一个卷积核大小为`nxn`（代码中是3x3）的卷积层做特征融合得到输出。

&#8195;&#8195;为了方便我们将`Unfold -> Transformer -> Fold`简写成`Global representations`，下面对此进行介绍。
1. 首先对特征图划分Patch（这里为了方便忽略通道channels），图中的Patch大小为2x2
2. 进行Self-Attention计算的时候，每个Token（图中的每个小颜色块），只和自己颜色相同的Token进行Attention，减少计算量。
>- 计算量降为原来的1/4。原始的Self-Attention计算，每个Token是需要和所有的Token进行Attention.
>- 为什么能这么做?
>1. 图像数据本身就存在大量的数据冗余，比如对于较浅层的特征图（`H, W`下采样倍率较低时），相邻像素间信息可能没有太大差异，如果每个Token做Attention的时候都要去看下相邻的这些像素，有些浪费算力。在分辨率较高的特征图上，增加的计算成本远大于Accuracy上的收益。
>2. 前面已经通过nxn的卷积层进行局部建模了，进行全局建模时就没必要再看这么细了
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c0f30fe993f0a2b7c6e34212cfd8c53d.png)
3. `Unfold/Fold`：只是为了将数据reshape成计算Self-Attention时所需的数据格式。
	- 普通的Self-Attention计算前，一般是直接展平H, W两个维度得到一个Token序列，即将`[N, H, W, C] -> [N, H*W, C]`其中N表示Batch维度
	- `MobileViT block`的`Self-Attention`计算中，只是将颜色相同的Token进行了Attention，所以不能简单粗暴的展平H, W维度
	- Unfold：将相同颜色的Token展平在一个序列中，以便使用普通的Self-Attention来进行
	- Fold：计算完后将其折叠回原特征图
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0cc8d7acba87848efebb56cf39b1e699.png)
### 3.5 Patch Size对性能的影响
&#8195;&#8195;大的patch_size能够提升网络推理速度，但是会丢失一些细节信息。论文中给的图8，展示了两组不同的patch_size组合，在图像分类，目标检测以及语义分割三个任务中的性能。
下采样倍率为8，16，32的特征图所采用的patch_size大小分别为：
- 配置A中，下采样倍率为8，16，32的特征图所采用的patch_size=[2, 2, 2]
- 配置B中，下采样倍率为8，16，32的特征图所采用的patch_size=[8, 4, 2]
- 对比发现，在图像分类和目标检测任务中（对语义细节要求不高的场景），配置A和配置B在Acc和mAP上没太大区别，只是配置B更快
- 但在语义分割任务中（对语义细节要求较高的场景）配置A的效果要更好。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/86cf13f9f57ce2bdab972761be24424d.png)
### 3.6 模型详细配置
&#8195;&#8195;在论文中，关于MobileViT作者提出了三种不同的配置，分别是：MobileViT-S(small)，MobileViT-XS(extra small)和MobileViT-XXS(extra extra small)，三者的主要区别在于特征图的通道数不同。下图中的标出的Layer1~5，这里是根据源码中的部分配置信息划分的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a8140988d37246764d3fe149040eb60b.png)
MobileViT-XXS，Layer1~5的详细配置信息如下：

```python
layer	out_channels	mv2_exp	transformer_channels	ffn_dim	  patch_h	patch_w	  num_heads
layer1		16				2			None				None	None	None	    None
layer2		24				2			None				None	None	None	    None
layer3		48				2			64					128		2		2			4
layer4		64				2			80					160		2		2			4
layer5		80				2			96					192		2		2			4
```

MobileViT-XS，Layer1~5的详细配置信息如下：

```python
layer	out_channels	mv2_exp	transformer_channels	ffn_dim	  patch_h 	patch_w 	num_heads
layer1		32				4			None				None	None	None			None
layer2		48				4			None				None	None	None			None
layer3		64				4			96					192		2		2				4
layer4		80				4			120					240		2		2				4
layer5		96				4			144					288		2		2				4
```
对于MobileViT-S，Layer1~5的详细配置信息如下：

```python
layer	out_channels	mv2_exp	transformer_channels	ffn_dim	 patch_h	patch_w	num_heads
layer1		32				4			None			 None	  None		  None	  None
layer2		64				4			None			 None	  None		  None	  None
layer3		96				4			144				 288	  2			  2		  4
layer4		128				4			192				 384	  2			  2		  4
layer5		160				4			240				 480	  2			  2		  4
```
其中：

- out_channels：表示该模块输出的通道数
- mv2_exp：表示Inverted Residual Block中的expansion ratio（倍率因子）
- transformer_channels：表示Transformer模块输入Token的序列长度（特征图通道数）
- num_heads：表示多头自注意力机制中的head数
- ffn_dim：表示FFN中间层Token的序列长度
- patch_h/patch_w：表示每个patch的高度/宽度

## 四、MobileNet系列模型
>- 参考博文[《MobileNet(v1、v2)网络详解与模型的搭建》](https://blog.csdn.net/qq_37541097/article/details/105771329)、[《轻量级神经网络MobileNet全家桶详解》](https://blog.csdn.net/qq_37555071/article/details/108393809?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166413382916782388096887%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166413382916782388096887&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-108393809-null-null.142%5Ev50%5Econtrol,201%5Ev3%5Econtrol_1&utm_term=mobilenet&spm=1018.2226.3001.4187)
>- 模型讲解视频 [《MobileNet（v1，v2）网络详解视频》](https://www.bilibili.com/video/BV1yE411p7L7/?vd_source=21011151235423b801d3f3ae98b91e94)、[《MobileNetv3网络详解》](https://www.bilibili.com/video/BV1GK4y1p7uE/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>- [github代码地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)、代码讲解[《使用pytorch搭建MobileNetV2》](https://www.bilibili.com/video/BV1qE411T7qZ/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[《使用Pytorch搭建MobileNetV3》](https://www.bilibili.com/video/BV1zT4y1P7pd/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、PyTorch API

### 4.1 前言
&#8195;&#8195;传统的CNN网络已经普遍应用在计算机视觉领域，并且已经取得了不错的效果。但是发展到现在，模型深度越来越深，模型越来越复杂，预测和训练需要的硬件资源也逐步增多，导致无法在移动设备以及嵌入式设备上运行。
&#8195;&#8195;深度学习领域内也在努力促使神经网络向小型化发展。在保证模型准确率的同时体积更小，速度更快。2016年直至现在，业内提出了SqueezeNet、ShuffleNet、NasNet、MnasNet以及MobileNet等轻量级网络模型，这些模型使移动终端、嵌入式设备运行神经网络模型成为可能。
&#8195;&#8195;MobileNet在轻量级神经网络中较具代表性，特别是谷歌在2019年5月份推出了最新的MobileNetV3。在ImageNet数据集上，最新的MobileNetV3-Large的Top1准确率达到75.2%。本章将对MobileNet进行详细解析。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3baba0ff95a692c6e55688d8f067775e.png)
MobileNet系列模型性能对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8783a86397cd9bd27a5ff53c74908580.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3e2f83fa2a897898e11d40a6d34c8fd2.png)


MobileNet可以在移动终端实现众多的应用，包括目标检测，目标分类，人脸属性识别和人脸识别等。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/62d66c3d344920ee78dff2ae4f941489.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8e5335decea6ae90f3b3cbe5492f4f29.png)

### 4.2 MobileNetV1
&#8195;&#8195;MobileNetV1网络类似于VGG，各个block串行连接，只不过在传统卷积层基础上加入了Depthwise Separable Convolution（深度可分卷积）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/07527cb5d565ecf3e62c5ad641c707ff.png)
&#8195;&#8195;为了适配更定制化的场景，MobileNet引入了两个超参数： 宽度因子width multiplier和分辨率因子resolution multiplier，分别记为α和β：
- α：控制卷积层卷积核个数，按比例减少/增加特征数据通道数大小，调整计算量
- β：控制输入图像大小，仅仅影响计算量，但是不改变参数量

#### 4.2.1 深度可分离卷积Depthwise separable convolution
&#8195;&#8195;MobileNet网络基本单元是<font color='deeppink'>深度可分离卷积（depthwise separable convolution） </font>，可以大大减少运算量和参数数量，是其一大亮点。
1. 传统卷积：每个卷积核的channel与输入特征矩阵的channel相等，即每个卷积核都会与输入特征矩阵的每一个维度进行卷积运算
2. <font color='deeppink'>DW卷积 （depthwise convolution）</font>：每个卷积核的channel都是等于1的，即每个卷积核只负责输入特征矩阵的一个channel。所以有$num_{kernel}=channel_{in}=channel_{out}$。（卷积核的个数=输入矩阵的通道数=输出矩阵的通道数）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ae8513db0b9ed96edc5c2e08eed2de51.png)



3.  <font color='deeppink'>深度可分卷积 </font>：DW卷积+PW卷积（pointwise convolution）。
刚刚说了使用DW卷积后$num_{kernel}=channel_{in}=channel_{out}$，如果想改变/自定义输出特征矩阵的channel，那只需要在DW卷积后接上一个PW卷积即可（普通的1×1卷积），通常DW卷积和PW卷积是放在一起使用的，合称深度可分卷积Depthwise Separable Convolution。

**计算量对比**

&#8195;&#8195;深度可分卷积与传统的卷积相比有到底能节省多少计算量呢，下图对比了这两个卷积方式的计算量。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/53c968072514cb75dfe789df81cebef9.png)
&#8195;&#8195;卷积计算量≈卷积核宽高×输入channel×卷积核个数×输出矩阵宽高。假设stride=1，输入输出特征矩阵大小一样，宽高都是Df。则有：
- Df是输入特征矩阵的宽高（这里假设宽和高相等）
- Dk是卷积核的大小
- M是输入特征矩阵的channel，N是输出特征矩阵的channel
- 普通卷积计算量≈$D_{k}\times D_{k}\times M \times N \times D_{f} \times D_{f}$
- DW计算量是≈$D_{k}\times D_{k}\times M \times D_{f} \times D_{f}$，PW计算量≈$M \times N \times D_{f} \times D_{f}$
- mobilenet网络中DW卷积都是是使用3x3大小的卷积核。所以理论上普通卷积计算量是DW+PW卷积的8到9倍。

#### 4.2.2 MobileNetV1网络结构
MobileNetV1网络结构如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8f33cf582d12f252338ad48d5983130e.png)

- Conv：普通卷积
- Conv dw：DW卷积
- s：步距
- Filter Shape：卷积核尺寸。比如：
	- 第一层3×3×3×32表示卷积核大小是3×3，输入输出channel分别是3和32。
	- 第二层3×3×32 dw，因为卷积核channel=1，所以只有卷积核size和输出矩阵channel

右边表格是对比MobileNetV1和其它网络，以及使用不同的α、β参数时的准确率、计算量和参数量。其中：
- α=0.75，表示卷积核个数缩减为原来的0.75倍。
- β表示不同的输入图像尺寸
### 4.3 MobileNet v2
&#8195;&#8195;在MobileNet v1的网络结构表中能够发现，网络的结构就像VGG一样是个直筒型的，没有残差结构。而且网络中的DW卷积很容易训练废掉，效果并没有那么理想。MobileNet v2网络是由google团队在2018年提出的，相比MobileNet V1网络，准确率更高，模型更小。
&#8195;&#8195;<font color='red'>如果说MobileNet v1网络中的亮点是DW卷积，那么在MobileNet v2中的亮点就是Inverted residual block（倒残差结构）</font>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c35f7dfe738a69b91e6a397eb5237d32.png)
#### 4.3.1 Inverted residual block
1. Inverted residual block结构如下图所示：
- 左侧是ResNet网络中的残差结构，采用的是1x1卷积降维->3x3卷积->1x1卷积升维（两头大中间小的瓶颈结构），使用的是ReLU激活函数。
- 右侧就是MobileNet v2中的倒残差结构。与上面相反，采用的是1x1卷积升维->3x3DW卷积->1x1卷积降维。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c06a50c4d37ff3ec455872f0c4ca8308.png)
2. 为什么要使用倒残差结构？
原文的解释是<font color='red'>高维信息通过ReLU激活函数后丢失的信息更少</font>。下图所示，作者做了一系列实验：
- 假设输入是2D矩阵（二维），channel=1
- 使用不同matrix （矩阵T）将其变换到高维空间，再使用RELU激活函数得到输出值
- 使用T的逆矩阵$T^{-1}$，将刚刚的输出还原为2D矩阵
	-  output/dim=2表示矩阵T的维度为2。可以看到还原回2D后，丢失了很多信息
	- 随着T的维度不断加深，比如output/dim=30，丢失的信息越来越少
- 结论：<font color='deeppink'>ReLU激活函数对低维信息造成大量损失，但是对高维信息，造成的损失较小 </font>
3.  Inverted residual block激活函数设计：
**中间使用ReLU6激活函数，最后一个1x1的卷积层使用的是线性激活函数**（最后一层输入是降维后的，像刚刚讲的，不能再使用ReLU激活函数）。所以也叫`Linear Bottleneck`线性瓶颈结构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/966e288f550226235cff5924d20db30f.png)
RELU6激活函数公式为：
$$y=RELU6(x)=min(max(x,0),6)$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/37f70afad03f5fc49ad5335c125f87b2.png)
&#8195;&#8195;可以看出，对比RELU函数，唯一的改变就是将最大值截断为6。
4. Inverted residual block参数：
- 第一层1x1卷积升维（BN+RELU6）：输入图片高宽为h和w，channel=k。 t是升维扩展因子，即卷积核个数 
- 第二层3×3的dw卷积（BN+RELU6）：输入channel=输出channel，步距=s
- 第三层1x1卷积降维（BN）：卷积核个数为$k^`$
- 注意：并不是每个倒残差结构都有shortcut。
只有当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接，比如下图stride=2 的结构就没有shortcut。因为只有当shape相同时，两个矩阵才能做加法运算。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d9a6150b1987456ed4b997fa460c364.png)
#### 4.3.2 MobileNet v2网络结构
&#8195;&#8195;`MobileNetV2`网络模型中有共有17个`Bottleneck`层（每个Bottleneck包含两个pw卷积层和一个dw卷积层），一个标准卷积层（conv），两个pw conv组成，共计有54层可训练参数层。
&#8195;&#8195;MobileNetV2中使用线性瓶颈和Inverted Residuals结构优化了网络，使得网络层次更深了，但是模型体积更小，速度更快了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0b2d321525a21a8d60406a9202c11e3c.png)
- t：上一节讲的倒残差结构第一层的扩展倍率，将输入矩阵channel扩展t倍。
- c：输出特征矩阵的channel，即上一节表中的$k^`$
- n：每个Operater模块中倒残差结构重复的次数
- s：每个模块第一层倒残差结构的步距，后面层的步距都是1（也就是每个模块只有第一层会改变channel）
- k：代表分类的类别数

&#8195;&#8195;按照上一节讲的，只有当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接。所以上图红色框选的模块：
- 第一层：虽然s=1，但是输入输出channel分别是64和96，所以没有shortcut
- 后两层：s=1，且输入输出channel都是96，size都是14×14，所以有shortcut
#### 4.3.3  MobileNet v2性能对比
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1cbcee9674223ec7bcf9fbad58b3108f.png)
- 第一张表中MobileNet v2 （1.4）表示倍率因子α=1.4，也就是卷积核个数增加为原来的1.4倍。
- 第二张表中，是将MobileNet和SSD联合使用进行目标检测。在CPU（自家安卓手机）上200ms检测一张图片，已经不错了。

MobileNet的提出，基本实现了在移动设备和嵌入式设备上跑深度学习模型了。 
### 4.4 MobileNetv3
>[github项目地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

`MobileNetv3`是Google在2019年提出的，主要有三点比较重要：
- 更新block，论文中称为bneck：引入Squeeze-and-excitation（SE）模块和 h-swish（HS）激活函数以提高模型精度
- 使用NAS（神经架构搜索）搜索参数
- 重新设计耗时结构：作者对搜索后的网络结构每一层进行耗时分析，针对耗时多的结构进行了进一步优化
	- 第一层普通卷积层卷积核个数由32降为16，耗时减少2ms
	- 最后一个stage去除3x3 Dconv，1x1Conv等卷积层，耗时减少7ms

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/197255b7b059496664bcc8b62453c43c.png)
**性能对比：**

&#8195;&#8195;在ImageNet数据集上，`V3-Large 1.0`和`V2 1.0`的Acc分别是75.2%和72.0%,前者推理速度快了20%。（MAdds是计算量，P-1、P-2、P-3是不同手机上的推理速度。这里的1.0是上面讲的宽度因子α）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/700721c7bf5da708cb5318af76e9279e.png)
#### 4.4.1 SE模块
&#8195;&#8195;SE模块首次提出是在2017年的`Squeeze-and-Excitation Networks(SENet)`网络结构中，在`MobileNetV3`中进行改进并大量使用。
&#8195;&#8195;研究人员期望通过**精确的建模卷积特征各个通道之间的作用关系来改善网络模型的表达能力**。为了达到这个期望，提出了一种能够**让网络模型对特征进行校准的机制，使得有效的权重大，无效或效果小的权重小的效果，这就是SE模块**。
&#8195;&#8195;`MobileNetV3`的SE模块由一个全局平均池化层，两个全连接层组成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/21848dd9410d913fc6136032db5ca2a7.png)

- `Squeeze`：将输入矩阵的每个channel都进行池化处理得到一维向量。向量维度=输入矩阵channel
- `Excitation`：再经过两个全连接层得到每个channel的权重。
这两步可以理解为对输入矩阵的每个channel分析出一个权重关系，重要的channel会赋予更大的权重。
	- 第一个全连接层节点个数=1/4*输入矩阵channel，激活函数为RELU
	- 第二个全连接层节点个数=输入矩阵channel，激活函数为H-sig，也就是Hard-sigmoid
- `scale`：将SE模块输入矩阵与通道权重相乘，得到最终输出

下面举一个简单的例子说明：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b181bcfa73ebdfde97134aa278f5fe89.png)
&#8195;&#8195;如上图所示，输入矩阵有两个channel（黄色和蓝色），池化后一维向量有两个元素[0.25,0.3]。将其经过两个Linear层处理得到两个channel的权重[0.5,0.6]。将这个权重每个元素分别和输入矩阵各个channel相乘，得到SE模块的最终输出。

#### 4.4.2 bneck
`MobileNetV3`更新了block，论文中称为bneck，其改动如下：
- 上部分是V2的倒残差结构，详情见上一节的4.3.1 Inverted residual block
- 下部分是V3的block结构
	- 加入SE模块（乘上channel权重，类似注意力机制）
	- 更新了激活函数，在下一节会详细将。图中NL表示非线性激活函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e804ccd9e08c3f1e9bfd170ab9660b11.png)
- ResNet中添加SE模块形成SE-ResNet网络，SE模块是在bottleneck结构之后加入的
- MobileNetV3版本中SE模块加在了bottleneck结构的内部，在dw卷积后增加SE块，scale操作后再做pw卷积，SERadio=0.25（模块中第一个全连接层的节点个数，是输入特征矩阵channels的1/4）。
- 使用SE模块后，MobileNetV3的参数量相比MobileNetV2多了约2M，但是MobileNetV3的精度得到了很大的提升。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d041159dc906e58f3164ef5849d7ee24.png)
#### 4.4.3 swish/sigmoid替换为h-swish/h-sigmoid
&#8195;&#8195;在V2中，`Inverted residual block`基本都是使用的`RELU6`激活函数。但在V3中，对其进行改进。如下图所示：
1. 激活函数`swish`：可以提高Acc，但是求导、计算复杂，而且量化过程不友好（移动端一般都会对计算过程量化）
$$swish(x)=x*sigmoid(x)$$
2. `h-sigmoid`激活函数：下图可见，其曲线接近于sigmoid函数。
3. `h-swish`激活函数：将公式中的sigmoid激活函数用h-sigmoid激活函数替代，则
$$h-swish(x)=x*\frac{RELU6(x+3)}{6}$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7680de2343dc793c7ec336d12eb7fb38.png)
&#8195;&#8195;原论文中，作者说<font color='deeppink'>将`swish`替换为`h-swish`，`sigmoid`替换为`h-sigmoid`之后，推理速度得到提升，而且对量化过程比较友好。</font>

#### 4.4.4 重新设计耗时层结构
对耗时层结构改进了两点：
- 第一个卷积层卷积核个数由32降为16，Acc不变，耗时减少2ms（可以看上面V1和V2的网络结构，第一个普通卷积的卷积核都是32）
- 精简Last Stage：NAS搜索出的网络最后一层称之为Last Stage，作者觉得非常耗时，将其精简为Efficient Stage。
精简后Acc基本不变，耗时减少7ms（占推理时间的11%）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b2e9161653190d3c4dddaa024fac3857.png)
#### 4.4.5 NAS
**1. NAS简介**

&#8195;&#8195;V3的特别之处在于网络结构是使用基于神经架构搜索技术（NAS）学习出来的，使得V3得以将其他很多网络中的优秀特性都集于一身，而在V3之前的V1、V2版本中，网络结构则是由研究人员经过人工设计和计算得到的。本节仅介绍MobileNetV3与NAS的一些相关内容，不对NAS的技术细节展开。
&#8195;&#8195;神经架构搜索（`Network Architecture Search, NAS`）：是一种试图使用机器，自动的在指定数据集的基础上<font color='red'>通过某种算法，找到在此数据集上效果最好的神经网络结构和超参数的方法 </font>，可以一定程度上解决研究人员设计一个网络的耗时的问题。NAS甚至可以发现某些人类之前未曾提出的网络结构，这可以有效的降低神经网络的使用和实现成本。
&#8195;&#8195;目前市面上也已经出现了许多运用NAS技术的产品，如AutoML、OneClick.AI、Custom Vision Service、EasyDL等。在之前一些网络模型中，如NasNet、MNasNet、AmoebaNet、MobileNet，也使用了NAS技术。

**2. NAS原理**
- 搜索：给定一个称为搜索空间的候选神经网络结构集合（即搜索空间包含了如：深度卷积、逐点卷积、常规卷积、卷积核、规范化、线性瓶颈、反向残差结构、SE模块、激活函数等等可作为原子的结构），用某种策略从中搜索出最优网络结构。
- 评估：用某些指标如精度、速度来度量神经网络结构的性能优劣。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4340fe9d69dcee4666c2ba29bf11383c.png)

搜索空间，搜索策略，性能评估策略是NAS算法的核心要素。
- 搜索空间：定义了可以搜索的神经网络结构的集合，即解的空间。
- 搜索策略：定义了如何在搜索空间中寻找最优网络结构。
- 性能评估策略：定义了如何评估搜索出的网络结构的性能。

对这些要素的不同实现得到了各种不同的NAS算法。
- 常见算法：全局搜索空间、cell-based 搜索空间、factorized hierarchical search space分层搜索空间、one-shot 架构搜索等
- 优化算法：基于强化学习的算法、基于进化算法的算法、基于代理模型的算法等。

**3. MobileNetV3中的NAS**
-  使用`platform-aware NAS`搜索全局网络结构的优化block，也就是从搜索空间的集合中根据预定义的网络模板搜索出网络结构；
- 使用`NetAdapt`算法针对block中的每一层搜索需要使用的卷积核个数。
- 性能评估：对搜索出的网络进行性能评估。MobileNetV3是使用真实手机的cpu（pixel-1手机）运行TFLite Benchmark Tool进行性能评估。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/efb62a748c53a392b67e36de5515efc9.png)

#### 4.4.6 MobileNetv3网络结构
`MobileNetv3`有small和large两个版本。
- Large版本共有15个bottleneck层，一个标准卷积层，三个逐点卷积层。
- Small版本共有12个bottleneck层，一个标准卷积层，两个逐点卷积层。

其详细网络结构如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0957a31e2c91bd8beab3df1509f520cd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7157efd4277e3d110ce6f14ca941656d.png)
- `out`：输出矩阵通道数
- `HL`：非线性激活函数。HS表示`h-swish`,RE表示RELU激活函数
- `bneck 3×3`：bneck就是左侧图结构，3×3是dw卷积核的大小
- `exp size`：bneck第一个卷积层升维后的channel数。经过dw卷积核SE模块后通道数不变。最后经过pw卷积进行降维，降维后的维度就是out
- `SE`:是否使用SE模块
- `NBN`：最后两个卷积层不使用Batch Norm
- 第一个bneck结构，exp size=out，即没有进行升维。所以这一层没有第一个1×1升维卷积层。其它bneck和之前讲的一样。
## 五、ConvNeXt
>参考博文[《ConvNeXt网络详解》](https://blog.csdn.net/qq_37541097/article/details/122556545)、[模型讲解视频](https://www.bilibili.com/video/BV1SS4y157fu/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)；[代码地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/ConvNeXt)、[代码讲解视频](https://www.bilibili.com/video/BV14S4y1L791/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)；[PyTorch API](https://pytorch.org/vision/stable/models/convnext.html)
### 5.1 前言
&#8195;&#8195;自从`ViT(Vision Transformer)`在CV领域大放异彩，越来越多的研究人员开始拥入Transformer的怀抱。回顾近一年，在CV领域发的文章绝大多数都是基于Transformer的，比如2021年ICCV 的best paper `Swin Transformer`，而卷积神经网络已经开始慢慢淡出舞台中央。
&#8195;&#8195;2022年一月份，Facebook AI Research和UC Berkeley一起发表了一篇文章`A ConvNet for the 2020s`，在文章中提出了`ConvNeXt`纯卷积神经网络，它对标的是2021年非常火的`Swin Transformer`。
&#8195;&#8195;通过一系列实验比对，**在相同的FLOPs下，`ConvNeXt`相比`Swin Transformer`拥有更快的推理速度以及更高的准确率**，在ImageNet 22K上ConvNeXt-XL达到了87.8%的准确率，参看下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/86dd12750105d2bf6c17b4ead7e3f78e.png)
&#8195;&#8195;上图红色框选这一栏是在A100上，每秒推理图片的数量。不仅是在ImageNet上，在coco数据集和ADE20K数据集上的效果也更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/648aa4934a858c4100952f5bad9e1da8.png)
&#8195;&#8195;阅读论文会发现，`ConvNeXt`使用的全部都是现有的结构和方法，没有任何结构或者方法的创新。而且源码也非常的精简，100多行代码就能搭建完成，相比`Swin Transformer`简直不要太简单。
&#8195;&#8195;为什么现在基于Transformer架构的模型效果比卷积神经网络要好呢？论文中的作者认为可能是**随着技术的不断发展，各种新的架构以及优化策略促使Transformer模型的效果更好，那么使用相同的策略去训练卷积神经网络也能达到相同的效果吗**？抱着这个疑问作者就以Swin Transformer作为参考进行一系列实验。
### 5.2 实验设计
&#8195;&#8195;作者首先利用训练`vision Transformers`的策略去训练原始的`ResNet50`模型，发现在ImageNet上的Top 1 ACC=78.8%，比原始效果要好很多，并将此结果作为后续实验的基准baseline。
&#8195;&#8195;然后作者罗列了接下来实验包含哪些部分，以及每个方案对最终结果的影响（Imagenet 1K的准确率，见下图）。很明显最后得到的ConvNeXt在相同FLOPs下准确率（82%）已经超过了Swin-T（81.3%）。接下来，针对每一个实验进行解析。
>ConvNeXt也有四个不同大小的版本，ConvNeXt 50对标的就是Swin-T，下图准确率也是这两者的对比数据。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/54cd698e1e9382e2390c9c6f7c86ca57.png)

>&#8195;&#8195;Our starting point is a ResNet-50 model. We first train it with similar training techniques used to train vision Transformers and obtain much improved results compared to the original ResNet-50. This will be our baseline. We then study a series of design decisions which we summarized as 1) macro design, 2) ResNeXt, 3) inverted bottleneck, 4) large kernel size, and 5) various layer-wise micro designs.

### 5.3 Macro design
在这个部分作者主要研究两方面：
1. Changing stage compute ratio：
	- 在原ResNet网络中，一般conv4_x（即stage3）堆叠的block的次数是最多的。如下图中的`ResNet50`中stage1到stage4堆叠block的次数是(3, 4, 6, 3)比例大概是1:1:2:1
	- 在`Swin Transformer`中，比如Swin-T的比例是1:1:3:1，Swin-L的比例是1:1:9:1。很明显，**在Swin Transformer中，stage3堆叠block的占比更高**。
	- 最后作者就将ResNet50中的堆叠次数由(3, 4, 6, 3)调整成(3, 3, 9, 3)，和Swin-T拥有相似的FLOPs。进行调整后，准确率由`78.8%`提升到了`79.4%`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f4096d49d5f262d08673fe3bbf610d40.png)


2.  Changing stem to “Patchify”
	- 最初的卷积神经网络中，我们将下采样模块称之为stem。在ResNet中，stem是由一个卷积核大小为7x7步距为2的卷积层，以及一个步距为2的最大池化下采样共同组成，最终高和宽都下采样4倍（上图红色框）。
	- 但在Transformer模型中，下采样一般都是通过一个卷积核非常大且相邻窗口之间没有重叠的（即stride等于kernel_size）卷积层完成。比如在`Swin Transformer`中，采用的是一个卷积核大小为4x4步距为4的卷积层构成`patchify`，同样是下采样4倍。
	- 最后作者将ResNet中的stem也换成了和Swin Transformer一样的patchify。替换后准确率从79.4% 提升到79.5%，并且FLOPs也降低了一点。

### 5.4 ResNeXt-ify
&#8195;&#8195;接下来作者借鉴了`ResNeXt`中的组卷积`grouped convolution`，而作者采用的是更激进的`depthwise convolution`，即group数和通道数channel相同。这样做的另一个原因是作者认为`depthwise convolution`和`self-attention`中的加权求和操作很相似。
1.  `grouped convolution`参考[《学习笔记五：卷积神经网络原理、常见模型》](https://blog.csdn.net/qq_56591814/article/details/124603340)3.8章节Resnext、[Resnext讲解视频](https://www.bilibili.com/video/BV1Ap4y1p71v/?vd_source=21011151235423b801d3f3ae98b91e94)。
	- 组卷积简单讲，就是将输入矩阵的channel划分为g个组，然后对每个组分别进行卷积操作，最后将其在channel维度拼接，得到最终卷积结果。这样算下来，组卷积参数个数是常规卷积的1/g
	- 组卷积在不明显增加参数量级的情况下提升了模型的准确率，同时由于拓扑结构相同，超参数也减少了。
2.  `depthwise convolution`参考[《MobileNet(v1、v2)网络详解与模型的搭建》](https://blog.csdn.net/qq_37541097/article/details/105771329)、[讲解视频](https://www.bilibili.com/video/BV1yE411p7L7)。简单讲就是每个卷积核的channel=1。
- 在传统卷积中，每个卷积核的channel与输入特征矩阵的channel相等（每个卷积核都会与输入特征矩阵的每一个维度进行卷积运算）。
	- 而在DW卷积中，每个卷积核的channel都是等于1的，即**每个卷积核只负责输入特征矩阵的一个channel**。故<font color='deeppink'>$num_{kernel}=channel_{in}=channel_{out}$ </font>。
	- 如果想改变/自定义输出特征矩阵的channel，那只需要在DW卷积后接上一个PW卷积（普通1×1卷积）即可。
	- <font color='deeppink'>DW卷积在准确率小幅降低的前提下，大大减少模型参数与运算量。 </font>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/99be7ec1bebacc49f9d9a2adeb33588b.png)
&#8195;&#8195;如上图所示，ResNet和ResNeXt唯一区别就是将残差模块中间的普通卷积层替换为了组卷积层。但这个模块都是两头粗中间细的瓶颈结构（ResNet首尾两层channel=256，中间层channel=64。ResNeXt三层通道数分别是256，128，256）。

3.  将中间3×3普通卷积层直接替换为DW卷积层后，准确率从`79.5%`降到`78.3%`，FLOPs差不多降低了一半。`80.5%`
4.  增大输入特征层宽度。原先`ResNet`第一个stage输入特征层channel=64，而在`Swin-T`中是96。所以作者将最初的通道数由64调整成96，和Swin Transformer保持一致，最终准确率达到了`80.5%`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/394cab87f4e2fb1cad901e8708eada6e.png)
### 5.5 Inverted Bottleneck
&#8195;&#8195;作者认为`Transformer block`中的`MLP`模块非常像`MobileNetV2`中的`Inverted Bottleneck`模块，即两头细中间粗。所以将`ResNet`中采用的`Bottleneck`模块替换为`Inverted Bottleneck`模块。在较大模型上acc由81.9%提升到82.6%。
- 下图a是`ResNet`中采用的`Bottleneck`模块，是一个两头粗中间细的瓶颈结构
-  b是`MobileNetV2`采用的`Inverted Botleneck`模块（图b的最后一个1x1的卷积层画错了，应该是384->96，后面如果作者发现后应该会修正过来）
- c是`ConvNeXt`采用的是`Inverted Bottleneck`模块。

>&#8195;&#8195;关于MLP模块可以参考本文1.2.4 MLP Head，关于Inverted Bottleneck模块可以参考之前讲的[MobileNetv2博文](https://blog.csdn.net/qq_37541097/article/details/105771329)。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4da173c1f8222a02e3002d6c04662cf3.png)

### 5.6 Large Kernel Sizes
&#8195;&#8195;Large Kernel Sizes部分的两个改动是`Moving up depthwise conv layer`（`depthwise conv`模块上移）和`Increasing the kernel size`，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7f330c66130d9a31b5e5eeba76214de5.png)
1. `depthwise conv`模块上移。上移后，准确率下降到了79.9%，同时FLOPs也减小了。这么做是因为作者认为`depthwise conv`类似于Transformer中的MSA，而在transformer中，MSA模块是放在MLP模块之前的，所以这里进行效仿，将depthwise conv上移。
2.  增大卷积核size，准确率从`79.9%` (3×3) 增长到 `80.6%` (7×7)。
	- 因为之前VGG论文中说通过堆叠多个3x3的窗口可以替代一个更大的窗口，而且现在的GPU设备针对3x3大小的卷积核做了很多的优化，会更高效，所以现在主流的卷积神经网络都是采用3x3大小的窗口。
	- 在Transformer中一般都是对全局做self-attention，比如`Vision Transformer`。即使是`Swin Transformer`也有7x7大小的窗口，所以作者将kernel size增大为7×7.
	- 当然作者也尝试了其他尺寸，包括3, 5, 7, 9, 11；发现取到7时准确率就达到了饱和。
### 5.7 Micro Design
&#8195;&#8195;接下来作者在聚焦到一些更细小的差异，比如激活函数以及Normalization。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2310102d519da078fccf2463c2bee279.png)


1. <font color=' deep yellow'>将CNN常用的激活函数`ReLU`替换为`GELU`（transformer）</font>，替换后准确率没变化。
2.  <font color=' deep yellow'>使用更少的激活函数</font>。在卷积神经网络中，一般会在每个卷积层或全连接后都接上一个激活函数。但在Transformer中并不是每个模块后都跟有激活函数，比如MLP中只有第一个全连接层后跟了GELU激活函数。所以作者在ConvNeXt Block中也减少激活函数的使用，如下图所示，减少后发现准确率从80.6%增长到81.3%

3.  <font color=' deep yellow'>使用更少的Normalization</font>。同样在Transformer中，Normalization使用的也比较少，接着作者也减少了ConvNeXt Block中的Normalization层，只保留了depthwise conv后的Normalization层。此时准确率已经达到了81.4%，已经超过了Swin-T。
4. <font color=' deep yellow'>将BN替换成LN</font>，准确率小幅提升到81.5%。卷积神经网络中使用的是Batch Normalization（加速网络的收敛并减少过拟合），但在Transformer中基本都用的Layer Normalization（LN）。因为最开始Transformer是应用在NLP领域的，BN又不适用于NLP相关任务。
5. <font color=' deep yellow'>单独的下采样层</font>（Separate downsampling layers），更改后准确率就提升到了82.0%。
	- ResNet网络中下采样方式并不合理
		- 在`ResNet`网络中stage2-stage4的下采样，都是通过设置主分支和右侧捷径分支的stride=2来进行下采样的。其中主分支kernel size=3×3，捷径分支kernel size=1×1。
		- 但其实这样右侧分支设置是不合理的，kernel size=1×1，stride=2则意味着1×1的格子每次移动2格，这样就丢失了3/4的信息。所以有了改进型的ResNet50-D。
		- `ResNet50-D`中，主分支不变，右侧分支是通过步长为2的池化层实现的。这样两边在下采样时都不会丢失信息，更加合理。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/78685b1ff0f87d1a3cfff99ec0c057fb.png)

	- 在Swin Transformer中，下采样是通过一个单独的`Patch Merging`实现的。
	- 作者就为ConvNext网络单独使用了一个下采样层，就是通过`Laryer Norm+Conv 2×2，stride=2`的卷积层构成。
### 5.8 ConvNeXt模型结构和不同版本的参数
&#8195;&#8195;下面是小绿豆手绘的ConvNeXt模型结构图。仔细观察`ConvNeXt Block`会发现其中还有一个`Layer Scale`操作（即对每个channel的数据进行缩放，论文中并没有提到）。它将输入的特征层乘上一个可训练的参数，该参数就是一个向量，元素个数与特征层channel相同。
>Layer Scale操作出自于Going deeper with image transformers. ICCV, 2021这篇文章，有兴趣的可以自行了解。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/28e33736203f2d9c455f7b864072f217.png)
&#8195;&#8195;对于ConvNeXt网络，作者提出了`T/S/B/L`四个版本，计算复杂度好和Swin Transformer中的T/S/B/L相似。其中C代表4个stage中输入的通道数，B代表每个stage重复堆叠block的次数。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/51332ea7bdb315b78ddb77f6dcf90099.png)
### 5.9 代码讲解（待补充）
>参考[ConvNeXt代码地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/ConvNeXt)、[代码讲解视频](https://www.bilibili.com/video/BV14S4y1L791/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)

## 六、EfficientNet系列
### 6.1 EfficientNetV1
>- 原论文名称：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)、[源码地址](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
>- 小绿豆实现的 [pytorch代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test9_efficientNet)、 [tensorflow代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test9_efficientNet)、[代码讲解视频](https://www.bilibili.com/video/BV19z4y1179h/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)。博客[《EfficientNet网络详解》](https://blog.csdn.net/qq_37541097/article/details/114434046)、[模型讲解视频](https://www.bilibili.com/video/BV1XK4y1U7PX/?vd_source=21011151235423b801d3f3ae98b91e94)
#### 6.1.1 前言
&#8195;&#8195;在之前的一些手工设计网络中(AlexNet，VGG，ResNet等等)经常有人问，为什么输入图像分辨率要固定为224，为什么卷积的个数要设置为这个值，为什么网络的深度设为这么深？这些问题你要问设计作者的话，估计回复就四个字——工程经验。
&#8195;&#8195;而这篇论文主要是用`NAS（Neural Architecture Search）`技术来搜索网络的图像输入分辨率`r` ，网络的深度`depth`以及channel的宽度`width`三个参数的合理化配置。
&#8195;&#8195;论文中提到，本文提出的`EfficientNet-B7`在`Imagenet top-1`上达到了当年最高准确率`84.3%`，与之前准确率最高的GPipe相比，参数数量（Params）仅为其1/8.4，推理速度提升了6.1倍（看上去又快又轻量，但个人实际使用起来发现很吃显存）。下图是EfficientNet与当时主流的网络的对比（注意，参数数量少并不意味推理速度就快）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5e7e9afaba514c1a13447d121fba713a.png)
#### 6.1.2 论文思想
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8db54149a86a46c9c71b44951b2180bc.png)
- 图a：baseline
- 图b：增加网络宽度，能够获得更高细粒度的特征并且也更容易训练。但对于width很大而深度较浅的网络，往往很难学习到更深层次的特征。
- 图c：增加网络的深度，使用更多层结构。能够得到更加丰富、复杂的特征并且能够很好的应用到其它任务中。但网络的深度过深会面临梯度消失，训练困难的问，
- 图d：增加输入网络的分辨率，可以获得更高细粒度的特征模板。但是分辨率过高，acc增益会变少，且大幅增加计算量。
- 图e：同时增加网络的width、网络的深度以及输入网络的分辨率来提升网络的性能（综合b、c、d）

下图虚线是分别增加width、depth、resolution后的结果，可以看到Accuracy达到80%时就趋于饱和了。而红色实线是三者同时增加，同计算量下，效果更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d50eef35c9376d66a5a0f172dfc87262.png)
那么该如何同时增加width、depth和分辨率呢？
1. 作者在论文中对整个网络的运算进行抽象：作者在论文中对整个网络的运算进行抽象：
 $$N(d,w,r)=\underset{i=1...s}{\odot} {F}_i^{L_i}(X_{\left\langle{{H}_i, {W}_i, {C}_i } \right\rangle})$$
其中：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d15f2444283d1f4da97611bb757a9a0e.png)
论文中通过 NAS（Neural Architecture Search） 技术搜索得到的EfficientNetB0的结构，如下图所示:
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/224a0e4c2d148291b4ac2dfac9512fe8.png)
2. 为了探究d , r , w 这三个因子对最终准确率的影响，则将其加入到公式中，我们可以得到抽象化后的优化问题（在指定资源限制下，其中s , t 代表限制条件），：
其中s . t . s.t.s.t.代表限制条件：
 $$\underset{d, w, r}{max} \ \ \ \ \ Accuracy(N(d, w, r)) \\ s.t. \ \ \ \ N(d,w,r)=\underset{i=1...s}{\odot} \widehat{F}_i^{d \cdot \widehat{L}_i}(X_{\left\langle{r \cdot \widehat{H}_i, \ r \cdot \widehat{W}_i, \ w \cdot \widehat{C}_i } \right\rangle}) \\ Memory(N) \leq {\rm target\_memory} \\ \ \ \ \ \ \ \ \ \ \ FLOPs(N) \leq {\rm target\_flops} \ \ \ \ \ \ \ \ (2)$$
其中：
- `d`用来缩放深度 $\widehat{L}_i$
- `r` 用来缩放分辨率即影响H ^ i $\widehat{H}_i$和$\widehat{W}_i$
- `w`就是用来缩放特征矩阵的channel即$\widehat{C}_i$ 
- `target_memory`为memory限制
- `target_flops`为FLOPs限制
3. 接着作者又提出了一个混合缩放方法  `compound scaling method` 。在这个方法中使用了一个混合因子$\phi$去统一的缩放width，depth，resolution参数，具体的计算公式如下（s.t.代表限制条件）：

$$depth: d={\alpha}^{\phi} \\ width: w={\beta}^{\phi} \\ \ \ \ \ \ \ resolution: r={\gamma}^{\phi} \ \ \ \ \ \ \ \ \ \ (3) \\ s.t. \ \ \ \ \ \ \ {\alpha} \cdot {\beta}^{2} \cdot {\gamma}^{2} \approx 2 \\ \alpha \geq 1, \beta \geq 1, \gamma \geq 1 $$
其中：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c4226eee14fd03dadd09281a53c55f85.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c80fcc784c162674906d9f290c7d66ab.png)
#### 6.1.3 网络详细结构
&#8195;&#8195;下表为EfficientNet-B0的网络框架，网络总共分成了9个Stage（B1-B7就是在B0的基础上修改Resolution，Channels以及Layers）：
- `Stage1`：就是一个卷积核大小为3x3步距为2的普通卷积层（包含BN和激活函数Swish）
- `Stage2～Stage8`：都是在重复堆叠MBConv结构（最后一列的Layers表示该Stage重复MBConv结构多少次）
- `Stage9`：由一个普通的Conv 1×1（包含BN和激活函数Swish）、Pooling层和全连接层组成。
- 倍率因子n：表格中每个MBConv后会跟一个数字1或6就是倍率因子n，表示MBConv中第一个1x1的卷积层会将输入特征矩阵的channels扩充为n倍。
- k3x3或k5x5表示MBConv中Depthwise Conv所采用的卷积核大小。
- Channels表示通过该Stage后输出特征矩阵的Channels。
- Resolution是输入每个stage的特征图的高宽。
- $\widehat{L}_i$表示在该Stage中重复$\widehat{F}_i$的次数（$\widehat{F}_i$表示对应Stage的运算操作）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/44287d133244fff29e8e22e274256205.png)
#### 6.1.4 MBConv结构
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dbcd17dec7c09ec5691ae38ab6f402ab.png)
如图所示，`MBConv`（Mobile Conv）结构主要以下几个部分组成：
1. `Conv 1×1，stride=1`：普通卷积层，作用是升维，包含BN和Swish；卷积核个数是输入特征矩阵channel的n倍。（上面提的倍率因子n，当n=1时不需要这一层）
2. `Depthwise Conv`卷积，kernel size可以是3x3和5x5，也包含BN和Swish。
3. `SE`模块:由一个全局平均池化，两个全连接层组成。
	- 第一个全连接层的节点个数是输入该MBConv特征矩阵channels的1/4且使用Swish激活函数。
	- 第二个全连接层的节点个数等于Depthwise Conv层输出的特征矩阵channels，且使用Sigmoid激活函数。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5a1326c96f2fd95bef3f087b0ba447f1.png)

4. `Conv 1×1，stride=1`：普通卷积层，作用是降维
5. `Droupout`层

需要注意的是：
- 关于shortcut连接，仅当输入MBConv结构的特征矩阵与输出的特征矩阵shape相同时才存在（代码中可通过stride\==1 and inputc_channels==output_channels条件来判断）
- 在源码实现中只有使用shortcut的时候才有Dropout层
#### 6.1.5 EfficientNet(B0-B7)参数及性能对比
EfficientNetB0网络结构上面已经给出，其他版本的详细参数可见下表：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3a1c8e1df73b90a34398984b6e5be14e.png)
- `input_size`：训练网络时输入网络的图像大小
- `width_coefficient`：代表channel维度上的倍率因子，比如在 EfficientNetB0中Stage1的3x3卷积层所使用的卷积核个数是32，那么在B6中就是$32 \times 1.8=57.6$,接着取整到离它最近的8的整数倍即56，其它Stage同理。
- `depth_coefficient`：代表depth维度上的倍率因子（仅针对Stage2到Stage8），比如在EfficientNetB0中Stage7的L${\widehat L}_i=4$ ，那么在B6中就是$4 \times 2.6=10.4$接着向上取整即11。
- `drop_connect_rate`：在MBConv结构中dropout层使用的drop_rate，在官方keras模块的实现中MBConv结构的drop_rate是从0递增到drop_connect_rate的（具体实现可以看下官方[源码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet.py)，注意，在源码实现中只有使用shortcut的时候才有Dropout层）。
- 这里的Dropout层是Stochastic Depth，即会随机丢掉整个block的主分支（只剩捷径分支，相当于直接跳过了这个block）。也可以理解为减少了网络的深度。具体可参考Deep Networks with Stochastic Depth这篇文章。
- `dropout_rate`是最后一个全连接层前的dropout层（在stage9的Pooling与FC之间）的dropout_rate。

最后给出原论文中关于EfficientNet与当时主流网络的性能参数对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7c00d33362d806f841863616a1bd6059.png)
### 6.2 EfficientNetV2
>参考：
>- [《EfficientNetV2网络详解》](https://blog.csdn.net/qq_37541097/article/details/116933569)、[bilibil视频](https://www.bilibili.com/video/BV19v41157AU/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[项目地址](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test11_efficientnetV2)
>- [《论文推荐：EfficientNetV2 - 通过NAS、Scaling和Fused-MBConv获得更小的模型和更快的训练速度》](https://blog.csdn.net/deephub/article/details/122958432?ops_request_misc=&request_id=&biz_id=102&utm_term=EfficientNetV2&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-9-122958432.142%5Ev50%5Econtrol,201%5Ev3%5Econtrol_1&spm=1018.2226.3001.4187)
####  6.2.1EfficientNetV2性能对比
&#8195;&#8195;`EfficientNetV2`是在2021年4月份发布的，主要是做了两点改进：
- <font color='deeppink'>引入Fused-MBConv模块</font>：将原来`MBConv`主分支中的expansion conv1x1和depthwise conv3x3替换成一个普通的conv3x3。替换后参数更少，训练更快。
- <font color='deeppink'>渐进式学习策略</font>：根据训练图像的尺寸动态调节正则方法，提升了训练速度和准确率。
1. 对比当时主流网络：
	- <font color='deeppink'>紫色线</font>是S/M/L三种尺寸的EfficientNetV2模型的准确率和推理速度
	- <font color='red'>红色线</font>是S/M/L/XL四种尺寸的EfficientNetV2模型在ImageNet 21K上预训练，再在ImageNet 1K上进行微调，准确率更高。下图可以看到，其acc和推理速度比ViT-L16(21K)都更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d64cb3454ebb157de84bff6fe424d4af.png)
>`EfficientNetV2-M`对比`EfficientNetV1-B7`，训练速度提升11倍，参数数量减少为1/6.8。
2. 对比传统卷积神经网络/混合网络以及`ViT`网络：
`EfficientNetV2-S`对标的就是`EfficientNet=B5`，可以发现其训练和推理速度更快。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/54482fc82a92916136f905bc4c6dfb33.png)

3. 对比`EfficientNetV1`：
在`EfficientNetV1`中作者关注的是准确率，参数数量以及FLOPs（理论计算量小不代表推理速度快），在`EfficientNetV2`中作者进一步关注模型的训练和推理速度。
	- 表10 是`EfficientNetV2-M`和`EfficientNetV1-B7`的各种指标对比。（准确率、参数量、理论计算量、训练和推理时间）
	- 表11 也可以看出`EfficientNetV2`比`V1`在acc差不多的情况下，推理速度快很多。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/047f596ab7b3cb55886f64d3080979bc.png)
#### 6.2.2 V1中存在的问题和V2的改进
作者系统性的研究了EfficientNet的训练过程，并总结出了三个问题：
1. <font color='deepyellow'> 训练图像的尺寸很大时，训练速度非常慢。 </font>
	- 在之前使用EfficientNet时发现当使用到B3（img_size=300）- B7（img_size=600）时基本训练不动，而且非常吃显存。
	- 通过下表可以看到，在Tesla V100上当训练的图像尺寸为380x380时，batch_size=24还能跑起来，当训练的图像尺寸为512x512时，batch_size=24时就报OOM（显存不够）了。
	- 针对这个问题一个比较好想到的办法就是降低训练图像的尺寸，这样不仅能够加快训练速度，还能使用更大的batch_size。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9b5d8987a85c273601b81e5d37985be5.png)

2. <font color='deepyellow'> 网络浅层中使用Depthwise convolutions速度会很慢，故引入</font>`Fused-MBConv` 。
`Fused-MBConv`结构也非常简单，即将原来`MBConv`主分支中的expansion conv1x1和depthwise conv3x3替换成一个普通的conv3x3，如下图左侧所示。
下表是不同stage的`MBConv`替换为`Fused-MBConv`的准确率和训练速度。作者使用NAS技术去搜索最佳组合，将浅层（stage1-3）替换为`Fused-MBConv`时效果最好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa1a1e53e09067beb6b1962a304c246e.png)

3. <font color='deepyellow'> 非均匀缩放stage</font>
&#8195;&#8195;在`EfficientNetV1`中，每个stage的深度和宽度都是同等放大的（同等的看待每个stage）。但每个stage对网络的训练速度以及参数数量的贡献并不相同，所以直接使用同等缩放的策略并不合理。在这篇文章中，作者采用了非均匀的缩放策略来缩放模型（论文中没有给出策略，直接给出缩放后的参数）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7e39cdef75a0a75e3cd678e6b53a2620.png)
#### 6.2.3  EfficientNetV2网络框架
**V1、V2框架的不同点**

表4展示了作者使用NAS搜索得到的EfficientNetV2-S模型框架，相比与EfficientNetV1，主要有以下不同：
1. 浅层使用了`Fused-MBConv`模块（stage1-3）
2. 使用较小的expansion ratio，这样能够减少内存访问开销。
>&#8195;&#8195;`expansion ratio`就是上一节提到的倍率因子n，也就是下表中的MBConv4中的4，表示MBConv模块第一层会将输入通道数增大为原来的4倍。而在V1中，基本都是6。
3. 使用更小(3x3)的kernel_size，而在V1中使用了很多5x5的kernel_size，所以V2需要堆叠更多的层结构以增加感受野。
4. 移除了`EfficientNetV1`中的stage8（stride=1）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/38f9ee0e4143e06fd1c19445533e6471.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c06c89108acf0db3494c7121af1d0038.png)
**EfficientNetV2-S结构详情**
-  `Operator`：在当前Stage中使用的模块
- `stride`：该Stage第一个 `Operator`的步距，剩下的 `Operator`步距都是1（也就是每个模块只有第一层可能会下采样，后面层不变）
- `Channels`：该Stage输出的特征矩阵的Channels
- `Layers`：该Stage重复堆叠Operator的次数
-  `Conv3x3`：普通的3x3卷积 + 激活函数（SiLU）+ BN
- `Fused-MBConv1，k3×3`：1表示`expansion ratio`，k表示kernel_size。
	- `Fused-MBConv`在论文中有SE模块，源码中没有。估计是SAN搜索后发现效果不好，删了
	- 当expansion ratio等于1时是没有expand conv的
	- 当stride=1且输入输出Channels相等时才有shortcut连接，shortcut连接时才有Dropout层
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/33cd425c731e98f35122b8309e7f5d6e.png)
	- ` MBConv`和`Fused-MBConv`中（不包括全连接层中的dropout），Dropout层使用的是Stochastic Depth，随机丢掉整个block的主分支，只剩捷径分支，相当于直接跳过了这个block。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1a8f23bcd48fd700d9fc9c744c4c7651.png)

- ` MBConv`模块和EfficientNetV1中是一样的。SE0.25表示使用了SE模块，且模块中第一个全连接层的节点个数，是输入特征矩阵channels的1/4 。（当stride=1且输入输出Channels相等时，才有shortcut连接和Stochastic Depth，详细的可以参考本文《4.1.4 MBConv结构》）
​![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ee2a4e942c1e12880b63069207029fb9.png)
#### 6.2.4 Progressive Learning渐进学习策略
- 动态的图像尺寸导致ACC降低：前面提到过，训练图像的尺寸对训练模型的效率有很大的影响。而如果尝试使用动态的图像尺寸（比如一开始用很小的图像尺寸，后面再增大）来加速网络的训练，但通常会导致Accuracy降低。
- 动态的正则方法：作者猜想Acc的降低是不平衡的正则化`unbalanced regularization`导致的。在训练不同尺寸的图像时，应该使用动态的正则方法（之前都是使用固定的正则方法）。

&#8195;&#8195;于是作者接着做了一些实验，训练过程中尝试使用不同的图像尺寸以及不同强度的数据增强。当训练的图片尺寸较小时，使用较弱的数据增强；当训练的图像尺寸较大时，使用更强的数据增强。如下表所示，当Size=128，RandAug magnitude=5时效果最好；当Size=300，RandAug magnitude=15时效果最好：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8503970a0e153950379167bfe1beae69.png)
&#8195;&#8195;基于以上实验，作者就提出了渐进式训练策略`Progressive Learning`。如上图所示：
- <font color='red'> 在训练早期（epoch = 1开始）使用较小的训练尺寸以及较弱的正则方法 </font>weak regularization，这样网络能够快速的学习到一些简单的表达能力。
- <font color='red'> 接着逐渐提升图像尺寸，同时增强正则方法 。正则包括dropout rate，RandAugment magnitude以及mixup ratio。</font>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6bd24b1641d684414d08d1f9cde41daf.png)
训练流程的伪代码：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/148aa90f7f7f906feaebf8bc69dfb602.png)
下表给出了EfficientNetV2（S，M，L）三个模型的渐进学习策略参数：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd0bb39685d2ca14b096a592de637311.png)
&#8195;&#8195;作者还在Resnet以及EfficientNetV1上进行了测试，使用了渐进式学习策略后确实能够有效提升训练速度并且能够小幅提升Accuracy。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/611ae091493245822997134d35f12b63.png)

#### 6.2.5 配置文件
 `EfficientNetV2-S`：在baseline的基础上采用了width倍率因子1.4， depth倍率因子1.8得到的（block里不包含stage0这个普通卷积层）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7295e3add1f2bd9e0b1660419b2c0545.png)
其它模型配置参考见小绿豆的文章，这里就不写了。其它训练参数：

```python
# (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)

efficientnetv2_params = {
'efficientnetv2-s': # 83.9% @ 22M
(v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
'efficientnetv2-m': # 85.2% @ 54M
(v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
'efficientnetv2-l': # 85.7% @ 120M
(v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),
							}
```


