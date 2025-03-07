@[toc]

传送门：
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT](https://blog.csdn.net/qq_56591814/article/details/127313216?spm=1001.2014.3001.5501)
- [李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
- [李沐论文精读系列三：MoCo、对比学习综述（MoCov1/v2/v3、SimCLR v1/v2、DINO等）](https://blog.csdn.net/qq_56591814/article/details/127564330)

## 一、CLIP 
>参考：
>- 论文[Learning Transferable Visual Models From Natural Language Supervision](https://paperswithcode.com/paper/learning-transferable-visual-models-from)、[官方代码](https://github.com/OpenAI/CLIP)
>- 李沐论文精度系列之[《CLIP 论文逐段精读》](https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[《CLIP学习笔记》](https://blog.csdn.net/Qi__Xi/article/details/124185059?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166650615616782427451654%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166650615616782427451654&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-2-124185059-null-null.142%5Ev59%5Epc_new_rank,201%5Ev3%5Econtrol_2&utm_term=CLIP%20%E7%B2%BE%E8%AF%BB&spm=1018.2226.3001.4187)、[《神器CLIP：连接文本和图像，打造可迁移的视觉模型》](https://zhuanlan.zhihu.com/p/493489688)

###  1.1 简介
#### 1.1.1 前言
&#8195;&#8195;`CLIP`是OpenAI在2021年2月发表的一篇文章，其全称为`Contrastive Language-Image Pre-training`，即一种基于对比文本-图像对的预训练方法。`CLIP`用文本作为监督信号来训练可迁移的视觉模型，使得最终模型的zero-shot效果堪比ResNet50，泛化性非常好，而且`CLIP`还有很多好玩的应用。
>&#8195;&#8195;`zero-shot`就是直接推理，用见过的图片特征去判断没见过的图片的类别，而完全不用下游任务训练集进行微调。（相当于把模型用作特征提取，但是没有分类头）
>&#8195;&#8195;作者在30多个不同的计算机视觉数据集上进行基准测试，（这些数据集涵盖了OCR、视频中的动作识别、地理定位和许多类型的细粒度对象分类等任务）`CLIP`通常都能够与监督模型的baseline效果相媲美。
>&#8195;&#8195;例如在ImageNet数据集上，`CLIP`模型在不使用`ImageNet`数据集的任何一张图片进行训练的的情况下，最终模型精度能跟一个有监督的训练好的`ResNet-50`打成平手（在ImageNet上`zero-shot`精度为76.2%，这在之前一度被认为是不可能的）。

#### 1.1.2 模型结构
**训练过程**：
&#8195;&#8195;如下图所示，`CLIP`的输入是一对对配对好的的图片-文本对（比如输入是一张狗的图片，对应文本也表示这是一只狗）。这些文本和图片分别通过`Text Encoder`和`Image Encoder`输出对应的特征。然后在这些输出的文字特征和图片特征上进行对比学习。
&#8195;&#8195;假如模型输入的是`n`对图片-文本对，那么这`n`对互相配对的图像–文本对是正样本（下图输出特征矩阵对角线上标识蓝色的部位），其它$n^2-n$对样本都是负样本。这样模型的训练过程就是最大化n个正样本的相似度，同时最小化$n^2-n$个负样本的相似度。

>&#8195;&#8195;`Text Encoder`可以采用NLP中常用的`text transformer`模型；而`Image Encoder`可以采用常用`CNN`模型或者`vision transformer`等模型。
>&#8195;&#8195;相似度是计算文本特征和图像特征的余弦相似性`cosine similarity`
>&#8195;&#8195;为了训练`CLIP`，`OpenAI`从互联网收集了共4个亿的文本-图像对，论文称之为`WIT`(`Web Image Text`。`WIT`质量很高，而且清理的非常好，其规模相当于`JFT-300M`，这也是`CLIP`如此强大的原因之一（后续在WIT上还孕育出了`DALL-E`模型）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b5bc60a113e7f8b7e06f8aeb7feebee5.png#pic_center )
**分类**
&#8195;&#8195;CLIP可以直接实现`zero-shot`的图像分类，即不需要任何训练和微调，这也是CLIP亮点和强大之处。用CLIP实现zero-shot分类只需要简单的两步：
- 根据任务的分类标签构建每个类别的描述文本：`A photo of {label}`，然后将这些文本送入`Text Encoder`得到对应的文本特征。如果类别数目为n，那么将得到`n`个文本特征；
- 将要预测的图像送入`Image Encoder`得到图像特征，然后与`n`个文本特征计算缩放的余弦相似度（**和训练过程保持一致**），然后选择相似度最大的文本对应的类别作为图像分类预测结果。进一步地，可以将这些相似度看成logits，送入softmax后可以到每个类别的预测概率。

&#8195;&#8195;我们不再需要预先定义好的标签（类别）列表，直接将图片喂给不同的文本句子，就可以知道图片中是否有我们感兴趣的物体。即，CLIP的多模态特性（利用文本监督信号）为具体的任务构建了动态的分类器，使得模型不再受限于预先定义好的类别，更加具有通用性和可用性。
>&#8195;&#8195;比如新增三轮车的图片时，只需要在文本部分也加上三轮车这个类别，模型很有可能直接`zero-shot`推理出图片属于三轮车这个类。而之前的模型，是永远不会预测出ImageNet1000个类之外的类的，这也是`CLIP`最吸引人的地方。
>&#8195;&#8195;类别单词变成句子，有`prompt engineering`和`prompt ensemble`两种方法，进一步提高模型准确率，在论文后面会讲到
####    1.1.3 模型效果
##### 1.1.3.1 对自然分布偏移的鲁棒性
&#8195;&#8195;如下图所示，作者还比较了`zero-shot CLIP`与现有ImageNet模型在自然分布偏移上的性能来验证它的鲁棒性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/70d881fe6f3e60bba02ec66fa956cd0c.png#pic_center )
- 左图的横纵坐标是ImageNet的分布偏移。黑色虚线是理想的鲁棒模型，是线性的、正比例的。普通的模型无法达到这样的理想效果，画出来的曲线只会在黑色虚线的下面。但这里可以看出`zero-shot CLIP`的鲁棒性比标准的ImageNet训练的模型更好。
- `ImageNetV2`是从ImageNet数据集中筛选出新的数据集，其更接近原来的测试集。然而在ImageNet上预训练的模型，在ImageNetV2上的测试性能下降了不少（76.2→64.3）
- 右图中 `ImageNet Sketch`都是素描的图片、`ImageNet-A`包含很多对抗样本

&#8195;&#8195;`CLIP`和基于ImageNet上有监督训练的`ResNet101`，在`ImageNet`验证集都能达到`76.2%`，但是在剩下的五个数据集上，ResNet101性能下降得非常厉害，但是CLIP能依然保持较大的准确度。比如在`ImageNet-A`数据集上，`ResNet101`精度只有`2.7%`，而`CLIP`能达到`77.1%`。
&#8195;&#8195;这也说明`CLIP`学习到的视觉特征，已经和语言产生了很强的联系。这也不论是自然的香蕉还是动漫里的香蕉、素描的香蕉、加了对抗样本的香蕉，`CLIP`都知道图片是对应香蕉这个单词。

##### 1.1.3.2 StyleCLIP
>- 论文[《StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery》](https://paperswithcode.com/paper/styleclip-text-driven-manipulation-of)（ICCV2021）
>- 还有另一篇相关的：[《StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Synthesis》](https://paperswithcode.com/paper/styleclipdraw-coupling-content-and-style-in)

&#8195;&#8195;顾名思义，这是一篇CLIP+styleGAN的工作，可以通过文字的改变引导图像的生成。比如下面例子中，输入“Mohawk hairstyle”，就能改变奥巴马的发型；输入“Without makeup”，就可以一键卸装了；输入“Cute cat”（可爱的猫），猫的眼睛就睁大了。CLIP 也能理解各种抽象妆容，比如烟熏妆，吸血鬼妆。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd3b8b68bba95a0f8d730d5d70fed9e4.png)
##### 1.1.3.3 CLIPDraw
>论文[《CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders》](https://paperswithcode.com/paper/clipdraw-exploring-text-to-drawing-synthesis)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d5cecc70cdbc0b1f3de0549808b016b.png#pic_center )

&#8195;&#8195;这也是一个利用CLIP预训练模型指导图片的生成。 `CLIPDraw`不需要进行训练，通过在一组RGBA Béezier曲线上执行梯度下降，就可以从文本合成一些简笔画图像。（目标是最小化生成图像的`CLIP encodings`与文本提示之间的余弦距离）。
>在普通GPU上生成一张简笔画通常不需要一分钟。最后一张图的self表示自拍照
##### 1.1.3.4 zero-shot检测
>论文[《Open-vocabulary Object Detection via Vision and Language Knowledge Distillation》](https://paperswithcode.com/paper/zero-shot-detection-via-vision-and-language)（ICLR 2022）
>
&#8195;&#8195;CLIP可以应用在目标检测任务上，实现zero-shot检测，即检测训练数据集没有包含的类别。比如在CLIP出现的一个半月之后，谷歌提出的`ViLD`（见本文3.1章节）基于CLIP实现了`Open-vocabulary`的物体检测，其主体架构如下所示，其基本思路和zero-shot分类相似，只不过这里是用文本特征和ROI特征来计算相似度。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0f2bf2df12a0efb63947fb64faa16efd.png#pic_center )

&#8195;&#8195;下面的例子中，如果用传统的目标检测算法的话，模型只会判断这些物体都是玩具，也就是图中蓝色的基础类。使用CLIP之后，就可以摆脱基础类的限制（`Open-vocabulary Object`），可以检测出新的类（图中红色标识），比如颜色和动物类别。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2bc127135480f38c7a873d2d70397859.png#pic_center )
Meta AI的最新工作[Detic](https://link.zhihu.com/?target=https://arxiv.org/abs/2201.02605)可以检测2000个类，背后也用到了CLIP。
##### 1.1.3.5 CLIP视频检索
&#8195;&#8195;github上[johanmodin/clifs](https://github.com/johanmodin/clifs)仓库，展示了使用CLIP视频检索的工作。可以通过输入文本直接找到视频中出现的对应物体。比如输入“一辆印有odwalla的卡车”，就真的在视频中找到了这辆卡车（CLIP把这句话变成文本特征，然后将视频中每一帧都当成视觉特征，然后一帧帧的去和文本特征做对比，然后挑出相似性最高的那一帧）。![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/649702f8ff4202685ec239d899ca68ff.png#pic_center )
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9be8638770338f1366aa2e41b3f2f152.png#pic_center )
####    1.1.4 导言
&#8195;&#8195;现有的CV模型基本都是基于人工标注的数据集进行训练的，然后用来预测一组提前定义好的物体类别。这种提前定义好的标签集合，会大大简化问题本身（比如ImageNet固定的1000个类，COCO数据集固定80个类等等）。但正因如此，这种受限的监督信号限制了模型的泛化性和可用性。比如大多数模型都只能预测已知的图像类别。对于没有见过的图像类别，需要额外的信息才能识别。这样每次新增一些类别，都需要重新收集数据，训练一个新的模型。

&#8195;&#8195;而且无论是有监督还是自监督方法（基于对比学习的方法如MoCo和SimCLR，和基于图像掩码的方法如MAE和BeiT），在模型迁移时都需要需要进行有监督微调，比如微调固定类别的softmax分类器，而无法实现zero-shot。

&#8195;&#8195;作者认为，直接从自然语言中得到监督信息是一个很有前途的选择，因为其涵盖的范围更广（只要是语言描述过的物体，都有可能让视觉模型去识别）。CLIP利用多模态的对比学习，使得自然语言可以引导模型学习到视觉概念，从而实现非常灵活的`zero-shot`迁移（把分类问题转化为了跨模态检索问题）。

&#8195;&#8195;使用自然语言监督进行图像表示学习的工作很少，并且效果往往不如有监督模型，主要有两个原因：
1. 早期nlp模型不太好学。
比如早期的n-gram模型非常复杂，不好跨模态训练。但是随着transformer的兴起，像BERT和GPT这种具有上下文表示的自监督训练模型做的越来越好，nlp模型也终于有了取之不尽的文本监督信号，而且使用简单，泛化性好，为多模态训练铺平了道路。
2. **数据集或模型的规模不够**。
比如VirTex和ICMLM都只训练了十几万的图片；ConVIRT非常类似CLIP，但只在医疗图像上做了预训练。从本质上来讲，`CLIP`其实并没有太大的创新，它只是<font color='deeppink'>**将ConVIRT方法进行简化，并采用更大规模的文本-图像对数据集来训练。也可以说，相对于之前的对比学习，CLIP只是将单模态的样本，换成了多模态的样本。**</font>

### 1.2 方法
#### 1.2.1 自然语言监督的优势
使用自然语言监督信号来训练视觉模型，有两个最重要的优势：
- 不需要采用特别的标注数据，扩展性更强。
比如ImageNet需要先定义好1000个类，然后根据这些类去下载图片，清理数据集，再去标注所有图片，过程很复杂。而`CLIP`不要求这种经典的“”机器学习兼容“”的标注格式，只需要下载文字-图片对；且没有n选1的标签之后，模型的输入输出自由度大了很多。

- `CLIP`学习到的是图像结合文字的多模态特征，从而实现灵活的zero-shot迁移。如果只是单模态的特征，无论是类似MOCO还是MAE，都很难做到这一点（zero-shot必须要加入文字特征才能做到）。

#### 1.2.2 预训练方法（训练效率至关重要）

&#8195;&#8195;CV领域的模型都很大，训练起来也很贵。比如[noise student](https://paperswithcode.com/paper/self-training-with-noisy-student-improves)之前在ImageNet一直霸榜，但是这个模型需要在一个 TPUv3上训练33年，这还只是在包含1000类的ImageNet上预训练的，而且只训练视觉特征。
&#8195;&#8195;由于训练数据量和模型计算量都很大，训练效率成为一个至关重要的因素。作者做了很多尝试，最终选择了对比学习：
- `VirTex`模型：预测文本，对应下图蓝色线`Transformer Language Model` 
	- `Image Encoder`使用CNN模型，`Text Encoder`使用transformer模型，两个模型一起从头训练，任务是预测图片对应的文本（image caption）。
	-  这种方法的训练效率太慢，因为根据图片进行文本描述，可能性太多了，你可以从各个角度去描述一张图片。
- `Bag of Words Prediction`（橘色线）：不要求每个词都是按顺序的进行预测，所有词都预测出来就行。这样放宽了约束，训练速度提高了三倍。
- `CLIP`：简化版的`ConVIRT`，基于对比学习。
	- 只需要判断图文是否配对，进一步简化了训练任务，训练效率一下子提升4倍（绿色线）
	- 训练任务更加合理。因为训练数据所包含的文本-图像对是从互联网收集来的，它们存在一定的噪音，二者并不完全匹配。适当的降低训练目标，反而能取得更好的收敛。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/01a66ad2d9e48675b85be0654a82fd7b.png#pic_center )
>&#8195;&#8195;OpenAI是一家GPT化的公司，从GPT系列、DALL-E到Image-GPT等等都是基于GPT做的，唯有`CLIP`因为效率的原因，选择了对比学习进行训练。

&#8195;&#8195;最终`Text Encoder`固定选择一个包含63M参数的text transformer模型，而`Image Encoder`采用了两种的不同的架构。因为<font color='deeppink'>CLIP虽然是多模态模型，但它主要是用来**训练可迁移的视觉模型**。</font>
- `Image Encoder`架构
	- ResNet：ResNet50，ResNet101，RN50x4，RN50x16和RNx64（后面三个模型是按照EfficientNet缩放规则对ResNet分别增大4x，16x和64x得到）
	-  ViT：ViT-B/32，ViT-B/16和ViT-L/14。
- 所有的模型都训练32个epochs，采用AdamW优化器，batch size=32768。
- 只在ResNet50上训练一个epoch进行超参搜索，没有进行进一步的调参
- 两个最大的模型RN50x64需要在592个V100卡上训练18天，ViT-L/14需要在256张V100卡上训练12天
- ViT-L/14效果最好，所以作者还将其在336的分辨率下额外finetune了一个epoch来增强性能，记为 `ViT-L/14@336px`。后面论文中没有特别说明的情况下，进行对比实验的CLIP模型都是指这个。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/693df4f9af36fbbdc09a633afd78c220.png#pic_center )
- 训练细节
	- 数据集非常大，几乎不会出现过拟合，所以`Image Encoder`和`Text Encoder`不需要提前进行预训练。
	- 只使用线性投射层（线性非线性影响不大）。
	- 数据增强只使用图片的随机剪裁，这是因为数据集非常大。
	- 对比学习目标函数中的超参数`τ`，设置成可学习的标量，在训练中自动优化，而不用慢慢调参（还是因为数据集太大，训练很贵）。

>&#8195;&#8195;另外还有很多的训练细节，才使得CLIP真正能被训练出来。训练超大模型，可以参考来自OpenAI的博文：[《How to Train Really Large Models on Many GPUs?》](https://lilianweng.github.io/posts/2021-09-25-train-large/)及对应的[CSDN翻译](https://blog.csdn.net/BAAIBeijing/article/details/120735633?ops_request_misc=&request_id=&biz_id=102&utm_term=How%20to%20Train%20Really%20Large%20Mode&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-120735633.142%5Ev62%5Epc_rank_34_queryrelevant25,201%5Ev3%5Econtrol_2,213%5Ev1%5Et3_esquery_v1&spm=1018.2226.3001.4187)。
####  1.2.3  伪代码
```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - 输入图片维度
# T[n, l] - 输入文本维度，l表示序列长度

# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

#  分别提取图像特征和文本特征
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# 对两个特征进行线性投射，得到相同维度的特征d_e，并进行l2归一化，保持数据尺度的一致性
# 多模态embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 计算缩放的余弦相似度：[n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n) #  对角线元素的labels
loss_i = cross_entropy_loss(logits, labels, axis=0) # image loss
loss_t = cross_entropy_loss(logits, labels, axis=1) # text loss
loss = (loss_i + loss_t)/2 # 对称式的目标函数
```
&#8195;&#8195;在`MOCO`中，真实标签都是0，因为其正样本都是放在第一位，所以正样本对应的索引永远是0；但是在`CLIP`中，正样本都是在对角线上，即（$I_1,T_1$，$I_2,T_2$,......），所以真实标签为`np.arange(n)`。

### 1.3 实验
####  1.3.1  zero-shot 迁移
&#8195;&#8195;研究`zero-shot`的动机：之前的自监督或有监督训练的模型（MOCO、DINO等），主要是学习一种泛化好的特征，所以在做下游任务的时候，还是需要有监督的微调，就依然存在很多问题。比如下游任务的数据集不好收集，存在分布飘偏移（distribution shift）等等。而使用文本引导视觉模型训练，就可以很好的进行`zero-shot`迁移；模型就可以不再训练，不再微调。

如何用CLIP实现zero-shot分类？
&#8195;&#8195;这里我们给出了一个基于CLIP的一个实例（参考官方notebook），这里任务共有6个类别："dog", "cat", "bird", "person", "mushroom", "cup"，首先我们创建文本描述，然后提取文本特征：

```python
# 首先生成每个类别的文本描述
labels = ["dog", "cat", "bird", "person", "mushroom", "cup"]
text_descriptions = [f"A photo of a {label}" for label in labels]
text_tokens = clip.tokenize(text_descriptions).cuda()

# 提取文本特征
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
```

&#8195;&#8195;然后我们读取要预测的图像，输入Image Encoder提取图像特征，并计算与文本特征的余弦相似度：

```python
# 读取图像
original_images = []
images = []
texts = []

for label in labels:
    image_file = os.path.join("images", label+".jpg")
    name = os.path.basename(image_file).split('.')[0]

    image = Image.open(image_file).convert("RGB")
    original_images.append(image)
    images.append(preprocess(image))
    texts.append(name)

image_input = torch.tensor(np.stack(images)).cuda()

# 提取图像特征  
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

# 计算余弦相似度（未缩放）
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
```

&#8195;&#8195;进一步地，我们也可以对得到的余弦相似度计算softmax，得到每个预测类别的概率值，注意这里要对相似度进行缩放：

```python
logit_scale = np.exp(model.logit_scale.data.item())
text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
```

&#8195;&#8195;得到的预测概率如下所示，可以看到6个图像，CLIP模型均能够以绝对的置信度给出正确的分类结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29fbed31d96ad1cc359d81c56bb54ab0.png#pic_center )
#### 1.3.2 Prompt Engineering and Ensembling
1. **Prompt Engineering**

&#8195;&#8195;作者还验证了文本描述时采用prompt的有效性（精度提升`1.3%`）。简单来说，`prompt learning`的核心是通过构建合适prompt（提示）来使预训练模型能够直接应用到下游任务中。

推理时，只使用类别标签作为文本描述效果并不够好，原因有二：
1. 词语存在歧义性
 如果我们直接采用类别标签作为文本描述，那么很多文本就是一个单词，缺少具体的上下文，并不能很好的描述图片内容。
	- 比如在做物体检测时，有一个类别是remote（遥控器）。但如果直接喂给文本编码器，很可能被模型认为是遥远的意思。
	- 同一个词语在不同数据集中所表示的意思可能有所不同。例如在 Oxford-IIIT Pets 数据集中，boxer指的是狗的一个种类，在其他数据集中指的是拳击运动员。
	
	- 所以 CLIP预训练时，用来描述图片内容的文本是一个句子，比如`A photo of {label}`。这里的label就只能是名词，一定程度上消除了歧义性。

4. 使推理和预训练时保持一致（消除distribution gap）。

&#8195;&#8195;另外，还可以根据不同的数据集来调整这个模板，进而提升zero-shot的性能。
&#8195;&#8195;例如当数据集是Oxford-IIIT Pets数据集时（类别都是动物），就可以将模板写成： `A photo of a {label}, a type of pet.`   ；或者在做OCR任务时，在想找的那个文本或者数字上打上双引号，模型就可能知道你是想找双引号里面的内容。

2. **prompt ensembling**

&#8195;&#8195;作者尝试了集成多个模板的效果，即在多个zero-shot分类器上进行集成，这些分类器使用不同的提示模板来构造不同的文本。由于是在嵌入空间(embedding space)而不是概率空间(probability space)上集成的，因此节约了计算成本。在大多数数据集上，`prompt ensembling`都能够提升模型性能。

&#8195;&#8195;最终作者使用了80种模板来进行集成，每种模板使用了不同的形容词，来，描述不同的情境。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1981f7f7f06995b01594f93013cf4093.png#pic_center  =500x)
&#8195;&#8195;上图横坐标表示模型算力，纵坐标表示在多个数据集上的平均分数。绿色曲线表示本文中使用Prompt engineering and ensembling的结果，蓝色曲线表示直接使用无提示上下文的类名的结果。

#### 3.3.3 zero-shot分类效果对比（ResNet-50）
&#8195;&#8195;为了测试CLIP的zero-shot分类的效果怎么样，作者将在27个数据集上的分类效果做成了对比图，下图就是CLIP与基于ResNet-50做Linear Probe的对比。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4c56207ff30a98e96fc6ac54048e9f9b.png#pic_center =500x)
- Linear Probe on ResNet-50：
	- Linear Probe就是冻住预训练好的模型，只训练最后一层的分类器，相当于将预训练模型做特征提取器。
	- ResNet50是在ImageNet上用有监督的方式预训练好的
- 对比结果：
	- 绿色 + 表示相比ResNet-50提升了多少，蓝色 - 表示相比ResNet-50降低了多少。
	- 最终在27个数据集中，CLIP在16个数据集上都超越了有监督训练好的ResNet-50。
	- 对于普通的物体分类任务，CLIP可以很好的做zero-shot迁移，例如车、食物、CIFAR10等数据集，因为图像中有可以描述出来的物体，那对应的文本中也就有这种描述，因此可以很好的匹配；

	- 但CLIP对于更加复杂或抽象的任务就表现比较弱，例如卫星图像分类、淋巴结肿瘤检测等需要特定领域知识的分类任务，CLIP并没有预训练到这些标签信息。

#### 1.3.4 few-shot分类效果对比

&#8195;&#8195;作者认为，这种特别难的任务，完全不给任何标签信息，有点强人所难了，不是很合理。所以论文还对比`few-shot`性能，即只用少量的样本来微调模型，这里对比了3个模型：
- 在ImageNet21K上训练的`BiT-M` （[big transfer](https://paperswithcode.com/paper/large-scale-learning-of-general-visual)），是一个很强的baseline。
- 基于SimCLRv2训练的ResNet50，
- 有监督训练的ResNet50。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1888da2329751aa96bba829236083f78.png#pic_center =500x)
- 横坐标：每个数据集每个类别里，用了多少个标注样本进行Linear Probe的分类器训练。0就相当于`zero-shot`了。
- 纵坐标表示在20个数据集上的平均分类准确度（有7个数据集每个类别不够16个）
- 当每类有16个训练样本时，`BiT-M`模型的性能才和`zero-shot CLIP`打成平手。
- 紫色曲线说明：每类的训练样本只有1个或2个的时候，效果还不如zero-shot CLIP；但当每类的训练样本增加到8个或16个的时候，效果则超越了zero-shot CLIP。这说明对于一些难的数据集来说，有一些训练样本还是非常有必要的。

>&#8195;&#8195;CLIP在做`Linear Probe`的时候，需要扔掉文本编码器部分，接着在图像编码器之后加一层线性分类器，所以分类方式不再是看图像特征与文本特征最相近，而是重新训练一个线性分类器.
>&#8195;&#8195;新加的一层线性分类器是随机初始化的，所以每类有1个标注样本是不够的。这也是为什么一开始性能会比较差，但随着训练样本的增多，模型的分类性能会逐渐提升。

#### 1.3.5 `Linear probe CLIP`对比
&#8195;&#8195;对比完了zero--shot和few-shot，下面自然就是拿下游任务的所有训练集来训练，进行效果对比了。作者在这里选择`Linear probe CLIP`的方式。
&#8195;&#8195;之所以选择`Linear probe`而不是微调，因为Linear probe只有最后一层FC是可以训练的，可学习的空间比较小，相比微调没那么灵活。如果预训练模型没有训练好的话，在下游任务上训练再久也很难优化到一个特别好的结果，所以用Linear probe能更准确的反映预训练模型的好坏。另一个原因就是`Linear probe`不需要怎么调参（因为微调的话，不同数据集可调的参数就太多了）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5e01153c785f3f845744543bdd19e4ff.png#pic_center )
- 横坐标表示对于一张图像来说，做一遍前向过程用多少的计算量
- 纵坐标表示在多个数据集上的平均准确率。
- 对比模型有有监督的EfficientNet、用了伪标签的EfficientNet、弱监督的在Instagram上训练的模型、自监督的对比学习模型、以及一些经典的有监督的baseline模型。
- 结果越靠近左上角，模型的性能越好。
- 左图是在12个数据集上的平均结果，这12个数据集和ImageNet是类似的。所以有监督的在ImageNet上预训练的模型，效果比CLIP好是可以预见的
- 右图是在27个数据集上的平均结果。

&#8195;&#8195;从图中可以看到，在12个数据集上，用ViT结构的CLIP效果最好，用ResNet的效果也比大多数模型要好；在27个数据集上，CLIP的效果就吊打其他所有模型了。这个结果就证明了CLIP模型的强大。

####      1.3.6 与Noisy Student EfficientNet-L2 对比
&#8195;&#8195;作者还在27个数据集上可视化了CLIP模型和用伪标签训练的EfficientNet的性能差异（ImageNet上表现最好）。
&#8195;&#8195;从图中可以看到，CLIP在21个数据集上的性能都超过了EfficientNet，并且很多数据集都是大比分超过。在其余6个表现不如EfficientNet的数据集上，CLIP也只比EfficientNet稍微低一点，差距并不大。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fe6c0ff48cf678a711d2e96dca689f7d.png#pic_center =500x)
### 1.4   与人类的差异（略）
### 1.5 数据重叠分析
&#8195;&#8195;CLIP能实现这么好的zero-shot性能，大家很可能质疑CLIP的训练数据集可能包含一些测试数据集中的样例，即所谓的数据泄漏。关于这点，论文也采用一个重复检测器对评测的数据集重合做了检查，发现重合率的中位数为2.2%，而平均值在3.2%，去重前后大部分数据集的性能没有太大的变化，如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa1da56b7a7f77c076be939f9198d20d.png#pic_center )
- 左：虽然几个数据集在检测到的重叠和干净示例上的zero-shot准确度有高达±20%的明显差异，但在35个数据集中只有5个具有99.5%的Clopper-Pearson置信区间，排除了0%的准确度差异。其中2个数据集在重叠数据上表现更差。
- 右：由于检测到的重叠示例的百分比几乎总是个位数，因此由于重叠导致的整体测试准确度增益要小得多，Birdsnap的最大增幅仅为0.6%。同样，当使用单边二项式检验计算时，只有6个数据集的准确性提高具有统计学意义。

由此可以得出结论，这样的数据重叠不会带来明显的准确率提升。

### 1.6 局限性 
1. 性能有待提高
CLIP在很多数据集上，平均下来看可以和ResNet-50打成平手（ImageNet精度为76.2），但与现在最好的模型（VIT-H/14，MAE等精度可以上90）还存在十几个点的差距。预测大概还需要当前1000倍的规模才可以弥补上十几个点的这个差距，现有的硬件条件也无法完成。所以扩大数据规模是不行了，需要在数据计算和高效性上需要进一步提高。

2. 难以理解抽象/复杂概念
CLIP在一些更抽象或更复杂的任务上zero-shot表现并不好。例如数一数图片中有多少个物体，或者在监控视频里区分当前这一帧是异常还是非异常，因为CLIP无法理解什么是异常、安全。所以在很多情况下，CLIP都不行。

3. out-of-distribution泛化差
对于自然图像的分布偏移，CLIP还是相对稳健的。但如果在做推理时，数据和训练时的数据相差太远（out-of-distribution），CLIP泛化会很差。例如CLIP在MNIST数据集上精度只有88%，随便一个分类器都都能做到99%，可见CLIP还是很脆弱的。（作者研究发现，4亿个样本没有和MNIST很像的样本）

4. 虽然CLIP可以做zero-shot的分类任务，但它还是从给定的那些类别里去做选择，无法直接生成图像的标题。作者说以后可以将对比学习目标函数和生成式目标函数结合，使模型同时具有对比学习的高效性和生成式学习的灵活性。

5. 数据的利用不够高效
在本文的训练过程中，4亿个样本跑了32个epoch，这相当于过了128亿张图片。可以考虑使用数据增强、自监督、伪标签等方式减少数据用量。

6. 引入偏见
本文在研发CLIP时一直用ImageNet测试集做指导，还多次使用那27个数据集进行测试，所以是调了很多参数才定下来网络结构和超参数。这并非真正的zero-shot，而且无形中引入了偏见。

7. 社会偏见
OpenAI自建的数据集没有清洗，因为是从网上爬取的，没有经过过滤和审查，训练的CLIP模型很有可能带有一些社会偏见，例如性别、肤色。

8. 需要提高few-shot的性能
很多复杂的任务或概念无法用文本准确描述，这时就需要提供给模型一些训练样本。但当给CLIP提供少量训练样本时，结果反而不如直接用zero-shot。例如3.1.4中CLIP的few-shot分类。后续工作考虑如何提高few-shot的性能
### 1.7 demo
下面是复制自CLIP官网的一段代码，使用红包图片（不在ImageNet那1000个类里面）进行一下简单的测试：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2804f374308a197eafdd3139c1e4a87a.png#pic_center =400x)
```python
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # 加载base模型

image = preprocess(Image.open("red_envelope.png")).unsqueeze(0).to(device)
text = clip.tokenize(["plane", "dog", "a cat","bird"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text) # 计算图文特征相似性
    probs = logits_per_image.softmax(dim=-1).cpu().numpy() # 和图片最相似的文本就是图片的类别

print("Label probs:", probs) 

Label probs: [[0.3131486  0.3174914  0.08763372 0.28172636]]
```
&#8195;&#8195;可以看到对于text给出完全不相关的类别，模型很困惑。
&#8195;&#8195;下面添加红包这个类别，模型做出正确的检测。

```python
text = clip.tokenize(["plane", "dog", "a cat","bird","red_envelope"]).to(device)

Label probs: [[0.00437422 0.00443489 0.00122411 0.0039353  0.98603153]]
```
&#8195;&#8195;下面我们实验一下，模型到底是通过什么方式知道这张图片是红包呢？是颜色还是信封（envelogp）？下面添加这两个类：（不知道为啥，和老师演示的不一样）

```python
text = clip.tokenize(["plane", "red", "envelope","bird","red_envelope"]).to(device)

Label probs: [[0.00259908 0.39436376 0.01481757 0.00233828 0.5858813 ]]
```
&#8195;&#8195;最后看看CLIP能不能学到相关的语义。红包一般是中国特有的，新年的时候都会在里面塞钱。下面改成这几个词试试：

```python
text = clip.tokenize(["money", "new year","red","envelope","china"]).to(device)

Label probs: [[0.01408994 0.015231   0.05491581 0.00206337 0.91369987]]
```
&#8195;&#8195;可以看到模型没有选择红色或者是信封，而是选择了和红包紧密结合的china这个概念，可见模型的相关语义还是学的很好。但是光从分类角度，是否应该分类为信封呢？
##  二、CLIP语义分割
>参考：
>- 李沐论文精度系列之[《CLIP 改进工作串讲》](https://www.bilibili.com/video/BV1FV4y1p7Lm/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
>- 博文[《CLIP改进工作串讲（上）》](https://blog.csdn.net/weixin_44966641/article/details/127340680)

&#8195;&#8195;`CLIP`是OpenAI于2021年2月发表的，在过去的一年多时间里，已经被用到各个方面：
- 语义分割：Lseg、GroupViT
- 目标检测：ViLD、GLIP v1/v2
- 视频理解：VideoCLIP、CLIP4clip、ActionCLIP
- 图像生成：VQGAN-CLIP、CLIPasso、CLIP Draw
- 多模态：VL Downstream
- 其他：depthCLIP、pointCLIP、audioCLIP（语音）、CLIPasso

### 2.1 LSeg
>论文：[《Language-driven Semantic Segmentation》](https://paperswithcode.com/paper/language-driven-semantic-segmentation-1)、[官网代码](https://github.com/isl-org/lang-seg)


&#8195;&#8195;语义分割可以看做是像素级的分类，因此分类的新技术、新思路，一般可以直接用过来。`LSeg`是2022年1月10号发表在ICLR的文章。与 `CLIP` 实现 zero-shot 的方式类似，`LSeg`通过类别 prompt 作为文本输入，然后计算相似度，也实现了zero-shot 语义分割。

&#8195;&#8195;`LSeg`的意义在于将文本的分支加入到传统的有监督分割的pipeline模型中，通过矩阵相乘将文本和图像结合起来了。训练时可以学到language aware（语言文本意识）的视觉特征，从而在最后推理的时候能使用文本prompt得到任意你想要的分割结果。

#### 2.1.1 模型效果
&#8195;&#8195;下图展示了LSeg的检测效果。给定一张图片，然后通过文本提示，给出需要检测的类别，就可以实现对应的语义分割。
![LSeg的zero-shot分割结果](https://i-blog.csdnimg.cn/blog_migrate/b9d143faca76ab8186ec2c6be9b53bdb.png#pic_center )
- 第一张图中，给出`dog,tree,others`，就可以把狗和树检测出来，其它为背景色
- 为了验证模型的容错能力，加一个汽车`vehicle`的标签，模型中也并没有出现汽车的轮廓
- 模型也能区分子类父类，标签中不再给出`dog`而是给出`pet`，`dog`的轮廓同样可以被分割开来
- 第三张图中，椅子、墙壁甚至地板和天花板这种极为相似的目标也被完美的分割开来

>&#8195;&#8195;值得一提的是，由于 CLIP 类的模型实质上都是通过计算图文相似度来实现分类或分割的，因此 ‘other’ 类的类别 prompt 文本实际可以是任何无意义的文本，如 ‘me’，‘a’，‘an’ 等，只要与目标类别不要太接近即可。

#### 2.1.2 模型框架
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/99a157a02d781f72dc8901427f9659b8.png#pic_center )
&#8195;&#8195;如上图 4 所示，模型整体看来与 `CLIP` 模型非常相似，只是将单个的图像文本特征换成语义分割中逐像素的密集特征。
&#8195;&#8195;另外除了上方的文本编码器提取的文本特征，要与密集图像特征相乘来计算像素级的图文相似度之外，整个网络与传统的有监督网络完全一致。

- 文本编码器提取$N\times C$的文本特征，图像编码器提取  $\tilde{H}\times\tilde{W}\times C$  的密集图像特征，二者相乘得到 $\tilde{H}\times\tilde{W}\times N$ 的特征，再经过 `Spatial Regularization Blocks` 上采样回原图尺寸。最后计算模型的输出与ground truth监督信号的交叉熵损失进行训练。
- $N,C,\tilde{H},\tilde{W}$ 分别是类别 个数（可变）、通道数和特征图的高宽，C一般取512或者768。
- `Text Encoder`： 直接用的是CLIP 文本编码器的模型和权重，并且训练、推理全程中都是冻结的。因为分割任务的数据集都比较小（10-20万），训练的话结果会不好。
- `Image Encoder`：[DPT](https://paperswithcode.com/paper/vision-transformers-for-dense-prediction)结构（使用了ViT进行有监督训练的语义分割模型，结构就是ViT+decoder），backbone可以是`ResNet`或者`ViT`。如果使用后者，其参数用的是`Vit/DEit`的预训练权重，直接使用CLIP的预训练权重效果不太好。
- `Spatial Regularization Blocks` 是本文提出的一个模块。在计算完像素级图文相似度后继续学习一些参数，可以进一步学习文本图像融合后的特征。模块由一些卷积和DW卷积组成（当加了两个模块时效果提升，加了四个模块后效果崩溃，作者未解释）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7820a707ca87f7ce8af16e1dfbe6b19d.png#pic_center =500x)

&#8195;&#8195;模型在 7 个分割数据集上进行训练，这些数据集都是由有标注的分割图组成，所以模型是以有监督的方式进行训练的（损失函数是交叉熵损失而非无监督的对比学习目标函数）。推理时，可以指定任意个数、任意内容的类别 prompt 来进行 zero-shot 的语义分割。

####    2.1.3 实验结果
&#8195;&#8195;作者将PascalVOC数据集和COCO数据集，都按照类别分成四份。比如COCO有80个类，前20类作为当前已知类，后60类为未知类，然后就可以做zero-shot 和few-shot 了。
&#8195;&#8195;对比zero-shot推理的话，LSeg的效果确实好很多；但是与 few-shot 哪怕是 one-shot 相比，还是有很大的距离。再考虑到LSeg用的是ViT结构，可见LSeg需要提升的空间还是非常大的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2d7bdc34a51e52c0d0d0c91dda64bdf0.png#pic_center =500x)
**Failure cases**
&#8195;&#8195;比如下图左侧，标签给定是`toy, grass`，在嵌入空间中（embedding space），狗的视觉特征明显更接近“玩具”而不是“草”，并且没有其他标签可以解释视觉特征，所以狗被检测为toy。如果标签是`face,grass`，狗会被检测为face。
&#8195;&#8195;<font color='deeppink'>也就是所有使用CLIP模型的工作，都是在计算图像和文本之间的特征相似度，谁相似就选谁，而不是真的在做分类。</font>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b9e2dd54f5b98983c44067feb61885b6.png#pic_center )

### 2.2 GroupViT
>- 论文[《GroupViT: Semantic Segmentation Emerges from Text Supervision》](https://paperswithcode.com/paper/groupvit-semantic-segmentation-emerges-from)、[官网代码](https://github.com/NVlabs/GroupViT)、[huggingface/transformers](https://github.com/huggingface/transformers/tree/803475fb69097e802985eb4f85b52199c66a52de/src/transformers/models/groupvit)
>- 博文[《Group ViT》](https://blog.csdn.net/weixin_45104951/article/details/127074187)、 博文[《CLIP改进工作串讲（上）》](https://blog.csdn.net/weixin_44966641/article/details/127340680)
#### 2.2.1 前言
&#8195;&#8195;上一节讲的`LSeg` 虽然能够实现 zero-shot 的语义分割，但是训练方式并不是对比学习（无监督训练），没有将文本作为监督信号来使用。因此`LSeg`还是需要手工标注的分割掩模（`segmentation mask`）进行训练。其使用的 7 个数据集加起来可能也就一二十万个样本，跟别的有监督无监督训练是远远不能比的。

&#8195;&#8195;`GroupViT`是2022年2月22发表在CVPR的文章。从标题可以看出，其监督信号来自文本而非segmentation mask。`GroupViT`通过文本自监督的方式进行训练，从而实现简单的分割任务（不再依赖segmentation mask）。

#### 2.2.2 模型结构
&#8195;&#8195;`GroupViT` 的核心思想，是利用了之前无监督分割工作中的的 `grouping`。简单说如果有一些聚类的中心点，从这些中心点开始发散，把周围相似的点逐渐扩散成一个group，最后这个group即相当于一个Segmentation mask（感觉类似DBSCAN）。
&#8195;&#8195;`Group ViT`的贡献就是在现有的ViT模型中加入计算单元`Grouping Block`，同时加入了可学习的`Group Tokens`。这样模型在初期学习的时候就能慢慢一点点的将相邻的元素group起来，最后变成一个个`segmentation mask`。
>&#8195;&#8195;比如下图，浅层的时候学习到的还是一些五颜六色的块，到深层大象、房子、草地等都已经分割出来了

&#8195;&#8195;下面来看一下 GroupViT 模型框架和具体的训练过程：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7dbca5db95c41d8a97fc5d874c4a5021.png#pic_center )
- `Image Encoder`：
	- 结构就是Vision Transformer，一共是12层 Transformer  Layer，其输入除了原始图像的Pacth embeddings，还有可学习的 group token 。
	- 假设输入图像尺寸是224,×224，选择的`Image Encoder`是ViT-Small/16，则输出Pacth embeddings尺寸是196×384，也就是图中的token  $\mathbf{s}_i^1$ 。对应group token $\mathbf{g}_i^1$的尺寸是64×384。
	- 这里的 group tokens 就相当于分类任务中的 cls token。然后通过Transformer  Layer的自注意力来学习到底哪些patch属于哪些group token。
>由于分类任务一张图像只需要一个全图的特征，因此只用一个 token 即可。而语义分割中一张图有多个目标，所以需要多个特征，也就是多个 group tokens。最初是选择64个 group tokens（聚类中心），不大不小，后期可以合并。

- 训练：
	- 经过六层 Transformer Layers 的之后，学的差不多了。加入一个`Grouping Block` 来完成 grouping ，将图像块 token 分配到各个 group token 上，合并成为更大的、更具有高层语义信息的 group，即Segment Token（维度64×384，相当于一次聚类的分配）。
	- `Grouping Block`结构如上图右侧所示，其grouping做法与自注意力机制类似。计算 grouping token （64×384）与图像块 token （196×384）的相似度矩阵（64×196），将token分配到相似度最大的grouping token上面（聚类中心的分配）。这里为了克服 argmax 的不可导性，使用了 可导的gumbel softmax。合并完成后得到$\mathbf{s}_i^2$（64×384） 。
	- 重复上述过程：添加新的 Group tokens $\mathbf{g}_i^2$（8×384），经过 3 层 Transformer Layers 的学习之后，再次经过grouping block 分配，得到 $\mathbf{s}_i^3$（8×384） 。
	- 为了和文本特征进行对比学习，将最后一层Transformer Layers输出的序列特征（8×384）进行全局平均池化Avg Pooling，得到1×384的图片特征。再经过一个MLP层变成$\mathbf{z}^I$维的图片特征。最后与文本特征$\mathbf{z}^T$ 计算对比损失。

- 推理
文本和图像分别经过各自的编码器得到文本特征和图像特征，然后计算相似度，得到最匹配的图像文本对，就可以知道每个group embedding对应什么class。局限性在于最后的聚类中心（Group Tokens）只有8类，所以一张图像中最多分割出八个目标。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/60d92f2a72c3ed23b6ce9e28eada3486.png#pic_center =500x)

总结：GroupViT 没有在ViT基础上加很复杂的模块，目标函数也和CLIP保护一致，所以其scale性能很好。即更大模型更多数据，其性能会更好。

其他细节：
- 论文中选用的是ViT-Small，数据集是2900万图文对。
- 除了与图形配对的文本本身，还将文本中的名词提取出来，按照类似 CLIP 中的方法生成 prompt （如 “A photo of a {tree}.”），与某个图像特征计算对比损失，见原文 Figure 3；
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8947e431ad73ed19a2dc54cd9114f43f.png#pic_center =500x)

- 消融实验中Group tokens个数选择64,8的组合效果最好
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b8c4acf88078fe6c08bda771867b87d2.png#pic_center =500x)
#### 2.2.3 Group tokens可视化
&#8195;&#8195;为了验证加入的`Group tokens`、 `Grouping Block`到底有没有工作，`Group tokens`有没有成为聚类中心，又是否对应了某一个类别；作者将不同阶段不同 `group token` 对应的注意力区域进行可视化，结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bd9dd18786f9f7acd1666f7d41932da5.png#pic_center =500x)
- 在第一阶段，每个 token 都注意到一些语义明确的区域，如group5表示的是眼睛，group36表示的是四肢，并且都是些相对较小的区域；
- 在第二阶段，每个 token 注意到的语义区域则相对较大，如脸、身体。这正符合了作者想要的 group 分组合并的效果。

#### 2.2.4 实验
1. **Comparison with Zero-Shot Baselines**
下表是和其它的一些 Zero-Shot 推理模型效果对比：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aaa498ddbadc6d42c8a00e0c8e08ca78.png#pic_center =500x)
2.  **Comparison with Fully-Supervised Transfer**
- 在PASCAL VOC 2012数据集上 ，Zero-Shot GroupViT（无微调）优于所有自监督预训练的ViT变体（有监督的微调）
- 在 PASCAL Context数据集上，Zero-Shot GroupViT的效果也和它们相当。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/76ad608b8e2589540b671e9ca80fcc19.png#pic_center )
&#8195;&#8195;GroupViT 是第一个实现 zero-shot 语义分割的工作，相较于其他自监督语义分割方法提升显著。但是跟有监督训练的模型上限比起来，还是差了非常多。
>&#8195;&#8195;`DeepLabv3+`(Xception-65-JFT) 在`PASCAL VOC`上 `Mean IoU`已经达到了`89`，	
`ViT-Adapter-L`(Mask2Former, BEiT pretrain)在`PASCAL Context`上 `Mean IoU`达到了68.2。

#### 2.2.5 局限性
现在的无监督语义分割还是很难做的，作者也列出了GroupViT 的两个局限性：
-  GroupViT更偏向于是一个图片编码器，没有使用dense prediction的特性，如空洞卷积、金字塔池化以及U-Net的结构，从而获取更多的上下文信息和多多尺度信息；
- 背景类干扰问题难以处理。
推理过程中，最大相似度也可能很低，比如0.2；为了提高前景类的分割性能，作者设定了相似度阈值，但是因为背景类干扰问题，这个阈值很难设定好。
>&#8195;&#8195;比如`PASCAL VOC`中相似度阈值是`0.9`或`0.95`，图片和文本的相似度取最大值且最大值大于`0.9`时，才会认为物体是这个类；否则认为是背景类。
>&#8195;&#8195;`PASCAL VOC`中类别少，且物体都有明确的语义，所以背景类干扰少；但是`PASCAL Context`或者`COCO`数据集，类别很多，前景的相似度一般都很低，和背景类差别不大。如果阈值设的高，很多物体都会被检测为背景类；如果设的低，容易误分类，即相似度最高的那一类，并不是真实类别。

&#8195;&#8195;因为背景类干扰问题，作者发现`Group tokens`其实已经学的很好了，但是最后分割时容易分类错了。为了验证这一点，作者做了 oracle 对比的实验。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c2b5505e47f136057129835d24ac966b.png#pic_center )
&#8195;&#8195;`oracle mask mIoU`：模型在完成分割之后，不用模型预测的类别结果，而是将每个 mask 与 GT mask 计算 IoU，直接取最大的类别标签作为类别结果。这相当于，模型只要分割的准就行了，语义类别的预测是肯定准的。
&#8195;&#8195;上图 可以看到，`oracle` 结果相比于原结果有多达二三十个点的巨大提升；这说明，语义类别预测错误是 `GroupViT` 模型的一大瓶颈。

&#8195;&#8195;**结论**： `GroupViT` 图片分割做得好（segmentation mask生成的好），但是语义分割做的不够好，这是由于CLIP这种对比学习的训练方式，对于明确语义物体信息能学的很好；但是对于背景这种语义比较模糊类别很难识别，因为背景可以代表很多类。后续改进可以是每个类设置不同阈值，或者使用可学习的阈值，或者是更改 Zero-Shot 推理过程、训练时加入约束，融入背景类概念等等。

### 2.3  总结
&#8195;&#8195;`Lseg`使用CLIP的预训练模型和大概框架，融合了文本和图片特征去做分割，但依旧是一个有监督的学习过程，还是需要手工标注的数据集；`GroupViT` 从头训练了一个分割模型，但是用的目标函数和`CLIP`的对比学习目标函数一样，局限之一就是背景类处理得不够好。

##  三、CLIP目标检测  
> 李沐论文精度系列之[《CLIP 改进工作串讲》](https://www.bilibili.com/video/BV1FV4y1p7Lm/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)
### 3.1 ViLD
>论文[《Open-vocabulary Object Detection via Vision and Language Knowledge Distillation》](https://paperswithcode.com/paper/zero-shot-detection-via-vision-and-language)、[tensorflow代码](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
#### 3.1.1 简介
&#8195;&#8195;ViLD是21年4月28上传到Arxiv上的，也就是CLIP发表仅仅两个月之后，而且训练了约460epoch，所以速度是很快的。`ViLD`即Vision and Knowledge Language  Distillation，即用CLIP当做teacher蒸馏网络，从而能达到Zero-Shot检测。简单来说，ViLD 想要做到的事情是：<font color='deeppink'>在训练时只需要训练基础类，然后通过知识蒸馏从 CLIP 模型中学习，从而**在推理时能够检测到任意的新的物体类别（`Open-vocabulary Object`）**</font>。

&#8195;&#8195;下面的例子中，如果用传统的目标检测算法的话，模型只会判断这些物体都是玩具，也就是图中蓝色的基础类，而无法检测到更细致的类别。使用CLIP之后，在现有检测框上，不需要额外标注，就可以检测出新的类（图中红色标识类）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2bc127135480f38c7a873d2d70397859.png#pic_center )
#### 3.1.2 模型结构
&#8195;&#8195;ViLD 方法的研究重点在两阶段目标检测方法的第二阶段，即得到提议框（proposal）之后。其思想还是最简单的分别抽取文本和图片特征，然后通过点积计算相似度。其模型结构如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/416a8b0e72356b49c9d56669dcd815ac.png#pic_center )
- (a)：Mask R-CNN框架。
一阶段得到的候选框`proposals`经过检测头得到 `region embeddings`，然后经过分类头得到预测的`bounding box`以及对应的类别。损失分为定位损失（回归损失）和分类损失。
- (b) ：`ViLD-text` 分支
	- 	N个`proposals`经过一些处理得到类似图a中的N个 `region embeddings`（图片特征）。
	- 将物体类别（基础类）处理为prompt 句子就得到了文本，然后将这些文本扔给文本编码器得到Text Embeddings（文本特征）。和Lseg类似，这些Text Embeddings也是冻住权重的，不参与训练。
	- 上面物体类别就是 `Base categories`（也叫CB，Class Base），和Mask R-CNN有监督训练的基础类一样，所以`ViLD-text` 做的还是有监督训练。
	- 因为是有监督训练，所以需要额外添加一个背景类进行训练，即可学习的Background  embedding（基础类之外的类别全部归为背景类）。
	- Text Embeddings加上分别和可学习的背景 embedding以及 `region embeddings`进行点积来计算图文相似度得到logics，然后计算logics和GT的交叉熵损失来进行训练。
	 - 在 `ViLD-text` 模型中，只是将文本特征和图像特征做了关联（感觉到这里只是类似Lseg），模型可以做文本查询的 zero-shot 检测。但是由于模型还不了解基础类CB之外的其他语义内容（X新类别CN），因此直接做 zero-shot 的效果不会很好。
>- `ViLD-text` 点积计算公式：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1c2042e854167afc8d5e4e4a0252a7b2.png#pic_center =500x)
>- `I`表示图片，`φ(I)`表示抽取图片特征，`r`为`proposals`。`φ(I)`和`r`一起进行`R`计算得到 **$e_r$**（`region embeddings`，图片特征）
>- $e_{bg}$表示背景embedding，t_1到$t_{\left | CB \right |}$表示基础类CB的文本特征`Text Embeddings`。
>- 图像特征$e_r$分别和背景特征$e_{bg}$和文本特征做点积计算相似度，最后得到 `ViLD-text` 模型输出`z(r)`（logics）。
>- `z(r)`做softmax后和`groud truth`计算交叉熵得到这部分的损失。
>- `Projection` 层的引入是为了统一图像文本特征的尺寸。


- (/c) ：`ViLD-image`分支：引入CLIP特性，这部分只在训练时进行蒸馏，推理时不蒸馏。
考虑到 `CLIP`的图片编码器训练的很好，而且和文本紧密关联，所以希望`ViLD-image-encoder`输出的`region embeddings`和CLIP输出的`image embedding`尽可能的接近，这样就可以学习到CLIP图像编码器中开放世界的图像特征提取能力。做到这一点的最简单方式就是知识蒸馏（Knowledge Distillation）。
	- 右侧`teacher`分支：将M个`proposals` resize到比如224×224的尺寸，然后输入预训练好的`CLIP-image-encoder`（冻结，不参与训练，保证抽出来的特征和CLIP一样好）得到`M image embeddings`
	- 左侧`student`分支：和`ViLD-text` 分支前面的结构一样，输入M个`proposals` 得到M个`region embeddings`
	- 计算`region embeddings`和 `image embeddings`的L1 loss来进行知识蒸馏，让检测器学习 CLIP 提取的特征。
	- 为了加速模型训练，在训练 ViLD-image 时先用 CLIP 模型提取好图像区域特征，保存在硬盘中，在训练时直接从硬盘读取即可。
	- <font color='red'> 此分支监督信号不再是人工标注的数据，而是CLIP的图像编码，所以就不再受基础类CB的限制了，对于任何的语义区域都可以由 CLIP 抽取图像特征。利用`ViLD-image`，大大加强了做Open-vocabulary检测的能力 </font>
>`ViLD-image`分支的弊端：预加载训练好的proposals，而不是随时可以变的`N proposals`
>- 此分支输入是`M pre-complete proposals`，这是为了训练加速。
> 
>- 理论上第一阶段输出的`N proposals`应该输入text和image两个分支进行训练，但如果每次训练时再去抽取CLIP特征就太慢了。因为`ViLD`选用的CLIP-L模型非常大，做一次前向过程非常贵。比如M=1000时等于每一次迭代都需要前向1000次才能得到所有图像特征，这样训练时间会拉到无限长。
> 
>- 作者在这里的做法就是在`ViLD-image`开始训练之前，利用RPN网络预先抽取`M pre-complete proposals`，然后按照图中顺序算好 `M image embeddings`。`ViLD-image`训练时，只需要将其load进来，这样loss算起来就很快，蒸馏过程也就训练的很快。

- (d) ：ViLD-text 和 ViLD-image的合体
	为了训练简单，将`M pre-complete proposals`和`N proposals`一起输入检测头Head得到n+m个embedding，然后拆分为`N region embeddings`和`M region embeddings`。前者算`ViLD-text` 分支的交叉熵损失，后者算`ViLD-image`的蒸馏L1损失。

#### 3.1.3 模型总览
下面快速过一下模型总览图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0282ef327a29bdd5951102e4a94ed135.png#pic_center )
- 训练：
	- 图片通过RPN网络得到`proposals`。然后经过RoIAlign和一些卷积层得到`N region embeddings`，也就是图中$R_1$和$R_2$。
	- 基础类通过prompt得到文本，经过文本编码器得到文本编码$B_1$到$B_n$。然后和$R_1$、$R_2$一起计算交叉熵。
	- 将已经抽取好的`M image embeddings`（图中的骰子、停车标识等等）输入CLIP图像编码器得到特征$I_1$、$I_2$，用它们对$R_1$、$R_2$进行蒸馏（计算L1损失）

- 推理：
	- $image\overset{backbone+RPN}{\rightarrow}proposals\overset{RoIAlign+Conv}{\rightarrow}region-embeddings$
	- $CN+CB\overset{prompt}{\rightarrow}Text\overset{Text-Encoder}{\rightarrow}text-embedding(B_1..B_n+N_{1}...N_{k})$
	- $class=argmax(region-embeddings\cdot text-embedding)$
#### 3.1.4 实验
1. LVis 数据集 zero-shot 效果对比

&#8195;&#8195;LVis 数据集图片采样自COCO数据集，但却是一个十分长尾的数据集。在标注了的1203 类中，有很多类只标注了几次，即罕见类，所以将各个类别分为 fequent、common、rare，标注数依次递减，也就是下图中的$AP_f,AP_c,AP_r$。
&#8195;&#8195;在本文的实验中，将 $AP_f,AP_c$作为模型见过的基础类（共 886 类），$AP_r$ 作为新类（共 337 类，模型没有见过，所以可以做zero-shot检测）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4842beb5496ec30c3fd4e210a8da9fbc.png#pic_center )

&#8195;&#8195;可以看到 `ViLD` 模型在新类上的 AP 大幅领先基线模型`Supervised-RFS`（RFS是尽量采样尾部的类，用于解决长尾问题，所以这是一个很强的极限模型），并且是做的zero-shot检测。但这是可以预见的，因为对于有监督训练，只有一两个样本可能越训练越差，还不如直接zero-shot。

2.  其它数据集zero-shot效果
下图是LVis 数据集上预训练的`ViLD`在`PASCAL 	VOC` 和`COCO`数据集上zero-shot迁移效果，对比有监督训练的模型还是有一些差距。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b41b876ca5c6ea4d89b8899ed7040aa2.png#pic_center )

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ae7837cb555175070c4aa03c2ce6c422.png#pic_center )

- 第一行：ViLD能够正确定位，并识别新的类别。为了清楚起见，我们只显示检测到的新类。
- 第二行：ViLD同时检测基础类和新类，并不会降低基本类别的检测性能。
- 后两行：ViLD可以直接迁移COCO和Objects365，无需进一步微调。

#### 3.1.5 结论
&#8195;&#8195;ViLD是第一个在LVis这么难的数据集上做Open-vocabulary目标检测的模型，是一个里程碑式的工作。ViLD借鉴了CLIP的思想，也借鉴了CLIP的预训练参数，最后的结果也不错。

### 3.2 GLIP v1
>- 论文[《Grounded Language-Image Pre-training》](https://paperswithcode.com/paper/grounded-language-image-pre-training)、[官网代码](https://github.com/microsoft/GLIP)
>- `GLIP`光听名字和`CLIP`很像，只是把`Contrastive Language-Image Pre-training`（基于对比文本-图像对的预训练方法）中的`Contrastive`换成了`Grounded`。
#### 3.2.1 前言

**1. 研究动机：`Open-vocabulary Object Detection`是必要的**

&#8195;&#8195;目标检测和分割一样，标注数据集都很贵，对于边边角角的类和层出不穷的新类，我们没有办法训练一个模型把这些都检测的很好。我们只能依赖于`Open-vocabulary`的目标检测模型，来把这些corner case都处理的很好。
&#8195;&#8195;而如果想训练一个很强的`Open-vocabulary`的目标检测模型，就只能像`CLIP`一样，可以利用上亿规模的的数据集，而且还要把图片-文本对应关系和定位都学的很好。那么<font color='red'> 重点就是使用图片-文本对数据集的高效使用 ，因为很好收集</font>。

**2. 解决办法：`phrase grounding+Object Detection`+伪标签训练**
&#8195;&#8195;Vision Language任务（图片-文本多模态任务）里有一类定位任务`Vision grounding`，主要就是根据文本定位出图片中对应的物体（短语定位phrase grounding），这与目标检测任务非常类似，都是去图中找目标物体的位置。

&#8195;&#8195;`GLIP` 的文章的出发点，就是将检测问题转换为短语定位（phrase grounding）问题，这样GLIP 模型就统一了目标检测和定位两个任务，可以使用更多的数据集。再配合伪标签的技术来扩增数据，使得训练的数据量达到了前所未有的规模（3M人工标注数据和24M图文对数据）。最后训练出来的模型`GLIP-L`，直接以 `zero-shot` 的方式在`COCO` 和`LVIS` 上进行推理，mAP分别达到了 `49.8` 和`26.9`，可见其性能非常的强。
>- `groudning`模型的输入是短语、短语中名词对应的框和图片。
>- 目标检测转为`phrase grounding`：通过`prompt`的方式将标签名转化为短语。例如coco有80个类别标签，将80个标签用逗号连接，并在短语前加“`Detect：`”来组成短句。这样做有两个好处：
	>	- 目标检测和`phrase grounding`的数据集就都可以拿来训练
	>	- 对于基础类 和其它各种类，可以都构建到 prompt 短语中一起检测，更加的灵活，可以方便的将任务迁移到开放式目标检测任务当中。
>- 伪标签训练（self training）：
	>	- 将所有目标检测任务和`phrase grounding`任务的数据集（一共`3M`）全部拿来做有监督训练，得到`GLIP-T(C)`模型。
	>	- 将这个模型对网上爬取到的`24M`图像-文本对数据进行推理，得到`bounding box`。然后将这些bounding box全部作为`GroundTruth`（伪标签），这样就得到了24M的有监督数据。
	>	- 最在这`24M`的有监督数据上继续训练，得到最终模型`GLIP-L`。由此可见整个`GLIP`都是用有监督的方法进行训练的。

**3. `zero-shot`推理效果展示**
&#8195;&#8195;直接像ViLD一样给出物体类别生成一句话（Prompt : person. bicycle.car. motorcycle…）或者是像phrase grounding任务一样生成短语“马路上有很多坑”（Prompt : there are some holes on the road），都可以将物体检测出来。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/41f918192a3877d627a62463525ddc71.png#pic_center )
#### 3.2.2 损失计算

&#8195;&#8195;目标检测的损失函数由分类损失和定位损失组成。对于目标检测和`Vision grounding`而言，定位部分都差不多，二者的区别主要在于如何计算分类loss。因为 `detection`的标签是one-hot的类别单词，而`Vision grounding`的标签是一个句子。所以需要把二者的分类loss统一到一个框架下面，也就是：
$$L = L _{cls}+ L_{loc} .$$

- `detection`分类损失计算：
$$O=Enc_{I}(Img),S_{cls}=OW^{T},L_{cls}=loss(S_{cls};T).$$
	1. $Enc_{I}$表示图片编码器（例如swin transformer），处理img之后得到`N region embeddings`，即$O\in \mathbb{R}^{N\times d}$（n个bounding box，每一个的维度是d）；
	2.  `N region embeddings`$\overset{cls-Head}{\rightarrow}S_{cls}$。其中分类头cls-Head由矩阵$W\in \mathbb{c}^{N\times d}$表示，和`N region embeddings`相乘后得到$S_{cls}\in \mathbb{R}^{N\times c}$；
	3. $L_{cls}=loss(S_{cls};T).$：使用nms筛选这些bounding box，然后和GroundTruth计算交叉熵，得到分类损失。

- `Vision grounding`分类损失计算：（其实和`ViLD text`分支一模一样）
$$O=Enc_{I}(Img),P=Enc_{L}(Prompt),S_{ground}=OP^{T}$$
1.  表示图片编码器$Enc_{I}$处理img之后得到`N region embeddings`，即$O\in \mathbb{R}^{N\times d}$；
2. 文本编码器$Enc_{L}$（比如BERT）处理Prompt得到`text embedding`，即$P\in \mathbb{R}^{M\times d}$；
3. 图像特征O和文本特征P相乘得到相似度结果$S_{ground}\in \mathbb{R}^{N\times M}$，即论文中说的region-word aligment scores。


上式中，M（(sub-word tokens数量）总是大于短语数c，原因有四：
- 一个短语总是包含很多单词
- 一个单词可以分成几个子词，比如toothbrush分成了 tooth#, #brush
- 还有一些添加词added tokens，像是“Detect:”，逗号等，或者是语言模型中的特殊token
-  tokenized序列末尾会加入token `[NoObj]`

&#8195;&#8195;在训练的时候，如果短语`phrase`都是正例（ positive match）并且`added tokens`都是负例negative match（added tokens和任何图片的物体都无法匹配），那就使用subwords（subwords也都是正例，此时标签矩阵由$T\in [0,1]^{N\times  c}$扩展为$T\in [0,1]^{N\times  M}$）。测试时多个token的平均pro作为短语的probability。

&#8195;&#8195;使用上面的方式统一损失之后，就可以用grounding模型方法来预训练检测任务，从而使`GLIP`模型可以做zero-shot检测。之后作者使用统一过后的框架验证了在 COCO 数据集上的指标，发现是完全匹配的 ，因此从实验上也验证了自己的想法。

#### 3.2.3 训练数据集
- 上面三行A,B,C展示的是`GLIP`模型可以同时使用目标检测的数据集，例如`Objects365`和`Grounding`的数据集`GoldG`（几个数据集的合并，还是很大的）。
- `GLIP-L`：backbone为Swin-L模型，然后同时使用`FourODs`（目标检测有监督训练中能用的所有的数据集）、`GoldG`和图片文本对`Cap24M`数据集一起训练，此时数据集已经非常大了，足以训练出一个很强的模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1055d20896e59f9f6d7918741f0e01c2.png#pic_center )
>&#8195;&#8195;`Cap24M`就是`24M`伪标签数据。生成的伪标签肯定有错误，但是实验表明，经过扩充大量伪标签数据训练得到的 GLIP-L 模型仍然会有性能提高。

#### 3.2.4 模型框架
**1. 总体框架**

&#8195;&#8195;如下图所示，由于所有数据集都是有标注的，所以模型是以有监督的方式进行训练。计算得到文本特征与图像特征的相似度之后，直接与 GT box计算对齐损失alignment loss即可（和ViLD-text分支一样）。这样就完成了文本和图像的特征融合，就可以进行zero-shot检测了。而定位损失也是直接与GT box计算L1 损失。

&#8195;&#8195;模型中间的融合层（`Deep Fusion`）和LSeg的做法一样，都是为了使图像特征和文本特征进一步交互，使最终的图像-文本联合特征空间（joined embedding space）训练得更好（相似的embedding拉近，不相似的拉远），图像特征和文本特征被训练的更强更有关联性，这样后面计算相似度矩阵的效果肯定就更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27cd74aaf7fd0a24310b39bee7c355a8.png#pic_center )

**2. `Deep Fusion`层**
- 图片编码器是`DyHead`（L层），第一层输出图片特征表示为$O^0$
- 文本编码器是预训练好的`BERT`（L层），第一层输出文本特征表示为$P^0$
- 	`X-MHA`表示跨模态多头注意力模块。
- 从结构图和公式可以看出，每一层输出的图文特征$O^i,P^i$都会在`X-MHA`中进行交互，交互后的特征和原特征相加之后一起输入到下一层进行编码，得到下一层的特征$O^{i+1},P^{i+1}$。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f91ac8919533fa25ebd793895dbe5b84.png#pic_center  =500x)
在`X-MHA`模块中，图像特征和文本特征交互使用的是`Cross Attention`：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a7c023ca2a2c7f049ed971984b8df1f2.png#pic_center  =500x)
>- 分割和检测都属于稠密性任务（dence prediction）的一种，都需要同时分类和定位，所以很多方法是可以互相借鉴的，所以`Deep Fusion`也可以用到分割领域，比如`GroupViT`。`GroupViT`只有在图像分支和文本分支的最后做了一下对比学习，如果在此之前做了一些`Deep Fusion`，可能效果更好。

**3. 推理展示**
上图右侧作者展示了两个非常难的任务：
- 检测两个针管和一瓶疫苗。现有的数据集中似乎没有针管和疫苗这种类别。但是GLIP自己做出来对文本的理解，给出了疫苗和针管的检测结果。
- 给出了一张图片的描述：“在古巴的playa esmeralda，从上往下俯瞰海滩，看到了漂亮的海绿色加勒比海”。这些描述都是一些比较抽象的概念，已经不太像是物体了，但是GLIP依旧做得很好。

#### 3.2.5 对比实验
1. COCO数据集结果对比。
可以看到`GLIP` 模型直接做 zero-shot 检测的mAP已经达到了 49.8 ，如果再在 COCO 上进行微调，`GLIP` 的 结果能够超过当前最好的一些有监督方法。当然`GLIP`和其它模型的预训练数据集规模和一些trick是不一样的，但也足以看出`GLIP`的强大之处。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3ce03ee524cbad8be45a4d73a3202dd6.png#pic_center )
2.  LVIS数据集结果对比![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c4edbc35c5e7cd4e2da605d5ee3a280e.png#pic_center )
### 3.3 GLIPv2
>论文[《GLIPv2: Unifying Localization and Vision-Language Understanding》](https://paperswithcode.com/paper/glipv2-unifying-localization-and-vision)、[代码](https://github.com/microsoft/GLIP)

#### 3.3.1简介
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aba241ded1fe909329dc5bbdc04d0354.png#pic_center )
&#8195;&#8195;GLIPv2和GLIPv1架构基本一样，只是融合了更多的任务和数据集。从论文题目 Unifying Localization and Vision-Language Understanding可以看出，其统一了所有的定位任务（比如分割和检测）和Vision-Language任务。
>Vision-Language：语言-视觉任务，包括：
>- `vision Caption`：图像描述生成，根据一张图片生成描述性文本；
>- ` VQA`：给定一张图片和一个与该图片相关的自然语言问题，计算机能产生一个正确的回答。文本QA即纯文本的回答，与之相比，VQA把材料换成了图片形式，所以这是一个典型的多模态问题；
>- `Vision grounding`：根据短语定位图片中对应的物体。

&#8195;&#8195;通过下图可以看到，比起GLIPv1，GLIPv2加了一些`text encoder`的训练任务，使其表征更加丰富。比如定位任务不光有目标检测还有实例分割，Understanding任务包含了`Vision grounding`、`vision Caption`和`VQA`任务。
&#8195;&#8195;然后就是图片特征和文本特征做`Deep Fusion`，后面就是一样的处理了。像这样在统一框架下囊括更多任务更多数据集更多模态也是当前的一种趋势，比如去年的OFA、今年的Unified-IO等等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6c3f7c54f21d34aef02d0af02ddc82f8.png#pic_center )
####  3.3.2损失函数
在GLIPv2 当中对损失函数做了改进，在原有`ground`损失的基础上加入两种损失：
**$${L_{GLIPv2}=\underset{L_{ground}}{\underbrace {L_{loc}+L_{intra}}}+L_{inter}+L_{mlm}}$$**

1. 添加`MLM` 损失：添加这一损失可以强化模型的语言特性。能够使得训练出来的模型能够扩展到 VQA / ImageCaption 任务上。

2. 图片间的对比学习损失$L_{inter}$。
原先的`image-text pair`，只能看到`pair`内部的信息。比如一对数据是一个人抱着猫的照片和对应的文本描述。按照原先的 loss 设计，图片中的`人`只能够做到和  ‘person’ 相似， 和 ‘cat” 不相似，但是没有办法和所有其它图片中各种各样的实体进行区分。所以在此考虑加入图片间的对比损失。

对比损失的计算方法：
- 对一个batch 当中所有的pair ，抽取其未交互的图片特征和文本特征$\overset{{\circ }}{O}=Enc_{V}(Img),\overset{{\circ }}{P}=Enc_{L}(Text)$
- 计算一个batch内，所有图片特征和文本特征的相似度$S_{ground}^{batch}[i,j]=\overset{{\circ }}{O}{_{}}^{i}(\overset{{\circ }}{P}{_{}}^{j})^{T}$，这样就可以通过跨图像匹配的方式，使得每一个object/token 都能够看到更多的负样本。所以我们不仅仅对图片和文字交互后的特征建模，也要对于图片和文本交互前的特征建模，类似loopiter。
- 跨样本匹配的时候，图片A 当中的‘人’这个物体，和图片B 对应的prompt 当中的 ‘person’类别，也应该是匹配的
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/510e36217dc15938283e267e26752499.png#pic_center )

#### 3.3.3 模型结构
模型总览图如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e08430b18b3f6748a05106d288e20e96.png#pic_center )
#### 3.3.4 模型效果
1. 我们将`GLIPv2`和下表中当前的目标检测和vision-language预训练模型，在8个下游任务上进行对比。
实验结果表明，单个 GLIPv2 模型（所有模型权重共享）在各种定位和理解任务上实现了接近 SoTA 的性能。该模型还展示了在开放词汇目标检测任务上的强大的zero-shot和few-shot性能以及在 VL 理解任务上的出色的grounding能力。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c55eca0abeb8352f6698d5f22387fedc.png#pic_center )
> `SOTA`：state-of-the-art ，当前最好/最先进的模型
2. 下面是不同规格的`GLIPv1`/`GLIPv2`模型，在直接推理和 prompt tuning时的对比结果：（灰色表示训练时用了这个数据集，所以无法进行zero-shot推理）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/52d71afa4ac3c24e373a2911ed0ba40b.png#pic_center )
3. 消融试验
- 左侧x轴表示使用不同数量的下游任务样本，y轴是13个数据集上的平均AP；
- 右侧是使用不同结构的loss时在 `ODinW`数据集上的的消融试验结果；
- `zero-shot GLIPv2-T` (48.5) 超过了5-shot DyHead-T (46.4)
- `one-shot GLIPv2-H` (61.3) 超过了用所有数据（ALL）进行有监督微调的DyHead-T (60.8).
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a214dbf4c26cee72d44a3ae73e29df70.png#pic_center )
## 四、CLIP图像生成
### 4.1 CLIPasso生成极简画
>- 论文：[CLIPasso: Semantically-Aware Object Sketching](https://paperswithcode.com/paper/clipasso-semantically-aware-object-sketching)、[代码](https://github.com/yael-vinker/CLIPasso)
>- 李沐论文精读系列之[《CLIP 改进工作串讲（下）》](https://www.bilibili.com/video/BV1gg411U7n4/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、笔记[《CLIP 改进工作串讲（下）》](https://blog.csdn.net/weixin_44966641/article/details/127365311)

#### 4.1.1 前言：为何又是 CLIP？
&#8195;&#8195;`CLIPasso`获得了2022年的SIGGRAPH最佳论文奖，其论文题目Semantically-Aware Object Sketching，意思就是**语义感知的物体素描**。从下面包含有毕加索（Picasso）名画的这张图，可以看出`CLIPasso`就是CLIP和毕加索的缩写，这些都表明了这是一篇研究从图片生成简笔画的文章。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7657fc4a576b276dc6e4e76e3fa1ec51.png#pic_center )

本文为什么又选择了`CLIP`？

1. 保持`Semantically-Aware`（语义感知）。
&#8195;&#8195;因为作者想做的就是用最简单的素描，几笔就把物体描述出来，同时大家又能认出来，这样就必须保证语义上和结构上都能被识别才行。可以看出这种素描是很难的，必须抓住物体最关键的特征才行，也就是摘要提到的**要有对物体抽象的能力**。
&#8195;&#8195;下图是毕加索的名画一头公牛，这个系列从第一张图画到最后，花了大概一年。作者展示的就是想输入一张图，输出最后的简笔画，可见这其中抽象是很重要的，也是很难的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cd7e3b2f2b89d608e5b324400e22e577.png#pic_center )
2. 摆脱有监督训练的数据集
&#8195;&#8195;之前也有一些相关工作，但都是收集一些素描数据集（sketch datasets）进行训练，而且抽象程度也是被固定的。这种data-driven（数据驱动）的方式，有什么数据集就学出什么模型，所以最后生成的素描的形式和风格非常受限，违背了图像生成的初衷了。
>&#8195;&#8195;素描数据集很少，学到的种类和风格不够丰富。比如下图几个素描数据集，SketchyCOCO只有9类物体，都是常见的动物。最新的google收集的QuickDraw（来自于大家在网上的涂鸦），虽然有5000万图像，但是只有300多个类别。训练完的模型碰到这些类别之外的物体可能是无法生成准确的素描的，还需要收集相应的数据进行微调。
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/075a6d4059252a51aea01629b961cf6f.png#pic_center )

&#8195;&#8195;所以做到这两点最直接的答案就是CLIP。CLIP图文配对学习的方式，使其对物体特别敏感，对物体的语义信息抓取的非常好；而且还有出色的zero-shot能力，完全不用在下游任务上进行任何的微调，拿过来就可以直接用，所以就有了`CLIPasso`。
>&#8195;&#8195;在可视化期刊`distill`上发表的博客[《Multimodal Neurons in Artificial Neural Networks》](https://distill.pub/2021/multimodal-neurons/)中，作者对CLIP模型的对抗性攻击、OCR攻击、稳健性等等都分析的非常透彻，非常值得一读，这其中就包括CLIP对简笔画的迁移问题。因为之前CLIP都是处理的自然图像，所以迁移到检测分割效果都很好。但是简笔画和自然图像分布完全不同，无法判断CLIP是否能很好的工作。
&#8195;&#8195;在文章中，作者观察到不管图片风格如何，CLIP都能把物体的视觉特征抽取的很好 ，也就是非常的稳健，由此才奠定了`CLIPasso`的工作基础。（其实本文也借鉴了`CLIPDraw`）

#### 4.1.2 摘要
&#8195;&#8195;由于线描具有简单和最小化的特性，因此<font color='red'> 抽象是速写图的核心。抽象需要识别一个物体或场景的基本视觉属性，需要语义理解和对高级概念的先验知识 </font>。因此，抽象描绘对艺术家来说是一种挑战，对机器来说更是如此。

&#8195;&#8195;本文提出的`CLIPasso`，是一种可以在几何简化和语义简化的指导下实现不同程度抽象的物体速写方法。虽然速写生成方法往往依赖明确的素描数据集进行训练，但是本文<font color='red'>利用CLIP的强大能力，从速写和图像中提炼语义概念 </font>，将速写定义为一组贝塞尔曲线。然后用一个可微调光栅化器直接针对基于CLIP的感知损失，优化曲线参数。

&#8195;&#8195;简笔画的抽象程度可以通过改变笔画数量来控制，其生成的草图展示了多层的次抽象性，同时保持了可识别性、基本结构和所画对象的基本视觉成分。该方法可推广到各种类别，并应对具有挑战性的抽象水平，同时保持语义上的视觉线索，以实现实例级和类别级的识别。
#### 4.1.3 模型结构
&#8195;&#8195;作者对训练方式、loss选择和简笔画初始设置都有所改进，才达到最终非常好的效果。比如下图，通过设置不同的笔画数，可以对图像进行不同层次的抽象：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/842023b152de45f738651e9023d4d67c.png#pic_center )
**1. 训练过程**

&#8195;&#8195;生成简笔画的方法不是直接做图到图的生成，而是使用图形学中的贝塞尔（贝兹）曲线来完成简笔绘画。贝塞尔曲线就是通过在平面上定义的几个点来确定一条曲线。本文中，每条曲线是通过四个点来确定，每个点都有其x,y坐标，即$s_{i}=\left \{ p_{i}^{j} \right \}_{j=1}^{4}=\left \{(x_{i},y_{i})^{j}\right \}_{j=1}^{4}$。其中s是笔画Stroke的缩写，j从1到4表示其由4个点控制。

&#8195;&#8195;所以本文的方法就是随机初始化一些贝塞尔曲线，然后经过不停的训练，更改这些点的位置，从而更改贝塞尔曲线，得到最终的简笔画。训练过程如下图所示：
>- `Rasterizer`：光栅化器，图形学方向根据参数绘制贝塞尔曲线的一种方法，可导。所以这部分是是以前就有的方法，不做任何改动。
>- 本文研究的重点：如何选择一个更好的初始化；以及如何选择合适的loss进行训练。
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/36493c52db771315d89ee3d4c3dde538.png#pic_center )
- 初始定义一些贝塞尔曲线$s_1$到$s_n$，然后扔给光栅化器`Rasterizer`，就可以在二维画布上绘制出我们看得到的图像。
- 根据loss训练笔画参数，得到最终的模型输出。

**2. 目标函数**
&#8195;&#8195;生成的简笔画有两个要求，即在语义和结构上和原图保持一致。比如马还是马、牛还是牛；而且不能生成了马，但是马头的朝向反了，或者马从站着变成趴着。在 CLIPasso 中，这两个要求分别由两个损失函数——语义损失$L_s$和几何距离损失$L_g$来保证。

- $L_s$：semantics loss，计算原图特征和简笔画特征，使二者尽可能相似。
&#8195;&#8195;使用 `CLIP`蒸馏`CLIPasso`模型（类似ViLD），可以让模型提取到的图像特征和 CLIP 图像编码器提取的特征接近。这样就借助了刚刚提到的CLIP的稳健性，即无论在原始自然图像上还是简笔画上都能很好的抽取特征。如果二者描述的是同一物体，那么编码后的特征都是同一语义，其特征必然相近。
- $L_g$：geometric distance loss，计算原图和简笔画的浅层编码特征的loss。
&#8195;&#8195;借鉴了一些LowerLevel的视觉任务。因为在模型的前几层，学习到的还是相对低级的几何纹理信息，而非高层语义信息，所以其包含了一些长宽啊这些信息，对几何位置比较敏感。因此约束浅层特征可以保证原图和简笔画的几何轮廓更接近。（比如CLIP预训练模型backbone是ResNet50，就将ResNet50的stage2,3,4层的输出特征抽出来计算loss，而非池化后的2048维特征去计算）

**3. 初始化**
&#8195;&#8195;作者发现，如果完全随机初始化贝塞尔曲线的参数，会使得模型训练很不稳定。生成的简笔画有的既简单又好看，有的怎么训练都恢复不了语义，甚至就是一团糟，所以需要找到一种更稳定的初始化方式。

&#8195;&#8195;基于saliency（显著性）的初始化：将图像输入ViT模型，对最后的多头自注意力取加权平均，得到`saliency map`。然后在`saliency map`上更显著的区域采点完成贝塞尔曲线参数的初始化，这样训练稳定了很多，效果也普遍好了很多。
>&#8195;&#8195;在显著性区域采点，相当于你已经知道这里有个物体（语义更明确），或者已经相当于沿着物体的边界去绘制贝塞尔曲线了。这样初始化曲线和最终简笔画曲线已经比较接近了。

&#8195;&#8195;下图a展示了显著性初始化生成结果（Proposed ）和随机初始化的生成结果（Random）的效果对比，可以看到Proposed的脸部特征更接近原图，而且头发更加简约。
&#8195;&#8195;这里作者还把自注意力的图和最后的采点分布图可视化了出来。可以看到采点分布图已经和最终的简笔画图像非常接近了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0a94020302374bb0b276edb7f945a08c.png#pic_center )
&#8195;&#8195;图b是本文的一种后处理操作。`CLIPasso`对每张图生成三张简笔画，最后计算每张简笔画和原图的loss（$L_s+L_g$），调出loss最低的一张作为最终结果（蓝色框）。
>&#8195;&#8195;这种后处理在文生图中很常见，比如`DALL-E`。根据文本生成很多图像，然后将这些生成的图片在CLIP中又去计算了一下和原文本的相似性，挑出相似性最高的展现出来，往往可以达到最好的效果。

**4. 训练可视化**

&#8195;&#8195;训练一般需要2000次迭代，但一般迭代100次就能看出大概轮廓了。而且作者在附录里说，`CLIPasso`的训练很快。在一张V100的卡上，用6min就可以完成这2000次迭代。所以在计算资源不足的时候可以试试这种跨界研究。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4df3d9fca3c36d4a334c8a46fdd3c8e1.png#pic_center )

#### 4.1.4 实验结果
1. 借助CLIP的`zero-shot`能力，`CLIPasso`对不常见物体也能生成简笔画
之前的方法只能对数据集中有的物体生成简笔画，而很难做到对罕见物体的生成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/498c5b095c8dc08f20cc796141e62f9e.png#pic_center )
2. 任意控制抽象程度
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e199fbb1b73678014ca8cbdf00cbbcf6.png#pic_center )

3. 对比其它方法
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b14bdeb99f2abcbe287d686191b1c350.png#pic_center )
#### 4.1.5 局限性
1. 输入图片有背景时，生成的效果大打折扣。
&#8195;&#8195;输入图片必须是一个物体，且在纯白色的背景上，生成的效果才最好。因为只有这样，自注意力图才更准，初始化效果才会好，而有了背景，自注意力就会复杂很多。
&#8195;&#8195;所以作者是先将图片输入U2Net，从背景中抠出物体，然后再做生成。这样就是两阶段的过程，不是端到端，所以不是最优的结构。如何能融合两个阶段到一个框架，甚至是在设计loss中去除背景的影响，模型适用就更广了。
2. 简笔画是同时生成而非序列生成。
如果模型能做到像人类作画一样，一笔一画，每次根据前一笔确定下一笔的作画位置，不断优化，生成效果可能更好
3. 复杂程度不同物体，需要抽象你的程度不同。
`CLIPasso`控制抽象程度的笔画数必须提前指定，所以最好是将笔画数也设计成可学习的参数。这样对不同的图片上不同复杂程度的物体，都能很好的自动抽象。目前用户每次输入图片，还得考虑用几笔去抽象。

#### 4.1.6 结论
&#8195;&#8195;CLIPasso可以适应任意语义类别的输入图像，而不再局限于数据集中固有的几个类别；并且可以做到对物体不同程度的抽象，同时保持和原图的语义和结构的一致性。

### 4.2 DALL-E2（放到另一篇，后续补）
## 五、CLIP视频理解（略）
&#8195;&#8195;这部分包括CLIP4clip和ActionCLIP，在[《CLIP 改进工作串讲（下）》](https://www.bilibili.com/video/BV1gg411U7n4/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)中有讲解，我对这一块暂时不感兴趣，就不写了。
## 六、其它方向
- 多模态 CLIP-ViL：
	- 论文[How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383)、[Code](https://github.com/clip-vil/CLIP-ViL)
	- 作者将 CLIP 的预训练参数用来初始化 ViL 模型，然后再各种视觉-文本多模态任务上进行微调，测试结果。
- 语音 AudioCLIP：暂不感兴趣
- 3D PointCLIP：
	- 论文： [PointCLIP: Point Cloud Understanding by CLIP](https://arxiv.org/abs/2112.02413)、[代码](https://github.com/ZrrSkywalker/PointCLIP)
	- 作者通过现将 3D 点云投射为多张 2D 的深度图，实现了在3D图上利用 2D 图像数据训练 的CLIP 模型。
	
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a2a9867b3dda646c0d38c5c032030fd3.png#pic_center )


