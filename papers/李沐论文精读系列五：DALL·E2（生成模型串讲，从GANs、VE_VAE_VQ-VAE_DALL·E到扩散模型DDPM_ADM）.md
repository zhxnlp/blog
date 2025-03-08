@[toc]
传送门：
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT](https://blog.csdn.net/qq_56591814/article/details/127313216?spm=1001.2014.3001.5501)
- [李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
- [李沐论文精读系列三：MoCo、对比学习综述（MoCov1/v2/v3、SimCLR v1/v2、DINO等）](https://blog.csdn.net/qq_56591814/article/details/127564330)
- [李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5502)
## 一、 前言
>- 论文[《Hierarchical Text-Conditional Image Generation with CLIP Latents》](https://paperswithcode.com/paper/hierarchical-text-conditional-image)、[DALL-E2官网](https://openai.com/dall-e-2/)、[代码](https://github.com/lucidrains/DALLE2-pytorch)
>- 参考李沐[《DALL·E2论文精读》](https://www.bilibili.com/video/BV17r4y1u77B/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[《论文笔记：DALL-E2详解》](https://blog.csdn.net/zcyzcyjava/article/details/126946020)


### 1.1 DALL·E简介
>参考[《OpenAI祭出120亿参数魔法模型》](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247571522&idx=1&sn=380ab14b7cf34783fd412e60713b6b48&scene=21#wechat_redirect)

&#8195;&#8195;2021年1月， OpenAI发布了 包含120亿参数的大模型DALL·E 。只要「阅读」文本，DALL·E 就能根据文本的内容「自动」生成栩栩如生的大师级画像，因此已经发布，迅速火爆全网。当时，DALL·E 的画风是这样的：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/37d277c01981e7c5248a8469ef0664bd.png)
&#8195;&#8195;OpenAI发现它具有多种功能，包括创建拟人化的动物和物体、以合理的方式组合无关概念、渲染文本并将转换应用于现有的图像。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5ec7976b96ffa01c3a740b06fd6d5c3d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8e989e0aa81618aa21943f67e7eeb25e.png)
更多生成展示可以参考[《OpenAI祭出120亿参数魔法模型》](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247571522&idx=1&sn=380ab14b7cf34783fd412e60713b6b48&scene=21#wechat_redirect)。
### 1.2 DALL·E2简介
&#8195;&#8195;`OpenAI`继去年1月推出`DALL·E`，年底推出`GLIDE`之后，时隔一年又在2022.4 推出`DALL·E2`。相比 `DALL·E` ，`DALL·E2` 可以生成更真实和更准确的画像：综合文本描述中给出的概念、属性与风格等三个元素，生成「现实主义」图像与艺术作品！分辨率更是提高了4倍！打开DALL·E2的[官网](https://openai.com/dall-e-2/)，可以看到其介绍的DALL·E2有以下几个功能：
1. 根据文本直接生成图片
`DALL·E2`可以根据文本描述生成原创性的真实图片（因为模型将文本和图片的特征都学的非常好），它可以任意的去组合这些概念、属性或者风格。下面还给了一个例子——描述是泰迪熊，属性是做最新的AI研究，风格是1980年的月球上，然后模型生成了下面的图片：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b167eca961cd7882170921d946f1efdf.png)

如果改变这些变量，又会生成新的图片。又比如下面的例子：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b6032660cf27ed14a244773b48fd6d9c.png#pic_center =700x980)
2. 扩展图像
`DALL·E2`可以将图像扩展到原始画布之外，创造出新的扩展构图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9f49e4e5e95990905663b2355b35ec8c.png#pic_center)

3. 根据文本编辑图片
`DALL·E2`可以对已有的图片进行编辑和修改，添加和删除元素，同时考虑阴影、反射和纹理。比如下面的例子，在房间里加入火烈鸟。如果在2这个位置加入，就是一个火烈鸟的游泳圈，因为常识里，火烈鸟不大可能在水面。如果是3这个位置，还可以生成不同角度样式的火烈鸟，并且这些还有倒影：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f208eade1e719fda787f15e615cf2d1c.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e4d0742b50e0355ea18265b7f2c46b81.png)
4. 给定一张图片，`DALL·E2`可以生成不同的变体，并保持原风格（不需要文本）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/39d795ba410880834fedb9fbc89db525.png)
5. 相比DALL·E ，给定同样的文本，`DALL·E2`可以生成4倍分辨率的图片。

&#8195;&#8195;最后，因为考虑到一些伦理道德（比如`DALL·E2`有一些[黑话](https://zhuanlan.zhihu.com/p/523020005)可能绕过审查机制），所以`DALL·E2`还没有开源，也不能release模型，甚至其API也是开放给了一小部分用户做内侧或者研究（主要是推特或reddit上的大V）。对于大部分没有排上waitlist的小伙伴，又想体验的用户，可以试试[DALL·E Mini](https://github.com/borisdayma/dalle-mini)。
>&#8195;&#8195;Boris Dayma等人根据论文创建了一个迷你但是开源的模型`Dall·E Mini`，只是训练集和模型都比较小，生成的质量会差一些。可以直接上打开github主页上提供的colab跑一跑，或者是`huggingface`的spaces [dalle-mini](https://huggingface.co/spaces/dalle-mini/dalle-mini)里面使用。
>&#8195;&#8195;`huggingface`提供了一个sapces，模型代码上传到里面之后，就可以做成一个APP直接玩。
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/76a246bd89c503211b8736cfafaca8b4.png)
###  1.3  文生图模型进展
&#8195;&#8195;无论是DALL·E还是DALL·E Mini，根据文字生成图片的效果都非常好，所以自从2021.1 DALL·E发布之后，后续就有一大批类似的工作跟进。
- 2021年：
	- 1月OpenAI推出了DALL·E（GPT+VQ-VAE）
	- 5月清华推出支持中文生成图像的[CogView](https://paperswithcode.com/paper/cogview-mastering-text-to-image-generation)
	- 11月微软和北大在推出的[NUWA（女娲）](https://paperswithcode.com/paper/nuwa-infinity-autoregressive-over)，可以生成图像和短视频
	- 12月OpenAI又推出了[GLIDE](https://paperswithcode.com/paper/glide-towards-photorealistic-image-generation)模型（ Classifier-Free Guidance 扩散模型）
	- 12月百度推出了[ERNIE-ViLG](https://paperswithcode.com/paper/ernie-vilg-unified-generative-pre-training)模型，也支持中文。
- 2022年：
	- 4月份OpenAI推出了DALL·E2（CLIP+GLIDE）。同月清华推出了[CogView2](https://paperswithcode.com/paper/cogview2-faster-and-better-text-to-image)
	- 5月清华后又推出[CogVideo](https://paperswithcode.com/paper/cogvideo-large-scale-pretraining-for-text-to)，专门针对视频生成。谷歌在五月也推出了[Imggen](https://paperswithcode.com/paper/photorealistic-text-to-image-diffusion-models)（可参考[此贴](https://blog.csdn.net/amusi1994/article/details/125013555?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166774229316800186590280%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=166774229316800186590280&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-125013555-null-null.nonecase&utm_term=%E8%B0%B7%E6%AD%8C%E6%96%B0%E4%BD%9CImagen&spm=1018.2226.3001.4450)），生成效果和DALL·E不相上下，但模型简单很多（和DALL-E2一样用的是扩散模型）。
	- 6月谷歌又推出了新一代AI绘画大师[Parti](https://zhuanlan.zhihu.com/p/532753671)。
>&#8195;&#8195;扩散模型是一个很火的方向。现在的扩散模型发展程度，就类似18年左右的GAN，还有很多可以改进的地方，而GAN已经做了五六年，被挖掘的已经差不多了。

## 二、 引言
&#8195;&#8195;论文题目——Hierarchical Text-Conditional Image Generation with CLIP Latents，意为根据`CLIP`特征，来做层级式的、依托于文本特征（条件）的图像生成。这里层级式（Hierarchical）的意思是，`DALL·E2`先生成小分辨率的图片（64\*64），再生成256\*256的图片，最后生成1024*1024的高清大图，所以是一个层级式的结构。另外`DALL·E2`中还用到了CLIP模型训练的文本图片对特征，以及使用扩散模型来解码生成图片，后面会细讲。
>&#8195;&#8195;论文的一作Aditya Ramesh参加过CLIP和DALL-E的工作，另外还有两个扩散模型的专家，以及GPT-3/codex的作者。`DALL·E2`其实就是`CLIP`模型加上`GLIDE`模型的融合。

### 2.1 摘要

&#8195;&#8195;之前的对比学习模型，比如OpenAI自己训练的`CLIP`，是使用互相匹配的图片-文本对进行训练的，可以学习到很稳健的图像特征。这些特征既可以捕捉到语义信息，也可以捕捉到风格信息，如果仅仅用来做分类任务就有点可惜了。

&#8195;&#8195;为了借助`CLIP`模型的特征来做图像生成，作者提出了一个两阶段模型。首先给定文本描述，根据CLIP生成文本特征（这里的CLIP是冻住的，不再训练），然后：
-  `prior`：根据文本特征生成类似CLIP的图像特征。作者试了AR（自回归模型）和扩散模型，后者效果更好。
- `decoder`：根据prior输出的图像特征生成图像，这里使用的也是扩散模型

&#8195;&#8195;作者发现，显式的生成图片特征的方式（也就是prior这一步），可以显著的提升生成图像的多样性（ diversity），并且对于图像的逼真程度以及和文本的匹配程度，都没有什么损失。而基于扩散模型的解码器，可以根据图片特征，生成多种多样的图片（风格相近但细节不一）。另外根据文本生成图像的架构，让模型很容易的可以利用CLIP的特性，从而达到可以利用文本直接编辑图像的功能，而且是zero-shot（直接推理，无需训练）。
>&#8195;&#8195;这里也就是说这种两阶段生成的框架，得到的图片既逼真又有多样性。GAN生成的图片也很逼真，因为其本来的目标就是以假乱真；但是多样性不太好，生成的图片都差不多，不具备原创性。这也是最近的`DALL·E2`和`Imagen`都使用扩散模型的原因，因为其多样性好，有创造力。
>&#8195;&#8195;`DALL·E2`在这里的成功，很重要的一点就是借助了`CLIP`模型，而且作者在method这一章，写的方法比较简单，默认大家是学过CLIP模型的。所以对CLIP不清楚的，可以看我之前的博文[《李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）》](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5501)。

### 2.2 引言&模型结构
&#8195;&#8195;引言第一段，作者又吹了一下CLIP模型，说CLIP学到的图像特征非常的稳健（对各种分布/风格的鲁棒性都很强，比如不管是漫画的香蕉、素描的香蕉还是自然界的香蕉，都能分辨出来），而且可以做zero-shot，在各种下游的视觉领域都被证明效果非常好。

&#8195;&#8195;扩散模型很早（15年之前）就提出了，它是一种概率分布模型，其生成的图片是从概率分布里采样的，所以多样性非常好。现在的扩散模型在图像和视频生成上，都达到了目前最好的效果。

>&#8195;&#8195;扩散模型逼真度不如GAN（GAN的目标就是以假乱真，比如非常火爆的[Deepface](https://blog.csdn.net/LuohenYJ/article/details/125564426?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166778486616782417048519%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166778486616782417048519&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125564426-null-null.142%5Ev63%5Ewechat,201%5Ev3%5Econtrol_2,213%5Ev1%5Et3_esquery_v1&utm_term=deepface&spm=1018.2226.3001.4187)）。不过从20年开始，从DDPM到improved DDPM、Diffusion Models Beat GANs到最近的`DALL·E2`和`Imagen`，使用了一系列的技巧来提高扩散模型的保真度，其中之一就是引导技巧`guidance technique`。
>&#8195;&#8195;`guidance technique`可以牺牲一定的多样性来提高保真度，使得扩散模型的分数可以媲美GANs，而且多样性还是更强，成为当前的`SOTA`模型。在`DALL·E2`和中，作者都特意提到`guidance technique`至关重要，使模型效果提高了很多。

&#8195;&#8195;下面是作者引言部分贴出的9张高清大图，比如柴犬带着贝雷帽，身穿黑色毛衣；疯狂的熊猫科学家将化学试剂混合在一起；土星上有一只身穿宇航员服的海豚等等。这些图片分辨率很高，细节生成的都很好。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/613d2b0c17f08def881e9d218089987f.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cc7b6c44a188223fd02b66d507002a8a.png)

**模型结构**

&#8195;&#8195;这里作者给出了DALL·E2的整体架构，上部分就是一个`CLIP`，下部分才是`DALL·E2`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b0535fd9282fed0ef7327a9a367ddeb3.png)
**1. CLIP训练过程**
&#8195;&#8195;如下图所示，`CLIP`的输入是一对对配对好的的图片-文本对（比如输入是一张狗的图片，对应文本也表示这是一只狗）。这些文本和图片分别通过`Text Encoder`和`Image Encoder`输出对应的特征。然后在这些输出的文字特征和图片特征上进行对比学习。
&#8195;&#8195;假如模型输入的是`n`对图片-文本对，那么这`n`对互相配对的图像–文本对是正样本（下图输出特征矩阵对角线上标识蓝色的部位），其它$n^2-n$对样本都是负样本。这样模型的训练过程就是最大化n个正样本的相似度，同时最小化$n^2-n$个负样本的相似度（余弦相似性`cosine similarity`）。
&#8195;&#8195;`CLIP`使用的数据集是`OpenAI`从互联网收集的4个亿的文本-图像对，所以模型的两个编码器训练的非常好，而且文本和图像特征紧紧的联系在一起，这也是`CLIP`可以直接实现`zero-shot`的图像分类的原因。

>&#8195;&#8195;`Text Encoder`可以采用NLP中常用的`text transformer`模型；而`Image Encoder`可以采用常用`CNN`模型或者`vision transformer`等模型。
&#8195;&#8195;`zero-shot`分类，即不需要任何训练和微调，而且`CLIP`分类不再受限于类别列表（open vocabulary）。在类别标签中随意的添加类别prompt（也就是下图`A photo of {label}`这种句子），模型就可以将新的类别检测出来，所以`CLIP`在ImageNet上可检测的类别远大于1000，这也是其最吸引人的地方。


![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7ae65ae1820e3330385127726ffc1afd.png)
**2. DALL·E2**
&#8195;&#8195;上面的CLIP训练好之后，就将其冻住了，不再参与任何训练和微调。DALL·E2训练时，输入也是文本-图像对，下面就是摘要提到的两阶段训练：
- `prior`：根据文本特征生成图像特征
	- 文本和图片分别通过锁住的`CLIP text encoder`和`CLIP image encoder`得到编码后的文本特征和图片特征。（这里文本和文本特征是一一对应的，因为这部分是始终锁住的，图片部分也一样）
	- `prior`模型的输入就是上面CLIP编码的文本特征，其ground truth就是CLIP编码的图片特征，利用文本特征预测图片特征，就完成了 `prior`的训练。
	- 推理时，文本还是通过`CLIP text encoder`得到文本特征，然后根据训练好的`prior`得到类似CLIP生成的图片特征（此时没有图片，所以没有`CLIP image encoder`这部分过程）。此时图片特征应该训练的非常好，不仅可以用来生成图像，而且和文本联系的非常紧（包含丰富的语义信息）。
- `decoder`：常规的扩散模型解码器，解码生成图像。
这里的decoder就是升级版的`GLIDE`，所以说`DALL·E2=CLIP+GLIDE`。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0d2875be02a446052521b23896c21108.png)
>&#8195;&#8195;其实最暴力的方法，就是直接根据文本特征生成图像，只要中间训练一个融合特征的大模型就行。但是就像之前作者再在摘要说的说的，如果有显式的生成图片特征的过程（也就是`文本→文本特征→图片特征→图片`），再由图像特征生成图像，模型效果就会好很多，所以采用了两阶段生成方式。
>&#8195;&#8195;另外论文中，作者将本文的模型称为`unCLIP`而非`DALL·E2`，因为CLIP是从文本/图像训练得到特征，然后可以拿训练好的图像特征去做分类、检测等任务（本身是做分类，后续改进工作拿来做检测和分割等等），是一个从输入→特征的过程。而`DALL·E2`是从`文本特征→图片特征→图片`的过程，是CLIP的反过程，所以作者称其为`unCLIP`。

## 三、 算法铺垫
&#8195;&#8195;`DALL·E2`是基于GLIDE、CLIP和扩散模型做的，但作者这部分只讲了和GLIDE的区别及一些实现细节，prior部分也是实现细节，而没有讲方法本身。所以仅从这部分，无法知道整个模型的具体算法。

&#8195;&#8195;下面简单介绍一下图形生成之前的工作，从GANs到AE（[Auto encoder](https://paperswithcode.com/method/autoencoder)）、VAE系列工作（[Auto-Encoding Variational Bayes](https://paperswithcode.com/paper/auto-encoding-variational-bayes)）再到扩散模型及一系列后续工作。
### 3.1 GANs
&#8195;&#8195;`GANs`（Generative Adversarial Networks，生成对抗网络）是从对抗训练中估计一个生成模型，其由两个基础神经网络组成，即生成器神经网络`G`（Generator Neural Network） 和判别器神经网络`D`（Discriminator Neural Network）。
&#8195;&#8195;生成器`G`从给定噪声中（一般是指均匀分布或者正态分布）采样来合成数据，判别器`D`用于判别样本是真实样本还是G生成的样本。`G`的目标就是尽量生成真实的图片去欺骗判别网络`D`，使`D`犯错；而`D`的目标就是尽量把`G`生成的图片和真实的图片分别开来。二者互相博弈，共同进化，最终的结果是`D(G(z)) = 0.5`，此时G生成的数据逼近真实数据（图片、序列、视频等）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c3c4d6e97b3fc8a65e0e6d55119a5008.png)
>&#8195;&#8195;GAN就是对分布进行建模，希望模型可以生成各种分布。最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此`D(G(z)) = 0.5`，此时噪声分布接近真实数据分布。

GANs也有很多局限性，比如：
- 训练不够稳定。
因为要同时训练两个网络，就涉及到平衡问题。经常训练不好，模型就坍塌了，所以这个缺点非常致命。
- GANs生成的多样性不够好。
GANs其主要的优化目标，就是让图片尽可能的真实。其生成的多样性，主要就来自于初始给定的随机噪声，所以创造性（原创性）不够好。
- GANs是隐式生成，不够优美
GANs不是概率模型，其生成都是通过一个网络去完成，所以GANs的生成都是隐式的。不知道模型都训练了什么，也不知道其遵循什么分布，在数学上就不如后续的VAE或扩散模型优美。
### 3.2 AE
>介绍：[Autoencoder](https://paperswithcode.com/method/autoencoder)

&#8195;&#8195;`AE`自编码器是一种瓶颈架构（ bottleneck），它使用编码器将高维输入$x$转换为潜在的低维Code $h$，然后使用解码器将潜在Code $h$进行重构，得到最终的输出${x}'$。目标函数就是希望${x}'$能尽量的重建${x}$。因为是自己重建自己，所以叫Autoencoder。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e6ff8b4f65157b89be07db5ac20ef8a2.png#pic_center)
### 3.3 DAE/MAE
>&#8195;&#8195;MAE论文[《Masked Autoencoders Are Scalable Vision Learners》](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)。其主要是将图片masked掉75%的像素区域之后，输入`ViT encoder`提取图片特征，然后用`decoder`重构原图，所以是一个无监督的训练过程，数据集和模型都可以做得很大
>&#8195;&#8195;推理时，使用训练好的`MAE encoder`提取图片特征进行下游任务微调。感兴趣可以看我的帖子[《李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer》](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)

&#8195;&#8195;紧跟着AE之后，出来了`DAE`（Denoising Autoencoder），就是先把原图$x$进行一定程度的打乱，变成$x_c$（corrupted x）。然后将$x_c$传给编码器，后续都是一样的，目标函数还是希望${x}'$能尽量的重建原始的${x}$。
&#8195;&#8195;这个改进非常有用，会让训练出来的模型非常的稳健，也不容易过拟合，尤其是对于视觉领域来说。因为**图像的像素信息，冗余性非常高**。即使把图片进行一定的扰乱（污染），模型还是能抓取其主要特征，重构原图。

&#8195;&#8195;这一点也类似`MAE`（Masked Autoencoder，掩码自编码器）的意思。作者在训练时，将图像75%的像素区域都masked掉（下图序列中灰色就是被masked的区域，不会传入decoder）。即使这样，模型也能将最终的图像重构出来，可见图像的冗余性确实是很高。作者在MAE论文中反复强调，高掩码率是非常重要的，所以说DAE或者MAE这种操作还是非常有效的。
![aa](https://i-blog.csdnimg.cn/blog_migrate/3ca1347d0875031c9465ad46b8d2bde1.png)<center>MAE主要结构</center>

### 3.4 变分自编码器VAE
>论文：《[Auto-Encoding Variational Bayes](https://paperswithcode.com/paper/auto-encoding-variational-bayes)》

&#8195;&#8195;上面的`AE/DAE/MAE`都是为了学习中间的特征$h$，然后拿这些特征去做后续的分类、检测、分割这些任务，而并不是用来做生成的。因为中间学到的$h$不是一个概率分布，只是一个专门用于重构的特征，所以没法对其进行采样。

&#8195;&#8195;`VAE`（Variational Auto-Encoder）就是借助了这种encoder-decoder的结构去做生成，和AE最主要的区别就是不再去学习中间的bottleneck特征了，而是去学习一种分布。
 
&#8195;&#8195;作者假设中间的分布是一个高斯分布（用均值$\mu$ 和方差$\sigma$来描述）。具体来说，就是将输入$x$进行编码得到特征之后，再接一些FC层，去预测中间分布的$\mu$ 和$\sigma$。

&#8195;&#8195;$\mu$ 和$\sigma$训练好之后，就可以扔掉encoder了。推理时直接从训练好的分布去采样一些$z$出来（$z=\mu +\sigma \cdot \varepsilon$），然后进行解码，这样VAE就可以用来做生成了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/55a321f38fb294b00cb55888fc216a6a.png)
&#8195;&#8195;从贝叶斯概率的角度看，前面的从x预测z的过程就是后验概率$q_{\theta }(z|x)$，学出来的分布就是先验分布$p_{\theta }P(x|z)$。给定z预测x就是likelihood，模型的训练过程就是maximize likelihood。这样从数学上看，就干净优美很多，而且VAE也有一些不错的性质，比如生成的多样性好。

&#8195;&#8195;VAE是从概率分布中去采样，所以其生成的多样性比GANs好得多。所以这也是为什么后续有一系列基于VAE的工作，比如VQ-VAE/VQ-VAE-2，以及基于VQ-VAE的DALL·E。

>以下摘自苏建林老师的[《变分自编码器（一）：原来是这么一回事》](https://spaces.ac.cn/archives/5253)：
> 
>  /![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/78fa66885f70600924c07499b5b32af1.png)
>  后面内容（公式）太多了，感兴趣可以看看。

### 3.5 VQ-VAE/VQ-VAE2
>- 论文：VQ-VAE[《Neural Discrete Representation Learning》](https://paperswithcode.com/paper/neural-discrete-representation-learning)、VA-VAE-2[《Generating Diverse High-Fidelity Images with VQ-VAE-2》](https://paperswithcode.com/paper/190600446)
>- 参考苏建林博客[《VQ-VAE简明介绍》](https://spaces.ac.cn/archives/6760)、CSDN博客[《VQ-VAE-2》](https://blog.csdn.net/weixin_38858621/article/details/105234094)

&#8195;&#8195;`VQ-VAE`（Vector Quantised - Variational AutoEncoder）首先出现在论文[《Neural Discrete Representation Learning》](https://paperswithcode.com/paper/neural-discrete-representation-learning)，跟VQ-VAE-2一样，都是Google团队的大作。

&#8195;&#8195;VQ即Vector Quantised，它编码出的向量是离散的，也就是把VAE做量化，所以`VQ-VAE`最后得到的编码向量的每个元素都是一个整数。
#### 3.5.1  为何要做Quantised Vector？

&#8195;&#8195;现实生活中，很多信息（声音、图片）都是连续的，你的大部分任务都是一个回归任务。但是等你真正将其表示出来，等你真正解决这些任务的时候，我们都将其离散化了。图像变成了像素，语音也抽样过了，大部分工作的很好的也都是分类模型（回归任务→分类任务）。

&#8195;&#8195;如果还是之前VAE的模式，就不好把模型做大，分布也不好学。取而代之的不是去直接预测分布$z$，而是用一个`codebook`代替。codebook可以理解为聚类的中心，大小一般是K*D（K=8192，Dim=512/768），也就是有8192个长为D的向量（聚类中心）。

#### 3.5.2   VQ-VAE算法

&#8195;&#8195;$x$输入编码器得到高宽分别为$(h,w)$的特征图$f$，然后计算特征图里的向量和codebook里的向量（聚类中心）的相似性。接着把和特征图最接近的聚类中心向量的编号（1-8192）存到矩阵$z$里面。

&#8195;&#8195;训练完成之后，不再需要编码特征$f$，而是取出矩阵$z$中的编号对应的codebook里面的向量，生成一个新的特征图$f_q$（量化特征quantised feature）。最后和之前一样，使用$f_q$解码重构原图。此时这个量化特征就非常可控了，因为它们永远都是从codebook里面来的，而非随机生成，这样优化起来相对容易。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6dce50df4cc6284e363cb1c853658834.png)
- 左图：`VQ-VAE`的模型结构
- 右图：`embedding space`可视化。编码器输出$z(x)$会mapped到最相近（nearest）的点$e_2$。
- 红色线的梯度$\triangledown _{z}L$迫使encoder在下一次forword时改变其输出（参数更新）。
- 由于编码器的输出和解码器的输入共享D维空间，梯度包含了编码器如何改变参数以降低损失的有效信息。

下面是一些重构效果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cc3dd624604d92acf10cc675c8a84fe9.png)
>&#8195;&#8195;`VQ-VAE`也可以用来做CV领域的自监督学习，比如`BEIT`就是把`DALL·E`训练好的`codebook`拿来用。将图片经过上面同样的过程`quantise`成的特征图作为`ground truth`，自监督模型来训练一个网络。后续还有`VL-BEIT`（vision language BEIT）的工作，也是类似的思路，只不过是用一个Transformer编码器来做多模态的任务。

#### 3.5.3 局限性
&#8195;&#8195;`VQ-VAE`学习的是一个固定的`codebook`，所以它又没办法像VAE这样随机采样去做生成。所以说VQ-VAE不像是一个VAE，而更像是一个AE。它学到的codebook特征是拿去做high-level任务的（分类、检测）。

&#8195;&#8195;如果想让VA-VAE做生成，就需要单独训练一个`prior`网络，在论文里，作者就是训练了一个`pixcl-CNN`（利用训练好的codebook去做生成）。
#### 3.5.4 VQ-VAE2（图片生成效果超越 BigGAN）
&#8195;&#8195;本身是对VQ-VAE的简单改进，是一个层级式的结构。VQ-VAE2不仅做局部的建模，而且还做全局的建模（加入attention），所以模型的表达能力更强了。同时根据codebook学了一个prior，所以生成的效果非常好。总体来说VQ-VAE2是一个两阶段的过程：
- 训练编解码器，使其能够很好的复现图像
- 训练PixelCNN自回归模型，使其能够拟合编码表分布，从而通过随机采样，生成图片

stage1：训练一个分层的VQ-VAE用于图像编码到离散隐空间
- 输入图像 $x$ ，通过编码器生成向量$E(x)$ ，然后采用最近邻重构，将$E(x)$替换为codebook的中的一个nearest prototype vector。
- codebook可以理解为离散的编码表，举一张人脸图像为例，codebook就包括头发颜色，脸型，表情和肤色等等。因此，量化就是通过编码表，把自编码器生成的向量$E(x)$离散化：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d0c64dacd3538f30695c1b88087a89b4.png)
- 解码器通过另一个非线性函数重建数据

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9d43e316d1a5bfb02180a517eeda6df5.png)
>&#8195;&#8195;作者提到其动机是把全局信息（如shape和geometry）以及局部信息（如纹理）分开建模。如图所示，`top level`用于model全局信息（256 * 256下采样得到 64 * 64），`bottom level`用于model局部细节（64 * 64再降为32 * 32）。解码器分别从两个隐层中重建图像。
>&#8195;&#8195;原文中还有再加`middle level`，实验结果表明加了middle level之后，生成的图像清晰度更高）。

stage2：在离散隐空间上拟合一个PixelCNN先验
- 经过Stage1，将图片编码为了整数矩阵，所以在Stage2用自回归模型PixelCNN，来对编码矩阵进行拟合（即建模先验分布）。
- 通过PixelCNN得到编码分布后，就可以随机生成一个新的编码矩阵，然后通过编码表$E$映射为浮点数矩阵，最后经过deocder重构得到一张图片。

**生成效果**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/71b1a04c55e76c96d3d2f1f8e7636def.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f4bc66c2276d60390c142caafd2603f4.png)
### 3.6  DALL·E
>论文[《Zero-Shot Text-to-Image Generation》](https://paperswithcode.com/paper/zero-shot-text-to-image-generation)、OpenAI的[DALL·E官网](https://openai.com/blog/dall-e/)、[代码](https://github.com/openai/DALL-E)

&#8195;&#8195;VQ-VAE的生成模式是`pixcl-CNN +codebook`，其中`pixcl-CNN`就是一个自回归模型。OpenAI 一看，这不是到了施展自己看家本领`GPT`的时候了吗，所以果断将`pixcl-CNN`换成`GPT`。再加上最近多模态相关工作的火热进展，可以考虑使用文本引导图像生成，所以就有了`DALL·E`。

DALL·E模型结构如下：
>论文里没有画出模型结构，网上找了一圈暂时也没发现，只好贴出视频里的手绘图了，将就看一下
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b653294611cdf64e990e6ce7478fd7e8.png)
DALL·E和VQ-VAE-2一样，也是一个；两阶段模型：
-  Stage1：Learning the Visual Codebook
	- 输入：一对图像-文本对（训练时）。
	- 编码特征
文本经过BPE编码得到256维的特征$f_t$；图像（256×256）经过VQ-VAE得到图片特征$f_q$（就是上面训练好的VQ-VAE，将其codebook直接拿来用。$f_q$维度下降到32×32）
- Stage2：Learning the Prior
	- 重构原图
将$f_q$拉直为1024维的tokens，然后连上256维的文本特征$f_t$，这样就得到了1280维的token序列。然后直接送入GPT（masked decoder）重构原图。

&#8195;&#8195;推理时，输入文本经过编码得到文本特征，再将文本通过GPT利用自回归的方式生成图片。生成的多张图片会通过CLIP模型和输入的文本进行相似度计算，然后调出最相似（描述最贴切）的图像。
&#8195;&#8195;另外还有很多训练细节，DALL·E中有近一半的篇幅是在说明如何训练好这120亿参数的模型，以及如何收集足以支撑训练如此大规模模型的数据集（大力出奇迹）。

### 3.7 扩散模型（原始）
>参考知乎：[《怎么理解今年CV比较火的扩散模型（DDPM）》](https://www.zhihu.com/question/545764550/answer/2670611518)

&#8195;&#8195;正如前面所说，一些主流的文生图模型如`DALL·E 2`、`Imagen`都采用了扩散模型（`Diffusion Model`）作为图像生成模型，这也引发了对扩散模型的研究热潮。相比GAN来说，扩散模型训练更稳定，而且能够生成更多样的样本，OpenAI的论文`DDPM`也证明了扩散模型能够超越GAN。

简单来说，扩散模型包含两个过程：前向扩散过程（forword）和反向生成过程（reverse）：
- 前向扩散过程：对数据逐渐增加高斯噪音直至数据变成随机噪音的过程（噪音化）
- 反向生成过程：从随机噪音开始逐步去噪音直至生成一张图像（去噪）

扩散模型与其它主流生成模型的对比如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/763ff3fd0f528dc71ba3536fe331ef1d.png#pic_center =650x)
&#8195;&#8195;如下图所示。无论是前向过程还是反向过程都是一个参数化的马尔可夫链（Markov chain）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aa1e566cf9660071ca670fe772787a12.png#pic_center)
1. 扩散过程：噪音化

&#8195;&#8195;详细解释就是：给定一张图片$x_0$，然后逐步往里面添加一个很小的正态分布噪声。$x_0$添加很小的噪声（比如几个杂点）得到$x_1$，然后继续添加得到$x_2$......，累计$t$步之后，我们得到了$x_t$。以上描述的加噪过程可以写成公式：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7b7124d7934e79c5efd8903ca244bccf.png#pic_center)
&#8195;&#8195;上式意思是：由​  $x_{t-1}$得到$x_{t}$​  的过程​ $q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1})$ ，满足分布​ $N$。这里的$(\beta _{t})_{t=1}^{T}$是每一步采用的方差。
&#8195;&#8195;我们看到这个噪声$z$只由​ $\beta _{t}$ 和  $x_{t-1}$ ​来确定，是一个固定值而不是一个可学习过程。因此，只要我们有了​$x_{0}$ ，并且提前确定每一步 $\beta _{t}$的固定值 ，我们就可以推出任意一步的加噪数据$x_{t}$，整个扩散过程也就是一个马尔卡夫链`Markov chain`（上图框选的公式）。
&#8195;&#8195;在一个设计好的`variance schedule`下（每一步设定的方差，通常越靠后方差越大），如果扩散步数$T$足够大，那么最终得到的$\mathbf{x}_{T}$就完全丢失了原始数据而变成了一个随机噪声（各同向性的正态分布）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e22bd5d05737b51281c72212c95ec126.png#pic_center =650x)
>&#8195;&#8195;扩散这个词来自于热力学的启发。在热力学中，高密度的物体会慢慢向低密度物体渗透，这种过程就叫扩散 `diffusion`。比如你喷的香水会慢慢扩散到整个房间，最后达到一种`balance`（各同向性正态分布，趋近于随机噪声）。
>&#8195;&#8195;CV领域就引用了这种概念，将上面的模型称之为扩散模型`Diffusion Model`。
>&#8195;&#8195;扩散过程往往是固定的，即采用一个预先定义好的variance schedule，比如`DDPM`就采用一个线性的`variance schedule`。

>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a5cd160a3478e7e3827442fee4da2080.png#pic_center =650x)
2. 反向过程：去噪

&#8195;&#8195;如果我们知道反向过程的每一步的真实分布$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，那么从一个随机噪音$z:\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$开始（类似GANs里面的z），逐渐去噪就能生成一个真实的样本（$x_{t}\rightarrow x_{t-1}\rightarrow x_{t-2},...,\rightarrow x_{0}$）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f4dc0894bd4836e5e032ea1478b3e22e.png#pic_center =650x)

&#8195;&#8195;估计分布$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$可以用神经网络来完成。具体地说，如果现在随机抽样一个噪声，比如$x_t$（或者其它任意一步）。我们可以训练一个模型将其逐步变为$x_0$。所有使用的这些模型都共享参数（也就是只有一个模型），只是要抽样生成多次。扩散模型就是要得到这些训练好的网络，因为它们构成了最终的生成模型。

&#8195;&#8195;训练好之后，采样任意时刻$t$下的加噪结果$x_t$  ，将  $\alpha _{t}=1-\beta _{t}$和$\bar{\alpha }_{t}=\prod _{s=1}^{t}\alpha_{s}$（缺的是累乘符），则我们可以得到：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ad4f53e2844bb65fd936020f903604f0.png#pic_center)
>&#8195;&#8195;无论是前向过程还是反向过程都是一个参数化的`Markov chain`。后续还有一大推公式，马尔科夫链、隐变量模型、优化函数等等，再写就跑题了。感兴趣可以看看上面给的知乎文章，还有好多推导帖子。

stable diffusion也是通过这个过程，可以进行图生图操作（扩散之后才有更多的空间取发挥）：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d9fb38ad35d66d115d1b1a3720d715d2.png)


3. backbone（大部分扩散模型选用`U-Net`）

&#8195;&#8195;从上面可以看到，整个前向后向过程，模型的输入输出维度都是不变的。所以`diffusion model`采用了最常见的模型结构——`U-Net`。
&#8195;&#8195;`U-Net`就是用编码器将图片一点点的压缩，再用一个解码器将其一步步的恢复回来，所以其输入输出大小始终是一样的，非常适合做扩散模型的backbone。
&#8195;&#8195;另外为了恢复效果更好，`U-Net`里还有一些`skip connection`的操作，可以直接将前面的信息传递给后面，以恢复更多的细节。后续还有一些改进，比如在`U-Net`里加一些attention操作，可以使图像生成的更好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f907969b8639c6b3960cbe6c60fa96da.png#pic_center)
4. 局限性：训练推理慢

&#8195;&#8195;扩散模型最大的一个局限性就是训练和推理特别的慢。对于GANs来说，推理时forword一次就能生成结果了，但是扩散模型要forword多次，非常的慢。特别是最原始的扩散模型，$t$=1000等于选了一个随机噪声之后，要forword一千次，一点点的把图像恢复出来，才能得到最终的生成结果，所以扩散模型做推理是最慢的。

5. 发展历程
下面会慢慢介绍
### 3.8 DDPM
>- 论文：[《Denoising Diffusion Probabilistic Models》](https://paperswithcode.com/paper/denoising-diffusion-probabilistic-models)
>- [《从DDPM到GLIDE：基于扩散模型的图像生成算法进展》](https://zhuanlan.zhihu.com/p/449284962)
#### 3.8.1 主要贡献
&#8195;&#8195;`DDPM`是2020年6月发表的，其对原始的扩散模型做了一定的改进，使器优化过程更加的简单。`DDPM`第一次使得扩散模型能够生成很好的图片，算是扩散模型的开山之作。主要来说，`DDPM`有两个重要的贡献：
1. 从预测转换图像改进为预测噪声
- 作者认为，每次直接从$x_{t}$预测$x_{t-1}$，这种图像到图像的转化不太好优化。所以作者考虑直接去预测从$x_{t}$到$x_{t-1}$这一步所添加的噪声$\varepsilon$，这样就简化了问题。
- 这种操作就有点类似`ResNet`的残差结构。每次新增一些层，模型不是直接从$x$去预测$y$（这样比较困难），而是让新增的层去预测$(y-x)$。这样新增层不用全部重新学习，而是学习原来已经学习到的$x$和真实值$y$之间的残差就行（`residual`）
- 目标函数
	- DDPM采用了一个`U-Net` 结构的Autoencoder来对t时刻的高斯噪声$z$进行预测。训练目标即希望预测的噪声和真实的噪声一致，所以目标函数为预测噪声和$z$的L1 Loss：$p(\mathbf{x}_{t-1} \vert \mathbf{x}_t)=\left \| z-f_{\theta  }(x_{t},t) \right \|$
	- 这里的标签$z$是正向扩散过程中，我们每一步添加的，所以是已知的。这里的$f_{\varepsilon }$就对应了`U-Net` 模型结构，t就是`U-Net` 另一个输入`time embedding`。
	- 通过这个简单的L1损失函数，模型就可以训练起来了。
- `time embedding`
	- `U-Net`模型输入，除了当前时刻的$x_{t}$，还有一个输入`time embedding`（类似transformer里的正弦位置编码），主要用于告诉 `U-Net`模型，现在到了反向过程的第几步。
	- `time embedding`的一个重要功能就是引导`U-Net`生成。
&#8195;&#8195;`U-Net`的每一层都是共享参数的，那怎样让其根据不同的输入生成不同的输出呢？因为我们希望从随机噪声开始先生成大致轮廓（全局特征），再一步步添加细节生成逼真的图片（局部特征，边边角角）。
&#8195;&#8195;这个时候，有一个`time embedding`可以提醒模型现在走到哪一步了，我的生成是需要糙一点还是细致一点。所以添加`time embedding`对生成和采样都很有帮助，可以使模型效果明显提升。
	- $x_{t}$和`time embedding`可以直接相加，拼接或者是其它操作。

2. 只预测正态分布的均值

&#8195;&#8195;正态分布由均值和方差决定。作者在这里发现，其实模型不需要学方差，只需要学习均值就行。逆向过程中高斯分布的方差项直接使用一个常数，模型的效果就已经很好。所以就再一次降低了模型的优化难度。


#### 3.8.2 总结：和VAE的区别
&#8195;&#8195;`DDPM`也有些类似`VAE`，也可以将其当做一个encoder-decoder的结构，但是有几点区别：
1. 扩散过程是编码器一步步的走到$z$（$x_t$），而且是一个固定的过程；而VAE的编码器是可以学习的；
2. DDPM的每一步输出输出都是同样维度大小的，但对一般的自编码器（AE/VAE等），往往中间的bottleneck特征会比输入小很多；
3. 扩散模型有步数`step`的概念（time step、time embedding），模型要经过很多步才能生成图片。在所有step中，`U-Net`都是共享参数的。

#### 3.8.3 improved DDPM
&#8195;&#8195;DDPM使得扩散模型可以在真实数据集上work得很好之后，一下子吸引了很多人的兴趣。因为DDPM在数学上特别的简洁美观，无论正向还是逆向，都是高斯分布，可以做很多推理证明；而且还有很多不错的性质。
DALL·E2的二作和三作看到之后，立马着手研究。所以在2020年底左右，OpenAI又推出了 `improved DDPM`。
`improved DDPM`相比DDPM做了几点改动：
- DDPM的逆向过程中，高斯分布的方差项直接使用一个常数而不用学习。`improved DDPM`作者就觉得如果方差效果应该会更好，改了之后果然取样和生成效果都好了很多。
- `DDPM`添加噪声时采用的线性的`variance schedule`改为余弦schedule，效果更好（类似学习率从线性改为余弦）。
- 简单尝试了**scale大模型之后生成效果更好**。
### 3.9  ADM Nets:扩散模型比GANs强
#### 3.9.1 主要改进
&#8195;&#8195;上面第三点对OpenAI来说，无疑是个好消息。所以这两人马上着手研究，出来了[《Diffusion Models Beat GANs on Image Synthesis》](https://paperswithcode.com/paper/diffusion-models-beat-gans-on-image-synthesis)这篇论文，比之前的`improved DDPM`又做了一些改进：
- 使用大模型：加大加宽网络、使用更多的自注意力头attention head，加大自注意力scale（single-scale attention改为multi-scale attention）。
- 使用`classifier guidance`的方法，引导模型进行采样和生成。这样不仅使生成的图片更逼真，而且加速了反向采样过程。论文中，只需要25次采样，就可以从噪声生成图片。
- 提出了新的归一化方式——`Adaptive Group Normalization`，在文章就是根据步数进行**自适应的归一化**。这个方法是对group归一化的一个改进：
 $$\text{AdaGN}(h,y=[y_s,y_b]) = y_s\text{GroupNorm}(h)+y_b$$

上面公式中的$h$是残差块激活函数的输出，$y$ 是一个线性层对时步和后面用到的类别信息的嵌入。组归一化是对输入的通道方向进行分组归一化的归一化方法，可以理解为局部LayerNorm：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bce992fe1311042a3ca3feea58645915.png)
#### 3.9.2 模型效果
&#8195;&#8195;Diffusion Models Beat GANs出来之后，在ImageNet的生成任务中打败了最先进的BigGAN，并且使得扩散模型的分数第一次超过了`BigGANs`。下面是定性的生成结果，左边是BigGAN-deep，中间是他们的ADM，右边是真实图像。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4f62b7bb3b0d3ae5c2174322b65b3f5e.png#pic_center =700x)

#### 3.9.3 classifier guidance
&#8195;&#8195;在`Diffusion Models Beat GANs`这篇论文之前，扩散模型生成的图片也很逼真，但是在算IS score、FID score时，比不过GANs。这样的话，大家会觉得你论文里生成的那些图是不是作者精心挑选的呢，所以结果不够有信服力，这样就不好过稿不好中论文了。
>&#8195;&#8195;IS（Inception Score）和FID（Frechet Inception Distance score）是当前流行的图像生成模型判别指标。简单说就是`IS`从生成图片的真实性和多样性评价生成模型，分数越高越好；`FID`用于衡量真实图像和生成图像的“距离”，分数越小越好。
>

&#8195;&#8195;刷分还是很重要的，同时扩散模型采样和生成图片非常的慢，所以作者考虑如果有一种额外的指导可以帮助模型进行采样和生成就好了。于是作者借鉴了之前的一种常见技巧`classifier guided diffusion`，即在反向过程训练`U-Net`的同时，也训练一个简单的图片分类器。这个分类器是在ImageNet上训练的，只不过图片加了很多噪声，因为扩散模型的输入始终是加了很多噪声的，跟真实的ImageNet图片是很不一样的，所以是从头训练的。

&#8195;&#8195;当采样$x_t$之后，直接扔给分类器，就可以看到图片分类是否正确，这时候就可以算一个交叉熵目标函数，对应的就得到了一个梯度。之后使用分类器对$x_t$​的梯度信息 $\nabla_{x_t}\text{log}p_{\theta}(x_t)$指导扩散模型的采样和生成。
>&#8195;&#8195;这个梯度暗含了当前图片是否包含物体，以及这个物体是否真实的信息。通过这种梯度的引导，就可以帮助`U-Net`将图片生成的更加真实，要包含各种细节纹理，而不是意思到了就行，要和真实物体匹配上。

&#8195;&#8195;使用了 classifier guidance之后，生成的效果逼真了很多，在各种inception score上分数大幅提高。也就是在这篇论文里，扩散模型的分数第一次超过了`BigGANs`。不过作者也说了，这样做其实是牺牲了一些多样性（对于无条件的扩散模型，分类器梯度的指导提升了模型按类输出的性能），去换取了生成图片的逼真性（写实性）。但这样取舍还是值得的，因为其多样性和逼真度还是比GANs好，一下子奠定了扩散模型在图像生成领域的地位。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/074368da76097642d039f68654506931.png#pic_center =700x)
&#8195;&#8195;上图可以看到，加了`classifier guidance`，调大其强度，精度和IS提升了，FID和Recall下降了（ conditional模型加大强度FID升高，但还是比没有classifier guidance高。后面还有各种实验分数，就不贴了）。

除了最简单最原始的`classifier guidance`之外，还有很多其它的引导方式。
- `CLIP guidance`：将简单的分类器换成CLIP之后，文本和图像就联系起来了。此时不光可以利用这个梯度引导模型采用和生成，而且可以利用文本指导其采样和生成。（原来文生图是在这里起作用）
- image侧引导：除了利用图像重建进行像素级别的引导，还可以做图像特征和风格层面的引导，只需要一个gram matrix就行。
- text侧：可以用训练好的NLP大模型做引导

以上所有引导方式，都是下面目标函数里的$y$，即模型的输入不光是$x_{t}$和time embedding，还有condition。加了condition之后，可以让模型的生成又快又好。
$$p(\mathbf{x}_{t-1} \vert \mathbf{x}_t)=\left \| z-f_{\theta }(x_{t},t,y) \right \|$$

#### 3.9.4 classifier free guidance（有条件生成监督无条件生成）
>[《基于扩散模型的文本引导图像生成算法》](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/124641910?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166788366116782428690275%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166788366116782428690275&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-124641910-null-null.142%5Ev63%5Econtrol,201%5Ev3%5Econtrol_1,213%5Ev1%5Econtrol&utm_term=classifier%20free%20guidance&spm=1018.2226.3001.4187)、[《从DDPM到GLIDE：基于扩散模型的图像生成算法进展》](https://zhuanlan.zhihu.com/p/449284962)
>
&#8195;&#8195;额外引入一个网络来指导，推理的时候比较复杂（扩散模型需要反复迭代，每次迭代都需要额外算一个分数），所以引出了后续工作classifier free guidance。
&#8195;&#8195;`classifier free guidance`的方式，只是改变了模型输入的内容，除了 conditional输入外（随机高斯噪声输入加引导信息）还有 unconditional 的 采样输入。两种输入都会被送到同一个 diffusion model 从而让其能够具有无条件和有条件生成的能力。
&#8195;&#8195;得到有条件输出$f_{\theta }(x_{t},t,y)$和无条件输出$f_{\theta }(x_{t},t,\phi )$后，就可以用前者监督后者，来引导扩散模型进行训练了。最后反向扩散做生成时，我们用无条件的生成，也能达到类似有条件生成的效果。这样一来就摆脱了分类器的限制，所以叫classifier free guidance。
>&#8195;&#8195;比如在训练时使用图像-文本对，这时可以使用文本做指导信号，也就是训练时使用文本作为$y$生成图像。然后把$y$去掉，替换为一个空集$\phi$（空的序列），生成另外的输出。

&#8195;&#8195;扩散模型本来训练就很贵了，`classifier free guidance`这种方式在训练时需要生成两个输出，所以训练更贵了。但是这个方法确实效果好，所以在`GLIDE` 、`DALL·E2`和`Imagen`里都用了，而且都提到这是一个很重要的技巧。用了这么多技巧之后，`GLIDE`终于是一个很好的文生图模型了，只用了35亿参数，生成效果和分数就比`DALL·E`（120亿参数）还好。

&#8195;&#8195;OpenAI一看`GLIDE`这个方向靠谱，就马上跟进，不再考虑`DALL·E`的VQ-VAE路线了。将`GLIDE`改为层级式生成（56→256→1024）并加入prior网络等等，最终得到了`DALL·E2`。
## 四、`DALL·E2`算法
>参考[《论文阅读：AI虚拟人画家 OpenAI DALL-E2》](https://zhuanlan.zhihu.com/p/544529290)
### 4.1 两阶段生成
&#8195;&#8195;铺垫一个小时，下面开始讲解`DALL·E2`的算法部分。
&#8195;&#8195;论文的训练数据集由图像 $x$ 及其相应标题（captions） $y$   组成。给定图像$x$  ，经过训练好的CLIP模型分别得到文本特征$z_t$和图像特征$z_i$。然后训练两个组件来从标题生成图像：

- `prior`：先验模型$P(z_i|y)$，根据标题$y$生成CLIP的图像特征$z_i$。
- `decoder`  ：解码器$P(x|z_i,y)$，生成以CLIP图像特征$z_i$  （和可选的文本标题$y$  ）为条件的图像$x$  。

>&#8195;&#8195;跟上面讲的一样，prior模型的输入就是CLIP编码的文本特征，其ground truth就是CLIP编码的图片特征，因为是图文对输入模型，CLIP是都能编码的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ab524bfd151ed20439421c5418876ba2.png)
&#8195;&#8195;所以`DALL·E2`是一个两阶段的生成器。在先训练好CLIP之后，任意给定文本$y$，通过CLIP文本编码器生成文本特征。再根据prior生成图片特征，最后利用decoder解码图像特征得到生成的图片。CLIP+prior+decoder就是整个文生图模型：
$$P(x,y)=P(x,z_i|y)=P(x|z_i,y) P(z_i|y)$$
- $P(x,y)$：根据文本$y$生成图片$x$
- $P(x,z_i|y)$：根据文本$y$生成图片特征$z_i$和图片$x$。因为CLIP始终是锁住的，给定图片$x$，会生成固定的$z_i$，也就是说二者是对等的，所以说$P(x,y)=P(x,z_i|y)$。
- 上面已经介绍了$P(x|z_i,y)$和$P(z_i|y)$分别是`decoder`和`prior`，二者相乘就得到了$P(x,z_i|y)$。所以从概率学上讲，这种两阶段生成是有依据的。

### 4.2 decoder
&#8195;&#8195;decoder就是使用扩散模型，生成以CLIP图形特征（和可选标题$y$）为条件的图像，这部分就是在GLIDE基础上改进的。
&#8195;&#8195;首先，decoder利用了`CLIP guidance`和`classifier-free guidance`，也就是这里反向扩散过程中，指导信息要么来自CLIP模型，要么来自标题$y$，当然还有一些时候是guidance fredd的。具体操作就是随机地在10%的时间里将CLI特征设置为零，在50%的时间内删除文本标题$y$。这样做训练就更贵了，但是为了更好地效果，OpenAI还是把能用的都用了。
>&#8195;&#8195;`CLIP guidance`和`classifier-free guidance`在上面3.9.3和3.9.4有介绍，这里CLIP guidance还有一些训练细节，得看代码才知道，视频里也没有细讲。

&#8195;&#8195;其次，为了提高分辨率，`DALL·E2`还用了层级式的生成，也就是训练了两个上采样扩散模型。一个将图像的分辨率从64×64上采样到256×256，另一个接着上采样的1024×1024。同时，为了提高上采样器的鲁棒性，还添加了噪声（第一个上采样阶段使用高斯模糊，对于第二个阶段，使用更多样化的BSR退化）。

&#8195;&#8195;最后，作者还强调它们只是用了空洞卷积（只有卷积没有自注意力，也就是没有用Transformer），所以推理时，模型可以适用任何分辨率。论文发现标题上采样没有任何益处，并且使用了 no guidance的unconditional ADM Nets。
### 4.3 prior
&#8195;&#8195;`prior`用于从文本特征生成图像特征，这部分作者试验了两种模型，两种模型都用了classifier-free guidance，因为效果好。
 - AR（自回归模型）
 	- 类似DALL·E或者GPT，将CLIP图像特征$z_i$转为离散的code序列，masked掉之后从标题$y$进行自回归预测就行。
 	- 在CLIP中OpenAI就说过，这种自回归的预测模型训练效率太低了，为了使模型训练加速，还使用了PCA降维。
- 扩散模型：
	- 使用的是`Transformer decoder`处理序列。因为这里输入输出都是embedding序列，所以使用U-Net不太合适。
	- 输入序列包括文本、CLIP的文本特征、3.8.1里讲的timestep embedding、加了噪声之后的CLIP图像特征，还有Transformer本来就有的embedding（CLS token之类的）。最终这个序列拿来预测没有加过噪声的CLIP图像特征$z_i$。
	- 论文发现直接去预测没有污染过的图像特征$z_i$，要比预测噪声效果更好（因为自从`DDPM`提出之后，大家都改为预测噪声了），预测时使用均方误差：（可以看到下面公式里是$z_i$而非噪声）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8ef100b326bafffae33cfc93d36063ba.png)
### 4.4 总结：大力出奇迹
&#8195;&#8195;图像生成这一块的技巧很多，经常连一个模型总览图都很难画出来。但是讲完这么多之后，会发现这些技巧有的时候有用，有的时候没用。
- `DDPM`提出将直接预测图像改为预测噪声，可以简化优化过程。但是`DALL·E2`这里又没有沿袭这种预测噪声的做法。
- `DALL·E2`提出如果有显式的生成图片特征的过程，模型效果会好很多，所以采用了两阶段生成方式。但是`Imagen`直接上一个`U-Net`就解决了，更简单，效果也很好。
- CLIP和`DALL·E2`都说自回归模型训练太贵了，训练太不高效了。但是在7月左右，谷歌又推出了Parti，用pathways模型做自回归的文本图像生成，效果直接超越了`DALL·E2`和`Imagen`。

所以最后总结，都是大力出奇迹。只有scale matters，其它模型、训练技巧都可以商量。

## 五、图像处理
论文描述了三种操作：
### 5.1  变化
给定一张图片生成很多类似图片（风格和物体类似）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d33914780a89986e1e1092d1e30cd4ec.png#pic_center =600x)
&#8195;&#8195;在上面模型总览图就可以知道，给定一张柯基图像，输出CLIP图像编码器就得到图像特征，然后可以得到对应的CLIP文本特征（CLIP图文特征是有一一对应的）。再将这个文本特征输入prior得到图像特征，最后就解码生成新的柯基图像。两张柯基图像风格语义相似，细节有所变化。
$$\mathbf {x\overset{CLIP-image-endoer}{\rightarrow}z_i\overset{CLIP}{\rightarrow}z_t\overset{prior}{\rightarrow}{z_i}'\overset{decoder}{\rightarrow}{x}'}$$

&#8195;&#8195;这样可以方便做设计，输入一段描述性文字，模型可以生成各种图像，最后挑一张喜欢的就行（或者再次输入模型进行生成）。这样不用费力去设计，只要会挑图就行，大大简化了工作量。

### 5.2 图像内插
&#8195;&#8195;`DALL·E2`可以混合两个图像以得到变体，比如下图第一张，越靠近左侧表示风格画占比越大，生成的图片越接近这张风格画；越到右侧生成的图片越接近柯基犬图片。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/25724cf26039a47669ba15c64a23928b.png#pic_center =600x)
### 5.3 文本内插
&#8195;&#8195;`DALL·E2`可以在两个文本之间进行插值，生成的图片也在逐渐改变。再做的细一点估计就可以用文字进行P图，而不用学Photoshop了。
- 第一行是从猫到动画中的超级赛亚猫的变化
- 维都利亚风格的房子变为现代的房子
- 第三行是从成年狮变为幼年狮
- 第四行从冬天景色变为秋天
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d7f2b8eb2e46253fb491c9b0418cad3.png)
>&#8195;&#8195;还有人觉得可以用`DALL·E2`做数据增强。比如先写一个包含几个单词的prompt，输入GPT3去生成一大段文字。再把这段话扔给`DALL·E2`生成图片，这样就有了图片-文本对了。这样就可以无穷无尽的生成图片-文本对，再将其扔给CLIP模型，继续训练`DALL·E2`。
## 六、实验
### 6.1 MS-COCO数据集对比
&#8195;&#8195;下图是在MS-COCO数据集上和其它模型对比了一下FID分数。可以看到Zero-shot FID分数从`DALL·E`到`GLIDE`降了非常多，说明扩散模型还是很好用的。prior用扩散模型比用AR，效果稍微好了一点，而且训练也容易一些，所以这部分也用了扩散模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa1f19efeb6d27b634b1a4fae050ba07.png)<center>表2:MS-COCO 256×256上的FID比较。对于AR和扩散prior，解码器引导scale= 1.25</center>


### 6.2 图像生成效果对比
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c903ed960695e6a6a0ee468361ce979f.png#pic_center )
- 最上面是真实图片
- 绿色火车沿着铁轨开过来：
	- `DALL·E`生成的类似动漫火车，不够真实
	- `GLIDE`生成的火车太大了，几乎占了整个画面，没有开过来的感觉
	- `DALL·E2`生成的两张都不错。
- 另外`DALL·E2`生成的大象图，水面有反光；滑雪图还有阳光和倒影，所以说质量真的不错。

## 七、局限性
1. 不能很好的结合物体和属性
	- 给定文本：红色方块在蓝色方块上面。右图`GLIDE`生成的不错，但是`DALL·E2`的结果惨不忍睹。作者觉得使用了CLIP模型的原因。
	- 使用CLIP 一方面可以让文本和图像联系的更加紧密，更容易去做文生图的任务；但是另一方面，CLIP在对比学习时只考虑相似性，比如这里的红方块蓝方块，但其实CLIP是不了解什么是“on top of”的，不知道什么是上下左右、yes or no这些概念。<font color='red'> CLIP从头到尾都是在寻找物体的相似性，所以在做下游任务时，不能很好的区别物体和它的属性 </font>，导致这里的生成效果很差。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/626753381c7717b23a22d53b208977ab.png#pic_center =600x)

2. 文字生成不够好，无法正确生成含有字幕的图像
比如下图让生成含有“deep learning”的标志，结果生成的图片连deep都拼错了，learning更是没有了。作者考虑是因为最开始的文本使用了BPE编码的编码器（将词拆为子词，比如前后缀），但应该还有其它原因。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5e7874017ac5553518ae80861137b823.png#pic_center =600x)
3. 无法生成太复杂的场景
	- 上图提示是狗在草地玩耍，旁边有个湖泊。可以看到都是狗的近景，其实应该远景更合适
	- 下图表示生成时代广场图，可以看到各个屏幕和广告牌上都是颜色块。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/afb543f54e174690cff1eaa281eb2c3c.png#pic_center =400x)
4. 黑话（这部分不在论文里）
>知乎：[《研究者意外发现DALL-E 2在用自创语言生成图像：全文黑话，人类都看不懂》](https://zhuanlan.zhihu.com/p/523020005)

&#8195;&#8195;有个研究者发现DALL-E 2有自己的黑话，人类看不懂，但是模型自己可以认出来。他摸索出了一个简单的方法来发现 DALLE-2 的黑话。
&#8195;&#8195;假设我们想要找到「蔬菜」（vegetables）对应的黑话，可以将 prompt 设置为「两个农民在谈论蔬菜，带字幕」就会得到下面左侧这张图。里面的字幕上一段代表蔬菜，下面的字幕代表鸟。将这两段字幕作为prompt，模型输出下图b和c，分别是一堆食物和一些鸟的图片。

&#8195;&#8195;从这里也可以看出，`DALL·E2`的物体和属性匹配的不是很好，也就是上面的第二条局限性。感兴趣的可以看上面的知乎帖子。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/66d623be5223476887146623785e6224.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ca7c13070d0c9066484e237507cf22aa.png)<center>左：使用 prompt：「两只鲸鱼讨论食物，配字幕」生成的图像；右：使用 prompt：乱码文本「Wa ch zod ahaakesrea.」结果生成一堆海鲜</center>

&#8195;&#8195;帖子中作者说NLP系统可以根据一些政策法规等等过滤有害信息，一般是比较准确和高效的。但是这种黑话、垃圾提示词（Gibberish），是可以绕过这种过滤机制的产生一些安全绳的问题。

>帖子就写到这里了，是在是有点心力憔悴。或者后面有时间会再优化、补充一下。
