@[toc]
>本文参考：
>- paddle课程[《生成对抗网络七日打卡营》](https://aistudio.baidu.com/aistudio/education/group/info/16651)、博客文章[《NLP 中的对抗训练（附 PyTorch 实现）》](https://wmathor.com/index.php/archives/1537/)及[bilibili视频](https://www.bilibili.com/video/BV1hX4y137dL?spm_id_from=333.999.0.0&vd_source=21011151235423b801d3f3ae98b91e94)、[天池新闻文本分类——bert模型源码（加入生成对抗网络）](https://github.com/MM-IR/rank4_NLP_textclassification/blob/master/bert/bert_mini_lstm_pl.py)
>- 生成式对抗网络系列论文地址在[《PaddleGAN预习课程》](https://aistudio.baidu.com/aistudio/projectdetail/4259072)中有。<font color='red'> 建议对照[打卡营视频讲解](https://aistudio.baidu.com/aistudio/education/group/info/16651)观看，更容易理解。 </font>
## 一、day1：生成对抗网络介绍
&#8195;&#8195;kaggle在2019年曾经举办一项奖金高达100万美元的比赛[《Deepfake Detection Challenge》](https://www.kaggle.com/competitions/deepfake-detection-challenge)，主要是识别视频中哪些人脸是真实哪些是AI生成的。
### 1.1 生成对抗网络概述
#### 1.1.1 GAN的应用
&#8195;&#8195;生成式对抗网络，简称GAN，在图像/视频领域、人机交互领域都有应用，比如:
- 图像视频生成、图像上色、图像修复、超分辨率（视频信息增强、遥感成像）
- Text to Image Generation：根据文字描述生成对应图像
- Image to Image Translation：图像到图像的转化（比如马转成斑马）
- Photo to Cartoon：图像翻译：人物/实景动漫化、风格迁移
- Motion Driving：人脸表情动作迁移
- Lip Sythesis：唇形动作合成
- 对抗神经机器翻译

以下这幅图中的人脸就都是神经网络生成的：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9a8a6cf6aff481fc154a1d4a3e8c6fe8.png)
根据文字描述生成对应图像、医疗影像由生成对抗网络进行数据增广和生成
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cb0914b69edeaf774e79d6e4726f80cc.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9c57b959992c740c61750e3135c43faa.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7d4b1aa2ddc261e2a4e5dd75495c7b48.png)
#### 1.1.2 GAN发展历史
2014年提出以来，生成对抗网络快速发展
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a4359c9efedc839defe5400cf4552fab.png)
以下红色部分在本次课程中会讲到：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2db06185321df3bd31500ee711814be3.png)

### 1.2 GAN原理
&#8195;&#8195;我们之前学习的图片分类、语义分割、目标检测都是判别模型，根据图片特征训练后得到标签，而GAN是生成模型，根据噪声和标签生成需要的图片。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7ba06c3a3dfb1bc2ee177f2633fe2b29.png)
&#8195;&#8195;`GANs`（Generative Adversarial Networks，生成对抗网络）是从对抗训练中估计一个生成模型，其由两个基础神经网络组成，即生成器神经网络`G`（Generator Neural Network） 和判别器神经网络`D`（Discriminator Neural Network）。
&#8195;&#8195;生成器`G`从给定噪声中（一般是指均匀分布或者正态分布）采样来合成数据，判别器`D`用于判别样本是真实样本还是G生成的样本。`G`的目标就是尽量生成真实的图片去欺骗判别网络`D`，使`D`犯错；而`D`的目标就是尽量把`G`生成的图片和真实的图片分别开来。二者互相博弈，共同进化，最终的结果是`D(G(z)) = 0.5`，此时G生成的数据逼近真实数据（图片、序列、视频等）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/82b154c4c9db07a15d2c2cd6d9e45091.png)
- 左图表示，均匀分布的噪声Random noise输入生成器得到假的图片Fake Image，Fake Image和Real Image一起输入判别器中，得到判别分数(分数1为真实图片，0为生成图片）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/24fff0eaac3afe8187a184958fd5fb93.png)

- 右图是GAN的数学描述：
	- G是一个生成图片的网络，接收随机噪声z，通过这个噪声生成图片记做G(z)，D(G(z))是D网络判断G生成的图片的是否真实的概率；
	- D是一个判别网络，输入参数x(表示一张图片)，输出D(x)代表x为真实图片的概率；
	-  $P_{r}$ → 真实数据的分布，X → $P_{r}$的样本（真实图片）
	-  $P_{z}$ → 生成数据的分布，Z → $P_{z}$的样本（噪声）
	- G的目的：希望生成的图片“越接近真实越好，D(G(z))变大，V(D, G)会变小。记做$\underset{G}{min}$。
	- D的目的：希望判别越来越准，D(x)变大，D(G(x))变小，V(D,G)会变大。记做$\underset{D}{max}$。
	- <font color='red'>最后博弈的结果D(G(z)) = 0.5</font>。最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

&#8195;&#8195;最终通过不断的训练，生成的图片会相当真实。但是目前的局限性，就是生成的内容非常逼真（GANs的目标就是以假乱真）不够多样性。
>&#8195;&#8195;现在图片生成领域，最火的还是扩散模型。扩散模型从20年开始，从DDPM到improved DDPM、Diffusion Models Beat GANs到最近的`DALL·E2`和`Imagen`，使用了一系列的技巧来提高扩散模型的保真度，使得扩散模型的保真度可以媲美`GANs`，而且多样性和原创性更强。
### 1.3 生成对抗网络的训练
&#8195;&#8195;生成器G希望从数据的真实分布中采样到一种分布，加入随机噪声（比如0-1之间的均匀分布的噪声）后映射成接近真实分布的生成器分布。可视化就是：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0799000adc3d3c2e6c9ec01839637d55.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/58dc99d6e5aaf1c25a7f064e5b1475da.png)
如上图所示：
1. 初始训练出生成器网络G和判别器网络D；
2. 固定判别器的权重，训练生成器，生成更逼真的图片，所以此时Fake Image标签为1（表示接近真实图片）。以生成器和真实分布的差异作为损失函数训练生成网络。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/432c6ee0af1e047efb81f15ae6c63fa9.png)
4. 固定生成器的权重，训练判别器，识别出生成图片，所以此时Fake Image标签为0（表示生成图片）。 以真实图片和生成图片的二分类问题训练判别网络。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fdeb6c84a2db3864d971b4a46c97dc7b.png)
5. 重复2、3步

### 1.4 DCGAN及代码实现
>参考：论文[《UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS》](https://arxiv.org/pdf/1511.06434.pdf)、[代码链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/dcgan_mnist.yaml)

&#8195;&#8195;由于卷积神经网络(Convolutional neural network, CNN)比MLP有更强的拟合与表达能力，并在判别式模型中取得了很大的成果。因此，Alec等人将CNN引入生成器和判别器，称作深度卷积对抗神经网络（Deep Convolutional GAN, DCGAN），Convolutional表示卷积算子。另外还讨论了 GAN 特征的可视化、潜在空间插值等问题。

GAN最初用在手写数字识别的网络结构入如下：
![二、](https://i-blog.csdnimg.cn/blog_migrate/a538fc69e5c1b9ec7b3be1d0cdc83196.png)
- 如上所示，输入图片是尺寸是[B,1,28,28]，转为B×784维向量；随机噪声是100维向量经过全连接层也转为784维向量。最后判别器经过一个[784,1]的全连接层得到判别结果。
- 一层神经网络太浅，效果不好，所以网络D和G都加到了三层。（G其实主要是训练噪声，然后加上图片向量成为生成图片向量，所以G的初始输入是噪声的维度100，而不是图片的维度784）

DCGAN的改进：
- 使用更深的网络、添加BatchNorm
- 判别器使用卷积算子Convolutional，这样相比全连接层，参数量大大减少，而且卷积层更能提取图片信息，更适用于计算机视觉任务。另外激活函数使用LeakyRelu。
- 生成器需要上采样，所以使用转置卷积，激活函数使用Relu。转置卷积原理可参考我的另一篇笔记：[《动手深度学习13：计算机视觉——语义分割、风格迁移》](https://blog.csdn.net/qq_56591814/article/details/124934701?spm=1001.2014.3001.5501)第二章。

代码如下：（来自[《DCGAN实践》](https://aistudio.baidu.com/aistudio/projectdetail/1795662?channelType=0&channel=0)、）
1. 加载数据集

```python
import os
import random
import paddle 
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.datasets as dset
import paddle.vision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

demo_dataset = paddle.vision.datasets.MNIST(mode='train')
#rezize到32×32，然后归一化到-1和1之间
dataset = paddle.vision.datasets.MNIST(mode='train', transform=transforms.Compose
									  ([transforms.Resize((32,32)),
                                        transforms.Normalize([127.5], [127.5])]))                                    
dataloader = paddle.io.DataLoader(dataset, batch_size=32,shuffle=True, num_workers=4)
```

2. Generator网络输入z经过四个转置卷积层之后，形状由[B,100,1,1]变成[B,1,32,32]。
```python
# Generator Code
class Generator(nn.Layer):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input is Z, [B,100,1,1] -> [B,64*4,4,4]
            nn.Conv2DTranspose(100,64*4,4,1,0, bias_attr=False),
            nn.BatchNorm2D(64*4),
            nn.ReLU(True),
            # state size. [B,64*4,4,4] -> [B,64*2,8,8]
            nn.Conv2DTranspose(64*4,64*2,4,2,1, bias_attr=False),
            nn.BatchNorm2D(64*2),
            nn.ReLU(True),
            # state size. [B,64*2,8,8] -> [B,64,16,16]
            nn.Conv2DTranspose(64*2,64,4,2,1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(True),
            # state size. [B,64,16,16] -> [B,1,32,32]
            nn.Conv2DTranspose(64,1,4,2,1, bias_attr=False),
            nn.Tanh()#最后输出值在-1到1之间
        )

    def forward(self, x):
        return self.gen(x)
```
2. Discriminator网络输入x经过四个转置卷积层之后，形状由[B,1,32,32]变成[B,1]。
```python
class Discriminator(nn.Layer):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(

            # input [B,1,32,32] -> [B,64,16,16]
            nn.Conv2D(1,64,4,2,1, bias_attr=False),
            nn.LeakyReLU(0.2),

            # state size. [B,64,16,16] -> [B,128,8,8]
            nn.Conv2D(64,64*2,4,2,1, bias_attr=False),
            nn.BatchNorm2D(64*2),
            nn.LeakyReLU(0.2),

            # state size. [B,128,8,8] -> [B,256,4,4]
            nn.Conv2D(64*2,64*4,4,2,1, bias_attr=False),
            nn.BatchNorm2D(64*4),
            nn.LeakyReLU(0.2),

            # state size. [B,256,4,4] -> [B,1,1,1] -> [B,1]
            nn.Conv2D(64*4,1,4,1,0,bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)
```

```python
#定义的初始化函数weights_init略去了
netG = Generator()
netG.apply(weights_init)
netD = Discriminator()
netD.apply(weights_init)
```

```python
loss = nn.BCELoss()#二分类损失函数
# 创建噪声
fixed_noise = paddle.randn([32, 100, 1, 1], dtype='float32')
# 设置真实图片和生成图片的标签
real_label ,fake_label= 1.,0.

# 设置两个优化器，训练一个网络时固定另一个网络的权重
optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=0.0002, beta1=0.5, beta2=0.999)
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=0.0002, beta1=0.5, beta2=0.999)
```

```python
losses = [[], []]
#plt.ion()
now = 0
for pass_id in range(100):
    for batch_id, (data, target) in enumerate(dataloader):
        """
        (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        optimizerD.clear_grad()#梯度清零
        real_img = data
        bs_size = real_img.shape[0]
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype='float32')#判别器真实图片标签为1
        real_out = netD(real_img)
        errD_real = loss(real_out, label)
        errD_real.backward()

		"""
		生成器根据噪声生成图片，并且把标签设为0
		"""
        noise = paddle.randn([bs_size, 100, 1, 1], 'float32')
        fake_img = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), fake_label, dtype='float32')
        fake_out = netD(fake_img.detach())
        errD_fake = loss(fake_out,label)
        errD_fake.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        errD = errD_real + errD_fake
        losses[0].append(errD.numpy()[0])

        """
        (2) Update G network: maximize log(D(G(z)))
        唯一不同是生成器生成的图片标签改为1，因为生成器要生成接近真实的图片
        """
        optimizerG.clear_grad()
        noise = paddle.randn([bs_size, 100, 1, 1],'float32')
        fake = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype=np.float32,)
        output = netD(fake)
        errG = loss(output,label)
        errG.backward()
        optimizerG.step()
        optimizerG.clear_grad()

        losses[1].append(errG.numpy()[0])


        """
        每一百步做一次可视化，打印输出
        """
        if batch_id % 100 == 0:
            generated_image = netG(noise).numpy()
            imgs = []
            plt.figure(figsize=(15,15))
            try:
                for i in range(10):
                    image = generated_image[i].transpose()
                    image = np.where(image > 0, image, 0)
                    image = image.transpose((1,0,2))
                    plt.subplot(10, 10, i + 1)
                    
                    plt.imshow(image[...,0], vmin=-1, vmax=1)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(pass_id, batch_id, errD.numpy()[0], errG.numpy()[0])
                print(msg)
                plt.suptitle(msg,fontsize=20)
                plt.draw()
                plt.savefig('{}/{:04d}_{:04d}.png'.format('work', pass_id, batch_id), bbox_inches='tight')
                plt.pause(0.01)
            except IOError:
                print(IOError)
    paddle.save(netG.state_dict(), "work/generator.params")
```
训练结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/562cb03d743afecaaaf9f3580ed34a67.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9ebc2d315eb9fccf502033deedc1ed0c.png)
### 1.5 PaddleGAN介绍
>[paddle官网](https://www.paddlepaddle.org.cn/paddlegan)、[github地址](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)。官网没啥用，主要看github。

paddle代码仓库结构预览：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d7021566b72e0a2d7e9d34acaff9985a.png)
## 二、day2：GAN的技术演进及人脸生成应用
### 2.1 GAN技术的演进
#### 2.1.1 GAN和DCGAN的问题
GAN和DCGAN存在以下问题：
- 模式坍塌：生成器生成非常窄的分布，仅覆盖真实数据分布中的单一模式。生成器只能生成非常相似的样本（比如MNIST中的单个数字），多样性不够。
- 没有指标可以告诉我们收敛情况。生成器和判别器的 loss并没有告诉我们任何收敛相关信息
- 训练不稳定

模式坍塌的原因一句话概括就是：<font color='red'>等 价优化的距离衡量（KL散度、JS散度）不合理，生成器随机初始化后的生成分布很难与真实分布有不可忽略的重叠。</font>
1. GAN网络训练的重点在于均衡生成器与判别器，我们越训练判别器，它就越接近最优。<font color='red'> 在最优判别器的下，我们可以把原始GAN定义的生成器loss等价变换为最小化真实分布与生成分布之间的JS散度。</font> （推导见下图）
2. JS散度存在的问题：通过优化JS散度就能将生成分布拉向真实分布，最终以假乱真，前提是两个分布有所重叠。但是如果两 个分布完全没有重叠的部分，或者它们重叠的部分可忽略， 那它们的JS散度就一直是 log2
3. 生成器随机初始化后的生成分布很难与真实分布有不可忽略的重叠（上升到高维时），所以在判别器太强时，梯度为0，loss没不再下降，生成器学习不到东西，生成图像的质量便不会再有提升。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4f813d20dc5de21f59db653802dd9e8e.png)
目标函数推导：
- 设真实数据分布$P_{r}(x)=a$,生成数据分布为$P_{G}(x)=b$，通过导数求极值，最终可以得到判别器函数的极值点$D^{*}(x)$。
- 将$D^{*}(x)$代入生成器目标函数中（只有后一项），根据KL散度和JS散度公式，可以得到生成器函数为-2log2+JS散度值。-2log2是因为log式子中分母除以2
- <font color='red'> 所以最终判别器收敛到接近最优点$D^{*}(x)$时，生成器函数是常数加上生成分布和真实分布之间的JS散度</font>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/55f062b40ebea18d6322584d116ef06c.png)
&#8195;&#8195;而JS散度的问题是：两个不重合分布的JS散度等于常数log2，梯度为0，网络无法继续优化。
#### 2.1.2 LSGAN：MSE损失函数代替二分类损失函数
>论文：[Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)

&#8195;&#8195;针对GAN存在的JS散度导致的问题，LSGAN（LeastSquare GAN）提出用MSE损失函数代替二分类损失函数，改善了传统 GAN 生成的图片质量不高，且训练过程十分不稳定的问题。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/26d7c1b75401e749fbb210596401e489.png)
&#8195;&#8195;[训练营第二课作业《代码题 DCGAN改写LSGAN》](https://aistudio.baidu.com/aistudio/projectdetail/4268463)中需要改的代码就两处：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b2bcd406ed59fe40936b94977868a3fc.png)

#### 2.1.3 WGAN和WGAN-GP：EM距离代替JS，KL散度
>参考：[论文Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)、[代码链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/wgan_mnist.yaml)、论文解读[《WGAN(Wasserstein GAN)看这一篇就够啦，WGAN论文解读》](https://blog.csdn.net/m0_62128864/article/details/124258797?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165601682516781667878692%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165601682516781667878692&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124258797-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=WGAN&spm=1018.2226.3001.4187)

1. WGAN

&#8195;&#8195;WGAN利用EM距离代替JS，KL散度来表示生成与真实分布的距离衡量，从而改进了原始GAN存在的两类问题。（Wasserstein距离 优越性在于： 即使两个分布没有任何重叠，也可以反应他们之间的距离。）

&#8195;&#8195;假设真实分布是$P_r$，生成器分布是$P_\theta$，两种分布就像两堆土，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a9302ac0b88305ec662b465cefbc0bf8.png)
&#8195;&#8195;将右边土堆堆成左边土堆的方式有无数种，其中一种消耗最少的称为推土机距离EM（Earth-Moverdistance）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ee79126637b150d4196b1dc520cd89f0.png)
 - 推土机距离公式代入GAN网络经过一堆推导得到中间那行式子，其中判别器D要满足$D\in 1-lipschitz$限制。这个限制直观来说会让生成器的标签足够平滑，即输出的变化要小于输入的变化。
 - 输入x是不好限制的，那么可以限制参数w。在神经网络中的实现就是判别器参数截断，即w∈[c,-c]，用clip即可实现。
 - WGAN与原始GAN第一种形式相比，只改了四点：
	 - 判别器最后一层去掉sigmoid 
	 - 生成器和判别器的loss不取log 
	 - 每次更新判别器的参数之后把它们的值截断到不超过一个 固定常数c
 	 - 不要用基于动量的优化算法（包括momentum和 Adam），推荐RMSProp

2. WGAN-GP
在神经网络中，w即使很小，累积多层之后输出也可能很大，不能保证输入一定小于输出，由此提出WGAN-GP,其目标函数如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d9d086d9e9e8778bfafe3e1255e3eb3e.png)
如上图所示，目标函数改成第二行的式子。其中：
- $x \sim P_{penalty}$表示$P_r$和$P_G$之间的采样。
- $\left \| \bigtriangledown _{x} D(x)\right \|$表示判别器输出分数对x的导数的范数。
- WGAN-GP目标函数第三项表示希望$\left \| \bigtriangledown _{x} D(x)\right \|$小于1。如果大于1那么max之后得到一个正值，前面乘以-λ作为惩罚。
- interpolates就是上图的$x \sim P_{penalty}$采样，最终得到的惩罚项gradient_penalty作为损失。
- 代码最后和式子有点不一样，是作者觉得这样效果更好。
### 2.2 GAN在人脸生成的改进
- 从2014年的GAN、2015年DCGAN、2017年PGGAN、到2018年的StyleGAN，GAN生成的图片越来越清晰。DCGAN要生成高分辨率一点的图片，发现会生成一些很奇怪的图片，分辨率继续扩大，问题会越来越明显。
- PGGAN损失函数使用了WGAN-GP的损失函数，网络结构如下：（左边是DCGAN生成的奇怪人脸）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0b954ce1f87b56a5b2d59d451719d583.png)

#### 2.2.1 渐近式增长生成对抗网络PGGAN
>参考：[论文PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/pdf/1710.10196.pdf)

##### 2.2.1.1 渐进式增长
&#8195;&#8195;如果直接生成大分辨率的图片，建立从latent code 到 1024x1024 pixels样本的映射网络G，肯定是很难工作的。因为，在生成的过程中， 判别器D很容易就可以识别出G生成的“假图像”，G难以训练 。因此，提出PGGAN（progressive gan）来进行逐层训练。
&#8195;&#8195;这项技术首先通过学习即使在低分辨率图像中也可以显示的基本特征，来创建图像的基本部分，并且随着分辨率的提高和时间的推移，学习越来越多的细节。由于每次前面的层已经训练好，所以会集中训练后添加的层，所以提高分辨率后，新的训练难度不会提高。低分辨率图像的训练不仅简单、快速，而且有助于更高级别的训练，因此，整体的训练也就更快。
&#8195;&#8195;如下图所示，模型先训练一个生成4\*4分辨率图片的的生成器和对应的判别器，效果不错之后再添加一层，训练8\*8分辨率的生成器和判别器。。。。。。不断逐层添加卷积层和转置卷积层，最终得到分辨率为1024*1024的生成对抗网络。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4fef4aaea0babc09c709d260f86bb47c.png)
PGGAN网络结构如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27d848b180104de93400e761949391c5.png)
##### 2.2.1.2 平滑过度：
- Generator 内部的网络只有一个，但是在训练过程中网络的结构是在动态变化的。引入这些层时，不是立即跳到该分辨率，而是通过参数α（介于0-1之间，从0到1线性缩放）平滑的增加高分辨率的新层。
- 如果从 4×4 的输出直接变为 8×8 的输出的话，网络层数的突变会造成 GANs 原有参数失效，导致急剧不稳定这会影响模型训练的效率（新添加的层，参数一开始是初始化的。如果直接输出，那么之前训练好的结果也被破坏了）。所以PGGAN 提出了平滑过渡技术。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f32613099bd5e39c3a2dd20cb4bf058d.png)
&#8195;&#8195;当把生成器和判别器的分辨率加倍时，会平滑地增大新的层。我们以从16 × 16 像素的图片转换到 32 × 32 像素的图片为例。在转换（b）过程中，把在更高分辨 率上操作的层视为一个残缺块，权重 α 从 0 到 1 线性增长。当 α 为 0 的时候，相当于图(a),当 α 为 1 的时候，相当于图(c)。所以，在转换过程中，生成样本的像素，是从 16x16 到 32x32 转换的。同理，对真实样本也做了类似的平滑过渡，也就是，在这个阶段的某个 训练 batch，真实样本是:
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/541b480c4f68487b583b764862231a0f.png)
- 上图中的 2× 和 0.5× 指利用最近邻卷积和平均池化分别对图片分辨率加倍和折半。
- toRGB 表示将一个层中的特征向量投射到 RGB 颜色空间中，
- fromRGB 正好是相反的过程； 这两个过程都是利用 1 × 1 卷积。

&#8195;&#8195;当训练判别器时，插入下采样后的真实图片去匹配网络中 的当前分辨率。在分辨率转换过程中，会在两张真实图片的分辨率之间插值，类似于将两个分辨率结合到一起用生成器输出。其它改进还有：
- 生成器中的像素级特征归一化。 动机是训练的稳定性，训练发散的早期迹象之一是特征的爆炸式增长，将图像中的所有点映射到一组向量，然后对其进行归一化 。
-  小批量标准差（仅应用判别器）、均衡学习率：略
##### 2.2.1.3 PPGAN的缺陷：特征纠缠
&#8195;&#8195;由于 PPGAN 是逐级直接生成图片，我们没有对其增添控制，我们也就无法获知它在每一级上学 到的特征是什么，这就导致了它<font color='red'> 控制所生成图像的特定特征的能力非常有限，即PPGAN 容易发生特征纠缠。</font>换句话说，这些特性是互相关联的，因此尝试调整一下输入，即使是一点儿，通常也会同时影响多个特性。
&#8195;&#8195;如下图，比如我们希望噪声第二个维度可以控制人脸的肤色，理想是第二维向量由0.9改为0.7之后，会生成第二张图片。但是结果可能生成完全不一样的图片，比如第三张图，这就是相互纠缠的一个例子。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd151a29f283dc0dbdb008c68e2cea9d.png)
&#8195;&#8195; 我们希望有一种更好的模型，能让我们控制住输出的图片是长什么样的，也就是在生成 图片过程中每一级的特征，要能够特定决定生成图片某些方面的表象，并且相互间的影响尽 可能小。于是，在 PPGAN 的基础上，StyleGAN 作出了进一步的改进与提升。
##### 2.2.1.4 PPGAN的TF实现
```python
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
 
with tf.Graph().as_default():
    # 提前从TFHub导入PGGAN
    module = hub.Module("progan-128_1")
    #运行时采样的维度
    latent_dim = 512
 
    # 改变种子得到不同的人脸
    latent_vector = tf.random.normal([1, latent_dim], seed=1337)
 
    # 使用该模块从潜在空间生成图像
    interpolated_images = module(latent_vector)
 
    # 运行Tensorflow session 得到（1，128，128，3）的图像
    with tf.compat.v1.Session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      image_out = session.run(interpolated_images)
 
plt.imshow(image_out.reshape(128,128,3))
plt.show()
```

#### 2.2.2 StyleGAN：基于样式的生成对抗网络
>参考：
>-  论文[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
>- [《StyleGAN 架构解读（重读StyleGAN ）精细》](https://blog.csdn.net/weixin_43135178/article/details/116331140?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165606448816782388023407%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165606448816782388023407&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-116331140-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=StyleGAN&spm=1018.2226.3001.4187)

&#8195;&#8195;PGGAN的问题：<font color='red'> 控制生成图像特定特征的能力有限。 </font>以下图来说：
- 图a表示，假设真实数据中有两个人脸特征，x轴越往左表示越man；y轴越往上表示头发越长。一般认为不存在头发长又很man的人，所以左上角区域是不存在的。
- b图表示噪声分布，噪声一般是从简单对称分布中取出，所以其区域是一个圆形。为了填补左上角空缺，就会对特征分布做一个扭曲。<font color='deeppink'> 这样当图片向量仅仅改变一个维度时，输出图片的多个特征都会变化，这就是特征纠缠现象。</font>
- c图表示，<font color='deeppink'>StyleGAN引入映射网络之后，会拟合真实数据分布的形状，缓解特征纠缠。</font>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4c3d3455dff6a0e33c4e8f155902d334.png)
##### 2.2.2.1 StyleGAN 总览
&#8195;&#8195;<font color='red'> StyleGAN 用风格（style）来影响人脸的姿态、身份特征等，用噪声 ( noise ) 来影响头发丝、皱纹、肤色等细节部分。</font>StyleGAN 的网络结构包含两个部分：映射网络Mapping network和Synthesis network。
&#8195;&#8195;Mapping network，即下图 (b)中的左部分，由隐藏变量 z 生成 中间隐藏变量 w的过程，这个 w 就是用来控制生成图像的style，即风格。
&#8195;&#8195;Synthesis network，它的作用是生成图像，创新之处在于给每一层子网络都喂了 A 和 B，A 是由 w 转换得到的仿射变换，用于控制生成图像的风格，B 是转换后的随机噪声，用于丰富生成图像的细节，即每个卷积层都能根据输入的A来调整"style"，通过B来调整细节。
&#8195;&#8195;整个网络结构还是保持了 PG-GAN （progressive growing GAN） 的结构。最后论文还提供了一个高清人脸数据集FFHQ。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/45749426ca8a70149fc441ed4efaa109.png)
架构解读：
&#8195;&#8195;StyleGAN 首先重点关注了 ProGAN 的生成器网络，它发现，渐进层的一个的好处是，如果使用得当，它们能够控制图像的不同视觉特征。层和分辨率越低，它所影响的特征就越粗糙。简要将这些特征分为三种类型：

>1、粗糙的——分辨率不超过82，影响姿势、一般发型、面部形状等；
2、中等的——分辨率为162至322，影响更精细的面部特征、发型、眼睛的睁开或是闭合等；
3、高质的——分辨率为642到10242，影响颜色（眼睛、头发和皮肤）和微观特征。

&#8195;&#8195;然后，StyleGAN 就在 ProGAN 的生成器的基础上增添了很多附加模块以实现样式上更细微和精确的控制。
##### 2.2.2.2 映射网络：为输入向量的特征解缠提供一条学习的通路
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dc7022a0b3e1d304d3419a28b3a41a7d.png)

&#8195;&#8195;<font color='red'>StyleGAN的第一点改进是:Mapping network 对隐藏空间（latent space）进行解耦,缓解特征纠缠 </font>。Generator的输入加上了由8个全连接层组成的Mapping Network，并且 Mapping Network 的输出W′与输入层Z（512×1）的形状大小相同。中间向量W′（或者叫潜在因子）后续会传给生成网络得到 18 个控制向量，使得该控制向量的不同元素能够控制不同的视觉特征。
&#8195;&#8195;如果不加这个 Mapping Network 的话，后续得到的 18个控制向量之间会存在特征纠缠的现象——比如说我们想调节 8\*8 分辨率上的控制向量（假 设它能控制人脸生成的角度），但是我们会发现 32\*32 分辨率上的控制内容（譬如肤色）也被改变了，这个就叫做特征纠缠。所以 Mapping Network 的作用就是为输入向量的特征解缠提供一条学习的通路。
&#8195;&#8195;为何 Mapping Network 能够学习到特征解缠呢？简单来说，<font color='deeppink'> 如果仅使用输入向量来控制视觉特征，能力是非常有限的，因此它必须遵循训练数据的概率密度。</font>例如，如果黑头发 的人的图像在数据集中更常见，那么更多的输入值将会被映射到该特征上。因此，该模型无法将部分输入（向量中的元素）映射到特征上，这就会造成特征纠缠。然而，<font color='deeppink'> 通过使用另一个神经网络，该模型可以生成一个不必遵循训练数据分布的向量，并且可以减少特征之间的相关性。</font>
##### 2.2.2.3  Synthesis network样式模块：AdaIN精确控制样式信息，而保留图片的关键信息
&#8195;&#8195;<font color='red'> StyleGAN第二点改进是，将特征解缠后的中间向量W′变换为样式控制向量，从而参与影响生成器的生成过程。</font>AdaIN表示自适应实例归一化。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8b96efb1041ce57db7ee383af8d4ba61.png)
&#8195;&#8195;实例归一化是上图Instance Norm中，对蓝色部分进行归一化。每个batch中只取一个样本，计算其在每个通道上的均值$\mu (x)$和标准差$\sigma  (x)$，γ和β表示缩放因子和偏置。自适应归一化AdaIN是其变体。

&#8195;&#8195;上图右下是风格迁移任务的网络示意图，我们希望上面实景图有下面那张漫画图的风格。论文实验发现，在实例归一化中，将实景图的γ和β换成漫画图的均值和标准差，最终会取得比较好的风格迁移效果。这就是自适应归一化的过程。StyleGAN就借鉴了这一种思路。
>&#8195;&#8195;风格迁移任务更多细节，可以参考我另一篇帖子：[《动手深度学习13：计算机视觉——语义分割、风格迁移》](https://blog.csdn.net/weixin_43135178/article/details/116331140?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165606448816782388023407%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165606448816782388023407&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-116331140-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=StyleGAN&spm=1018.2226.3001.4187)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0ba51628011471b4b2c06554f9fde3fa.png)

&#8195;&#8195; AdaIN 的具体实现过程如上右图所示：将潜在因子W′通过一个可学习的仿射变换A（简单理解就是一个全连接层）后输出，输出扩大为原来的两倍（2×n），分别作为缩放因子$y_{s,i}$和偏差因子$y_{b,i}$。输入$x_i$进过标准化（减均值除方差）后，与两个因子进行AdaIN，就完成了一次W′影响原始输出$x_i$的过程。
&#8195;&#8195;  AdaIN 代码见左下，W′经过FC层之后变成原来两倍，reshape成前后两部分。这两部分分别作为两个因子，最后$x=y_{s,i}*x+y_{b,i}$。（x在AdaIN之前先标准化）

&#8195;&#8195;生成器从 分辨率4\*4，变换到 8\*8，并最终到 1024\*1024，一共由 9 个生成阶段组成，而每个阶段都会受两个控制向量（A）对其施加影响。其中一个控制向量在 Upsample之后对其影响一次，另外一个控制向量在 Convolution 之后对其影响一次，影响的方式都采用 AdaIN。因此，中间向量W′总共被变换成 18 个控制向量（A）传给生成器。
&#8195;&#8195;<font color='red'> 这种影响方式能够实现样式控制，主要是因为它让变换后的W′影响图片的全局信息（注意标准化抹去了对图片局部信息的可见性），而保留生成人脸的关键信息由上采样层和卷积层来决定，因此W′只能够影响到图片的样式信息。</font>

##### 2.2.2.4 常数输入(ConstantInput)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/812e84c3f86f4280eb1c3b66799b1cd5.png)
&#8195;&#8195;上图左侧网络表示传统的GAN网络输入是一个随机变量或者隐藏变量 z，右侧表示Synthesis network中最开始的输入变成了常数张量。
&#8195;&#8195;既然 StyleGAN 生成图像的特征是由 𝑊 ′ 和 AdaIN 控制的，那么生成器的初始输入可以 被忽略，并用常量值4×4×512输入替代（分辨率，通道数）。这样做的理由是，首先可以降低由于初始输入取值不当而生成出 一些不正常的照片的概率（这在 GANs 中非常常见），另一个好处是它有助于减少特征纠缠， 对于网络在只使用𝑊 ′ 不依赖于纠缠输入向量的情况下更容易学习。
&#8195;&#8195;左下代码是将input先定义为[batch_size=1,channel,size,size]，然后获取实际输入的batch_size，再对其进行铺开（tile函数），最终得到[input_batch_size,512,4,4]的输入。
##### 2.2.2.5 噪声输入改进
&#8195;&#8195;人脸很多小特征是随机性的，比如头发、皱纹、雀斑；不同时间、角度、地点都可能发生变化。将这些小特征插入 GAN 图像的常用方法是 在输入向量中添加随机噪声 （即通过在每次卷积后添加噪声 ）。为了控制噪声仅影响图片样式上细微的变化， StyleGAN 采用类似于 AdaIN 机制的方式添加噪声。
&#8195;&#8195;噪声输入是由不相关的高斯噪声组成的单通道数据，它们被馈送到生成网络的每一层。即在 AdaIN 模块之前向每个通道添加一个缩放过的噪声，并稍微改变其操作的分辨率级别特征的视觉表达方式。 加入噪声后的生成人脸往往更加逼真与多样 。
&#8195;&#8195;左下代码中weight表示可学习的缩放因子，初始化shape=1，value=0。noise从高斯分布中取得。
&#8195;&#8195;风格影响的是整体（改变姿势、身份等），噪音影响无关紧要的随机变化（头发、胡须等）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ca1bc0e25aa297b2bdea291a82c7c94b.png)
##### 2.2.2.6 混合正则化
&#8195;&#8195;StyleGAN 生成器在合成网络的每个层级中都使用了潜在因子，这有可能导致网络学习到这些层级是相关的。为了降低关联性，一个简单的想法是使用不同的潜在因子。论文中采用随机选择两个输入向量，映射后生成了两个潜在因子𝑊 ′ 。然后在所有网络层级中随机选取一个点，这个点之前的层级使用第一个它用第一个𝑊 ′，之后的层级使用第二个𝑊 ′。随机的切换确保了网络不会学习并依赖于一个合成网络级 别之间的相关性。下图代码中inject_index表示随机选取的点。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ecf6383c1235d74548cf99d5fe5d8718.png)
&#8195;&#8195;混合正则化并不会提高所有数据集上的模型性能，但是它能够以一种连贯的方式来组合多个图像。该模型生成了两个图像 A 和 B（第一行的第一张图片和第二行的第一张图片），然后通过从 A 中提取低级别的特征并从 B 中提取其余特征再组合这两个图像，这样能生成出混合了 A 和 B 的样式特征的新人脸 。
- Source A:gender,age,hair length,glasses,pose
- Source B: everything else
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/15e70089702dafb6d024de4170d7109c.png)

根据交叉点选取位置的不同，style组合的结果也不同。下图中分为三个部分，
>- 第一部分是 Coarse styles from source B，分辨率(4x4 - 8x8)的网络部分使用B的style，其余使用A的style, 可以看到图像的身份特征随souce B，但是肤色等细节随source A；
>- 第二部分是 Middle styles from source B，分辨率(16x16 - 32x32)的网络部分使用B的style，这个时候生成图像不再具有B的身份特性，发型、姿态等都发生改变，但是肤色依然随A；
> - 第三部分 Fine from B，分辨率(64x64 - 1024x1024)的网络部分使用B的style，此时身份特征随A，肤色随B。

&#8195;&#8195;由此可以 大致推断， 低分辨率的style 控制姿态、脸型、配件 比如眼镜、发型等style，高分辨率的style控制肤色、头发颜色、背景色等style。
#### 2.2.3 StyleGAN 2
>参考：
>- 论文[《Analyzing and Improving the Image Quality of StyleGAN》](https://arxiv.org/pdf/1912.04958.pdf)
>- [《StyleGAN2学习笔记》](https://blog.csdn.net/weixin_39538889/article/details/114947559?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165607759816782395337700%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165607759816782395337700&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-114947559-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=StyleGAN2&spm=1018.2226.3001.4187)、[代码](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/stylegan_v2_256_ffhq.yaml)

##### 2.2.3.1 消除伪影
&#8195;&#8195;StyleGAN 中，通过AdaIN实现特征解耦和风格控制，但是会带来水印问题，即生成的图片有水滴状伪影，在特征图上很明显。在StyleGAN2中，AdaIN被重构为权重解调(Weight Demodulation)。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fddc0bfa520f73c6464ba3ac3c42de07.png)

下图左侧是StyleGAN 结构，右图是StyleGAN2结构，可以看出：
- 移除初期常数
- normalization中不再需要mean，只计算std即可
- 将noise模块移除style box中
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/de758f5b2898344805109db42991cef9.png)
&#8195;&#8195;第二个不同是权重解调。Mod表示可学习的放射变换A，Std表示除以标准差。StyleGAN中把mod std和卷积Conv参数结合在一起，即下图蓝色框的公式。<font color='red'> 权重解调就是对权重做归一化，即红色框的式子。StyleGAN2认为AdaIN的做法有问题，所以把AdaIN去掉了，而是使用权重归一化。即对i,j,k是三个维度归一化。</font>这样做之后，伪影就都消除了。
&#8195;&#8195;尽管这种方式与Instance Norm在数学上并非完全等价，但是weight demodulation同其它normalization 方法一样，使得输出特征图有着standard的unit和deviation。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a670457a279d3b1e645e2ba35432897a.png)
代码如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6a0cd3352a84e0497fa90dad5e61f12a.png)

&#8195;&#8195;modelation是可学习的放射变换（FC层），scale是和weight形状有关的一个固定值，style是仿射变换之后的潜在因子。demodulate就是解调部分，rsqrt(x)就是x平方的导数。加一个小的ϵ 是为了避免分母为0，保证数值稳定性
##### 2.2.3.2 改进渐进式增长网络结构
&#8195;&#8195;StyleGAN2作者发现，生成图片时，部分细节不随主体变化而变化。例如下图的牙齿，在人脸变化后还是保持不变。作者认为在逐步增长的过程中，每个分辨率都会瞬间用作输出分辨率，迫使其生成最大频率细节，然后导致受过训练的网络在中间层具有过高的频率。神经网络中要产生细节充足和高频率的图片，那么网络的参数频率也要很高，从而损害了位移不变性。
&#8195;&#8195;作者根据MSG-GAN设计了b和c两种结构，解决了这个问题。（实验中人脸的眼珠子会转了，牙齿也会变化）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3385248a093951c707d184fe726f7546.png)
&#8195;&#8195;在生成方法的背景下，Skip connections，残差网络和分层方法也被证明是非常成功的。三种生成器（虚线上方）和判别器体系结构如上图。Up和Down分别表示双线性上和下采样。 在残差网络中，这些还包括1×1卷积以调整特征图的channel数。tRGB和fRGB在RGB和高维每像素数据之间转换。 Config E和F中使用的体系结构以绿色突出显示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/76af1f18ccc3f60dff99cb0904d5d995.png)
&#8195;&#8195;我们可以看到，从一开始，网络就专注于低分辨率图像，并随着训练的进行逐渐将其注意力转移到较大分辨率上。
- 在（a）中，生成器基本上输出512x512图像，并对1024x1024进行一些细微锐化。
- 在（b）中，较大的网络更多地关注高分辨率细节。通过将两个网络的最高分辨率层中的特征图的数量加倍来进行测试，这使行为更加符合预期。图（b）显示了贡献的显著增加。

总结
- 使用Weight demodulation代替AdaIN
- 发现PPL与生成图像质量的关系（略）
- 去除渐进式网络，在生成器和判别其中采用不同的网络结构

##### 2.2.3.3 StyleGAN2的应用体验
应用体验教程参考：[StyleGAN V2](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/styleganv2.md)
使用方法：
1. 用户使用如下命令中进行生成，可通过替换seed的值或去掉seed生成不同的结果：

```python
cd applications/
python -u tools/styleganv2.py \
       --output_path <替换为生成图片存放的文件夹> \
       --weight_path <替换为你的预训练模型路径> \
       --model_type ffhq-config-f \
       --seed 233 \
       --size 1024 \
       --style_dim 512 \
       --n_mlp 8 \
       --channel_multiplier 2 \
       --n_row 3 \
       --n_col 5 \
       --cpu
```
weight_path可以不设置，会默认下载已经训练好的权重。

2. 训练模型、推理（略）
### 2.3 PaddleGAN的使用
>参考[《PaddleGAN》](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)

&#8195;&#8195;关于PaddleGAN的代码、各种应用，可以参考github资源上的教程，例如：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6a00ba3e7a50f51c6894513bb621b3d9.png)
&#8195;&#8195;或者有时间我会再写个应用的笔记。
## 三、day3：图像翻译及人像卡通化
### 3.1 背景介绍
&#8195;&#8195;卡通画一直以幽默、风趣的艺术效果和鲜明直接的表达方式为大众所喜爱。近年来，随着多部动漫电影陆续成为现象级爆款，越来越多的人开始在社交网络中使用卡通画作为一种表意的文化载体。人们对于定制卡通画的需求与日俱增，然而高质量的卡通画需要经验丰富的画师精心绘制，从线稿设计到色彩搭配，整个流程耗时费力，对于大众而言购买成本较高。（淘宝上这种服务的店铺众多）
&#8195;&#8195;定制卡通画痛点：耗时长、成本高、要求高的话可能需要反复沟通修改、涉及隐私。
&#8195;&#8195;计算机生成卡通画任务要点：图像精美好看、男女老少都覆盖且保留其鲜明特点、卡通画和原照片有相同的身份信息(长得像）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4131d8edf8f37aa460f686f3792e5a7d.png)

&#8195;&#8195;图像翻译：指从一副图像到另一副图像的转换。可以类比机器翻译，一种语言转换为另一种语言。下图就是一些典型的图像翻译任务：比如语义分割图转换为真实街景图，灰色图转换为彩色图，白天转换为黑夜......（Pixel2Pixel的效果图，下面会讲到）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2c2ded7d2a11921d1601d76ed7587279.png)
>图像翻译的三个比较经典的模型pix2pix，pix2pixHD, vid2vid。可参考[《图像翻译三部曲：pix2pix, pix2pixHD, vid2vid》](https://blog.csdn.net/Yong_Qi2015/article/details/112130735?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165612861816781683913092%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165612861816781683913092&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-112130735-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=%E5%9B%BE%E5%83%8F%E7%BF%BB%E8%AF%91&spm=1018.2226.3001.4187)

本课任务就是将人物画翻译为动漫画。
### 3.2 技术原理
&#8195;&#8195;鉴别器可以当做是一种可自行优化的损失函数，训练完生成器就可以丢掉了。
&#8195;&#8195;GAN中的噪声是随机的，无法控制生成器生成哪一种特征的图片。所以提出了Conditional GAN。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d9f4f9d9536f2480f38d8cc62f37e435.png)
#### 3.2.1 Conditional GAN
>论文：[《Conditional Generative Adversarial Nets》](https://arxiv.org/pdf/1411.1784.pdf)、[代码](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/cond_dcgan_mnist.yaml)

&#8195;&#8195;Conditional GAN希望可以控制GAN 生成的图片，而不是单纯的随机生成图片。具体地，Conditional GAN 在生成器和判别器的输入中增加了额外的条件信息y，生成器生成的图片只有足够真实且与条件y相符，才能够通过判别器。
&#8195;&#8195;条件信息y，可以是类别标签 或者是 其他类型的数据，使得 图像生成能够朝规定的方向进行。

网络模型：
- 在生成器中，作者将输入噪声 z 和 y 连在一起隐含表示，而对抗性训练框架在如何构成这种隐藏表示上具有相当大的灵活性。
- 损失函数：和GAN区别是，在在生成器和判别器都加入条件信息y。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ea2a4b7f544006597194592e488be350.png)
论文在MNIST数据集上结果：（类别标签的one-hot编码作为条件信息y，控制生成的数字）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e5998e4579cf815a863c54c86f9abb0d.png)
&#8195;&#8195;Conditional GAN的想法在各方面的细节上，比如条件y的具体内容（类别标签、实际的图片…），生成器、判别器中条件y的表示方式，判别器的打分方式（真实度和条件符合度放在一起打还是分开来打）等，有各种实现形式，因而延伸出了丰富的应用。：text-to-image（文本生成图像）、image-to-image（图像转换）、Speech Enhancement（语音增强 ）、Video Generation（视频生成）等。
&#8195;&#8195;结合人像卡通画任务考虑，如果条件信息是一张卡通画，输入真实照片能不能引导模型输出人像卡通画呢？
#### 3.2.2 pixel2pixel
>论文[《Image-to-Image Translation with Conditional Adversarial Networks》](https://arxiv.org/pdf/1611.07004.pdf)、[代码](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/pix2pix_cityscapes_2gpus.yaml)
>更多原理参考[《图像翻译三部曲：pix2pix, pix2pixHD, vid2vid》](https://blog.csdn.net/Yong_Qi2015/article/details/112130735?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165612861816781683913092%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165612861816781683913092&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-112130735-null-null.142%5Ev21%5Econtrol,157%5Ev15%5Enew_3&utm_term=%E5%9B%BE%E5%83%8F%E7%BF%BB%E8%AF%91&spm=1018.2226.3001.4187)

pix2pix是一个经典的图像翻译模型,使用成对数据进行训练。
- 模型使用的训练数据是真实人像以及画师画的对应人像卡通画（像素级对应）。真人像要收集不同光照、姿态表情的图像，提高鲁棒性。真人像好收集，数据量较大。
- 生成器输入是真人照片，输出是卡通画，输入输出图片轮廓位置信息相同，采用的结构是U-Net，适合传递位置信息。
- 判别器输入类似Conditional GAN，是将真人照和卡通画在通道维度拼接，然后判断卡通画是真实的还是生成的，以及判断真人照和卡通画是否相符（成对）。
- 判别器采用Patch GAN，输出是一个单通道的特征图，而不是GAN中的一个判断真假的概率值。特征图中每个元素值表示卡通图每个小区域的逼真程度，Pacth就是块的意思。
- 损失函数采用L1l oss，用真实卡通画约束生成卡通画
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f330bc8c4bc4510725f7bff13515c823.png)

&#8195;&#8195;pixel2pixel可用于生成街景、建筑物、黑白图→彩色图、线稿→实物等等（见上一节效果图）。
&#8195;&#8195;有些任务的成对数据是容易收集的，比如用技术手段将很多彩色图转为少见的黑白图，扩大了黑白图数据的规模，这样黑白-彩色成对数据就容易收集了。但是有些任务的成对数据很难通过简单的技术手段收集。CycleGAN就是一种基于非成对数据的图像翻译方法。

这里解释一下成对数据和非成对数据：
&#8195;&#8195;成对数据：两组数据有相似度级别的对应，比如图片的风格和纹理可以不同，但是位置信息要是一致的，比如脸型、五官等空间信息一致。成对数据一般获取难度较大。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fe08de67e2b6f459612a614a3fb76784.png)

#### 3.2.3 CycleGAN
>论文：[《Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks》](https://arxiv.org/pdf/1703.10593.pdf)、[代码](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/cyclegan_horse2zebra.yaml)

&#8195;&#8195;简单来说， CycleGAN功能就是：自动将某一类图片转换成另外一类图片。CycleGAN不需要配对的训练图像。当然了配对图像也完全可以，不过大多时候配对图像比较难获取。所以CycleGAN可以做配对图像转换，也可以做图像从一个模式到另外一个模式的转换，转换的过程中，物体发生了改变，比如从猫到狗，从男人到女人。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/66d306463afcf72f255d6596f6de2fff.png)
CycleGAN结构如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aebd61eb46d1797b5f6b03f64a7b6184.png)
CycleGAN其实是由两个判别器(Dx和Dy)以及两个生成器(G和F）组成。
- G将X域图像映射到Y域，本任务就是真人照映射到卡通画
- F将Y域图像映射到X域，本任务就是卡通画映射到真人照
- Dx和Dy分别判断两个域的图像的真假。
- b、c是模型的两个部分，都是将输入x转换过去又转换回来，生成新的图像。通过输入输出图像的loss（L1或L2都行）约束生成图像在结构上不发生大的变化。
&#8195;&#8195;为什么要连两个生成器和两个判别器呢？论文中说，是为了避免所有的X都被映射到同一个Y，比如所有男人的图像都映射到范冰冰的图像上，这显然不合理，所以为了避免这种情况，论文采用了两个生成器的方式，既能满足X->Y的映射，又能满足Y->X的映射，这一点其实就是变分自编码器VAE的思想，是为了适应不同输入图像产生不同输出图像


&#8195;&#8195;CycleGAN虽然可以使用非成对数据训练，但是两个域的目标要规定好，即每个域的图像要具有一定的规则。比如X域图像都是油画，风格越统一越好。
&#8195;&#8195;CycleGAN缺陷：缺少有监督信息（pixel2pixel有成对数据，位置信息对应），所以需要的数据量会更多，收敛也更慢。
>代码链接[《CycleGAN算法原理（附源代码，可直接运行）》](https://blog.csdn.net/qq_29462849/article/details/80554706)

#### 3.2.4 U-GAT-IT
>论文[《U-GAT-IT: UNSUPERVISED GENERATIVE ATTENTIONAL NETWORKS WITH ADAPTIVE LAYERINSTANCE NORMALIZATION FOR IMAGE-TO-IMAGE TRANSLATION》](https://arxiv.org/pdf/1907.10830.pdf)、[代码](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/ugatit_selfie2anime_light.yaml)

U-GAT-IT接近本次人像生成动漫画的任务，结构继承了CycleGAN的设计，也是有两个生成器和判别器，下图简化，只展示一个生成器；而loss有四种。
- 生成器中，编码器提取输入图像特征，解码器将特征转为动漫图。
- GAN loss，是为了消除模糊，使图像更精美，分辨率更高。
- Cycle loss：输入转为动漫图再转回来，用L1 loss约束转换前后的图像，使生成的图像结构不发生大的变化。
- Identity loss：输入输出都是动漫图，约束输入输出颜色尽量相似
- CAM loss：二分类loss。编码器提取特征后有一个网络分支，特征输入全连接层做一个分类任务（真人or动漫）。全连接层权重拿出乘以特征权重，类似attention机制，注意力放在需要重点关注的特征上。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6bb485c070a43ddd7383d33cc9a07fb5.png)
&#8195;&#8195;U-GAT-IT论文一大贡献是AdaLIN，自适应实例归一化。输入是解码器各层特征，经过IN和LN之后使用可学习的ρ加权求和，然后使用MLP提取的统计特征γ和β来归一化。作者发现IN更关注内容信息，LN更关注全局信息，ρ可以自动调整二者比重。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/abe286d160dbbbeb91dcc52f9621df10.png)
以下是U-GAT-IT的一些应用举例，对比CycleGAN有比较明显的提升。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5e8dfdec970912ac5d23a47112ffb8e9.png)
&#8195;&#8195;U-GAT-IT的缺点是转换为比较夸张的漫画之后，无法辨认出漫画的真实身份信息。要保留身份信息，实现写实风格的卡通画，有以下难点：
- 卡通图像往往有清晰的边缘，平滑的色块和经过简化的纹理，与其他艺术风格有很大区别。<font color='deeppink'>使用传统图像处理技术生成的卡通图无法自适应地处理复杂的光照和纹理，效果较差；基于风格迁移的方法无法对细节进行准确地勾勒。 </font>(如果使用传统图像处理技术，模型鲁棒性会比较差。因为我们会人为加一些阈值、设定的规则或者自己设计的参数，这样在复杂的光线或背景的场景下容易失控。基于神经网络的风格迁移算法在风景、建筑等宏观场景比较适用。而人像处理的风格化要求比较精细，眼睛多一笔少一笔最终观感影响很大）
- 数据获取难度大。绘制风格精美且统一的卡通画耗时较多、成本较高，且卡通画和原照片的脸型及五官形状有差异，因此不构成像素级的成对数据，难以采用基于成对数据的图像翻译（Paired Image Translation）方法。
- 照片卡通化后容易丢失身份信息。基于非成对数据的图像翻译（Unpaired Image Translation）方法中的循环一致性损失（Cycle Loss）无法对输入输出的ID进行有效约束。（CycleGAN只能保证输入输出形状不发生明显变化，但是无法有效约束五官位置形状等）

解决方法：Photo2Cartoon
#### 3.2.5 Photo2Cartoon
##### 3.2.5.1 Photo2Cartoon模型结构
Photo2Cartoon生成器有以下三个部分：
- 特征提取：提取真人图像不同尺度的特征；
- 特征融合：不同尺度特征和高层特征融合，反归一化到解码特征中；
- 特征重建：解码器将融合特征重建为卡通形象。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/74d7c35b22c009b92630bd4cf99be19d.png)
Photo2Cartoon希望既可以生成精美的卡通画，又可以保留身份信息可以识别，所以做了三个设计：
- 在输入输出部分都加了两个Hourglass 模块，强化特征提取和重建，用于提取内容不变性的特征；
- Face ID Loss：约束卡通画的身份信息。使用预训练的人脸模型，提取输入输出的ID特征，并用余弦距离进行约束。实验中Face ID Loss可以明显提升五官相似性。
- Soft-AdaLIN：为了更好的利用不同尺度的编码特征，将不同尺度编码特征和高层CAM特征融合，再应用于解码特征中。底层特征中有丰富的纹理、色彩等信息，这样做可以更好的将照片信息迁移到卡通画上。

##### 3.2.5.2 递进训练
&#8195;&#8195;Photo2Cartoon生产落地时，面临多种多样的人脸数据。在绘制训练数据时，需要为不同类型的人群设计不同的风格。比如小朋友可以加红晕，更可爱。女青年睫毛更长，男性有胡须，老年人皱纹更明显。如果这些数据混合训练，会导致最终输出风格不确定。（比如输入短发女生，可能会匹配到男性风格，用户体验差）

&#8195;&#8195;如果分开训练，由于数据获取成本高，每个类别数据量更少。所以采用了递进训练，这样即使某一类数据匮乏，也能得到很好的训练效果。训练过程如下：
- 先所有数据混合训练，得到基础模型
- 根据年龄将数据分为少年、青年、老年三个部分，基于基础模型分别训练三个模型；
- 进一步加入性别信息，基础之前的三个模型训练出6个模型。
- 模型推断时，先收集用户的年龄、性别属性，再使用对应的模型进行输出。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e05b161e99bc415705771c313cc637ab.png)
##### 3.2.5.3 效果展示和扩展应用
对比其它模型的结果：（精美程度和ID相似度都更胜一筹）
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7995d189d0a4f4423dd2fa8f7504a14b.png)
基于Photo2Cartoon的扩展应用：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3de19f612c023da8b7ab79393b2a8724.png)

### 3.3 卡通化实战
#### 3.3.1 Pixel2Pixel实现人像卡通化
>项目地址[《Pixel2Pixel：人像卡通化》](https://aistudio.baidu.com/aistudio/projectdetail/1813349?channelType=0&channel=0)

在AI Studio中搜索卡通，有四个数据集,第一个就是人像卡通化数据集。

### 数据准备：
&#8195;&#8195;Pixel2Pixel需要成对数据训练，卡通画没有找画师画，而是photo2cartoon生成的真实照片对应的卡通画。由于是有监督训练，收敛很快。
- 真人数据来自[seeprettyface](http://www.seeprettyface.com/mydataset.html)(AI生成的照片）
- 数据预处理（详情见[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目）
<div>
  <img src='https://i-blog.csdnimg.cn/blog_migrate/9c7a0df240edc5e5ee437f5b2b0b737b.jpeg' height='150px' width='1000px'>
</div>

- 使用[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目生成真人数据对应的卡通数据。

数据预处理：
- 将图像数据转换为标准形式；
- 检测人脸、关键点。根据关键点对人脸进行旋转校正；
- 根据人像分割模型，去除背景并填充为白色。

其它代码内容请参考[《Pixel2Pixel：人像卡通化》](https://aistudio.baidu.com/aistudio/projectdetail/1813349?channelType=0&channel=0)
#### 3.3.2 Photo2cartoon
>项目地址：[Photo2cartoon](https://aistudio.baidu.com/aistudio/projectdetail/1428373?channelType=0&channel=0)

##### 3.3.2.1 测试、推理
1. 安装ppgan、dlib、scikit-image
```python
%cd /home/aistudio/work/
!git clone https://gitee.com/hao-q/PaddleGAN.git
%cd PaddleGAN/
!pip install -v -e .

!pip install dlib -t /home/aistudio/external-libraries
!pip install scikit-image -t /home/aistudio/external-libraries
```

```python
# 导入依赖库
import sys 
sys.path.append('/home/aistudio/external-libraries')


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ppgan.apps import Photo2CartoonPredictor
```

```python
# 下载测试图片
!wget https://raw.fastgit.org/minivision-ai/photo2cartoon-paddle/master/images/photo_test.jpg -P /home/aistudio/work/imgs

img_src = plt.imread('../imgs/photo_test.jpg')

plt.imshow(img_src)
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/01b81e6496b3dc8af607979f476ee2ab.png)

```python
# 测试
p2c = Photo2CartoonPredictor()
output = p2c.run('../imgs/photo_test.jpg')#使用Photo2CartoonPredictor的run方法得到卡通化结果
#查看测试效果
plt.figure(figsize=(10, 10))
img_input = plt.imread('./output/p2c_photo.png')
img_output = plt.imread('./output/p2c_cartoon.png')

img_show = np.hstack([img_input, img_output])
plt.imshow(img_show)
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d28ac7423a1a0d0bc7fce3289e7cdfd.png)
##### 3.3.2.2 训练
训练过程如下：
1. 从aistudio数据集中导入人像卡通化数据集。
2. 将数据解压并放置在PaddleGAN/data路径下
3. 设置训练参数configs/ugatit_photo2cartoon.yaml
4. 开始训练

代码如下：

```python
# 解压数据至PaddleGAN/data/
!unzip -q /home/aistudio/data/data68045/photo2cartoon_dataset.zip -d /home/aistudio/work/PaddleGAN/data/ 
```

```python
# 训练数据统计
trainA_names = os.listdir('data/photo2cartoon/trainA')
print(f'训练集中真人照数据量: {len(trainA_names)}')

trainB_names = os.listdir('data/photo2cartoon/trainB')
print(f'训练集中卡通画数据量: {len(trainB_names)}')

testA_names = os.listdir('data/photo2cartoon/testA')
print(f'测试集中真人照数据量: {len(testA_names)}')

testB_names = os.listdir('data/photo2cartoon/testB')
print(f'测试集中卡通画数据量: {len(testB_names)}')

# 训练数据可视化
img_A = []
for img_name in np.random.choice(trainA_names, 5, replace=False):
    img_A.append(cv2.resize(cv2.imread('data/photo2cartoon/trainA/'+img_name), (256,256)))

img_B = []
for img_name in np.random.choice(trainB_names, 5, replace=False):
    img_B.append(cv2.resize(cv2.imread('data/photo2cartoon/trainB/'+img_name), (256,256)))

img_show = np.vstack([np.hstack(img_A), np.hstack(img_B)])[:,:,::-1]
plt.figure(figsize=(20, 20))
plt.imshow(img_show)
plt.show()
```

```python
# 一行代码开始训练
!python -u tools/main.py --config-file configs/ugatit_photo2cartoon.yaml
```

