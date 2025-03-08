@[toc]
- [《opencv优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5502)
- [《learnopencv系列一：使用神经网络进行特征匹配（LoFTR、XFeat、OmniGlue）、视频稳定化、构建Chrome Dino游戏机器人》](https://blog.csdn.net/qq_56591814/article/details/143252588?spm=1001.2014.3001.5502)
- [《learnopencv系列二：U2-Net/IS-Net图像分割（背景减除）算法、使用背景减除实现视频转ppt应用》](https://blog.csdn.net/qq_56591814/article/details/143317678?spm=1001.2014.3001.5501)
- [《learnopencv系列三：GrabCut和DeepLabv3分割模型在文档扫描应用中的实现》](https://blog.csdn.net/qq_56591814/article/details/143612087)

>本文三篇文章均来自[learnopencv](https://github.com/spmallick/learnopencv/blob/master/README.md)
## 一、使用神经网络进行特征匹配
>[learnopencv原文](https://learnopencv.com/feature-matching/#aioseo-feature-matching-classical-vs-deep-learning)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Feature-Matching-Using-Neural-Networks)

&#8195;&#8195;你使用相机的全景模式来拍摄广角照片。但是，这个全景模式究竟是如何在后台工作的呢？或者假设你有一段不稳定的自行车骑行视频，你在你的编辑应用中选择视频稳定化选项，它给出了完全稳定化版本的相同视频，那么它是如何工作的呢？答案就是**特征匹配（Feature Matching ）**。

![img](https://img-blog.csdnimg.cn/img_convert/85c02f2de14096b916e16d06f78d53aa.gif)

那么，在这篇文章中，我们将看到：

- 什么是特征匹配？
- 2024年为什么还要进行特征匹配？
- 特征匹配的最新进展：经典与深度学习
- 特征匹配算法如何在代码中工作——WebUI的代码管道
- 特征匹配实验结果
- 结论与参考文献

### 1.1 什么是图像特征？

&#8195;&#8195;图像由多个对象或单个对象组成。每个对象在该图像中都带有不同的描述。**图像特征是描述对象独特品质的信息片段，这些特征包括从简单的边缘和角点到更复杂的纹理（比如强度梯度）或独特的形状（比如斑点）**。考虑一个人拿着一本书的图像。人类可以通过查看图像帧中的照明条件或对象周围的轮廓和形状来理解某些对象（人或书）存在于框架中。计算机如何解释相同的内容？

![img](https://img-blog.csdnimg.cn/img_convert/8efc6f4b637b3e40bb93897afd8ec657.gif#pic_center) <center> **图像特征的样子**</center>


&#8195;&#8195;为此，我们使用图像特征。我们取每个图像像素并计算这些像素的强度梯度（与周围像素的强度值变化相比）。梯度值高的区域，通常是图像特征（角落或边缘）。我们如何提取这些图像特征？或者这些图像特征如何用于识别对象？一个简单的答案是**特征匹配**，我们现在将探索这个问题。

![img](https://img-blog.csdnimg.cn/img_convert/cf3c56d185ddfc3ace212c9cabc3b42b.png#pic_center =800x) <center> **局部与全局图像特征**</center>


在此之前，我们可以将这些图像特征分为两种类型：

- **局部特征** ： 这些指的是图像的特定部分，捕获有关小区域的信息。这些特征特别适合于理解图像的详细方面，如纹理、角落和边缘。
- **全局特征** -：这些描述整个图像作为一个整体，并捕获整体属性，如形状、颜色直方图和纹理布局。

根据用例，两者都可以使用。在本文中，我们将主要使用局部特征。

### 1.2 特征匹配的应用场景——为什么在2024年还要进行特征匹配？

&#8195;&#8195;特征匹配是一项始于1990年代末的旧计算机视觉技术，最初的边缘检测算法如`Sobel`、`Canny`和角点检测算法`Harris`角点检测。多年来，它不断改进，并将神经网络引入了特征匹配，如`Superpoint`。然后，`LoFTR`通过引入变换器进入了特征匹配流程，改变了游戏规则。但我们今天为什么要使用特征匹配呢？下面介绍特征匹配的应用场景：

1. **3D重建**：3D重建是这个时代最重要的研究课题之一。研究人员正在努力从即使是单个图像中生成3D结构，将这些3D结构扩展到AR/VR空间。**特征匹配是整个3D重建流程的关键部分。当您从不同角度拍摄对象或场景的多张照片时，特征匹配识别这些图像中的公共点**。通过分析这些点在照片中的移动，`3D-CNN`模型计算点的3D坐标，重建场景的3D模型。

![img](https://img-blog.csdnimg.cn/img_convert/06161fe5bc88891f73b0760e9433aa37.gif#pic_center )



2. **医学图像配准**：医学图像配准技术，旨在对齐不同时间或不同类型的医学扫描图像，例如MRI或CT扫描，通常用于治疗前后的比较。通过特征检测算法识别扫描图像中的对应点（通常是肿瘤或器官的边缘），然后在不同扫描之间进行匹配来对齐图像，从而帮助医生获得更好的诊断。
![img](https://img-blog.csdnimg.cn/img_convert/b47ff0cf30cd43b9326a6e157f132e2d.gif#pic_center)

3. **面部识别**
面部识别是依赖于特征匹配的主流应用之一。特征检测器（Haar Cascades）提取面部标志点（描述面部的特征或点），然后使用匹配器通过这些标志点在整个框架中匹配面部并识别它。我们还可以利用这些特征在其上实现一些AR滤镜。请参阅我们关于[创建Instagram滤镜的详细文章](https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/)，进行一些实践操作。
![img](https://img-blog.csdnimg.cn/img_convert/e8cbc4c0bd9b11bbde9fc7d49f937db7.gif#pic_center)



4. **图像拼接（全景）**
图像拼接算法使用诸如Harris角点检测或SURF之类的方法，在重叠区域检测特征，并通过暴力或FLANN方法进行匹配。然后它使用单应性矩阵来拼合图像，校正失真并创建单个连续的全景。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4de1947b1bc94de99e7aa766c546125b.gif#pic_center)

6. **SLAM (Simultaneous Localization and Mapping，实时定位与地图构建)**：
SLAM是一种使机器人或设备能够在未知环境中同时进行定位和地图构建的技术。SLAM的主要目标是在不依赖于预先构建地图的情况下，通过传感器数据实时感知环境，并跟踪其在其中的位置（更多内容，详见[Visual SLAM](https://learnopencv.com/monocular-slam-in-python/)）。其基本原理为：
	1. **传感器数据获取**：SLAM系统通常使用激光雷达、摄像头、IMU（惯性测量单元）等传感器获取环境数据。这些数据帮助系统识别周围的物体和特征。	
	2. **特征提取**：从传感器数据中通过FAST或BRIEF等方法检测提取特征点，如墙壁、角落或其他显著物体，这些特征用于地图构建和定位。	
	3. **特征匹配**：通过比较当前观测到的特征与之前的特征，SLAM系统可以确定它们是否相同。这对于保持地图的一致性至关重要。
	4. **位置估计**：使用滤波算法（如卡尔曼滤波、粒子滤波等）来估计设备在环境中的位置。这些算法会处理传感器数据的不确定性，提供更可靠的位置估计。	
	5. **地图构建**：随着设备移动，SLAM系统不断更新地图，将新识别到的特征添加到地图中，从而构建出环境的完整表示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b9639540dfa4be785b915ab19f4f9e8.gif#pic_center)
SLAM技术广泛应用于：
		- **机器人导航**：使自主移动机器人在未知环境中导航。
		- **增强现实（AR）**：帮助设备理解周围环境，以便在其中叠加虚拟物体。
		- **自动驾驶**：为自动驾驶车辆提供实时定位和环境感知。
		- **无人机**：帮助无人机在复杂环境中自主飞行和导航。

6. **视频稳定化**
使用特征匹配技术可以稳定你骑自行车时拍摄的抖动视频。具体步骤如下：
	1. **特征检测**：在每一帧中使用光流法或Lucas-Kanade方法检测特征点。
	2. **特征匹配**：将当前帧的特征与前一帧中的特征进行匹配。
	3. **应用变换**：基于匹配的特征应用变换（如仿射变换或投影变换），对齐帧，平滑运动并产生稳定的视频。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/33d14fac0011466e9bcc01fffd786e5f.gif#pic_center)


&#8195;&#8195;以上是特征匹配的主流任务。它还可以用来做一些更疯狂的事情，比如使用OpenCV特征匹配构建Chrome Dino游戏机器人（见第三章）。


### 1.3 特征匹配——经典方法与深度学习

&#8195;&#8195;特征匹配技术始于1990年代末，本节将介绍其整个演化过程，包括特征匹配经典方法、深度学习方法以及最新研究。

#### 1.3.1 经典特征匹配
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41654d44a619495d8c0403010736d220.webp#pic_center =600x)<center>**特征匹配方法概述**</center>

特征匹配方法由三个主要部分组成：
- **检测**
检测算法如`Harris`角点检测器、`Sobel`或`Canny`边缘检测器、`Haar`级联等，通过简单的数学运算（例如计算像素强度、应用高斯模糊或阈值处理）来寻找关键点（特征）。当算法检测到图像中像素强度的变化时，便将其计为一个关键点或特征（可能是物体的角落或边缘），因为图像边缘和角落的颜色饱和度或拥挤度较高，会出现强度变化的梯度。

- **描述**
	- 获得关键点后，特征描述符（如`SIFT`或`SURF`）会处理这些关键点并生成特征向量（描述符向量），类似于数值“指纹”，可用于区分不同的特征。图像是二维向量（矩阵），因此算法检测到的像素梯度（亮度、平移、缩放和面内旋转的像素强度变化）仅是数字数组（向量）。
	- 应用程序应根据图像内容选择合适的检测器和描述符。例如，如果图像包含细菌细胞，应使用斑点检测器，而对于城市的航拍图像，则适合使用角点检测器。

- **匹配**
	- 生成两张图像的特征向量之后，可以使用匹配算法（如暴力搜索或FLANN）来进行特征匹配。算法遍历一张图像的所有特征向量，计算其与另一张图像中所有特征向量的相似度距离，并将最近的特征向量对视为完美匹配。
	- `ORB`（定向FAST和旋转BRIEF）是一种使用`FAST`进行特征检测，并使用`BRIEF`计算描述符的算法。它被广泛使用，是特征匹配的经典方法之一。更多信息详见[feature matching with ORB](https://learnopencv.com/how-to-build-chrome-dino-game-bot-using-opencv-feature-matching/).

#### 1.3.2 深度学习特征匹配
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/233b06c18c4749d99b3260953b677461.png#pic_center =800x)<center>**深度学习在特征匹配中的演变** </center>



&#8195;&#8195;深度学习或神经网络逐渐取代了传统的机器学习和计算机视觉算法，以提高在边缘情况下的准确性和鲁棒性。在特征匹配方面，深度学习引入了卷积神经网络（CNN），使特征匹配流程更加高效。最初，[Superpoint](https://paperswithcode.com/paper/superpoint-self-supervised-interest-point)将关键点检测和描述结合在一个网络中，随后是[D2Net](https://paperswithcode.com/paper/190503561)和[R2D2](https://paperswithcode.com/paper/r2d2-reliable-and-repeatable-detector-and)，进一步整合了这些过程。

&#8195;&#8195;接着，[NCNet](https://paperswithcode.com/paper/neighbourhood-consensus-networks)引入了四维成本体积（一个大立方体，每个切片表示两个不同图像在不同位置或偏移下的特征匹配效果），推动了无检测器的方法，如`Sparse-NCNet,DRC-Net,GLU-Net`和`PDC-Net`。[SuperGlue](https://paperswithcode.com/paper/superglue-learning-feature-matching-with)采用了将匹配视为图问题的不同方法，`SGMNet`和`ClusterGNN`对这一概念进行了优化。后来，[LoFTR](https://paperswithcode.com/paper/loftr-detector-free-local-feature-matching)和[Aspanformer](https://paperswithcode.com/paper/aspanformer-detector-free-image-matching-with)等方法结合了Transformer或注意力机制，进一步扩大了感受野，推动了基于深度学习的匹配技术的发展。目前，我们还有两个最新的深度学习方法，[XFeat](https://paperswithcode.com/paper/xfeat-accelerated-features-for-lightweight)和[OmmiGlue](https://arxiv.org/abs/2405.12979)，来自CVPR 2024。下面，我们将讨论最新的研究工作：
- [XFeat](https://paperswithcode.com/paper/xfeat-accelerated-features-for-lightweight)：一个优化的深度神经网络，专门用于特征匹配，仅使用CPU；
- [OmniGlue](https://arxiv.org/abs/2405.12979)：一个完美的模型，在特征匹配流程中结合了Transformer和CNN；
- [LoFTR](https://paperswithcode.com/paper/loftr-detector-free-local-feature-matching)：这个模型将Transformer架构引入了传统的特征匹配流程。

#### 1.3.3 XFeat
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/40f09766c73647b2b136a13ab07b9953.webp#pic_center =800x)<center>  **XFeat模型架构**</center>





&#8195;&#8195;**局部特征提取的准确性取决于输入图像的细节程度**。对于相机位置确定、图像定位和从照片构建3D模型（SfM）等任务，需要图像点之间非常精确的匹配。高分辨率图像提供这种精度，但会对计算过程提出极高要求，即使对于像SuperPoint这样简化的网络也是如此。XFeat主要通过一系列优化和架构简化来提升特征匹配的效率和准确性，特别是在计算资源有限的情况下。

##### 1.3.3.1 网络结构

1. **轻量化网络骨干**：XFeat使用了一个轻量化的网络骨干架构，将图像从浅层开始处理，通过逐步减少图像分辨率，同时增加通道数（提高深层表达能力）的方式，实现了更高效的特征提取，降低了整体计算成本。以灰度图像$I\in \mathbb{R}^{H\times W\times C}$为例（`H,W,C`分别是高度宽度和通道数）。每层的计算成本由公式给出：
$$F_{ops}=H_{i}\cdot W_{i}\cdot C_{i}\cdot C_{i+1}\cdot k^{2}$$

2. **使用深度可分离卷积**
	- 但仅仅减少通道可能会影响网络处理不同光照或视角的能力。使用**深度可分离卷积**可以降低计算成本，参数更少。然而，在需要高细节的局部特征提取中，这种方法并没有节省太多时间，反而限制了网络的表现能力。	
	- XFeat网络的骨干由称为“基本层”的简单单元组成，这些层是2D卷积与`ReLU`和`BatchNorm`的结合，按块结构逐步减半图像分辨率并增加深度，结构为{4, 8, 24, 64, 64, 128}，最后以一个融合块结束，将来自多个分辨率的特征结合在一起。

##### 1.3.3.2 局部特征提取
1. **多分辨率特征融合（特征金字塔）**： 通过多分辨率融合提升特征的鲁棒性
**Descriptor Head**从不同尺度聚合特征，生成一个稠密特征图`F`（一个紧凑的64维稠密描述符图），扩大网络在每层“看到”的区域，增强对视角变化的鲁棒性，这对小型网络设计至关重要。另一个卷积块用于回归可靠性图`R`，建模给定局部特征$F_{ij}$的无条件匹配概率$R_{ij}$。

2. **关键点检测的简化方法**
**Keypoint Head**：与`UNet`或`ResNet`等复杂架构的典型方法不同，XFeat采用简化的卷积操作，通过将图像区域划分为$8×8$块，并使用$1×1$卷积快速生成关键点热图`K`。这不仅加快了关键点检测过程，也减少了网络复杂度和计算负担。



3. **稠密匹配模块（Dense Matching）**：XFeat在匹配模块中采用了一种稠密匹配策略，不依赖于高分辨率的特征图，而是基于低分辨率下的特征实现有效匹配，因此适用于计算能力有限的场景。
	-  **原理**：通过只在低分辨率（原始图像的1/8）上进行粗略特征匹配，XFeat**大大减少了计算和内存需求**。此外，使用简单的多层感知机（MLP）微调初始匹配位置，根据可靠性评分优化匹配结果，确保在低计算资源下达到良好的匹配效果。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf13d8431a14432ca533cbc6d846817c.webp#pic_center)<center> **Xfeat Matching Module**</center>

##### 1.3.3.3 网络训练
&#8195;&#8195;**可靠性图与损失函数优化**：训练过程使用实际匹配的图像点来教网络如何正确识别和描述局部特征。该方法使用特定的损失函数，通过将网络的预测与已知对应关系进行比较，处理不匹配问题。网络在训练过程中也学习这些特征的可靠性，帮助模型学习匹配置信度。
#### 1.3.4 OmniGlue
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bd05c4ef8bcb47f3bbac53c9dadbfaf3.webp#pic_center)<center> **OmniGlue模型架构**</center>
##### 1.3.4.1 模型概览

&#8195;&#8195;`OmniGlue` 是首个以广泛**泛化能力**为核心设计的可学习图像匹配器，能够在不同的视觉任务、图像场景和数据集之间泛化得更好。模型包括四个主要阶段：

1. **特征提取（Feature Extraction）**
   - **SuperPoint Encoder**：用于提取高分辨率、细粒度的局部特征，这些特征对于复杂场景中的精确匹配至关重要。
   - **DINOv2 Encoder**：利用预训练的视觉 Transformer 模型，提取更**通用**和**稳健**的特征，这些特征能够捕捉更广泛的视觉模式，从而在领域差异较大的情况下实现更好的匹配效果。

2. **关键点关联图（Keypoint Association Graphs）**
   - **图像内关联图（Intra-image Graphs）**：在每张图像中构建密集连接的图结构，以便全面地互相关联特征，并增强局部描述符的细化能力。
   - **图像间关联图（Inter-image Graphs）**：在 `DINOv2` 的引导下，选择性地在不同图像中类似的关键点之间建立连接，优先考虑高概率的匹配对，同时通过剪除不太可能的连接来降低计算开销。

3. **信息传播（Information Propagation）**：采用双重注意力机制进行信息传播。这些注意力机制具有适应性，能够根据特征的复杂度和独特性来动态调整关注焦点，确保信息在图像内外的平衡传播。
     - **自注意力（Self-Attention）**：专注于单个图的细化，基于图像内上下文线索来优化关键点。
     - **交叉注意力（Cross-Attention）**：在图像之间的关键点之间建立桥梁，选择性地融合特征，依据 DINOv2 提供的相似性和重要性线索，动态调整注意力。


4. **描述符优化和匹配（Descriptor Refinement and Matching）**
   - 在信息传播和特征优化之后，描述符通过一系列操作进行优化，使得匹配不仅仅依赖于直接的特征相似度，还融入了空间和外观属性的整体信息。
   - 最终匹配通过学习到的度量和几何约束结合来实现，确保匹配在局部和全局图像结构上均具备一致性。
##### 1.3.4.2 模型细节

1. **详细特征提取（Detailed Feature Extraction）**：
   - 每个关键点的局部描述符通过 `SuperPoint` 和 `DINOv2` 的输出进行上下文增强，以平衡特征的**特异性**和**通用性**。
   - 动态调整的位置嵌入通过多层感知机（MLP）优化，将局部图像特征和 `DINOv2` 提供的图像全局上下文结合，实现丰富的空间编码。

2. **图结构构建（Graph Construction）**：
   - **动态图剪枝（Dynamic Graph Pruning）**：采用 `DINOv2` 提供的阈值机制对图像间的图结构连接进行动态剪枝，只保留潜在有意义的连接，以提高计算效率和匹配精度。

3. **高级信息传播（Advanced Information Propagation）**：
   - 集成了**新型混合注意力机制**，结合了传统注意力和领域自适应组件，使得模型能够根据领域的特定性和特征的独特性调整关注焦点。
   - 使用位置信息来调制注意力机制，以提高其对上下文重要特征的敏感度，同时避免因位置因素过度拟合。
#### 1.3.5 LoFTR

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2d2f1c4bfe54b2d913b2711874ac191.webp#pic_center =800x)<center>**LoFTR模型架构** </center>
`LoFTR` 引入了 `Transformer` 架构到传统特征匹配流程中，其架构包括以下四个部分：

1. **局部特征 CNN**
	* **特征图提取**：通过卷积神经网络（CNN）从成对图像 $I_A$ 和 $I_B$ 中提取两级特征图。粗级特征图提供全局视图，而细级特征图捕捉细节信息。该双重提取有助于实现一般对齐和匹配细化。

2. **局部特征 Transformer**
	* **扁平化与编码**：首先将粗级特征图扁平化为 1D 向量，并添加位置编码以保留特征在图像中的原始位置信息。
	* **Transformer 处理**：合并后的特征向量通过 LoFTR 模块处理，包含 $N_c$ 层的自注意力（同一图像向量内部）和交叉注意力（不同图像向量之间），允许模型在整个图像中整合信息。

3. **可微匹配层**
	* **置信度矩阵与选择**：Transformer 模块的输出传递至匹配层生成置信度矩阵 $P_c$，表示潜在匹配的置信水平。基于预定义的置信度阈值和最近邻准则，从矩阵中选出一组粗级匹配预测 $M_c$。

4. **匹配细化**
	* **局部窗口裁剪**：对于粗匹配对 $(M_c)$ 中的每一对，在细级特征图上裁剪大小为 $w \times w$ 的局部窗口。
	* **亚像素级细化**：在这些局部窗口内，将粗匹配调整到亚像素精度，得到最终的匹配预测 $M_f$。这一步确保了匹配不仅基于全局对齐，还在细节层面实现精确调整。

#### 1.3.6 总结

- `XFeat`：一种轻量化神经网络，设计用以在CPU上高效进行特征匹配。它使用简化的网络骨干并结合特征金字塔技术，提取图像的细节信息。XFeat的关键点检测模块采用快速卷积操作生成关键点热图，从而降低计算成本。
- `OmniGlue`：集成了变换器与卷积神经网络，其特征提取由SuperPoint编码器和DINOv2编码器完成。通过构建密集的图节点连接和信息传播，OmniGlue能够在跨域场景下实现高精度的图像特征匹配。
- `LoFTR`：首个将变换器引入特征匹配流程的模型，通过跨图像的自注意力和交叉注意力机制，增强了特征匹配的整体一致性和准确性。
### 1.4 特征匹配 – WebUI 代码流程

&#8195;&#8195;接下来，我们将测试经典的 `SIFT,LoFTR,XFeat` 和 `OmniGlue`模型并对比结果。为了测试所有模型，我们将使用[Image Matching WebUI ](https://github.com/Vincentqyw/image-matching-webui)，并根据我们的用例做一些额外的调整。

首先，将[Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui)存储库克隆到本地目录中：

```python
git clone https://github.com/0xSynapse/image-matching-webui.git
cd image-matching-webui/
```

然后使用miniconda创建一个Python 3.10的虚拟环境：

```python
conda create -n imw python=3.10.0
conda activate imw
```

接着安装`Pytroch,CUDA,Torchmetrics,PyTorch Lightning`以及`requirements.txt`文件中的依赖项。

```python
# 安装pytorch 2.2.1和CUDA 12.1
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install torchmetrics=0.6.0 pytorch-lightning=1.4.9
pip install -r requirements.txt
```

现在，运行`app.py`脚本启动WebUI：

```python
cd image-matching-webui/
python app.py
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6fdd1fca61f146f5ab701db373b3700c.gif#pic_center)<center> **Feature Matching WebUI Gradio Demo**</center>

&#8195;&#8195;这个WebUI集合了不同特征匹配模型，我们将模型的所有代码库收集在 `third_party` 文件夹中，并在`hloc`文件夹中为所有特征匹配模型提供了一个通用的pipeline ：

```
./hloc/
├── extractors
├── matchers
├── pipelines
├── __pycache__
└── utils
```

在`ui`文件夹中有我们的主要代码集成基座：

```
./ui/
├── api.py
├── app_class.py
├── config.yaml
├── __init__.py
├── __pycache__
├── sfm.py
├── utils.py
└── viz.py
```

下面是`app.py`：

```python
import argparse
from pathlib import Path
from ui.app_class import ImageMatchingApp
 
 
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "--server_name",
       type=str,
       default="0.0.0.0",
       help="server name",
   )
   parser.add_argument(
       "--server_port",
       type=int,
       default=7860,
       help="server port",
   )
   parser.add_argument(
       "--config",
       type=str,
       default=Path(__file__).parent / "ui/config.yaml",
       help="config file",
   )
   args = parser.parse_args()
   ImageMatchingApp(
       args.server_name, args.server_port, config=args.config
   ).run()
```

&#8195;&#8195;在后台，它调用了`app_class.py`中的`ImageMatchingApp(...)`类。`app_class.py`有三个类：

```python
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
 
import gradio as gr
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf
 
from ui.sfm import SfmEngine
from ui.utils import (
    GRADIO_VERSION,
    gen_examples,
    generate_warp_images,
    get_matcher_zoo,
    load_config,
    ransac_zoo,
    run_matching,
    run_ransac,
    send_to_match,
)
 
DESCRIPTION = """
# Image Matching WebUI
This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!
<br/>
 
## It's a modified version of the original code for better structure and optimization.
"""
 
class ImageMatchingApp:
 
class AppBaseUI:
 
class AppSfmUI(AppBaseUI):
```

- `ImageMatchingApp(...)` – 这个类是我们的主要gradio应用程序；它包含了gradio的所有组件，包括所有输入、方法、参数设置和输出。
- `AppBaseUI(...)` – 这个类包括一些额外的参数输入，我们可以考虑更复杂的任务。
- `AppSfmUI(AppBaseUI)` – 这个类是用于SFM（从运动中恢复结构），在我们的用例中不考虑。
- 所有类都包含了我们最小化的函数和代码，以便于可视化。

```python
# button callbacks
button_run.click(
    fn=run_matching, inputs=inputs, outputs=outputs
)
```

&#8195;&#8195;现在，在`ImageMatchingApp(...)`类中，它使用了`utils.py`中的`run_matching(...)`。这个`utils.py`从`hloc`模块中提取了所有特征匹配流程代码。

```python
from hloc import (
   DEVICE,
   extract_features,
   extractors,
   logger,
   match_dense,
   match_features,
   matchers,
)
```

&#8195;&#8195;在`hloc`中，所有代码都被结构化以适应管道，从`third_party `文件夹中获取。如果你记得，这是我们下载所有模型和代码库的文件夹。让我们简化这个过程；假设你正在使用**OmniGlue**进行特征匹配。流程将如下所示：

1. `app.py`将从`ui/app_class.py`调用`ImageMatchingApp(...)`类
2. `ImageMatchingApp(...)`调用`ui/utilis.py`中的`run_matching(...)`函数
3. 在`utils.py`中，`run_matching(...)`函数调用`hloc/matchers//omniglue.py`
4. 在`omniglue.py`中，它从`third_party/omniglue`导入了omniglue模块
5. 最后，整个流程运行特征匹配，并将匹配项提供给我们的`app.py` Gradio界面。

以上是主要的可执行流程。现在，我们对代码结构有了了解，让我们将进入实验。

### 1.5 特征匹配实验
#### 1.5.1使用LoFTR
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3c044a30728440ea3b4779ef8b0cd33.webp#pic_center =800x)
运行日志：
```
The Log for inference time:
 
Loaded LoFTR with weights outdoor
Loading model using: 0.368s
Matching images done using: 0.528s
RANSAC matches done using: 2.302s
Display matches done using: 1.434s
TOTAL time: 5.411s
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06f86cf12332416cbfe3061ff9bd04bf.webp#pic_center =800x)

#### 1.5.2 使用XFeat(Dense)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/858e92e1142348ce8a699db74cffeb5d.webp#pic_center =800x)


```
The Log for inference time:
 
Load XFeat(dense) model done
Loading model using: 0.983s
Matching images done using: 0.194s
RANSAC matches done using: 2.161s
Display matches done using: 1.358s
TOTAL time: 5.463s
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca431e0e082042dd99da591f5b5200ee.webp#pic_center =800x)
#### 1.5.3 使用XFeat(Sparse)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bf883f2a5ad4850be94dcf56a3d0d3d.webp#pic_center =800x)

```
The Log for inference time:
 
Load XFeat(sparse) model done.
Matching images done using: 1.351s
RANSAC matches done using: 2.177s
Display matches done using: 1.392s
TOTAL time: 5.704s
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c5148001d9694302ad99fed7168c8b06.webp#pic_center =800x)

#### 1.5.4 使用OmniGlue
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3352713ad3be4d13bbaa194de84acc84.webp#pic_center =800x)


```
The Log for inference time:
 
Loaded OmniGlue model done!
Loading model using: 4.561s
Matching images done using: 7.700s
RANSAC matches done using: 2.166s
Display matches done using: 1.395s
TOTAL time: 16.606s
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6a5bff20939a4f30a6e226500dd5c103.webp#pic_center =800x)

####  1.5.5 实验总结

- 我们应用了`1000`的匹配阈值。
- `LoFTR`在所有模型中耗时最少。
- `XFeat`仅使用CPU，并且在可接受的时间内给出不错的匹配。
- `Omniglue`匹配最准确和优化，尽管处理时间更长。

### 1.6 参考文献

1. [Xu, Shibiao, et al. “Local feature matching using deep learning: A survey.” _Information Fusion_ 107 (2024): 102344.](https://arxiv.org/abs/2401.17592)
2. [Sun, Jiaming, et al. “LoFTR: Detector-free local feature matching with transformers.” Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.](https://arxiv.org/abs/2104.00680)
3. [Potje, Guilherme, et al. “XFeat: Accelerated Features for Lightweight Image Matching.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.](https://arxiv.org/abs/2404.19174)
4. [Jiang, Hanwen, et al. “OmniGlue: Generalizable Feature Matching with Foundation Model Guidance.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.](https://arxiv.org/abs/2405.12979)
5. 除了实验结果外，所有资源都来自Google Image Search、Medium、YouTube、研究论文的项目页面等。


## 二、使用特征匹配进行视频稳定化
>[learnopencv原文](https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/VideoStabilization)

### 2.1 视频稳定化简介
&#8195;&#8195;视频稳定化是指用于减少摄像机运动对最终视频效果影响的一系列方法。摄像机的运动包括平移（即在x、y、z方向上的移动）和旋转（偏航、俯仰、翻滚）。视频稳定化的应用领域非常广泛，包括：

1. **消费级和专业级摄像**：在这些领域中，视频稳定化至关重要，因此存在许多机械、光学和算法解决方案。即使在静态图像摄影中，稳定化技术也可以帮助拍摄长时间曝光的手持照片。

2. **医疗诊断应用**：例如在内窥镜检查和结肠镜检查中，需要稳定视频以确定问题的确切位置和范围。

3. **军事应用**：在侦察飞行中，由空中飞行器拍摄的视频需要稳定化，以便进行定位、导航、目标跟踪等。

4. **机器人应用**：同样需要视频稳定化技术。

视频稳定化的方法包括：

1. **机械视频稳定化**：机械图像防抖系统通过使用陀螺仪和加速度计等特殊传感器检测到的运动来移动图像传感器，从而补偿相机的运动。

2. **光学视频稳定化**：这种方法不是移动整个摄像机，而是通过移动镜头的部分来实现稳定化。它采用可移动的镜头组件，可变地调整光线通过摄像机镜头系统时的光路长度。

3. **数字视频稳定化**：这种方法无需特殊传感器来估计相机运动，分为三个主要步骤：1）运动估计，2）运动平滑，3）图像合成。首先计算相邻帧之间的变换参数，然后过滤掉不必要的运动，最后重构稳定的视频。

&#8195;&#8195;本文将介绍一种快速且稳健的数字视频防抖算法实现。该算法基于二维运动模型，应用了欧氏（即相似性）变换，包括平移、旋转和缩放。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0d19a64da57e44ab92a86315bfca0451.webp#pic_center)

&#8195;&#8195;如上图所示，在欧氏运动模型中，图像中的一个正方形可以变换为另一个位置、大小或旋转角度不同的正方形。尽管相比仿射和单应变换，这种方法较为局限，但足以用于运动稳定，因为视频连续帧之间的相机移动通常较小。

### 2.2 使用光流估计稳定视频
&#8195;&#8195;本方法通过跟踪两个相邻帧之间的若干特征点，我们可以估算帧间的运动并进行补偿，其基本步骤为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c86e35494a2e457d88925c9855387f29.webp#pic_center)

- **特征检测**：通过`Shi-Tomasi` 角点检测算法检测前一帧的特征点
- **光流估计**：使用**光流**进行特征点跟踪，根据上一帧特征点算出当前帧对应的特征点。光流估计是一种技术，用于检测和跟踪这些特征点在连续帧中的运动，以确定哪些点在不同帧中是相同的。
- **计算变换矩阵**：根据前后帧的特征点，使用函数`cv2.estimateRigidTransform`计算出前后帧的刚性变换矩阵。
- **计算并平滑运动轨迹，应用变换**：一旦估计出帧间的变换，算法会应用这些变换来对图像进行校正，以补偿摄像机的运动。此外，为了使视频看起来更平滑，算法还会对这些变换进行平滑处理，以减少帧与帧之间的突然变化（将所有的帧间运动累加就得到了运动轨迹 ）。
- **Output Frame Sequence**：将稳定后的视频帧写入输出文件。

### 2.3 代码实现
#### 2.3.1 设置输入和输出视频
```python
# Import numpy and OpenCV
import numpy as np
import cv2
 
# 读取视频，获取视频中的总帧数
cap = cv2.VideoCapture('video.mp4') 
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
# 获取帧的宽度和高度
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# 定义输出视频的解码器
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
 
# 创建一个视频文件
out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))
```

#### 2.3.2 计算帧间运动
 
&#8195;&#8195;这是算法最关键的部分。通过遍历所有帧，计算当前帧与前一帧之间的运动。欧氏运动模型只需知道两个点的运动，但实践中通常会跟踪50-100个点，以便更稳健地估算运动模型。

1. **选择合适的跟踪点**  
选择用于跟踪的点至关重要。跟踪算法基于点周围的小区域来进行跟踪，因此平滑区域不适合跟踪，而带有大量角点的纹理区域更适合。OpenCV提供了`goodFeaturesToTrack`函数来检测适合跟踪的特征点，该函数是[Shi-Tomasi 角点检测算法](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)的实现。
2. **Lucas-Kanade光流算法**  
找到特征点后，可通过Lucas-Kanade光流算法在下一帧中跟踪这些点。该函数使用图像金字塔处理不同尺度的图像，并提供状态标志，用于过滤掉因遮挡等原因而无法跟踪的特征点。这一步在opencv中通过`calcOpticalFlowPyrLK`函数实现：

```python
cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts=None, winSize=(15, 15), maxLevel=2, 
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), 
                          flags=0, minEigThreshold=0.0001)
```
- 参数：
	* **prevImg**: 先前帧的灰度图像。
	* **nextImg**: 当前帧的灰度图像。
	* **prevPts**: 要跟踪的特征点列表，通常是一个 $N×1×2$ 的数组，每个元素表示点的坐标 (x, y)。
	* **nextPts**: 用于存储在当前帧中找到的特征点。如果为 None，函数将创建一个新的数组。
	* **winSize**: 计算光流时的窗口大小，默认为 (15, 15)。这个窗口影响算法的计算精度和性能。
	* **maxLevel**: 图像金字塔的最大层数，默认为 `2`。越高的层数会提高计算速度，但可能会降低精度。
	* **criteria**: 终止条件，通常使用 `cv2.TERM_CRITERIA_EPS` 和 `cv2.TERM_CRITERIA_COUNT` 来设置最大迭代次数和精度。
	* **flags**: 用于指定计算选项，默认为 0。可以设置为 `cv2.OPTFLOW_USE_INITIAL_FLOW` 等。
	* **minEigThreshold**: 最小特征值阈值，用于过滤特征点。
- 返回值
	* **nextPts**: 在当前帧中找到的特征点。
	* **status**: 每个特征点的跟踪状态，1 表示跟踪成功，0 表示跟踪失败。
	* **error**: 特征点跟踪的误差。

3. **估算运动**  
通过跟踪算法，我们在当前帧中获得了特征点的位置，并且知道它们在前一帧中的位置。可以利用这两组点来估算将前一帧映射到当前帧的刚性（欧氏）变换。通过`estimateRigidTransform`函数估算出运动后，将运动分解为x、y方向的平移和旋转角度，并将这些值存储以便平滑处理。


```python
cv2.estimateRigidTransform(src, dst, fullAffine=False)
```


* **src**：输入的源点集，通常是一个 $N×1×2$ 的数组，表示源图像中的特征点坐标。
* **dst**：目标点集，形状与源点集相同，表示目标图像中的对应特征点坐标。
* **fullAffine**：布尔值，默认为`False`，表示是否使用仿射变换。如果为 `True`，函数将返回一个 $2×3$ 的仿射变换矩阵；如果为 `False`，函数将返回一个 $2×3$ 的刚性变换矩阵。如果无法估计变换，则返回 `None`。

```python
# 读取第一帧并灰度化
_, prev = cap.read() 
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
```

```python
# 预定义变换存储数组
transforms = np.zeros((n_frames-1, 3), np.float32) 
 
# 主循环，处理每一帧
for i in range(n_frames-2):
	# 在前一帧中检测适合跟踪的特征点
	prev_pts = cv2.goodFeaturesToTrack(prev_gray,
	                                 maxCorners=200,
	                                 qualityLevel=0.01,
	                                 minDistance=30,
	                                 blockSize=3)
	
	# 读取下一帧
	success, curr = cap.read()
	if not success:
	break
	
	# 将当前帧转换为灰度
	curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
	
	# 计算光流（即跟踪特征点）
	curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
	
	# 合理性检查
	assert prev_pts.shape == curr_pts.shape 
	
	# 过滤出有效点
	idx = np.where(status==1)[0]
	prev_pts = prev_pts[idx]
	curr_pts = curr_pts[idx]
	
	# 计算变换矩阵，仅在OpenCV-3或更低版本中有效
	# 在 OpenCV 4.x 及更高版本中，考虑使用cv2.estimateAffinePartial2D
	m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
	
	# 提取平移量
	dx = m[0,2]
	dy = m[1,2]
	
	# 提取旋转角度
	da = np.arctan2(m[1,0], m[0,0])
	
	# 存储变换
	transforms[i] = [dx,dy,da]
	
	# 移动到下一帧
	prev_gray = curr_gray
	
	print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
```
#### 2.3.3 计算帧间平滑运动

&#8195;&#8195;在上一步中，我们估算了帧间运动并将其存储在数组中。现在需要通过累加上一步的差分运动来计算运动轨迹。

1. **计算轨迹**  
将帧间运动累加以得到运动轨迹，目的是平滑该轨迹。  在Python中，可以使用numpy的`cumsum`（累加和）函数轻松实现。

```python
# 使用变换的累积和计算轨迹
trajectory = np.cumsum(transforms, axis=0)
```
2. **计算平滑轨迹**
	- 在上一步中，我们计算了运动轨迹，得到三条曲线，分别表示 x、y 平移和角度随时间的变化。现在，我们将对这三条曲线进行平滑处理。	
	- 平滑曲线的最简单方法是使用**滑动平均滤波器（moving average filter）**，它通过取窗口内相邻点的平均值来平滑曲线。例如，对于存储在数组 `c` 中的曲线，用宽度为`5`的滑动平均滤波器处理后得到平滑曲线 `f`。第 `k` 个平滑点的计算公式为：
$$f[k] = \frac{c[k-2] + c[k-1] + c[k] + c[k+1] + c[k+2]}{5}$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a19117d190d44a17a3b9bfb4bfeaa010.webp#pic_center)<center> 左：噪声曲线；右：使用大小为5的box filter进行平滑后的曲线</center>

&#8195;&#8195;可以看出，平滑曲线上的值是噪声曲线在小窗口内的平均值。在Python实现中，我们定义了一个滑动平均滤波器，用于平滑任意一维曲线（即一维数字），返回平滑后的曲线。

```python
# 定义移动平均函数，用于平滑曲线
def movingAverage(curve, radius):
  window_size = 2 * radius + 1										# 定义滤波器
  f = np.ones(window_size)/window_size  
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')			# 为边界添加填充  
  curve_smoothed = np.convolve(curve_pad, f, mode='same')			# 应用卷积  
  curve_smoothed = curve_smoothed[radius:-radius]					# 移除填充  
  return curve_smoothed												# 返回平滑后的曲线
```
我们还定义了一个函数，它接受轨迹并对三个分量进行平滑：

```python
# 定义修复边界函数
def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory
```

```python
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)
```
3. **计算平滑变换**
到目前为止，我们已经得到了一个平滑的轨迹。在这一步中，我们将使用这个平滑轨迹来获得可以应用于视频帧的平滑变换，以稳定视频。具体来说，需要找到平滑轨迹与原始轨迹之间的差异，并将这个差异加回到原始变换中来完成计算。

```python
# 计算平滑轨迹和轨迹之间的差异
difference = smoothed_trajectory - trajectory
 
# 计算新的变换数组
transforms_smooth = transforms + difference
```
#### 2.3.4 将平滑后的相机运动应用到帧上
现在只需遍历所有帧并应用刚刚计算的平滑变换。对于指定的运动参数 (x, y, θ)，对应的变换矩阵为：

$$T = \begin{bmatrix} \cos \theta & -\sin \theta & x \\ \sin \theta & \cos \theta & y \end{bmatrix}$$

```python
# 重置视频流到第一帧 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# 写入n_frames-1个变换后的帧
for i in range(n_frames-2):
	# 读取下一帧
	success, frame = cap.read()
	if not success:
	break
	
	# 从新的变换数组中提取变换
	dx = transforms_smooth[i,0]
	dy = transforms_smooth[i,1]
	da = transforms_smooth[i,2]
	
	# 根据新值重构变换矩阵
	m = np.zeros((2,3), np.float32)
	m[0,0] = np.cos(da)
	m[0,1] = -np.sin(da)
	m[1,0] = np.sin(da)
	m[1,1] = np.cos(da)
	m[0,2] = dx
	m[1,2] = dy
	
	# 应用仿射变换到给定帧
	frame_stabilized = cv2.warpAffine(frame, m, (w,h))
	
	# 修复边界伪影
	frame_stabilized = fixBorder(frame_stabilized) 
	
	# 将视频帧写入文件
	frame_out = cv2.hconcat([frame, frame_stabilized])
	
	# 果图像太大，则调整大小。
	if(frame_out.shape[1] &gt; 1920):
		frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
	
	cv2.imshow("Before and After", frame_out)
	cv2.waitKey(10)
	out.write(frame_out)
```
#### 2.3.5 修复边界伪影
&#8195;&#8195;在视频稳定过程中，可能会出现黑色边界伪影，这是因为为稳定视频，某些帧可能需要缩小。  可以通过围绕中心将视频微缩放（如放大4%）来缓解这一问题。我们使用 `getRotationMatrix2D` 函数，因为它可以在不移动图像中心的情况下对图像进行缩放和旋转。只需设置旋转角度为0、缩放比例为1.04（即4%放大）。

```python
def fixBorder(frame):
	s = frame.shape
	# 在不移动中心的情况下将图像放大4%
	T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
	frame = cv2.warpAffine(frame, T, (s[1], s[0]))
	return frame
```
&#8195;&#8195;整个代码过程讲完了。我们的目标是显著减少运动，而不是完全消除它。我们留给读者思考如何修改代码以完全消除帧间运动。如果尝试消除所有相机运动，可能会产生什么副作用？

&#8195;&#8195;目前的方法仅适用于固定长度的视频，而不适用于实时视频流。为了实现实时视频输出，我们需要对该方法进行大量修改，这超出了本帖的范围，但这是可以实现的，更多信息可以在[这里](https://abhitronix.github.io/2018/11/30/humanoid-AEAM-3/)找到。
## 三、使用OpenCV特征匹配构建Chrome Dino游戏机器人
>[learnopencv原文](https://learnopencv.com/how-to-build-chrome-dino-game-bot-using-opencv-feature-matching/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Chrome-Dino-Bot-using-OpenCV-feature-matching)、[轮廓分析教程](https://learnopencv.com/contour-detection-using-opencv-python-c/)


&#8195;&#8195;Chrome Dino游戏是一款简单而有趣的游戏，具有无限生成的障碍物，且难度会不断增加。玩家需要让小恐龙跳跃或低头，以避免碰到空洞。虽然玩法简单，但难度较高。我们来看看游戏中不同类型的障碍物。




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bfa1c7ce70545a8a4c55faf982109cd.webp#pic_center)
&#8195;&#8195;要开发一个控制该游戏的外部机器人，需要实现实时屏幕捕捉和图像处理，以检测恐龙的位置，并在障碍物即将撞上时自动触发按键。为实现屏幕捕捉，可以使用 [mss库](https://python-mss.readthedocs.io/examples.html)，并在恐龙前方区域执行[轮廓分析](https://learnopencv.com/contour-detection-using-opencv-python-c/)来检测障碍物。而为了模拟按键，我们可以使用 [AutoPyGUI](https://pyautogui.readthedocs.io/en/latest/)。

&#8195;&#8195;至于如何获取分析区域，虽然可以建议训练 TensorFlow 或 Caffe 模型，然后使用 [OpenCV  DNN 模块](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/)来检测玩家的位置，但有时我们会忽略 OpenCV 中的一些经典工具，它们在这种任务中同样非常有效，那就是**特征匹配**。


在计算机视觉中，特征匹配包含三部分：

1. **特征检测（Feature Detector）或关键点检测（Keypoint Detector）：** 图像中的某些特征容易被检测和定位，特征检测器会提供图像中特征的x和y坐标，比如[Harris角点检测器](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)。

2. **特征描述（Feature Descriptor）：** 在特征检测器提供特征位置后，特征描述器会生成该特征的“签名”以便在不同图像间匹配。特征描述器的**输入是图像中特征的(x, y)位置，输出是一个描述向量（签名）**。一个优秀的特征描述器会为同一特征在不同图像中生成相似的向量比如[FREAK](https://docs.opencv.org/3.4/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html)，而[SIFT](https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html)、[SURF](https://docs.opencv.org/3.4/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html)和[ORB](https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html)包含了特征检测和特征描述的组合。

3. **特征匹配算法（Feature Matching Algorithm）：** 为了在不同图像中识别同一对象，我们在两张图像中识别特征并使用特征匹配算法进行匹配。当需要匹配的图像数量较少时，可以使用[暴力匹配算法（Brute Force Algorithm）](https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html)。如果要在大量图像数据库中进行搜索，则需使用[FLANN（Fast Library for Approximate Nearest Neighbors）](https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html)。



### 3.1 常见特征检测算法
#### 3.1.1 SIFT和SURF

&#8195;&#8195;SIFT（Scale Invariant Feature Transform，尺度不变特征变换）由David G. Lowe在`2004`年发明，是一种重要的计算机视觉技术。在深度学习兴起之前，SIFT一直是手工特征的主流方法。SIFT基于[高斯差分检测器（DoG）](https://arxiv.org/pdf/1311.2561.pdf)，并且在图像进行缩放、平移、旋转、仿射等几何变换后依然保持特征描述的一致性。然而，由于其较慢且被专利保护的局限性，使用SIFT需要支付专利费用。2020年3月，SIFT专利到期，OpenCV已将其算法从opencv-contrib移动到主库中。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1422f34273c64f77a7ae19f48b35dbb3.webp#pic_center)

&#8195;&#8195;`SURF`（Speed Up Robust Features）于`2006`年发布，比`SIFT`更快，但也存在专利问题。为了使用这些算法，OpenCV需要特定编译选项来启用，例如`OPENCV_ENABLE_NONFREE = ON`。

&#8195;&#8195;`ORB`、`BRISK`和`FREAK`是速度较快且无专利限制的特征算法。`BEBLID`（Boosted Efficient Binary Local Image Descriptor）也是最近在OpenCV 4.5.1发布的改进算法。


#### 3.1.2 ORB概述
&#8195;&#8195;[ORB（Oriented FAST and Rotated BRIEF）](https://ieeexplore.ieee.org/document/6126544)由Ethan Rublee于`2011`年发布。ORB结合了`FAST`检测器和`BRIEF`描述符，其在计算速度和特征效果方面表现优异。即使到如今，ORB仍然在计算机视觉研究中具有重要地位，并在2021年的ICCV上获得了Helmholtz奖，以表彰其在计算机视觉中的重大影响。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78ecef6c0e4a41a6a03d6d1aaddc8a44.jpeg#pic_center)


&#8195;&#8195;为了理解ORB的工作原理，我们可以考虑上面形状的顶部角落。在转换为灰度图后，图像中的任意角点要么有一个亮像素和较暗的背景，要么有一个暗像素和较亮的背景。FAST算法观察邻域中的四个像素，并将它们的亮度与中心像素进行比较。以下插图展示了该算法的工作原理。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2d56b4137d0a4a488d69bd649ebe8d04.webp#pic_center#pic_center =600x)<center>**角点检测示意图** </center>


&#8195;&#8195;取一个图像块（image patch），设中心像素的亮度为$I_c$。考虑周围的`16`个像素，其亮度分别为$I_1, I_2, …, I_{16}$。选择一个阈值亮度`T`。如果满足以下条件之一，则中心像素被标记为角点。

- 对于较暗的角点：存在n个连续的像素，满足$I_n > I_c + T$。
- 对于较亮的角点：存在n个连续的像素，满足$I_n < I_c – T$。

&#8195;&#8195;对于大多数像素，我们可以只查看像素`1,5,9,13`来判断是否为角点。如果`4`个像素中至少有`3`个像素比中心像素更暗或更亮，则可以将其声明为角点。

&#8195;&#8195;接下来，我们需要**计算特征的方向**。这是通过找到image patch的质心来完成的。质心和中心像素之间连线的方向即为特征的方向，比如下图中的蓝色像素。该方向是通过图像矩来计算的，计算方式详见[此博客](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3cc56ef3da264db0861b5de344ee79c3.webp#pic_center#pic_center =600x)
&#8195;&#8195;为了成功匹配特征，我们还需要一些东西，即image patch的简要描述。`BRIEF`描述符随机选择两个点进行亮度比较，并构建一个向量作为image patch的签名，比较的次数等于描述符的比特长度。一旦我们有了**关键点和描述符**，就可以将数据传递给描述符匹配器来进行匹配。在OpenCV中，有两种类型的描述符匹配器：基于暴力匹配和基于FLANN。

#### 3.1.3  OpenCV中的ORB实现

&#8195;&#8195;在OpenCV中，有三个主要函数：[ORB_create](https://docs.opencv.org/3.4.15/db/d95/classcv_1_1ORB.html)、`detect` 和 `compute`。不过，可以通过直接使用 `detectAndCompute` 函数来同时实现这三个函数的功能。下面创建一个ORB特征检测器：

```python
orb = cv2.ORB_create(nFeatures, scaleFactor, nlevels,  edgeThreshold, 
                        firstLevel, WTA_K, scoreType, patchSize, fastThreshold)
```




1. **nFeatures**: 指定要检测的特征点的最大数量。这个值越高，检测到的特征点就越多，但可能会影响处理速度。

2. **scaleFactor**: 每一层金字塔之间的尺度因子，范围`1 - 2`，默认设置为`1.2`，表示在每个尺度上图像大小会缩小到80%。这个参数控制金字塔的层级数和特征点的尺度变化。

3. **nlevels**: 金字塔的层级数，默认为`8`。增加层级数可以在不同的尺度上检测特征，但会增加计算量。

4. **edgeThreshold**: 边缘检测的阈值，默认`31`。用于过滤掉图像中太接近边缘的特征点，通常较小的值可以更精确地控制边缘。

5. **firstLevel**: 开始构建金字塔的层级数，默认从`0`开始，也可以设置为其他值以跳过某些层。

6. **WTA_K**: 每个关键点的描述符中包含多少个最佳响应的关键点，一般取值为`2`。

7. **scoreType**: 特征点的评分类型。可以是ORB特有的评分类型（例如`cv2.ORB_HARRIS_SCORE`或`cv2.ORB_FAST_SCORE`），决定特征点的选择标准。

8. **patchSize**: 计算描述符时所用的image patch大小，默认值为`31`。较大的patchSize可以捕捉更多信息，但也会增加计算时间。

9. **fastThreshold**: FAST特征点检测的阈值，值越小，检测的特征点数量可能越多，默认为`20`。

```python
# 检测特征
keypoints = orb.detect(image, mask)
# 计算描述符
keypoints, des = orb.compute(image, keypoints, mask)
```
你也可以使用一行代码完成这两个步骤：

```python
keypoints, des = orb.detectAndCompute(image, mask)
```

#### 3.1.4 OpenCV中的匹配算法
&#8195;&#8195;在OpenCV中有两种类型的描述符匹配器，基于两种不同的算法——`BRUTE FORCE`和`FLANN`。就像ORB一样，这里我们也需要创建一个描述符匹配器对象，然后使用`match`或`knnMatch`查找匹配。

```python
# 创建描述符匹配器
retval = cv2.DescriptorMatcher_create(descriptorMatcherType)
```
`descriptorMatcherType`可以是以下几种类型：

1. **`"BruteForce"`**: 这是最简单和最直接的匹配器，使用`L2`距离，通过暴力比较每个描述符进行匹配。适合小规模数据集，但对于大型数据集，效率较低。

2. **`"BruteForce-L1"`**: 这种类型使用`L1`距离（绝对值距离）进行匹配，适用于某些特定的描述符类型，如`BRIEF`。

3. **`"BruteForce-Hamming"`**: 使用汉明距离进行匹配，适合二进制描述符（如ORB、BRIEF等）。

4. **`"FlannBased"`**: 基于FLANN（快速近似最近邻库）的匹配器，适合大规模数据集，速度快且效率高。需要事先设置好FLANN参数。

5. **`"FlannBased-Hamming"`**: 结合FLANN和汉明距离，用于匹配二进制描述符。

```python
# 进行描述符匹配
matches = match(queryDescriptors, trainDescriptors[, mask])
```
- **queryDescriptors**：第一个图像的特征描述符集，通常是一个Numpy数组，其中每一行代表一个特征描述符。
- **trainDescriptors**：第二个图像的特征描述符集。

&#8195;&#8195;由于特征匹配的计算成本较低，因此它被广泛用于各种有趣的应用程序，例如文档扫描仪应用程序、全景图、图像对齐、对象检测、面部检测等。

### 3.2 游戏玩法观察与挑战

&#8195;&#8195;你可以在任何浏览器中打开这个游戏，但在Chrome中优化效果最佳，因此推荐使用Chrome（在地址栏输入`chrome://dino`加载游戏）。。在游戏中，我们观察到所有的仙人掌障碍物都可以通过适当的跳跃时机来跳过，但对于鸟类，有时需要低头，具体取决于鸟的飞行高度。

&#8195;&#8195;鸟类有三种不同的飞行高度：高、中、低。当鸟处于低高度时，可以跳跃；当处于中高度时，可以选择跳跃或低头。跳跃使用空格键，低头使用下箭头键。如果鸟的高度超过头部水平，则不需要做任何动作，T-Rex可以直接通过。

&#8195;&#8195;当游戏开始时，T-Rex会向前移动一小段距离，此后在整个游戏过程中位置保持不变。游戏模式会在白天和夜晚之间变化。根据上述观察，应用程序工作流程设计如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d755aaa1cc0d43a78eeadbf42e3ff5f7.png#pic_center)
1. 处理循环
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9e6f673e06744ffa8b7a97ba058ae0ee.png#pic_center)
### 3.3 启动项目
&#8195;&#8195;我们使用opencv-python、mss和PyAutoGUI来完成上述任务。直接下载[项目代码](https://github.com/spmallick/learnopencv/tree/master/Chrome-Dino-Bot-using-OpenCV-feature-matching)，导航到下载的代码目录，然后输入以下命令：
	
```python
pip install -r requirements.txt
```
然后启动：
- 将Chrome浏览器窗口放置在屏幕的右半部分。
- 在左半部分打开终端/powershell，如下图所示。
- 导航到工作目录并运行脚本`Trex.py`。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9200e0583a9a420f8dc9e92cc943b351.gif#pic_center)
&#8195;&#8195;如果没有像上图这样运行，请调整方框高度百分比，并检查显示器是否开启了自动缩放。更详细的说明已在代码中提供。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aced424a61dd4d9f818da5752f6881b1.jpeg#pic_center)<center> **冗余功能触发：这是一个鼠标回调函数，当特征匹配失败时触发**</center>
### 3.4 代码说明
#### 3.4.1 导入必要的包
`mss`用于屏幕捕获，`Tkinter`用于显示错误消息，`pyautogui`用于虚拟按键。

```python
import cv2
import numpy as np
from mss import mss
from tkinter import *
import pyautogui as gui
import tkinter.messagebox
```
#### 3.4.2 恐龙检测函数
&#8195;&#8195;根据我们所拥有的T-Rex参考图像（真实图），我们使用特征匹配在屏幕上找到T-Rex。这些函数以真实图像和捕获的屏幕图像作为参数。如下所示，使用ORB检测和计算两幅图像的关键点和描述符。然后，我们使用描述符匹配器找到匹配对。最后，对匹配进行排序和过滤，取前25%的匹配作为良好匹配。你可以调整这个值以生成最佳结果。

`getMatches`函数以列表的形式返回关键点，稍后将使用该列表来估计霸王龙的位置。

```python
def getMatches(ref_trex, captured_screen):
    # 初始化列表
    list_kpts = []
    # 初始化ORB。
    orb = cv2.ORB_create(nfeatures=500)
    # 检测和计算。
    kp1, des1 = orb.detectAndCompute(ref_trex, None)
    kp2, des2 = orb.detectAndCompute(captured_screen, None)
    # 匹配特征。
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    # 转换为列表。
    matches = list(matches)
    # 按分数排序匹配项。
    matches.sort(key=lambda x: x.distance, reverse=False)
    # 只保留前25%的匹配项。
    numGoodMatches = int(len(matches) * 0.25)
    matches = matches[:numGoodMatches]
    # 可视化匹配项。
    match_img = cv2.drawMatches(ref_trex, kp1, captured_screen, kp2, matches[:50], None)
    # 对于每个匹配项...
    for mat in matches:
        # 获取匹配图像的关键点。
        img2_idx = mat.trainIdx
        # 获取坐标。
        (x2, y2) = kp2[img2_idx].pt
        # 添加到每个列表。
        list_kpts.append((int(x2), int(y2)))
    # 调整图像大小以方便显示。
    cv2.imshow('Matches', cv2.resize(match_img, None, fx=0.5, fy=0.5))
    # cv2.imwrite('Matches.jpg', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list_kpts
```
#### 3.4.3 冗余功能
&#8195;&#8195;这是一个鼠标回调函数，当特征匹配失败时触发。添加点击-拖动-释放功能以定义障碍物检测patch坐标。这是我们执行轮廓分析以检查其中是否有任何障碍物的区域。当按下左键时，该点的坐标存储在`top_left_corner`列表中，当释放左键时，坐标存储在`botton_right_corner`列表中。一旦释放左键，就在帧上绘制矩形。这个区域决定了机器人的成功。你可以自由尝试恐龙前方的不同区域。

```python
def drawBboxManual(action, x, y, flags, *userdata):
    global bbox_top_left, bbox_bottom_right
    # 文本原点坐标估计在右侧的一半上，使用以下逻辑。
    '''
    将屏幕分成12列和3行。文本的原点定义在
    第3行，第6列。
    '''
    org_x = int(6 * img.shape[1] / 12)
    org_y =  int(3 * img.shape[0] / 5)

    # 显示错误文本。
    cv2.putText(img, 'Error detecting Trex', (org_x + 20, org_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Please click and drag', (org_x + 20, org_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'To define the target area', (org_x + 20, org_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    # 鼠标交互。
    if action == cv2.EVENT_LBUTTONDOWN:
        # 获取坐标（存储为列表）。
        bbox_top_left = [(x, y)]
        # center_1：要绘制的点圆的中心。
        center_1 = (bbox_top_left[0][0], bbox_top_left[0][1])
        # 绘制一个小的实心圆。
        cv2.circle(img, center_1, 3, (0, 0, 255), -1)
        cv2.imshow("DetectionArea", img)

    if action == cv2.EVENT_LBUTTONUP:
        # 获取坐标（存储为列表）。
        bbox_bottom_right = [(x, y)]
        # center_1：要绘制的点圆的中心。
        center_2 = (bbox_bottom_right[0][0], bbox_bottom_right[0][1])
        # 绘制一个小的实心圆。
        cv2.circle(img, center_2, 3, (0, 0, 255), -1)
        # 定义边界框的左上角和右下角坐标为元组。
        point_1 = (bbox_top_left[0][0], bbox_top_left[0][1])
        point_2 = (bbox_bottom_right[0][0], bbox_bottom_right[0][1])
        # 绘制边界框。
        cv2.rectangle(img, point_1, point_2, (0, 255, 0), 2)
        cv2.imshow("DetectionArea", img)
    cv2.imshow("DetectionArea", img)
    # cv2.imwrite('MouseDefinedBox.jpg', cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
```
#### 3.4.4 检查白天或夜晚
&#8195;&#8195;在检测障碍物区域的上方，检查游戏模式是白天还是夜晚，逻辑非常简单。在夜间模式下，由于图像是黑色的，图像中的大多数像素亮度接近零。相反，在白天模式下，大多数像素的亮度接近255。

&#8195;&#8195;这里，`checkDayOrNight`函数返回平均亮度值。这个值将在后面的条件语句中使用，以确保[轮廓分析](https://learnopencv.com/contour-detection-using-opencv-python-c/)在两种情况下都能正常工作。

```python
def checkDayOrNight(img):
    # 初始化像素patch强度列表
    pixels_intensities = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 只对图像的四分之一区域进行采样，以减少计算量
	h = int(img.shape[0] / 4)
	w = int(img.shape[1] / 4)
	# 遍历像素并存储强度值
	for i in range(h):
		for j in range(w):
			pixels_intensities.append(img[i, j])
	# 计算列表中所有像素强度的总和，并除以像素数量，得到平均强度值val
	val = int(sum(pixels_intensities) / len(pixels_intensities))
	# 如果大于195，则认为是白天模式。
	if val > 195:
		return True
	else:
		return False
```
#### 3.4.5 初始化
这里，我们将pyautogui按键延迟设置为0毫秒，以便循环能够实时平滑运行

```python
# 设置按键延迟为0
gui.PAUSE = 0
 
# 初始化列表，用于保存边界框坐标
bbox_top_left = []
bbox_bottom_right = []
```
#### 3.4.6 主函数


##### 3.4.6.1 加载真实图像并捕获屏幕
- 下载的代码包含用于正常和黑暗模式的参考图像，需根据系统模式调整代码

- 使用`mss`库捕获屏幕，确保检查多显示器设置。
`screen.monitors[1]`返回一个字典，包含显示器的顶部、左侧、宽度和高度。在我的例子中（没有连接外部显示器），`monitors[0]`和`monitors[1]`返回相同的值。另外在在多显示器设置下要检查输出，例如使用MacBook时，因为它具有自动缩放功能。
```python
# 加载参考图像。
ref_img = cv2.imread('trex.png')

# 如果你处于Dark Mode，请取消注释以下行
# ref_img = cv2.imread('tRexDark.jpg')
screen = mss()
# 确定要捕获的显示。
monitor = screen.monitors[1]
# 检查mss返回的分辨率信息
print('MSS resolution info : ', monitor)
# 抓取屏幕并转为numpy array.
screenshot = screen.grab(monitor)
screen_img = np.array(screenshot)
```
##### 3.4.6.2 分析T-Rex的尺寸与屏幕分辨率的关系
&#8195;&#8195;本项目假设Chrome窗口覆盖屏幕约一半，我们在全高清和4K显示器上进行测试，根据分辨率得出了T-Rex的近似大小。如果你的系统分辨率不同，建议测试一下比例。只需捕获屏幕，裁剪出T-Rex，获取尺寸并将其除以系统分辨率即可。

```python
# 根据屏幕分辨率测试的TRex的高度和宽度

box_h_factor = 0.062962
box_w_factor = 0.046875
hTrex = int(box_h_factor * screen_img.shape[0])
wTrex = int(box_w_factor * screen_img.shape[1])
tested_area = hTrex * wTrex
# print('测试尺寸：', hTrex, '::', wTrex)
```

##### 3.4.6.3查找匹配
&#8195;&#8195;使用`getMatches`函数获取T-Rex关键点，并将其转换为数组以传递给`boundingRect`函数得到T-Rex的边界框。可以将其与测试尺寸进行比较，以决定是否为一个好的检测。

```python
# 获取关键点.
trex_keypoints = getMatches(ref_img, screen_img)
# 转换为numpy数组
kp_arary = np.array(trex_keypoints)
# 获取边界框的尺寸
x, y, w, h = cv2.boundingRect(np.int32(kp_arary))
obtained_area = w * h
```

##### 3.4.6.4 估计障碍物检测区域
&#8195;&#8195;经过测试发现，如果边界框面积在测试区域的10%到300%范围内，则为良好检测。障碍物检测区域设置为T-Rex前方区域，通过将边界框向右移动来实现。

&#8195;&#8195;你可以调整移动量，以找到最佳效果。如果检测未能满足区域条件，则触发冗余功能，要求你使用鼠标或触控板在T-Rex前绘制边界框。

```python
# 如果边界框面积在测试区域的10%到300%范围内，则为良好检测
if 0.1*tested_area < obtained_area < 3*tested_area:
       print('匹配项良好')
       # 设置目标区域的边界框坐标
       xRoi1 = x + wTrex
       yRoi1 = y
       xRoi2 = x + 2 * wTrex
       """
       将边界框的高度设置为原始高度的50%，以确保不捕捉到T-Rex下方的线条。你可以调整这个值以获得更好的定位。
       """
       yRoi2 = y + int(0.5*hTrex)
       # 在屏幕图像上绘制绿色矩形表示检测区域
       cv2.rectangle(screen_img, (xRoi1, yRoi1), (xRoi2, yRoi2), (0, 255, 0), 2)
       cv2.imshow('DetectionArea', cv2.resize(screen_img, None, fx=0.5, fy=0.5))
       cv2.imwrite('ScreenBox.jpg', screen_img)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
        
   else:
       print('匹配不佳，请手动设置目标区域')
       # 将图像宽高缩小到原来的一半并显示
       img = cv2.resize(screen_img, None, fx=0.5, fy=0.5)
       cv2.namedWindow('DetectionArea')
       # 设置鼠标回调函数drawBboxManual，允许用户手动绘制边界框
       cv2.setMouseCallback('DetectionArea', drawBboxManual)
       cv2.imshow('DetectionArea', img)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       
       # 根据手动绘制的边界框调整坐标
       xRoi1 = 2 * bbox_top_left[0][0]
       yRoi1 = 2 * bbox_top_left[0][1]
       xRoi2 = 2 * bbox_bottom_right[0][0]
       yRoi2 = 2 * bbox_bottom_right[0][1]
```
&#8195;&#8195;这段代码的逻辑是：首先检查自动检测的结果是否在合理范围内，如果是，则计算并绘制边界框；如果不是，则允许用户手动指定边界框位置。这种方法结合了自动检测和手动干预，以提高障碍物检测的准确性。
##### 3.4.6.5 若边界框未正确绘制则停止程序
&#8195;&#8195;在边界框绘制过程中，程序需要具备容错能力，能够识别不当输入并及时提醒用户，以确保最终的障碍物检测能够正常进行。

&#8195;&#8195;如果用户在图像上绘制的边界框不符合预期，程序将会停止运行，比如用户可能由于操作不当（点击、拖动、释放的方式不正确）导致错误的输入。这种错误常见于使用触控板时，比如用户可能不小心进行了多次点击或未正确释放鼠标。当发生这种情况时，我们使用`Tkinter`生成错误消息。
>&#8195;&#8195;`Tkinter`是Python的一个GUI库，能够创建窗口和消息框。当检测到输入错误时，程序将利用`Tkinter`弹出一个错误消息，提醒用户出现了问题。

```python
# 如果你执行了错误的点击-拖动操作，请重新开始.

# 检查边界框的左上角坐标和右下角坐标是否相等。如果是，意味着用户没有正确绘制边界框（例如，只进行了点击，而没有拖动和释放）。
if xRoi1 == xRoi2 and yRoi1 == yRoi2:
    print('请再次使用点击-拖动-释放方法绘制边界框')
    
    # 创建Tkinter窗口，并使用wm_withdraw方法将其隐藏。这是为了后续显示错误消息而不需要显示一个完整的窗口。
    window = Tk()
    window.wm_withdraw()
    # 屏幕的宽度和高度，计算出屏幕的中心位置
    win_width = str(window.winfo_screenwidth()//2)
    win_height = str(window.winfo_screenheight()//2)
    # 将窗口的几何形状设置为1x1像素，位置居中。
    window.geometry("1x1+"+win_width+"+"+win_width)
    # 使用Tkinter的消息框显示错误信息。这会弹出一个对话框，提示用户如何正确操作
    tkinter.messagebox.showinfo(title="Error", message="Please use click-drag-release")
    exit()
```

##### 3.4.6.6 调整缩放并定义屏幕捕获字典
&#8195;&#8195;MSS（Python屏幕捕获库）要求以字典的形式定义屏幕捕获的区域。如果MSS返回的屏幕分辨率与实际分辨率不同，则需要进行缩放调整。这通常是因为不同的显示器（特别是高分辨率显示器）可能会自动缩放显示内容。

&#8195;&#8195;通过打印`mss.monitors[1]`可以查看MSS检测到的显示器的宽度和高度。返回的字典包含了屏幕的宽度和高度。通过将mss返回的尺寸除以实际分辨率，可以获得缩放因子。

&#8195;&#8195;通过计算得到的坐标和尺寸，可以创建两个字典：一个用于障碍物检测，另一个用于判断白天或夜晚模式。

```python
# xRoi1, yRoi1, xRoi2, yRoi2 = (xRoi1 // 2, yRoi1 // 2, xRoi2 // 2, yRoi2 // 2)
 
# 创建用于MSS的字典，定义要捕获的屏幕大小
obstacle_check_bbox = {'top': yRoi1, 'left': xRoi1, 'width': xRoi2 - xRoi1, 'height': yRoi2 - yRoi1}
# 定义用于检测游戏模式（白天或夜晚）的区域，位于障碍物检测区域的上方
day_check_bbox    = {'top': yRoi1 - 2*hTrex, 'left': xRoi1, 'width': xRoi2, 'height': yRoi2 - 2*hTrex}
```
- `obstacle_check_bbox`：定义了障碍物检测区域的四个参数
- `day_check_bbox`定义了用于检测游戏模式（白天或夜晚）的区域，位于障碍物检测区域的上方：
	 * `top`: 设置为`yRoi1 - 2*hTrex`，即在障碍物检测区域的上方（向上偏移两倍的T-Rex高度）。
	* `left`: 保持与障碍物检测区域相同。
	* `width`: 宽度与障碍物检测区域相同。
	* `height`: 设置为`yRoi2 - 2*hTrex`，使得区域上部也向上偏移。

### 3.5 主循环
下面是主循环部分，实时捕获屏幕并分析游戏中的障碍物，触发T-Rex跳跃。

1. **捕获屏幕**：在每个循环中，捕获两个图像区域：一个用于障碍物检测，另一个用于判断游戏模式（白天或夜晚）。

2. **判断游戏模式**： `checkDayOrNight`函数返回一个布尔值，指示环境是白天还是夜晚。在白天模式下，环境亮度高，背景为白色，障碍物（空白处）为黑色；在夜晚模式下，环境较暗，背景为黑色，障碍物为白色。

3. **阈值处理**：在白天模式下，使用`THRESH_BINARY`进行阈值处理；在夜晚模式下，使用`THRESH_BINARY_INV`。这两种方法是图像处理中常用的二值化（黑白）处理技术。

4. **边界处理**：使用`copyMakeBorder`函数创建10像素的边界，以确保即使障碍物触碰到边界，轮廓检测也能正常工作。白天模式下边界为黑色，夜晚模式下边界为白色。

5. **轮廓分析和动作触发**：进行轮廓分析，如果检测到的轮廓数量大于1，则触发空格键，使T-Rex跳跃。

```python
while True:
    # 捕获障碍物检测区域
    obstacle_check_patch = screen.grab(obstacle_check_bbox)
    obstacle_check_patch = np.array(obstacle_check_patch)

    # 捕获游戏模式检测区域
    day_check_patch = screen.grab(day_check_bbox)
    day_check_patch = np.array(day_check_patch)

    # 将障碍物检测区域转换为灰度图
    obstacle_check_gray = cv2.cvtColor(obstacle_check_patch, cv2.COLOR_BGR2GRAY)

    # 检查游戏模式。
    day = checkDayOrNight(day_check_patch)

    # 根据游戏模式执行轮廓分析。
    if day:
        # 添加10像素的白色边界，使用阈值处理将图像转换为二值图
        obstacle_check_gray = cv2.copyMakeBorder(obstacle_check_gray, 10, 10, 10, 10,
                                         cv2.BORDER_CONSTANT, None, value=255)
        ret, thresh = cv2.threshold(obstacle_check_gray, 127, 255,
                                    cv2.THRESH_BINARY)
    else:
        # 添加10像素的黑色边界，使用反向阈值处理。
        obstacle_check_gray = cv2.copyMakeBorder(obstacle_check_gray, 10, 10, 10, 10,
                                         cv2.BORDER_CONSTANT, None, value=0)
        ret, thresh = cv2.threshold(obstacle_check_gray, 127, 255,
                                    cv2.THRESH_BINARY_INV)

    # 查找轮廓。
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
    # 打印轮廓数量。
    # print('检测到的轮廓：', len(contours))

    # 触发T-Rex跳跃
    if len(contours) > 1:
        gui.press('space', interval=0.1)
	
	# 显示障碍物检测区域图像
    cv2.imshow('Window', obstacle_check_gray)
    key = cv2.waitKey(1)
    # 按下‘q’键时退出循环并关闭窗口
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
```

