@[toc]
传送门：
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT](https://blog.csdn.net/qq_56591814/article/details/127313216?spm=1001.2014.3001.5501)
- [李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)
- [李沐论文精读系列三：MoCo、对比学习综述（MoCov1/v2/v3、SimCLR v1/v2、DINO等）](https://blog.csdn.net/qq_56591814/article/details/127564330)
- [李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5502)
- [李沐论文精读系列五：DALL·E2（生成模型串讲，从GANs、VE/VAE/VQ-VAE/DALL·E到扩散模型DDPM/ADM）](https://blog.csdn.net/qq_56591814/article/details/127749105?spm=1001.2014.3001.5501)
- [李沐论文精度系列之七：Two-Stream双流网络、I3D](https://blog.csdn.net/qq_56591814/article/details/127873069?spm=1001.2014.3001.5501)


## 一、DETR
>- 论文：[《End-to-End Object Detection with Transformers》](https://paperswithcode.com/paper/end-to-end-object-detection-with-transformers)、[官方代码](https://github.com/facebookresearch/detr)
>- 论文：[《Deformable DETR: Deformable Transformers for End-to-End Object Detection》](https://paperswithcode.com/paper/deformable-detr-deformable-transformers-for-1)、[官方代码](https://github.com/fundamentalvision/Deformable-DETR)
>- 参考李沐[《DETR 论文精读》](https://www.bilibili.com/video/BV1GB4y1X72R/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、[《DETR精读笔记》](https://blog.csdn.net/weixin_44966641/article/details/126559666)
### 1.1 前言
#### 1.1.1 研究动机：端到端目标检测的意义
&#8195;&#8195;DETR（DEtection TRansformer）是2020年5月发布在Arxiv上的一篇论文，可以说是近年来目标检测领域的一个里程碑式的工作。从论文题目就可以看出，DETR其最大创新点有两个：end-to-end（端到端）和 引入Transformer。
&#8195;&#8195;目标检测任务，一直都是比图片分类复杂很多，因为需要预测出图片中物体的位置和类别。以往的主流的目标检测方法都不是端到端的目标检测，因为：
1. 会加入很多的先验知识，预先生成一些锚框。比如one-stage方法（YOLO系列）中的Anchor模板；two-stage（R-CNN系列）中的proposal。
2. 这些方法都不是直接预测物体，而是利用anchor/proposal去做近似，或者像Anchor Free这样利用角点/中心点定位，设计一些回归/分类任务，间接出框。
3. Anchor Based、Anchor Free方法最后都会生成大大小小很多的预测框，必须在后处理时使用NMS去除这些冗余的框。

&#8195;&#8195;正是因为需要很多的人工干预、先验知识（Anchor）还有NMS，所以整个检测框架非常复杂，难调参难优化，并且部署困难（NMS需要的算子普通的库不一定支持，即不是所有硬件都支持）。所以说，一个端到端的目标检测是大家一直以来梦寐以求的。

#### 1.1.2 简介
1. DETR如何做到end-to-end

&#8195;&#8195;`DETR`利用`Transformer`这种全局建模的能力，直接把目标检测视为集合预测问题（即给定一张图像，预测图像中感兴趣物体的集合）。然后使用可学习的`object query`替代了生成`anchor`的机制；使用了新的目标函数，并利用二分图匹配的方式，强制模型对每个物体生只生成一个预测框，从而替代了NMS这一步（后面会细讲）。
&#8195;&#8195;`DETR`把之前不可学习的东西（anchor、NMS）变成可学的东西，删掉了这些依赖先验知识的部分，从而得到了一个简单有效的端到端的网络。所以`DETR`不需要费尽心思的设计anchor，不需要NMS后处理，也就没有那么多超参需要调，也不需要复杂的算子。

&#8195;&#8195;除了端到端这一点，`DETR`使用了 Transformer Encoder-Decoder 的架构。相比于原始的 Transformer，`DETR`是并行预测的（in parallel），即所有预测框是一起出框的。
>&#8195;&#8195;原始 `Transformer Decoder`用于自然语言处理，为了屏蔽未来信息，使用了masked self attention，所以是使用自回归的方式，一个接一个的顺序预测。
>&#8195;&#8195;但是目标检测任务是不需要顺序的，不存在说先得预测大物体才能预测小物体，或者说预测图片右边的物体必须依赖于图片左边的物体，所以没法做自回归预测；而且并行预测明显是更快更高效的。
2. 简单架构
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/feccf73a0e79d6e5da18a14e7145f8cb.png)
整个模型前向流程如上，训练分四个步骤：
- 使用CNN网络提取图片特征
-  全局建模：图片特征拉成一维，输入`Transformer Encoder` 中进行全局建模，进一步通过自注意力学习全局特征。
之所以使用`Transformer Encoder`，是因为Transformer 中的自注意力机制，使得图片中的每个点（特征）都能和图片中所有其他特征做交互了，这样模型就能大致知道哪块区域是一个物体，哪块区域又是另一个物体，从而能够尽量保证每个物体只出一个预测框。所以说这种全局特征非常有利于移除冗余的框。
- 通过`Transformer Decoder` 生成N个预测框set of box prediction（默认取N=100，也就是一张图固定生成100个预测框）。
-  计算二分图匹配损失（bipartite matching loss），选出最优预测框，然后计算最优框的损失。
计算N个预测框与所有GT box（真实框）的matching loss，然后通过二分图匹配算法来选出与每个物体最匹配的预测框。比如上图中有两个物体，那么最后只有两个框和它们是最匹配的，归为前景；剩下98个都被标记为背景（`no object`）。最后和之前的目标检测算法一样，计算这两个框的分类损失和回归损失。

&#8195;&#8195;推理时，前三步是一样的。通过decoder生成N个预测框后，设置一个置信度阈值进行过滤，得到最终的预测框。（比如设阈值=0.7，表示只输出置信度大于0.7的预测框，剩下都当做背景框）

&#8195;&#8195;总的来说，`Transformer Encoder`全局建模，用于区分物体；`Transformer Decoder`用于描绘物体边界，将物体位置补充的更完整（见4.2可视化）。

3. 性能

在摘要中，作者卖了一下DETR的优点：
- 简单性：不仅框架简单，可以进行端到端的检测，而且只要硬件支持CNN和Transformer就一定可以支持DETR。
- 在COCO数据集上的性能，和一个 训练好的Faster R-CNN baseline是差不多的，无论从内存、速度还是精度来说。
- 迁移性好：DETR框架可以很容易的拓展到其它任务上，比如在全景分割上的效果就很好（加个分割头就行）。

局限性：
- DETR对大物体检测效果不错，但是对小物体的检测效果不好（见实验4.1）。
&#8195;&#8195;前者归功于transformer可以进行全局建模，这样无论多大的物体都可以检测，而不像anchor based方法检测大物体时会受限于anchor的尺寸。后者是因为作者只是使用了一个简单的结构，很多目标检测的针对性设计还没有使用，比如多尺度特征、针对性的检测头。
- 训练太慢。
为了达到好的效果，作者在COCO上训练了500epoch，而一般模型训练几十个epoch就行了。

4. 改进

&#8195;&#8195;DETR精度只有`44 AP`，比当时SOTA模型差了近10个点，但是想法特别好，解决了目标检测里面的很多痛点，所以影响还是很大的。而且其本身只是一个简单的模型，还有很多可以改进的。比如 半年后提出的`Deformable-DETR`, 融入了多尺度特征，成功解决小物体检测效果不好的问题，还解决了训练慢的问题。

&#8195;&#8195;另外`DETR`不仅是一个目标检测方法，还是一个拓展性很强的框架。其设计理论，就是适用于更多复杂的任务，使其更加的简单，甚至是使用一个框架解决所有问题。后续确实有一系列基于它的改进工作，比如Omni-DETR, up-DETR, PnP-DETR, SMAC-DETR, DAB-DETR, SAM-DETR, DN-DETR, OW-DETR, OV-DETR等等，将DETR应用在了目标追踪、视频领域的姿态预测、语义分割等多个视觉任务上。（感觉类似CLIP出来之后，有一系列基于它的工作）


### 1.2 相关工作
这一块介绍了三部分：
1. 介绍之前的集合预测工作
2. 如何使用Parallel Decoding让transformer可以并行预测
3. 目标检测

- 集合预测：以前也有集合预测这一类的方法，也做了二分图匹配，也可以做到每个物体只得到一个预测框，而不需要NMS。但是这些方法性能低，要不就是为了提高性能加了很多人工干预，显得复杂。
- encoder-decoder：以前也有用encoder-decoder做检测，但都是17年以前的工作，用的是RNN的结构，效果和性能都不好（RNN自回归，效率慢）。

&#8195;&#8195;所以对比以前的工作发现，能让DETR工作的好最主要的原因就是使用了Transformer。比如上面两点，都是backbone学的特征不够好，才需要使用很多人工干预，或者说模型效果性能都不好。<font color='red'> 所以说DETR的成功，还是Transformer的成功。 </font>

### 1.3  算法
#### 1.3.1 目标函数
&#8195;&#8195;DETR 模型每次输出固定个数（N=100）的预测框，如何判断哪个预测框匹配哪个GT box呢？这就涉及到二分图匹配算法。
&#8195;&#8195;假设现在有 3 个工人和 4 个任务，由于每个工人的特长不一样，他们完成不同任务的时间（成本）也是不一样的，那如何分配任务能够使总的成本最低呢？最直接的最暴力的方法，就是用直接遍历，找出各种排列组合中的最优组合，但这样的复杂度无疑是很高的。匈牙利算法是解决该问题的一个知名且高效的算法，能够以较低的复杂度得到唯一的最优解。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2ae6fef675e5f796f0e931ead5f6a207.png)
&#8195;&#8195;在 scipy 库中，已经封装好了匈牙利算法，只需要将成本矩阵（cost matrix）输入进去就能够得到最优的排列。在 DETR 的官方代码中，也是调用的这个函数进行匹配（`from scipy.optimize import linear_sum_assignment`）。

&#8195;&#8195;从N个预测框中，选出与M个GT Box最匹配的预测框，也可以转化为二分图匹配问题，这里需要填入矩阵的“成本”，就是每个预测框和GT Box的损失。对于目标检测问题，损失就是分类损失和边框损失组成。即： $$L_{match}=(y_{i},\widehat{y}_{\sigma (i)})=-\mathbb{1}_{\{c_i\ne\empty\}}\hat{p}_{\sigma(i)}+\mathbb{1}_{\{c_o\ne\empty\}}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})−1$$

所以整个步骤就是：
1. 遍历所有的预测框和GT Box，计算其loss。
2. 将loss构建为cost matrix，然后用scipy的`linear_sum_assignment`（匈牙利算法）求出最优解，即找到每个GT Box最匹配的那个预测框。
3. 计算最优的预测框和GT Box的损失。常规目标检测算法损失为：（分类+回归）
$$\mathcal{L}_{Hungarian}(y,\hat{y})=\sum_{i=1}^N[-\log \hat{p}_{\hat{\sigma}(i)}(c_i)+\mathbb{1}_{c_i\ne\empty}\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}(i))}]$$

但是在DETR 中，损失函数有两点小改动：
- 去掉分类损失中的log
对于前一项分类损失，通常目标检测方法计算损失时是需要加 log的，但是 DETR 中为了保证两部分损失的数值区间接近，便于优化，选择了去掉 了log；
- 回归损失为L1 loss+GIOU
 对于后一项回归损失，通常方法只计算一个 `L1 loss`（预测框和真实框坐标的L1 损失）。但是L1 loss和预测框的大小有关，框越大损失越大。 DETR 中，用 Transformer 提取的全局特征对大物体比较友好，经常出一些大框，这样就不利于优化，因此作者这里还添加了一个 `GIoU Loss`。

>&#8195;&#8195;其实这里使用匈牙利算法找最优匹配，和之前使用anchor/proposal这种先验知识来匹配预测框和真实框是差不多的，只不过这里的约束更强，也就是强制模型对每个物体只输出一个预测框。
>&#8195;&#8195;关于`GIOU Loss`，可以参考我之前的帖子[《YOLOv1——YOLOX系列及FCOS目标检测算法详解》](https://blog.csdn.net/qq_56591814/article/details/125940060?spm=1001.2014.3001.5501)4.3章节`CIoU Loss`，其中详细介绍了回归损失使用L1 loss、IoU Loss、GIoU Loss和CIoU Loss的优劣。
#### 1.3.2 模型结构
&#8195;&#8195;作者在这部分给出了模型更详细的框架，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f9c09ac227ed97c1aab747d4c5e3457c.png)

下面参考官网的一个demo，以输入尺寸3×800×1066为例进行前向过程：
- CNN提取特征（`[800,1066,3]→[25,34,256]`）
backbone为ResNet-50，最后一个stage输出特征图为25×34×2048（32倍下采样），然后用1×1的卷积将通道数降为256；
- `Transformer encoder` 计算自注意力（`[25,34,256]→[850,256]`）
将上一步的特征拉直为850×256，并加上同样维度的位置编码（Transformer本身没有位置信息），然后输入的Transformer encoder进行自注意力计算，最终输出维度还是850×256；
- `Transformer decoder`解码，生成预测框
&#8195;&#8195;decoder输入除了encoder部分最终输出的图像特征，还有前面提到的`learned object query`，其维度为100×256。在解码时，`learned object query`和全局图像特征不停地做`across attention`，最终输出100×256的自注意力结果。
&#8195;&#8195;这里的`object query`即相当于之前的anchor/proposal，是一个硬性条件，告诉模型最后只得到100个输出。然后用这100个输出接FFN得到分类损失和回归损失。
- 使用检测头输出预测框
检测头就是目标检测中常用的全连接层（FFN），输出100个预测框（$x_{center},y_{center},w,h$）和对应的类别。
- 使用二分图匹配方式输出最终的预测框，然后计算预测框和真实框的损失，梯度回传，更新网络。


>&#8195;&#8195; `object query`准确来说是`learned positional embedding`，我感觉有点就类似[Group ViT](https://paperswithcode.com/paper/groupvit-semantic-segmentation-emerges-from)中的 grouping操作。简单说如果有一些聚类的中心点，从这些中心点开始发散，把周围相似的点逐渐扩散成一个group。
>&#8195;&#8195; Group ViT使用计算单元Grouping Block，将可学习的Group Tokens一点点的group起来，最终变成物体掩模（segmentation mask）。Group ViT结构如下图所示：
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3999d6a254fb430e74f1850f1e0103da.png)
>- ViT 的Linear Projection层将图片分割成patch然后映射为`Pacth embeddings`，即图中token  $\mathbf{s}_i^1$ （维度196×384），然后和`learned group token` $\mathbf{g}_i^1$一起输入Transformer Layer。
>- 学习6层之后使用Grouping Block模块，将图像块 token 分配到各个 group token 上，合并成为更大的、更具有高层语义信息的 group，即Segment Token（维度64×384，相当于一次聚类的分配）。
>- 重复上述过程：添加新的 Group tokens $\mathbf{g}_i^2$（8×384），经过 3 层 Transformer Layers 的学习之后，再次经过grouping block 分配，得到 $\mathbf{s}_i^3$（8×384） 。


>&#8195;&#8195;有兴趣的可以看帖子[《李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）》](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5501)


除此之外还有部分细节：
- Transformer-encode/decoder都有6层
- 除第一层外，每层Transformer encoder里都会先计算object query的self-attention，主要是为了移除冗余框。这些query交互之后，大概就知道每个query会出哪种框，互相之间不会再重复（见实验）。
- decoder加了auxiliary loss，即每层decoder输出的100×256维的结果，都加了FFN得到输出，然后去计算loss，这样模型收敛更快。（每层FFN共享参数）

#### 1.3.3 伪代码
下面是论文中给出的简化代码，可以直接跑，只是精度会差两个点。
```python
import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1) # 1×1卷积层将2048维特征降到256维
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1) # 类别FFN
        self.linear_bbox = nn.Linear(hidden_dim, 4)                # 回归FFN
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim)) # object query
        # 下面两个是位置编码
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
       					 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
       					 ], dim=-1).flatten(0, 1).unsqueeze(1) # 位置编码
       					 
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200)
logits, bboxes = detr(inputs)
```
### 1.4 实验
#### 1.4.1 对比 Faster RCNN 
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/24a9baf30e77527356aa25302e583fc0.png)

- 最上面一部分是 Detectron2 实现的 Faster RCNN ，但是本文中作者使用了很多trick
- 中间部分是作者使用了GIoU loss、更强的数据增强策略、更长的训练时间来把上面三个模型重新训练了一次，这样更显公平。重新训练的模型以+表示，参数量等这些是一样的，但是普偏提了两个点
- 下面部分是DETR模型，可以看到参数量、GFLOPS更小，但是推理更慢。模型比 Faster RCNN 精度高一点，主要是大物体检测提升6个点AP，小物体相比降低了4个点左右
- 参数量、计算量和推理速度之间并没有必然的关系

2. transformer encoder/decoder层数消融试验，结果是层数越多效果越好，但是考虑到计算量，作者最后选择6层。
#### 1.4.2 可视化
1. 编码器自注意力图可视化

&#8195;&#8195;下图展示了对于一组基准点（图中红点）的 Encoder 注意力热力图的可视化，即基准点与图像中所有其他点的自注意力分布。
&#8195;&#8195;可以观察到，Transformer Encoder的自注意力已经做得非常好了， 基本能够非常清晰地区分开各个物体，甚至已经有一点实例分割的 mask 图的意思了。而且在严重遮挡的情况下，也能够清楚地区分左侧的两头牛。
&#8195;&#8195;所以Transformer Encoder的作用，正是可以把图片中的物体清楚地区分开，再在这个基础上做分割或者检测就会简单很多，效果也更好。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/53c24eb9bdbce61ce25a3d7f2194160f.png)
2. 解码器注意力图可视化

&#8195;&#8195;通过前面的可视化，我们已经看到Encoder 学习的全局特征，基本已经能够区分开图中不同的物体。但是对于目标检测来说，大致地区分开不同的物体是不够的，我们还需要物体边界框的精确坐标，这部分就由 Decoder 来做。

&#8195;&#8195;下图是 将Decoder自注意力用不同的颜色可视化出来 ，比如左图中的两头大象分别由蓝色和橙色表示。右侧斑马也用三个颜色表示。

&#8195;&#8195;可以观察到，即使在严重遮挡的情况下，每个物体边界的注意力还能区分开来，如大象尾巴、象腿等处。而且两头象的皮肤还有斑马上的花纹都差不多，但是轮廓都分的很清楚。作者认为这是 Decoder 在区分不同物体边界的极值点（extremities），在 Encoder 能够区分开不同的物体之后，Decoder 就只需要关注物体的边界位置，解决遮挡这些问题，最终精准地预测出不同物体的边框位置。因此，Encoder-Decoder 的结构是必要的（类似U-Net）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a956e22a1db42a563ab3e45e9a64c3b1.png)
3. object query 的可视化

&#8195;&#8195;下图将 COCO2017 验证集中所有图片的预测框可视化了出来，在N=100个预测框中只取了20个。下图每一个框代表一个object query，并且每张图都根据其尺寸进行了归一化（相当于每张图都除以高宽，得到1×1大小）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2edcd97f562458d6216442f7dfd792ba.png)
- 这些点用不同颜色进行区分，绿色表示小的bounding box，蓝色表示大的纵向box，红色表示大的横向box。
- 不同query负责检测不同的物体
&#8195;&#8195;从上面可以看出，不同的query负责检测不同位置不同大小的物体。比如上图第一个query，就是负责检测图片左侧靠下部分的小物体，中心部分的大物体，其它以此类推，遍历100个query后，图片中存在这个物体的就返回预测框。
- 对比anchor
&#8195;&#8195;query也类似anchor，都是检测图片中某个部位有没有某种物体。只不过anchor 需要先验地手动设置，而query是与网络一起端到端学习的。
- COCO数据集中心都有大物体
上图还可以看到，每张图中心都有红色的竖线，表示每个query都会检测图片中心是否有横向的大物体。这是因为COCO数据集图片中心往往都有一个大的物体，query则学到了这个模式，或者说分布。

## 二、ViLT
>- 论文[《ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision》](https://paperswithcode.com/paper/vilt-vision-and-language-transformer-without)、[代码](https://github.com/dandelin/vilt)
>- 李沐[《ViLT 论文精读》](https://www.bilibili.com/video/BV14r4y1j74y/?spm_id_from=333.788&vd_source=21011151235423b801d3f3ae98b91e94)、帖子[《2021： ViLT》](https://blog.csdn.net/weixin_42653320/article/details/123041729)、知乎[《ViLT：最简单的多模态Transformer》](https://zhuanlan.zhihu.com/p/369733979)
>- ViLT是直接在ViT的基础上改进的，建议先了解一下ViT模型。具体可以参考我的帖子[《李沐论文精读系列二：Vision Transformer、MAE、Swin-Transformer》](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5501)

### 2.1 前言
&#8195;&#8195;**天下苦目标检测久矣！！！**

&#8195;&#8195;`DETR`一经出世就广受热捧，因为它可以进行端到端的目标检测，使得目标检测的框架和流程都大大简化；另外引入Transformer之后，整个检测性能也不错，所以推动着整个目标检测工作都往这个方向走。

&#8195;&#8195;`ViLT`也是一个极其简单的视觉文本多模态的框架，其最主要贡献，就是把多模态学习框架中的目标检测，也就是论文中反复强调的`Region Feature`（区域性特征）直接拿掉了。这个操作简直是大快人心，因为它极大地简化了视觉模态特征的抽取过程，大大提高了模型的推理速度，可称之为多模态领域一个里程碑式的工作。

**1. 抽取视觉特征的三种方式**

&#8195;&#8195;现有的`VLP`模型（Vision-and-Language Pre-training，视觉文本多模态模型）抽取文本特征基本上都使用 `pre-trained BERT`的 tokenizer来得到`text embedding`，但抽取视觉特征存在着差异。往往处理视觉特征的网络越复杂，模型效果就越好，所以抽取视觉特征是现有VLP模型的瓶颈。图下图所示，获取`visual embedding`的方法总共有三大类：
- `Region Feature`：通常采用Faster R-CNN二阶段检测器提取区域性特征，这种操作也是最贵的；
>比如图像经过ResNet101 backbone提取特征，再经过RPN得到一些RoI，然后使用NMS过滤冗余的RoI，最后经过RoI Head得到一些一维的向量（Region Feature），也就是一个个bounding box。
- `grid feature`：将CNN backbone得到的feature map，作为网格特征，大大简化了计算量
>比如将ResNet50最后得到的7×7特征图拉直为一个序列，或者是上一层的14×14的特征图
- `patch projection`：使用类似[ViT](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5502)模型中的`patch projection`层直接得到patch embeddings，ViLT是首个这么做的，有三个原因：
	- **不需要使用额外的网络**，无论是CNN backbone还是目标检测，都非常贵。
	- **不需要缓存特征**。前两种方法都需要在线下使用预训练的模型提前抽取好图片特征，然后再训练。虽然这样训练还是比较轻量的，但在部署的时候是一个很大的局限性。真实场景里每时每刻都在生成新数据，都需要抽取新数据的特征，这时推理速度就是一大瓶颈了，所以作者才想设计一个更轻量更简单的视觉特征抽取方案。
	- **ViT的`patch projection`层表现很好**。在`ViT`论文中，其作者对比了使用CNN  backbone先抽特征再使用patch projection层（[ViT Hybrid混合模型](https://blog.csdn.net/qq_56591814/article/details/127358168?spm=1001.2014.3001.5502)）和直接使用patch projection层（`ViT`）两种将图片映射成`patch embedding`的方式，发现最终结果差不多，可见只用`patch projection`层模型也能工作的很好。受此启发，<font color='red'> 作者直接将ViT的`patch projection`层拿过来用，替代之前的提取网络！</font>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/59a32a3449ae17fe893b9a2723831b5c.png#pic_center)
&#8195;&#8195;这三种方法都是将抽取到的`visual embedding`当做一个序列，和同样长度的`text embedding`序列一起输入Transformer做后续的特征融合，其性能和运行时间如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6d6d07bf791adba36a24750e99e73a40.png#pic_center)
- `Region Feature`：整个运行时间900ms，其中视觉特征抽取就要885ms，处理文本特征只有15ms，浪费了太多计算资源在视觉特征的处理上，比后面处理多模态融合的时间还多，所以也不是很合理（VLP应该花费更多精力在多模态特征的融合上）。
- `ViLT`相比`Region Feature`方法性能下降了很多，但是高于`grid feature`方法，而且抽取视觉特征抽取只需要0.4ms，训练时间上千倍的减少，这也是本文的最大卖点。

**2. 模态交互**

 多模态特征的融合有两种常见方式：
 - Single-stream：单通路结构，文本特征和图像特征直接concat连接，然后输入一个transformer进行交互；
 - Dual-stream：双通道结构，文本特征和图像特征分别过一个文本模型和图像模型，充分挖掘单模态特征，然后再经过一个transformer layer做融合。

这两种方法的效果其实差不多，dual-stream明显更贵，参数量、计算量更多，所以作者采用了Single-stream。
### 2.2 引言
&#8195;&#8195;2017年以来，NLP领域基本就是被transformer一统江湖了，所以VLP模型的文本处理也只能这么做，没有什么好改的。但VLP要做Vision-Language Pre-training，就**必须将图像的像素，转换为带有语义性质的离散的特征**，这样才可以和文本tokens匹配起来，才能在后续输入transformer时进行特征融合，这也是大家研究的重点。
1. 目标检测抽特征
&#8195;&#8195;图像的像素不能直接扔给transformer，不然序列长度就太长了。`ViT`提出将图片分割成一个个固定大小的patch，然后使用线性层映射为patch embedding输入网络（比如patch size=16×16时，处理后序列长度从224×224降为14×14）。但`ViT`是2021年的工作，之前的工作这部分处理都是依赖于一个目标检测器。
>选用目标检测器来处理图像特征有很多原因：
>- 目标检测是一个天然的离散化过程，其得到的`bounding box`代表一个个物体，有明确的语义信息（可类比文本中的token），而且还是离散化的。所以这种方法简单粗暴，效果也好。
>- 以前的`VLP`下游任务（包括VLP领域的数据集），不管是VQA（视觉问答，给定图像回答问题）、visual captioning（VC，视觉字幕，给定图片或视频，生成对应的文本描述）还是image-text-retrieval（图文检索）等等，这些任务都跟物体有非常强烈的联系，一旦检测到物体，就很可能做出正确的答案。所以选择目标检测作为多模态模型的一部分，也是很合理的。
>
>目前VLP模型的目标检测器都是在 [Visual Genome数据集](https://paperswithcode.com/paper/visual-genome-connecting-language-and-vision)上预训练的，其包含1400类物体和400类属性。如果物体类别太少，就和文本token匹配不起来了，因为文本token基本是无穷无尽的。

2. Pixel-BERT抽取网格
使用在ImageNet上预训练好的ResNet抽取特征，将ResNet最后得到的特征图当成是一个离散的序列，然后和文本特征一起输入transformer做融合，速度就快很多。

3. `ViLT`三大贡献
	- 使用 `patch projection`层抽取视觉特征，极大简化了多模态学习框架，减少了运行时间和参数量
	- `ViLT`是第一个不使用卷积特征和区域性特征的同时（Without Convolution or Region Supervision），模型性能还表现的比较好的模型
	-  首次在`VLP`训练中使用了整词掩码和图像数据增强，并被证明可以明显提升模型性能。
>&#8195;&#8195;CV领域早已证明数据增强是一个很有用的trick，但在多模态领域，始终要考虑图文匹配的问题，所以一直没有使用。比如文本是“草地上有只小白兔”，对图像使用数据增强，可能就不是白色兔子和绿色的草地了，这时新生成的图文对就不是一个正确的对。

### 2.3 背景知识
>这部分相当于多模态工作的简单综述，很多介绍多模态的工作都使用了下面这张图。
>
&#8195;&#8195;首先，作者根据1）图像和文本的表达力度（参数量/计算量）是否平衡（图像和文本特征一样重要，理论上比重应该差不多）；2）多模态特征怎样融合；将VLP模型归结为四类：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0fae1a29b70d52c23783bb64e997fa55.png)
- VE, TE和 MI 分别表示visual embedder, textual embedder以及modality interaction（模态融合）
- a：VSE/ SCAN等模型的做法，视觉特征的处理远大于文本特征，模态融合只使用了简单的点乘操作或很简单的浅层attention网络；即VE > TE > MI
- b：[CLIP](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5501)，每个模态单独使用transformer encoder，两者计算量差不多。特征融合部分，只是简单的计算了一下图文特征的相似性；即 VE = TE > MI
CLIP特别适合需要图文特征（[GroupViT/GLIP](https://blog.csdn.net/qq_56591814/article/details/127421979?spm=1001.2014.3001.5501)等）或者是图文检索的任务，但做VQA或者visual reasoning（视觉推理，更难的VQA）这种需要视觉推理的任务时，会稍逊一筹。因为一个简单的不可学习的点乘，是没法做深层次的特征融合和分析的。
- c：这些年80%的工作都是这个方向，比如ViLBERT、UNITER、Pixel-BERT等等。文本侧很轻量，但图像侧使用很重的CNN抽取特征；最后特征融合使用了Transformer，所以VE > MI > TE；
>&#8195;&#8195;对于大部分视觉文本多模态任务来说，模态融合一定要做的比较好，最后的效果才会比较好，跟之前抽取的特征关系不太大，即理想框架应该是 `MI > VE = TE`。
- d：文本视觉特征的抽取都很轻量，特征融合使用transformer，即 `MI > VE = TE`。

### 2.4算法
#### 2.4.1 模型结构
`ViLT`模型结构如下图所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8ef935d8be359f6b5026d8901bad6d05.png)
- 文本经过pre-trained BERT tokenizer得到word embedding（前面有CLS token，图中*表示）
- 图片经过ViT patch projection层得到patch embedding（也是用*表示CLS token）；
- 文本特征+文本位置编码+模态嵌入得到最终的text embedding，图像这边也是类似的操作得到image embedding；二者concat拼接之后，一起输入transformer layer，然后做MSA交互（多头自注意力）
>&#8195;&#8195;模态嵌入即Modal-type embedding，使用0代表文本，1代表图像。因为在`Single-stream`模型中，图文特征是直接拼在一起输入一个transformer。如果不进行标注，模型是不知道哪一块是文本，哪一块是特征，这样不利于学习。加了模态嵌入可以区分之后，模型就可以在训练时找出图文之间的关系，学习的更好。

论文中也给出了前向过程的数学表达式：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d405273f6a124483dc9bcd754135c5e9.png)
- 文本$t$tokenizer后得到L×H维的word embedding，再加上(L+1)×H维的位置编码和cls token得到文本嵌入$\bar{t}$，同理得到N×H维的图片嵌入 $\bar{v}$；
- $\bar{t}$和$\bar{v}$分别加上各自的模态嵌入$t^{type}$和$v^{type}$之后，拼接得到输入序列$z_0$；
- $z_0$输入transformer layer做后续的MSA等操作得到最后的输出$p$。
#### 2.4.2 目标函数
&#8195;&#8195;`ViLT`使用了一般VLP模型常用的目标函数，即图文匹配loss（ ITM，image text matching）和 BERT的掩码学习loss（MLM，Masked Language Modeling）。另外`ViLT`还使用了Word Patch Alignment（WPA）。
-  `ITM loss`：以50%的概率将文本对应的图片随机替换成数据集中的其它图片，然后将文本CLS token对应输出使用一个FC层映射成一个二值logits，用来判断图像文本是否匹配；
- `MLM loss`：随机mask一个文本token，然后将其重建出来。
其实图片这边也可以使用masked patch 重构任务，但是当时MAE还没出来，重构效果还不够好，所以作者没有这么做。后续有[VL-BEiT](https://paperswithcode.com/paper/vl-beit-generative-vision-language)，就使用了图像-文本掩码任务（masked vision-language modeling ）。
- `WPA`：简单理解就是将文本和图像的输出都当做一个概率分布，然后使用最优运输理论计算一下两者的距离

**模型总结**
&#8195;&#8195;`ViLT`模型确实很简单，如果将图片这边patch embedding也看做token embedding，那这就是一个`BERT`模型；如果将文本特征拿掉，那这就是一个`ViT`。

#### 2.4.2 整词掩码

&#8195;&#8195;  另外`ViLT`还使用了`whole word masking`技巧，即将整个token masked掉而不是只掩码子词，避免了只通过单词上下文就可以进行预测。比如将“giraffe”词tokenized成3个部分["gi", "##raf", "##fe"]，可以mask成["gi", "[MASK]", "##fe"]，但是前后分别为"gi"和"e"的单词本来就没多少，模型很可能只通过文本的上下文信息就预测出这个单词就是“giraffe”，导致图像信息没有利用到，图文匹配loss就失去了意义。
#### 2.4.3 图像增强
&#8195;&#8195;上面提到的c类VLP模型，需要缓存特征，即在训练前就提前抽取好视觉特征，所以在下游任务微调时没法做图像数据增强的（如果想使用图像增强，就得重新抽取，成本太高，所以直到21年都还没有人这么做）。
&#8195;&#8195;`ViLT`是一个端到端的模型，作者在微调时直接就上了 [RandAugment](https://paperswithcode.com/paper/randaugment-practical-data-augmentation-with)。考虑到需要图文匹配，作者改动了其中两处，即去掉了cutout和color inversion（前者是随机去掉图像中某一区域，后者是进行颜色变换）。
### 2.5 实验
#### 2.5.1 训练数据集
&#8195;&#8195;`ViLT`使用四个多模态数据集进行预训练：MSCOCO、VG、SBU、GCC。这四个数据集也叫4M，因为所有图片加在一起是400万张左右。
- MSCOCO：即Microsoft COCO，每张图片有五个 captions（描述图像的标题），平均标题长度12。模型生成的标题要尽量和这56万标题相似。（图像11万，图文对56万）
- VG：即Visual Genome，10.8万图片，541万标题，标题平均长度5.5
- SBU/GCC ：即SBU Captions和Google Conceptual Captions。这两个数据集都是一张图配一个标题，但数据集建造者只给了图片链接，其中不少失效了，所以作者只使用了其中能用的部分，在下图用`†`表示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98e728b56c4e5be193eed2ad54e4d40a.png)
#### 2.5.2 性能对比
**1. 分类任务**
&#8195;&#8195;下面在`VQAv2`和`NLVR2`两个数据集上对比了 `ViLT-B/32`和其它模型的性能。这两个都可以简单理解为（转化为）多模态领域的分类任务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2751d68ec30b93fa835915f9aaa60b47.png)
- 上图a表示使用改进的`RandAugment`数据增强策略；+表示训练更长的时间（20万steps）
- `ViLT-B/32`在速度和精度之间平衡的比较好
**2. 检索任务**

作者比较了 `ViLT-B/32`和其它模型在Flickr30k和MSCOCO两个检索任务上的性能
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b223ebe274a670fa72be2b566f762369.png)
上图是Zero-Shot的结果，下图是微调的结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e05ebb53ac959fcb5cca53e8d72afdc7.png)
可以看到`ViLT-B/32`的取舍做的不错，速度很快，不过精度还有待提高。
#### 2.5.3 消融试验
下面是作者做的一些消融试验：
>w表示整词掩码，m是图像的完形填空后重建，论文称之为MPP、a表示数据增强
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3f172097251930124b3673cbe2cfcab2.png)
- ViLT也是一种自监督训练的形式，可以看到随着训练时间的增加，模型性能一直提升
- 对比三种策略，整体掩码和数据增强都比较有效，特别是数据增强。MPP效果不好，作者后续没有再使用

### 2.6 结论
&#8195;&#8195;`ViLT`提出了一个极简的多模态框架，成功将`BERT`和`ViT`应用于多模态Transformer中。`ViLT-B/32`证明了不使用卷积特征或者`Region Feature`，只需要一个patch projection层，模型效果也不错，但性能还是有待提高。作者提出三种改进方向：
- Scalability：模型越大越好，数据集越多越好
- 使用masked vision-language modeling，即图像部分也做掩码重建（完形填空）。后续[VL-BEiT](https://paperswithcode.com/paper/vl-beit-generative-vision-language)做到了这一点。
- 数据增强：消融试验中数据增强的提升是最大的，作者希望可以优化这一块。
