@[toc]
- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)
## 一、图像金字塔
&#8195;&#8195;图像金字塔是图像处理中的一种常用技术，它通过对原始图像进行一系列的降采样操作来创建一组图像。在OpenCV中，图像金字塔有两种类型：高斯金字塔和拉普拉斯金字塔。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45108a4f49124836a621aec94ec31da6.png)
图像金字塔在多种图像处理任务中有应用，包括：

- 图像缩放：快速放大或缩小图像。
- 图像融合：将不同分辨率的图像融合在一起。
- 图像分割：在不同的分辨率层次上分析图像。
- 多尺度目标检测：在不同尺度上检测目标。
### 1.1 高斯金字塔
&#8195;&#8195;高斯金字塔 (Gaussian Pyramid)是通过连续应用高斯模糊和降采样来构建的。每一层的图像都是上一层的图像经过高斯模糊后，删除其偶数行和列得到的。这样，金字塔的每一层都比上一层小，分辨率也低。

&#8195;&#8195;在构建高斯金字塔时，通常使用的是5x5的高斯卷积核来进行高斯模糊。这个卷积核的权重是根据高斯分布（正态分布）计算得出的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce5f68dad783411494851db5a565d691.png#pic_center =600x)
&#8195;&#8195;将$G_i$（表示不同层级的图像）与高斯卷积核进行卷积之后，去除所有偶数行和列，就得到一次下采样的结果。每次下采样之后，图像尺寸都减半，多次处理就得到整个高斯金字塔。

&#8195;&#8195;高斯模糊的过程，类似于将每个像素的特征分配一部分到邻域像素中，所以减去一半的行和列，图像基础特征不变。不过每次下采样，还是会丢失部分图像信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1345e0653fc945c0a178a03d6c4ba98d.png#pic_center =300x)
具体来说，我们使用下面两个函数进行操作：
- `cv2.pyrDown`：使用高斯金字塔进行一次降采样。
- `cv2.pyrUp`：上采样，通过插入0来扩大图像，然后使用与pyrDown相同的卷积核进行卷积。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b73433d2d2847c49eac52bd76646ec7.png#pic_center =600x)
&#8195;&#8195;使用下采样时相同的高斯卷积核进行卷积，可以达到将原先像素分配到邻近插入的0像素的效果，近似恢复原图信息。下面进行演示：

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('./lena.png')
dst1 = cv2.pyrDown(img)
dst2 = cv2.pyrUp(dst1)
print(f' {img.shape=} {dst1.shape=} {dst2.shape=}')

plt.figure(figsize=[16,8])
plt.subplot(131); plt.imshow(img[:,:,::-1]);  plt.title("img");
plt.subplot(132); plt.imshow(dst1[:,:,::-1]);  plt.title("pyrDown");
plt.subplot(133); plt.imshow(dst2[:,:,::-1]);  plt.title("pyrDown+pyrUp");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fd1a4c97ce1449a920d94f300325753.png)
可以看到，先下采样再上采样，原图的信息还是有部分丢失，处理后的图像没有原图清晰。
### 1.2 拉普拉斯金字塔
&#8195;&#8195;拉普拉斯金字塔是基于高斯金字塔构建的，它主要用于图像的重建。拉普拉斯金字塔的每一层都是通过当前层原图减去其先下采样后上采样的图像得到的，代表了二者之间的残差。用数学公式表示就是：
 $$L_{i}=G_{i}-PyrUp(PyrDown(G_{i}))$$
 其中，$L_{i},G_{i}$分别是某一层的原始图像及其拉普拉斯金字塔图像。
 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8fce31fae685426fb675af4eaa0e36b9.png#pic_center =600x)

```python
lap0=img-dst2

plt.figure(figsize=[16,8])
plt.subplot(131); plt.imshow(img[:,:,::-1]);  plt.title("img");
plt.subplot(132); plt.imshow(dst2[:,:,::-1]);  plt.title("pyrDown+pyrUp");
plt.subplot(133); plt.imshow(lap0[:,:,::-1]);  plt.title("lap0");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/697d11c490744001bf6a839178766079.png)
&#8195;&#8195;拉普拉斯金字塔包含了图像的细节和高频信息，所以可以间接地包含轮廓信息，多用于图像的多尺度表示和图像重建任务。



## 二、特征检测
### 2.1 基本概念

&#8195;&#8195;在OpenCV中，特征检测是一种用于从图像中识别出关键点或局部显著区域的技术，这些关键点通常对物体的识别、跟踪、匹配等任务非常有用。特征检测可以帮助我们在不同视角、尺度、光照条件下识别出同一对象。下面介绍一些特征检测的基本概念和相关算法。

&#8195;&#8195;**特征 (Features)**：图像中的特征指的是一些局部的、在周围区域中显著的点、边缘或区域。常见的特征包括角点、边缘和斑点。特征应该具有如下性质：
- **独特性**：与其他区域有显著区别。
- **可重复性**：在不同视角、尺度下，依然能被检测到。
- **抗干扰性**：对光照、噪声等的变化具有一定的鲁棒性。

&#8195;&#8195;**关键点 (Keypoints)**：关键点是图像中特别突出的点，通常与图像的局部结构（如角点、斑点）相关。关键点的检测是特征检测的第一步，它们是后续特征描述和匹配的基础。

&#8195;&#8195; **特征描述符 (Feature Descriptors)**：特征描述符是用于描述关键点周围的局部图像信息的向量。每个描述符都是一个高维向量，能够唯一地表示某个关键点。通过对特征描述符的比较，我们可以在不同图像中匹配相似的特征点。


| 常见的特征检测算法 | **Harris 角点检测** | **Shi-Tomasi 角点检测** | **SIFT** | **SURF** | **ORB** |
| --- | --- | --- | --- | --- | --- |
| **简介** | 一种基于角点响应函数的经典算法，可以快速检测图像中的角点 |  Harris的改进算法|一种尺度不变特征变换算法，能够在不同尺度和旋转下检测关键点，并生成特征描述符。 | SIFT 的改进版，提高了速度  | 一种基于快速角点检测算法 (FAST) 和旋转不变的 BRIEF 描述符的组合方法，适合实时计算 |
| **时间** | 1988 | 1994 | 1999 | 2006 | 2011 |
| **特征类型** | 角点 | 角点 | 角点+特征描述符 | 角点+特征描述符 | 角点+特征描述符 |
| **优缺点** | 对噪声和旋转有一定鲁棒性，但对尺度变化不敏感| 对噪声和旋转有一定鲁棒性，但对尺度变化不敏感 |对光照、尺度、旋转有良好鲁棒性，准确率很高 | 对光照、尺度、旋转有良好鲁棒性，准确率略低于SIFT| 且对光照、旋转和噪声有一定鲁棒性.速度快，但准确率较低 |
| **描述符维度** | 无（只检测角点） | 无（只检测角点） | 128 维 | 64/128 维 | 32 维 |
| **速度** | 快 | 快 | 慢 | 较快 | 快 |
| **应用场景** | 基础角点检测，快速响应 | 精确角点检测，图像跟踪 | 图像匹配、物体识别 | 图像匹配、物体识别、实时处理 | 实时应用、目标跟踪、SLAM |


### 2.2 Harris 角点检测 (Harris Corner Detection)
>Harris 角点检测的原理，可参考[《数字图像处理【14】特征检测——Harris角点检测》](https://blog.csdn.net/a360940265a/article/details/140158249)。
#### 2.2.1 泰勒展开
&#8195;&#8195;先复习一下泰勒公式。泰勒公式是一种数学中常用的方法，它将一个在某点附近可微的函数用多项式来近似表示。具体来说，如果函数 $f(x)$ 在点 $a$ 附近可导，泰勒展开可以写作：

$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \ldots+R_n(x)$$

&#8195;&#8195;当a很小时，$R_n(x)$是一个无穷小的量，可以忽略。这个级数可以无限地展开，直到无穷项。通常，我们可以将其表示为：

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n$$

在 $a = 0$ 的情况下，称为麦克劳林展开：
$$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \ldots$$

&#8195;&#8195;这种形式便于直接计算函数在原点附近的近似值，常用于分析和逼近函数行为。而一般用的最多的是一阶泰勒展开，即只考虑到一阶导数：

$$f(x) =f(a) + f'(a)(x - a)+o(x-x_0)\approx f(a) + f'(a)(x - a)$$
>对于$y=f(x)$来说，当$a$很小时，$f^{'}(a)=\frac{\Delta y}{\Delta x}$，$f^{'}(a)(x-a)=\frac{\Delta y}{\Delta x} *\Delta x=\Delta y$。

&#8195;&#8195;泰勒展开在数值分析、物理和工程等领域中非常有用，可以用来近似复杂函数。例如对于函数$f(x) = e^x$在 $x = 0$ 附近的一阶泰勒展开：
* 函数： $f(0) = e^0 = 1$
 * 导数：$f'(x) = e^x$，因此 $f'(0) = e^0 = 1$
 * **一阶泰勒展开**：    
     $$f(x) \approx f(0) + f'(0)(x - 0) = 1 + 1 \cdot x = 1 + x$$
    
&#8195;&#8195;这表示在 $x = 0$ 附近，指数函数 $e^x$ 可以被近似为 $1 + x$。假设我们想要估计 $e^{0.1}$：

* 实际值：$e^{0.1} \approx 1.10517$
* 近似值：使用一阶泰勒展开得 $1 + 0.1 = 1.1$

&#8195;&#8195;可见一阶泰勒展开提供了一个非常简洁的估计。这个方法在科学和工程中非常实用，尤其是在需要快速计算时。


#### 2.2.2 Harris 角点检测的原理
&#8195;&#8195;在众多的检测算法里最经典的角点特征检测就是Harris角点检测，由Chris Harris和Mike Stephens于1988年提出。角点是图像中两条边缘的交点，将整个图形角点的检测分成三种情况，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19b9b9f4ec6745c196fa1162f7f42ddc.png)<center>Harris角点检测原理</center>

1. **平坦区**：检测窗口朝任意方向进行移动，窗口内的像素没有任何差异的变化
2. **边缘**：检测窗口沿着边缘的方向进行移动，中心像素是没有明显的差异变化；但如果检测窗口垂直边缘进行移动，中心像素会剧烈的变化。这可以简单的确定为边缘特征。
3. **角点**：检测窗口无论朝哪个方向移动的时候都会产生剧烈的变化，这个时候就可以简单的确定为角点特征了。


简单的物理原理是这样的，如果用数学表示图像$I(x,y)$在移动$( \Delta x, \Delta y )$后的自相似性就是：
$$c(x,y;\Delta x, \Delta y) = \sum_{(u,v)\in W(x,y)} w(u, v) [I(u+\Delta x, v+\Delta y) - I(u, v)]^2$$
其中：
* 计算结果表示统计窗口内整体像素值变化，是一个标量（一般是灰度值变化差）
* $I(u, v)$ 是图像在窗口 $(u, v)$ 处的灰度值，平方是为了使变化度量值非负（不使用绝对值是因为绝对值需要进行判断，不好消掉）。
 * $\Delta x, \Delta y$ 是窗口的小位移。
* $w(u, v)$ 是窗口的加权函数（窗口相当于一个卷积核，里面每一个像素都有对应的权重，通常为高斯函数）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d99c8c9f464241a38dedd68c2f326964.png)<center> 左：窗口内像素权重都是1；右：窗口内像素权重为高斯分布</center>



为了简化上述灰度变化公式，通常对 $I(x + \Delta x, y + \Delta y)$ 进行一阶泰勒展开，保留一阶项：
 $$I(u+\Delta x,v+\Delta y)=I(u,v)+I_{x}(u,v)\Delta x+I_{y}(u,v)\Delta y+O(\Delta x^{2},\Delta y^{2})\approx I(u,v)+I_{x}(u,v)\Delta x+I_{y}(u,v)\Delta y$$
其中，$I_x,I_y$是图像$I(x,y)$的偏导数 $\left( \frac{\partial I}{\partial x} \Delta x , \frac{\partial I}{\partial y} \Delta y \right)$, 对于图像来说就是水平方向和竖直方向的梯度了。将该近似代入灰度变化公式中，得到：

$$c(x,y;\Delta x, \Delta y) \approx  \sum_{(u,v)\in W(x,y)} w(u, v)(I_{x}(u,v)\Delta x+I_{y}(u,v)\Delta y)^{2}=\begin{bmatrix}
\Delta x,\Delta y \\
\end{bmatrix}M(x,y)\begin{bmatrix}
\Delta x \\\Delta y
\end{bmatrix}$$

其中：
$$M(x,y)=\sum_{w}\begin{bmatrix}
I_{x}(x,y)^{2} & I_{x}(x,y)I_{y}(x,y)\\
I_{x}(x,y)I_{y}(x,y) &I_{y}(x,y)^{2}  \\
\end{bmatrix}=\begin{bmatrix}
\sum_{w}I_{x}^{2} &  \sum_{w}I_{x}I_{y}\\
\sum_{w}I_{x}I_{y} &\sum_{w}I_{y}^{2}  \\
\end{bmatrix}=\begin{bmatrix}
A & C \\
C& B \\
\end{bmatrix}$$

化简可得：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/229a204096ae4cb188251b9435181915.png#pic_center =400x)

$\begin{bmatrix}
A & C \\
C& B \\
\end{bmatrix}$是一个实对称矩阵（元素都是实数，且矩阵的转置等于其本身），它可以对角化为以下形式（对角阵上的元素即为矩阵本身特征值，需要具体计算）：
$$\begin{bmatrix}
A & C \\
C& B \\
\end{bmatrix}=\begin{bmatrix}
 \lambda _{1} & 0 \\
0&  \lambda _{2} \\
\end{bmatrix}$$
代入有：
$$c(x,y;\Delta x, \Delta y)=\lambda _{1}\Delta x^{2}+\lambda _{2}\Delta y^{2}$$

椭圆方程标准方程为：
$$\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}=1$$

所以二次项函数本质是一个椭圆函数，$a,b$分别是椭圆的长轴和短轴。
$$\lambda _{1}=\frac{1}{a^{2}}，a=\frac{1}{\sqrt{\lambda _{1}}}=\lambda _{1}^{-\frac{1}{2}}$$

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/404efe3ad1284d93a2f4dae57e80f798.png#pic_center =450x)





&#8195;&#8195;矩阵 $M$ 称为自相关矩阵或结构张量，反映了窗口内灰度的梯度变化情况。自相关矩阵 $M$ 的两个特征值 $\lambda_1$ 和 $\lambda_2$ 可以用来描述局部窗口内的变化：

* 如果 $\lambda_1$ 和 $\lambda_2$ 都很大，且相差不大，则说明该窗口在两个方向上都有显著变化，即为**角点**。
* 如果 $\lambda_1\geqslant \lambda_2$ 或反过来 $\lambda_2\geqslant \lambda_1$ ，则说明该窗口主要在一个方向上有显著变化，这对应**边缘**。
* 如果 $\lambda_1$ 和 $\lambda_2$ 都很小且近似相等，则窗口几乎没有变化，这对应**平坦区域**。
#### 2.2.3 角点响应
&#8195;&#8195;按上面步骤可以计算出$\lambda_1\lambda_2$，但是我们没必要分别进行以上三种情况的判断。为了有效地描述符合这三种情况的特征，Harris 提出了一个**响应函数** $R$，它可以通过矩阵的行列式（det）和迹（trace）来近似计算：

$$R = \text{det}(M) - k \cdot (\text{trace}(M))^2$$

其中：
* $\text{det}(M) = \lambda_1 \lambda_2$ ，是矩阵的行列式，反映了该区域的强度变化。
* $\text{trace}(M) = \lambda_1 + \lambda_2$， 是矩阵的迹，反映了强度变化的整体程度。
* $k$ 是一个经验常数，通常取值在 $0.04 \sim 0.06$ 之间。

根据 $R$ 的值，可以判断出该区域是角点、边缘还是平坦区域：

* 当 $R>0$ 且其绝对值很大时，该点是角点。
* 当 $R<0$ 且绝对值很大时时，该点位于边缘。
* 当 $R$ 值接近0时，该区域是平坦区域。

#### 2.2.4 代码演示

```python
cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) -> dst
```
-  `blockSize`：检测窗口尺寸（不是卷积核，可以是偶数）
- `ksize`：sobel算子卷积核尺寸（sobel算子用于计算图像表示的离散数值的梯度）
- `k`：上面公式中的 $\alpha$ ，是个经验值, 一般取0.04~0.06之间，默认0.04。

```python
import cv2
import numpy as np

img = cv2.imread('./chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 角点检测,返回图像中每个像素对应的角点响应R
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 设定阈值，结果大于0.01倍的dst.max()视为角点，画成红色来显示
img[dst > (0.01 * dst.max())] = [0, 0, 255]

cv2.imshow('Harris', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2.3  Shi-Tomasi 角点检测
&#8195;&#8195;Shi-Tomasi是对Harris角点检测的改进。Harris角点检测计算的稳定性和K有关（上面的α值）, 而K是一个经验值, 不太好设定最佳的K值。

&#8195;&#8195;Shi-Tomasi 发现，角点的稳定性其实和矩阵 M 的较小特征值有关，于是直接用较小的那个特征值作为分数，这样就不用调整k值了，即角点响应为：
$$R=min( \lambda_1 , \lambda_2)$$

和 Harris 一样，如果该分数大于设定的阈值，我们就认为它是一个角点。

```python
goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
```
* **image**: 输入的灰度图像，需要从彩色图像转换为灰度图像。
* **maxCorners**: 最多检测的特征点数量。如果图像中的角点数量超过该值，则返回的角点数量会少于该值。
* **qualityLevel**: 质量水平参数，用于确定角点的质量。值在 0 到 1 之间，表示最好的角点相对于最差角点的比例，通常取值为 0.01 到 0.1。
* **minDistance**: 允许检测到的角点之间的最小距离。这个参数可以用来避免在相邻区域检测到过多特征点。
* **corners**: 可选参数，用于存储检测到的特征点。如果为 None，函数将创建一个新的数组。
* **mask**: 可选参数，用于指定一个区域，只在该区域内进行角点检测。该区域为二进制掩码，非零像素表示检测区域。
* **blockSize**: 计算导数时使用的邻域大小。较大的块大小可以检测到更大的特征。
* **useHarrisDetector**: 布尔值，指示是否使用 Harris 角点检测器。如果为 True，则使用 Harris 检测器；否则，使用 Shi-Tomasi 检测器。
* **k**: Harris 检测器中的参数，用于控制响应函数的计算。

函数最终返回一个$N×1×2$ 的数组，表示检测到的特征点的坐标 (x, y)。
```python
import cv2
import numpy as np

img = cv2.imread('chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# shi-tomasi焦点检测
corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.1, minDistance=10)

print(corners.shape)  			# 返回的结果是检测出的角点的坐标，浮点类型
corners = np.int0(corners)

# 画出角点
for i in corners:
    # i相当于corners中的每一行数据，ravel()把二维变一维了.即角点的坐标点
    x,y = i.ravel()
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
cv2.imshow('Shi-Tomasi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```python
(191, 1, 2)			# 检测出191个角点，每行一个列表，列表中是角点坐标
```

### 2.4 SIFT (Scale-Invariant Feature Transform)
>检测原理可参考[《数字图像处理【15】特征检测——SIFT特征检测》](https://blog.csdn.net/a360940265a/article/details/140673760)

`Harris`角点具有旋转不变的特性，但是缩放后，原来的角点有可能就不是角点了：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60435fedc95c4c0ab50b8795378b61bc.png#pic_center =500x)

&#8195;&#8195;`SIFT`，即尺度不变特征变换（Scale-invariant feature transform），是用于图像处理领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。不仅如此，当图像旋转，改变图像亮度，移动拍摄位置时，`SIFT`仍可得到较好的检测效果；对视角变化、仿射变换、噪声也保持一定程度的稳定性。其缺点也很明显：
- 实时性不高，因为要对输入图像进行多个尺度的下采样和插值等操作；
- 对边缘光滑的目标无法准确提取特征（比如边缘平滑的图像，检测出的特征点过少，对圆更是无能为力）
#### 2.4.1 算法详解
1.  **图像尺度空间**
	- 在一定的范围内，无论物体是大还是小，人眼都可以分辨出来，然而计算机要有相同的能力却很难，所以要让机器能够对物体在不同尺度(距离)下有一个统一的认知，就需要考虑图像在不同的尺度下都存在的特点。
	- 尺度空间的获取通常使用高斯模糊来实现，高斯滤波器（卷积核）可以平滑图像，从而消除图像中的细微细节。随着`σ`值不断增大，图像也越来越模糊，直到只剩下最基本的特征。`SIFT`可以做到对不管是清晰还是模糊的图片都能识别出特征点，且是同一个位置。
![sift_3.png](https://img-blog.csdnimg.cn/img_convert/0280b7a92ac447e079a7cd0a1f3ab999.png#pic_center =600x)![sift_2.png](https://img-blog.csdnimg.cn/img_convert/ee0a66046c1baa9c652d59cab0ef2ad2.png#pic_center =600x)
>图像的尺度空间是指图像经过几个不同高斯核后形成的模糊图片的集合，这些图片模拟了人眼在不同距离观察物体时的视觉效果。尺度不是指图像分辨率尺寸，是模拟人眼远近观察的空间距离。

2.  图像金字塔
	- 图像金字塔是尺度空间的一种具体实现方式，是图像处理中的一种常用技术。它通过对原始图像进行一系列的降采样操作来创建一组不同分辨率的图像。在OpenCV中，图像金字塔有两种类型：高斯金字塔和拉普拉斯金字塔，详见[《OpenCV系列教程三：形态学、图像轮廓、直方图》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5501)第三章。
	- 高斯金字塔 (`Gaussian Pyramid`)是通过连续应用高斯模糊和降采样来构建的。每一层的图像都是上一层的图像$G_i$经过高斯模糊后，删除其偶数行和列得到的。每次下采样之后，图像尺寸都减半，多次处理就得到整个高斯金字塔。
	- 在构建高斯金字塔时，通常使用的是5x5的高斯卷积核来进行高斯模糊。这个卷积核的权重是根据高斯分布（正态分布）计算得出的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f164c0dc11ff4e3e970b9fa83f112cd0.png)

3. **SIFT金字塔**
`SIFT`金字塔，即尺度不变特征变换金字塔，是SIFT算法中的一个重要组成部分。它主要用于在不同尺度空间上查找图像中的关键点（特征点），并计算出这些关键点的方向。SIFT金字塔的构建其实就是高斯金字塔和差分金字塔（Difference of Gaussian，DoG）的融合，其构建过程如下：
	- **初始化**：将原始图像作为高斯金字塔的第一层（Octave）的第一张（Interval）。
	- **高斯模糊**：对原始图像应用具有不同 `σ` 值的高斯核进行卷积操作，生成多张逐渐模糊的图像。一般设初始`σ=1.6`，之后每次将`σ`乘以一个比例系数`k`（如`k=√2`），重复此过程（通常执行4~5次操作）。
	- **图像降采样**：在每一层高斯模糊之后，图像可以进行降采样（删除偶数行和列，尺寸缩小一半），以生成下一层的图像。重复上述步骤，我们就得到多层不同尺寸的图像金字塔，每一层都有5张不同清晰度的图像。
![sift_4.png](https://img-blog.csdnimg.cn/img_convert/40c5684ebbf85114cfb6cc0a931d12b7.png#pic_center =600x)
	- **构建差分高斯金字塔**：对于高斯金字塔的每一层，将相邻两张图像进行相减操作，计算它们之间的差值，这个差值图像称为差分高斯（DoG）图像。最终每层（Octave）有$N$张高斯图像，有$N-1$张DoG图像。DOG定义公式如下，其中 $k$ 是与尺度空间的层级数有关的比例系数，$G$表示使用不同$\sigma$的高斯模糊函数，$I(x,y)$表示图像（像素值）。
$$D(x,y,\sigma )=[G(x,y,k\sigma)-G(x,y,\sigma)]*I(x，y)=L(x,y,k\sigma)-L(x,y,\sigma)$$
	- 
![sift_5.png](https://img-blog.csdnimg.cn/img_convert/cfa5147c3f2bfc8cd2eca9f647249c68.png#pic_center =600x)

4. **DoG空间极值检测**
	- 在差分高斯金字塔中，通过寻找局部极值点来检测关键点。每个像素点要和其图像域（同一尺度空间）和尺度域（相邻的尺度空间）的所有相邻点进行比较，当其大于（或者小于）所有相邻点时，该点就是极值点。
	- 如下图所示，中间的检测点要和其所在图像的3×3邻域8个像素点，以及其相邻的上下两层的3×3领域18个像素点，共26个像素点进行比较。如果一个像素在所有比较中都是局部极值，那么它就是一个**关键点候选**。


![sift_7.png](https://img-blog.csdnimg.cn/img_convert/ddc12678e1b3fe3f428b399d48a9cc57.png#pic_center =500x)

5. **关键点的精确定位**
	- 这些候选关键点是DOG空间的局部极值点，是通过离散的高斯模糊图像相减得到的，因此存在一定的误差。为了更精确地确定关键点的位置和尺度，可以使用泰勒多项式来拟合DoG函数，来获得极值点的准确位置。
	- D函数有三个未知数 $\Delta x,\Delta y,\Delta \sigma$ 。分别对其求一阶导和二阶导，写成矩阵的形式，得到其二阶泰勒展开式$D(x)$。求导并让令展开式=0，可求出位置点的$\Delta x$。
![sift_8.png](https://img-blog.csdnimg.cn/img_convert/0fdf1eb88229737de4458cee5c580cae.png#pic_center =800x)
![sift_9.png](https://img-blog.csdnimg.cn/img_convert/8e491cc0e45b3997cd526f05b0aed466.png#pic_center =800x)
- 将$\Delta x$代入公式中，即可得到其拟合的极值点位置的(灰度)值表达式：
$$D(\Delta x)=D+\frac{1}{2}\frac{\partial D}{\partial x}\Delta x$$

6. **消除边界响应**：DoG 算法对边界非常敏感，所以我们必须要把边界去除。前面讲Harris 算法除了可以用于角点检测之外其实还可以用于检测边界的。作者就是使用了同样的思路。作者使用 2x2 的 Hessian 矩阵计算主曲率。从 Harris 角点检测的算法中，我们知道当一个特征值远远大于另外一个特征值时检测到的是边界。Sift算法论文中建议边界阈值为 R=10。R大于此阈值就会被剔除。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2c1fdb62776841a0acb145a2ca1a9564.png#pic_center =800x)

7. **特征点的主方向**：图像的特征点不仅具有尺度不变性，还需要具备旋转不变性。通过为每个关键点指定一个主方向，SIFT算法能够确保即使图像发生旋转，关键点的描述子也能够保持一致，从而在图像匹配时能够正确对应。
	- **确定极值点的领域范围**：对每个特征点，以特征点为中心，根据其尺度在图像上定义一个固定大小的区域。这个区域通常是特征点尺度的大小的若干倍（通常是 $4 \times 4$ 个像素的区域），用来计算该区域内的图像梯度信息。
	- **计算该邻域内每个像素点的梯度幅值和梯度方向**：每个特征点需要包含三个信息$(x,y,σ,θ)$，即位置、尺度和方向。邻域内每个点$L(x,y)$的梯度的模$m(x,y)$以及方向 $\theta(x,y)$为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2901753a7f2c4e38892be3c75ca400b0.png#pic_center =600x)
其中，$L(x,y)$是图像在点$(x,y)$处的灰度值。$m(x,y)$公式右侧根号中的两项，分别表示像素点x轴和y轴方向的梯度。图像中都是离散的像素点，其梯度通过中心差分来求解，详见[《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)2.3章节sobel算子。
	- **构建方向梯度直方图**： 将梯度方向的范围（0°~360°）划分为多个方向区间（bin），常用的划分方式有每10°一个bin（共36个bin）或每45°一个bin（共8个bin）。然后对于邻域内的每个像素点，进行梯度方向的直方图统计，将每个点的梯度幅值累加到对应的bin中。
	- **加权处理**：在计算梯度幅值时，引入高斯加权函数，给予靠近特征点中心的像素较大的权重，远离特征点中心的像素较小的权重。这样做是为了减少噪声的影响，并增强对特征点局部结构的捕捉。
	- **确定特征点的主方向**：最终构建完成的直方图中，峰值代表了邻域内图像梯度的主方向，将其作为关键点的主方向。 如果梯度直方图中存在另一个峰值，其能量达到或超过主峰值的80%，则将该方向作为关键点的辅方向，这样可以增强匹配的鲁棒性。具有多个方向的关键点可以被复制成多份，然后将方向值分别赋给复制后的特征点，一个特征点就产生了多个坐标、尺度相等，但是方向不同的特征点。
![](https://img-blog.csdnimg.cn/img_convert/80ace6859c55aa98b7645d2518ada02a.png#pic_center =600x)

8. 生成特征描述：通过以上步骤，我们检测出的含有位置、尺度和主方向的SIFT关键点$(x,y,σ,θ)$。
	- **将坐标系旋转到关键点主方向**：以特征点为中心，在附近邻域内将坐标轴旋转θ角度，将坐标轴旋转对齐到关键点主方向（下图红色箭头指示的方向）。这样，所有梯度方向都以特征点的主方向为参考进行旋转调整，从而实现旋转不变性。
![sift_14.png](https://img-blog.csdnimg.cn/img_convert/73afd3a7f0f12c4ed211c46d38365d53.png#pic_center =700x)
	- **使用种子点进行特征描述**：以旋转之后的主方向为中心取8x8的窗口，求每个像素点的梯度幅值和方向。箭头方向代表梯度方向，长度代表梯度幅值，然后利用高斯窗口对其进行加权运算，最后在每个4x4的小块上绘制8个方向的梯度直方图，计算每个梯度方向的累加值，即可形成一个种子点。即每个特征描述由4个种子点组成，每个种子点有8个方向的向量信息。（也就是将360度的角度分成8个45度）
![sift_15.png](https://img-blog.csdnimg.cn/img_convert/3a5c4af67106dec209018da795d4cd83.png#pic_center =700x)
	- 论文中建议对每个关键点使用4x4共16个种子点来描述，这样一个关键点就会产生128维的SIFT特征向量（4x4x8）。它表示了特征点附近的局部图像信息，这个向量就是最终的特征描述符。
 ![sift_17.png](https://img-blog.csdnimg.cn/img_convert/d33400de18b464ebbfb4bdf1131a3fd2.png#pic_center =700x)
 9.  **归一化与截断**
为了消除光照变化的影响，特征向量需要进行归一化处理，即将特征向量的所有值除以该向量的欧几里得范数，使得整个向量的长度为1：
$$\text{descriptor} = \frac{\text{descriptor}}{\|\text{descriptor}\|}$$
这种归一化的目的是为了使得特征描述符对光照变化具有鲁棒性。在实际应用中，为了进一步减少光照的影响，通常会将向量中的每个值进行截断，防止某些值过大而造成对少数维度的过度依赖。通常的做法是将特征向量中的值限制在一个最大阈值（如 0.2）以内。
	* 截断操作的具体步骤是：如果特征向量中的某个值超过 0.2，则将其设定为 0.2。
	* 截断后再次归一化，以确保描述符的总能量为 1。

#### 2.4.2 总结
&#8195;&#8195;SIFT算法的核心原理可分为四个步骤：特征点检测、特征点定位、主方向确定和特征描述符生成。

1. **特征点检测**
首先，SIFT算法通过构建图像的多尺度金字塔来实现尺度不变性。使用高斯模糊函数对图像进行不同尺度的模糊处理，然后通过差分高斯（DoG, Difference of Gaussian）对模糊后的图像进行处理，获取图像的不同尺度空间。这些空间中的极值点（即在图像空间中比周围像素更亮或更暗的点）被认为是潜在的特征点。

2. **特征点精确定位**
在检测到的极值点中，SIFT对每个点进行精确定位。通过二次插值来提升特征点的定位精度，滤除对比度过低或位于边缘的特征点，这样可以去除不稳定或噪声干扰较大的点。最终剩下的特征点具有较好的定位精度和稳定性。

 3. **确定特征点主方向**
为了使特征点具有旋转不变性，SIFT根据每个特征点邻域内的梯度方向信息确定主方向。具体来说，首先计算特征点邻域内每个像素的梯度幅值和方向，然后构建一个方向直方图，选择幅值最高的方向作为该特征点的主方向。在某些情况下，可能会有多个主方向，每个方向都会生成一个独立的特征描述符。

4. **生成特征描述符**
基于确定的特征点位置、尺度和主方向，SIFT算法在特征点的邻域内划分为多个 $4 \times 4$ 的子区域。对于每个子区域，计算其内像素的梯度方向并构建8个方向的梯度直方图，从而形成128维的特征向量（16个子区域，每个子区域8个方向，$16 \times 8 = 128$）。为了增强描述符的鲁棒性，梯度幅值经过高斯加权处理，邻域内的梯度方向也会进行双线性插值。最后，对128维的特征向量进行归一化与截断，以消除光照和对比度变化的影响。

5. **匹配与应用**
&#8195;&#8195;经过上述步骤，最终得到的128维向量就是该特征点的特征描述符。由于在特征描述符构建过程中考虑了尺度、旋转和光照不变性，它可以在不同的图像中被用来进行特征点匹配。匹配时，通过计算不同图像中特征描述符之间的欧几里得距离，找出相似的特征点对。另外在物体识别、三维重建等任务中具有高鲁棒性和广泛应用。
#### 2.4.3 代码示例

```python
import cv2
import numpy as np

img = cv2.imread('chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建sift对象，使用默认参数就行
sift = cv2.SIFT_create()		
# 进行检测,kp是一个列表，里面是cv2.KeyPoint对象，可通过cv2.drawKeypoints绘出。			
kp = sift.detect(gray)							

# 绘制关键点
cv2.drawKeypoints(gray, kp, img)
cv2.imshow('SIFT', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- `sift.detect()` 函数主要有image和mask两个参数，表示输入图像和掩码图像。函数返回检测到的关键点列表。每个关键点是一个 `cv2.KeyPoint` 对象，包含位置（`pt`）、尺度（`size`）、方向（`angle`）等信息。
- drawKeypoints函数语法为：`drawKeypoints(image, keypoints, outImage[, color[, flags]]) -> outImage`
	 * **`image`**：原始图像，在该图像上绘制关键点。通常是灰度图（输入给 SIFT 检测器的图像）。
	    
	* **`keypoints`**：关键点列表，这是由 `sift.detect()` 返回的关键点结果。
	    
	* **`output_image`**：输出图像。如果传入 `None`，函数将会返回一张新图像，并在上面绘制关键点。你也可以传入一个空的同尺寸图像，以便保存绘制的结果。
	    
	* **`color`**（可选）：绘制关键点的颜色，默认情况下绘制颜色为随机颜色。如果指定颜色，则可以使用 (B, G, R) 颜色格式。
    
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e973b08a0e34f00a18101d8ea727eaa.png#pic_center =400x)

&#8195;&#8195;使用`sift.compute`函数，可以计算图像的关键点描述子（描述符），也就是那个 记录了关键点周围对其有共享的像素点的128维向量值, 其不受仿射变换, 光照变换等影响。描述子的作用就是进行特征匹配, 在后面进行特征匹配的时候会用上。
```python
# 计算描述子des
kp, des = sift.compute(gray, kp)	
print(des.shape)			
```

```python
(768, 128)			# 766个关键点，每个关键点描述子是一个128维的向量。
```

你也可以一步到位，直接计算出关键点和关键点描述子：

```python
kp, des = sift.detectAndCompute(gray, mask=None)
```

### 2.5 SURF (Speeded-Up Robust Features)
>原理见[《SURF算法》](https://blog.csdn.net/qq_30815237/article/details/86545950)

&#8195;&#8195;如果想对一系列的图片进行快速的特征检测, 使用SIFT会非常慢（主要是求关键点的精确定位时二阶泰勒展开的求导步骤，以及后面计算128维向量时）。Speeded Up Robust Features（SURF，加速稳健特征），是2006年提出的一种稳健的局部特征点检测和描述算法，是对Sift算法的改进，提升了算法的执行效率，为算法在实时计算机视觉系统中应用提供了可能。

1. **使用方盒滤波器（Box Filter）进行近似卷积**
	
	* **SIFT**：SIFT在图像的不同尺度空间中进行高斯卷积，逐层计算高斯差分，卷积核是标准的高斯核，这需要较大的计算量。
	* **SURF**：SURF引入了盒式滤波器（Box Filter）来近似高斯卷积。盒式滤波器可以利用积分图像进行快速计算（直接查找积分表得到结果）。虽然这种方法是高斯卷积的近似，但在实践中效果仍然非常好，而计算效率显著提高。

2. **特征点方向的计算方式**：SURF在特征点邻域内计算水平方向和竖直方向的Haar小波响应，并使用加权的局部加总来确定主方向。这种方法比SIFT中的梯度计算更快。
3. **简化特征描述符的构建**：SURF的特征描述符简化为基于Haar小波的响应，最终生成一个64维的描述符。SURF也有扩展的128维版本，但通常使用64维版本，因为其计算速度快且描述能力强。

```python
import cv2
import numpy as np

img = cv2.imread('./chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建SURF对象
surf = cv2.xfeatures2d.SURF_create()
# 检测关键点, 并计算描述子。URF算法的特征描述子是一个64维的向量, 比SIFT少了一半
kp, des = surf.detectAndCompute(img, None)

# 绘制关键点
cv2.drawKeypoints(gray, kp, img)
cv2.imshow('SURF', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b080640490024ba0bbaf0291dfac6041.png#pic_center =400x)

### 2.6 ORB (Oriented FAST and Rotated BRIEF)
>[《ORBSLAM2学习（一）：ORB算法原理》](https://blog.csdn.net/lwx309025167/article/details/80365075)、[《图像特征算法(三)——ORB算法简述》](https://blog.csdn.net/qq_43616471/article/details/107855268)

&#8195;&#8195;ORB（Oriented FAST and Rotated BRIEF）是一个用于图像特征检测和描述的算法，它结合了**FAST**（Features from Accelerated Segment Test）特征检测器和**BRIEF**（Binary Robust Independent Elementary Features）特征描述子，并对它们进行了改进和优化。

&#8195;&#8195;ORB算法最大的特点就是计算速度快。这首先得益于使用FAST检测特征点，FAST的检测速度正如它的名字一样是出了名的快。再次是使用BRIEF算法计算描述子，该描述子特有的2进制串的表现形式不仅节约了存储空间，而且大大缩短了匹配的时间。 <font color='deeppink'>ORB最大的优势就是可以做到实时检测，常用于诸如**图像拼接**、**目标跟踪**、**视觉SLAM**等需要高效、实时处理的场景。</font >其算法原理为：

1. **特征点检测（FAST改进版）**：
   - ORB使用FAST算法（一种快速的角点检测方法）进行特征点检测，还对其进行了改进。通过使用图像灰度质心来计算特征点的方向，让ORB对图像的旋转更具鲁棒性。
   - ORB算法进一步引入了多尺度金字塔，从而能够在不同尺度上检测特征点，以提高对图像缩放和旋转的鲁棒性。


2. **特征点描述（BRIEF改进版）**：BRIEF是一种二值描述子，通过随机采样图像中的像素对并比较它们的亮度差异，生成一串二进制向量来描述特征点。 为了使描述子能够应对旋转变化，ORB对BRIEF描述子进行了改进，根据特征点的方向，对BRIEF描述子进行相应的旋转调整，增加了旋转不变性。

3. **Harris角点评分**：结合了Harris角点响应的得分，用于排序和筛选质量更高的特征点。这一做法可以有效提升特征点的稳健性。

ORB算法的优势：
- **速度快**：ORB是为了提高特征检测和描述的效率而设计的，特别是在实时应用中具有很好的表现，适合资源受限的场景。
- **旋转不变性**：通过计算每个特征点的方向，ORB可以处理图像的旋转变化。
- **尺度不变性**：ORB使用图像金字塔进行多尺度检测，使得它对尺度变化具有一定的鲁棒性。
- **二进制描述子**：ORB使用的BRIEF描述子是二进制的，存储和计算的代价较低，非常适合计算机视觉中的实时任务。

```python
import cv2
import numpy as np

img = cv2.imread('chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建ORB对象
orb = cv2.ORB_create()
# 一步到位, 检测出把关键点和描述子（32维向量）.
kp, des = orb.detectAndCompute(img, None)

# 绘制关键点
cv2.drawKeypoints(gray, kp, img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/135e09b540f84145bb479a6138025d6d.png)
可以看到，ORB的准确性差了很多，同样的数字7.8.9等，被检测出的效果不一样。
## 三、特征匹配 (Feature Matching)

&#8195;&#8195;一旦提取了图像中的关键点和特征描述符，我们就可以在不同图像中进行匹配。常见的匹配方法包括：
   - `Brute-Force Matcher`：**通过穷举的方式**，将一幅图像的每个特征与另一个图像的所有特征进行比较，找到最相似的特征对。
   - `FLANN Matcher`：基于**近似最近邻搜索**的快速匹配方法，适合大数据集的特征匹配。

**应用场景：**

- **图像拼接**：通过匹配两张或多张图像中的特征点，可以实现图像拼接。
- **物体识别**：检测和匹配物体特征，从而在复杂的场景中识别物体。
- **运动跟踪**：通过跟踪图像中的特征点，可以实现目标的实时运动跟踪。
### 3.1 暴力特征匹配
#### 3.1.1 原理

&#8195;&#8195;`Brute-Force Matcher`的基本思想是：给定两个图像的特征点描述子（descriptors），逐个比较每个特征点在两个图像中的描述子，寻找相似性最高的一对特征点。

- **描述子（Descriptors）**：描述子是对图像中局部区域特征的高维向量表示，不同算法生成的描述子维度和特征有所不同。例如，SIFT生成128维的浮点描述子，而ORB生成32维的二进制描述子。
- **距离度量**：BFMatcher通过计算两个描述子之间的距离来判断它们的相似程度，通常使用欧氏距离（针对浮点数描述子，如SIFT、SURF）或者汉明距离（针对二进制描述子，如ORB）。距离越小，描述子越相似。


BFMatcher主要分为以下步骤：

1. 特征检测和描述子计算
首先，使用特征检测算法（如SIFT、ORB等）在两幅图像中检测关键点（Keypoints），并为每个关键点生成对应的描述子。描述子是对关键点周围区域的特征的高维向量表示。

```python
# SIFT例子
sift = cv2.SIFT_create()

# 检测关键点(cv2.KeyPoint类型)和计算描述子（32维向量）
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
```

2. 匹配描述子
BFMatcher会将第一幅图像的每个描述子与第二幅图像中的每个描述子逐一进行比较。对于每个描述子，它会找到距离最近的匹配。

```python
# 使用BFMatcher进行匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(descriptors1, descriptors2，mask=None)
```

- `NORM_L2`：默认值，欧氏距离，常用于SIFT和SURF等浮点描述子的（也可以使用NORM_L1距离）。
- `crossCheck=True`：启用交叉验证模式，即只有当图像A中的关键点与图像B中的关键点互相匹配时才保留这对匹配点。

&#8195;&#8195;返回的`matches` 是一个 DMatch 对象的列表，每个元素是一个特征匹配对象（一般用`m`表示）。它存储了一个特征点的匹配信息，该对象具有以下属性:
- `m.distance` - 描述符之间的距离，越低越好。
 - `m.queryIdx` - 匹配结果中，在第一幅图像中的特征点的索引。你可以通过 `keypoints1[m.queryIdx]` 访问第一幅图像中相应的特征点对象。
- `m.trainIdx` – 匹配结果中，在第二幅图像中的特征点的索引。你可以通过 `keypoints2[m.trainIdx]` 访问第二幅图像中相应的特征点对象。
- `m.imgIdx` – 第一幅图像的索引

```python
for m in matches:
    # 通过 m.queryIdx 从 keypoints1 中找到第一个图像的匹配点
    pt1 = keypoints1[m.queryIdx].pt

    # 通过 m.trainIdx 从 keypoints2 中找到第二个图像的匹配点
    pt2 = keypoints2[m.trainIdx].pt

    print(f"第一幅图像中的匹配点坐标: {pt1}, 第二幅图像中的匹配点坐标: {pt2}")
```

```python
第一幅图像中的匹配点坐标: (96.88207244873047, 258.9445495605469), 第二幅图像中的匹配点坐标: (190.9088134765625, 343.1674499511719)
第一幅图像中的匹配点坐标: (101.16863250732422, 197.38674926757812), 第二幅图像中的匹配点坐标: (195.0311737060547, 281.328369140625)
...
```

3. 排序匹配结果（可选）
BFMatcher会输出若干对匹配点，可以根据距离对匹配点进行排序，保留距离最小的前几对匹配点。

```python
# 按照距离对匹配点排序
matches = sorted(matches, key=lambda x: x.distance)
```

4. 筛选匹配点（可选）
有时候最近距离的描述符不一定最匹配。对于复杂场景，可以使用KNN匹配，每个特征点会找到k个最近的匹配点，然后根据比率测试（Lowe’s Ratio Test）筛选出有效的匹配点。

>&#8195;&#8195;比率测试（Lowe's Ratio Test）是David Lowe在SIFT算法中提出的一种筛选匹配点的方法，旨在减少错误匹配。它的主要思想是：**对于每个特征点，找到两个最近的匹配点，并比较它们的距离。如果最近邻和次最近邻的距离差异较大，说明该匹配点是可靠的**；如果两个匹配点的距离差异不大，说明这个特征点可能在多处找到相似的匹配，存在混淆，应该排除掉。


```python
# KNN匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 比率测试
good_matches = []
for m, n in matches:
	# 如果最近邻匹配点m的距离远远好于次最近邻匹配点n，那么我们认为m是一个可靠的匹配点
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

5. 可视化匹配结果
最后，可以将匹配结果绘制出来，直观地展示两幅图像之间的匹配点。

```python
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

&#8195;&#8195;Brute-Force Matcher通过逐个比较两幅图像的描述子，基于距离最小原则寻找匹配点。它的匹配过程非常直观且简单，但由于是穷举匹配，在处理大量特征点时可能效率较低。为了提高匹配精度，可以结合KNN匹配和比率测试等方法。此外，BFMatcher对于不同类型的描述子可以选择不同的距离度量，如欧氏距离（浮点描述子）或汉明距离（二进制描述子）。
#### 3.1.2 代码示例
以下面两幅图为例，完整代码示例为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84c10a8cd3de43cebcef69beaea0b1cc.png#pic_center =600x)


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


img1 = cv2.imread('./opencv_search.png')
img2 = cv2.imread('opencv_orig.png')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建特征检测对象
sift = cv2.SIFT_create()

# 计算描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_L1)
```
```python
# 1. 直接进行匹配
match1 = bf.match(des1, des2)
result1 = cv2.drawMatches(img1, kp1, img2, kp2, match1, None)

# 2. 使用KNN匹配
match2 = bf.knnMatch(des1, des2, k=2)
# 比率测试进行筛选。如果不筛选，绘制时第一张图每个特征点会匹配到两个第二张图中的特征点，画起来不好看
good_matches = []
for m, n in match2:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        
# 绘制匹配特征
result2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good_matches], None)


plt.figure(figsize=[18,8]);
plt.subplot(121); plt.imshow(result1[:,:,::-1]);plt.axis('off');plt.title("Match");
plt.subplot(122); plt.imshow(result2[:,:,::-1]);plt.axis('off');plt.title("knnMatch");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/182a35e4858941c78681ac638f5e9061.png)
### 3.2 FLANN特征匹配
#### 3.2.1 原理
&#8195;&#8195;`FLANN`（Fast Library for Approximate Nearest Neighbors）的核心思想是使用近似搜索算法代替精确搜索来提高匹配效率。对于大规模数据集，精确搜索（如`Brute-Force`）会变得非常缓慢，而`FLANN`基于近似最近邻搜索，并通过构建搜索索引加速匹配过程，因其具有更高的效率和速度，适合于大规模数据集下的快速匹配。

- **近似最近邻（Approximate Nearest Neighbors）**：在高维空间中寻找距离最近的特征点会非常耗时，FLANN使用一些近似算法（如k-d树、KMeans树等）来加速搜索过程，这种方式牺牲了一定的匹配精度，但显著提高了计算效率。
- **索引构建**：FLANN会根据特征描述子数据自动选择合适的索引算法，并预先对数据进行分层索引，从而减少搜索时的计算量。


与Brute-Force Matcher类似，FLANN的特征匹配过程也分为几个主要步骤：

1. 特征检测和描述子计算
首先，像Brute-Force一样，我们使用特征检测算法（如SIFT、SURF、ORB等）来检测关键点，并计算每个关键点的描述子。

	```python
	# 使用SIFT检测关键点并计算描述子
	sift = cv2.SIFT_create()
	
	keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
	keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
	```

2. 设置FLANN参数。FLANN需要根据描述子的类型构建不同的FLANN索引参数`index_params`和最近邻搜索的参数`search_params`。
	- 对于浮点数描述子（如SIFT、SURF），使用KDTREE算法。
		```python
		index_params = dict(algorithm=1, trees=5)   # 使用KD树，适合SIFT、SURF等浮点描述子
		search_params = dict(checks=50)  			# 搜索时递归遍历的次数，值越大越精确，速度越慢
		```
	- 对于二进制描述子（如ORB），可以使用`LSH`（局部敏感哈希），#后的数字是官方文档中推荐可以改大的数值。
		```python
		FLANN_INDEX_LSH = 6
		index_params = dict(algorithm=FLANN_INDEX_LSH,
		                    table_number=6,  		     # 12
		                    key_size=12,    			 # 20
		                    multi_probe_level=1)  		 # 2
		search_params = dict(checks=50)  				 # or pass empty dictionary
		```



- **algorithm**：FLANN使用的算法类型。1表示KD树，适合SIFT、SURF等浮点描述子。
- **trees**：KD树的树数量，通常5是一个合适的值。
- **checks**：经验值，指定递归遍历的次数，值越高匹配精度越高。如KDTREE设为5，那么搜索次数设为50。

3. 进行KNN匹配
FLANN通常与KNN（k近邻）匹配一起使用，而不是直接精确匹配。对于每个特征点，它会寻找k个最近的匹配点。

```python
# 使用FLANN进行KNN匹配
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)
```

4. 比率测试筛选匹配点

```python
# 比率测试
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

5. 可视化匹配结果
最后，可以将匹配结果进行可视化，展示两个图像之间的匹配特征点。

```python
# 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
&#8195;&#8195;`FLANN Matcher`基于近似最近邻搜索，通过索引结构显著加快了匹配过程，特别适合处理大量特征点数据。它的核心优势在于速度快，但代价是匹配的精度略有下降（**一般图像拼接都用暴力特征匹配**）。结合KNN匹配和比率测试，FLANN可以在保持高效的同时，提供良好的匹配精度，是大规模特征匹配任务中的常用工具。
#### 3.2.2 代码示例

```python
# 创建FLANN特征匹配对象
index_params = dict(algorithm=1, tree=5)
# 根据经验, kdtree设置5个tree, 那么checks一般设置为50
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.match(des1, des2)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49bfb32e800c4ad5a474259d5c633082.png#pic_center =600x)
## 四、图像查找、矫正和拼接
&#8195;&#8195;当两幅图像中有**共面物体**（例如拍摄相同平面物体的两张图片），我们可以通过特征点匹配求解出**单应性矩阵（Homography Matrix）**。单应性矩阵是一个 $3 \times 3$ 的矩阵，表示了图像间的**投影变换**，它描述了两个图像之间的**平面映射关系**。通过单应性矩阵，我们可以将一个图像中的某个平面映射到另一个图像中的对应平面（要解出两张图的单应性矩阵，至少要在原图选取4个点才行）。
 ![image.png](https://img-blog.csdnimg.cn/img_convert/0d8d930a95dae8dbe956ad9869b80e9a.png#pic_center =600x)
&#8195;&#8195;使用`cv2.findHomography` 函数，可以通过匹配的特征点计算两幅图像之间的单应性矩阵，其语法为：

```python
findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]) -> retval, mask
```
* **`src_pts`**：源图像中的点集，通常是通过特征匹配得到的特征点坐标。可以是CV_32FC2类型，形状为 `(N, 2)`；也可以是`vector <Point2f>`类型，形状为 `(N, 1, 2)` 。
* **`dst_pts`**：目标图像中的点集，与 `src_pts` 对应，表示这些点在目标图像中的位置。
* **`method`**：用于计算单应性矩阵的算法，有以下几种可选方法：
    * `0`（默认）：直接使用最小二乘法计算单应性矩阵，适用于噪声较小的情况。
    * **`cv2.RANSAC`**：使用 RANSAC（随机采样一致性）算法来计算单应性矩阵，适合存在噪声或异常匹配的情况。RANSAC 会自动剔除不符合投影关系的点对。
    * `cv2.LMEDS`：使用最小中值平方法（Least-Median of Squares），也可用于存在噪声的情况。
* **`ransacReprojThreshold`**：RANSAC 方法中的参数，用于设定当使用 RANSAC 时的重投影误差阈值。默认值为 3.0，表示允许最大3个像素的误差。


单应性矩阵可以用于：

* **图片对齐（摆正）**：矫正图片中的透视失真（如检测到书页或名片后，将其摆正）。
* **图片查找和替换**：在场景中查找目标物体的位置，将其替换为新的物体
* **图像拼接**：将多张图片拼接成全景。

![image.png](https://img-blog.csdnimg.cn/img_convert/9d78e2eaf729f2edb5435447106889a0.png#pic_center =600x)
 ![image.png](https://img-blog.csdnimg.cn/img_convert/607088f0b3ca32fd7bfb80fd4a890547.png#pic_center =600x)
&#8195;&#8195;在[《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)3.7章透视变换中，我们通过`cv2.warpPerspective`函数，在原图中选取四个点来将图像进行变换摆正。这种方法适用于较简单的情况，当我们已知要处理的物体的四个角点时，可以迅速得到想要的结果，且实现更为简单。然而，它缺乏灵活性，对错误的角点选择敏感。

&#8195;&#8195;特征匹配与单应性矩阵方法是一个更通用、自动化的解决方案，适合于处理复杂场景，特别是目标物体在图像中的位置不确定时。

类别 | 特征匹配+单应性矩阵 | `cv2.warpPerspective` 透视变换 |
| --- | --- | --- |
| **自动化程度** | 自动：基于特征匹配找到对应点，可以处理变形、旋转、缩放等情况。 | 手动：需手动指定角点，适用于已知位置的矩形物体。 |
| **适用场景** | 适合复杂场景，物体不规则变形，目标物体在图像中的位置不确定。 | 适合固定位置的已知矩形物体，简单场景中。 |
| **精确度** | 高：通过特征匹配得到的单应性矩阵通常能较准确地反映图像间的关系。 | 取决于手动角点的选择，选择不当可能导致失真。 |
| **处理速度** | 较慢：计算特征点和匹配的时间开销。 | 快：直接计算角点，无需特征匹配。 |
| **灵活性** | 高：适应各种不同的图像变换，包括旋转、缩放、剪切等。 | 低：只能用于矩形变换，灵活性受限。 |

### 4.1 图像查找和矫正

&#8195;&#8195;`findHomography`函数中，常用的是RANSAC（Random Sample Consensus）算法。这是一种迭代算法，从数据中随机抽取最小数量的点来估计模型，然后用这个模型验证其他数据点是否符合（符合是“内点”，否则是“外点”）。通过多次迭代，最终选择使“内点”数量最多的模型作为最终结果。它特别适用于**存在噪声或离群值**（outliers）的数据，允许有一些错误的特征点。RANSAC 工作流程：
1. **随机采样**：从数据集中随机选择最少数量的点来拟合模型（比如，估计单应性矩阵时，至少需要4对点）。
   
2. **模型拟合**：基于这些随机采样的点，估计模型参数（比如单应性矩阵）。
   
3. **计算内点**：将所有其他数据点代入该模型，计算它们是否符合模型。符合模型的点称为“内点”，即满足某个误差阈值条件。

4. **重复迭代**：重复前面三个步骤若干次，每次随机选择不同的采样点，估计出多个模型，记录下内点最多的模型。

5. **最终模型选择**：经过多次迭代后，选择内点最多的模型作为最终的结果。


&#8195;&#8195;在几何变换中，源图像中的点通过单应性矩阵映射到目标图像时，实际映射点和期望的目标点之间的欧氏距离称之为**重投影误差**，该误差衡量了变换后的点与实际位置的偏差。在 `cv2.findHomography` 中，`ransacReprojThreshold` 是 用于定义在使用RANSAC方法计算内点时的**重投影误差阈值**，它决定了一个点是否可以被视为“内点”。

&#8195;&#8195;阈值越小，模型精度要求越高，这可能会导致某些略微偏差的点也被视为外点，模型可能会排除掉一些有用的数据点。反之会降低模型精度要求，有可能导致一些误差较大的外点被错误地包含进来。通常，`ransacReprojThreshold` 的默认值在3左右（像素单位）。


```python
import cv2
import numpy as np


img1 = cv2.imread('opencv_search.png')
img2 = cv2.imread('opencv_orig.png')
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#创建特征检测器
sift = cv2.SIFT_create()

# 计算特征点和描述子
kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)

# 进行FLANN特征匹配
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 对描述子进行knn特征匹配,然后筛选出好的特征点
matches = flann.knnMatch(des1, des2, k=2)
good_matches = []
for (m, n) in matches:
    # 阈值一般设0.7到0.8之间.
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        

# 根据匹配的特征点计算单应性矩阵
if len(good_matches) >= 4:
	# 提取两张图中的匹配点
	# 因为findHomography函数需要的是(N, 1, 2)形状的`vector <Point2f>`类型来表示点，所以需要reshape
	# 因为不确定good_matches中有多少个点，reshape中-1表示自动匹配，但最后两个维度必须是1和2
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)    
    # 根据匹配上的关键点去计算单应性矩阵，使用RANSAC算法过滤掉错误匹配
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5)
    
    # 获得第一张图的四个角点
    h, w = img1.shape[:2]    
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # 使用单应性矩阵将角点从第一张图变换到第二张图
    dst = cv2.perspectiveTransform(pts, H)
    print(dst)
    # 在第二张图中使用绘制多边形的方法画出找到的目标位置
    cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 2)
    # 使用单应性矩阵将图像矫正，如果需要矫正的话
	# img_aligned = cv2.warpPerspective(img2, H, (w, h))
    
else:
    print('not enough point number to compute homography matrix')
    exit()

# 画出匹配的特征点
ret = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [goods], None)
cv2.imshow('ret', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- `kp1,kp2`：关键点列表，中每个元素都是一个 `cv2.KeyPoint` 对象，包含位置（`pt`）、尺度（`size`）、方向（`angle`）等信息。
- `des1,des2`：列表形式，每个元素都是关键点对应的描述子，是一个高维向量（SIFT、SURF和ORB中分别是128维、64维和32维）
- `matches`,`good_matches`：特征点匹配结果，也一个列表，列表中的每个元素是一个匹配对象 `m`。
- `cv2.perspectiveTransform`：用于将图A中的一组点坐标`src_pts`通过一个透视变换矩阵（通常是单应性矩阵H）映射到图B的平面上，返回图B中对应的坐标`dst_pts`。
- `cv2.warpPerspective`：用于对整个图像进行透视变换，常用于图像对齐、全景拼接、透视矫正等。函数签名为：

	```python
	dst_image = cv2.warpPerspective(src_image, H, (dst_width, dst_height))
	```
	* **`src_image`**：源图像，输入图像需要进行透视变换。
	* **`H`**：3x3 的透视变换矩阵（单应性矩阵），通常通过 `cv2.findHomography` 计算得到。
	* **`(dst_width, dst_height)`**：输出图像的大小（宽度和高度），指定输出图像的尺寸。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0603df304aef4bd28ec7b2560fc185b5.png#pic_center =600x)
### 4.2 图片拼接
#### 4.2.1 手动拼接

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取6张图像
image_files =  ["./map1.png","./map2.png"]
images = [cv2.imread(file) for file in image_files]

# 初始化SIFT特征检测器
sift = cv2.SIFT_create()

# 暴力匹配器初始化，使用欧氏距离
bf = cv2.BFMatcher(cv2.NORM_L2)

def stitch_images(img1, img2):
    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 进行暴力匹配
    matches = bf.knnMatch(des1, des2, k=2)
    goods_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goods_matches.append(m)
            
    if len(goods_matches) >= 4:   

        # 提取匹配的关键点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in goods_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in goods_matches]).reshape(-1, 1, 2)
    
        # 计算单应性矩阵（使用RANSAC算法去除误匹配）
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    else:
        print('not enough point number to compute homography matrix')
        exit()

    # 计算拼接后的尺寸（基于透视变换的结果）
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    pts1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)

    # 透视变换后的点
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    print(pts)

    # 计算拼接后的新尺寸边界
    # 转为int类型时小数部分被截断，为此在展平为一维之后，扩展一个像素。
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 1)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 1)

    # 手动构造平移矩阵,确保没有负坐标。如果不平移, img1很大一部分都在显示窗口外面, 我们看不到。
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    # 对第一张图像进行仿射变换
    result = cv2.warpPerspective(img1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    # 将第二张图像复制到拼接结果中
    result[-ymin:-ymin+h2, -xmin:-xmin+w2] = img2

    return result

# 依次拼接多张图像
stitched_image = images[0]
for i in range(1, len(images)):
    stitched_image = stitch_images(stitched_image, images[i])

# 显示最终拼接结果
plt.figure(figsize=[20,10])
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

```python
array([[[   0.     ,    0.     ]],

       [[   0.     ,  479.     ]],

       [[ 639.     ,  479.     ]],

       [[ 639.     ,    0.     ]],

       [[-481.34628, -175.11775]],

       [[-773.80646,  545.1025 ]],

       [[ 197.02798,  477.85214]],

       [[ 326.34628,   43.06862]]], dtype=float32)
```
构建`stitch_images`函数，可以实现多张图的拼接。
- 变换后的坐标出现负值，表示变换后的图像有部分不会被显示，所以需要对图像尺寸进行调整。先计算 `result_pts`在x轴和y轴方向的最小值`(xmin, ymin)` 和最大值 `(xmax, ymax)`，就得到了拼接后整个图像的边界范围（拼接后图像的左上角和右下角）。  
-  为了确保图像的所有点都位于正坐标系内，需要对图像进行平移操作，`translation_dist` 就是平移的量：
      * `-xmin`: 将图像的最小 `x` 值平移到0，以确保所有 `x` 坐标非负。
      * `-ymin`: 将图像的最小 `y` 值平移到0，以确保所有 `y` 坐标非负。
- 构建一个3x3的仿射平移矩阵move_matrix，平移矩阵的形式为：
$$ \begin{bmatrix}
1 & 0 &tx  \\
 0& 1 &ty  \\
 0&0  & 1 \\
\end{bmatrix}$$
- 将原来的单应性矩阵 `H` 与平移矩阵 `move_matrix` 进行矩阵乘法（`dot`），得到一个新的变换矩阵。这个新矩阵不仅包含了透视变换，还考虑了坐标的平移，确保拼接后的图像不会出现负坐标。最终，`cv2.warpPerspective()` 会生成将第一张图像 img1 透视变换后的结果，并放置到新的平移后的坐标系统中。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5641cb36eb404d6285f2be44507853db.png#pic_center =600x)



下一种方式效果看起来好一些，但依赖于图像的初始特征和拼接时的对齐精度，在处理更加复杂的图像场景时，需要额外的调整。

```python
...
	# 计算单应性矩阵（使用RANSAC算法去除误匹配）
	H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	# 矩阵 np.linalg.inv(M) 是对单应性矩阵 M 求逆矩阵。这样做的目的是将 img2 反向映射到 img1 所在的坐标空间。后面广播的时候高度会缺失6个像素
	warpImg = cv2.warpPerspective(img2, np.linalg.inv(H), (img1.shape[1]+img2.shape[1], img2.shape[0]+6))
	# 深拷贝一份，以免修改 warpImg 时影响原始数据。
    direct=warpImg.copy()									
    # 因为 img1 已经处于原始的坐标系，而 warpImg 是 img2 经过透视变换后的图像，因此直接将 img1 放在左边，实现了简单的拼接。
    direct[0:img1.shape[0], 0:img1.shape[1]]=img1			
    
# 处理中间黑线问题. 
# 经过仔细观察, 中间的黑线是左图第743列的位置。水平拼接这两部分，中间跳过了第743列，相当于删除了那条黑线
direct3 = np.hstack((direct[:, :742].copy(), direct[:, 744:].copy()))
# 然后再对局部做一个高斯模糊. 
dst = cv2.GaussianBlur(direct3[:, 740:747], (5, 5), sigmaX=0)
# 替换
direct3[:, 740:747] = dst

cv2.imshow('result', direct3) 
cv2.imshow('ret',ret)
cv2.waitKey(0) 
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/25262c5cade344ba92de611c520fffe0.png#pic_center =800x)
#### 4.2.2 使用Stitcher自动拼接
&#8195;&#8195;`cv2.Stitcher` 是 OpenCV 提供的用于图像拼接的高层次类，它可以自动处理多张图像的全景拼接。这个类简化了拼接流程，让用户无需自己实现特征点检测、匹配、单应性计算等步骤。
```python
import cv2
import glob
import matplotlib.pyplot as plt
import math

imagefiles = glob.glob("boat/*")
imagefiles.sort()

images = []
for filename in imagefiles:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

num_images = len(images)
```

```python
# Display Images
plt.figure(figsize=[30,10]) 
num_cols = 3
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
    plt.subplot(num_rows, num_cols, i+1) 
    plt.axis('off')
    plt.imshow(images[i])
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/07047474a5096a39b0d7507bad5a06ba.png)

```python
# 创建一个 Stitcher 对象，对象内部包含了完成图像拼接的所有必要步骤
# 包括：特征点检测、特征匹配、单应性矩阵计算、图像变换和融合等。
# 有cv2.Stitcher_PANORAMA（全景）和cv2.Stitcher_SCANS（扫描）两种模式
stitcher = cv2.Stitcher_create()
# 返回拼接结果和状态码status，status用来判断拼接是否成功
status, result = stitcher.stitch(images)
if status == 0:
    plt.figure(figsize=[30,10]) 
    plt.imshow(result)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8c9409f427355282c0cb9f8e1222cae2.png)
 **维度** | **cv2.Stitcher** | **手动拼接（特征查找+匹配+单应性+透视变换）** |
| --- | --- | --- |
| **操作复杂度** | 简单易用，只需提供图像列表，自动完成拼接 | 复杂，需要编写特征提取、匹配、单应性计算、透视变换的步骤 |
| **灵活性** | 灵活性有限，用户难以控制每一步 | 灵活性高，用户可完全控制拼接过程 |
| **拼接质量** | 常见场景下表现良好，局限于特定复杂场景 | 处理复杂场景效果更好，拼接质量可通过调试优化 |
| **处理速度** | 通常较快，适合准实时任务 | 较慢，视使用的特征提取方法而定 |
| **后处理** | 自动完成，用户难以干预 | 用户可定制后处理步骤，如曝光补偿、图像混合 |
