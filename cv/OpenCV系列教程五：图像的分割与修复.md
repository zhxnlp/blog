@[toc]

- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)
## 一、图像分割
&#8195;&#8195;图像分割分为传统图像分割和基于深度学习的图像分割方法。传统图像分割就是使用OpenCV进行的图像分割，主要有:分水岭法、GrabCut法、MeanShift法、背景扣除。
### 1.1 分水岭法
#### 1.1.1 基本原理

>论文：[《IMAGE SEGMENTATION AND MATHEMATICAL MORPHOLOGY》](https://people.cmm.minesparis.psl.eu/users/beucher/wtshed.html)




&#8195;&#8195;**分水岭算法**是一种图像分割技术，广泛应用于从复杂背景中分割目标区域。分水岭算法可以直观地理解为<font color='deeppink'>“水流从高处向低处汇聚”的过程</font >。最朴素的理解是：

* **图像的每个像素值（灰度值）可以看作高度**。也就是说，一个亮的区域（高灰度值）被看作“高地”，而暗的区域（低灰度值）被看作“低地”或“谷底”。这种高度变化可以看作是一个复杂的地形起伏。
* **水流从高地汇聚到低地**。如果我们假想向这个地形中倒水，水将从山顶流向低谷并形成不同的汇流区域。分水岭算法就是在这种图像地形中寻找那些自然形成的边界（分水岭），它们是不同“流域”之间的分界线。

&#8195;&#8195;简单说就是：分水岭算法将图像视为一个地形，其中灰度值代表高度。算法的目标是找到这些“地形”中的山脊线（分水岭线），以此分离出不同的物体或区域。

1. 基本概念
	- **将图像看作一个三维地形**: 将每个像素的灰度值解释为一个高度，形成一个虚拟的“地形模型”，亮的区域是高地，暗的区域是低地。
	- **模拟“注水”过程**: 假设水从地形模型的最低点开始缓慢注入。水会沿着“低谷”流动并充满各个“盆地”（图像中的不同灰度值区域）。随着水位不断上升，原本属于不同区域的“水域”（代表不同物体的区域）最终会在某些地方相遇。
	- **形成分割边界（分水岭）**: 当水从两个不同的区域相遇时，算法将这些相遇点标记为“分水岭”。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/579645f30eb84f83a63fb3f07845efa5.gif#pic_center)


2. 算法步骤
	1. **预处理**: 首先需要对图像进行一些预处理，例如灰度化和高斯模糊，以减少噪声。
	2. **阈值处理**: 应用阈值分割，生成前景和背景的初始标记图像（marker）。
	3. **距离变换**: 使用距离变换函数（`cv2.distanceTransform`），帮助识别图像中的前景物体。距离变换会计算出每个像素到背景的最小距离，帮助分离重叠的物体。
	4. **生成Marker**: 根据前景物体和背景生成初始的标记（marker）。标记区域会以不同的标签（通常是不同的整数）表示。
	5. **应用分水岭算法**: 调用OpenCV的`cv2.watershed()`函数，将marker作为输入，进行分水岭算法。算法会返回一个与输入图像大小相同的标记图像，其中不同的区域有不同的标记值。
	分水岭算法会将原图像中每个像素分配给某个标记区域，边界区域则被标记为 -1，表示这是分水岭线。

3. 算法分析：优点是能有效分割有明确边界的图像，尤其是可以处理一些具有复杂形状的物体。同时适用于不均匀光照或图像模糊的情况。缺点是**容易过分割**，需要一些预处理步骤，并<font color='deeppink'>在启用分水岭算法前精确的标记图片</font >。比如下图有很多小的低地，应该被整个当成一个低地，而不是一连串高地和低地。
![](https://img-blog.csdnimg.cn/img_convert/546a9ae3bf4e414f800fa68c73349211.png#pic_center)
#### 1.1.2 实例：使用分水岭算法分割硬币图像
- **读取图像并预处理**: 使用高斯模糊消除图像中的噪声。
- **阈值分割**: 使用二值化将图像分为前景和背景，方便后续分割。
- **通过距离变换获取确定的前景**: 计算前景物体到背景的距离，再使用一次阈值化，提取硬币的中心区域（确定的前景）。
- **膨胀操作获取确定的背景**
- **计算未知区域**（前景与背景之间的过渡区域）
- **标记Marker**: 利用连接组件的方式为不同的区域创建标记，并为未知区域赋值为0。
- **分水岭算法**: `cv2.watershed`对图像执行分割，结果中的边界线用红色标记。分水岭算法将标记的`0`的区域视为不确定区域，将标记为`1`的区域视为背景区域，将标记大于`1`的正整数表示我们想得到的前景。

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. 图像预处理 - 高斯模糊
gray = cv2.GaussianBlur(gray, (5, 5), 0)


# 2. 阈值分割 - 图像的灰度直方图是一个典型的双峰结构，可以使用OTSU算法进行二值化处理
_ = plt.hist(gray.ravel(), bins=256, range=[0, 255])
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. 距离变换 - 获取明确的前景物体
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 4. 获取背景
sure_bg = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
sure_fg = np.uint8(sure_fg)
# 5. 获取未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 6. 求连通域, 用0标记图像的背景，用大于0的整数标记其他对象
# connectedComponents要求输入的图片是个8位的单通道图片, 即单通道的0到255的图片.
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 7. 应用分水岭算法,返回的markers已经做了修改. 边界区域标记为-1了.
markers = cv2.watershed(img, markers)

# 标记分水岭边界
img[markers == -1] = [0, 0, 255]

# 显示结果
plt.figure(figsize=[9,8]);
plt.subplot(231);plt.imshow(gray,cmap='gray');plt.title("gray");plt.axis('off');
plt.subplot(232);plt.imshow(thresh,cmap='gray');plt.title("thresh");plt.axis('off');
plt.subplot(233);plt.imshow(sure_fg,cmap='gray');plt.title(" sure_fg");plt.axis('off');
plt.subplot(234);plt.imshow(sure_bg,cmap='gray');plt.title(" sure_bg");plt.axis('off');
plt.subplot(235);plt.imshow(unknown,cmap='gray');plt.title(" unknown");plt.axis('off');
plt.subplot(236);plt.imshow(img[:,:,::-1]);plt.title("Result of Watershed Algorithm");plt.axis('off');
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dc47bbaebf204f31b974939ef352a31c.png#pic_center =600x)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba025d29a2d34bccacc9232abfaf50f2.png)


下面对代码进行更详细的解析：

```python
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```
&#8195;&#8195;这段代码的作用是将灰度图像进行二值化处理，生成一个二值图。Otsu's阈值算法会分析图像直方图，并根据像素的分布情况，自动计算一个最优的阈值，将图像进行二值化。由于分水岭算法通常要求**背景为黑色，前景为白色**，所以需要使用反转二值化（cv2.THRESH_BINARY_INV），将灰度值小于阈值的部分（通常是前景）设为白色（255），将灰度值大于阈值的部分（通常是背景）设为黑色（0）。

&#8195;&#8195;二值图像无法准确区分重叠物体，只将图像分为前景和背景，且所有前景像素都会被归为同一类，所以此时还不能直接进行分割。


```python
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
```
- `img` ：要处理的图像
- `distanceType` ：计算距离的方式—— `DIST_L1`, `DIST_L2`
- `maskSize`：进行扫描时的kernel的大小, `L1`用`3`, `L2`用`5`
- 返回结果是一个和图像同尺寸的数组，每个元素是每个像素到最近背景的距离值。

&#8195;&#8195;分水岭算法的本质是基于标记的分割，为了让分水岭算法正确地找到物体的边界，需要明确的前景和背景标记以及未确定的区域（即前景和背景之间的模糊区域，需要算法来决定边界）。为了更好地处理复杂的场景，**距离变换和膨胀**等操作用于生成可靠的初始标记（markers），为分水岭算法提供更准确的分割信息。

&#8195;&#8195;**距离变换区分重叠物体**：距离变换计算前景中每个像素到最近背景的距离。越靠近物体中心的区域，距离肯定越大，多个重叠物体的中心区域将表现出显著的差异。通过对距离变换图像进行阈值化，可以保留离背景较远的前景像素，生成“确定的前景”区域。通过距离变换可以把多个物体的中心部分分开，从而使它们能够被单独标记。

```python
sure_bg = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
sure_fg = np.uint8(sure_fg)
```
&#8195;&#8195;通过对`thresh`图像进行膨胀（`cv2.dilate`）操作，获取**“确定的背景区域”（sure background）**。上述代码中使用了一个`3x3`的全1矩阵，对图像的每个前景像素进行3次膨胀操作，将前景物体的轮廓向外扩展，剩下的就是sure background。

&#8195;&#8195;通过距离变换得到前景的确定区域（sure_fg），通过膨胀得到背景的确定区域（sure_bg），再通过差集得到前景和背景之间的未知区域（unknown），这三部分的明确划分是成功应用分水岭算法的关键。

```python
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
```
&#8195;&#8195;**`cv2.connectedComponents(sure_fg)`** 函数会对前景区域`sure_fg`进行连通分量标记，将每个独立的前景区域（物体）赋予不同的标签（Label）。返回值`markers`是一个与输入图像同尺寸的标记图，此时背景被标记为`0`，各个前景物体被标记为 `1,2,3...`。
&#8195;&#8195;`markers+1`并将`unknown`区域设为`0`之后，未知区域被标记为 `0`，背景被标记为 `1`。各个前景物体被标记为 `2,3,4...`。

```python
markers = cv2.watershed(img, markers)
```
&#8195;&#8195;标记为`0`的区域是分水岭算法中待分割的未知区域。分水岭算法会尝试在不同标记之间找到边界（山脊线），并将这些边界标记为`-1`，代表分水岭线。

&#8195;&#8195;另外，我们使用膨胀操作来找到确定的背景，如果同样的思路，使用腐蚀操作来寻找确定的前景会出错。因为硬币是有重叠的，这样操作之后硬币之间会形成白色通道，如下图所示，这样肯定是不对的，后续使用分水岭算法会有问题。

```python
# 封装显示图片的函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

fg = cv2.erode(opening,np.ones((3, 3), np.uint8), iterations=2)
unknown = cv2.subtract(bg, fg)
cv_show('unknown', np.hstack((thresh, bg, fg, unknown)))
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bcd1e521de9b4dfb9418af689802fa22.png)

&#8195;&#8195;如果要扣出前景物体，可以使用以下代码。另外使用`canny`等目标检测算法，或使用`findContours`查找轮廓，虽然也可以检测出物体边缘，但是要想进行抠图等后续操作，就不好处理了。

```python
# 1. 使用makers进行抠图
mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)		# 初始化一个全为黑色的掩膜mask
mask[markers > 1] = 255										# 将mask中所有前景区域标记为白色
# 使用与运算，任何颜色和白色进行与运算结果不变，任何颜色和黑色进行与运算结果都是黑色
coins = cv2.bitwise_and(img, img, mask=mask)
	
# 2. 使用canny直接检测轮廓
img_canny = cv2.Canny(img, 100, 150)

# 3. 使用findContours查找轮廓，复习一下以前的内容
img2 = cv2.imread('water_coins.jpeg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 二值化
_, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# findContours要求是单通道, 0到255的整数的图片, 最好是二值化的图片. 
contours, _ = cv2.findContours(thresh2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 显示轮廓, 会直接修改img2
cv2.drawContours(img2, contours, -1, (0, 0, 255), 3)

	
plt.figure(figsize=[12,6]);
plt.subplot(141); plt.imshow(img[:,:,::-1]);plt.axis('off');plt.title("img");
plt.subplot(142); plt.imshow(coins[:,:,::-1]);plt.axis('off');plt.title("coins");
plt.subplot(143); plt.imshow(img_canny,cmap='gray');plt.axis('off');plt.title("canny");		
plt.subplot(144); plt.imshow(img2[:,:,::-1]);plt.axis('off');plt.title("findContours");	
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b44dc550b7f468d8be01ddfaa126558.png#pic_center)
可以看到结果还是有一些瑕疵，比如图中三个硬币之间白色的部分。
>- `canny`算法原理见[《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)第2.3章节边缘检测；
>- `findContours`函数见[《OpenCV系列教程三：形态学、图像轮廓、直方图》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5501)第二章图像轮廓。
### 1.2 GrabCut算法
>论文《GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts》

#### 1.2.1 原理

&#8195;&#8195;GrabCut算法于2004年提出，是一种用于图像分割的**交互式算法**，用户可以通过提供初始标记（矩形框或mask）来指定前景的大体区域，其余区域则是背景。然后基于图割（Graph Cut）理论，采用分段迭代的方法分析前景物体, 形成模型树。最后通过最大化前景和背景之间的对比度来实现目标物体的精确提取。其实现步骤为：
1. **初始化**：用户在图像中绘制一个矩形框，框内的区域被认为包含前景，框外区域被认为是背景。框内部的区域将进一步分为前景（目标物体）和背景（目标物体内部的背景）。
    
2. **高斯混合模型（GMM）**：GrabCut算法假设前景和背景可以分别用两个独立的高斯混合模型来描述。首先，它会对矩形框内的像素进行聚类，并根据颜色分布分别生成前景和背景的GMM。
    
3. **图模型构建**：算法把图像像素表示成图的节点，每个节点代表一个像素，节点之间的边表示像素之间的相似性。前景和背景的可能性用GMM进行建模，基于这些模型将像素连接到两个超级终端（前景或背景），从而每条边都有一个属于前景或者背景的概率。
    
4. **图割（Graph Cut）**：通过最大流/最小割算法，算法计算出最优的分割方式，将像素划分为前景和背景。如下图所示，一些节点连接前景终端，一些节点连接背景终端，那么就可以从中分开（右下图中的`cut`操作）。
![](https://img-blog.csdnimg.cn/img_convert/ec51adef7bbae1edfe09a73b6ab61564.png#pic_center =600x)    

5. **迭代优化**：分割结果并不总是一次就能达到最优，因此GrabCut会在每次迭代后调整前景和背景的GMM，直到收敛到一个理想的分割。

| GrabCut优点                         | GrabCut缺点                         |
|------------------------------|------------------------------|
| **灵活**：可以通过提供不同的初始矩形或掩码来调整分割结果。 | 对于复杂背景或低对比度图像（前景和背景的颜色相似时）效果不佳  |
| **高效**：相较于全自动的图像分割方法，GrabCut能够通过少量用户输入得到精确的结果。     | 计算复杂度较高，处理大图像时速度慢 |

#### 1.2.2 代码示例    
GrabCut算法函数签名为：
```python
grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) -> mask, bgdModel, fgdModel
```
- `img`：待分割的源图像，必须是8位3通道，在处理的过程中原图不会被修改
- `mask`：掩码图像，用于标记每个像素的状态，初始为全0。分割完成后，有四种结果：    
    * `cv2.GC_BGD` (0)：确定的背景。
    * `cv2.GC_FGD` (1)：确定的前景。
    * `cv2.GC_PR_BGD` (2)：可能的背景。
    * `cv2.GC_PR_FGD` (3)：可能的前景。
- `rect`：用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
- `bgdModel,fgdModel`：背景模型/前景模型，这些模型是分割的基础。如果为None，函数内部会自动创建一个。模型必须是单通道浮点型图像，且行数只能为`1`，列数只能为`13x5`；
- `iterCount`：迭代次数；
- `mode`：初始化前景和背景的方式，可选的值有：
	- GC_INIT_WITH_RECT（=0），通过矩形框（rect）来初始化GrabCut算法；
	- GC_INIT_WITH_MASK（=1），通过掩码来初始化GrabCut算法。比如你已经进行了一次分割，得到了更新后的mask，但是效果不好。此时可以手动修改标记，然后再次传入进行继续分割。
	- GC_EVAL（=2），执行分割。
- 算法返回`mask,bgdModel,fgdModel`，后两个不需要；mask会在计算后被修改，所以只需要执行就行。

&#8195;&#8195;下面举例说明。假设要分割出人脸部分，通过plt以坐标形式显示出图像，可以估计出前景矩形框的坐标（矩形框左上角、右下角）为`(200, 170), (380,390)`，矩形框宽高为`(180,220)`。    
```python
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./lena.png')
image_copy = img.copy()
cv2.rectangle(image_copy, (200, 170), (380,390), (255, 0, 255), thickness=2, lineType=cv2.LINE_8);

plt.figure(figsize=[16,12]);
plt.subplot(121); plt.imshow(img[:,:,::-1]);plt.title("img");
plt.subplot(122); plt.imshow(imageRectangle[:,:,::-1]);plt.title("Rect");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/491e49bac3fb4144803f455e0bba2859.png#pic_center =600x)


```python
# 画出矩形框矩形框(x, y, w, h)，初始化前景区域
rect =(200, 170, 180, 220)

# 创建一个与图像大小相同的掩码mask，并初始化为0
mask = np.zeros(img.shape[:2], np.uint8)
# 执行GrabCut算法，第一次使用grabcut是用户指定rect
cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

# 将背景、可能的背景标记为0，前景、可能的前景标记为255
mask1 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
# 同样的与运算方式进行抠图
img_cut1 = cv2.bitwise_and(img, img, mask=mask1)
```
&#8195;&#8195;现在我们可以在此基础上进行二次计算。比如除了脸部，我们还想抠出上面的帽子部分，帽子坐标为`(200:110),(380:170)`，可以进行第二次`grabcut`分割。

```python
# 第二次使用grabcut, 对mask进行修改，将帽子部分区域设为前景。
mask[110:170, 200:380] = 1
cv2.grabCut(img, mask, None, None, None, 5, mode=cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
# 使用与运算.
img_cut2 = cv2.bitwise_and(img, img, mask=mask2)
cv2.rectangle(image_copy, (200, 110), (380, 170), (0, 255, 0), 3)

plt.figure(figsize=[16,12]);
plt.subplot(131); plt.imshow(img_cut1[:,:,::-1]);plt.title("img_cut1");
plt.subplot(132); plt.imshow(imageRectangle[:,:,::-1]);plt.title("Rect");
plt.subplot(133); plt.imshow(img_cut2[:,:,::-1]);plt.title("img_cut2");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b0dd4db16b24721a8be3f87c7fc529f.png#pic_center )
#### 1.2.3 交互式grabCut程序    
&#8195;&#8195;本节需要实现在图片上拖动鼠标标注前景区域，然后使用grabCut进行分割的脚本。所有内容封装在名为`grabCutAPP`的类里面，通过运行`run`方法实现以上功能。

解题思路：
1. 首先实现图片标注功能
	-  读取图片 
	- 当用户按下鼠标左键后，拖动鼠标时，跟随鼠标位置绘制矩形（绿色）
	-  当用户松开鼠标左键时，绘制并固定此时的矩形（红色）。最终经过一次操作后（按下鼠标左键、拖动、最后释放左键），图像上只留下一个红色矩形  
2. 实现grabCut分割功能
	- 同步展示标注窗口和分割窗口
	- 标注完成后，设置标注区域为前景区域，进行grabCut分割
	- 更新分割后的窗口

>鼠标控制详见[《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)2.4章节

```python
import cv2
import numpy as np

class grabCutAPP:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)       
        self.drawing_image = self.image.copy()  						# 创建图片副本用于绘制           
        self.drawing = False											# 鼠标按下标志        
        self.start_point = (0, 0)										# 起始点坐标        
        self.end_point = (0, 0)											# 结束点坐标        
        self.draw_window = "Draw Rectangle"								# 标注窗口名称
        self.Segment_window = "Segment_image"							# 分割窗口名称
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  	# 初始mask
        self.output = np.zeros(self.image.shape[:2], dtype=np.uint8)	#初始分割图片
        self.rect = (0, 0, 0, 0)										# 初始标注矩形
        
        

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:								# 按下鼠标左键
            # 记录起始点，设置绘制状态
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:								# 移动鼠标
            if self.drawing:                
                self.end_point = (x, y)									# 更新结束点位置                
                self.drawing_image = self.image.copy()					# 复制原图
                # 绘制绿色矩形
                cv2.rectangle(self.drawing_image, self.start_point, 
                            self.end_point, (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:								# 释放鼠标            
            self.end_point = (x, y)										# 更新结束点位置
            self.drawing = False
            # 在原图上绘制红色矩形
            cv2.rectangle(self.image, self.start_point, 
                         self.end_point, (0, 0, 255), 2) 
            self.drawing_image = self.image.copy()						# 更新显示图片

    def run(self):        
        cv2.namedWindow(self.draw_window)								# 创建窗口        
        cv2.setMouseCallback(self.draw_window, self.mouse_callback)		# 设置鼠标回调函数

        while True:
            # 显示图片
            cv2.imshow(self.draw_window, self.drawing_image)
            cv2.imshow(self.Segment_window, self.output)
            # 检测按键，按ESC退出
            key=cv2.waitKey(1) 
            if key==27:
                break
            elif key==ord('g'):             							# 按下g键，开始分割
                self.rect = (min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1]),
                         abs(self.start_point[0]-self.end_point[0]), abs(self.start_point[1]-self.end_point[1]))
                cv2.grabCut(self.image, self.mask,self.rect , None, None, 5, mode=cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((self.mask == 1) | (self.mask == 3), 255, 0).astype(np.uint8)
                # 使用与运算.
            self.output = cv2.bitwise_and(self.image, self.image, mask=mask2)
                
        # 释放资源
        cv2.destroyAllWindows()
```

```python
app = grabCutAPP("cat.png")  
app.run()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bcef933c46b4dc89b98b84c69e3e2a2.png)
&#8195;&#8195;还是有一些可以继续改进的地方。比如设置同时按下鼠标左键和ctrl键，框选的区域被加入前景再一次进行识别；或者是同时按下鼠标左键和alt键，框选的区域从前景中删除，再一次进行识别。这个功能可以通过onMouse参数中的flag参数实现。


## 二、 图像修复
&#8195;&#8195;OpenCV 中的 `inpaint` 算法用于图像修复（图像去污或恢复损坏区域）。它的作用是通过插值图像中缺失或损坏的部分，使其尽量恢复原始图像的自然效果。典型的应用场景包括**去除水印、文本、划痕或多余的对象，修复老照片等**。其函数原型为：
```python
cv2.inpaint(src, inpaintMask, inpaintRadius, flags)
```
- `src`: 输入图像（可以是灰度图或彩色图像）。
- `inpaintMask`: 掩码图像，和输入图像相同大小的单通道二值图，损坏的部分用白色（255）标记，其他部分为黑色（0）。
- `inpaintRadius`: 修复的半径，即算法考虑的周围像素的距离。
- `flags`: 表示修复算法的类型，分别是：
	-  `cv2.INPAINT_NS` ：Navier-Stokes， 基于流体动力学的方法。模拟流体的运动，计算出损坏区域的“光滑路径”，适用于大面积损坏区域的修复，但可能在某些复杂细节处表现不佳。
	-  `cv2.INPAINT_TELEA`。 (`cv2.INPAINT_NS`)：基于傅里叶变换的传输扩散方法，能够快速扩散邻近区域的信息，效果较好，适合小面积或简单结构的修复。


&#8195;&#8195;掩码是修复算法的关键部分，定义了哪些区域需要修复。可以通过手动绘制或自动生成方法（如图像分割或边缘检测）来创建掩码。


&#8195;&#8195;下面是一个简单的例子，展示如何用 OpenCV 的 `inpaint` 函数去除图像中的污点或遮挡物。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('inpaint.png')
# 掩码图像为二值图，损坏部分为白色(255)
mask = cv2.imread('inpaint_mask.png', 0)  

# 使用inpaint方法修复图像
dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

# 显示原图和修复后的图像
plt.figure(figsize=[16,12]);
plt.subplot(131); plt.imshow(img[:,:,::-1]);plt.title("img_cut1");
plt.subplot(132); plt.imshow(mask,cmap='gray');plt.title("Rect");
plt.subplot(133); plt.imshow(dst[:,:,::-1]);plt.title("img_cut2");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a60e518f11d4656a6d7d17ee1a4da80.png#pic_center)

&#8195;&#8195;实际使用中，mask比较难创建。可以参考交互式grabCut程序中设计思路，通过创建一个类，实现根据用户鼠标擦掉的部分更新mask。

