@[toc]

- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)

## 一、图像直方图及阈值处理
### 1.1 图像直方图基本概念
>参考[《相机直方图：色调和对比度》](https://www.cambridgeincolour.com/tutorials/histograms1.htm)

&#8195;&#8195;**图像直方图**是一种显示图像中像素值分布情况的统计图表。它表示图像中各个像素强度值出现的频率，可以用来分析图像的对比度、亮度、动态范围等特性。直方图的横轴表示像素值（如果是灰度图，0表示最暗，255表示最亮），纵轴表示各像素值的像素数量。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac1368447187478fb61e0102030885ce.png)
&#8195;&#8195;如上图所示，左图水面整体偏亮，此部分对应于图像直方图右侧高亮度区域。右图将其分为上中下三个部分分别进行统计，上部像素分布均匀；中间是水面，像素过于集中；下方整体偏亮。

图像直方图既可以统计灰度图，也可以统计彩色图：
1. **灰度图像直方图**：横轴为0-255的灰度值，纵轴为该灰度值出现的频率。
2. **彩色图像直方图**：对RGB图像，可以为每个通道（红、绿、蓝）绘制单独的直方图，显示各通道像素值的分布。

### 1.2 统计直方图
#### 1.2.1 直接统计
由于图像直方图是统计图像中像素值分布情况，所以可以直接使用`plt.hist`对图像的灰度值进行统计。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度
img = cv2.imread('./lena.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建图形
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121);ax1.imshow(gray, cmap='gray');ax1.axis('off');ax1.set_title('Grayscale Image')
ax2 = plt.subplot(122);ax2.hist(gray.ravel(), 256, [0, 256]);ax2.set_title('Histogram')

# tight_layout自调整子图参数，使之填充整个图像区域，同时确保子图之间的标签和标题不会重叠。
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf96fe37e25d408aa1bbd9aad193d682.png#pic_center =800x)
>使用`plt.subplot`的方式并排显示，由于其默认显示坐标轴，两张图坐标会互相重叠

#### 1.2.2 使用OpenCV统计图像直方图

OpenCV 中可以使用`cv2.calcHist`进行图像直方图计算，其语法为：

```python
calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
```
1. **`images`**：输入图像（列表形式，可以对一批图像进行统计）。即使只传入一张图像，也要放在列表中。
2. **`channels`**：需要计算直方图的通道。对于灰度图只能为`[0]`（单通道）；对于彩色图像， `[0]`、`[1]`、`[2]` 分别表示蓝、绿、红三个通道。
3. **`mask`**：掩膜图像。如果只想计算图像某一部分的直方图，可以传入一个与原图像大小相同的二值掩膜图像，白色部分表示计算区域，黑色部分忽略。若不需要则设 `None`
4. **`histSize`**：直方图的 bins 数量，一般设置为 `[256]`，表示256个灰度值都单独统计。假设设为16，则每15个像素区间统计一次。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be613b3c970e48f5a7af37d30ae79028.png#pic_center =600x)

5. **`ranges`**：统计的像素值范围，一般为 `[0, 256]`。
6. **`accumulate`**：是否累积，默认为`False`。如果对一组图像进行统计，可以设为`True`，表示统计图像时，在上一个直方图的基础上累积结果，而不是从0开始。


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('./lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 统计直方数据
histb = cv2.calcHist([img], [0], None, [256], [0, 255])
histg = cv2.calcHist([img], [1], None, [256], [0, 255])
histr = cv2.calcHist([img], [2], None, [256], [0, 255])

# 创建一个图形窗口，包含两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 在第一个子图中显示原图
ax1.imshow(img_rgb);ax1.set_title('Original Image');ax1.axis('off')  # 不显示坐标轴

# 在第二个子图中绘制直方图
ax2.plot(histb, color='b', label='blue');ax2.plot(histg, color='g', label='green');
ax2.plot(histr, color='r', label='red');ax2.set_title('Histogram using opencv');ax2.legend()

plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c2a444e4260d49fbb837c7f5e97ca04b.png#pic_center =800x)
#### 1.2.3 使用掩膜
&#8195;&#8195;我们可以通过使用掩膜，只统计图中感兴趣的区域。掩膜是与原图像大小相同的二值掩膜图像，只有白色区域会被统计。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a19ec2ce010048d4a048534731497633.png#pic_center)


```python
# 生成灰度图，并创建掩膜
img = cv2.imread('./lena.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros(gray.shape, np.uint8)				# 生成掩膜图像
mask[200:400, 200: 400] = 255						# 直接设置掩码区域

# 生成掩码部分的灰度图
# gray和gray做与运算结果还是gray, 结果再和mask做与运算，黑色部分置0，白色部分不变
gray_mask=cv2.bitwise_and(gray, gray, mask=mask)    

# 对是否使用掩膜进行分别统计
hist_mask = cv2.calcHist([gray], [0], mask, [256], [0, 255])
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 255])


plt.figure(figsize=[10,5])
plt.subplot(121); plt.imshow(gray_mask,cmap='gray'); plt.title("gray_mask");
plt.subplot(122); plt.plot(hist_mask, label='mask');plt.plot(hist_gray, label='gray');plt.legend();
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de46eb2c6b22414e84a0fdd1a70093a2.png#pic_center =800x)
### 1.3 直方图均衡化
&#8195;&#8195;有的时候拍出的图片整体偏亮或偏暗，或者亮度很不均匀。直方图均衡化可以改善图像的对比度。它通过重新分配图像像素的灰度值，使得图像中灰度值的分布更加均匀，从而**增强细节，使图像看起来更清晰**（使灰度值扩展到整个范围，从而增加图像的全局对比度）。

直方图均衡化的实现原理：
1. **计算图像的直方图**： 首先统计出原始图像中每个灰度级（0-255）所出现的频率，即构建图像的直方图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5562de0d158848288c9aee3d432a5d8f.png#pic_center =600x)

    
2. **计算累积直方图**： 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/059f92b7718c43309e2b44407836a39c.png#pic_center =600x)
|
    

  
3. **映射原始像素值**：将累计直方图结果直接乘以255就是最终均衡直方图的结果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73827144c45a43a692cef23fb41c3cfa.png#pic_center =600x)
&#8195;&#8195;在OpenCV中，可以使用`cv2.equalizeHist()`函数来实现直方图均衡化。该函数只适用于灰度图像。下面我们对一张图进行整体增亮和增暗处理，然后进行直方图均衡化看看效果。

```python
img=cv2.imread('lena.png')
matrix = np.ones(img.shape, dtype = "uint8") * 50

img_brighter = cv2.add(img, matrix) 
img_darker   = cv2.subtract(img, matrix)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_darker[:,:,::-1]);  plt.title("Darker");
plt.subplot(132); plt.imshow(img[:,:,::-1]);         plt.title("Original");
plt.subplot(133); plt.imshow(img_brighter[:,:,::-1]);plt.title("Brighter");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d7bc52e4b431431e876140131ddaef5f.png)
&#8195;&#8195;彩色图像有多个通道（如 RGB 或 HSV 颜色空间），直接对每个通道进行直方图均衡化可能会导致颜色失真。因此，通常不会对 RGB 三个通道直接进行均衡化。比较常见的方法是将图像转换到亮度通道可分离的颜色空间（如 YUV 或 HSV），然后只对亮度通道进行直方图均衡化，再将处理后的图像转换回原来的颜色空间。

```python
# 将图像从 BGR 转换到 YUV 颜色空间
yuv_darker=cv2.cvtColor(img_darker, cv2.COLOR_BGR2YUV)
yuv_brighter=cv2.cvtColor(img_brighter, cv2.COLOR_BGR2YUV)

# 对 Y 通道（亮度通道）进行直方图均衡化
yuv_darker[:,:,0] = cv2.equalizeHist(yuv_darker[:,:,0])
yuv_brighter[:,:,0] = cv2.equalizeHist(yuv_brighter[:,:,0])

# 将图像从 YUV 转换回 BGR 颜色空间
darker_equ=cv2.cvtColor(yuv_darker, cv2.COLOR_YUV2BGR)
brighter_equ=cv2.cvtColor(yuv_brighter, cv2.COLOR_YUV2BGR)

# # 显示原图和均衡化后的图像
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(darker_equ[:,:,::-1]);  plt.title("darker_equ");
plt.subplot(132); plt.imshow(img[:,:,::-1]);         plt.title("Original");
plt.subplot(133); plt.imshow(brighter_equ[:,:,::-1]);plt.title("brighter_equ");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2977af77734f4fd78e22a1af33e50b72.png)
### 1.4 自适应直方图均衡化 (CLAHE)
#### 1.4.1 实现原理
直方图均衡化的局限性：

* **噪声增强**：对于含有大量噪声的图像，均衡化可能会使噪声也得到增强，导致图像质量下降。
* **细节丢失**：直方图均衡化是一种全局处理方法，无法处理局部区域对比度问题。如果图像中存在不同亮度的区域，全局均衡化可能会使局部细节丢失。

&#8195;&#8195;针对上述问题，OpenCV提供了自适应直方图均衡化（CLAHE, Contrast Limited Adaptive Histogram Equalization），它通过对图像的局部区域（称为“子图块”）分别进行直方图均衡化，从而增强局部对比度，同时避免过度增强噪声。
1. **将图像分割成多个子图块**： CLAHE将图像划分为多个较小的矩形区域（称为“子图块”或“窗口”，通常是8x8或16x16的网格）。每个子图块会单独进行直方图均衡化，这样可以增强每个局部区域的对比度。
    
2. **对每个子图块进行直方图均衡化**： 在每个子图块上执行和普通直方图均衡化类似的操作，计算该子图块的直方图，然后根据该直方图的累积分布函数 (CDF) 来重新分配像素值。
    
3. **应用对比度限制**： 在局部直方图均衡化时，某些子图块中的像素可能集中在特定的灰度范围内，导致对比度过度增强，尤其是在图像包含噪声时。因此，CLAHE引入了一个对比度限制参数`clipLimit`，用于限制每个灰度级的像素频率。当某个灰度级的频率超过 `clipLimit` 时，多余的部分会均匀分配到其他灰度级。
    
    * **clipLimit**：表示限制直方图中某个灰度级出现的最大频率，防止噪声被过度放大。
4. **插值平滑**： 对于每个像素，由于它位于多个子图块的边界上，CLAHE对这些子图块的均衡化结果进行插值平滑，避免由于直接均衡化子图块而产生块状效应（blocky effect）。通过插值，这些子图块的边界变得平滑，使得过渡更加自然。
    
CLAHE的效果：

* **局部对比度增强**：相比全局直方图均衡化，CLAHE能够有效增强图像中不同区域的对比度，因此在处理具有复杂光照或局部对比度差异大的图像时效果更好。
* **防止过度增强噪声**：由于引入了对比度限制参数，CLAHE可以防止对比度过度增强，从而避免了噪声的放大。
* **适合自然图像**：CLAHE常用于医学图像、卫星图像和低光照图像的处理，这些图像通常需要增强局部区域的对比度，而不希望整体图像变得太过刺眼。

| **属性** | **普通直方图均衡化** | **CLAHE（自适应直方图均衡化）** |
| --- | --- | --- |
| **处理范围** | 全局 | 局部，分块处理 |
| **效果** | 提高全局对比度，可能导致局部细节丢失 | 提高局部对比度，增强细节 |
| **噪声处理** | 可能过度增强噪声 | 使用clipLimit限制对比度增强，避免噪声过度增强 |
| **适用场景** | 适用于灰度值集中分布的图像，全局对比度不高 | 适用于包含复杂光照或局部对比度差异大的图像（如医学、卫星图像） |
| **常见问题** | 对局部细节处理不佳，可能丢失对比度 | 使用不当时，可能引入分块效应，不过插值技术可以有效减缓 |
#### 1.4.2 代码实现
&#8195;&#8195; OpenCV 中使用`cv2.createCLAHE` 函数进行自适应直方图均衡化，它生成一个 CLAHE 对象，可以通过该对象对图像应用自适应直方图均衡化。

```python
createCLAHE([, clipLimit[, tileGridSize]]) -> retval
```

1. **`clipLimit`**：对比度限制阈值，浮点型，默认为`2.0`。`clipLimit` 限制了每个灰度级像素频率的最大值，超过 `clipLimit` 的频率会被平摊到其他灰度级，从而避免过度增强局部噪声。
	* 如果 `clipLimit` 值较低，对比度增强较弱。
	* 如果 `clipLimit` 值较高，则会增强对比度。

2. **`tileGridSize`**：整型元组，表示子图块的大小。默认为`(8, 8)`，即将图像分为 8×8 个子图块。对每个子图块单独进行直方图均衡化，然后在子图块之间进行插值以避免边界出现突变现象。
	* **值越大**：处理的大块区域更多，图像整体的对比度调整幅度更大，但局部细节增强不明显。
	* **值越小**：处理的小块区域更多，图像局部对比度更强，但可能会引入噪声和块效应。

        
```python
# 将图像从 BGR 转换到 YUV 颜色空间
yuv_darker=cv2.cvtColor(img_darker, cv2.COLOR_BGR2YUV)
yuv_brighter=cv2.cvtColor(img_brighter, cv2.COLOR_BGR2YUV)

# 创建CLAHE对象，设定clipLimit和tileGridSize
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
# 对 Y 通道（亮度通道）进行直方图均衡化
yuv_darker[:,:,0] = clahe.apply(yuv_darker[:,:,0])
yuv_brighter[:,:,0] = clahe.apply(yuv_brighter[:,:,0])

# 将图像从 YUV 转换回 BGR 颜色空间
darker_equ=cv2.cvtColor(yuv_darker, cv2.COLOR_YUV2BGR)
brighter_equ=cv2.cvtColor(yuv_brighter, cv2.COLOR_YUV2BGR)

# # 显示原图和均衡化后的图像
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(darker_equ[:,:,::-1]);  plt.title("darker_equ");
plt.subplot(132); plt.imshow(img[:,:,::-1]);         plt.title("Original");
plt.subplot(133); plt.imshow(brighter_equ[:,:,::-1]);plt.title("brighter_equ");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/986f3fe950e24fb9a1e737332f8ba9d5.png)
### 1.5 阈值处理
&#8195;&#8195;阈值处理的主要意义是将图像中的某些区域分离出来，通常是为了突出前景（如物体）和背景（如场景）。通过二值化，可以将灰度图像转化为黑白图像（即二值图像），使后续的图像分析、边缘检测、目标识别等任务更加简便和高效。

&#8195;&#8195;例如，图像由暗色背景上的亮目标组成，这时可以通过设定适当的阈值 T，将图像的像素划分为两类：灰度值大于 T 的像素集是目标，小于 T 的像素集是背景。当 T 是应用于整幅图像的常数，称为<font color='deeppink'>全局阈值处理</font >；当 T 对于整幅图像发生变化时，称为<font color='deeppink'>可变阈值处理</font >。有时，对应于图像中任一点的 T 值取决于该点的邻域的限制，称为<font color='deeppink'>局部阈值处理</font >。
#### 1.5.1 全局阈值处理
&#8195;&#8195;全局阈值处理（Global Thresholding）是对图像的所有像素点应用同一个阈值。如果像素值高于阈值，则将其设为一个值（通常是白色），否则设为另一个值（通常是黑色）。OpenCV 提供了函数 `cv2.threshold` 函数来实现此功能，其语法为：

```python
retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
```
- `retval`：阈值，浮点型
- `dst`：阈值处理后的图像（numpy数组），与src具有相同大小和类型以及通道数。
- `src`：输入数组，最好是灰度图。
- `thresh`：阈值。

- `maxval`：用于`THRESH_BINARY`和`THRESH_MINARY_INV`阈值类型的最大值，一般取 255。
- `type`：阈值类型（详见[阈值类型](https://docs.opencv.org/4.5.1/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)）。 

| **Type 类型** | **描述** |
| --- | --- |
| `cv2.THRESH_BINARY`（输出二值图像） | 超过阈值的像素值设为`maxValue`，否则设为0 |
| `cv2.THRESH_BINARY_INV`（输出二值图像） | 超过阈值的像素值设为0，否则设为`maxValue` |
| `cv2.THRESH_TRUNC` | 超过阈值时置为阈值 `thresh`，否则不变 |
| `cv2.THRESH_TOZERO` | 超过阈值的像素值保持不变，否则置0 |
| `cv2.THRESH_TOZERO_INV` | 超过阈值的像素值设为0，否则不变 |
| `cv2.THRESH_OTSU` | 使用 OTSU 算法选择阈值，需要与其他类型（如`cv2.THRESH_BINARY`）结合使用。 |
`cv2.THRESH_TRIANGLE`|	使用三角算法自动计算阈值，需要与其他类型结合使用。|

&#8195;&#8195;特殊值`THRESH_OTSU`或`THRESH_TRIANGLE`可以与上述值之一组合。在这些情况下，函数使用Otsu或Triangle算法确定最佳阈值，并使用它代替指定的阈值。Otsu和Triangle方法仅用于8位单通道图像。

&#8195;&#8195;当图像中存在高斯噪声时，通常难以通过全局阈值将图像的边界完全分开。如果图像的边界是在局部对比下出现的，不同位置的阈值也不同，使用全局阈值的效果将会很差。如果图像的直方图存在明显边界，容易找到图像的分割阈值；但如果图像直方图分界不明显，则很难找到合适的阈值，甚至可能无法找到固定的阈值有效地分割图像。
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img_read = cv2.imread("building-windows.jpg", 0) # 灰度图
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(121); plt.imshow(img_read, cmap="gray");         plt.title("Original");
plt.subplot(122); plt.imshow(img_thresh, cmap="gray");       plt.title("Thresholded");

print(retval,img_thresh.shape)
```

```python
(572, 800) 100.0 (572, 800)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9f2936d32af116d3b738a2d9aab3e0e5.png)

#### 1.5.2 全局阈值处理之Otsu's 阈值法
&#8195;&#8195;当阈值范围无法人工确定时, 可以使用Otsu's方法（又称大津算法）自动计算全局阈值。Otsu适用于图片的灰度直方图是双峰结构的图形。

&#8195;&#8195;Otsu's方法使用最大化类间方差（intra-class variance）作为评价准则，基于对图像直方图的计算，可以给出类间最优分离的最优阈值。

&#8195;&#8195;任取一个灰度值 T，可以将图像分割为两个集合 F 和 B，集合 F、B 的像素数的占比分别为 pF、pB，集合 F、B 的灰度值均值分别为 mF、mB，图像灰度值为 m，定义类间方差为：
$$ ICV = p_F * (m_F - m)^2 + p_B * (m_B - m)^2$$
&#8195;&#8195;使类间方差 ICV 最大化的灰度值 T 就是最优阈值。因此，只要遍历所有的灰度值，就可以得到使 ICV 最大的最优阈值 T。
```python
img = cv2.imread("../images/Fig1039a.tif", flags=0)

deltaT = 1  # 预定义值
histCV = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
grayScale = range(256)  # 灰度级 [0,255]
totalPixels = img.shape[0] * img.shape[1]  # 像素总数
totalGray = np.dot(histCV[:,0], grayScale)  # 内积, 总和灰度值
T = round(totalGray/totalPixels)  # 平均灰度
while True:
    numC1, sumC1 = 0, 0
    for i in range(T): # 计算 C1: (0,T) 平均灰度
        numC1 += histCV[i,0]  # C1 像素数量
        sumC1 += histCV[i,0] * i  # C1 灰度值总和
    numC2, sumC2 = (totalPixels-numC1), (totalGray-sumC1)  # C2 像素数量, 灰度值总和
    T1 = round(sumC1/numC1)  # C1 平均灰度
    T2 = round(sumC2/numC2)  # C2 平均灰度
    Tnew = round((T1+T2)/2)  # 计算新的阈值
    print("T={}, m1={}, m2={}, Tnew={}".format(T, T1, T2, Tnew))
    if abs(T-Tnew) < deltaT:  # 等价于 T==Tnew
        break
    else:
        T = Tnew

# 阈值处理
ret1, imgBin = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)  # 阈值分割, thresh=T
ret2, imgOtsu = cv2.threshold(img, T, 255, cv2.THRESH_OTSU)  # 阈值分割, thresh=T
print(ret1, ret2)

plt.figure(figsize=(7,7))
plt.subplot(221), plt.axis('off'), plt.title("Origin"), plt.imshow(img, 'gray')
plt.subplot(222, yticks=[]), plt.title("Gray Hist")  # 直方图
histNP, bins = np.histogram(img.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])
plt.subplot(223), plt.title("global binary(T={})".format(T)), plt.axis('off')
plt.imshow(imgBin, 'gray')
plt.subplot(224), plt.title("OTSU binary(T={})".format(round(ret2))), plt.axis('off')
plt.imshow(imgOtsu, 'gray')
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/76f74a892528ed0b17301bf58fad5a62.png#pic_center =600x)

简单使用就是：

```python
import cv2 
import numpy as np
import matplotlib.pyplot as plt

naza = cv2.imread('naza.png')
naza_gray = cv2.cvtColor(naza, cv2.COLOR_BGR2GRAY)
hist = plt.hist(naza_gray.ravel(), bins=256, range=[0, 255])
histCV = cv2.calcHist([naza], [0], None, [256], [0, 256])

# 普通阈值处理
ret1, dst1 = cv2.threshold(naza_gray, 80, 255, cv2.THRESH_BINARY)

# ostu阈值处理
ret2, dst2 = cv2.threshold(naza_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


plt.figure(figsize=(12,6))

plt.subplot(131);plt.imshow(naza[:,:,::-1]),plt.axis('off');plt.title('naza');
plt.subplot(132);plt.imshow(dst1,cmap='gray'),plt.axis('off');plt.title('normal');
plt.subplot(133);plt.imshow(dst2,cmap='gray'),plt.axis('off');plt.title('ostu');
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/138ea6fb49374767ba61a3e8d406020d.png#pic_center =400x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be503a92c6244b69b98717e991f3a32a.png#pic_center =800x)

&#8195;&#8195;全局阈值处理还有一些其它改进方法，比如[处理前先对图像进行平滑](https://blog.csdn.net/youcans/article/details/124281345)、[基于边缘信息改进全局阈值处理](https://blog.csdn.net/youcans/article/details/124281390)等等。

#### 1.5.3 自适应阈值处理
&#8195;&#8195;噪声和非均匀光照等因素对阈值处理的影响很大，例如光照复杂时 全局阈值分割方法的效果往往不太理想，需要使用可变阈值处理。

&#8195;&#8195;自适应阈值处理（Adaptive Thresholding）**对图像中的每个点，根据其邻域计算其对应的阈值**，非常适合处理光照条件不均匀的图像。`cv2.adaptiveThreshold`函数语法为：

```python
adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
```
- `maxValue`：为满足条件的像素指定的非零值，详见[阈值类型](https://docs.opencv.org/4.5.1/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)说明。
- `adaptiveMethod`：要使用的自适应阈值算法，详见[AdaptiveThresholdTypes](https://docs.opencv.org/4.5.1/d7/d1b/group__imgproc__misc.html#gaa42a3e6ef26247da787bf34030ed772c)。
	- `cv.ADAPTIVE_THRESH_MEAN_C`：阈值是邻域的均值；
	- `cv.ADAPTIVE_THRESH_GAUSSIAN_C`：阈值是邻域的高斯核加权平均值；
- `thresholdType`：阈值类型，只有两种
	- `cv2.THRESH_BINARY`：大于阈值时置 maxValue，否则置 0
	- `cv2.THRESH_BINARY_INV`：大于阈值时置 0，否则置 maxValue
- `blockSize`：用于计算像素阈值的像素邻域的尺寸，例如3、5、7。
- `C`： 偏移量，从平均值或加权平均值中减去该常数。


&#8195;&#8195;假设您想构建一个可以读取（解码）乐谱的应用程序，这类似于文本文档的光学字符识别（OCR）。处理管道的第一步是隔离文档图像中的重要信息（将其与背景分离）。这项任务可以通过[阈值技术](https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html)来完成。
```python
# 示例：乐谱阅读器

img_read = cv2.imread("Piano_Sheet_Music.png", 0)

# 全局阈值1
retval, img_thresh_gbl_1 = cv2.threshold(img_read,50, 255, cv2.THRESH_BINARY)

# 全局阈值2
retval, img_thresh_gbl_2 = cv2.threshold(img_read,130, 255, cv2.THRESH_BINARY)

# 自适应阈值
img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

# Show the images
plt.figure(figsize=[18,15])
plt.subplot(221); plt.imshow(img_read,        cmap="gray");  plt.title("Original");
plt.subplot(222); plt.imshow(img_thresh_gbl_1,cmap="gray");  plt.title("Thresholded (global: 50)");
plt.subplot(223); plt.imshow(img_thresh_gbl_2,cmap="gray");  plt.title("Thresholded (global: 130)");
plt.subplot(224); plt.imshow(img_thresh_adp,  cmap="gray");  plt.title("Thresholded (adaptive)");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/71e840eb06bc38a1e876da2d2734f946.png)

## 二、图像轮廓
### 2.1 轮廓的查找与绘制
&#8195;&#8195;轮廓可以看作是具有相同强度或颜色的所有连续点的边界，通常在处理二值图像时使用。通过轮廓，图像中的物体形状和结构可以被有效提取，这在对象检测、识别和分析中非常有用。

`cv2.findContours` 是 OpenCV 中用于检测图像中轮廓的函数，其语法为：

```python
findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
```

* **`image`**：输入图像，通常是二值图像（黑白图像）。可以使用 `cv2.threshold` 或 `cv2.Canny` 将图像转换为二值图像。
    
* **`mode`**：轮廓检索模式，决定如何检索轮廓。常见的模式有：    
    * `cv2.RETR_EXTERNAL`：只检测最外层轮廓。
    * `cv2.RETR_TREE`：按照树型检测所有轮廓， 从里到外，从右到左
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1a6924c56c0467fb8d1907b9317c768.png)


* **`method`**：轮廓近似方法，决定如何处理轮廓点。常见的方法有：    
    * `cv2.CHAIN_APPROX_NONE`：存储所有轮廓点，但这通常是没必要的，会产生很多冗余。
    * `cv2.CHAIN_APPROX_SIMPLE`：常用，只保留轮廓的关键拐点。


函数最终返回两个值：

* **`contours`**：轮廓点列表。列表中每个元素是一个 ndarray 数组，表示一个轮廓（轮廓上所有点的坐标）。
* **`hierarchy`**：层级信息。对于每个轮廓，存储其父轮廓、子轮廓、下一轮廓和前一轮廓的索引。

&#8195;&#8195;轮廓查找完之后，返回的只是轮廓点的坐标信息。我们可以使用`cv2.drawContours`函数，将轮廓绘制出来，其语法为：

```python
drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
```
- **image**：将要绘制轮廓的图像，会被直接修改，可以考虑拷贝一份来绘制。
- **contours**：轮廓列表，每个轮廓都是一个`numpy`数组，表示轮廓上的点。
- **contourIdx**： 轮廓的索引，如果设置为负数，所有的轮廓都会被绘制。
- **color**： 轮廓线的颜色，用`(B, G, R)`元组表示。
- **thickness**： 轮廓线的厚度。如果为负数，轮廓内部会被填充指定的颜色。
- **lineType**：轮廓线类型，默认是`cv2.LINE_8`。其他选项包括`cv2.LINE_4`、`cv2.LINE_AA`等。
- **hierarchy**: 轮廓的层次结构信息，只有在绘制轮廓的子集时才需要。
- **maxLevel**: 绘制轮廓的最大级别。如果为0，只绘制指定的轮廓；如果为1，绘制轮廓及其子轮廓；以此类推。
- **offset**: 轮廓的偏移量，所有的轮廓都会按照这个偏移量进行移动。


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 原图是一个3通道彩色图，但显示出来是黑白图。
img = cv2.imread('./contours1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用二值化处理
thresh, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) 

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓会直接修改原图，如果想保持原图不变, 建议copy一份
img_copy = img.copy()
# -1表示绘制所有轮廓，2为轮廓线厚度
cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 2)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img[:,:,::-1]);  plt.title("img");
plt.subplot(122); plt.imshow(img_copy[:,:,::-1]);  plt.title("img_copy");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fef7751ef0c44f37839cf3f2e20fd116.png#pic_center =600x)

### 2.2 计算轮廓面积和周长

&#8195;&#8195;轮廓面积是指每个轮廓中所有的像素点围成区域的面积，单位为像素。使用 `cv2.contourArea()` 函数可以计算轮廓的面积，其语法为：

```python
contourArea(contour[, oriented]) -> retval
```

&#8195;&#8195;使用 `cv2.arcLength()` 函数可以计算轮廓的周长，其语法为：
```python
arcLength(curve, closed) -> retval
```
- **`curve`**：轮廓，一般是用`findContours` 函数返回的轮廓列表中的一个轮廓
- **`closed`**：布尔值，如果为 `True`，表示轮廓是封闭的，计算周长；如果为 `False`，表示轮廓是开放的，计算曲线长度。

轮廓面积和周长有多种应用：
1. **物体大小分析**：通过计算面积，可以比较不同物体的大小。例如，机器人视觉可以通过面积判断不同物体的大小，从而做出选择和处理；
2. **形状特征提取**：结合面积和周长可以分析物体形状。例如，通过周长和面积的比率，可以判断轮廓是接近圆形、方形还是其他形状，以便检测图像中的指定形状的物体；
3. **形状筛选**：在特定场景中，可能需要过滤掉面积或周长过小或过大的轮廓。例如，在车牌识别中，可以通过设定面积阈值只保留符合条件的轮廓；
4. **物体检测与分类**：在图像中通过轮廓的面积来分类不同类型的物体。例如，根据物体的大小将它们分为大、中、小三类。
5.  **过滤噪声**：在物体检测任务中，可能会检测到一些非常小的噪声点，可以通过面积筛选将它们过滤掉。



下面是车牌识别中，轮廓面积的简单应用示例：
```python
import cv2
import numpy as np

# 读取图像并转换为灰度图像
image = cv2.imread('car.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊，减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 进行轮廓检测,返回轮廓列表及其索引
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 设定车牌的面积阈值
min_area = 1000    # 车牌的最小面积
max_area = 15000   # 车牌的最大面积

# 遍历所有轮廓，筛选符合面积条件的轮廓
for contour in contours:
    area = cv2.contourArea(contour)
    
    # 过滤掉不在面积范围内的轮廓
    if min_area < area < max_area:
        # 在图像上绘制轮廓
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        
        # 计算轮廓的边界框（矩形）
        x, y, w, h = cv2.boundingRect(contour)
        
        # 提取轮廓对应的区域并显示
        plate_region = image[y:y+h, x:x+w]
        cv2.imshow("Plate Region", plate_region)

# 显示最终筛选后的图像
cv2.imshow("Filtered Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.3 多边形近似

&#8195;&#8195;`findContours`找到的轮廓比较精细，有时候我们只想得到一个大致的轮廓。`cv2.approxPolyDP`是 OpenCV 中用于轮廓近似的算法，可以对找出的轮廓进行多边形近似，来简化轮廓。

&#8195;&#8195;`cv2.approxPolyDP`的实现是基于Douglas-Peucker 算法，其原理如下（详见[《DP算法——道格拉斯-普克 Douglas-Peuker》](https://zhuanlan.zhihu.com/p/438689157)）：
1. **初始设定**：选择轮廓的两个端点作为多边形的起点和终点。
2. **寻找最大距离点**：在轮廓中找到离这条线段距离最远的点，如果该距离大于设定的阈值 `epsilon`，则保留该点。
3. **递归处理**：将轮廓分成两个子段，分别递归执行该过程，直到所有剩余点的距离小于 `epsilon`，最终形成近似的多边形。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7079cf5203dd4989984ca22d39dbbb0e.png#pic_center =400x)



```python
approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve
```
- `curve`：要简化的轮廓
- `epsilon` ：DP算法使用的阈值，阈值越大精度越低，保留的轮廓点数越少
- `closed`：布尔值，指示轮廓是否封闭。如果为 True，输出的近似轮廓是封闭的。

### 2.4 凸包
&#8195;&#8195;逼近多边形是轮廓的高度近似，但是有时候，我们希望使用一个多边形的凸包来进一步简化它。


&#8195;&#8195;凸包是包含给定点集的最小凸多边形。换句话说，它是能够包围所有给定点的最小凸形状。凸包的每一处都是凸的，即在凸包内连接任意两点的直线都在凸包的内部。

&#8195;&#8195;`cv2.convexHull`是OpenCV库中用于计算凸包(convex hull)的函数，其语法是：

```python
convexHull(points[, hull[, clockwise[, returnPoints]]]) -> hull
```
- `points`：要简化的轮廓
- `colckwise`：方向标志，默认为False。如果为True，输出的凸包为顺时针方向。
- `returnPoints`：默认为True，表示返回凸包顶点的坐标，否则只返回凸包顶点的索引。

下面进行多边形近似和凸包的演示：

```python
import cv2
import numpy as np


img = cv2.imread('./hand.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,binary= cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 查找轮廓并画出，contours[0]是手的轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
cv2.drawContours(img_contours, contours, 0, (0, 0, 255), 2)

# 进行多边形逼近, 返回的是多边形上一系列的点, 即多边形逼近之后的轮廓
# 凸包和多边形都可以使用drawContours函数绘制，只是其接受的是轮廓列表格式
approx = cv2.approxPolyDP(contours[0], 20, True)
img_approx=img.copy()
cv2.drawContours(img_approx, [approx], 0, (0, 255, 0), 2)

# 计算凸包
hull = cv2.convexHull(contours[0])
img_hull=img.copy()
cv2.drawContours(img_hull, [hull], 0, (255, 0, 0), 2)

plt.figure(figsize=[16,8])
plt.subplot(141); plt.imshow(img[:,:,::-1]);  plt.title("img");
plt.subplot(142); plt.imshow(img_contours[:,:,::-1]);  plt.title("img_contours");
plt.subplot(143); plt.imshow(img_approx[:,:,::-1]);  plt.title("img_approx");
plt.subplot(144); plt.imshow(img_hull[:,:,::-1]);  plt.title("img_hull");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12eb0f050f514de2be4f5e64a759ec20.png)
### 2.5 外接矩形和外接圆
关于轮廓还有一些其它的操作，比如最小外接矩阵、最大外接矩阵和最小外接圆。

```python
cv2.minAreaRect(points) -> retval
```
- `points`：轮廓
- 返回一个元组 `(center(x, y), (width, height), angle)`，表示最小外接矩形的 中心点坐标，高宽，以及矩形相对于水平轴的旋转角度。

>&#8195;&#8195;`cv2.minAreaRect`返回的结果是外接矩形的中心点坐标、高宽以及旋转角度，可以使用opencv提供的`cv2.boxPoints`函数，自动计算出矩形的四个角点坐标，也就得到了轮廓数据。然后就可以使用`cv2.drawContours`将其标记出来。

```python
cv2.boundingRect(array) -> retval
```
- `array`：轮廓
- 返回一个元组`(center(x, y), (width, height))`。最大外接矩形一定是水平的，所以没有旋转角度，所以可以直接用画矩形的函数在图像上`cv2.rectangle`画出来。

```python
minEnclosingCircle(points) -> center, radius
```
下面进行简单的演示：
```python
img = cv2.imread('./hello.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contours,_= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 最外面的轮廓是整个图像, contours[1]才是Hello语的轮廓
# rect是一个元组，包括(x, y), (w, h), angle
rect = cv2.minAreaRect(contours[1])
# 快速把rect转化为轮廓数据，得到的结果是浮点类型,要转为整型
box = cv2.boxPoints(rect)
box = np.round(box).astype('int64')

# 绘制最小外接矩形
img1=img.copy()
cv2.drawContours(img1, [box], 0, (255, 0, 0), thickness=2)

# 绘制最大外接矩形
x,y, w, h = cv2.boundingRect(contours[1])
img2=img.copy()
cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# 绘制最小外接圆,返回的也是浮点类型
center, radius=cv2.minEnclosingCircle(contours[1])
center, radius= np.round(center).astype('int64'),np.round(radius).astype('int64')
img3=img.copy()
cv2.circle(img3,center,radius,(0, 0, 255),thickness=2) 

plt.figure(figsize=[16,8])
plt.subplot(141); plt.imshow(img[:,:,::-1]);  plt.title("img");
plt.subplot(142); plt.imshow(img1[:,:,::-1]);  plt.title("minAreaRect");
plt.subplot(143); plt.imshow(img2[:,:,::-1]);  plt.title("boundingRect");
plt.subplot(144); plt.imshow(img3[:,:,::-1]);  plt.title("minEnclosingCircle");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9cf51095cf5c4f6591600503987a9869.png)
## 三、形态学
&#8195;&#8195;形态学（`Morphology`）是指一系列用于处理图像形状和结构的算法，其基本思想是利用一种特殊的结构元(本质上就是卷积核)来测量或提取输入图像中相应的形状或特征。形态学操作通常用于预处理、图像分割、特征提取、图像滤波和图像增强等任务。形态学的基本操作包括：
1. 腐蚀（`Erosion`）：它将图像中的前景物体缩小，这种操作可以去除图像中的小物体，分离相互接触的物体，以及平滑物体的边界。
2. 膨胀（`Dilation`）：与腐蚀相反，膨胀操作将图像中的前景物体增大，可以用来填补物体中的小洞，连接相邻的物体，或者增加物体的面积。
3. 开运算（`Opening`）：先腐蚀后膨胀的过程，用于去除小的物体，平滑较大物体的边界，而不改变其面积。
4. 闭运算（`Closing`）：先膨胀后腐蚀的过程，用于填充物体内的小洞，连接邻近的物体，而不明显改变物体的边界。
5. 形态学梯度（Morphological Gradient）：膨胀图与腐蚀图之差，可以突出物体的边缘。
6. 顶帽（`Top Hat`）和黑帽（`Black Hat`）：这两种操作分别是原图与开运算结果之差（顶帽）和闭运算结果与原图之差（黑帽），用于突出比周围区域亮或暗的区域。



### 3.1 腐蚀与膨胀
- **腐蚀**用于消除小的白色噪声，减小前景区域。
- **膨胀**用于填补物体中的空洞，增加前景区域。
#### 3.1.1 腐蚀操作

&#8195;&#8195;**腐蚀**是一种将前景（白色区域）缩小的操作，其原理是滑动窗口中的结构元素（kernel），如果卷积区域内所有被覆盖的像素都是前景像素（白色），中心像素保留为前景（白色），否则变为背景（黑色）。这使得前景物体逐渐缩小，细小的噪声点会被消除。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/72e45ace940f47ef8c65deaadd1d93da.png#pic_center =400x)
&#8195;&#8195;如上图所示，腐蚀操作的卷积核设为5×5，只有图中虚线方框内的像素，被卷积时区域内都是白色像素，所以卷积后也是白色（设为255）。其它区域都将被置为黑色（设为0）。


腐蚀操作使用`erode`函数，其语法为：


```python
erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
```
- `src`: 输入图像，一般是二值图像。
- `kernel`: 卷积核（即结构元素）。
	- 核越大，腐蚀的效果越强。
	- 不同的形状会影响腐蚀的方向性和图像特征的保留。矩形核适合均匀腐蚀，而椭圆核能更好地保留圆滑的边缘。
- `iterations`: 腐蚀操作的迭代次数，默认为1。次数越多，腐蚀效果越明显。


&#8195;&#8195;下面是一个简单的示例，处理之后，白色的字体像是被橡皮擦去了一圈，变小了。
```python
img = cv2.imread('msb.png')

# 定义核
kernel = np.ones((5, 5), np.uint8)
dst = cv2.erode(img, kernel, iterations=1)

plt.figure(figsize=[18,15])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("dst");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/145b407909a1471d97f75e9d75f6e531.png#pic_center =800x)

#### 3.1.2 创建形态学卷积核
&#8195;&#8195;`cv2.getStructuringElement` 是 OpenCV 中用于生成 **结构元素**（也叫形态学核）的函数，常用于形态学操作（如腐蚀、膨胀、开运算、闭运算等）。结构元素决定了形态学操作（卷积核）的形状和尺寸。

```python
getStructuringElement(shape, ksize[, anchor]) -> retval
```
- **`shape`**：指定结构元素（卷积核）的形状，常见的形状有：
  - `cv2.MORPH_RECT`：矩形
  - `cv2.MORPH_ELLIPSE`：椭圆形
  - `cv2.MORPH_CROSS`：十字形
  
- **`ksize`**：结构元素的大小，通常以 `(width, height)` 的形式给出。例如 `(5, 5)` 表示 5x5 的结构元素。

- **`anchor`**（可选）：结构元素的锚点，表示结构元素的参考中心点。默认是结构元素的中心 `(ksize[0]//2, ksize[1]//2)`，但也可以指定其他锚点。

```python
kernel_RECT = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ELLIPSE=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_CROSS=cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

kernel_RECT,kernel_ELLIPSE,kernel_CROSS
```

```python
 array([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]], dtype=uint8
```

```python
 array([[0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0]], dtype=uint8)
```

```python
 array([[0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]], dtype=uint8)
```

#### 3.1.3 膨胀操作

&#8195;&#8195;**膨胀**（Dilation）是一种将前景（白色区域）扩大的操作。膨胀的原理与腐蚀相反，只要滑动窗口中的结构元素覆盖下有一个像素是前景像素(白色），中心像素就保留为前景。这可以使前景区域扩大，填补物体中的小孔，并连接分离的小物体，其函数语法为：


```python
dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
```

- `src`：输入图像，一般是二值图像。
- `kernel`：卷积核，定义操作的结构元素。
- `iterations`： 膨胀操作的迭代次数，默认为1。

```python
img = cv2.imread('./j.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.dilate(img, kernel, iterations=1)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("dst");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/efbed556b3d74b33a35502fcbbfc95cb.png#pic_center =600x)


### 3.2 开运算和闭运算
&#8195;&#8195;`cv2.morphologyEx` 是 OpenCV 中一个用于执行更复杂的形态学操作的函数，它基于基础的腐蚀和膨胀操作，并提供了一系列的高级形态学变换。通过这个函数，我们可以实现诸如开运算、闭运算、形态学梯度、顶帽操作和黑帽操作等操作。

| 操作类型   | 操作顺序          | 用途                                   |
|------------|-------------------|----------------------------------------|
| **开运算** | 先腐蚀，后膨胀    | 消除小的噪声点，保留前景物体的整体形状   |
| **闭运算** | 先膨胀，后腐蚀    | 填补前景物体中的小孔，连接分散的小区域 |
| **梯度**   | 膨胀与腐蚀的差    | 提取物体的边缘，提取图像中的轮廓                         |
| **顶帽**   | 输入图像 - 开运算  | 提取前景外的亮区域，常用于不均匀光照的图像处理中  |
| **黑帽**   | 输入图像 - 闭运算 | 提取前景中的暗区域，适合分析背景中的暗部特征 |

`cv2.morphologyEx` 语法为：

```python
morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
```

* **`src`**：输入图像，通常是二值图像（黑白图像）。
* **`op`**：表示要执行的形态学操作。常见的操作包括：
    * `cv2.MORPH_OPEN`：开运算。先腐蚀，后膨胀 ，在消除噪声的同时保持前景部分不变。
    * `cv2.MORPH_CLOSE`：闭运算。先膨胀扩大前景，再腐蚀，擦掉前景中的黑色部分。
    * `cv2.MORPH_GRADIENT`：形态学梯度；
    * `cv2.MORPH_TOPHAT`：顶帽操作；
    * `cv2.MORPH_BLACKHAT`：黑帽操作；
* **`kernel`**：结构元素（卷积核），通常由 `cv2.getStructuringElement()` 生成（包括形状和大小）。
* **`iterations`**：迭代次数，默认为 1。


```python
# 开运算，先腐蚀后膨胀，前景保持不变

img = cv2.imread('./dotj.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 直接调用cv2.morphologyEx更方便
# dst = cv2.erode(img, kernel, iterations=2)
# dst = cv2.dilate(dst, kernel, iterations=2)
dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("open");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7fc904a7295b47d78aa741de16625fa4.png#pic_center =600x)

```python
# 闭运算

img = cv2.imread('dotinj.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("close");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/731a815fa58e4941a15540775c4bb652.png#pic_center =600x)
### 3.3 形态学梯度
&#8195;&#8195;**形态学梯度 = 原图 - 腐蚀**，也就是得到被腐蚀掉的部分。这会突出显示物体的边缘，生成的是前景物体的轮廓。

```python
img = cv2.imread('./j.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=1)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("GRADIENT");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4161466aaa949759d644ec2a303e12a.png#pic_center =600x)
### 3.4 顶帽操作(tophat)
&#8195;&#8195;**顶帽 = 原图 - 开运算**。开运算的效果是去除图形外的噪点,，原图 - 开运算就**得到了图形外的噪点**，可以用于突出显示图像中的亮区域。

```python
import cv2 
import numpy as np

img = cv2.imread('./dotj.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=2)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("TOPHAT");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/343fbac6d29843febf0868d55b03a784.png#pic_center =600x)
### 3.5 黑帽操作（Black Hat）
&#8195;&#8195;**黑帽 = 原图 - 闭运算**。闭运算可以将图形内部的噪点去掉，那么原图 - 闭运算的结果就是**图形内部的噪点**，用于突出显示图像中的暗区域。

```python
img = cv2.imread('./dotinj.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=2)

plt.figure(figsize=[8,4])
plt.subplot(121); plt.imshow(img,cmap="gray");  plt.title("img");
plt.subplot(122); plt.imshow(dst,cmap="gray");  plt.title("BLACKHAT");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24eaa4d456494a78b6184889286c5919.png#pic_center =600x)
## 四、车辆统计项目
下面是项目要实现的效果图：

![](https://img-blog.csdnimg.cn/img_convert/a362345659a62cf02db20c451e1980a2.png#pic_center =600x)
### 4.1 背景减除算法
&#8195;&#8195;**Background Subtraction**（背景减除）是一种在计算机视觉中广泛使用的技术，主要用于静态相机场景下的前景检测。通过从当前帧中减去背景，背景减除可以有效地检测出移动的物体（如行人、车辆等），因此非常适合用于视频监控、智慧交通、运动分析等场景，监测人、车的流量、轨迹等。


&#8195;&#8195;背景减除的基本思想是将视频序列中的每一帧与背景模型进行比较，找出前景区域。实现这一点的关键在于建立并更新背景模型。算法通常分为三个步骤：

1. **背景模型初始化**：通过一系列图像帧建立初始的背景模型。
2. **前景检测**：当前帧与背景模型进行差分，差异较大的像素点被视为前景。
3. **背景模型更新**：将当前帧的部分信息更新到背景模型中，以适应场景中的变化（如光照变化）。


&#8195;&#8195;OpenCV 提供了多种背景减除算法，最终返回的是一个包含前景的掩模，移动的物体会被标记为白色，背景会被标记为黑色。
| **背景减除算法** | **主要原理** | **优点** | **缺点** | **典型应用场景** |
| --- | --- | --- | --- | --- |
| `MOG2` | 一种参数建模方法，利用高斯混合概率密度函数实现高效自适应算法，能够更好地处理随时间变化的背景以及具有多种颜色和纹理的复杂背景。 | 适应光照变化，支持阴影检测，适合动态背景 | 会产生很多细小的噪点。对剧烈的光照变化、复杂场景效果有限 | 视频监控、交通监控、动态背景场景 |
| `KNN` | 一种非参数建模方法，使用K近邻技术，利用历史帧信息判断背景和前景 | 对复杂背景和光线变化适应性强，适合动态环境 | 阴影处理能力弱，参数选择敏感 | 复杂环境的前景检测 |
| `GMG` | 一种结合了统计学建模和贝叶斯估计的背景减除算法。它通过短期的前景帧进行背景更新，并使用贝叶斯估计来获得最有可能的背景模型。 | 适应剧烈变化的环境，快速检测新运动物体 | 需要初始化时间较长，处理复杂场景效果不稳定 | 运动分析、场景剧烈变化的场合 |


1. **MOG2背景减除算法**函数解析：
        
      ```python
      cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
      ```
      
      * `history`: 用于背景模型的训练帧数，越大背景模型越稳健。
      * `varThreshold`: 决定是否将一个像素标记为前景的阈值。
      * `detectShadows`: 是否启用阴影检测功能（True 时前景中的阴影部分会被标记为灰色区域）。
2. **KNN背景减除算法**函数解析      
    ```python
    cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
    ```
    
    * `history`: 背景模型使用的帧数。
    * `dist2Threshold`: 控制阈值，较大的值会减少误检。
    * `detectShadows`: 是否检测阴影。



背景减除算法的局限性：

* **光照变化**：快速变化的光照可能会被误认为是前景。
* **动态背景**：例如摇摆的树叶、波动的水面，这些场景中的变化难以通过简单的背景建模处理。
* **阴影处理**：部分算法可以处理阴影，但阴影依旧可能导致误检。

### 4.2 项目实现
1. 读取视频并查看背景去除效果

```python
import cv2

cap = cv2.VideoCapture('video.mp4')  					# 打开视频文件
backSub = cv2.createBackgroundSubtractorMOG2()			# 创建背景减除对象

# 后面将原视频和处理后的视频并排显示，由于单个视频尺寸太大，拼接后会超过窗口范围，故需要进行缩放
# 设置缩放比例，比如 0.5 表示缩小到原始大小的一半
scale_percent = 0.5										

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    Mask = backSub.apply(frame)								# 应用背景减除算法  
    Mask_colored = cv2.cvtColor(Mask, cv2.COLOR_GRAY2BGR)	# 将前景掩码转换为三通道（彩色），以便与原始帧拼接
    combined_frame = cv2.hconcat([frame, Mask_colored])		# 水平拼接原始帧和前景检测帧    
    width = int(combined_frame.shape[1] * scale_percent)		# 获取拼接后图像的宽度和高度
    height = int(combined_frame.shape[0] * scale_percent)

    # 调整拼接后图像的大小，显示缩放后的拼接结果
    resized_combined_frame = cv2.resize(combined_frame, (width, height))
    cv2.imshow('Original and FG Mask', resized_combined_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/438f792a014e4782ae017f580376cd98.png#pic_center)
2. 形态学处理，得到前景，识别并标出车辆
&#8195;&#8195;可以看到，右侧图像中有很多小白点，还有路边那棵树，这些都是噪声，需要去除。此时可以考虑对原图进行灰度化，然后使用高斯滤波器进行去噪。

&#8195;&#8195;高斯去噪之后，小白点消除了大部分，剩下小部分（比如那棵树），可以通过先腐蚀再膨胀的方法进一步去除。只是这样做了之后，车在行驶中，会变成一团破碎的白色小方块，可以使用闭运算（先膨胀后腐蚀）进行处理。

&#8195;&#8195;接下来们就是使用findContours方法，在处理好的灰度图中，查找物体的轮廓，然后使用画最大外接矩形的方法标出车辆。
```python
import cv2

cap = cv2.VideoCapture('video.mp4')  					# 打开视频文件
backSub = cv2.createBackgroundSubtractorMOG2()			# 创建背景减除对象
# 创建矩形框形状的卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
min_w，min_h  = 80，75

while True:
    ret, frame = cap.read()
    if ret == True:
        # 把原始帧进行灰度化, 然后去噪
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        Mask = backSub.apply(blur)
        
        # 腐蚀
        erode = cv2.erode(Mask, kernel)
        # 膨胀, 多操作一次，把图像还原回来
        dialte = cv2.dilate(erode, kernel, iterations=2)
        # 闭运算整合破碎的小方块
        close = cv2.morphologyEx(dialte, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓，返回轮廓点列表和层级信息
		contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
            # 最大外接矩形
            (x, y, w, h) = cv2.boundingRect(contour)
            # 宽高必须同时大于设定的最小阈值，才会被认为是车辆否则跳过。
            if w < min_w or h < min_h:
                continue
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.imshow('frame', frame)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8a6e640efa54760b19aeaca24130be2.png)
&#8195;&#8195;可以看到，车辆及其阴影被一起检测出来。另外车体检测框内，有些还会有一些被误检出的小框（车牌也容易被误检出来），所以需要通过设定最小尺寸，将其过滤。

3. 标出检测线，过线统计车辆数，并在图像上方进行显示
标出检测线之后，通过检测框（最大外接矩形）计算出车辆中心点`cpoint`。当这个中心点y轴坐标落在离检测线非常近的区间（±offset），视为过线，就计数一次。以下是完整代码：

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('./video.mp4')
# 创建MOG2背景减法器对象，用于检测视频中的移动物体
mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
# 创建形态学操作卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

min_w,min_h = 80,75					# 检测框最小的宽度和高度，高于此阈值才被认为是车辆
line_high = 580						# 检测线的垂直位置，车辆经过这条线时会触发计数
offset = 7							# 检测线的上下容差，避免因细微位置差异导致误计数
cars = []							# 车辆中心点列表
carno = 0							# 车辆数

# 通过最大外接矩形（检测框）计算其中心点坐标
def center(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 灰度化并去噪
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    # 背景减除与形态学处理
    mask = mog.apply(blur)
    erode = cv2.erode(mask, kernel)
    dialte = cv2.dilate(erode, kernel, iterations=2)
    close = cv2.morphologyEx(dialte, cv2.MORPH_CLOSE, kernel)
    # 轮廓检测
    contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 3)
    for contour in contours:
    	# 通过轮廓面积、轮廓最小宽高、车辆形状比例等三个方面过滤掉形状异常的物体。
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_w or h < min_h:
            continue
        aspect_ratio = float(w) / h
        if aspect_ratio < 1.0 or aspect_ratio > 3.0:
            continue
        # 计算车辆中心点并使用实心圆标出
        cpoint = center(x, y, w, h)
        cars.append(cpoint)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, cpoint, 5, (0, 0, 255), -1)
	
	# 判断车辆是否经过检测线并计数
    for (x, y) in cars:
        if y > (line_high - offset) and y < (line_high + offset):
            carno += 1
            cars.remove((x, y))
            print(carno)
            
	# 显示车辆计数并展示图像
    cv2.putText(frame, 'Vehicle Count:' + str(carno), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)    
    cv2.imshow('frame', frame)

	# 按下ESC键退出
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69e63d7e078d41ac8e84358d0b406120.png)
&#8195;&#8195;在这段代码中，将车辆的中心点 `cpoint` 加入列表 `cars`，并在后续循环中判断车辆是否经过检测线后再移除，是为了处理车辆在多帧图像中的持续运动。这种设计的好处有以下几点：
1. **多帧处理，避免瞬时抖动**

	* 如果直接在遍历 `contour` 时判断 `cpoint` 是否过线，可能会因为车辆在一帧中刚好到达检测线，但下帧离开检测线时计数可能会发生错误。车辆的移动通常会跨越多帧，因此通过记录其中心点在 `cars` 列表中，可以确保车辆在多帧中一致检测。
	* 这种方法允许在多帧中跟踪车辆，避免车辆因为瞬时抖动或者检测不准而漏计。

 2. **处理单个车辆多次检测**
	
	* 如果直接在遍历轮廓时进行判断，没有列表存储，就无法确保同一辆车不会被多次计数。因为同一个 `cpoint` 在多帧中可能会多次经过检测线附近，直接判断会导致多次重复计数。
	* 将车辆的中心点存入 `cars` 列表，且只在其经过检测线时移除，可以保证每辆车只被计数一次。

目前存在的问题：
- 误检：对车辆的识别检测还是不够准确，有些莫名其妙的检测框会出现，导致计数错误。
如果视频背景复杂，光线变化大，传统的背景减法和形态学操作可能不够准确。可以考虑使用更鲁棒的车辆检测模型，如YOLO、SSD等深度学习模型，这样可以大幅提高检测的精度，并减少误检。
- 漏检：车辆过线未被统计。
改善车流跟踪机制：当前的车辆跟踪只依赖 cars 列表记录车辆中心点的单一坐标，由于视频帧率问题或车速过快，可能出现上一帧还未过线，下一帧已经过线，导致漏检。可以改用更稳定的追踪机制，如使用 cv2.Tracker 系列的跟踪算法：`tracker = cv2.TrackerCSRT_create()`



 







