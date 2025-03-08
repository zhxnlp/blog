@[toc]
- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)


## 一、信用卡数字识别
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('./ocr_a_reference.png')
image = cv2.imread('./credit_card_05.png')

plt.figure(figsize=[12,6]);
plt.subplot(121); plt.imshow(img[:,:,::-1]);plt.axis('off');plt.title("template");
plt.subplot(122); plt.imshow(image[:,:,::-1]);plt.axis('off');plt.title("card");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb1a06a6a1f04f0da7bf123079a10a19.png)
&#8195;&#8195;如上图所示，我们有一些信用卡图片，和一张信用卡数字模板。用AI算法训练推理会有一定的错误率。为了更高的准确率，考虑使用模板识别，解题思路为：
1. 先对模板处理, 获取每个数字的模板及其对应的数字标识；
2. 对信用卡处理, 通过一系列预处理操作, 取出信用卡数字区域；
3. 取出每一个数字去和模板中的10个数字进行匹配
### 1.1 模板匹配
&#8195;&#8195;[matchTemplate](https://docs.opencv.org/4.10.0/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d) 是 OpenCV 中用于模板匹配的函数，它通过滑动模板图像在输入图像上，计算每个位置的相似度，从而找到最佳匹配区域，其函数签名为：

```python
matchTemplate(image, templ, method[, result[, mask]]) -> result
```
1. **image**：输入图像，通常是灰度或彩色图像。
2. **template**：要匹配的模板图像，通常比输入图像小。
3. **method**：匹配方法，常用的有：
    * `cv2.TM_CCOEFF`：相关系数匹配，值越大越相关；
    * `cv2.TM_CCOEFF_NORMED`：归一化的相关系数匹配。
    * `cv2.TM_CCORR`：计算相关性，值越大越相关
    * `cv2.TM_CCORR_NORMED`：归一化的相关匹配。
    * `cv2.TM_SQDIFF`：平方差匹配，值越小越相关
    * `cv2.TM_SQDIFF_NORMED`：归一化的平方差匹配。

&#8195;&#8195;最终结果是一个矩阵，表示每个位置的相似度。假如原图形是$A \times B$大小，而模板是$a \times b$大小，则输出结果的矩阵尺寸为$(A-a+1) \times(B-b+1)$（和卷积计算一样）。

&#8195;&#8195;一般计算完之后，会接着使用`minMaxLoc`函数，找到最大最小值位置，以确定最佳匹配位置。函数返回四个值，分别是最小值,、最大值、最小值坐标、 最大值坐标：
```python
minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc
```
下面是一个演示示例：

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('lena.jpg')
template = cv2.imread('face.jpg')
h,w = template.shape[:2]

res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print(f'{img.shape=},{template.shape=},{res.shape=}')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(f'{min_val=},{min_loc=},{max_val=},{max_loc=}')

# 标记出最佳匹配位置,即以min_loc为左上角画一个template同尺寸的矩形。
img_copy=img.copy()
cv2.rectangle(img_copy,min_loc,(min_loc[0]+w,min_loc[1]+h),(0,0,255),2)

plt.figure(figsize=[12,6]);
plt.subplot(131); plt.imshow(img[:,:,::-1]);plt.axis('off');plt.title("lena");
plt.subplot(132); plt.imshow(template[:,:,::-1],aspect='equal');plt.axis('off');plt.title("template");
plt.subplot(133); plt.imshow(img_copy[:,:,::-1]);plt.axis('off');plt.title("matchTemplate");
```

```python
img.shape=(263, 263, 3),template.shape=(110, 85, 3),res.shape=(154, 179)
min_val=256897.0,min_loc=(107, 89),max_val=200943056.0,max_loc=(157, 45)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4d3f9bf37c14597bb06d5ac81954d2e.png)
### 1.2 匹配多个对象
&#8195;&#8195;上一节代码中，我们只是匹配图中的一个对象，而信用卡上有很多数字，要逐一匹配多个对象。方法也很简单，可考虑使用cv2.TM_CCOEFF_NORMED方法（归一化的相关系数匹配，值越大越相关），设置阈值为0.8。以下是演示效果：

```python
# 读取马里奥图片和金币模板图
img = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

# 匹配相关系数>0.8，可认为是匹配到的位置
# loc是匹配到的位置的x轴和y轴坐标，也可直接使用argwhere方法
loc = np.where(res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```python
(array([ 40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  41,  41,  41,
        41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  42,
        42,  42,  42,  42,  42,  42,  42,  42,  42,  42,  42,  42,  42,
        43,  43,  43,  43,  43,  72,  72,  72,  72,  72,  72,  72,  72,
        72,  72,  72,  72,  72,  73,  73,  73,  73,  73,  73,  73,  73,
        73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,
        74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,
        74,  74,  74,  74,  74,  74,  74,  75,  75,  75,  75,  75,  75,
        75, 104, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106,
       106, 106, 106], dtype=int64), 
array([ 69,  70,  83,  84,  97,  98, 111, 112, 125, 126,  68,  69,  70,
        82,  83,  84,  96,  97,  98, 110, 111, 112, 124, 125, 126,  68,
        69,  70,  82,  83,  84,  96,  97,  98, 110, 111, 112, 125, 126,
        69,  83,  97, 111, 125,  54,  55,  69,  83,  84,  97,  98, 111,
       112, 125, 126, 139, 140,  54,  55,  56,  68,  69,  70,  82,  83,
        84,  96,  97,  98, 110, 111, 112, 124, 125, 126, 138, 139, 140,
        54,  55,  56,  68,  69,  70,  82,  83,  84,  96,  97,  98, 110,
       111, 112, 124, 125, 126, 139, 140,  55,  69,  83,  97, 111, 125,
       139,  55,  55,  69,  83,  97, 111, 125, 139,  55,  69,  83,  97,
       111, 125, 139], dtype=int64))
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/896ffe17a7fa452a924f9a91736a749d.png#pic_center =400x)

### 1.3 处理数字模板
1. 将数字模板图进行灰度化和二值化处理，使数字部分更加鲜明
2. 使用findContours方法查找数字轮廓并展示（此时轮廓可能是乱序的）
3. 画出每个数字轮廓ref_contours的最大外接矩形bounding_boxes，按照每个数字的bounding_boxes的x轴坐标从小到大进行排序，排序后的轮廓就是有序的
4. 重新计算排序后的数字轮廓的最大外接矩形，将其resize到统一大小(57, 88)，便与后续匹配。此时的外接矩形就是从0到9这9个分割开的模板数字区域，将其添加到字典digits中。
5. 最后，为了使整个py文件可以作为脚本一样在命令行可以输入参数来执行类似`python card_ocr.py --image './credit_card_05.png'`这样的操作，使用[argparse](https://docs.python.org/zh-cn/3.11/library/argparse.html#adding-arguments)命令行解析器来加入参数。

```python
import argparse

# 创建命令解析器对象，添加命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-i','--image',required=True,help='path to image')
parser.add_argument('-t','--template',required=True,help='path to template ocr image')

# 解析参数（结果是元组类型），并使用vars将结果转为字典类型
args = vars(parser.parse_args())
print(args)
```
如果我们在命令行输入：

```python
python card_ocr.py -i credit_card_05.png -t ocr_a_reference.png
```
结果会是：

```python
{'image': 'credit_card_05.png', 'template': 'ocr_a_reference.png'}
```

```python
# 封装显示图片的函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

```python
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# 读取模板图片
template  = cv2.imread(args['template'])
# 灰度化处理
gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# 二值化处理
_, ref = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓，mode为只查找最外层轮廓。
# ref_contours是轮廓点列表，列表中每个元素是一个 ndarray 数组，表示轮廓上所有点的坐标
ref_contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 画出所有轮廓,原图会被直接修改
template_copy=template.copy()
cv2.drawContours(template_copy, ref_contours, -1, (0, 0, 255), 3);

# 计算每个轮廓的最大外接矩形，其与轮廓一一对应，但是是乱序的
bounding_boxes = [cv2.boundingRect(c) for c in ref_contours]
# 使用 zip 将 bounding_boxes 和 ref_contours 组合在一起
combined = list(zip(bounding_boxes, ref_contours))
# 根据 bounding_boxes 的 x 坐标排序
sorted_combined = sorted(combined, key=lambda item: item[0][0])
# 解压排序后的结果
sorted_boxes, sorted_contours = zip(*sorted_combined)

# 创建字典digits，存储排序后的数字区域
template_digits = {}
for (idx, c) in enumerate(sorted_contours):
    # 重新计算外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # 取出每个数字区域，roi表示感兴趣的区域region of interest）
    template_roi = ref[y:y + h, x: x + w]
    # resize成合适的大小
    template_roi = cv2.resize(template_roi, (57, 88))
    template_digits[idx] = template_roi
    # 逐个显示roi，其结果应该是数字从0到9
    cv_show('template_roi',template_roi)
    

plt.figure(figsize=[12,3]);
plt.subplot(221); plt.imshow(template[:,:,::-1]);plt.axis('off');plt.title("template");
plt.subplot(222); plt.imshow(gray,cmap='gray');plt.axis('off');plt.title("gray");
plt.subplot(223); plt.imshow(ref,cmap='gray');plt.axis('off');plt.title("ref");
plt.subplot(224); plt.imshow(template_copy[:,:,::-1]);plt.axis('off');plt.title("template_copy");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8ab8a1eb83344139e6d3ac9529c66a2.png)

### 1.4 预处理卡片信息，得到4组数字块。
1. 统一将信用卡图片resize到300的宽度，高宽比不变
2. 对其灰度化图像进行顶帽操作，突出显示比周围区域更亮的部分（数字部分）
3. 梯度幅值能够突出图像中的边缘和纹理，帮助检测和识别对象的边界。所以这一步要计算图像的 x 方向梯度幅值，然后归一化梯度图像，便于后续处理
4. 使用闭运算（先膨胀再腐蚀），可以把数字连在一起
5. 全局二值化，进一步突出数字区域
6. 查找轮廓并计算外接矩形，然后根据实际信用卡数字区域的长宽比, 找到真正的数字区域，最终得到4个数字块的外接矩形`digit_group_boxes`。

```python
# 读取信用卡
card = cv2.imread(args['image'])
# 对信用卡图片进行resize
h, w = card.shape[:2]
width = 300
r = width / w
card = cv2.resize(card, (300, int(h * r)))
gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

# 顶帽操作, 突出更明亮的区域。信用卡是长方形，这一步使用长方形卷积核，效果更好
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
tophat = cv2.morphologyEx(gray_card, cv2.MORPH_TOPHAT, rect_kernel)

# 使用sobel算子计算x轴方向梯度
grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# 使用绝对值得到梯度幅值
grad_x = np.absolute(grad_x)
# 归一化使得所有梯度值统一到 0 到 255 的范围内
min_val, max_val = np.min(grad_x), np.max(grad_x)
grad_x = ((grad_x - min_val) / (max_val - min_val)) * 255
# 修改一下数据类型
grad_x = grad_x.astype('uint8')

# 闭操作, 先膨胀, 再腐蚀, 可以把数字连在一起.
close = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)

# 通过OTSU算法找到合适的阈值, 进行全局二值化操作.
_, thresh_card = cv2.threshold(close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 中间还有空洞, 再来一个闭操作
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
close2 = cv2.morphologyEx(thresh_card, cv2.MORPH_CLOSE, sq_kernel)

# 查找轮廓
card_contours, _ = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 在原图上画轮廓
card_copy = card.copy()
cv2.drawContours(card_copy, card_contours, -1, (0, 0, 255), 3)

plt.figure(figsize=[12,6]);
plt.subplot(231); plt.imshow(card[:,:,::-1]);plt.axis('off');plt.title("img");
plt.subplot(232); plt.imshow(tophat,cmap='gray');plt.axis('off');plt.title("topcat");
plt.subplot(233); plt.imshow(grad_x,cmap='gray');plt.axis('off');plt.title("grad_x");
plt.subplot(234); plt.imshow(close,cmap='gray');plt.axis('off');plt.title("close");
plt.subplot(235); plt.imshow(close2 ,cmap='gray');plt.axis('off');plt.title("close2");
plt.subplot(236); plt.imshow(card_copy[:,:,::-1]);plt.axis('off');plt.title("card_copy");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4455dae5ec04d16bbe9eca1a578fab5.png)

```python
# 遍历轮廓, 计算外接矩形, 然后根据实际信用卡数字区域的长宽比, 找到真正的数字区域
digit_group_boxes = []
for c in card_contours:
    # 计算外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算外接矩形的长宽比例
    ar = w / float(h)
    # 选择合适的区域
    if ar > 2.5 and ar < 4.0:
        # 在根据实际的长宽做进一步的筛选,下面是测试效果比较好的数值
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合条件的外接矩形留下来
            digit_group_boxes.append((x, y, w, h))
            
# 对符合要求的轮廓进行从左到右的排序.
digit_group_boxes = sorted(digit_group_boxes, key=lambda x: x[0])
digit_group_boxes
```

```python
[(30, 105, 50, 17), (93, 105, 48, 17), (155, 105, 49, 17), (218, 106, 49, 16)]
```
### 1.5 遍历数字块，将卡片中每个数字与模板数字进行匹配

```python
output=[]    # 最终输出

# 1. 遍历每个数字块, 把原图中的每个数字抠出来.
for (i, (gx, gy, gw, gh)) in enumerate(digit_group_boxes):
    # 抠出每个数字块, 并且加点余量
    digit_group = gray_card[gy - 5: gy + gh + 5, gx - 5: gx + gw + 5]
    # 全局二值化处理，使数字更明显
    digit_group = cv2.threshold(digit_group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # 2. 通过轮廓查找，分割出单个数字的轮廓和外接矩形
    digit_contours, _ = cv2.findContours(digit_group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的最大外接矩形，其与轮廓一一对应，但是是乱序的
    digit_boxes = [cv2.boundingRect(c) for c in digit_contours]
	# 使用 zip 将 bounding_boxes 和 ref_contours 组合在一起
    combined2 = list(zip(digit_boxes, digit_contours))
	# 根据 bounding_boxes 的 x 坐标排序
    sorted_combined2 = sorted(combined2, key=lambda item: item[0][0])
	# 解压排序后的结果，得到排序后的单个数字
    sorted_digit_boxes, sorted_digit_contours = zip(*sorted_combined2)

    # 定义每个数字块的输出结果
    group_output = []

    # 3. 遍历排好序的数字轮廓
    for c in sorted_digit_contours:
        # 找到当前数字的轮廓, resize成合适的大小, 然后再进行模板匹配
        (x, y, w, h) = cv2.boundingRect(c)
        # 取出每个数字区域
        card_roi = digit_group[y: y + h, x: x + w]
        card_roi = cv2.resize(card_roi, (57, 88))
        
        # 4. 将每个数字区域和模板中的每个数字区域进行匹配，取最佳结果
        match_scores = []
        for (idx, template_roi) in template_digits.items():
            result = cv2.matchTemplate(card_roi,template_roi, cv2.TM_CCOEFF)
            # 只要最大值, 即分数
            # minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(result)
            match_scores.append(result)
        # 找到分数最高的数字, 即我们匹配到的数字
        group_output.append(str(np.argmax(match_scores)))
        
    # 画出轮廓和显示数字    
    cv2.rectangle(card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(card, ''.join(group_output), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    output.extend(group_output)
cv_show('card', card)
print(''.join(output))
```

```python
5476767898765432
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/42abb2e2f916437896b3bf3d6d0b5276.png#pic_center =400x)
## 二、人脸检测
&#8195;&#8195;OpenCV进行人脸检测的基本思想基于经典的**Haar级联分类器**，它是由Paul Viola和Michael Jones在2001年提出的快速目标检测算法，被称为**Viola-Jones算法**。该算法通过一系列的简单分类器进行多层次的过滤，达到高效检测的目的。

### 2.1人脸检测算法原理

1. **Haar-like特征**
	Haar特征是图像中矩形区域的亮度对比，主要通过对比不同区域的像素值来提取图像中的特征。典型的Haar特征包括：
	- 边缘特征：用于检测图像中物体的边缘。
	- 线条特征：用于检测线性结构，如鼻梁或嘴唇。
	- 四方形特征：用于检测复杂的区域，如眼睛周围的阴影。

Haar特征通过矩形区域（黑白区域）的像素加权求和来计算，分类器通过这些特征来判断一个区域是否可能包含人脸。

2. **积分图**
为了加快特征的计算，Haar级联分类器使用了**积分图**（Integral Image）来快速计算任意矩形区域的像素和。积分图的构建使得计算矩形区域的和只需在常数时间内完成（复杂度O(1)），这极大地提高了检测速度。

3. **AdaBoost算法**
	- Haar级联分类器使用**AdaBoost算法**来选择有效的Haar特征并组合成一个强分类器。AdaBoost是一种自适应提升算法，它通过选择一系列弱分类器（每个分类器只根据一个简单特征做出决策），并将这些弱分类器加权组合，形成一个强分类器。它逐步强化那些能够正确分类的数据点，并弱化那些误分类的数据点。
	- 在训练过程中，分类器会从数十万个Haar特征中选择出那些最能区分目标的特征，通常只需几千个特征就足以构成一个有效的分类器。

4. **级联分类器（Cascade Classifier）**
为了进一步提高检测效率，Viola-Jones算法采用了**级联分类器**的策略。级联分类器是一系列逐层过滤的分类器，形成了一种分阶段的检测方法：
	- **初始阶段**：使用简单且快速的特征检测器，迅速排除大部分不包含人脸的区域。
	- **后续阶段**：逐步增加检测的复杂度，对初始阶段通过的区域进行更精细的检测。

&#8195;&#8195;这种设计大大提高了检测效率，因为大部分无关区域会在前几层被快速排除，只有少数可能包含目标的区域会通过全部层的检测。

### 2.2 OpenCV中的人脸检测流程
一个高准确率的级联分类器的主要生成步骤如下： 
1. 大量样本集合，特征值的提取 
2. 通过adaboost 训练多个弱分类器并迭代为强分类器 
3. 多层级联强分类器，得到最终的级联分类器 。

&#8195;&#8195;这些训练流程完成之后结果以`xml`的方式保存起来，就是分类器文件。opencv中包含了以上的实现，并且已经存放了许多已经训练好的不同类型的不同特征值提取生成的的级联分类器。 在Opencv中，可以直接加载这些分类器文件，并且给出了便捷的检测API。以下是一个示例：

```python
# 1.读取图片
# Haar特征是基于亮度差异的，不需要颜色信息，使用灰度图检测可以加快速度
img = cv2.imread('./p3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2.创建haar级联器
face = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('./haarcascade_eye.xml')

# 3.检测人脸，画出检测出的人脸框
faces = face.detectMultiScale(gray)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 在人脸区域继续检测眼睛
    roi_img = img[y: y + h, x: x + w]
    eyes = eye.detectMultiScale(roi_img)
    for (ox, oy, ow, oh) in eyes:
        cv2.rectangle(roi_img, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
        roi_eye = roi_img[oy: oy + oh, ox: ox + ow]
        img[y: y + h, x: x + w] = roi_img
        
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6dbc87e67aed486582eaff9c591f4133.png#pic_center =400x)


- `cv2.data.haarcascades`：是OpenCV库中一个特定的属性，它指向存储预训练Haar级联分类器模型文件的目录。这个目录包含了一系列用于对象检测的XML文件，包括人脸、眼睛、微笑等常见目标的检测模型。比如我的电脑，打印`cv2.data.haarcascades`，结果是`'D:\\Miniconda\\Lib\\site-packages\\cv2\\data\\'`
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8251ee058fa148989508af1d4c0ae97a.png#pic_center =600x)

- `detectMultiScale`：用于检测图像中的目标对象（如人脸）。这个函数会扫描整个图像，在不同尺度上寻找符合分类器特征的区域。它返回检测到的目标区域的矩形列表。其参数为：
	* **`image`**：输入图像，通常为灰度图。
	* **`scaleFactor`**：默认值`1.1`。人脸在图像中的大小可能各不相同，OpenCV通过图像金字塔技术，在不同的尺度上检测人脸。scaleFactor控制图像金字塔的缩放步长，值越小检测越精细，但速度越慢。
	* **`minNeighbors`**：确定检测到的人脸矩形框需要有多少个“Neighbors”才会被认为是有效检测，这个值越大，检测结果越可靠，但可能会漏掉一些目标，默认值为 `3`。
	* **`flags`**：检测模式标志，通常使用默认值 `0`。
	* **`minSize`**：要检测目标的最小尺寸，默认值为 `(30, 30)`。
	* **`maxSize`**：要检测目标的最大尺寸，默认为 `None`。


| 优点                          | 缺点                             |
|-------------------------------|----------------------------------|
| **实时性强**：Haar级联分类器的检测速度非常快，尤其适用于实时应用，如摄像头中的人脸检测。                   | ***鲁棒性较差** ：对于光照变化、遮挡、非正面人脸等情况，检测效果不佳。                      |
| **轻量级**：相比深度学习模型，Haar级联分类器需要的资源较少，可以在低计算能力的设备上运行。                      | **精度有限** ：Haar特征较为简单，无法很好地处理复杂场景。相对于基于深度学习的检测方法（如CNN、YOLO），其精度较低。                        |
| **简单易用**：OpenCV中已经提供了预训练的模型，可以直接使用                      | **姿态敏感**：该方法对检测对象的姿态变化（如侧脸）比较敏感。                         |


&#8195;&#8195;随着深度学习的发展，基于卷积神经网络（CNN）的目标检测方法（如YOLO、SSD、MTCNN等）在复杂场景下表现出了更高的精度和鲁棒性。因此，在精度要求较高的应用中，通常会使用深度学习方法进行人脸检测。然而，对于资源受限的设备或需要高实时性的场景，OpenCV中的Haar级联分类器依然是一个快速、轻量的选择。

## 三、车牌识别
使用opencv进行车牌识别的主要思路是：
1. 使用级联分类器检测出车牌区域
2. 对车牌区域进行形态学处理
3. 使用tesseract进行OCR识别

### 3.1 安装tesseract
&#8195;&#8195;[Tesseract](https://github.com/tesseract-ocr/tessdoc)是目前最准确的开源OCR（光学字符识别）引擎之一，由Google赞助并进行进一步的开发和维护，能够识别多种语言的文字；支持Windows、Linux和macOS等多操作系统。其安装方法为：
- `macOS`: brew install tesseract tesseract-lang
- `ubantu`: apt install tesseract tesseract-lang
- `windows`: 网上下载[tesseract安装包](https://digi.bib.uni-mannheim.de/tesseract/)，流程详见[《Tesseract-OCR 下载安装和使用》](https://blog.csdn.net/weixin_51571728/article/details/120384909)

输入`tesseract --help extra`可以看到其进阶说明：

```python
tesseract imagename|imagelist|stdin outputbase|stdout [options...] [configfile...]

OCR options:
  --tessdata-dir PATH   Specify the location of tessdata path.
  --user-words PATH     Specify the location of user words file.
  --user-patterns PATH  Specify the location of user patterns file.
  --dpi VALUE           Specify DPI for input image.
  --loglevel LEVEL      Specify logging level. LEVEL can be
                        ALL, TRACE, DEBUG, INFO, WARN, ERROR, FATAL or OFF.
  -l LANG[+LANG]        Specify language(s) used for OCR.
  -c VAR=VALUE          Set value for config variables.
                        Multiple -c arguments are allowed.
  --psm NUM             Specify page segmentation mode.
  --oem NUM             Specify OCR Engine mode.
```
即基本的 Tesseract 命令格式为：

```python
tesseract imagename outputbase [-l lang] [-psm pagesegmode] [-oem ocrenginemode] [configfile...]
```
- `imagename` ：你想要识别的图像文件的路径。
- `outputbase` ：输出文件名（自动添加.txt后缀）
- `-l lang` ：是可选的语言参数，用于指定 OCR 使用的语言。
- `-psm pagesegmode`： 页面分割模式。
- `-oem ocrenginemode`： OCR 引擎模式。
- `configfile...` ：是可选的配置文件

其中，页面分割模式和OCR 引擎模式分别是：
```python
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.

OCR Engine modes:
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
```


比如在cmd中直接使用命令行进行图片识别：

```python
# 使用中文语言包进行识别，输出到output.txt文件
tesseract 横渠四句.png output -l chi_sim
```

```python
Estimating resolution as 713
为天地立心
为生民立命
为往圣继绝学
为万世开太平
```
&#8195;&#8195;如果要在代码中使用Tesseract，需要安装pytesseract（`pip  install pytesseract`）。`pytesseract` 使用的是 `Tesseract` 的命令行界面，因此你可以在 `config` 参数中传递任何有效的 `Tesseract` 命令行参数。

```python
import matplotlib.pyplot as plt
import cv2
import pytesseract

img = cv2.imread('1.png')
plt.imshow(img[:,:,::-1])
# 默认模式
text = pytesseract.image_to_string(img,lang="chi_sim")
# 或是自选模式
custom_config = r'--oem 1 --psm 3 -l chi_sim'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4873594ebe1146b4926db328c871966c.png#pic_center =700x)
### 3.2 车牌识别

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract

img = cv2.imread('./chinacar.jpeg')

# 变成黑白图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建haar级联器
car = cv2.CascadeClassifier('./haarcascade_russian_plate_number.xml')
car_plate  = car.detectMultiScale(gray)
for (x, y, w, h) in car_plate :
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    roi = gray[y: y + h, x: x + w]
    
    # 二值化
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('roi', roi_bin)
    # 开操作
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    roi_close= cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    #cv2.imshow('roi2', roi)
    result = pytesseract.image_to_string(roi_close, lang='chi_sim+eng', config='--psm 8 --oem 3')
    
plt.figure(figsize=[12,6]);
plt.subplot(131); plt.imshow(img[:,:,::-1]);plt.axis('off');plt.title("img");
plt.subplot(132); plt.imshow(roi,cmap='gray');plt.axis('off');plt.title("roi");
plt.subplot(133); plt.imshow(roi_close,cmap='gray');plt.axis('off');plt.title("roi_close");
print(result)
```
```python
*G5N555
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e3f7c696154b496face1cbd59d743b33.png)
&#8195;&#8195;这个模型进行车牌识别，还不够精准，经常会出错。可以考虑通过车牌的固定形状进行轮廓筛选，而且车牌大多是蓝色和绿色。

## 四、答题卡识别

- 图片预处理，找到答题卡的四个角点
- 透视变换,，把答题卡的视角拉正
- 找出所有轮廓，根据圆圈的面积筛选出正确的轮廓
- 通过计算非零值来判断是否答题正确.

### 4.1 查看样例图片，找出答题卡轮廓

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./images/test_01.png')
img2 = cv2.imread('./images/test_03.png')
img3 = cv2.imread('./images/test_05.png')

plt.figure(figsize=[16,12]);
plt.subplot(131); plt.imshow(img1[:,:,::-1]);plt.title("img1");
plt.subplot(132); plt.imshow(img2[:,:,::-1]);plt.title("img2")
plt.subplot(133); plt.imshow(img3[:,:,::-1]);plt.title("img3");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b5e453969484f0a88f1fdf79f209b68.png)

 找出答题卡轮廓
```python
img = cv2.imread('./images/test_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 去掉一些噪点
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 为了仿射变换，需要先边缘检测，再找出答题卡轮廓，找出四个角。
edged = cv2.Canny(blurred, 75, 200)

# 只检测最外层轮廓
contours,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画轮廓会修改原图，所以拷贝一份原图
contours_img = img.copy()
cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 3)

plt.figure(figsize=[20,12]);
plt.subplot(141); plt.imshow(img[:,:,::-1]);plt.title("img");
plt.subplot(142); plt.imshow(blurred,cmap='gray');plt.title("blurred")
plt.subplot(143); plt.imshow(edged,cmap='gray');plt.title("edged");
plt.subplot(144); plt.imshow(contours_img[:,:,::-1]);plt.title("contours_img");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/beb390c6fe424dc18b9cd49b75f620db.png)
&#8195;&#8195;为了防止图片中混入其他物体造成检测错误，需要对识别出的轮廓进行判断，确保我们拿到的轮廓是答题卡的轮廓。
- 按照轮廓面积对所有轮廓进行排序
- 答题卡可能拍的是斜的，不是一个标准的矩形，需要使用多边形逼近（`cv2.approxPolyDP`）的方式找出近似轮廓。（近似前答题卡轮廓可能有很多个点，近似后只会有四个点）


```python
if len(contours) > 0:
    # 根据轮廓面积对轮廓进行排序.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 遍历每一个轮廓
    for c in contours:
        # 计算周长
        perimeter = cv2.arcLength(c, True)
        # 得到多边形近似轮廓，处理后答题卡轮廓应该只有四个点
        # 经过调试，epsilon取0.02倍周长效果最好。True表示是闭合轮廓
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx) == 4:
            bbox = approx
            # 找到答题卡近似轮廓, 直接推荐.
            break

print(bbox)
```

```python
array([[[131, 206]],

       [[119, 617]],

       [[448, 614]],

       [[430, 208]]], dtype=int32
```
### 4.2 仿射变换，矫正答题卡
&#8195;&#8195;上面找出的轮廓有四个点，但顺序是乱的，需要先确认每个点的位置，然后再进行透视变换。下面将这两个操作都封装成一个函数。

```python
def order_points(bbox):
    # 创建全是0的矩阵, 来接收等下找出来的4个角的坐标.
    rect = np.zeros((4, 2), dtype='float32')
    # 求每个点横纵坐标的和
    s = bbox.sum(axis=1)						
    # 左上的坐标一定是x,y加起来最小的坐标. 右下的坐标一定是x,y加起来最大的坐标.
    rect[0] = bbox[np.argmin(s)]
    rect[2] = bbox[np.argmax(s)]
    
    # np.diff是列表中后一个元素对前一个元素做差值，这里就是y-x
    # 左下角的y-x一定是最大，右上角的y-x一定是最小（自己画个矩形就知道了）
    diff = np.diff(bbox, axis=1)
    rect[1] = bbox[np.argmin(diff)]
    rect[3] = bbox[np.argmax(diff)]
    return rect

def four_point_transform(image, bbox):
    # 对输入的4个坐标排序
    rect = order_points(bbox)
    (tl, tr, br, bl) = rect
    
    # 计算上下两条边的长度（空间中两点的距离公式），取最长的长度为变换后的宽度
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(widthA), int(widthB))
    
    #  同理得到变换后的高度
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(heightA), int(heightB))
    
    # 构造变换之后的对应坐标位置.
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped
```

```python
warped_gray = four_point_transform(gray, bbox.reshape(4, 2))

plt.figure(figsize=[20,12]);
plt.subplot(121); plt.imshow(gray,cmap='gray');plt.title("gray");
plt.subplot(122); plt.imshow(warped_gray,cmap='gray');plt.title("warped_gray")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/58886de4108a405a92d2ba7879592bec.png)

### 4.3 找出答案轮廓

```python
# 二值化处理
_, thresh= cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 只最外层轮廓，忽略轮廓内部的任何嵌套轮廓或子轮廓
cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 复制一份，画的时候不修改原图
thresh_contours = thresh.copy()
cv2.drawContours(thresh_contours, cnts, -1, 255, 3)

plt.subplot(121); plt.imshow(thresh,cmap='gray');plt.title("thresh");
plt.subplot(122); plt.imshow(thresh_contours,cmap='gray');plt.title("thresh_contours");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f59910c732094ec596039e034a74319e.png#pic_center =600x)
通过轮廓形状从所有轮廓中筛选出圆圈选项轮廓：

```python
# 遍历所有的轮廓, 找到特定宽高和特定比例的轮廓, 即圆圈的轮廓.
question_cnts = []
for c in cnts:
    # 找到轮廓的外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算宽高比
    ar = w / float(h)
    
    # 根据实际情况制定标准.
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        question_cnts.append(c)

len(question_cnts),len(question_cnts[0])
```

```python
25,54
```
每个轮廓都有很多个点，只能根据轮廓的最大外界矩形进行标记。

```python
# 求每个轮廓的最大外接矩形
bounding_boxes = [cv2.boundingRect(c) for c in question_cnts]
# 根据外接矩形的坐标进行排序(先排y轴，再排x轴）
(sort_cnts, sort_boxes) = zip(*sorted(zip(question_cnts, bounding_boxes), key=lambda b: (b[1][1],b[1][0]),reverse=False))

# 下面的代码只是确认一下排序和标记的逻辑是否正确
texts=['A','B','C','D','E']*5
print(texts)

warped_img = four_point_transform(img, bbox.reshape(4, 2))
for (idx,box) in  enumerate(sort_boxes):
    x,y,w,h=box
    text=texts[idx]
    cv2.putText(warped_img, text, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
plt.imshow(warped_img[:,:,::-1]);
```

```python
['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E']
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/294cab6e9b114eb78bf8123ff1449c18.png#pic_center)
### 4.4 识别选项
- 逐个遍历所有答案轮廓，使用drawContours函数将轮廓涂白
- 涂白后做与运算，这样处理后，没有被涂抹的选项是不变的，被涂抹的选项会出现涂抹痕迹
- 通过cv2.countNonZero() 函数，计算二值图像中非零像素的数量。被涂抹的选项会比其它选项数量更多。

```python
warped_img = four_point_transform(img, bbox.reshape(4, 2))

answer=[1,4,0,2,1]													# 正确结果
correct = 0															# 统计识别正确的数量
bubbled = None														# 存储每行最大非零像素结果		
count=0																# 统计循环次数，每5个是一行，进行判断


for (i,c) in enumerate(sort_cnts):
    mask = np.zeros(thresh.shape, dtype='uint8')
    # 在灰度图上依次画出每个轮廓，-1表示内部填充，即整个涂白作为掩膜
    cv2.drawContours(mask, [c], -1, 255, -1)
    # 逐个答案选项做与运算，只留下答案轮廓
    thresh_and = cv2.bitwise_and(thresh, thresh, mask=mask)
    cv_show('result', np.hstack((mask,thresh_and)))
    
    # 计算非零像素个数，被涂抹的选项应该是最多的
    non_zero=cv2.countNonZero(thresh_and)
    # 每一行bubbled重置为None，第一个点的数值直接写入；之后依次比较，更大的值才会被写入
    if bubbled is None or non_zero > bubbled[0]:
    	# 最终每一行都留下最大的non_zero值及其索引，也就是这一行被涂抹的答案序号
        bubbled = (non_zero, i)                
        
    count+=1        
    # 每遍历完一行进行判断
    if count ==5:
        # 将答案序号与5取余数，得到一个0到4的数
        result=bubbled[1]%5
        print(f'{result=},{answer[i//5]=}')
        # 答题正确画红线
        if result == answer[i//5]:
            correct += 1
            color = (0, 0, 255)  
            # 正确选项是当前行数*5+result
            idx=result+(i//5)*5
            cv2.drawContours(warped_img, [sort_cnts[idx]], -1, color, 2)
        else:
            color = (255, 0, 0) 
            cv2.drawContours(warped_img, [sort_cnts[idx]], -1, color, 2)
            
        bubbled = None
        count=0


score = (correct / 5.0) * 100
cv2.putText(warped_img, str(score) + '%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
plt.imshow(warped_img[:,:,::-1]);
```

```python
result=1,answer[i//5]=1
result=4,answer[i//5]=4
result=0,answer[i//5]=0
result=2,answer[i//5]=2
result=1,answer[i//5]=1
```
运行时窗口会依次显示每个选项的mask图，及这个选项的原图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b1c600d7c6d24963a45f37dc0518e8dc.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf9a2c2c29f94941947226d5170032f4.png#pic_center =400x)
## 五、光学字符识别（OCR）&光流估计
1. 对图片进行预处理，查找出最大的闭合轮廓
2. 进行仿射变换，将图片进行矫正
3. 使用`pytesseract`进行OCR


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('images/page.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊去掉一些噪点
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用开运算，进一步去除噪点，确保后面只得到一个最大外轮廓
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
blurred = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
# 为了仿射变换，需要先边缘检测，再找出答题卡轮廓，找出四个角。
edged = cv2.Canny(blurred, 75, 200)


# 只检测最外层轮廓
contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours),len(contours[0]))

contours_img = img.copy()
cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 5)

plt.figure(figsize=[20,12]);
plt.subplot(221); plt.imshow(img[:,:,::-1]);plt.title("img");
plt.subplot(222); plt.imshow(gray,cmap='gray');plt.title("blurred")
plt.subplot(223); plt.imshow(edged,cmap='gray');plt.title("edged");
plt.subplot(224); plt.imshow(contours_img[:,:,::-1]);plt.title("contours_img");
```

```python
(1, 1165)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/081c941dd2c54fa88cf9aa483da08dd3.png)

>&#8195;&#8195;如果不进行开运算，会检测出大量轮廓。按照轮廓面积从大到小进行排序，`contours[0]`死活画不出最大轮廓，不知道为什么。但是换另一张图就可以。

```python
# 不同的图片可能得到多个轮廓，筛选出最大面积的闭合轮廓
if len(contours) > 0:
	contours = sorted(contours, key=cv2.contourArea, reverse=True) 
    for c in contours:
        # 计算周长
        perimeter = cv2.arcLength(c, True)
        # 得到多边形近似轮廓，处理后答题卡轮廓应该只有四个点
        # 经过调试，epsilon取0.02倍周长效果最好。True表示是闭合轮廓
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx) == 4:
            bbox = approx                    
            break

cv2.drawContours(contours_img, [bbox], -1, (0, 0, 255), 5)
plt.imshow(contours_img[:,:,::-1]);
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e900d356bbd34c10b780ceb0130e1fb0.png#pic_center =400x)


```python
warped_gray = four_point_transform(gray, bbox.reshape(4, 2))

plt.figure(figsize=[20,12]);
plt.subplot(121); plt.imshow(gray,cmap='gray');plt.title("gray");
plt.subplot(122); plt.imshow(warped_gray,cmap='gray');plt.title("warped_gray")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0d46352f775945e6b4177bc263cab37a.png)

```python
import pytesseract
from PIL import Image


cv2.imwrite('./page1.jpg', warped_gray)
# pytesseract要求的image不是opencv读进来的image, 而是pillow这个包, 即PIL,按照 pip install pillow
text = pytesseract.image_to_string(Image.open('./page1.jpg'))
print(text)
```

```html
4.3 ACCESSING AND MANIPULATING PIXELS

On Line 14 we manipulate the top-left pixel in the im-
age, which is located at coordinate (0,0) and set it to have
a value of (0, 0, 255). If we were reading this pixel value
in RGB format, we would have a value of 0 for red, 0 for
green, and 255 for blue, thus making it a pure blue color.

However, as I mentioned above, we need to take special
care when working with OpenCV. Our pixels are actually
stored in BGR format, not RGB format.

We actually read this pixel as 255 for red, 0 for green, and
0 for blue, making it a red color, not a blue color.

After setting the top-left pixel to have a red color on Line
14, we then grab the pixel value and print it back to con-
sole on Lines 15 and 16, just to demonstrate that we have
indeed successfully changed the color of the pixel.

Accessing and setting a single pixel value is simple enough,
but what if we wanted to use NumPy’s array slicing capa-
bilities to access larger rectangular portions of the image?
The code below demonstrates how we can do this:

Listing 4.3: getting and_setting.py

17 corner = image[0:100, 0:100]
18 cv2.imshow("Corner", corner)

20 image[0:100, 0:100] = (0, 255, 0)

22 cv2.imshow("Updated", image)
23 cv2.waitKey(0)

On line 17 we grab a 100 x 100 pixel region of the image.
In fact, this is the top-left corner of the image! In order to
grab chunks of an image, NumPy expects we provide four

22
```

