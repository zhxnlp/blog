@[toc]
- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)



>参考：
>- [opencv官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)、[opencv 4.10.0官方文档](https://docs.opencv.org/4.10.0/)
>- [opencv官方免费课程（含视频、代码）](https://opencv.org/opencv-free-course/#elementor-action:action=popup:open&settings=eyJpZCI6Ijk5NjciLCJ0b2dnbGUiOmZhbHNlfQ==)（验证之后可以下载，资源大概一个G，包含14章代码和YouTube视频）
>- CSDN帖子[《【youcans的OpenCV例程200篇】总目录》](https://blog.csdn.net/youcans/article/details/125112487)


## 一、图像处理入门（读取、显示、转换、拆分合并、保存）
### 1.1 机器视觉基本概念
&#8195;&#8195;机器视觉(Machine Vision)在早些年一般指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等，并进一步做图形处理，使电脑处理成为更适合人眼观察或传送给仪器检测的图像。简单来说就是研究如何使机器看懂东西，所以更偏向硬件侧。

&#8195;&#8195;计算机视觉(Computer Vision)更偏向算法侧，指使用图像处理、模式识别、人工智能等技术，对图像进行分析。计算机视觉为机器视觉提供图像和景物分析的理论及算法基础，机器视觉为计算机视觉的实现提供传感器模型、系统构造和实现手段。发展到现在，很多时候这两个概念以及不加区分了。

机器视觉的常见应用有：
- 物体识别: 人脸识别, 车辆检测
- 识别图像中的文字(OCR)
- 图像拼接, 修复, 背景替换

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81768027f74243c3885eb40bf340e7f2.png#pic_center =600x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3dc1538a82aa41729cefb9dbbaae6a31.png#pic_center =600x)
### 1.2 OpenCV简介和安装
>[《OpenCV官方教程节选》](https://blog.csdn.net/qq_56591814/article/details/127275045)、[OpenCV官网教程](https://docs.opencv.org/4.9.0/)

 &#8195;&#8195; Opencv（Open Source Computer Vision Library）是一个基于开源发行的跨平台计算机视觉库，它实现了图像处理和计算机视觉方面的很多通用算法，已成为计算机视觉领域最有力的研究工具。
 
1. OpenCV的特点：
	- 底层用C++语言编写，速度快。
	- 具有C ++，Python，Java和MATLAB接口，如今也提供对于C#、Ch、Ruby，GO的支持
	- 支持Windows，Linux，Android和Mac OS， 可跨平台使用。
	- 经过20多年的发展，具有成熟的生态链。

2. 安装OpenCV
	- 本体安装： `pip install opencv-python==3.4.1.15` （3.4.2之后有些算法申请了专利,用不了了。）
	- opencv扩展包(选装): `pip install opencv-contrib-python==3.4.1.15`
	- pip安装失败，可在[官网](https://www.lfd.uci.edu/~gohlke/pythonlibs/)下载相应的包手动安装。

```python
# 查看安装版本
cv2.__version__
3.4.1
```

3. 虚拟环境安装OpenCV
virtualenv 是一个在 Python 中创建隔离的虚拟环境的工具。它允许你为每个项目创建一个独立的Python运行环境，这样就可以避免不同项目之间依赖包的版本冲突问题（可pip方式直接安装）。

| 命令                           | 功能说明                                               |
| ------------------------------ | ------------------------------------------------------ |
| `virtualenv <env_name>`         | 创建一个名为 `<env_name>` 的虚拟环境                     |
| `source <env_name>/bin/activate`| 激活虚拟环境 (Linux/Mac)                                |
| `<env_name>\Scripts\activate`   | 激活虚拟环境 (Windows)                                  |
| `deactivate`                    | 退出当前虚拟环境                                        |
| `virtualenv --version`          | 查看 `virtualenv` 的版本                                |
| `virtualenv -p python3 <env_name>` | 使用特定 Python 版本创建虚拟环境 (如 `python3`)          |
`virtualenv -p /usr/bin/python3.8 [env_name]`	|使用指定路径的 Python 解释器创建虚拟环境（例如，使用 Python 3.8）。
| `virtualenv --system-site-packages <env_name>` | 创建一个虚拟环境，并将系统的全局包也包含在内。         |
| `rm -rf <env_name>`             | 删除虚拟环境                                            |
| `pip freeze > requirements.txt` | 导出当前虚拟环境的依赖包列表至 `requirements.txt` 文件  |
| `pip install -r requirements.txt` | 根据 `requirements.txt` 安装依赖包                     |


创建并激活虚拟环境`venv`之后，直接进行pip安装我们需要的库：

```python
# 创建虚拟环境
virtualenv venv
# 可在venv\Scripts文件夹的地址栏打开cmd，即切换到Scripts目录，然后输入activate命令激活虚拟环境
venv\Scripts\activate

pip install opencv-python==3.4.1.15 opencv-contrib-python==3.4.1.15 jupyter matplotlib -i https://pypi.douban.com/simple
```
>`virtualenv` 和 `conda` 都是创建和管理虚拟环境的工具，但它们有一些显著的不同。
>- virtualenv仅支持 Python 虚拟环境，conda支持 Python、R 等多种语言的包管理与虚拟环境
>- virtualenv只管理 Python 包，不能管理系统库（如 C 库等），conda可以管理非 Python 的库，比如 C 库、MKL、NumPy 等系统依赖，处理依赖冲突能力较强
>- virtualenv 适合需要轻量的 Python 虚拟环境管理，尤其在标准 Python 项目开发中非常流行，依赖 pip 管理 Python 包。conda 更加适合科学计算、机器学习、数据科学等场景，特别是在需要处理 Python 之外的库时表现优异，它提供了强大的包管理系统和多语言支持。

### 1.3  图像显示
#### 1.3.1 使用Image直接显示图像
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import Image
```
我们将使用以下图片作为示例，使用ipython图像函数来加载和显示图像。
```python
# Display 84x84 pixel image.
Image(filename='checkerboard_84x84.jpg') 
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d741262cecf84e6ea3dad3914b4f6d3f.png)
#### 1.3.2 使用`cv2.imread`读取图像
&#8195;&#8195;OpenCV可以使用使用[cv2.imread](https://docs.opencv.org/4.5.1/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)函数读取不同类型的图像（JPG、PNG等）。您可以加载灰度图像、彩色图像，也可以使用Alpha通道加载图像。其语法为：
`retval = cv.imread(filename[, flags])`
>读取模式：[ImreadModes](https://docs.opencv.org/4.5.1/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)

- `retval`：读取的 OpenCV 图像，nparray 多维数组。如果无法读取图像（文件丢失，权限不正确，格式不支持或无效），该函数返回一个空矩阵None(此时不报错）。
- `filename`：读取图像的文件路径和文件名，绝对路径相对路径都行。
- `flags`：读取图片的方式，可选项
	- `1`或cv2.IMREAD_COLOR：**默认方式**，始终将图像转换为 3 通道`BGR`彩色图像（PIL、PyQt、matplotlib 等库使用的是 `RGB` 格式），忽略透明度通道（如果有的话）。如果文件是灰度图像，仍会转换为彩色图像（每个通道都相同）。
	- `0`或`cv2.IMREAD_GRAYSCALE`：始终将图像转换为单通道灰度图像。
灰度图像有256个灰度级，用数值区间[0,255]来表示，其中255表示为纯白色，0表示为纯黑色。256个灰度级的数值恰好可以用一个字节（8位二进制值）来表示
	- `-1`或`cv2.IMREAD_UNCHANGED`：加载图像，包括alpha通道（存储透明度信息的附加通道，如果存
- 注意事项：
	- OpenCV 读取图像文件，返回值是一个nparray 多维数组。OpenCV 对图像的任何操作，本质上就是对 Numpy 多维数组的运算。
	- OpenCV 中彩色图像使用 `BGR` 格式，而 。
	- `cv2.imread()` 指定图片的存储路径和文件名，在 python3 中不支持中文和空格（但并不会报错）。必须使用中文时，可以使用 cv2.imdecode() 处理，参见扩展例程。

>&#8195;&#8195;通常情况下，我们根据需求来选择适合的图像读取方式。例如，如果你正在处理图像识别任务，并且颜色信息对任务来说并不重要，那么读取灰度图像会更快，同时还能节省内存。如果图像处理任务需要透明度信息，那么你需要使用`cv2.IMREAD_UNCHANGED`模式。


```python
# 将图像读取为灰度图
cb_img = cv2.imread("checkerboard_18x18.png",0) # 就是上面那张18*18的图像
# 打印图像数据（像素值），2D numpy数组
# 每个像素值为8位[0，255]
print(cb_img)
```
```python
[[  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]
```

&#8195;&#8195;某些时候 `cv2.imread`无法读取包含中文路径的图像，此时课使用 `imdecode` 方法读取，此方法也可读取网络图像。
```python
 img = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), -1)
 plt.imshow(img)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/43d65abd3f2eef30b4a59ece3e18de8f.png)
```python
# 读取网络图像
import urllib.request as request
response = request.urlopen("https://profile.csdnimg.cn/8/E/F/0_youcans")
imgUrl = cv2.imdecode(np.array(bytearray(response.read()), dtype=np.uint8), -1)
plt.imshow(imgUrl)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c70f70686e1ed89061bf111c562aef72.png)

>在AI stadio平台创建的notebook中，可以直接使用cv2.imread正确读取中文名图像。


#### 1.3.3  Mat数据结构&图像属性显示
&#8195;&#8195;Mat 是 C++ 中 OpenCV 用来表示图像的核心数据结构，是 C++ OpenCV 库中最重要的类之一，由两部分组成：
- `Header`：存储图像的元数据信息，如图像的尺寸（宽度和高度）、通道数、数据类型（如 CV_8U、CV_32F 等）、步幅（每行占用的字节数）等。这些信息帮助 OpenCV 知道如何解析和操作图像数据。

| 属性名        | 数据类型     | 功能描述                                                |
|---------------|--------------|---------------------------------------------------------|
| `rows`        | `int`        | 图像的行数（高度）。                                     |
| `cols`        | `int`        | 图像的列数（宽度）。                                     |
| `data`        | `uchar*`     | 指向图像像素数据的指针，包含实际的图像数据。              |
| `step`        | `size_t`     | 每一行的字节数（步幅），即图像中一行数据的大小。           |
| `type()`      | `int`        | 返回图像的类型（如 `CV_8UC1`、`CV_8UC3` 等）。            |
| `channels()`  | `int`        | 返回图像的通道数（如灰度图像为 1，彩色图像为 3）。         |              |                  |
| `flags`       | `int`        | 存储矩阵的标志位，包含维度、深度、通道数等信息。            |
| `dims`        | `int`        | 矩阵的维数（通常图像是 2 维，深度图像或其他复杂数据可能更高）。|


- `Data`：指向图像的像素数据的指针。这个数据可以存储在连续的内存块中，数据存储的结构根据图像的格式和维度变化。
>-  `CV_8UC1`中CV_是图像数据类型在 OpenCV 中的前缀，8是8位，U 表示无符号（unsigned），8U表示图像中每个像素的每个通道占用 8 位无符号整数（0-255）。C1表示图像是 单通道（1 channel），即灰度图。因此，`CV_8UC1` 表示每个像素由 1 个通道 构成，且每个通道占用 8 位无符号整数，常用于灰度图像。
>- `CV_8UC3`：表示 8 位无符号整数的 3 通道图像（通常是彩色图像，BGR 格式）。

&#8195;&#8195;Mat 类采用了 引用计数 的机制，因此多个 Mat 对象可以共享相同的数据。当一个 Mat 对象被复制时，只会复制头部，而不会立即复制底层数据。只有当一个对象进行修改时，才会执行 写时拷贝（Copy-on-Write），以确保数据的一致性和节省内存。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8270ccf07494f3a961acb257732af15.png#pic_center =600x)


&#8195;&#8195;在 Python 中，OpenCV 使用 `numpy.ndarray` 来代替 C++ 中的 Mat 类，因此，在 Python 中处理图像数据时，并不需要直接操作 C++ 的 Mat 类，而是使用 numpy 库中的数据结构来完成类似的工作，所以你可以通过ndarray主要属性（ndim、shape、size、dtype、data）来访问Mat图像的属性。
- `img.ndim`：查看图像的维数。
	- 彩色图像读取结果是一个三维数组，通常称为图像矩阵。这三个维度分别代表：
		- 高度（Height）：图像的垂直方向上的像素数。
		- 宽度（Width）：图像的水平方向上的像素数。
		- 通道（Channel）：图像的颜色通道数。彩色图像通常是使用RGB颜色模型，所以有红绿蓝三个通道。
	- 灰度图像是一个二维数组，其形状为(高度, 宽度)。每个像素点仅由一个灰度值表示（0到255之间，0表示黑色，255表示白色），而没有颜色通道。
- `img.shape`：查看图像的形状，即图像栅格的行数（高度）、列数（宽度）、通道数。
- `img.size`：查看图像数组元素总数，灰度图像的数组元素总数为像素数量，彩色图像的数组元素总数为像素数量与通道数的乘积。
- `img.__sizeof__()`：图像的存储大小（字节数）

```python
imgBGR = cv2.imread("../images/imgLena.tif", 1)
print("img shape is ", imgBGR.shape)
print("img size is ", imgBGR.size,'\n')

img_gry = cv2.imread("../images/imgLena.tif", 0)
print("img_gry shape is: ", img_gry.shape)
print("img_gry size is ", img_gry.size)

print("Data type of img is ", imgBGR.dtype)
```

```python
img shape is  (599, 1440, 3)
img size is  2587680

img_gry shape is:  (599, 1440)
img_gry size is  862560
Data type of img is  uint8
```
#### 1.3.4 使用窗口显示图像（imshow）
opencv显示图像的常用函数如下：
| 函数 | 功能 |
| --- | --- |
| `imshow` （notebook不兼容，IDLE中运行）| 显示一幅图像。函数的第一个参数是窗口名称，第二个参数是要显示的图像。 |
| `namedWindow`（notebook不兼容，IDLE中运行） | 创建一个窗口，允许自定义窗口大小、位置等属性。窗口名必须是唯一的。 |
 `destroyWindow()`|关闭指定的显示窗口
| `destroyAllWindows` | 关闭所有的窗口，释放与窗口相关的所有资源。 |
| `resizeWindow` | 调整指定窗口的大小。第一个参数是窗口名称，接下来的参数是宽度和高度。 |
| `waitKey` | 暂停程序一段时间，等待用户按键。时间以毫秒为单位，返回值是按下键的 ASCII 码。如果传入 0，则无限等待。 |

##### 1.3.4.1 imshow
&#8195;&#8195;`imshow`函数用于在指定窗口中显示图像，其其语法为：`None=cv2.imshow(winname,img)`。其中`winname`为窗口名，`img`是 OpenCV 图像，nparray 多维数组。下面是一个简单的示例：
```python
import os
>>> os.chdir('E:\CV\opencv-python-free-course-code')
>>> print (os.getcwd())
E:\CV\opencv-python-free-course-code
>>> import cv2
>>> img=cv2.imread("images/imgLena.tif",-1)
>>> cv2.imshow("winname", img)
```
电脑自动弹出窗口winname，以原尺寸（1280×768）打开图片
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2c3d4d38d1e5701bd72ec95a2413c8b5.png)
##### 1.3.4.2 namedWindow
&#8195;&#8195;[namedWindow](https://docs.opencv.org/4.5.1/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b)函数：用于创建一个具有合适名称和大小的窗口，以在屏幕上显示图像和视频。（默认情况下，图像以其原始大小显示）函数语法为：`None	=	cv.namedWindow(	winname[, flags]	)`
- `window_name`：窗口的名称
- `flag`： 表示窗口大小是自动设置还是可调整。
	- `WINDOW_NORMAL` –允许手动更改窗口大小
	- `WINDOW_AUTOSIZE(Default)` –自动设置窗口大小
	- `WINDOW_FULLSCREEN` –将窗口大小更改为全屏
- 图像窗口将在 `waitKey()` 函数所设定的时长（毫秒）后自动关闭，`waitKey(0)` 表示窗口显示时长为无限，按下任意按键后才关闭窗口。

1. `WINDOW_AUTOSIZE`自动调整窗口
```python
import cv2
path = '../images/imgLena.tif'
image = cv2.imread(path)
  
# 使用namedWindow()函数，创建名为Display的窗口
# flag为WINDOW_AUTOSIZE, 表示自动调整窗口大小，目测是大小为图片原尺寸
cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
cv2.imshow('Display', image)
  
cv2.waitKey(1000) 					# 等待1000ms后退出窗口。  
# 也可以设置只有按下Q键，才销毁窗口。
key = cv2.waitKey(0)  				# 等待按键命令, 0表示任意按键。
# 如果此时按下键盘上的Q这个键，key的值就是其ASCII 码的值，即113.
if key == ord('q'):
  cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2c3d4d38d1e5701bd72ec95a2413c8b5.png)

2. `cv2.WINDOW_NORMAL`手动调节窗口大小（弹出的窗口可以拉伸），也可按指定大小的窗口显示图像
```python
cv2.namedWindow("Demo1", cv2.WINDOW_NORMAL) # 手动调节窗口
cv2.resizeWindow("Demo2", 400, 300) # 指定窗口大小
cv2.imshow('Demo1', image)
cv2.imshow('Demo2', image)
cv2.destroyAllWindows()
```
#### 1.3.5 使用Matplotlib显示图像（notebook兼容）


1. 直接显示彩色图像
```python
# 读取并显示 Coca-Cola logo.
Image("coca-cola-logo.png")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/25313aced95d9ab09ad52d629d7c0997.png#pic_center )

2. 使用plt模块显示图像，语法：`matplotlib.pyplot.imshow(img[, cmap])`
	- `img`：图像数据，nparray 多维数组，对于 openCV（BGR）格式图像要先进行格式转换（OpenCV 使用 BGR 格式，matplotlib/PyQt 使用 RGB 格式）
	- `cmap`：颜色图谱（colormap），默认为 RGB(A) 颜色空间
		- gray：灰度显示
		- hsv：hsv 颜色空间
	- `plt.imshow()` 可以使用 matplotlib 库中的各种方法绘图，如标题、坐标轴、插值等，详见 [matploblib Document](https://matplotlib.org/stable/contents.html#)。
```python
# 读取图像
import matplotlib.pyplot as plt

coke_img = cv2.imread("coca-cola-logo.png",1)
plt.imshow(coke_img)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/54d2a6ba5e1e08695763abc7608a61ec.png#pic_center )
&#8195;&#8195;上面显示的颜色与实际图像不同。这是因为matplotlib需要RGB格式的图像，而OpenCV则以BGR格式存储图像。因此，为了正确显示，我们需要反转图像的通道。

```python
# 直接手动转换，也可使用cv.cvtColor()函数

coke_img_channels_reversed = coke_img[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1eb063aed8a93591d55aa7206f1c7f9e.png#pic_center )
>`:` 表示所有的行和所有的列（即整个图像的所有像素）;`::` 表示对第三个维度（通道）的切片操作，`::-1` 意思是把通道的顺序逆序。

2.  使用pylab模块显示图像

```python
import matplotlib.pylab as pylab
pylab.imshow(coke_img_channels_reversed)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1eb063aed8a93591d55aa7206f1c7f9e.png#pic_center )

```python
# 显示成灰度图

coke_gray = cv2.imread("coca-cola-logo.png",0)
plt.imshow(coke_gray,cmap='gray')
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/658fe8c9adc44c42a012d4253efec820.png#pic_center =400x)


二者的区别：
- `pylab`：结合了pyplot和numpy，将numpy导入了其命名空间中，对交互式使用来说比较方便，既可以画图又可以进行简单的计算，pylab表现的和matlab更加相似
- `pyplot`：相比pylab更加纯粹，如果只是打印图片，使用这个就行。

### 1.4 保存图像（imwrite）
&#8195;&#8195;函数 [cv2.imwrite()](https://docs.opencv.org/4.5.1/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) 用于将图像保存到指定的文件。其语法为：`retval = imwrite(filename, img[, params])`

- `filename`：要保存的文件的路径和名称，包括文件扩展名（图像格式是根据文件扩展名选择）
- `img`：要保存的 OpenCV 图像，nparray 多维数组
- `params`：不同编码格式的参数，详见[Imwrite flags文档](https://docs.opencv.org/4.5.1/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac)。
	- `cv2.CV_IMWRITE_JPEG_QUALITY`：设置 .jpeg/.jpg 格式的图片质量，取值为 0-100（默认值 95），数值越大则图片质量越高；
	- `cv2.CV_IMWRITE_WEBP_QUALITY`：设置 .webp 格式的图片质量，取值为 0-100；
	- `cv2.CV_IMWRITE_PNG_COMPRESSION`：设置 .png 格式图片的压缩比，取值为 0-9（默认值 3），数值越大则压缩比越大。

- `retval`：返回值，保存成功返回 True，否则返回 False。

注意：
- `cv2.imwrite()` 保存的是 OpenCV 图像（多维数组），不是 cv2.imread() 读取的图像文件，所保存的文件格式是由 filename 的扩展名决定的，与读取的图像文件的格式无关。
- 此功能只能保存8位单通道或BGR 3通道图像，或 PNG/JPEG/TIFF 16位无符号单通道图像。对 4 通道 BGRA 图像，可以使用 Alpha 通道保存为 PNG 图像。
- `cv2.imwrite(`) 指定图片的存储路径和文件名，在 python3 中不支持中文和空格（但并不会报错）。必须使用中文时，可以使用 `cv2.imencode()` 处理，参见扩展例程。

1. 基本示例
```python
# cv2读取的原始BGR图，保存后发现是RGB图，再次读取还是BGR图
imgBGR = cv2.imread("../images/imgLena.tif", flags=1)  # 读取为BGR彩色图像

cv2.imwrite('../images/SaveBGR.png', imgBGR)
SaveBGR=cv2.imread('SaveBGR.png',1)
plt.imshow(SaveBGR)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/367340ee1307f5c7b5044c362b2e1939.png#pic_center )

```python
# 转换为RGB格式的图像，发现保存后图片是BRG格式，但是cv2默认方式打开，打印后是RGB格式
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)  # BGR 转换为 RGB

cv2.imwrite('../iamges/SaveRGB.png', imgRGB)
SaveRGB=cv2.imread('SaveRGB.png',1)
plt.imshow(SaveRGB)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0d16acca121a3f192b98bf5487e4e0a4.png#pic_center )
&#8195;&#8195; <font color='deeppink'>可见：图像的原始格式和cv2默认读取的格式是倒序的，和cv2保存和格式也是倒序的。</font>只不过一般我们读取的图片是RGB格式，所以cv2默认读取是BGR格式。如果图片本身就是BGR格式，默认读取就是正常的RGB格式。
 - RGB默认读取→BGR，保存→RGB，默认读取→BGR
 - RGB默认读取→BGR，转为RGB→RGB，保存→BGR，默认读取→RGB

2. 扩展示例：保存中文路径的图像

```python
 saveFile = "../images/测试图.jpg"  # 带有中文的保存文件路径
 # cv2.imwrite(saveFile, img3)  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
 img_write = cv2.imencode(".jpg", imgBGR)[1].tofile(saveFile)
```
### 1.5 图像的色彩空间转换（cv.cvtColor）

&#8195;&#8195;色彩空间是指通过多个（通常为 3个或4个）颜色分量构成坐标系来表示各种颜色的模型系统。色彩空间中的每个像素点均代表一种颜色，各像素点的颜色是多个颜色分量的合成或描述。

&#8195;&#8195;彩色图像可以根据需要映射到某个色彩空间进行描述。在不同的工业环境或机器视觉应用中，使用的色彩空间各不相同。

&#8195;&#8195;常见的色彩空间包括：GRAY 色彩空间（灰度图像）、XYZ 色彩空间、YCrCb 色彩空间、HSV 色彩空间、HLS 色彩空间、CIELab 色彩空间、CIELuv 色彩空间、Bayer 色彩空间等。

&#8195;&#8195;计算机显示器采用 RGB 色彩空间，数字艺术创作经常采用 HSV/HSB 色彩空间，机器视觉和图像处理系统大量使用 HSl、HSL色彩空间。各颜色分量的含义分别为：

- `RGB`：`RGB` 模型是一种加性色彩系统，色彩源于红、绿、蓝三基色。用于CRT显示器、数字扫描仪、数字摄像机和显示设备上，是当前应用最广泛的一种彩色模型。
- `HSV/HSB`：
	- `Hue`：色相，如红色, 蓝色.，用角度度量，取值范围为0°～360°。从红色开始按逆时针方向计算，红色为0°，绿色为120°，蓝色为240°，60度代表黄色，180度代表青色。
	- `Saturation`：饱和度，控制纯色中混入白色的量。一种颜色，可以看成是其光谱色与白色混合的结果，其中光谱色占比越高，颜色越纯越饱和。通常取值范围为0%～100%，0表示完全无色（灰色），1表示纯色。
	- `Value/Brightness`：明度，控制纯色中混入黑色的量，取值范围为0%（黑）到100%（白）。
<table>
  <tr>
    <td><img src="https://i-blog.csdnimg.cn/direct/fb8128a71daf4fdc8937d5540548a7af.png" alt="Image 1"></td>
    <td><img src="https://i-blog.csdnimg.cn/direct/a8bbe6692c0d4d12992ccf654e30a418.png" alt="Image 2" ></td>
  </tr>
</table>

- `HSL`：包括色调（Hue）、饱和度（Saturation）和亮度（Lightness）。和HSV不同的是，HSL中的亮度控制纯色中混入黑白两种颜色的量，所以其顶部是白色，底部是黑色；而其饱和度和黑白没有关系。

<table>
  <tr>
    <td><img src="https://i-blog.csdnimg.cn/direct/ed0cd2c2bedf4b9b84e01b9f749751d0.png" alt="Image 1" width="300" ></td>
    <td><img src="https://i-blog.csdnimg.cn/direct/d5fc9d43ccc14e5282b86e42981d896b.png" alt="Image 2" width="600" ></td>
  </tr>
</table>

&#8195;&#8195;使用函数 [cv.cvtColor()](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)可以进行色彩空间类型转换，。例如，在进行图像的特征提取、距离计算时，往往先将图像从 RGB 色彩空间转换为灰度色彩空间。函数语法为：

```python
cvtColor(src, code[, dst[, dstCn]]) -> dst
```
- src：输入图像，nparray 多维数组
- code：颜色空间转换代码，详见 [ColorConversionCodes](https://docs.opencv.org/4.5.1/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0)
- dst：输出图像，大小和深度与 src 相同
- dstCn：输出图像的通道数，0 表示由src和code自动计算。



```python
# 读取原始图像
imgBGR = cv2.imread("../images/imgLena.tif", flags=1)  # 读取为BGR彩色图像
print(imgBGR.shape)

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)  # BGR 转换为 RGB, 用于 PyQt5, matplotlib
imgGRAY = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)  # BGR 转换为灰度图像
imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)  # BGR 转换为 HSV 图像
imgYCrCb = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YCrCb)  # BGR转YCrCb
imgHLS = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HLS)  # BGR 转 HLS 图像
imgXYZ = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2XYZ)  # BGR 转 XYZ 图像
imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)  # BGR 转 LAB 图像
imgYUV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YUV)  # BGR 转 YUV 图像

# 调用matplotlib显示处理结果
titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
images = [imgBGR, imgRGB, imgGRAY, imgHSV, imgYCrCb,
          imgHLS, imgXYZ, imgLAB, imgYUV]
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3c8b03be35a4ebad7830fc6b7ad421bb.png)
### 1.6 图像通道的拆分和合并（cv2.split&cv2.merge）

- `cv2.split`：将多通道arrays划分为多个单通道array。（[spilt函数文档](https://docs.opencv.org/4.5.1/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a)）
- `cv2.merge`：将多个array合并为一个多通道数组arrays。所有输入矩阵的大小必须相同。
>- 直接用 imshow 显示返回的单通道对象，将被视为 (width, height) 形状的灰度图像
>- 如果要正确显示某一颜色分量，需要增加另外两个通道值（置 0）转换为 BGR 三通道格式，再用 imshow 才能显示为拆分通道的颜色。

```python
# 将图像拆分为B,G,R components
img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg",cv2.IMREAD_COLOR) # IMREAD_COLOR就是flag=1，默认读取彩色图像
b,g,r = cv2.split(img_NZ_bgr)

# 显示各个通道
plt.figure(figsize=[20,5])
plt.subplot(131);plt.imshow(r,cmap='gray');plt.title("Red Channel");
plt.subplot(132);plt.imshow(g,cmap='gray');plt.title("Green Channel");
plt.subplot(133);plt.imshow(b,cmap='gray');plt.title("Blue Channel");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/171bf251f9bc45aea89daf0dbd1836c7.png#pic_center )

```python
# 将单通道扩展为三通道
imgZeros = np.zeros_like(img_NZ_bgr)  # 创建与原图相同形状的全0数组（黑色）
imgZeros[:,:,1] = g  				  # 在黑色图像模板添加绿色分量 g

# 将各个通道合并到BGR图像中
imgMerged = cv2.merge((b,g,r))
imgStack = np.stack((b, g, r), axis=2) # 效果和cv2.merge等价，但是操作更简单
# 显示合并后的图像
plt.subplot(121);plt.imshow(imgZeros,cmap='gray');plt.title("Green Channel");
plt.subplot(122);plt.imshow(imgMerged[:,:,::-1]);plt.title("Merged Output");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/80c675b4765f4754bee74f151b435c81.png#pic_center)


### 1.7 修改单个通道

```python
h,s,v = cv2.split(imgHSV)
# 显示H、S、V三通道和原始RGB图像
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(imgRGB);plt.title("Original");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ac815cffd2013971bda95a4e01bc113c.png)

```python
# 将色相通道h的强度值加10，然后对比显示

h_new = h+10
img_merged = cv2.merge((h_new,s,v))
img_rgb = cv2.cvtColor(img_merged, cv2.COLOR_HSV2RGB)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(img_rgb);plt.title("Modified");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ac2db03a5bf72439a04e9cbc5acda4c8.png)
>&#8195;&#8195;`h, s, v` 都是单通道图像，每个像素值代表的是该通道的强度。在matplotlib中，如果不指定`cmap`参数，它会默认使用RGB颜色映射，这会导致显示的结果不正确，因为单通道图像没有红、绿、蓝三个颜色分量的信息，而且使用灰度映射可以更容易地观察和分析每个通道的细节。
### 1.8 图片拼接和拷贝
1. 图片拼接
用 Numpy 的数组堆叠方法可以进行图像的拼接，操作简单方便。
	
	- `retval = numpy.hstack((img1, img2, …))` # 水平拼接
	- `retval = numpy.vstack((img1, img2, …))` # 垂直拼接
```python
imgFile1 = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile1, flags=1)  # flags=1 读取彩色图像(BGR)
imgFile2 = "../images/imgGaia.tif"  # 读取文件的路径
img2 = cv2.imread(imgFile2, flags=1)  # # flags=1 读取彩色图像(BGR)

imgStack = np.hstack((img1, img2))  # 相同大小图像水平拼接
cv2.imshow("Demo4", imgStack)  # 在窗口 "Demo4" 显示图像 imgStack
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b457e66a9b026a22b31a353207282c0a.png#pic_center )

2. 图像的拷贝
Python 中的 “复制” 有无拷贝、浅拷贝和深拷贝之分
	- 无拷贝相当于引用（例如直接赋值得到的新图像）
	- `view`：浅拷贝，创建一个新的数组对象，但它引用的仍然是原始数组的数据（切片操作也是浅拷贝）。这意味着，如果你修改了浅拷贝中的数据，原始数组的数据也会被修改，因为它们共享同一块数据缓冲区。
	 - `copy`：对原变量（ndarray数组）的所有数据的拷贝，来创建一个全新的数组对象。在深拷贝之后，原始数组和拷贝的数组之间没有任何关联，修改一个不会影响到另一个。

```python
import cv2
import numpy as np

cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
img = cv2.imread('./cat.jpeg')

img2 = img.view()					#浅拷贝
img3 = img.copy()					#深拷贝

img[10:100, 10:100] = [0, 0, 255]
cv2.imshow('Display', np.hstack((img, img2,img3)))

cv2.waitKey(0)
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/02a4d6571f0542ffbd2c7de635beaec5.png#pic_center )
## 二、视频处理
### 2.1 读取视频
#### 2.1.1  VideoCapture类简介
&#8195;&#8195;[VideoCapture](https://docs.opencv.org/4.9.0/d8/dfe/classcv_1_1VideoCapture.html)  是 OpenCV 库中的一个类，用于从视频文件、图像序列或者摄像头捕获视频。你可以通过创建 VideoCapture 类的实例来使用它，例如：

```python
cap=cv2.VideoCapture(video_source)
```
- cap：视频捕获对象，可以使用read方法读取视频
- video_source：可以是设备索引（整数）或视频文件路径（字符串格式）。

>&#8195;&#8195;设备索引就是摄像头的ID号码，默认值为`-1`，表示随机选取一个摄像头。如果电脑有多个摄像头，则用数字`0`表示第一个，用数字`1`表示第二个，以此类推。


cap常用的属性和方法如下：
| 属性/方法                           | 说明                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `cv2.VideoCapture(video_source)`   | 初始化视频捕获对象。`video_source` 可以是设备索引（如 0 表示摄像头）或视频文件路径（字符串）。         |
| `.isOpened()`                      | 检查视频捕获设备是否成功打开，返回布尔值。                                                  |
| `.read()`                          | 读取视频帧，返回 `(ret, frame)`，其中 `ret` 是布尔值，表示读取是否成功，`frame` 是帧数据。 |
| `.release()`                       | 释放捕获对象，释放摄像头或视频文件资源。                                                    |
| `.get(propId)`                     | 获取指定的视频属性，`propId` 是属性的编号，如帧宽、高等。                                    |
| `.set(propId, value)`              | 设置指定的视频属性，`propId` 是属性的编号，`value` 是设置的值。                             |
| `.grab()`                          | 抓取下一帧，但不解码并返回。可以加快帧的处理速度。                                           |
| `.retrieve([flag])`                | 解码并返回抓取的帧。`flag` 是可选的解码参数。                                               |
| `.open(filename)`                  | 打开指定的视频文件或设备，类似于构造函数。                                                  |
| `.getBackendName()`                | 返回当前使用的视频捕获后端的名称（字符串）。                                                |
| `.setExceptionMode(enable)`        | 设置是否在捕获失败时抛出异常。                                                             |
| `.getExceptionMode()`              | 获取当前是否启用了异常模式。                                                               |


| 属性编号  (`propId`)                          | 说明                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `cv2.CAP_PROP_FRAME_WIDTH`  ，值为`3`      | 帧的宽度（像素）。                                                                          |
| `cv2.CAP_PROP_FRAME_HEIGHT`  ，值为`4`      | 帧的高度（像素）。                                                                          |
| `cv2.CAP_PROP_FPS`  ，值为`5`                | 帧率。                                                                                      |
| `cv2.CAP_PROP_FRAME_COUNT`         | 视频中的总帧数。                                                                            |
| `cv2.CAP_PROP_POS_FRAMES`          | 当前捕获到的帧序号。                                                                        |
| `cv2.CAP_PROP_POS_MSEC`            | 当前帧对应的时间戳（毫秒）。                                                                |
| `cv2.CAP_PROP_BRIGHTNESS` ，值为`10`         | 图像的亮度（摄像头支持时）。                                                                |
| `cv2.CAP_PROP_CONTRAST`  ，值为`11`            | 图像的对比度（摄像头支持时）。                                                              |
| `cv2.CAP_PROP_SATURATION`   ，值为`12`         | 图像的饱和度（摄像头支持时）。                                                              |
| `cv2.CAP_PROP_GAIN`   ，值为`14`                | 图像的增益（摄像头支持时）。                                                                |
| `cv2.CAP_PROP_EXPOSURE`   ，值为`15`         | 曝光时间（摄像头支持时）。                                                                  |
#### 2.1.2  从摄像头读取视频
```python
import cv2

# 打开摄像头,默认值为-1，表示随机选取一个摄像头.
# 如果电脑有多个摄像头，则用数字0表示第一个，用数字1表示第二个，以此类推
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
else:
	# 设置窗口大小
    cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)
	
    # 打印常用属性
    print("帧宽度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("帧高度:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("帧率 (FPS):", cap.get(cv2.CAP_PROP_FPS))
    print("总帧数:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("当前帧号:", cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("当前帧时间戳 (毫秒):", cap.get(cv2.CAP_PROP_POS_MSEC))
    
	# 逐帧读取视频数据
    while True:
        ret, frame = cap.read() # ret是布尔类型
        if not ret:
            print("无法接收帧 (流结束或出现错误)")
            break

        # 显示视频帧
        cv2.imshow('Camera Feed', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
```

```python
帧宽度: 640.0
帧高度: 480.0
帧率 (FPS): 30.0
总帧数: -1.0
当前帧号: 0.0
当前帧时间戳 (毫秒): 0.0
```

1.  `cv2.waitKey(delay)`
	- 这 是 OpenCV 中的一个函数，用来等待键盘输入。它会等待 delay 毫秒（必须是整数，否则报错），检查用户是否按下了某个键。如果是整数0，表示会无限等待，直到有任意键盘输入，才关闭窗口。
	- `cv2.waitKey(delay)`还在 GUI 窗口的刷新过程中起到了重要作用。必须使用它来更新显示的图像（如使用 cv2.imshow() 时），否则窗口不会正常刷新。
	- 如果在这段时间内没有键盘输入，它会返回 -1，如果有输入，它会返回对应键的 ASCII 值，比如键盘上的"Q"按键，对应的ASCII 值为ord('q')，即133。
2. `& 0xFF`：在某些系统上，`cv2.waitKey()` 返回的值可能包含额外的无用信息（高位值），而 `& 0xFF` 操作会将其限制在 8 位（即 0-255 的范围），从而确保只保留低 8 位（也就是我们感兴趣的 ASCII 值）。

所以`if cv2.waitKey(1) & 0xFF == ord('q'):`的意思是：
- 隔 1 毫秒检查一次是否有键盘输入，如果用户按下某个键，`cv2.waitKey(1)` 会返回对应键的ASCII 值。
- `& 0xFF` 操作确保只检查键值的低 8 位，以避免系统差异带来的问题
- 当用户按下 'q' 键时，返回的值等于 ord('q') ，循环退出。


&#8195;&#8195;另外，如果打开一个30帧的视频，如果设置`cv2.waitKey(1)`，则会发现视频被加速了，因为此视频正常是1秒刷新30张图片，所以应该设置为`cv2.waitKey(1000//30)`。另外`VideoCapture`读取的只是每一帧的图片，所以是没有声音的。

#### 2.1.3 从视频文件读取视频

```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

source = './race_car.mp4'  # source = 0 for webcam
cap = cv2.VideoCapture(source)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# 读取并显示第一帧
ret, frame = cap.read()
print(frame.shape)

plt.imshow(frame[...,::-1]) # 等价于plt.imshow(frame[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/06ecac6061e6adc8aaf4b9800a32195b.png#pic_center)
### 2.2 播放视频
下面的操作可以直接在当前notebook中播放视频：
```python

from IPython.display import HTML
HTML("""
<video width=1024 controls>
  <source src="race_car.mp4" type="video/mp4">
</video>
""")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d25c08f5cc28b0dc8d41298159baa7b7.png#pic_center)
### 2.3 视频录制
&#8195;&#8195;`cv2.VideoWriter` 是 OpenCV 中用于保存视频的类，你可以通过设置帧宽度、高度、帧率和编码格式等参数，将录制的帧写入视频文件，生成 MP4、AVI 等格式的视频，其语法为：

```python
cv2.VideoWriter(filename, fourcc, fps, frameSize[, isColor])
```


| 参数名        | 类型            | 说明                                                                 | 示例                            |
|---------------|-----------------|----------------------------------------------------------------------|---------------------------------|
| `filename`    | 字符串           | 保存视频的文件路径和文件名，支持多种格式（如 .mp4、.avi）。           | `'output.mp4'`                 |
| `fourcc`      | 四字符编码       | 用于指定视频的编码格式。常用编码包括：<br> `'XVID'`：AVI 格式的视频<br> `'MP4V'`：MP4 格式的视频<br> `'MJPG'`：Motion-JPEG 编码 | `cv2.VideoWriter_fourcc(*'MP4V')` |
| `fps`         | 浮点数           | 视频的帧率（每秒的帧数）。                                             | `20`                         |
| `frameSize`   | 元组 `(宽, 高)`  | 视频帧的宽度和高度，通常从摄像头获取（如 `cap.get(3)`、`cap.get(4)`）。 | `(640, 480)`                   |
| `isColor`     | 布尔值（可选）   | 指定是否为彩色视频。默认为 `True` 表示彩色，`False` 表示黑白视频。    | `True` 或 `False`              |

```python
import cv2

# 打开默认摄像头,创建显示窗口
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)

# 设置视频的分辨率
# .get(propId)用于获取指定的视频属性，编号3 和4分别代表视频的宽和高
frame_width = int(cap.get(3))                  			 
frame_height = int(cap.get(4))

# 定义编码器和创建VideoWriter对象，保存为MP4格式
vw = cv2.VideoWriter('output.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 	# 编码格式
                      20, # 帧率
                      (frame_width, frame_height)) 		# 视频分辨率

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:        
        vw.write(frame)									# 写入帧到视频文件	                
        cv2.imshow('video', frame)						# 在窗口中显示帧                
        if cv2.waitKey(1) & 0xFF == ord('q'):			# 按下 'q' 键退出
            break
    else:
        break

# 释放资源
cap.release()
vw.release()
cv2.destroyAllWindows()
```
### 2.4 鼠标控制
&#8195;&#8195;`setMouseCallback` 是 OpenCV 中的一个函数，用于在指定的窗口上设置鼠标事件的回调函数。通过使用这个函数，你可以检测并处理用户在窗口中通过鼠标执行的各种操作，比如点击、移动、拖拽等。典型的应用包括在图像窗口中进行交互式的区域选择或标注。

```python
cv2.setMouseCallback(windowName, onMouse, param=None)
```
- `windowName`：窗口名称
- `onMouse`：这是你要设置的鼠标回调函数，每当鼠标事件发生时，这个函数就会被调用。
- `param`:：（可选）可以向回调函数传递的额外参数，默认值为 None。你可以利用这个参数传递一些自定义数据。

回调函数的格式如下：

```python
def onMouse(event, x, y, flags, param):
    # event 表示发生的鼠标事件
    # x, y 表示鼠标在窗口中的位置坐标
    # flags 表示鼠标事件相关的标志（如按住键的状态）
    # param 是通过 setMouseCallback 传递的参数
```
| event类型                  | 值 | 描述                     |
|--------------------------|----|--------------------------|
| EVENT_MOUSEMOVE          | 0  | 鼠标移动                 |
| EVENT_LBUTTONDOWN        | 1  | 按下鼠标左键             |
| EVENT_RBUTTONDOWN        | 2  | 按下鼠标右键             |
| EVENT_MBUTTONDOWN        | 3  | 按下鼠标中键             |
| EVENT_LBUTTONUP          | 4  | 左键释放                 |
| EVENT_RBUTTONUP          | 5  | 右键释放                 |
| EVENT_MBUTTONUP          | 6  | 中键释放                 |
| EVENT_LBUTTONDBLCLK      | 7  | 左键双击                 |
| EVENT_RBUTTONDBLCLK      | 8  | 右键双击                 |
| EVENT_MBUTTONDBLCLK      | 9  | 中键双击                 |
| EVENT_MOUSEWHEEL         | 10 | 鼠标滚轮上下滚动         |
| EVENT_MOUSEHWHEEL        | 11 | 鼠标左右滚动             |


| flags类型                  | 值  | 描述                   |
|--------------------------|-----|------------------------|
| EVENT_FLAG_LBUTTON       | 1   | 按下左键               |
| EVENT_FLAG_RBUTTON       | 2   | 按下右键               |
| EVENT_FLAG_MBUTTON       | 4   | 按下中键               |
| EVENT_FLAG_CRTLKEY       | 8   | 按下Ctrl键             |
| EVENT_FLAG_SHIFTKEY      | 16  | 按下Shift键            |
| EVENT_FLAG_ALTKEY        | 32  | 按下Alt键              |

以上内容详见[MouseCallback官方文档](https://docs.opencv.org/4.10.0/d7/dfc/group__highgui.html#gab7aed186e151d5222ef97192912127a4)

示例一：用鼠标在图像上绘制图形（线条、圆形、长方形）

```python
# 按下l, 拖动鼠标, 可以绘制直线.
# 按下r, 拖到鼠标, 可以绘制矩形
# 按下c, 拖动鼠标, 可以绘制圆. 拖动的长度可以作为半径.
import cv2
import numpy as np

curshape = 0										# 这是一个全局标志, 判断要画什么类型的图.
startpos = (0, 0)
#img = np.zeros((480, 640, 3), np.uint8)
img=np.full((480, 640, 3), 255, dtype=np.uint8)		# 创建纯白背景图

# 要监听鼠标的行为, 所以必须通过鼠标回调函数实现.
def mouse_callback(event, x, y, flags, userdata):
    # 引入全局变量
    global curshape, startpos
    if event == cv2.EVENT_LBUTTONDOWN:				# 左键按下        
        startpos = (x, y)							# 记录起始位置
        
    elif event ==0 and flags == 1: 					# 表示按下鼠标左键（1）并移动鼠标（0）
        if curshape == 0: # 画直线
            cv2.line(img, startpos, (x, y), (0, 0, 255), 1)
        elif curshape == 1: # 画矩形
            cv2.rectangle(img, startpos, (x, y), (0, 0, 255), 1)
        elif curshape == 2: # 画圆
            # 注意计算半径
            a = (x - startpos[0])
            b = (y - startpos[1])
            r = int((a ** 2 + b ** 2) ** 0.5)		 # 画圆的时候, 半径必须是整数           
            cv2.circle(img, startpos, r, (0, 0, 255), 1)
        else: # 按其他的按键
            print('暂不支持绘制其他图形')            
            
cv2.namedWindow('drawshape', cv2.WINDOW_NORMAL)		# 创建窗口
cv2.setMouseCallback('drawshape', mouse_callback)	# 设置鼠标回调函数

while True:
    cv2.imshow('drawshape', img)
    # 检测按键
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('l'):
        curshape = 0
    elif key == ord('r'):
        curshape = 1
    elif key == ord('c'):
        curshape = 2
        
cv2.destroyAllWindows()
```
示例二：按下鼠标右键，关闭显示窗口

```python
import cv2
import numpy as np

# flags鼠标的组合按键
def mouse_callback(event, x, y, flags, userdata):
	
    print(event, x, y, flags, userdata) 
    # 按下鼠标右键退出
    if event == 2:
        cv2.destroyAllWindows()
    

# 创建窗口
cv2.namedWindow('mouse', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mouse', 640, 360)

# 设置鼠标回调函数
cv2.setMouseCallback('mouse', mouse_callback, '123')

# 生成全黑的图片
img = np.zeros((360, 640, 3), np.uint8)
while True:
    cv2.imshow('mouse', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()
```

```python
0 1 15 0 123	# 移动鼠标
0 2 15 0 123
0 4 15 0 123
0 7 15 0 123
1 10 15 0 123   # 点击鼠标左键
4 10 15 0 123	# 释放鼠标左键
...
...
```
- 当鼠标有任何操作时，都会触发`mouse_callback`中的打印函数，打印出鼠标事件的值、鼠标坐标和自定义的`userdata`的值。
- `userdata`是传递给鼠标回调函数的自定义用户数据，代码中传入的是"123"，帮助你确认或使用一些外部信息。

###  2.5  trackbar
`trackbar` 是opencv 中的一个图形界面控件（类似滑块），可以用来动态调节参数。它通常与窗口一起使用，使得用户可以通过拖动滑块实时调整图像的参数值，其基础语法为：

```python
cv2.createTrackbar(trackbar_name, window_name, value, max_value, on_change)
```


- `trackbar_name`：滑动条的名称。
- `window_name`：滑动条所依附的窗口名称。必须是已经通过 `cv2.namedWindow()` 创建的窗口。
- `value`：滑动条的初始值。
- `max_value`：滑动条的最大值。
- `on_change`：回调函数，当滑动条的值发生变化时调用。

示例：使用滑块控制RGB色

```python
# trackbar的使用
import cv2 
import numpy as np

# 创建窗口
cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
cv2.resizeWindow('trackbar', 640, 480)

# 定义回调函数
def callback(value):
#     print(value)
    pass
    
# 创建3个trackbar
cv2.createTrackbar('R', 'trackbar', 0, 255, callback)
cv2.createTrackbar('G', 'trackbar', 0, 255, callback)
cv2.createTrackbar('B', 'trackbar', 0, 255, callback)

# 创建背景图片
img = np.zeros((480, 640, 3), np.uint8)

while True:
    # 获取当前trackbar的值
    r = cv2.getTrackbarPos('R', 'trackbar')
    g = cv2.getTrackbarPos('G', 'trackbar')
    b = cv2.getTrackbarPos('B', 'trackbar')
    
    # 用获取到的三个值修改背景图片颜色
    img[:] = [b, g, r]
    cv2.imshow('trackbar', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db62ae52ce8140a7a83def212969ac86.png#pic_center =600x)
## 三、基本图像操作

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from IPython.display import Image
%matplotlib inline
```
### 3.1 访问单个像素
>&#8195;&#8195;`像素`是构成数字图像的基本单位，像素处理是图像处理的基本操作。对像素的访问、修改，可以使用 `Numpy 方法直接访问数组元素`。
1. 读取原始棋盘图像

```python
# Read image as gray scale.
cb_img = cv2.imread("checkerboard_18x18.png",0)

# Set color map to gray scale for proper rendering.
plt.imshow(cb_img, cmap='gray')
print(cb_img)
```

```python
[[  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/454af92a472bbbef0ea921f3fb2c6f1e.png#pic_center )
2. 访问单个像素
- 要访问numpy矩阵中的任何像素，必须使用矩阵表示法，例如matrix[r，c]，其中r是行号，c是列号。还要注意，矩阵是0索引的。
- 例如，如果要访问第一个像素，则需要指定矩阵[0,0]。让我们看一些例子。我们将从左上角打印一个黑色像素，从上中心打印一个白色像素。

```python
# 打印第一个黑色box的第一个像素
print(cb_img[0,0])
# 打印第一个黑色box右边的第一个白色像素
print(cb_img[0,6])
```

```python
0
255
```
### 3.2 修改图像像素（切片）

```python
cb_img_copy = cb_img.copy()
cb_img_copy[2,2] = 200
cb_img_copy[2,3] = 200
cb_img_copy[3,2] = 200
cb_img_copy[3,3] = 200

# Same as above
# cb_img_copy[2:3,2:3] = 200

plt.imshow(cb_img_copy, cmap='gray')
print(cb_img_copy)
```

```python
[[  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0 200 200   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0 200 200   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6e84c3ef365c53adc531d3842bd7a1f4.png#pic_center )
### 3.3 裁剪图像（切片）

```python
img_NZ_bgr = cv2.imread("New_Zealand_Boat.jpg",cv2.IMREAD_COLOR)
img_NZ_rgb = img_NZ_bgr[:,:,::-1] #  通道维度进行反转，也就是BGR→RGB

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(img_NZ_rgb);plt.title("RGB");
plt.subplot(142);plt.imshow(img_NZ_bgr);plt.title("BGR");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7f27572143d55af44a08f5b3b59be66e.png#pic_center )
下面裁剪图片中间的某一部分：
>&#8195;&#8195;Numpy 多维数组的切片是原始数组的**浅拷贝**，切片修改后原始数组也会改变。推荐采用 `.copy()` 进行**深拷贝**，得到原始图像的副本。
```python
# 裁剪中心200×300大小的图像
cropped_region = img_NZ_rgb[200:400, 300:600].copy()
plt.imshow(cropped_region)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/050b13da9a07b9838666cffc8d70f58b.png#pic_center )
### 3.4 调整图像大小（cv2.resize）
&#8195;&#8195;[resize](https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)函数可以调整图像的大小，其语法为

```python
dst = resize( src, dsize[, dst[, fx[, fy[, interpolation]]]] )
```

- `scr/ dst`：输入输出图像，二者类型相同
- `dsize`： 输出图像的大小，二元元组 (width, height)
- `fx, fy`：x 轴、y 轴上的缩放比例，可选项（dsize = None时）
- `interpolation`：用于在调整图像大小时计算新像素值，整型，可选项
	- `cv2.INTER_LINEAR`：双线性插值（默认方法）
	- `cv2.INTER_AREA`：使用像素区域关系重采样，缩小图像时可以避免波纹出现
	- `cv2.INTER_NEAREST`：最近邻插值
	- `cv2.INTER_CUBIC`：4x4 像素邻域的双三次插值
	- `cv2.INTER_LANCZOS4`：8x8 像素邻域的Lanczos插值


>图片的shape一般是`(width, height,ndim)`，而resize中 `fx, fy`分别是`（height,width)`，两者正好是相反的。
```python
#  1.使用`fx, fy`指定缩放比例
resized_cropped_region_2x = cv2.resize(cropped_region,None,fx=2, fy=2)

# 2.使用dsize指定输出图像大小
resized_cropped_region = cv2.resize(cropped_region, dsize=(100,200), interpolation=cv2.INTER_AREA)

# 3.保持高宽比的同时调整大小，使得宽度为100
desired_width = 100
aspect_ratio = desired_width / cropped_region.shape[1]  		# 宽/高得到比例
desired_height = int(cropped_region.shape[0] * aspect_ratio)	# 根据比例计算要调整的高度
dim = (desired_width, desired_height)
# Resize image
keep_aspect_ratio_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(resized_cropped_region_2x);plt.title("resized_cropped_region_2x");
plt.subplot(142);plt.imshow(resized_cropped_region);plt.title("(resized_cropped_region");
plt.subplot(143);plt.imshow(keep_aspect_ratio_region);plt.title("aspect ratio");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aa9058f1d5651b53be35357c69f86525.png)

4. `Image`显示裁剪图片真实大小
```python
resized_cropped_region_2x = resized_cropped_region_2x[:,:,::-1]
cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

# 显示resize图片真实大小
Image(filename='resized_cropped_region_2x.png')
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa31188e420e19dbb2ccaf8b5dd5dad3.png#pic_center )
### 3.5 图像的翻转和旋转（cv2.flip，cv2.rotate）
&#8195;&#8195;函数[cv2.flip](https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441)可以进行图像的翻转，包括水平翻转（沿x轴）、垂直翻转（沿y轴）和水平垂直翻转（两个轴同时翻转），函数语法为：

```python
dst=flip(src, flipCode[, src])  
```

- `scr/ dst`：输入输出图像
-  `flipCode`：指定如何翻转的标志；0表示围绕x轴翻转，正值表示围绕y轴翻转。负值表示围绕两个轴翻转。

 

```python
img = cv2.imread("../images/Fractal03.png")  # 读取彩色图像(BGR)

imgFlip1 = cv2.flip(img, 0)  # 垂直翻转
imgFlip2 = cv2.flip(img, 1)  # 水平翻转
imgFlip3 = cv2.flip(img, -1)  # 水平和垂直翻转

plt.figure(figsize=(9, 6))
plt.subplot(221), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 原始图像
plt.subplot(222), plt.axis('off'), plt.title("Flipped Horizontally")
plt.imshow(cv2.cvtColor(imgFlip2, cv2.COLOR_BGR2RGB))  # 水平翻转
plt.subplot(223), plt.axis('off'), plt.title("Flipped Vertically")
plt.imshow(cv2.cvtColor(imgFlip1, cv2.COLOR_BGR2RGB))  # 垂直翻转
plt.subplot(224), plt.axis('off'), plt.title("Flipped Horizontally & Vertically")
plt.imshow(cv2.cvtColor(imgFlip3, cv2.COLOR_BGR2RGB))  # 水平垂直翻转
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f5ce644b1446ed849a74c430eaf394fd.png#pic_center )


cv2.rotate 用于旋转图像，其语法为：

```python
dst	=	cv.rotate(	src, rotateCode[, dst]	)
```
其中，rotateCode为旋转模式，主要有：
- `cv2.ROTATE_90_CLOCKWISE`: 顺时针旋转90度。
- `cv2.ROTATE_180`: 旋转180度。
- `cv2.ROTATE_90_COUNTERCLOCKWISE`: 逆时针旋转90度。

### 3.6 仿射变换
#### 3.6.1 仿射变换方法
&#8195;&#8195;[cv2.warpAffine](https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)是 OpenCV 中用于执行图像仿射变换的函数。仿射变换是一种二维几何变换，它能够保持直线、平行性和比例不变，这种变换包括平移、旋转、缩放和剪切等操作，其语法为：

```python
cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
```
- `src`：输入图像（可以是灰度图或彩色图）
-  `M`:    2x3 的仿射变换矩阵。这个矩阵定义了如何将输入图像的像素映射到输出图像。仿射变换矩阵的一般形式如下： $$M = \begin{bmatrix} a_{11} & a_{12} & b_1 \\ a_{21} & a_{22} & b_2 \end{bmatrix}$$ 其中 `a11, a12, a21, a22` 控制旋转、缩放和剪切变换，`b1, b2` 控制平移。
- `dsize`:输出图像的大小，格式为 `(width, height)`。
- `dst`（可选）:输出图像。你可以提前定义这个参数，但一般情况下不需要显式定义，函数会根据 `dsize` 自动创建。
- `flags`（可选）:插值方法的标志，含义和resize中的interpolation参数一样。
-  `borderMode`（可选）：像素点变换到图像边界以外时的边界填充方式，常见选项有：
   * `cv2.BORDER_CONSTANT`: 使用常数值填充（由 `borderValue` 指定）
   * `cv2.BORDER_REPLICATE`: 重复最近的边界像素
-  `borderValue`（可选）:当 `borderMode` 设置为 `cv2.BORDER_CONSTANT` 时，填充的颜色值（默认为 0）。

#### 2.6.2 仿射变换矩阵的获取
&#8195;&#8195;通常，你可以通过 OpenCV 的函数如 `cv2.getRotationMatrix2D()` 来获得仿射变换矩阵，也可以手动定义。例如，常见的仿射变换包括：

1. **平移**:
    
    $$M = \begin{bmatrix} 1 & 0 & tx \\ 0 & 1 & ty \end{bmatrix}$$
    
    其中 `tx` 和 `ty` 是平移量。
    
2. **旋转和缩放**: 使用 `cv2.getRotationMatrix2D` 生成，其语法为：

	```python
	M = cv2.getRotationMatrix2D(center, angle, scale)
	```
	
    * `center`: 旋转中心坐标
    * `angle`: 旋转角度，正值表示 逆时针 旋转，负值表示 顺时针旋转。旋转后图像可能会出现部分区域超出原始图像的范围，这时你可以根据需要调整输出图像的大小或者处理边界（如填充黑色、复制边缘等）。
    * `scale`: 缩放比例

二维旋转的仿射变换矩阵的一般形式如下：
	$$M = \begin{bmatrix} \alpha & \beta & (1-\alpha) \cdot center_x - \beta \cdot center_y \\ -\beta & \alpha & \beta \cdot center_x + (1-\alpha) \cdot center_y \end{bmatrix}$$

其中：

* $\alpha = \text{scale} \cdot \cos(\text{angle})$
* $\beta = \text{scale} \cdot \sin(\text{angle})$
3. **任意仿射变换**: 可以使用 `cv2.getAffineTransform` 函数，它通过指定输入图像的三个点及其对应的输出图像的三个点来计算变换矩阵，其语法为：

	```python
	M = cv2.getAffineTransform(srcPoints, dstPoints)
	```
	-  `srcPoints`: 输入图像中的三个点的坐标，通常为 2x3 的 NumPy 数组，表示变换前的三角形的三个顶点，例如：`np.float32([[x1, y1], [x2, y2], [x3, y3]])`
	-  `dstPoints`：目标图像中的三个点的坐标，表示变换后的对应三个点位置。
	- 返回一个 2x3 的仿射变换矩阵。



```python
# 1. 向右平移150像素

import cv2
import numpy as np

cat = cv2.imread('./cat.jpeg')

h, w, ch = cat.shape
M = np.float32([[1, 0, 150], [0, 1, 0]])			# 变换矩阵,最少是float32位
new_cat = cv2.warpAffine(cat,M, dsize=(w, h))		# 注意opencv中是先宽度, 后高度
cv2.imshow('Move', np.hstack((cat,new_cat)))

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf42f90af6944ab0ada6c2888e1fb24b.png#pic_center)

```python
# 2. 沿中心点逆时针旋转45度

cat = cv2.imread('./cat.jpeg')
h, w, ch = cat.shape
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
new_cat = cv2.warpAffine(dog, M, (w, h))

cv2.imshow('rotate', np.hstack((cat,new_cat)))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45ca0d13590c4523a234fd4d675a696f.png#pic_center )

```python
# 3. 通过三个点来确定变换矩阵
import cv2
import numpy as np

cat = cv2.imread('./cat.jpeg')
h, w, ch = cat.shape

src = np.float32([[100, 100], [200, 100], [200, 300]])
dst = np.float32([[100, 150], [200, 100], [200, 300]])
# 需要原始图片的三个点坐标, 和变换之后的三个对应的坐标
M = cv2.getAffineTransform(src, dst)
new_cat = cv2.warpAffine(cat, M, (w, h))

cv2.imshow('Transform', np.hstack((cat,new_cat)))

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bb340cf0964477bad2934840ab3fe27.png#pic_center )
### 3.7 透视变换
#### 3.7.1 参数解析
&#8195;&#8195;`cv2.warpPerspective` 是 OpenCV 中用于执行 **透视变换**（也称为投影变换）的函数。透视变换是一种更加复杂的几何变换，可以将图像从一个平面变换到另一个平面，不仅能改变图像的旋转、平移、缩放，还可以调整其视角（仿射变换无法改变视角）。函数定义：

```python
dst = cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
```
&#8195;&#8195;其参数含义和`cv2.warpAffine`相同，只不过变换矩阵`M`是一个3x3 的透视变换矩阵。该矩阵可以通过 `cv2.getPerspectiveTransform` 函数计算获得，用于定义输入图像到输出图像的像素映射关系，其语法为：

```python
M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
```

* `srcPoints`: 输入图像中的四个点，通常为一个 `4x2` 的 NumPy 数组。
* `dstPoints`: 输出图像中的四个点，表示变换后的四个点位置。

#### 3.7.2 透视变换的数学原理及应用场景
&#8195;&#8195;透视变换的基本原理是将图像中的四个点 $(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)$ 映射到目标图像中的四个点 $(x'_1, y'_1), (x'_2, y'_2), (x'_3, y'_3), (x'_4, y'_4)$ 。透视变换矩阵 `M` 是一个 3x3 的矩阵，形如：

$$M = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

该矩阵将输入图像中的任意点 $(x, y)$ 映射到目标图像中的点 $(x', y')$，变换关系为：

$$\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = M \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

其中，归一化后的坐标为 $x' = \frac{x'}{w}$，$y' = \frac{y'}{w}$。

 应用场景如下：

1. **图像校正**：比如校正因拍摄角度产生的透视畸变（如文档扫描时，调整角度使页面看起来为平面）。
2. **视角转换**：可以模拟从不同角度观察图像的效果。
3. **投影变换**：如在增强现实（AR）应用中，将二维图像投影到三维物体的表面。
4. **数据增强**：在机器学习中，透视变换可用于对训练数据进行数据增强。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import Image

img = cv2.imread('./123.png')
# 获取变换矩阵，src是原图的4个坐标
src = np.float32([[100, 1100], [2100, 1100], [0, 4000], [2500, 3900]])
dst = np.float32([[0, 0], [2300, 0], [0, 3000], [2300, 3000]])
M = cv2.getPerspectiveTransform(src, dst)

# 透视变换
new_img = cv2.warpPerspective(img, M, (2300, 3000))
print(img.shape,new_img.shape)
plt.figure(figsize=[20,15])
plt.subplot(121);plt.imshow(img[:,:,::-1]);plt.title("img");
plt.subplot(122);plt.imshow(new_img[:,:,::-1]);plt.title("new_img");
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dc416763785646c5bbb513508c4b67d5.png#pic_center )

## 四、 为图像添加注释（画线、圆、矩形和文本注释）

```python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)
from IPython.display import Image
```

```python
# 第三章的火箭发射场图不知道为啥算违规，所以这里拿第四章的图来用
image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

# 显示图片
plt.imshow(image[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/56037865f7ff5cdbc362dc3ceebe0b66.jpeg#pic_center )

### 4.1 画条线（cv2.line）
&#8195;&#8195;使用[cv2.line](https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2)函数可以在图片上画条线，其语法为：

```python
img = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
```

- `img`：要被画线的图像
- `pt1`：直线的一个端点，格式为坐标(x1, y1)
- `pt2`：直线的一个端点，格式为坐标(x2, y2)
- `color`：线条颜色
- `thickness`：线条厚度（正整数，负值报错），默认值为1，可选。
- `lineType`：线条类型
	- 8 (或cv2.LINE_8)：默认值，表示8-连通线。
	- 4 (或cv2.LINE_4)：4-连通线。
	- cv2.LINE_AA：抗锯齿线，这种类型的线看起来更平滑。
- `shift`: 表示坐标点的小数位数，默认为0。如果设置了 shift 参数，cv2.line 会假设输入的坐标已经乘以 2^shift（即左移了 shift 位）。
	- 如果 shift = 1，那意味着输入的坐标值已经乘以 2。
	- 如果 shift = 2，意味着坐标值已经乘以 4。

```python
imageLine = image.copy()

# 线段从(200,50)到(400,50)，颜色为青色
cv2.line(imageLine, (200, 50), (400, 50), (255, 255, 0), thickness=2, lineType=cv2.LINE_AA);
plt.imshow(imageLine[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c10aaf5f83c2533042f3e621de060ba1.png#pic_center )

```python
# 以 (200.5, 50.5) 和 (400.5, 50.5) 绘制一条线，使用 shift=1 来表示子像素坐标
cv2.line(image, (200*2, 50*2), (400*2, 50*2), (255, 255, 0), 2, shift=1)
```

### 4.2 圆（cv2.circle）和椭圆（cv2.ellipse）
&#8195;&#8195;使用 [cv2.circle](https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670)函数可以在图片上画个圆，其语法为：

```python
img = cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
```

其中 center, radius, color分别表示圆的圆心、半径和颜色。

```python
imageCircle = image.copy()
cv2.circle(imageCircle, (290,485), 30, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA);
plt.imshow(imageCircle[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6fdd5035991574fa9cf60a299e55a911.png#pic_center )
```python
cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)
```
- `center`: 椭圆中心的坐标，格式为(x, y)。
- `axes`: 椭圆主轴的长度，格式为(major_axis_length, minor_axis_length)。其中major_axis_length是长轴的长度，minor_axis_length是短轴的长度。两个长度一样就是圆形。
- `angle`: 旋转角度（逆时针方向），用于画斜着的椭圆。
- `startAngle` ， `endAngle`: 这椭圆弧的开始和结束角度。0度对应于椭圆的最右端点，角度沿逆时针方向增加，所以你可以只画部分椭圆。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/718457ca6abe4468a7e8e7edee007ce0.png#pic_center =500x)
### 4.3 画个矩形（cv2.rectangle）
&#8195;&#8195;使用[cv2.rectangle](https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9)函数可以在图像上绘制矩形，函数语法为：

```python
img = cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
```

其中，`pt1,pt2`为矩形顶点，通常使用左上角和右下角顶点。

```python
imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (350, 310), (550,430), (255, 0, 255), thickness=2, lineType=cv2.LINE_8);
plt.imshow(imageRectangle[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/38dcc8b336b5cc58ba2b1e13ad836836.png#pic_center )
### 4.4 添加文本注释（cv2.putText）
&#8195;&#8195;使用[cv2.putText](https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)函数可以在图像上添加文本注释（不支持中文字符），其语法为：

```python
img = cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
```

- `text`：要写入的文本字符串。
- `org`：文本字符串的左下角坐标，格式为 (x, y)
- `fontFace`：字体类型，例如 `cv2.FONT_HERSHEY_SIMPLEX`（推荐，抗锯齿效果好，尤其是在文本较大时）、`cv2.FONT_HERSHEY_PLAIN`（比前者更细）、`cv2.FONT_HERSHEY_DUPLEX` 等。
- `fontScale`：字体大小，相对于基本大小的一个比例因子。

```python
imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontFace = cv2.FONT_HERSHEY_PLAIN # 字体类型

# 2.3是字体比例因子，(0,255,255)表示黄色，线条厚度为2
cv2.putText(imageText,text,(150,580),fontFace, 2.3,(0,255,255),2,cv2.LINE_AA);

# Display the image
plt.imshow(imageText[:,:,::-1])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4af3e27bc68f2276a2d145d17afe6ba6.png#pic_center )
### 4.5 添加中文注释
&#8195;&#8195;在图像中添加中文字符，可以使用 python+opencv+PIL 实现（先安装pillow），或使用 python+opencv+freetype 实现。

```python
from PIL import Image, ImageDraw, ImageFont

imgBGR = cv2.imread("../images/imgLena.tif")  						# 读取彩色图像(BGR)
# 检查imgBGR是否为NumPy的ndarray类型，如果是，则将其从BGR格式转换为RGB格式，并转换为PIL图像对象。
if (isinstance(imgBGR, np.ndarray)):  								
    imgPIL = Image.fromarray(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
    
# 使用ImageDraw.Draw创建一个可以在PIL图像上绘制的对象，并加载一个中文字体（这里使用的是"simsun.ttc"字体文件）
drawPIL = ImageDraw.Draw(imgPIL)
fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
text = "OpenCV2021, 中文字体"
pos = (50, 20)  													# (left, top)，字符串左上角坐标
color = (255, 255, 255)  											# 字体颜色
textSize = 40														# 字体大小

drawPIL.text(pos, text, color, font=fontText)						# 在图像的指定位置绘制文本
imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR) 	# 转换回OpenCV图像的BGR格式

cv2.imshow("imgPutText", imgPutText)  	
key = cv2.waitKey(0)  					
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/247ad2d24b8827d93c2e11b4b5ea78e5.png#pic_center )

其中，`simsun.ttc`字体文件可以在C盘windows文件夹下的Fonts文件夹中 找到。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3bde5c57d0734eea82064099b16fb409.png#pic_center )



