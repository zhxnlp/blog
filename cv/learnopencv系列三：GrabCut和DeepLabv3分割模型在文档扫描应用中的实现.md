@[toc]
- [《opencv优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5502)
- [《learnopencv系列一：使用神经网络进行特征匹配（LoFTR、XFeat、OmniGlue）、视频稳定化、构建Chrome Dino游戏机器人》](https://blog.csdn.net/qq_56591814/article/details/143252588?spm=1001.2014.3001.5502)
- [《learnopencv系列二：U2-Net/IS-Net图像分割（背景减除）算法、使用背景减除实现视频转ppt应用》](https://blog.csdn.net/qq_56591814/article/details/143317678?spm=1001.2014.3001.5501)
 - [《learnopencv系列三：GrabCut和DeepLabv3分割模型在文档扫描应用中的实现》](https://blog.csdn.net/qq_56591814/article/details/143612087)
## 一、使用OpenCV实现自动文档扫描
>原文[《Automatic Document Scanner using OpenCV》](https://learnopencv.com/automatic-document-scanner-using-opencv/)、[GitHub代码](https://github.com/spmallick/learnopencv/tree/master/Automatic-Document-Scanner)、[Streamlit Web App](https://varunbal-document-scanning-app-uvmdn8.streamlitapp.com/)


&#8195;&#8195;**文档扫描** 是将纸质文档转换为数字形式的过程，这可以通过扫描仪或手机相机拍照来完成。本教程将使用计算机视觉和图像处理技术来实现这一过程，步骤如下：
- 使用形态学的闭运算操作，得到一个空白页面；
- 使用GrabCut分割技术去除背景，得到前景
- 使用Canny进行边缘检测
- 使用轮廓查找找出文档轮廓
- 找到文档的四个角点以及这些角点的目标坐标，计算出单应性矩阵
- 使用单应性矩阵执行透视变换，对齐后裁剪出文档
- 在Streamlit上打包和部署应用程序。

>- **图像基本处理**见[《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
>- **形态学操作、查找轮廓、计算轮廓周长面积、使用多边形近似查找轮廓**见[《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
>- **canny边缘检测**见[《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
>- **GrabCut图像分割原理与函数解析**见[《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
>- **单应性矩阵和仿射变换**见[《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
### 1.1 图片预处理

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 1. 读取图片，如果图片最大维度超过1080，则进行等比例缩放
img = cv2.imread('inputs/img22.jpg')

dim_limit = 1080
max_dim = max(img.shape)
if max_dim > dim_limit:
    resize_scale = dim_limit / max_dim
    img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

orig_img = img.copy()

# 2.重复迭代3次闭操作以从文档中移除文本，得到空白页面
kernel = np.ones((5,5),np.uint8)
close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
```
### 1.2 查找轮廓
```python
# 3.将角落20像素作为背景，GrabCut自动分割出前景，只留下文档。
mask = np.zeros(img.shape[:2],np.uint8)								
rect = (20,20,img.shape[1]-20,img.shape[0]-20)
# 进行grabCut分割，mask被更新，前景和可能的前景分别被标记为0和2
cv2.grabCut(close,mask,rect,None,None,5,cv2.GC_INIT_WITH_RECT)
# 将前景部分设置为白色
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_grabCut = cv2.bitwise_and(close, close, mask=mask2)

# 4. 高斯模糊+边缘检测+膨胀操作
gray = cv2.cvtColor(img_grabCut, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(gray, 0, 200)
canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
```
&#8195;&#8195;根据GrabCut图像分割的原理，rect 应该完整包裹文档，但不需要紧贴边缘。即使框稍大、包含一些背景，只要背景均匀，GrabCut 仍能有效区分前景（文档）和背景。关键是确保文档在 rect 内，因为所有框外区域会被视为背景。然后使用Canny边缘检测，就可以得到文档的精确轮廓。

&#8195;&#8195;Canny 检测出的边缘通常比较细，且有时会不连续，出现断裂。膨胀操作（cv2.dilate）会加粗并连接断裂的边缘，使它们在图像中更显眼，便于后续的轮廓检测。

&#8195;&#8195;在获得文档的边缘后，进行轮廓检测以获得这些边缘的轮廓。那为什么不跳过边缘检测直接检测轮廓。不推荐这样做，因为GrabCut有时会不经意地保留部分背景。

完成轮廓检测后：

- 根据大小对检测到的轮廓进行排序
- 只保留最大的检测轮廓
- 然后在空白画布上绘制这个最大的检测轮廓

```python
# 空白画布。
con = np.zeros_like(img)
# 查找所有轮廓，不保留轮廓之间的层次关系，并且保留轮廓上的所有点
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# 保留图像中面积最大的前5个轮廓
page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

plt.figure(figsize=[12,12]);
plt.subplot(231); plt.imshow(img[:,:,::-1]);plt.axis('off');plt.title("img");
plt.subplot(232); plt.imshow(close[:,:,::-1]);plt.axis('off');plt.title("close");
plt.subplot(233); plt.imshow(img_grabCut[:,:,::-1]);plt.axis('off');plt.title("img_grabCut");
plt.subplot(234); plt.imshow(gray,cmap='gray');plt.axis('off');plt.title("gray");
plt.subplot(235); plt.imshow(canny ,cmap='gray');plt.axis('off');plt.title("canny");
plt.subplot(236); plt.imshow(con ,cmap='gray');plt.axis('off');plt.title("con");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb7b5cef99be41b9b17d218d7bb25d91.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dbef22e1109f4f1695dde3d7f17b8839.png)

### 1.3 检测角点
&#8195;&#8195;此时的轮廓不是一个标准的矩形，需要使用多边形逼近（cv2.approxPolyDP）的方式找出近似轮廓，然后找到其四个角点。

```python
# 空白画布。
con = np.zeros_like(img)
# 循环遍历轮廓。
for c in page:
  # 计算轮廓c的周长，True表示轮廓是闭合的。如果为 False，表示轮廓是开放的，计算曲线长度
  # 多边形近似基于DP算法， epsilon是其最小阈值。True也表示轮廓是闭合的
  epsilon = 0.02 * cv2.arcLength(c, True)
  corners = cv2.approxPolyDP(c, epsilon, True)
  # 如果我们近似的轮廓有四个点
  if len(corners) == 4:
      break

# 画出最大近似轮廓及四个角点
cv2.drawContours(con, c, -1, (0, 255, 255), 3)
cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
# 将四个角点使用concatenate拼成一个列表然后排序，最后再转为列表格式
corners = sorted(np.concatenate(corners).tolist())

# 显示角点
for index, c in enumerate(corners):
  character = chr(65 + index)
  cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

plt.imshow(con ,cmap='gray');plt.axis('off');plt.title("con");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3fcc2a2d02249dca8da19ea9291fe02.png#pic_center)

### 1.4 仿射变换
1. 此时得到的角点顺序是乱的，要对其进行排序
2. 根据文档的四个角点及转换后的四个角点（图片边缘的四个点），求得单应性矩阵
3. 使用单应性矩阵对图片进行进行仿射变换



```python
def order_points(pts):
    '''重新排列坐标以顺序：
      左上角，右上角，右下角，左下角'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # 左上角点将具有最小的总和。
    rect[0] = pts[np.argmin(s)]
    # 右下角点将具有最大的总和。
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # 右上角点将具有最小的差异。
    rect[1] = pts[np.argmin(diff)]
    # 左下角将具有最大的差异。
    rect[3] = pts[np.argmax(diff)]
    # 返回有序坐标。
    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # 寻找最大宽度。
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 寻找最大高度。
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 最终目标坐标。
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)
```

```python
corners = order_points(corners)
destination_corners = find_dest(corners)
# 获取单应性矩阵
M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
# 进行仿射变换
warped_img = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),flags=cv2.INTER_LINEAR)

plt.figure(figsize=[20,12]);
plt.subplot(121); plt.imshow(img[:,:,::-1]);plt.title("img");
plt.subplot(122); plt.imshow(warped_img[:,:,::-1]);plt.title("warped_img");
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f63d2cd47ed8499a8798cb5ce75d3686.png#pic_center)

### 1.5  Streamlit Web App

&#8195;&#8195;现在我们的文档扫描器已经准备好了，让我们看看这个简单的[Streamlit Web App](https://varunbal-document-scanning-app-uvmdn8.streamlitapp.com/)。除了自动扫描和对齐，我们还添加了手动设置文档边缘的功能。
#### 1.5.1 设置扫描函数和图像下载链接函数

```python
### 扫描文档的主要函数：
def scan(img):
    img = cv2.imread(img)


	dim_limit = 1080
	max_dim = max(img.shape)
	if max_dim > dim_limit:
	    resize_scale = dim_limit / max_dim
	    img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
	
	orig_img = img.copy()
	
	# 2.重复迭代3次闭操作以从文档中移除文本，得到空白页面
	kernel = np.ones((5,5),np.uint8)
	close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)

	# 3.将角落20像素作为背景，GrabCut自动分割出前景，只留下文档。
	mask = np.zeros(img.shape[:2],np.uint8)								
	rect = (20,20,img.shape[1]-20,img.shape[0]-20)
	# 进行grabCut分割，mask被更新，前景和可能的前景分别被标记为0和2
	cv2.grabCut(close,mask,rect,None,None,5,cv2.GC_INIT_WITH_RECT)
	# 将前景部分设置为白色
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	
	img_grabCut = cv2.bitwise_and(close, close, mask=mask2)
	
	# 4. 高斯模糊+边缘检测+膨胀操作
	gray = cv2.cvtColor(img_grabCut, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (11, 11), 0)
	canny = cv2.Canny(gray, 0, 200)
	canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

	# 空白画布。
	con = np.zeros_like(img)
	# 寻找检测到的边缘的轮廓。
	contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	# 只保留最大的检测轮廓。
	page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
	con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

	con = np.zeros_like(img)
	# 循环遍历轮廓。
	for c in page:
	  # 计算轮廓c的周长，True表示轮廓是闭合的。如果为 False，表示轮廓是开放的，计算曲线长度
	  # 多边形近似基于DP算法， epsilon是其最小阈值。True也表示轮廓是闭合的
	  epsilon = 0.02 * cv2.arcLength(c, True)
	  corners = cv2.approxPolyDP(c, epsilon, True)
	  # 如果我们近似的轮廓有四个点
	  if len(corners) == 4:
	      break

	corners = sorted(np.concatenate(corners).tolist())
	
	corners = order_points(corners) 
    destination_corners = find_dest(corners) 
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),flags=cv2.INTER_LINEAR)
    
    return final
```

生成链接以下载特定图像文件的函数：

```python
def get_image_download_link(img, filename, text):
	# 创建一个 BytesIO 对象，作为图像数据的临时存储，然后将图像存入其中。
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    # 从 buffered 对象中获取图像数据（以字节形式），然后使用 base64 编码这些数据
    # 将编码后的字符串解码为 UTF-8 格式的字符串，以便在 HTML 中使用。
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # 生成HTML 链接字符串
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href
```
&#8195;&#8195;这段代码的目的是为了生成一个可以直接在网页上点击下载图像文件的 HTML 链接。用户点击这个链接时，浏览器会下载并保存名为 filename 的图像文件，文件内容是 img 参数所代表的图像。
#### 1.5.2 streamlit app

1. 在Streamlit 应用的侧边栏设置标题为“文档扫描器”

```python
st.sidebar.title('文档扫描器')
```



2. 在侧边栏创建一个文件上传器，允许用户上传 PNG 或 JPG 格式的图像文件

```python
uploaded_file = st.sidebar.file_uploader("上传文档图像：", type=["png", "jpg"])
# 初始化变量image和final,用于存储上传的图像和处理后的图像
image = None
final = None
```

3. 创建两列以并排显示输入图像和扫描文档，用于自动扫描模式。

```python
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # 如果用户上传了文件，将文件转换为NumPy 数组。
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # 使用 OpenCV 的 imdecode 函数将文件内容解码为图像
    image = cv2.imdecode(file_bytes, 1)
	
	# 在侧边栏创建一个复选框，允许用户选择是否进行手动调整
    manual = st.sidebar.checkbox('手动调整', False)
	# 计算调整后的图像高度和宽度，保持宽高比，使宽度为400像素
    h, w = image.shape[:2]
    h_, w_ = int(h * 400 / w), 400
```

&#8195;&#8195;为了提供手动选择文档边缘的功能，使用`streamlit_drawable_canvas`。如果切换到手动模式，你就不需要进行角点检测。用户可以通过点击角点简单地追踪文档的边缘，然后执行透视变换以获得对齐的文档。

```python
if manual:
    st.subheader('选择4个角点')
    # 显示 Markdown 文本，提供用户操作提示。
    st.markdown('### 双击重置上一个点，右键选择')

    # 创建一个画布组件，允许用户绘制多边形并选择图像的四个角点。
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 固定填充颜色，带一些透明度
        stroke_width=3,
        background_image=Image.open(uploaded_file).resize((h_, w_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='polygon',
        key="canvas",
    )
    # 在侧边栏显示提示，询问用户是否对选择的角点满意。
    st.sidebar.caption('对手动选择满意吗？')
    # 如果用户点击“获取扫描结果”按钮，执行以下操作
    if st.sidebar.button('获取扫描结果'):
        # 从画布结果中提取用户选择的四个角点，并对其进行排序。
        points = order_points([i[1:3] for i in canvas_result.json_data['objects'][0]['path'][:4]])
        # 将角点坐标缩放到原始图像大小。
        points = np.multiply(points, w / 400)
		# 调用 find_dest 函数，确定目标图像的四个角点。
        dest = find_dest(points)

        # 获取单应性矩阵。
        M = cv2.getPerspectiveTransform(np.float32(points), np.float32(dest))
        # 使用单应性进行透视变换。
        final = cv2.warpPerspective(image, M, (dest[2][0], dest[2][1]), flags=cv2.INTER_LINEAR)
        # 在页面上显示处理后的图像
        st.image(final, channels='BGR', use_column_width=True)
```

&#8195;&#8195;如果未选择手动模式，则使用scan函数执行前面讨论的所有步骤，并提供扫描文档。这些并排显示在streamlit应用程序上。

```python
else:
	# 在两个列中分别显示原始图像和扫描结果
    with col1:
        st.title('输入')
        st.image(image, channels='BGR', use_column_width=True)
    with col2:
        st.title('扫描')
        final = scan(image)
        st.image(final, channels='BGR', use_column_width=True)
```

扫描完成后，显示下载结果。

```python
if final is not None:
    # 处理后的图像转换为 PIL 图像对象，并调整颜色通道顺序
    result = Image.fromarray(final[:, :, ::-1])
    # 在侧边栏显示一个下载链接，允许用户下载处理后的图像
    st.sidebar.markdown(get_image_download_link(result, 'output.png', '下载 ' + '输出'),
                        unsafe_allow_html=True)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71dcd2afe3844549b9e4a125e75e7ec2.png#pic_center =800x)

&#8195;&#8195;这段代码实现了一个完整的文档扫描器功能，包括图像上传、手动或自动处理以及结果下载。用户可以选择手动调整角点或自动扫描，最后下载处理后的图像。

#### 1.5.3 测试结果

&#8195;&#8195;我们在23种不同的背景和各种方向上进行了测试，在几乎所有情况下，自动文档扫描器都能很好地工作，即使在背景为白色且与文档颜色相似的情况下。而如果选择使用其他方法（例如阈值处理），可能会遇到问题。另外，GrabCut算法的速度取决于图像的大小，对于非常高分辨率的图像可能会慢一些。但对于中等大小的图像，它只需要几秒钟就能完成扫描。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b3baafb158104c13be370650537c7216.png#pic_center =600x)





使用GrabCut进行文档扫描的局限性主要有以下几点：

1. **图像部分缺失**：如果文档的一部分（如角落）超出图像范围，无法查找出完整的文档轮廓，GrabCut无法正常检测出所有的前景部分。
2. **背景噪声影响**：在背景中有较多噪声，尤其是当噪声与文档边缘混淆时，可能会检测到许多不必要的边缘，会被认为是文档轮廓的一部分，干扰文档的识别。
3. **边缘与背景不区分**：如果文档的边缘与背景颜色或纹理相似，轮廓检测可能无法正常工作，导致检测失败。

由于这些局限性，深度学习技术因其鲁棒性而被更广泛地应用于文档扫描领域。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c49877041d4e4d589df6ea8778df358c.png)
## 二：DeepLabv3文档分割
>- 原文：[Document Segmentation Using Deep Learning in PyTorch](https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/)、[GitHub代码](https://github.com/spmallick/learnopencv/tree/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3)
>- 基础知识：[《Image segmentation》](https://learnopencv.com/image-segmentation/)、[《 getting started with PyTorch》](https://learnopencv.com/learn-pytorch/)、[《DeepLabv3 architecture 》](https://learnopencv.com/deeplabv3-ultimate-guide/)、图像数据集：[DocUNet: Document Image Unwarping via A Stacked U-Net](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf)

### 2.1 项目背景
&#8195;&#8195;文档扫描是一个背景分割问题，上一章我们使用传统的图像处理方法（形态学处理+GrabCut分割+canny检测等技术）在某些情况下会失败，主要原因在于我们对文档结构和位置以及背景变化的偏见性假设。

&#8195;&#8195;要解决这个问题，构建一个鲁棒性的文档扫描器，使其在多种场景下都能高效工作，所使用的算法必须避免带有偏见的假设。我们的解决方案采用了基于深度学习的图像分割模型`deeplabv3_mobilenet_v3_large`，该模型在不同场景下进行了训练，以构建鲁棒的分割模型。

&#8195;&#8195;[DeepLabv3](https://learnopencv.com/deeplabv3-ultimate-guide/)是一种语义分割模型架构，可以使用不同的主干网络来增强其特征提取能力，被认为是基于深度学习的语义分割模型的重要里程碑。自2017年发布以来，该架构因其出色的速度、精度和简单性而迅速受到欢迎。


&#8195;&#8195;[MobileNetV3-Large](https://blog.csdn.net/qq_56591814/article/details/126901999)是一种轻量级的卷积神经网络模型，专注于高效的图像特征提取，它属于MobileNet系列，专为移动和嵌入式设备设计，具有较小的模型大小和较高的推理速度，所以使用其作为DeepLabv3架构的骨干网络（另外还有Resnet50和Resnet101两个选择）。
### 2.2 合成数据集
&#8195;&#8195;在项目初期，定义问题陈述后，接下来的关键步骤是制定数据集的收集方案，即如何为任务收集数据集。任何机器学习或深度学习项目所花费的时间一般都有80%用于数据集的收集、准备和分析。剩下的时间只有20%用于实际的培训和改进。

&#8195;&#8195;为了创建稳健的文档分割模型，需要包含在多种背景和不同角度拍摄的各种文档的数据集。收集这样的数据集非常耗时，因此我们采取合成数据集的方式，尽可能模拟在捕捉真实世界图像时可能遇到的问题（如运动模糊、相机噪声等）。



### 2.2.1 图像收集与预处理
- **文档图像**：需要各种类型的文档图像，将其最大尺寸调整为640以保持结构并减少处理时间，并为每个裁剪过的文档生成值为255的掩码（掩码和文档大小一致）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/589db6de8af84aebae7e5b91f8c742c7.png#pic_center =600x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d3807542fc048a180c2afef4d107017.png#pic_center =600x)


- **背景图像**：合成数据集的目标之一是模拟文档被放置在不同背景下的情况，通过谷歌图片搜索下载，最终筛选得到1055张背景图像。
	>&#8195;&#8195;为了简化下载过程，使用了[Google Images Download仓库](https://github.com/Joeclinton1/google-images-download)的一个分支，通过搜索“桌面俯视图”、“层压板近景”、“木桌近景”等关键词，下载具有不同纹理和颜色的图像，以增强数据集的多样性。所有图片都被下载或转换为JPG格式。经过去重处理后，最终得到1055张背景图像。
- **数据集示例**：SmartDoc QA数据集中包含4260张原始图像，其中85张文档经过手动标注和提取。

#### 2.2.2 合成数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12b26bf01352422eaa04b50bc61a45eb.png#pic_center)

- **随机变换**：为了提高鲁棒性，每个文档-掩码对会进行6次**随机透视变换**（文档还有50%概率会进行**随机亮度和对比度调整**），且每次变换有70%的概率会被应用。
- **合并背景图**：为每个变换后的文档随机选择6张背景图像，将这些变换后的文档与选定的背景图像合并，生成新的合成图像和对应的掩码。
- **数据增强**：每个合并后的输出对（图像和掩模）都经过进一步的增强，以尽可能接近地复制真实世界的场景。增强操作包括：水平和垂直翻转、随机旋转、颜色抖动、通道混洗、亮度对比度调整、压缩/噪声/运动模糊、阴影/光晕/RGB偏移、光学失真等（使用Albumentations库实现）。

最终生成8058对图像 - 掩码，其中6715对用于训练，1343对用于验证。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60f399a9e47b4554a8ca311ceb2104ea.png#pic_center)
### 2.3 项目代码
>完整代码见[GitHub代码](https://github.com/spmallick/learnopencv/tree/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3)

整个项目流程如下：
* 合成数据集
* 创建自定义Dataset类生成器，负责加载和预处理图像-掩码对。
* 加载深度学习模型`deeplabv3_mobilenet_v3_large`
* 选择合适的损失函数和评估指标。通常图像分割常用的有**交并比（IoU）**和**Dice系数**，逐个测试。
* 训练定制的语义分割模型，并将其与上一章传统文档扫描器进行对比测试。

#### 2.3.1 自定义Dataset类
&#8195;&#8195;创建自定义Dataset类以加载图像 - 掩码对，并将其转换为适当的格式。除了图像的预处理变换外，训练集和验证集的其他步骤是相同的。

- 训练集额外应用了图像增强操作“RandomGrayscale”，即有40%的概率将图像转换为灰度，增加训练难度。
>&#8195;&#8195;这项决定是基于多次实验观察到的结果：单独训练灰度图像或RGB图像的模型虽然表现良好，但在一些情况下会出现对方处理更好的情况。
- 每个图像的掩码包含两个通道：一个表示背景，另一个表示文档。
- 返回的所有图像和掩码首先被缩放到[0, 1]范围内。图像进一步根据ImageNet的均值和标准差进行标准化。

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import albumentations as A
import PIL

from torchvision.utils import make_grid, save_image
from torchmetrics import MeanMetric
from livelossplot import PlotLosses 

# 定义训练时的图像转换操作
def train_transforms(mean=(0.4611, 0.4359, 0.3905), 
                      std=(0.2193, 0.2150, 0.2109)):
    transforms = torchvision_T.Compose([
        torchvision_T.ToTensor(),  						# 将图片转换为Tensor
        torchvision_T.RandomGrayscale(p=0.4),  			# 以40%的概率将图片转换为灰度图
        torchvision_T.Normalize(mean, std),  			# 归一化处理，参数为均值和标准差
    ])
    return transforms

# 定义非训练时的图像转换操作
def common_transforms(mean=(0.4611, 0.4359, 0.3905), 
                       std=(0.2193, 0.2150, 0.2109)):
    transforms = torchvision_T.Compose([
        torchvision_T.ToTensor(),  
        torchvision_T.Normalize(mean, std),  
    ])
    return transforms

# 定义用于图像分割的数据集类
class SegDataset(Dataset):
    def __init__(self, *, 
                 img_paths,  							
                 mask_paths,  							
                 image_size=(384, 384),  				
                 data_type="train"  					
    ):
 
        self.data_type  = data_type  					# 数据类型，训练或非训练
        self.img_paths  = img_paths  					# 图片路径
        self.mask_paths = mask_paths  					# 掩码路径
        self.image_size = image_size  					# 图片尺寸
 
        # 根据数据类型选择不同的转换操作
        if self.data_type == "train":
            self.transforms = train_transforms()
        else:
            self.transforms = common_transforms()

    # 读取文件并进行处理
    def read_file(self, path):
        file = cv2.imread(path)[:, :, ::-1]  			# 读取图片并转换颜色通道顺序，BGR转RGB
        file = cv2.resize(file, self.image_size,  		
                          interpolation=cv2.INTER_NEAREST)
        return file

    # 获取数据集中的样本数量
    def __len__(self):
        return len(self.img_paths)

    # 获取单个样本
    def __getitem__(self, index):
        image_path = self.img_paths[index]  			# 根据序号依次获取单张图片路径
        image = self.read_file(image_path)  			# 读取理图片
        image = self.transforms(image)  				# 应用转换操作

        mask_path = self.mask_paths[index]  
        gt_mask = self.read_file(mask_path).astype(np.int32)  
		# 创建一个空的掩码数组
		# self.image_size是一个元组，包含了图像的宽度和高度（例如(384, 384)）。
		# *操作符在这里用于将元组解包为独立的参数，所以(*self.image_size, 2)实际上变成了(384, 384, 2)
        _mask = np.zeros((*self.image_size, 2), dtype=np.float32)  

        # 背景，将gt_mask中第一个通道值为0的像素在_mask的第一个通道中标记为1.0，其他像素标记为0.0
        _mask[:, :, 0] = np.where(gt_mask[:, :, 0] == 0, 1.0, 0.0) 

        # 文档，将gt_mask中第一个通道值为255的像素在_mask的第一个通道中标记为1.0，其他像素标记为0.0
        _mask[:, :, 1] = np.where(gt_mask[:, :, 0] == 255, 1.0, 0.0)  

        mask = torch.from_numpy(_mask).permute(2, 0, 1)  
        return image, mask  							
```
- `torch.from_numpy`：将NumPy数组转换为PyTorch张量
- `.permute(2, 0, 1)`：原始的_mask数组维度是 **(高度, 宽度, 通道数)**，这在NumPy中是常见的图像表示方式。permute函数将维度重新排列为 **(通道数, 高度, 宽度)**，这是PyTorch中的标准图像张量格式。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a0a32b381534cdf858b7e39f729c834.png)
#### 2.3.2 加载预训练的DeeplabV3模型
我们将微调所有的模型层，因为我们的目标类与用于训练模型的类有很大的不同。

```python
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large  
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101  

def prepare_model(backbone_model="mbv3", num_classes=2):
    weights = 'DEFAULT'  					# 使用预训练权重初始化模型
    if backbone_model == "mbv3":  
        model = deeplabv3_mobilenet_v3_large(weights=weights)  
    elif backbone_model == "r50": 
        model = deeplabv3_resnet50(weights=weights)  
    elif backbone_model == "r101": 
        model = deeplabv3_resnet101(weights=weights)  
    else:
        raise ValueError("传入的骨干网络参数不正确，必须是'mbv3', 'r50' 或 'r101' ")  

    # 更新输出层的输出通道数为类别数，为每个像素提供一个类别标签
    model.classifier[4] = nn.LazyConv2d(num_classes, 1)  		# 更新主分类器的最后一层
    model.aux_classifier[4] = nn.LazyConv2d(num_classes, 1)  	# 更新辅助分类器的最后一层

    return model  

model = prepare_model(num_classes=2)
model.train() 

# 传递一个随机生成的输入张量，模拟一次前向传播
out = model(torch.randn((2, 3, 384, 384))) 
print(out['out'].shape)  
```
#### 2.3.3 损失函数和度量函数
图像分割常用的损失函数有IoU和Dice：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c1f1f090a245474288df8dd0cff5efa6.png)

- **交并比IoU**：预测分割区域与真实分割区域之间的重叠部分（交集）与它们的并集之比，不会受到某一类别中像素比例失衡的影响
- **Dice系数**：类似于F1分数，它简单地表示为重叠区域的两倍除以真值和预测中像素的总数。

&#8195;&#8195;这两种指标的范围都在0到1之间，且呈正相关；不同点在于对错误预测的惩罚。IoU对假阳性（FP）和假阴性（FN）的惩罚比Dice多一倍。

```python
# 定义中间度量计算函数，用于计算IoU或Dice系数
def intermediate_metric_calculation(predictions, targets, use_dice=False, smooth=1e-6, dims=(2,3)):
    # dims对应于图像的高度和宽度：[B, C, H, W]。
    # 交集：|G ∩ P|。形状：(batch_size, num_classes)
    # smooth是平滑参数，用于避免除以零的情况
    intersection = (predictions * targets).sum(dim=dims) + smooth 
  
    # 总和：|G| + |P|。形状：(batch_size, num_classes)。
    summation = (predictions.sum(dim=dims) + targets.sum(dim=dims)) + smooth
  
    if use_dice:
        # Dice形状：(batch_size, num_classes) 
        metric = (2.0 * intersection) / summation
    else:
        # 并集。形状：(batch_size, num_classes)
        union = summation - intersection
  
        # IoU形状：(batch_size, num_classes)
        metric = intersection / union
         
    # 计算剩余轴上的均值（批量和类别）。 形状：标量
    total = metric.mean()
    return total
```

```python
# 定义损失函数
class Loss(nn.Module):
    def __init__(self, smooth=1e-6, use_dice=False):
        super().__init__()
        self.smooth = smooth
        self.use_dice = use_dice


    def forward(self, predictions, targets):
        # predictions  --> (B, #C, H, W) 未归一化的预测结果
        # targets      --> (B, #C, H, W) 独热编码的真实标签	

        # 将模型预测结果归一化。
        predictions = torch.sigmoid(predictions)

        # 计算两个通道的逐像素损失。形状：标量
        pixel_loss = F.binary_cross_entropy(predictions, targets, reduction="mean")
        # 根据是否使用Dice系数计算mask损失。
        mask_loss  = 1 - intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)
        # 返回总损失
        total_loss = mask_loss + pixel_loss
        
        return total_loss
```

```python
# 将输入的多类别预测矩阵转为onehot形式
def convert_2_onehot(matrix, num_classes=3):
    '''
    在通道维度上执行独热编码
    '''
    # 调整矩阵的维度顺序，将通道维度移动到最后
    matrix = matrix.permute(0, 2, 3, 1)	
    # 在最后一个维度（通道维度）上找到最大值的索引，即每个像素的类别索引			
    matrix = torch.argmax(matrix, dim=-1)
    # 使用找到的类别索引，生成独热编码
    matrix = torch.nn.functional.one_hot(matrix, num_classes=num_classes)
    # 再次调整矩阵的维度顺序，将类别维度移动到通道维度的位置
    matrix = matrix.permute(0, 3, 1, 2)

    return matrix
    
class Metric(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6, use_dice=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.use_dice    = use_dice
    
    def forward(self, predictions, targets):
        # predictions  --> (B, #C, H, W) 未归一化的预测结果
        # targets      --> (B, #C, H, W) 独热编码的真实标签

        # 将未归一化的预测结果转换为通道上的独热编码。
        predictions = convert_2_onehot(predictions, num_classes=self.num_classes) 
        metric = intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)
        
        return metric
```

#### 2.3.4 读取数据集，查看图像预处理效果
```python
# 1. 固定随机种子
def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed = 41
seed_everything(seed)

DATA_DIR = r"./document_dataset_resized"
BESTMODEL_PATH = r"model_mbv3_iou_mix_2C_aux.pth"  # path to save model weights

IMAGE_SIZE = 384
NUM_WORKERS = 2
```
```python
# 2.下载数据集
!pip install -qU livelossplot
!pip install -qU torchmetrics

# final set
# https://drive.google.com/file/d/1tcPv-KT09eMgYcM3YQVoRmIjdEb_Wgtc/view?usp=sharing
# 使用gdown命令从Google Drive下载数据集并解压
!gdown 1tcPv-KT09eMgYcM3YQVoRmIjdEb_Wgtc
!unzip -qq './document_dataset_resized.zip'
```
>- `livelossplot` 是一个用于深度学习训练过程中实时可视化训练过程的库，可以实时显示训练和验证过程中的损失值、准确率等指标，支持多种图表类型，如线图、条形图等，可以在Jupyter Notebook中直接显示图表，方便用户监控训练进度。
>- `torchmetrics` 是一个为PyTorch提供的指标计算库，提供了一系列用于评估机器学习模型性能的指标，如准确率、F1分数、精确率、召回率等。
```python
# 3. 数据预处理
def get_dataset(data_directory, batch_size=16):

    train_img_dir = os.path.join(data_directory, "train", "images")
    train_msk_dir = os.path.join(data_directory, "train", "masks")

    valid_img_dir = os.path.join(data_directory, "valid", "images")
    valid_msk_dir = os.path.join(data_directory, "valid", "masks")
 
 
    train_img_paths = [os.path.join(train_img_dir, i) for i in os.listdir(train_img_dir)]
    train_msk_paths = [os.path.join(train_msk_dir, i) for i in os.listdir(train_msk_dir)]

    valid_img_paths = [os.path.join(valid_img_dir, i) for i in os.listdir(valid_img_dir)]
    valid_msk_paths = [os.path.join(valid_msk_dir, i) for i in os.listdir(valid_msk_dir)]

    train_ds = SegDataset(img_paths=train_img_paths, mask_paths=train_msk_paths, data_type="train")
    valid_ds = SegDataset(img_paths=valid_img_paths, mask_paths=valid_msk_paths, data_type="valid")

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True,  pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    return train_loader, valid_loader

train_loader, valid_loader = get_dataset(DATA_DIR, batch_size=1)

for i, j in valid_loader:
    print(i.shape, j.shape, j.dtype)
    break
```

```python
torch.Size([1, 3, 384, 384]) torch.Size([1, 2, 384, 384]) torch.float32
```

```python
# 4. 定义去归一化函数，用于将图像张量恢复到原始的图像数据范围
def denormalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """使用均值和标准差对图像张量进行去归一化"""

    for c in range(3):
        tensors[:,c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0., max=1.)

# 3. 查看图像预处理效果
for image, mask in valid_loader:
    # 对图像进行去归一化处理
    image = denormalize(image)
    # 取出第一个图像
    image = image[0]
    x = image.permute(1, 2, 0).numpy()
    labels = mask[0]
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1);plt.imshow(x);plt.title('Image')          
    plt.subplot(1, 3, 2);plt.imshow(labels[0].numpy(), cmap='gray');plt.title("Background Mask")    
    plt.subplot(1, 3, 3);plt.imshow(labels[1].numpy(), cmap='gray');plt.title("Document Mask")    
    plt.show()        
    plt.close()

    break
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cca7209f70d14497be8a89b5dc3e6c83.png#pic_center)

#### 2.3.5 定义训练辅助函数
为了确保代码在无论是CPU还是GPU上都能正常工作，定义以下函数和类。
```python
def to_device(data, device):
    """将张量(s)移动到选择的设备"""
    # 如果数据是列表或元组，递归地将每个元素移动到设备
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    # 否则，直接将单个张量移动到设备
    return data.to(device, non_blocking=True)

# 定义一个类，用于包装数据加载器，使其自动将数据移动到指定的设备
class DeviceDataLoader():
    """包装一个数据加载器，将数据移动到设备"""
    def __init__(self, dl, device):
        self.dl = dl				# 原始的数据加载器
        self.device = device		# 目标设备
        
    def __iter__(self):
        """产生一个批次的数据，并将数据移动到设备"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """批次数量"""
        return len(self.dl)

# 获取默认的计算设备
def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
def step(model, epoch_num=None, loader=None, optimizer_fn=None, loss_fn=None, metric_fn=None, is_train=False, metric_name="iou"):
	# 初始化损失和度量的平均值记录器
    loss_record   = MeanMetric()
    metric_record = MeanMetric()
    # 获取数据加载器的长度，即批次的数量
    loader_len = len(loader)

    text = "Train" if is_train else "Valid"

    for data in tqdm(iterable=loader, total=loader_len, dynamic_ncols=True, desc=f"{text} :: Epoch: {epoch_num}"):
        
        if is_train:
            preds = model(data[0])["out"]
        else:
            with torch.no_grad():
                preds = model(data[0])["out"].detach()
		# 使用损失函数计算预测结果和真实标签之间的损失
        loss = loss_fn(preds, data[1])

        if is_train:
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

        metric = metric_fn(preds.detach(), data[1])
		# 从损失和度量张量中提取数值，并更新记录器
        loss_value = loss.detach().item()
        metric_value = metric.detach().item()
        
        loss_record.update(loss_value)
        metric_record.update(metric_value)
	# 计算当前步骤的平均损失和度量值
    current_loss   = loss_record.compute()
    current_metric = metric_record.compute()

    # print(f"\rEpoch {epoch:>03} :: TRAIN :: LOSS: {loss_record.compute()}, {metric_name.upper()}: {metric_record.compute()}\t\t\t\t", end="")

    return current_loss, current_metric
```
上述代码中，`data[0]`和`data[1]`分别是`image`和`mask`，即预处理后的图像和其对应的真实标签。
#### 2.3.6 开始训练
整个训练过程的超参数和结果如下：
| 技术规格              | 细节                         |
|---------------------------|-----------------------------------|
| 骨干网络           | MobileNetV3-Large                  |
| 图像形状              | (384 x 384 x 3)                    |
| 掩膜形状             | (384 x 384 x 2)                    |
| 输出通道数（类别数） | 2 (0 – background, 1 – document)    |
| 训练轮次  | 50                               |
| 批次大小              | 64                              |
| 优化器               | Adam                              |
| 学习率          | 0.0001 (Constant)                  |
| 损失函数       | Binary Cross entropy + Intersection over union (BCE + IoU) |
| 度量指标        | Intersection over Union (IoU)       |
| Best Loss Values         | Training – 0.027 <br> Validation – 0.076 |
| Best Metric Scores       | Training – 0.989 <br> Validation – 0.976 |

```python
!nvidia-smi
```

```python
Wed Aug 17 16:16:12 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

```python
NUM_EPOCHS = 2 						# 50
BATCH_SIZE = 64
NUM_CLASSES = 2

device = get_default_device()

backbone_model_name = "mbv3" 		# mbv3 | r50 | r101

model = prepare_model(backbone_model=backbone_model_name, num_classes=NUM_CLASSES)
model.to(device)

# Dummy pass through the model
_ = model(torch.randn((2, 3, 384, 384), device=device))


train_loader, valid_loader = get_dataset(data_directory=DATA_DIR, batch_size=BATCH_SIZE)
train_loader = DeviceDataLoader(train_loader, device)
valid_loader = DeviceDataLoader(valid_loader, device)

metric_name = "iou"
use_dice = True if metric_name == "dice" else False 

metric_fn = Metric(num_classes=NUM_CLASSES, use_dice=use_dice).to(device)
loss_fn   = Loss(use_dice=use_dice).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

```python
liveloss = PlotLosses()  
best_metric = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    logs = {}

    model.train()
    train_loss, train_metric = step(model, epoch_num=epoch, loader=train_loader, optimizer_fn=optimizer, 
                                    loss_fn=loss_fn, metric_fn=metric_fn, is_train=True,metric_name=metric_name)

    model.eval()
    valid_loss, valid_metric = step(model, epoch_num=epoch, loader=valid_loader, loss_fn=loss_fn, 
                                    metric_fn=metric_fn, is_train=False,metric_name=metric_name)

    logs['loss']               = train_loss
    logs[metric_name]          = train_metric
    logs['val_loss']           = valid_loss
    logs[f'val_{metric_name}'] = valid_metric

    liveloss.update(logs)
    liveloss.send()

    if valid_metric >= best_metric:
        print("\nSaving model.....")
        torch.save(model.state_dict(), BESTMODEL_PATH)
        best_metric = valid_metric
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/935a742df68b4004a9f884c10c10a88d.png#pic_center =800x)

```python
iou
	training         	 (min:    0.886, max:    0.946, cur:    0.946)
	validation       	 (min:    0.935, max:    0.952, cur:    0.952)
Loss
	training         	 (min:    0.192, max:    0.429, cur:    0.192)
	validation       	 (min:    0.164, max:    0.262, cur:    0.164)

Saving model.....
```
&#8195;&#8195;我们还使用了`Resnet-50`作为骨干网络进行训练，训练了`25`个`epoch`，比`MobileNetv 3-large`提高了`0.07`左右。尽管如此，我们还是更喜欢轻量级的`MobileNetv 3-large`，因为它的速度很快，允许我们快速进行更多的实验。
### 2.4 测试
&#8195;&#8195;创建了一个包含51张图像的测试集，其中包括第一章中准备的23张图像（包括其badcase）和28张新的图像。下面是对比结果：




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13e20766e72e48d585d3203252142412.gif#pic_center)
&#8195;&#8195;可见，尤其对于文档边界超出图像范围的情况，上一章使用的传统视觉处理方法完全失效，而深度学习方法工作良好。此外，模型在`DocUNet`数据集的裁剪版本上进行了测试，结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0614082394574e7fa1b073accfafa660.gif#pic_center)
&#8195;&#8195;对`DocUNet`数据集的测试表明，合成数据集生成过程和训练过程仍有进一步改进的空间，但这需要另外的工作量。
### 2.5 Streamlit Web App（略）
本项目也实现了Streamlit Web App，具体见项目代码，不做讲解。

<iframe src="https://live.csdn.net/v/embed/433308" width="840" height="480" frameborder="0" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
