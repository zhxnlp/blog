- # 特征点检测和匹配

  ### 1. 特征检测的基本概念

  **特征检测**是[计算机视觉](https://baike.baidu.com/item/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)和[图像处理](https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86)中的一个概念。它指的是使用计算机提取图像信息，决定每个图像的点是否属于一个图像特征。特征检测的结果是把图像上的点分为不同的子集，这些子集往往属于孤立的点、连续的曲线或者连续的区域。

  特征检测包括边缘检测, 角检测, 区域检测和脊检测.

  特征检测应用场景:

  - 图像搜索, 比如以图搜图
  - 拼图游戏
  - 图像拼接

    ...

  以拼图游戏为例来说明特征检测的应用流程.

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/df521ed76ef24a1ab7d5488bd88a7ff2.png)

  - 寻找特征
  - 特征是唯一的
  - 特征是可追踪的
  - 特征是能比较的

    ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/736ed98bb0a0422c87301d5e457315a0.png)

    我们发现:

    - 平坦部分很难找到它在原图中的位置
    - 边缘相比平坦要好找一些, 但是也不能一下确定
    - 角点可以一下就找到其在原图中的位置

  图像特征就是值有意义的图像区域, 具有独特性, 易于识别性, 比较角点, 斑点以及高密度区.

  在图像特征中最重要的就是角点. 哪些是角点呢?

  - 灰度梯度的最大值对应的像素
  - 两条线的交点
  - 极值点(一阶导数最大, 二阶导数为0)

  ### 2. Harris角点检测

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/931bec8db4f64f09a8d2d77ad9f7fa9c.png)**Harris角点检测原理**

  ![harris_3.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/b9704da3be664769837d8e286b01b890.png)![harris_4.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/09cea5476a3646b79a452e3cec253ee6.png)![harris_5.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/98d6eb1def284cdc8c7b2d68e3a60859.png)![harris_6.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/db49e05744bd43a4b8950dea8cbc909c.png)![harris_7.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/c213c102300f49e8b46401e93ed23514.png)

  检测窗口在图像上移动, 上图对应着三种情况:

  - 在平坦区域, 无论向哪个方向移动, 衡量系统变换不大.
  - 边缘区域, 垂直边缘移动时, 衡量系统变换剧烈.
  - 在角点处,  往哪个方向移动, 衡量系统都变化剧烈.
  - cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
    - blockSize: 检测窗口大小
    - ksize: sobel的卷积核
    - k: 权重系数, 即上面公式中的$\alpha$ , 是个经验值, 一般取0.04~0.06之间.一般默认0.04

  ```python
  import cv2
  import numpy as np

  img = cv2.imread('./chess.png')

  # 变成灰度图片
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # harris角点检测
  dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

  # 返回的东西叫做角点响应. 每一个像素点都能计算出一个角点响应来. 
  # print(dst)
  print(dst.shape)
  # 显示角点
  # 我们认为角点响应大于0.01倍的dst.max()就可以认为是角点了.
  img[dst > 0.01 * dst.max()] = [0, 0, 255]
  cv2.imshow('img', img)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/2d1d125f3aee4e37ba8f75bee5d541b8.png)

  ### 3. Shi-Tomasi角点检测

  - Shi-Tomasi是Harris角点检测的改进.
  - Harris角点检测计算的稳定性和K有关, 而K是一个经验值, 不太好设定最佳的K值.
  - Shi-Tomasi 发现，角点的稳定性其实和矩阵 M 的较小特征值有关，于是直接用较小的那个特征值作为分数。这样就不用调整k值了。

    - Shi-Tomasi 将分数公式改为如下形式：$R= min(\lambda_1\lambda_2)$
    - 和 Harris 一样，如果该分数大于设定的阈值，我们就认为它是一个角点。
  - goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])

    - maxCorners: 角点的最大数, 值为0表示无限制
    - qualityLevel: 角点质量, 小于1.0的整数, 一般在0.01-0.1之间.
    - minDistance: 角之间最小欧式距离, 忽略小于此距离的点.
    - mask: 感兴趣的区域.
    - blockSize: 检测窗口大小
    - useHarrisDetector: 是否使用Harris算法.
    - k: 默认是0.04

    ```python
    import cv2
    import numpy as np

    #harris
    # blockSize = 2
    # ksize = 3
    # k = 0.04

    #Shi-Tomasi
    maxCorners = 1000
    ql = 0.01
    minDistance = 10

    img = cv2.imread('chess.png')

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners, ql, minDistance)
    corners = np.int0(corners)

    #Shi-Tomasi绘制角点
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img, (x,y), 3, (255,0,0), -1)

    cv2.imshow('Shi-Tomasi', img)
    cv2.waitKey(0)
    ```

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/5bf79c5738d14372b1522b1110f28b36.png)

  ### 4. SIFT关键点检测

  SIFT，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于[图像处理](https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/294902)领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。

  Harris角点具有旋转不变的特性.但是缩放后, 原来的角点有可能就不是角点了.

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/13b3fc7821bc4235bf8b57fd9e526334.png)

  **SIFT原理**

  - 图像尺度空间

    在一定的范围内，无论物体是大还是小，人眼都可以分辨出来，然而计算机要有相同的能力却很难，所以要让机器能够对物体在不同尺度下有一个统一的认知，就需要考虑图像在不同的尺度下都存在的特点。

    尺度空间的获取通常使用高斯模糊来实现

    不同σ的高斯函数决定了对图像的平滑程度，越大的σ值对应的图像越模糊。

  ![sift_3.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/1705b61fdf8b4f26b66491f918797604.png)![sift_2.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/0a1c4163e6504fce816d683cb26f1d46.png)

  - 多分辨率金字塔

  ![sift_4.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/6efa3475d48c41538f87282f1f75781d.png)

  - 高斯差分金字塔（DOG）

  ![sift_5.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/230b8c47b1b14b67ab78ca4cb1614a61.png)

  - DoG空间极值检测

    为了寻找尺度空间的极值点，每个像素点要和其图像域（同一尺度空间）和尺度域（相邻的尺度空间）的所有相邻点进行比较，当其大于（或者小于）所有相邻点时，该点就是极值点。如下图所示，中间的检测点要和其所在图像的3×3邻域8个像素点，以及其相邻的上下两层的3×3领域18个像素点，共26个像素点进行比较。

  ![sift_6.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/c83b034319904e90aa012aec9d1fc089.png)![sift_7.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/9d0d3b6e07384542ba5e33cd7dca8ce2.png)

  - 关键点的精确定位

    这些候选关键点是DOG空间的局部极值点，而且这些极值点均为离散的点，精确定位极值点的一种方法是，对尺度空间DoG函数进行曲线拟合，计算其极值点，从而实现关键点的精确定位。

  ![sift_8.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/98a92b30dc8640739e6c75cee14aade0.png)![sift_9.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/e18d3a2d7821401aa39daba67a813cbb.png)

  - 消除边界响应

  ![sift_10.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/6ff2e9bc48d448c0b1ec73c1e2369982.png)

  - 特征点的主方向

  每个特征点可以得到三个信息(x,y,σ,θ)，即位置、尺度和方向。具有多个方向的关键点可以被复制成多份，然后将方向值分别赋给复制后的特征点，一个特征点就产生了多个坐标、尺度相等，但是方向不同的特征点。

  ![sift_11.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/3167d452c0334759b2810b36c2564c67.png)

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/4ce759a09f8c4bc599dde5fb4c91e876.png)

  - 生成特征描述

    为了保证特征矢量的旋转不变性，要以特征点为中心，在附近邻域内将坐标轴旋转θ角度，即将坐标轴旋转为特征点的主方向。

    旋转之后的主方向为中心取8x8的窗口，求每个像素的梯度幅值和方向，箭头方向代表梯度方向，长度代表梯度幅值，然后利用高斯窗口对其进行加权运算，最后在每个4x4的小块上绘制8个方向的梯度直方图，计算每个梯度方向的累加值，即可形成一个种子点，即每个特征的由4个种子点组成，每个种子点有8个方向的向量信息。

    论文中建议对每个关键点使用4x4共16个种子点来描述，这样一个关键点就会产生128维的SIFT特征向量。

  ![sift_14.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/3b8166d126b54d2bb5c4253f7f94d215.png)![sift_15.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/0742592e1236441887751a59e1a4d572.png)![sift_17.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/1bc5994d0b5f493fb697ffc3867d1622.png)

  **使用SIFT的步骤**

  - 创建SIFT对象 sift = cv2.xfeatures2d.SIFT_create()
  - 进行检测, kp = sift. detect(img, ...)
  - 绘制关键点, drawKeypoints(gray, kp, img)

  ```python
   import cv2
  import numpy as np

  img = cv2.imread('chess.png')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 创建sift对象
  # 注意: xfeatures2d是opencv的扩展包中的内容, 需要安装opencv-contrib-python
  sift = cv2.xfeatures2d.SIFT_create()

  # 进行检测
  kp = sift.detect(gray)
  # print(kp)

  # 绘制关键点
  cv2.drawKeypoints(gray, kp, img)

  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/b7ff48cd8aa942e4bc09b77081d8604e.png)

  **关键点和描述子**

  关键点: 位置, 大小和方向.

  关键点描述子: 记录了关键点周围对其有共享的像素点的一组向量值, 其不受仿射变换, 光照变换等影响.描述子的作用就是进行特征匹配, 在后面进行特征匹配的时候会用上.

  ```python
   import cv2
  import numpy as np

  img = cv2.imread('chess.png')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 创建sift对象
  sift = cv2.xfeatures2d.SIFT_create()

  # 进行检测
  kp = sift.detect(gray)

  # 检测关键点, 并计算描述子
  kp, des = sift.compute(img, kp)
  # 或者一步到位, 把关键点和描述子一起检测出来.
  kp, des = sift.detectAndCompute(img, None)
  # print(kp)
  print(des)
  print(des.shape)


  # 绘制关键点
  cv2.drawKeypoints(gray, kp, img)

  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ### 5. SURF特征检测

  Speeded Up Robust Features（SURF，加速稳健特征），是一种稳健的局部特征点检测和描述算法。最初由Herbert Bay发表在2006年的欧洲计算机视觉国际会议（Europen Conference on Computer Vision，ECCV）上，并在2008年正式发表在Computer Vision and Image Understanding期刊上。

  Surf是对David Lowe在1999年提出的Sift算法的改进，提升了算法的执行效率，为算法在实时计算机视觉系统中应用提供了可能。

  SIFT最大的问题就是速度慢, 因此才有了SURF.

  如果想对一系列的图片进行快速的特征检测, 使用SIFT会非常慢.

  注意: SURF在较新版本的OpenCV中已经申请专利, 需要降OpenCV版本才能使用. 降到3.4.1.15就可以用了.

  ```python

  import cv2
  import numpy as np

  img = cv2.imread('chess.png')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 创建SURF对象
  surf = cv2.xfeatures2d.SURF_create()

  # 进行检测
  kp = surf.detect(gray)

  # 检测关键点, 并计算描述子
  kp, des = surf.compute(img, kp)
  # 或者一步到位, 把关键点和描述子一起检测出来.
  kp, des = surf.detectAndCompute(img, None)
  # print(kp)
  print(des)
  print(des.shape)


  # 绘制关键点
  cv2.drawKeypoints(gray, kp, img)

  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/9ce16b90629b4d9fbf6287ed7b4dbbdf.png)

  ### 6. OBR特征检测

  ORB（Oriented FAST and Rotated BRIEF）是一种快速特征点提取和描述的算法。这个算法是由Ethan Rublee, Vincent Rabaud, Kurt Konolige以及Gary R.Bradski在2011年一篇名为“ORB：An Efficient Alternative to SIFTor SURF”( http://www.willowgarage.com/sites/default/files/orb_final.pdf )的文章中提出。ORB算法分为两部分，分别是特征点提取和特征点描述。特征提取是由FAST（Features from  Accelerated Segment Test）算法发展来的，特征点描述是根据BRIEF（Binary Robust IndependentElementary Features）特征描述算法改进的。ORB特征是将FAST特征点的检测方法与BRIEF特征描述子结合起来，并在它们原来的基础上做了改进与优化。ORB算法最大的特点就是计算速度快。这首先得益于使用FAST检测特征点，FAST的检测速度正如它的名字一样是出了名的快。再次是使用BRIEF算法计算描述子，该描述子特有的2进制串的表现形式不仅节约了存储空间，而且大大缩短了匹配的时间。
  **ORB最大的优势就是可以做到实时检测**

  ORB的劣势是检测准确性略有下降.

  ORB还有一个优势是ORB是开源的算法, 没有版权问题, 可以自由使用.SIFT和SURF都被申请了专利.

  ```python
  import cv2
  import numpy as np

  img = cv2.imread('chess.png')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 创建ORB对象
  orb = cv2.ORB_create()

  # 进行检测
  kp = orb.detect(gray)

  # 检测关键点, 并计算描述子
  kp, des = orb.compute(img, kp)
  # 或者一步到位, 把关键点和描述子一起检测出来.
  kp, des = orb.detectAndCompute(img, None)
  # print(kp)
  print(des)
  print(des.shape)


  # 绘制关键点
  cv2.drawKeypoints(gray, kp, img)

  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/0a54e90880094147987a53a1b17c4920.png)

  **三种算法对比**

  - SIFT 最慢, 准确率最高
  - SURF 速度比SIFT快些, 准确率差些
  - ORB速度最快, 可以实时检测, 准确率最差.

  ### 7. 暴力特征匹配

  我们获取到图像特征点和描述子之后, 可以将两幅图像进行特征匹配.

  BF(Brute-Force) 暴力特征匹配方法, 通过枚举的方式进行特征匹配.

  暴力匹配器很简单。它使用第一组(即第一幅图像)中一个特征的描述子，并使用一些距离计算将其与第二组中的所有其他特征匹配。并返回最接近的一个。

  - BFMatcher(normType, crossCheck)
    - normType计算距离的方式.
      - NORM_L1, L1距离, 即绝对值, SIFT和SURF使用.
      - NORM_L2, L2距离, 默认值. 即平方. SIFT和SURF使用
      - HAMMING 汉明距离. ORB使用
    - crossCheck: 是否进行交叉匹配, 默认False.
  - 使用match函数进行特征点匹配, 返回的对象为DMatch对象. 该对象具有以下属性:
    * DMatch.distance - 描述符之间的距离。 越低，它就越好。
    * DMatch.trainIdx – 训练描述符中描述符的索引
    * DMatch.queryIdx - 查询描述符中描述符的索引
    * DMatch.imgIdx – 训练图像的索引
  - drawMatches 绘制匹配的特征点

  ```python
  import cv2
  import numpy as np

  img1 = cv2.imread('opencv_search.png')
  img2 = cv2.imread('opencv_orig.png')
  gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 创建sift对象
  sift = cv2.xfeatures2d.SIFT_create()

  # 进行检测
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  # 暴力特征匹配
  bf = cv2.BFMatcher(cv2.NORM_L1)
  match = bf.match(des1, des2)

  # 绘制匹配特征
  result = cv2.drawMatches(img1, kp1, img2, kp2, match, None)


  cv2.imshow('result', result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/8624b9987bb64a2a8f76d89e4648efdc.png)

  ### 8. FLANN特征匹配

  FLANN是快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）的简称。它是一个对大数据集和高维特征进行最近邻搜索的算法的集合，而且这些算法都已经被优化过了。在面对大数据集是它的效果要好于BFMatcher。

  特征匹配记录下目标图像与待匹配图像的特征点（KeyPoint），并根据特征点集合(即特征描述子)构造特征量（descriptor），对这个特征量进行比较、筛选，最终得到一个匹配点的映射集合。我们也可以根据这个集合的大小来衡量两幅图片的匹配程度。

  - FlannBasedMatcher(index_params)
    - index_params字典: 匹配算法KDTREE, LSH, SIFT和SURF使用KDTREE算法,  OBR使用LSH算法.
      - 设置示例: index_params=dict(algorithm=cv2.FLANN_INDEX_KDTREE, tree=5)
      - FLANN_INDEX_LSH = 6index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, # 12 key_size = 12, # 20 multi_probe_level = 1#2)
    - search_params字典: 指定KDTREE算法中遍历树的次数.经验值, 如KDTREE设为5, 那么搜索次数设为50.
      - search_params = dict(checks=50)
  - Flann中除了普通的match方法, 还有knnMatch方法.
    - 多了个参数--k, 表示取欧式距离最近的前k个关键点.

  ```python
  import cv2
  import numpy as np

  #打开两个文件
  img1 = cv2.imread('opencv_search.png')
  img2 = cv2.imread('opencv_orig.png')

  #灰度化
  g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  #他建SIFT特征检测器
  sift = cv2.xfeatures2d.SIFT_create()

  #计算描述子与特征点
  kp1, des1 = sift.detectAndCompute(g1, None)
  kp2, des2 = sift.detectAndCompute(g2, None)

  #创建匹配器
  index_params = dict(algorithm = 1, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  #对描述子进行匹配计算
  # 返回的是第一张图和第二张图的匹配点.
  matchs = flann.knnMatch(des1, des2, k=2)
  print(matchs)

  good = []
  for i, (m, n) in enumerate(matchs):
      # 设定阈值, 距离小于对方的距离的0.7倍我们认为是好的匹配点.
      if m.distance < 0.7 * n.distance:
          good.append(m)
    
  ret = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good], None)
  cv2.imshow('result', ret)
  cv2.waitKey(0)
  cv2.destroyAllWindows(
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/dd6e09221b32415d9b47f5276e576c69.png)

  ### 9. 图像查找

  通过特征匹配和单应性矩阵我们可以实现图像查找. 

  基本的原理是通过特征匹配得到匹配结果, 作为输入, 得到单应性矩阵, 再经过透视变换就能够找到最终的图像.

  #### 9.1 单应性矩阵

  **单应性（Homography）变换** ：可以简单的理解为它用来描述物体在世界坐标系和像素坐标系之间的位置映射关系。对应的变换矩阵称为单应性矩阵。

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/1a646916716e4314acb99623b7d87ee1.png)

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/76369d6473a5489b918c518406df491d.png)

  - 单应性矩阵的应用
  - 把图片摆正

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/686863c9fe4e4df387fd0bbb791c3ab7.png)

  - 图片替换

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/beed854d3fe54b09bd6d0500c5cd164e.png)


  - findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]])
    - srcPoints: 源平面中点的坐标矩阵，可以是CV_32FC2类型，也可以是vector `<Point2f>`类型
    - dstPoints: 目标平面中点的坐标矩阵，可以是CV_32FC2类型，也可以是vector `<Point2f>`类型
    - method: 计算单应矩阵所使用的方法。不同的方法对应不同的参数，具体如下：

      - **0** - 利用所有点的常规方法
      - **RANSAC** - RANSAC-基于RANSAC的鲁棒算法
      - **LMEDS** - 最小中值鲁棒算法
      - PROSAC-基于PROSAC的鲁棒算法
    - ransacReprojThreshold: 将点对视为内点的最大允许重投影错误阈值（仅用于RANSAC和RHO方法）。若srcPoints和dstPoints是以像素为单位的，则该参数通常设置在1到10的范围内。
    - mask: 可选输出掩码矩阵，通常由鲁棒算法（RANSAC或LMEDS）设置。 请注意，输入掩码矩阵是不需要设置的。
    - maxIters: RANSAC算法的最大迭代次数，默认值为2000。
    - confidence: 可信度值，取值范围为0到1.

  ```python
  import cv2
  import numpy as np

  #打开两个文件
  img1 = cv2.imread('opencv_search.png')
  img2 = cv2.imread('opencv_orig.png')

  #灰度化
  g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  #他建SIFT特征检测器
  sift = cv2.xfeatures2d.SIFT_create()

  #计算描述子与特征点
  kp1, des1 = sift.detectAndCompute(g1, None)
  kp2, des2 = sift.detectAndCompute(g2, None)

  #创建匹配器
  index_params = dict(algorithm = 1, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  #对描述子进行匹配计算
  matchs = flann.knnMatch(des1, des2, k=2)

  good = []
  for i, (m, n) in enumerate(matchs):
      if m.distance < 0.7 * n.distance:
          good.append(m)


  if len(good) >= 4:
      srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
      dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
      # 查找单应性矩阵
      H, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

      h, w = img1.shape[:2]
      pts = np.float32([[0,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
      dst = cv2.perspectiveTransform(pts, H)

      cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255))
  else:
      print('the number of good is less than 4.')
      exit()


    
  ret = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good], None)
  cv2.imshow('result', ret)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1638186851000/1942ee53a7a74e48bbb867364cd7ce5f.png)