@[toc]
- [《OpenCV优秀文章集合》](https://blog.csdn.net/qq_56591814/article/details/143195439?spm=1001.2014.3001.5501)
- [《OpenCV系列课程一：图像处理入门（读写、拆分合并、变换、注释）、视频处理》](https://blog.csdn.net/qq_56591814/article/details/127275045)
- [《OpenCV系列教程二：基本图像增强（数值运算）、滤波器（去噪、边缘检测）》](https://blog.csdn.net/qq_56591814/article/details/142146096?spm=1001.2014.3001.5502)
- [《OpenCV系列教程三：直方图、图像轮廓、形态学操作、车辆统计项目》](https://blog.csdn.net/qq_56591814/article/details/142421338?spm=1001.2014.3001.5502)
- [《OpenCV系列教程四：图像金字塔、特征检测与特征匹配，图像查找、对齐和拼接》](https://blog.csdn.net/qq_56591814/article/details/142467197?spm=1001.2014.3001.5501)
- [《OpenCV系列教程五：图像的分割与修复》](https://blog.csdn.net/qq_56591814/article/details/142906327?spm=1001.2014.3001.5501)
- [《OpenCV系列教程六：信用卡数字识别、人脸检测、车牌/答题卡识别、图片OCR》](https://blog.csdn.net/qq_56591814/article/details/143223687?spm=1001.2014.3001.5501)
- [《OpenCV系列教程七：虚拟计算器项目、目标追踪、SSD目标检测》](https://blog.csdn.net/qq_56591814/article/details/143161533?spm=1001.2014.3001.5501)
>- [opencv blog](https://www.opencv.ai/blog)、[learnopencv主页](https://learnopencv.com/)、[learnopencv 源码](https://github.com/spmallick/learnopencv/tree/master)、[KopiKat](https://www.kopikat.co/)、[CVAT标注工具](https://www.cvat.ai/)
>- 计算机视觉职位列表的首选平台 [job portal](https://learnopencv.com/detr-overview-and-inference/)

## 一、 CV领域
- [CVPR 2024：概述和关键论文——Part 1](https://learnopencv.com/cvpr2024/)、[CVPR 2024：概述和关键论文——Part 2](https://learnopencv.com/cvpr-2024-research-papers/)：CVPR - IEEE 计算机视觉和模式识别会议是计算机视觉领域规模最大、最负盛名的会议之一，汇集了计算机视觉领域最新、最前沿的研究成果。我们试图了解 CVPR 论文列表，并发现了一些有趣的论文，我们认为像您这样的用户可能会感兴趣。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/121b0e7828694dcaa1f18bdb87de65e8.gif#pic_center)
### 1.1 图像处理
- [《神经网络特征匹配入门》](https://learnopencv.com/feature-matching/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/VideoStabilization)：当你使用相机中的全景模式来拍摄一张广角照片时，你可能会好奇这个全景模式背后是如何工作的。再比如，你有一段骑自行车时拍摄的抖动视频，你打开编辑应用程序并选择视频稳定化选项，它能给你提供一个完全稳定的视频版本。这很酷，对吧？但是它是如何做到的呢？让我告诉你一个秘密：所有这些功能都是利用一种称为特征匹配的传统计算机视觉方法来实现的。
- [使用opencv特征匹配构建 Chrome Dino 游戏机器人](https://learnopencv.com/how-to-build-chrome-dino-game-bot-using-opencv-feature-matching/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Chrome-Dino-Bot-using-OpenCV-feature-matching)
- [使用OpenCV中的点特征匹配技术实现视频稳定化](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/VideoStabilization)：视频稳定化是一种技术，用于减少因摄像机抖动或移动而导致的视频质量下降。传统的摄像机在拍摄时常常受到外部因素的影响，导致画面不稳定。稳定化技术通过算法调整视频帧，平滑视觉效果，使观众获得更流畅的观看体验。

- [《使用U2-Net进行高效背景去除》](https://learnopencv.com/u2-net-image-segmentation/)：U2-Net是一种简单而强大的基于深度学习的语义分割模型，革命性地改善了图像分割中的背景去除。它在前景与背景隔离方面的有效方法在**广告、电影制作和医学影像**等应用中至关重要。本文还将讨论`U2-Net`的增强版本`IS-Net`，并展示其优越的结果，特别是在复杂图像上的表现。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71348ae07b7e4214bc0dcc1910d8e888.gif#pic_center)
- [《How Computer Vision Techniques Make People Look More Attractive》](https://www.opencv.ai/blog/how-computer-vision-makes-people-look-more-attractive)：探索计算机视觉技术用于面部增强的功能。我们深入研究了去除瑕疵、均匀肤色等的算法。此外，我们还概述了用于面部改善的流行商业解决方案，并附有各种案例研究。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78a8b64668ff4c58b131c5ce72311fc0.jpeg#pic_center)
-[《将视频转换为幻灯片并保存为PDF》](https://learnopencv.com/video-to-slides-converter-using-background-subtraction/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Build-a-Video-to-Slides-Converter-Application-using-the-Power-of-Background-Estimation-and-Frame-Differencing-in-OpenCV)：本文介绍如何利用OpenCV中的帧差分和背景减法技术，构建一个简单的视频转幻灯片应用。该应用特别适用于将带有动画的视频转化为PPT或PDF格式的幻灯片，方便在缺少原始幻灯片文件的情况下（如在YouTube上观看的讲座视频）获取内容。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e34e3bdbb064469ba866ea527386bc2.gif#pic_center)

- [《Create Snapchat/Instagram Filters Using Mediapipe》](https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/)：Snapchat 和 Instagram 提供了各种各样的滤镜功能，本文我们将了解这些增强现实滤镜的工作原理，以及如何使用 Mediapipe 框架创建自己的滤镜！
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bfb7afa40cf2498f987524f4f5952f23.gif#pic_center)


- [《A Closer Look at CVAT: Perfecting Your Annotations》](https://learnopencv.com/a-closer-look-at-cvat-perfecting-your-annotations/)、[YouTube视频](https://www.youtube.com/watch?v=yxX_0-zr-2U&list=PLfYPZalDvZDLvFhjuflhrxk_lLplXUqqB)、[如何使用CVAT标注骨架](https://www.youtube.com/watch?v=88TVX58GHIc)：[CVAT](https://www.cvat.ai/)是OpenCV发布的一个免费的图像和视频标注工具集。CVAT可以使用矩形（边界框）、多边形（遮罩）、关键点、椭圆、折线等来标注图像或视频帧。CVAT还提供了详尽的标注格式列表，以导出目标数据集标签。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf2bd487fc194a7aad9d99d163b4ba3a.webp#pic_center)
- [《CVAT SDK PyTorch adapter: 在您的机器学习流程中使用CVAT数据集》](https://www.cvat.ai/post/cvat-sdk-pytorch-adapter)：CVAT是一个视觉数据标注工具，以前在完成标注后，需要将将其转换为适合你的机器学习框架的数据结构。在CVAT SDK 2.3.0中，新引入了`cvat_sdk.pytorch`模块（PyTorch适配器），使得部分情况下可直接将CVAT项目作为PyTorch兼容的数据集使用，从而简化了数据导入的流程。
- [《使用 OpenCV 构建自定义图像注释工具》](https://learnopencv.com/automated-image-annotation-tool-using-opencv-python/)：注释是深度学习项目中最重要的部分。它是模型学习效果的决定性因素。但是，这是非常乏味和耗时的。一种解决方案是使用自动图像注释工具[pyOpenAnnotate](https://pypi.org/project/pyOpenAnnotate/) ，该工具可以大大缩短持续时间。我们还分享了一个 [streamlit Web 应用程序](https://kxborg-open-threshold-threshold-a2jyh4.streamlit.app/)，供您试用注释工具
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5799a4c35f794c8ca1415b3c54b366c2.png)


### 1.2 目标检测与识别
- [《人脸检测 - 终极指南》](https://learnopencv.com/what-is-face-detection-the-ultimate-guide/)：本文将介绍各种最先进的**人脸检测算法**，以及其历史演变。
- [《Face Recognition Models: Advancements, Toolkit, and Datasets》](https://learnopencv.com/face-recognition-models/)：本文重点介绍了对现有**人脸识别模型、工具包、数据集和 用于构建整合系统的人脸识别管道**的全面研究，本探索旨在丰富您对塑造现代人脸识别系统的潜在机制的理解。
- [《使用 OpenCV进行面部情感识别》](https://learnopencv.com/facial-emotion-recognition/)[（Github源码）](https://github.com/spmallick/learnopencv/tree/master/Facial-Emotion-Recognition)：面部情感识别 （FER） 是指根据面部表情对人类情绪进行识别和分类的过程。通过分析面部特征和模式，机器可以对一个人的情绪状态做出有根据的猜测。我们将尝试从哲学和技术的角度理解面部情绪识别的概念。我们还将**探索自定义 VGG13 模型架构和人脸表情识别数据集（FER+），以构建一个整合的实时面部表情识别系统**。最后，我们还将分析从实验中获得的结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/92939edb09374422b9d4eb5b1341dcf7.gif#pic_center)
- [《使用Mediapipe进行驾驶中的睡眠检测》](https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/)：在本文中，我们将创建一个驾驶员疲劳检测系统，使用Python中的Mediapipe人脸网格解决方案和眼宽高比公式，创建一个健壮且易于使用的应用程序。当用户的眼睛长时间闭合时，能够检测并提醒用户。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d6b9b79546e450f8d9ffc3640d2710b.gif#pic_center)

- [《Sponsor: Automated Self-Checkout with OpenVINO》](https://medium.com/openvino-toolkit/automated-self-checkout-29b2eb69afa9)：如何利用 OpenVINO 的强大功能实现自动自助结账系统，并配有一个套件，使其非常简单。该博文包括一个带有代码的 [Jupyter Notebook](https://click.convertkit-mail.com/e5u0qnr3xnt7hpvqm87t8h737o022/g3hnh5h36g395oar/aHR0cHM6Ly9naXRodWIuY29tL29wZW52aW5vdG9vbGtpdC9vcGVudmlub19ub3RlYm9va3MvYmxvYi9yZWNpcGVzL3JlY2lwZXMvYXV0b21hdGVkX3NlbGZfY2hlY2tvdXQvc2VsZi1jaGVja291dC1yZWNpcGUuaXB5bmI=) 及其工作原理的完整演练。在这个对象检测示例中，我们使用了 OpenVINO、™ Roboflow 的监督库和 Ultralytics YOLOv8。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8e8b1641b80466385f6066ab3db24c0.webp#pic_center)
- [《CenterNet: Objects as Points》](https://learnopencv.com/centernet-anchor-free-object-detection-explained)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/centernet-with-tf-hub)、[YouTube论文讲解](https://www.youtube.com/watch?v=h7WejF3QLDM)：无锚目标检测因其速度快、对其他计算机视觉任务的广泛适用性而备受关注。CenterNet是无锚目标检测算法的一个重要里程碑。本文将讨论目标检测的基本原理、无锚（anchor-free）与有锚（anchor-based）目标检测的对比、CenterNet的Object as Points论文、CenterNet姿态估计，以及CenterNet模型的推理过程。

- [《Mastering All YOLO Models from YOLOv1 to YOLOv9: Papers Explained 》](https://learnopencv.com/mastering-all-yolo-models/)：在本文中，我们将介绍所有不同版本的YOLO，从最初的YOLO到YOLOv 8和YOLO-NAS，并了解它们的内部工作原理，架构，设计选择，改进和自定义训练。
- [YOLO系列指南](https://www.opencv.ai/blog/yolo-unraveled-a-clear-guide)：我们以简短、结构化的材料（[YOLO年表](https://docs.google.com/spreadsheets/d/1Glzw3g7PasuMNyO4gzcF64OOivkg9mbai-TlMoStHvQ/edit?gid=0#gid=0)）汇编了有关 YOLO 的关键信息，您只需看一次即可理解 YOLO。
- [《Ultralytics Explorer API简介》](https://learnopencv.com/ultralytics-explorer-api/)：Ultralytics公司最近创建了一个新的数据分析工具`Ultralytics Explorer`，用于探索计算机视觉的图像数据集。 `Ultralytics Explorer`提供了Python API和GUI界面，允许您选择最合适的选项。我们将使用Explorer API可视化一个自定义野生动物数据集，而不是仅仅浏览文档。借助Ultralytics Explorer API，您可以在计算机视觉项目中获得新的见解和效率。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5457c40e4d364bc3828a7144f97620f7.gif#pic_center)

- [《使用YOLOv4训练自定义坑洞检测器》](https://learnopencv.com/pothole-detection-using-yolov4-and-darknet/)
- [《在自定义数据集上训练YOLOv5》](https://learnopencv.com/custom-object-detection-training-using-yolov5/)、[《使用 OpenCV DNN 进行 YOLOv5 对象检测（C++ and Python）》](https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/)
- [《YOLOv6 目标检测论文解析与推理》](https://learnopencv.com/yolov6-object-detection/)、[《YOLOv6 水下垃圾检测》](https://learnopencv.com/yolov6-custom-dataset-training/)
- [《YOLOR 目标检测论文解释和推理》](https://learnopencv.com/yolor-paper-explanation-inference-an-in-depth-analysis/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/YoloR-paper-explanation-analysis)
- [《YOLO X 目标检测论文解释和自定义训练》](https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/?ck_subscriber_id=1909892683)
- [《YOLOv7 目标检测论文解释和推理》](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/?ck_subscriber_id=1909892683)、[《在自定义数据集上训练YOLOv7》](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/)
- [《在无人机数据集上训练​YOLOX Object Detector》](https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/)
- [《YOLOv9简介》](https://learnopencv.com/yolov9-advancing-the-yolo-legacy/)、 [《在自定义数据集上微调YOLOv9》](https://learnopencv.com/fine-tuning-yolov9/)
- [《Train YOLOv8 on Custom Dataset – A Complete Tutorial》](https://learnopencv.com/train-yolov8-on-custom-dataset/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Train-YOLOv8-on-Custom-Dataset-A-Complete-Tutorial)。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e2b3d6ed8db421fbb572c4127da97d9.gif#pic_center)

- [使用KerasCV进行交通灯检测（YOLOv8）](https://learnopencv.com/object-detection-using-kerascv-yolov8/)（[Github源码](https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-KerasCV-YOLOv8)）

- PCB_YOLOv8_Quality_Control：微处理器板的自动检测，定位和验证；检测GPIO和焊接点等特定元素，确保全面的质量控制。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41a27436e62146b6bd9da7e60cad2a99.jpeg#pic_center#pic_center =600x)


- [《在 OAK-D-Lite 上部署 YOLOv8》](https://learnopencv.com/object-detection-on-edge-device/)：通过在 OAK-D-Lite 等流行的**边缘 AI 设备上部署模型**来展示一个有趣的嵌入式计算机视觉应用程序。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be3417d792894ae09eeccdd14133061e.gif#pic_center)
- [《YOLO-NAS：对比YOLOv6 & YOLOv8》](https://learnopencv.com/yolo-nas/)及[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/YOLO-NAS_Introduction)、[《如何在自定义数据集上训练YOLO-NAS》](https://learnopencv.com/train-yolo-nas-on-custom-dataset/)及[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Train-YOLO-NAS-on-Custom-Dataset)、[YouTube视频](https://www.youtube.com/watch?v=vfQYRJ1x4Qg)。
- [《DETR: Overview and Inference》](https://learnopencv.com/detr-overview-and-inference/)（目标检测）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e63ff2b759b846b3bb246be59e1261fd.gif#pic_center)
- [《YOLOv10: The Dual-Head OG of YOLO Series》](https://learnopencv.com/yolov10/)、[《Fine-Tuning YOLOv10 Models for Kidney Stones Detection》](https://learnopencv.com/fine-tuning-yolov10/)：微调YOLOv10模型以增强肾结石检测，将诊断时间从每份报告`15-25`分钟大幅缩短至每秒处理约`150`份报告。此研究面向医学研究人员、医疗专业人士和AI公司，通过以数据为中心的技术实现了`94.1`的mAP50，且无需更改模型架构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e24e155795174d0f9ccbe9090e7fa82f.gif#pic_center)

- [《YOLO11: Faster Than You Can Imagine!》](https://learnopencv.com/yolo11/)：YOLO11 是 Ultralytics 的 YOLO 系列的最新版本。YOLO11 配备超轻量级型号，比以前的 YOLO 型号更快、更高效。YOLO11 能够执行更广泛的计算机视觉任务（Object Detection 、Instance Segmentation 、Image Classification 、Pose Estimation 、Oriented Object Detection 、定向目标检测 OBB）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/44344444b5114dc9bc09a8c5a34ee108.gif#pic_center)
- [《微调SegFormer改进车道线检测 》](https://learnopencv.com/segformer-fine-tuning-for-lane-detection/)：Berkeley Deep Drive 100K（BDD 100K）数据集主要用于促进自动驾驶的研究和开发。该数据集非常大，包含大约100，000个视频，每个视频长度为40秒，涵盖了各种驾驶场景，天气条件。这些注释包括车道、可行驶区域、对象（如车辆、行人和交通标志）的标签，以及全帧实例分割。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff27168f808844619b9643b588f24eb5.gif#pic_center)

- [《Exploring SAHI: Slicing Aided Hyper Inference for Small Object Detection》](https://learnopencv.com/slicing-aided-hyper-inference/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Exploring-Slicing-Aided-Hyper-Inference)：本文我们将探讨现有的小目标检测方法，并使用YOLOv8通过`SAHI`技术进行推理。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2005b23c1b5b419a93ada49b6241d36e.gif#pic_center)
- [《使用Faster R-CNN在小目标检测任务中进行微调》](https://learnopencv.com/fine-tuning-faster-r-cnn/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b1bc943ec27344708cb4e5f3bc7d5b7d.gif#pic_center)

- [《用OpenCV分析大米》](https://click.convertkit-mail.com/mvum3gow8gc5hgvk60zumheownqqq/48hvhehrrlgpzvcx/aHR0cHM6Ly93d3cubGlua2VkaW4uY29tL3Bvc3RzL211aGFtbWFkcmF6YWFfaW1hZ2Vwcm9jZXNzaW5nLW9wZW5jdi1yaWNlYW5hbHlzaXMtYWN0aXZpdHktNjk4NDUxNzk2NjU3NDgwOTA4OC1pQmZLLw==)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/91c4988f5ecc42f2b9fd9ba5e88d4a65.gif#pic_center)
- [用激光分选橄榄](https://click.convertkit-mail.com/75u2ermn8rt8hkd75dvfzhq02z666/p8heh9hzz7vzgnaq/aHR0cHM6Ly93d3cubGlua2VkaW4uY29tL2ZlZWQvdXBkYXRlL3VybjpsaTphY3Rpdml0eTo3MTM3NDMxMzkwMTIzMzQzODczLw==)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c652754743f74142915e091d65ad57f2.webp#pic_center)
-  [《TrOCR：基于Transformer的OCR入门》](https://learnopencv.com/trocr-getting-started-with-transformer-based-ocr/)、[youtube视频](https://www.youtube.com/watch?v=2k7aOpiCU-I)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/TrOCR-Getting-Started-with-Transformer-Based-OCR)：本文将介绍TrOCR 的架构、TrOCR 系列模型、如何训练TrOCR 模型以及如何使用 TrOCR 和 Hugging Face 运行推理。
- [TrOCR的训练和微调](https://learnopencv.com/fine-tuning-trocr-training-trocr-to-recognize-curved-text/)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/Fine-Tuning-TrOCR)：TrOCR在某些时候是最佳性能的OCR模型，在处理非直线文本方面的表现并不理想。本文将对TrOCR进一步研究，介绍如何在曲线文本数据集上微调TrOCR模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/141fe0e92bd34123a2f2145f1a8826da.gif#pic_center)

- [《PaddleOCR: Unveiling the Power of Optical Character Recognition》](https://learnopencv.com/optical-character-recognition-using-paddleocr/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3fe3c8ebc6984f16969b6d02e47a5229.webp#pic_center)


- [使用 TrOCR 进行手写文本识别](https://learnopencv.com/handwritten-text-recognition-using-ocr/)：手写文本文档在研究和学习领域无处不在。它们根据用户的需求进行个性化设置，并且通常包含其他人难以理解的写作风格。对于处理手写学习笔记的网站来说是一个问题。在本文中，我们将通过微调 `TrOCR` 模型来使用 OCR 进行手写文本识别。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c27721cccc2045b9aae5c1d227c02a2f.gif#pic_center)
- [使用YOLOv4进行自动车牌识别](https://learnopencv.com/automatic-license-plate-recognition-using-deep-learning/)：本博文将重点介绍 ALPR（自动车牌识别）的端到端实施，过程分为 2 步——车牌检测和车牌OCR。
- - [《YOLO模型损失函数》](https://learnopencv.com/yolo-loss-function-siou-focal-loss/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8608b4d78bb64d7b8c33934fe2e1096c.gif#pic_center)
- [目标检测中WBF与NMS的比较](https://learnopencv.com/weighted-boxes-fusion/)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/Weighted-Boxes-Fusion-in-Object-Detection)：目标检测模型传统上使用非极大值抑制（NMS）作为默认的后处理步骤来过滤掉冗余的边界框。然而，这种方法无法有效地为多个模型提供统一的、平均化的预测，因为它们倾向于移除那些具有显著重叠的不太确定的框。为了缓解这个问题，我们将讨论一种称为加权框融合（WBF）的高效预处理步骤，它有助于在多个检测结果中实现统一的局部化预测。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9324d8596f634e78b7a4f14b51eac0ab.webp#pic_center)

### 1.3 图像分割、目标追踪
- [《SAM – A Foundation Model for Image Segmentation》](https://learnopencv.com/segment-anything/)[、YouTube视频](https://www.youtube.com/watch?v=mtHEBUdYRYU)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Segment-Anything-A-Foundation-Model-for-Image-Segmentation)：介绍SAM模型及其数据集，以及推理。
- [《Tech track #2. Fast SAM review》](https://www.opencv.ai/blog/fast-sam-review)：`Fast SAM` 是 Segment Anything 任务的一种创新方法，将SAM模型的处理速度提升了50倍。它引入了提示机制，以促进解决广泛的问题，打破了传统监督学习的界限。
 - [《SAM 2 – Promptable Segmentation for Images and Videos》](https://learnopencv.com/sam-2/)：图像分割是计算机视觉中最基本的任务之一。去年，Meta AI 凭借其 Segment Anything Model （SAM） 推出了世界上第一个图像分割基础模型。今天，我们有了 SAM 2 （Segmentation Anything Model 2），这是一个用于图像和视频分割的可提示基础模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e835a46c0b894a669178efe8ad234b80.gif#pic_center)
- [《训练3D U-Net进行脑肿瘤分割（BraTS2023-GLI）挑战赛》](https://learnopencv.com/3d-u-net-brats/)：3D U-Net 是一种高效的医疗分割范例，擅长分析 3D 体积数据，使其能够捕获脑部扫描的整体视图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27c308769f844665b9aa53627974c78b.gif#pic_center)
- [《探索DINO：使用ResNet50和U-Net进行道路分割的自监督变换器》](https://learnopencv.com/fine-tune-dino-self-supervised-learning-segmentation/)：`DINO`是一个自监督学习（SSL，self-supervised learning）框架，以视觉变换器（ViT，Vision Transformer）为核心架构。SSL作为一种替代的预训练策略，不依赖标记数据，最初在自然语言处理（NLP）任务中获得了广泛关注。在计算机视觉领域，DINO（无标签蒸馏）模型由Facebook AI开发，并于去年推出了改进版本DINOv2。本文深入探讨自监督学习的历史，分析DINO模型及其内部机制，并针对IDD数据集的印度道路分割任务对基于DINO的ResNet-50进行微调，最终在训练`DINO-ResNet50 U-Net`模型时达到了`0.95`的IOU。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf27c12130814242abf8a9a44678e7df.gif#pic_center)


- [《Sapiens: Foundation for Human Vision Models by Meta》](https://learnopencv.com/sapiens-human-vision-models/)：Sapiens是一个模型系列，针对四个基本的人本视觉任务——二维姿态估计、身体部位分割、深度估计和表面法线预测。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/933f95505a75415c9c3574821ddab984.gif#pic_center)


- [《DeepLabv3 & DeepLabv3+ The Ultimate PyTorch Guide》](https://learnopencv.com/deeplabv3-ultimate-guide/)：Google 研究团队在 17 年末发布了广受欢迎的DeepLabv3，在 Pascal VOC 2012 测试集上实现了SOTA性能。DeepLabv3+ 在 DeepLabv3 语义分割模型的基础上进行了根本性的架构改进，实现了更强的性能。

- [《Semantic Segmentation using KerasCV DeepLabv3+》](https://learnopencv.com/kerascv-deeplabv3-plus-semantic-segmentation/)（[github源码](https://github.com/spmallick/learnopencv/tree/master/Semantic-Segmentation-using-KerasCV-with-DeepLabv3-Plus)）：DeepLabv3+是一种流行的语义分割模型，可用于图像分割中的各种应用，例如医学成像，自动驾驶等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/abda448e8c554503b2dc9d92ced4ebe3.gif#pic_center)
- [使用YOLOv8 进行实例分割](https://learnopencv.com/train-yolov8-instance-segmentation/)、[youtube视频](https://www.youtube.com/watch?v=6J2PvzhO_Mk)
- [《YOLOv8 Object Tracking and Counting with OpenCV》](https://learnopencv.com/yolov8-object-tracking-and-counting-with-opencv/)：了解使用 YOLOv8 和 OpenCV 进行对象跟踪，以及如何在视频源中执行对象计数。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ccf36ece4676429da60a258409781fd9.gif#pic_center)
- [《YOLOv9 Instance Segmentation on Medical Dataset》](https://learnopencv.com/yolov9-instance-segmentation-on-medical-dataset/)

- [《Real Time Deep SORT with Torchvision Detectors》](https://learnopencv.com/real-time-deep-sort-with-torchvision-detectors/)：不论是人物还是车辆，目标跟踪在各个领域都非常重要。测试多种检测模型和再识别（Re-ID）模型比较繁琐，因此我们计划简化这个流程。我们将创建一个小的代码库，帮助用户轻松测试Torchvision中的任何目标检测模型，并将其与实时的Deep SORT库结合。Deep SORT库可以用来执行目标跟踪，并提供多种再识别模型（Re-ID）。我们还将还会对不同检测器和Re-ID模型组合的效果进行定性和定量分析，比如通过帧率（FPS）等指标来评估系统性能。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e7a28615ca214967870334fc1aa4ffe1.gif#pic_center)
- [《YOLOv5实例分割》](https://learnopencv.com/yolov5-instance-segmentation/)
### 1.4 姿态估计
- [《Object Keypoint Similarity in Keypoint Detection》](https://learnopencv.com/object-keypoint-similarity/)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/Object-Keypoint-Similarity-in-Keypoint-Detection)：无论是 在杂乱场景中检测特定对象，还是在实时分析人类姿势，关键点都发挥着关键作用。但是，我们如何测量这些检测到的关键点的相似性和精确度呢？这就引入了对象关键点相似性（OKS）的概念，这是一种用于衡量关键点检测准确性的特定指标。本文将介绍什么是OKS？OKS是如何计算的？OKS如何处理不同的尺度和对象大小？
- [《使用 Mediapipe 进行不良姿势检测》](https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/46fa4587debf40b3b41c893ad1ada2b6.gif#pic_center)
- [《使用 Mediapipe 创建 AI 健身教练》](https://learnopencv.com/ai-fitness-trainer-using-mediapipe/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/AI-Fitness-Trainer-Using-MediaPipe-Analyzing-Squats)：用于估计人体姿态的流行框架包括OpenPose、AlphaPose、Yolov7、MediaPipe等。然而，由于在CPU上的疯狂推理速度，我们选择了使用MediaPipe的`Pose`流程来估计人体关键点。基于此，我们构建一个AI健身教练，无论您是初学者还是专业人士，它都能帮助您顺畅地完成深蹲动作（初级模式和高级模式），并伴有适当的反馈。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c792c49d8cf42a2ba58621bd1ba07b7.gif#pic_center)
- [《YOLOv7 Pose vs MediaPipe in Human Pose Estimation》](https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/#What-is-MediaPipe-Pose?)：YOLOv7 Pose 是在 22 年 7 月首次发布几天后引入 YOLOv7 存储库的。它是一个单阶段、多人姿势估计模型。MediaPipe Pose 是一个单人姿势估计框架。本文对这两种模型进行了分析和对比。
- [《YOLOv7 Object Detection Paper Explanation & Inference》](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/)：本文将解读YOLOv7 论文及推理测试，并介绍YOLOv7 Pose。

- [《基于手势的三维操作交互》](https://click.convertkit-mail.com/38u2z0xwm0tkhomq40rfrh5d5ponn/wnh2hghwxqk3qmf7/aHR0cHM6Ly93d3cubGlua2VkaW4uY29tL2ZlZWQvdXBkYXRlL3VybjpsaTphY3Rpdml0eTo3MTYxMDA2MTA3NzIzNzE4NjU2Lw==)：Alireza Delisnav 最近在 LinkedIn 上分享了这个演示，展示了如何将用户的网络摄像头与 TensorFlow 机器学习库和 Handpose 库结合使用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b36429b006546c3a803a89fbf346541.gif#pic_center)

- [《微调YOLOv 8用于动物姿态估计》](https://learnopencv.com/animal-pose-estimation/)（[Github源码](https://github.com/spmallick/learnopencv/tree/master/Fine-tuning-YOLOv8-Pose-Models-for-Animal-Pose-Estimation)）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71851112179741a5bd5d9caf5c8624c5.gif#pic_center)

### 1.5 3D视觉
 - [《立体视觉简介及iphones 中的深度估计技术》](https://www.opencv.ai/blog/depth-estimation)
 - [《为什么相机校准在计算机视觉中如此重要》](https://www.opencv.ai/blog/camera-calibration)
- [《Tech track #4. NeRF: Photorealistic Image Synthesis》](https://www.opencv.ai/blog/nerf-short-review)：3D 视觉领域一项重要任务是 Novel View Synthesis，它旨在使用该场景的稀疏图像集从新颖的角度生成场景的图像。该领域的一个显着突破是 `NeRF` 模型（神经辐射场），它使用神经网络和体积渲染技术来生成场景的新视图。`NeRF` 的输入是一组具有相应相机位置（外在矩阵）的图像，该模型本质上是对给定3D场景的一种特殊表示，由一系列连续的点组成，每个点都有预测的密度和颜色。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a9d03e6f3a042d68e48a8b13ed989c9.jpeg#pic_center)

- [《Depth Anything: Accelerating Monocular Depth Perception》](https://learnopencv.com/depth-anything/)：单眼深度感知是 3D 计算机视觉的一个关键方面，它能够从单个二维图像中估计 3D 结构。与依赖多个视点来推断深度的立体技术不同，单眼深度感知算法必须从各种图像特征（如纹理渐变、对象大小、阴影和透视）中提取深度线索。应用领域包括**水下成像、人体动作观察、手势识别、野生动物监测、地形测绘**等等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca6bfdc0edfc4b30bef6f61bbebe6cbf.gif#pic_center)

- [《激光雷达SLAM简介：LOAM和LeGO-LOAM论文及代码解析与ROS 2实现》](https://learnopencv.com/lidar-slam-with-ros2/)：在机器人感知研究中，LiDAR SLAM是一个独特的领域，因为它必须处理各种场景，例如室内和室外环境，自我车辆的高速，动态对象，变化的天气条件以及实时处理的需求等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/79c0e64335ab4ae6b80740f28c00fd4c.gif#pic_center)


- [3D激光雷达可视化](https://learnopencv.com/3d-lidar-visualization/)（[github源码](https://github.com/spmallick/learnopencv/tree/master/3D-LiDAR-Perception)）：3D激光雷达传感器（或）三维光探测与测距是一种先进的光发射仪器，它能够像人类一样在三维空间中感知现实世界。这项技术特别是对地球观测、环境监测、侦察以及现在的自动驾驶领域产生了革命性的影响。它提供准确和详细数据的能力，对于提升我们理解和管理环境及自然资源起到了至关重要的作用。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6695a5dd087441768828233e52b7e306.gif#pic_center)
- [ADAS简介](https://learnopencv.com/advanced-driver-assistance-systems/#aioseo-what-are-the-core-components-of-adas)：随着汽车技术的快速发展，推动更安全、更智能、更高效的驾驶体验一直是汽车创新的前沿。高级驾驶辅助系统（ADAS）是这场技术革命的关键参与者，是指集成到现代车辆中的一系列技术和功能，以增强驾驶员安全性，改善驾驶体验，并协助完成各种驾驶任务。它使用传感器，摄像头，雷达和其他技术来监控车辆的周围环境，收集数据并向驾驶员提供实时反馈。在今天关于ADAS的文章中，我们将讨论不同级别的ADAS（0至5级）， ADAS系统如何工作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8190d41b9684f1f9b5eb9e0edecae90.gif#pic_center)

- [《 ADAS：立体视觉中超越LiDAR的深度感知先锋》](https://learnopencv.com/adas-stereo-vision/)：本文将探讨汽车中的ADAS立体视觉如何改变游戏规则，为深度感知提供一种智能替代方案，而不是传统的基于LiDAR的方法。该综合研究文章包括一个逐步流程，介绍如何设置和微调STereo TRansformer（STTR），以从两个摄像头流中预测视差图，类似于人眼的工作方式。除了纯粹的计算机视觉理论，本文还包含了在微调KITTI立体视觉数据集后的真实实验结果。[源码](https://click.convertkit-mail.com/68u6rowxknu8hkr6m75c9uzdz0nkk/p8heh9hzgle0exfq/aHR0cHM6Ly9naXRodWIuY29tL3NwbWFsbGljay9sZWFybm9wZW5jdi90cmVlL21hc3Rlci9BREFTLVN0ZXJlby1WaXNpb24=)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd3afe6960054372bff0b4238cb934e5.gif#pic_center)
- [《使用Python和OpenCV从零开始构建SLAM》](https://learnopencv.com/monocular-slam-in-python/)：`SLAM`是机器人和3D计算机视觉中的一个众所周知的术语。它是机器人感知的一个组成部分，负责使机器人能够在未知的地形中导航。特斯拉的自动驾驶汽车就使用了这种技术。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5d4af9a3af64ef980adc0ab65068db7.gif#pic_center)

### 1.6  图像生成
- [《Introducing the Kopikat》](https://www.kopikat.co/mscoco-experiment)：[KopiKat](https://www.kopikat.co/)是一款无代码的生成式数据增强工具，通过生成多样化的图片，保持原始标注，从而显著提升神经网络在小数据集（少于5000张图像）上的效果。它在以下方面具有重要应用：

	1. **目标检测**：提升YOLOX-Nano等模型的准确性，适用于零售流量分析、安全系统、自动驾驶等实时识别场景。
	2. **小数据集训练**：多样化小规模数据集，使得工业应用在数据有限的情况下也能得到有效模型。
	3. **迁移学习**：使用增强后的数据集进行迁移学习，有助于减少新任务或数据集的训练时间和资源。

KopiKat为有限数据量的项目提供了更高效的数据扩充方式，助力高质量产品的快速开发。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eed5b446a25f4d90a42a6d43a1159833.png)



- [《Introduction to Diffusion Models for Image Generation》](https://learnopencv.com/image-generation-using-diffusion-models/)
- [《​Introduction to Denoising Diffusion Models (DDPM)》](https://learnopencv.com/denoising-diffusion-probabilistic-models/)
- [《Top 10 AI Tools for Image Generation》](https://learnopencv.com/ai-art-generation-tools/)：简单介绍了DALL-E 2 、Midjourney、Stable Diffusion、OpenJourney、Playgroundai等图像生成工具。
- [《Mastering DALLE2》](https://learnopencv.com/mastering-dall-e-2/)
- [《Mastering MidJourney》](https://learnopencv.com/rise-of-midjourney-ai-art/)
- [《Introduction to Stable Diffusion》](https://learnopencv.com/stable-diffusion-generative-ai/)
- [《ControlNet – Achieving Superior Image Generation Results》](https://learnopencv.com/controlnet/)
- [《InstructPix2Pix – Edit Images With Prompts》](https://learnopencv.com/instructpix2pix/)
- [《OpenCV Face Recognition – Does Face Recognition Work on AI-Generated Images?》](https://learnopencv.com/opencv-face-recognition-api/)：面部识别技术能否像处理真实图像一样处理这些 AI 生成的人脸图像？我们将构建OpenCV 人脸识别系统，测试它在真实人脸上的准确性，最后评估它在 AI 生成的人脸上的性能。
- [《Hugging Face Diffusers简介》](https://learnopencv.com/hugging-face-diffusers/)
- [《Dreambooth using Diffusers》](https://learnopencv.com/dreambooth-using-diffusers/)：Dreambooth 是一种专门的训练技术，旨在微调预训练的 Diffusion 模型。本文深入探讨了使用 Diffusers 根据特定主题生成个性化图像的 Dreambooth 技术。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/59922896d5394902bc6e0bc4c3478d05.gif#pic_center)

- [使用SDXL进行图片重绘](https://learnopencv.com/sdxl-inpainting/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/735954e996124f7b908d88ca5e34eb70.gif#pic_center)

### 1.7 机器视觉
- [《ROS 2（机器人操作系统2）简介及使用教程》](https://learnopencv.com/robot-operating-system-introduction/)：ROS2（Robot Operating System 2）是一个用于机器人编程的开源框架，它提供了一组软件库和工具，帮助开发者构建机器人应用程序。
- [《ROS 2 and Carla Setup Guide for Ubuntu 22.04》](https://learnopencv.com/ros2-and-carla-setup-guide/)

- [OpenCV AI Competition Top 20 teams](https://mail.qq.com/cgi-bin/frame_html?t=newwin_frame&sid=WwpXDTqb59bOakMk&url=/cgi-bin/readmail?folderid=132&folderkey=132&&t=readmail&mailid=ZC0030_hNXNrpmMkKEuTGUAnZYWkdc&mode=pre&maxage=3600&base=11.53&ver=15864)
- [Rescue Bot](https://www.hackster.io/user102774/rescue-bot-b51484)：救援机器人是一个自主的地面漫游者，旨在导航灾后地形，定位幸存者和加强灾害响应。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe4bee93b7e94215a71e93684dd8f32c.png#pic_center =500x)

- [Guiding Gaze](https://www.hackster.io/optical-oddballs/guiding-gaze-074f80)：一个自动导航系统，旨在为那些有视觉挑战的人提供帮助
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec0ec59af7814e31a6e88467f851413f.png#pic_center =500x)
- [Opti Sentinel](https://www.hackster.io/robixlo/opti-sentinel-7afd73)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4d9a317adb74512a6a8edfe03f99a6b.png#pic_center =500x)


- [使用OpenCV进行相机校准](https://www.youtube.com/watch?v=EWqqseIjVqM)
### 1.8 其它
- [《如何为计算机视觉 AI 解决方案制定预算？Part 2 | Software》](https://www.opencv.ai/blog/what-impacts-your-computer-vision-ai-solution-budget-part-2-software)
- [《使用Hugging Face Spaces 和 Gradio部署深度学习模型》](https://learnopencv.com/deploy-deep-learning-model-huggingface-spaces/)、[GitHub源码](https://github.com/spmallick/learnopencv/tree/master/Deploying-a-Deep-Learning-Model-using-Hugging-Face-Spaces-and-Gradio)、[yolov8_nano目标检测应用](https://huggingface.co/spaces/sovitrath/pothole_yolov8_nano)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97a2c9c1a5bb4a97a7ae0ef034a5319a.gif#pic_center)
- [《Gradio与OpenCV DNN集成》](https://learnopencv.com/integrating-gradio-with-opencv-dnn/)：本文探讨深度学习模型部署方案——集成Gradio与OpenCV DNN，一种轻量级、高效且具有实时推理能力的网络应用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a869ef4154d84a198f86261448bb63b5.gif#pic_center)

- [在 PyTorch 中实现 Vision Transformer](https://learnopencv.com/the-future-of-image-recognition-is-here-pytorch-vision-transformer/)：我们将详细解释 vision transformer 架构的各个组件，最后继续在 PyTorch 中实现整个架构。
- [《PaddlePaddle: Exploring Object Detection, Segmentation, and Keypoints》](https://learnopencv.com/paddlepaddle/)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/Introduction-to-PaddlePaddle)：介绍如何使用PaddlePaddle完成目标检测、分割和关键点检测任务。
- [《使用CLIP构建图像检索系统》](https://learnopencv.com/clip-model/)：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c4fd74082b9a45329dc4b9895d404fdf.gif#pic_center)



- [《Building MobileViT Image Classification Model from Scratch In Keras 3》](https://learnopencv.com/mobilevit-keras-3/)：在快速发展的深度学习领域，挑战往往不仅在于设计强大的模型，还在于使其易于访问和高效地用于实际使用，尤其是在计算能力有限的设备上。MobileViT 模型是更大、更复杂的视觉转换器 （ViT） 的紧凑而强大的替代方案。我们的主要目标是提供使用 Keras 3 从头开始实施 MobileViT v1 模型的全面指南。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19b5658d61474fcb9a70539a68c361de.gif#pic_center)

- [《ControlNet – Achieving Superior Image Generation Results》](https://learnopencv.com/controlnet/)、[Github源码](https://github.com/spmallick/learnopencv/tree/master/ControlNet-Achieving-Superior-Image-Generation-Results)：ControlNet 是一种调节输入图像和图像生成提示的新方法。它允许我们通过各种技术（如姿势、边缘检测、深度图等）来控制最终图像的生成。














- [《使用OpenCV进行系外行星探索》](https://www.hackster.io/contests/opencv-ai-competition-2023)（[源码](https://github.com/katelyng7/ComputerVision-on-Exoplanet-Detection)）：Katelyn Gan 的项目“用于系外行星探测的径向速度提取的计算机视觉方法”在 2023 年 OpenCV AI 竞赛中获得第二名！此代码演示了应用计算机视觉技术来分析恒星光谱和提取径向速度 （RV） 以进行系外行星探测的概念和程序。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28d584e1c8cf423bbd1be7288d92a7f7.jpeg#pic_center)
## 二、 nlp
- [《Introducing OpenVINO™ 2024.4》](https://medium.com/openvino-toolkit/introducing-openvino-2024-4-28578870b264)、[《如何使用 OpenVINO™ 在本地运行 Llama 3.2》](https://medium.com/openvino-toolkit/how-to-run-llama-3-2-locally-with-openvino-60a0f3674549)、[《Retrieval Augmented Generation – RAG with LLMs》](https://learnopencv.com/rag-with-llms/)
- [《​Understanding the Attention Mechanism in Transformers》](https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/)
- [《使用ColPali 和 Gemini 多模态模型的 RAG：财务报告分析应用程序》](https://learnopencv.com/multimodal-rag-with-colpali/)：带有 ColPali 的多模态 RAG 通过将每个页面视为图像，提供了一种有效检索图像、表格、图表和文本等元素的新方法。这种方法利用视觉语言模型 （VLM） 来理解复杂文档（如财务报告、法律合同和技术文档）中的复杂细节。我们将通过使用 Colpali 和 Gemini 构建财务报告多模式 RAG 应用程序。具体来说，我们将研究如何分析 10-Q 季度报告，这是公司向美国证券交易委员会 （SEC） 提交的重要财务文件。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eaeead6f9a5a463391fadbb1654d097d.gif#pic_center)
- [《训练神经网络的秘诀》](https://karpathy.github.io/2019/04/25/recipe/)
- [《Text Summarization using T5: Fine-Tuning and Building Gradio App》](https://learnopencv.com/text-summarization-using-t5/?utm_source=email_button)、[《Fine Tuning T5 for Building a Stack Overflow Tag Generator》](https://learnopencv.com/fine-tuning-t5/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca969ec0b56c4c8e863ad35918094608.gif#pic_center)
## 三、语音
- [《在自定义数据集上微调Whisper》](https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/)
- [《自动语音识别（ASR）》](https://learnopencv.com/automatic-speech-recognition/)
## 四、推荐系统
- [《基于Qdrant的向量搜索推荐系统》](https://learnopencv.com/recommendation-system-using-vector-search/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e880ee48ae947a7a446b5ae8177d7ed.gif#pic_center)


