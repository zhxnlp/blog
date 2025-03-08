# 先处理数字模板图片
import cv2 
import numpy as np
import argparse

# 设置脚本的参数
parse = argparse.ArgumentParser()
parse.add_argument('-i', '--image', required=True, help='path to input image')
parse.add_argument('-t', '--template', required=True, help='path to template ocr image')
args = vars(parse.parse_args())
print(args)

# 封装显示图片的函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取模板图片
img = cv2.imread(args['template'])
# print(img.shape)
# cv_show('img', img)
# 灰度化处理
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref', ref)

# 二值化处理
_, ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
# cv_show('ref', ref)

# 计算轮廓
ref_, ref_contours, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画出外轮廓
cv2.drawContours(img, ref_contours, -1, (0, 0, 255), 3)
# cv_show('img', img)
# 表示数字的轮廓
print(np.array(ref_contours, dtype='object').shape)
# 对轮廓进行排序, 按照数字大小进行排序, 方便后面使用.
# 排序思路: 根据每个数字的最大外接矩形的x轴坐标进行排序
# 计算每个轮廓的外接矩形
bounding_boxes = [cv2.boundingRect(c) for c in ref_contours]
# print(bounding_boxes)
# print(sorted(bounding_boxes, key=lambda b: b[0]))
# 要把排序之后的外接矩形和轮廓建立对应关系.
(ref_contours, bounding_boxes) = zip(*sorted(zip(ref_contours, bounding_boxes), key=lambda b: b[1][0]))
digits = {}
for (i, c) in enumerate(ref_contours):
    # 重新计算外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # region of interest 感兴趣的区域
    # 取出每个数字
    roi = ref[y:y + h, x: x + w]
    # resize成合适的大小
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
# print(digits)

# 对信用卡图片进行处理
image = cv2.imread(args['image'])
# cv_show('image', image)
# 对信用卡图片进行resize
# 为了保证原图不拉伸, 需要计算出原图的长宽比.
h, w = image.shape[:2]
width = 300
r = width / w
image = cv2.resize(image, (300, int(h * r)))
# cv_show('image', image)
# 灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray', gray)

# 接下来是形态学的各种操作
# 顶帽操作, 突出更明亮的区域
# 初始化卷积核
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
# cv_show('tophat', tophat)

# sobel算子
grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# print(grad_x)
# 对grad_x进行处理
# 只用x轴方向的梯度
grad_x = np.absolute(grad_x)
# 再把grad_x变成0到255之间的整数
min_val, max_val = np.min(grad_x), np.max(grad_x)
grad_x = ((grad_x - min_val) / (max_val - min_val)) * 255
# 修改一下数据类型
grad_x = grad_x.astype('uint8')
# cv_show('grad_x', grad_x)

# grad_y = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
# # print(grad_x)
# # 对grad_x进行处理
# grad_y = np.absolute(grad_y)
# # 再把grad_x变成0到255之间的整数
# min_val, max_val = np.min(grad_y), np.max(grad_y)
# grad_y = ((grad_y - min_val) / (max_val - min_val)) * 255
# # 修改一下数据类型
# grad_y = grad_y.astype('uint8')
# cv_show('grad_y', grad_y)

# cv_show('gray', grad_x + grad_y)

# 闭操作, 先膨胀, 再腐蚀, 可以把数字连在一起.
grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
# cv_show('gradx', grad_x)

# 通过大津(OTSU)算法找到合适的阈值, 进行全局二值化操作.
_, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv_show('thresh', thresh)

# 中间还有空洞, 再来一个闭操作
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
# cv_show('thresh', thresh)

# 找轮廓
thres_, thresh_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 在原图上画轮廓
image_copy = image.copy()
cv2.drawContours(image_copy, thresh_contours, -1, (0, 0, 255), 3)
# cv_show('img', image_copy)

# 遍历轮廓, 计算外接矩形, 然后根据实际信用卡数字区域的长宽比, 找到真正的数字区域
locs = []
output = []
for c in thresh_contours:
    # 计算外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算外接矩形的长宽比例
    ar = w / float(h)
    # 选择合适的区域
#     print(ar)
    if ar > 2.5 and ar < 4.0:
        # 在根据实际的长宽做进一步的筛选
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合条件的外接矩形留下来
            locs.append((x, y, w, h))
            
# 对符合要求的轮廓进行从左到右的排序.
sorted(locs, key=lambda x: x[0])

# 遍历每一个外接矩形, 通过外接矩形可以把原图中的数字抠出来.
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    # 抠出数字区域, 并且加点余量
    group = gray[gy - 5: gy + gh + 5, gx - 5: gx + gw + 5]
#     cv_show('group', group)
    # 对取出灰色group做全局二值化处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cv_show('group', group)
    
    # 计算轮廓
    _, digit_contours, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 对轮廓进行排序
    bounding_boxes = [cv2.boundingRect(c) for c in digit_contours]
    (digit_contours, _) = zip(*sorted(zip(digit_contours, bounding_boxes), key=lambda b: b[1][0]))
    
    # 定义每一组匹配到的数字的存放列表
    group_output = []
    # 遍历排好序的轮廓
    for c in digit_contours:
        # 找到当前数字的轮廓, resize成合适的大小, 然后再进行模板匹配
        (x, y, w, h) = cv2.boundingRect(c)
        # 取出数字
        roi = group[y: y + h, x: x + w]
        roi = cv2.resize(roi, (57, 88))
#         cv_show('roi', roi)
        
        # 进行模板匹配
        # 定义保存匹配得分的列表
        scores = []
        for (digit, digit_roi) in digits.items():
            result = cv2.matchTemplate(roi,digit_roi, cv2.TM_CCOEFF)
            # 只要最大值, 即分数
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 找到分数最高的数字, 即我们匹配到的数字l
        group_output.append(str(np.argmax(scores)))
        
    # 画出轮廓和显示数字
    cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(image, ''.join(group_output), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    output.extend(group_output)
cv_show('image', image)
