# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# 
import argparse

# 创建命令解析器对象，添加命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-i','--image',required=True,help='path to image')
parser.add_argument('-t','--template',required=True,help='path to template ocr image')

# 解析参数（结果是元组类型），并使用vars将结果转为字典类型
args = vars(parser.parse_args())
print(args)

# 封装显示图片的函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




## 模板处理

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
    #cv_show('template_roi',template_roi)
    
# #print(template_digits)
# plt.figure(figsize=[12,3]);
# plt.subplot(221); plt.imshow(template[:,:,::-1]);plt.axis('off');plt.title("template");
# plt.subplot(222); plt.imshow(gray,cmap='gray');plt.axis('off');plt.title("gray");
# plt.subplot(223); plt.imshow(ref,cmap='gray');plt.axis('off');plt.title("ref");
# plt.subplot(224); plt.imshow(template_copy[:,:,::-1]);plt.axis('off');plt.title("template_copy");

# +
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

# plt.figure(figsize=[12,6]);
# plt.subplot(231); plt.imshow(card[:,:,::-1]);plt.axis('off');plt.title("img");
# plt.subplot(232); plt.imshow(tophat,cmap='gray');plt.axis('off');plt.title("topcat");
# plt.subplot(233); plt.imshow(grad_x,cmap='gray');plt.axis('off');plt.title("grad_x");
# plt.subplot(234); plt.imshow(close,cmap='gray');plt.axis('off');plt.title("close");
# plt.subplot(235); plt.imshow(close2 ,cmap='gray');plt.axis('off');plt.title("close2");
# plt.subplot(236); plt.imshow(card_copy[:,:,::-1]);plt.axis('off');plt.title("card_copy");

# +
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
# -

# 1. 遍历每个数字块, 把原图中的每个数字抠出来.
output=[]
for (i, (gx, gy, gw, gh)) in enumerate(digit_group_boxes):
    # 抠出每个数字块, 并且加点余量
    digit_group = gray_card[gy - 5: gy + gh + 5, gx - 5: gx + gw + 5]
    # 全局二值化处理，使数字更明显
    digit_group = cv2.threshold(digit_group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # 2. 通过轮廓查找，分割出单个数字
    digit_contours, _ = cv2.findContours(digit_group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的最大外接矩形，其与轮廓一一对应，但是是乱序的
    digit_boxes = [cv2.boundingRect(c) for c in digit_contours]
	# 使用 zip 将 bounding_boxes 和 ref_contours 组合在一起
    combined2 = list(zip(digit_boxes, digit_contours))
	# 根据 bounding_boxes 的 x 坐标排序
    sorted_combined2 = sorted(combined2, key=lambda item: item[0][0])
	# 解压排序后的结果，得到排序后的单个数字
    sorted_digit_boxes, sorted_digit_contours = zip(*sorted_combined2)

    # 定义每一组匹配到的数字的存放列表
    group_output = []

    # 遍历排好序的数字轮廓
    for c in sorted_digit_contours:
        # 找到当前数字的轮廓, resize成合适的大小, 然后再进行模板匹配
        (x, y, w, h) = cv2.boundingRect(c)
        # 取出每个数字区域
        card_roi = digit_group[y: y + h, x: x + w]
        card_roi = cv2.resize(card_roi, (57, 88))
        
        # 3. 匹配模板中的每个数字
        match_scores = []
        for (idx, template_roi) in template_digits.items():
            result = cv2.matchTemplate(card_roi,template_roi, cv2.TM_CCOEFF)
            #print(result)
            # 只要最大值, 即分数
            # minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(result)
            match_scores.append(result)
            # print(match_scores)
        # 找到分数最高的数字, 即我们匹配到的数字l
        group_output.append(str(np.argmax(match_scores)))
        #print(card_digits)
        
    # 画出轮廓和显示数字
    
    cv2.rectangle(card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(card, ''.join(group_output), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    output.extend(group_output)
cv_show('card', card)

print(''.join(output))


