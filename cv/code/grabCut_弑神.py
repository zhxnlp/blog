import numpy as np
import cv2 as cv
import  argparse

parse = argparse.ArgumentParser()
parse.add_argument('-i','--image',required=True,help='path to input image')
args = vars(parse.parse_args())

BLUE = [255,0,0]        # 矩形颜色
RED = [0,0,255]         # 可能是背景的颜色
GREEN = [0,255,0]       # 可能是前景的颜色
BLACK = [0,0,0]         # 确定背景的颜色
WHITE = [255,255,255]   # 确定前景的颜色

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_BG = {'color' : RED, 'val' : 2}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

rect = (0,0,1,1)
drawing = False         # 补充前景背景标记
rectangle = False       # 画矩形标记
rect_over = False       # 矩形画完标记
rect_or_mask = 100      # 设置rect或者mask模式标记
value = DRAW_FG         # 前景背景颜色,模式标记
thickness = 3           # 线类型

def onmouse(event,x,y,flags,userdata):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # 画矩形
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print('按n键进行查看 \n')

    # 画补充标记

    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print('请先按鼠标右键选择需要处理的区域')
        else:
            drawing = True
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

if __name__ == '__main__':

    img = cv.imread(args['image'])
    size = img.shape
    x = 800
    y = int(800*float(size[0])/size[1])
    res = cv.resize(img,(x,y),interpolation=cv.INTER_AREA)
    img = res
    img2 = img.copy()

    mask = np.zeros(img.shape[:2],dtype = np.uint8)
    output = np.zeros(img.shape,np.uint8)           # 展示窗口

    # 创建2个窗口
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input',onmouse)

    print('请按住鼠标右键画矩形，选择需要处理的区域 \n')

    while True:

        cv.imshow('output',output)
        cv.imshow('input',img)
        k = cv.waitKey(1)


        if k == 27:
            break
        elif k == ord('0'):
            print('请按下鼠标左键标记背景,标记完成后请按n进行查看. \n')
            value = DRAW_BG
        elif k == ord('1'):
            print('请按下鼠标左键标记前景,标记完成后请按n进行查看. \n')
            value = DRAW_FG
        elif k == ord('2'):
            print('请按下鼠标左键标记可能的背景,标记完成后请按n进行查看. \n')
            value = DRAW_PR_BG
        elif k == ord('3'):
            print('请按下鼠标左键标记可能的前景,标记完成后请按n进行查看. \n')
            value = DRAW_PR_FG
        elif k == ord('s'):
            bar = np.zeros((img.shape[0],5,3),np.uint8)
            res = np.hstack((img2,bar,img,bar,output))
            cv.imwrite('grabcut_output.png',res)
            print('存储成功 \n')
        elif k == ord('r'):
            print('重置 \n')
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8)
            output = np.zeros(img.shape,np.uint8)
        elif k == ord('n'):
            print('请按0-3对图像进行标记，调整完毕后再次按n进行查看,满足要求按s保存,不满足按r重置.(0:背景,1:前景.2:可能是背景,3:可能是前景)\n')
            if (rect_or_mask == 0):         # rect模式
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:         # mask模式
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv.bitwise_and(img2,img2,mask=mask2)

    cv.destroyAllWindows()