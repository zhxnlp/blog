import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, help='path to input video file')
ap.add_argument('-t', '--tracker', type=str, default='kcf', help='OpenCV object tracker type')
args = vars(ap.parse_args())

# MultiTracker_create以及一些其他的目标追踪算法在opencv4.5以后换了地方. 
# cv2.legacy.MultiTracker_create

# 定义OpenCV中的七种目标追踪算法
OPENCV_OBJECT_TRACKERS = {
    'boosting': cv2.legacy_TrackerBoosting.create,
    'csrt': cv2.legacy_TrackerCSRT.create,
    'kcf':  cv2.legacy.TrackerKCF.create,
    'mil': cv2.legacy.TrackerMIL.create,
    'tld': cv2.legacy_TrackerTLD.create,
    'medianflow': cv2.legacy_TrackerMedianFlow.create,
    'mosse': cv2.legacy_TrackerMOSSE.create  
}

trackers = cv2.legacy.MultiTracker_create()
cap = cv2.VideoCapture(args['video'])

while True:
    flag, frame = cap.read()
    if frame is None:
        break
        
    # 追踪目标
    success, boxes = trackers.update(frame)
    # 绘制追踪到的矩形区域
    for box in boxes:
        # box是个浮点型, 画图需要整型
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(100)
    if key == ord('s'):
        # 框选ROI区域
        roi = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)
#         print(roi)
        # 创建一个实际的目标追踪器
        tracker = OPENCV_OBJECT_TRACKERS[args['tracker']]()
        trackers.add(tracker, frame, roi)
    elif key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()