# -*- coding: utf-8 -*-

import cv2
import numpy as np
import face_detector
import os

from datetime import datetime, timedelta
from sklearn.externals import joblib


#t1,t2 元组表示坐标点
def distance(t1,t2):
    x1 = t1.x
    y1 = t1.y
    x2 = t2.x
    y2 = t2.y
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    dist = np.sqrt(np.sum(np.square(p1 - p2)))
    return dist



def get_feature(img,detector, predictor):
    faces = detector(img)
    shapes = []
    if len(faces) != 0:
        for face in faces:
            shape = predictor(img, face)
            shapes.append(shape)
    if len(shapes) != 0:
        face = faces[0]
        leftTop = (int(face.left()), int(face.top()))
        rightBottom = (int(face.right()), int(face.bottom()))
        cv2.rectangle(img, leftTop, rightBottom, (0, 0, 255), 1, 4, 0)

        shape = shapes[0]

        #采用脸部宽度做归一化衡量标准
        factor = distance(shape.part(0), shape.part(16))

        #找特征距离：
        dist = []
        #眉间距：
        dist.append(distance(shape.part(21), shape.part(22))/factor)
        #眉毛中心到眉头距离
        dist.append(distance(shape.part(19), shape.part(21))/factor)
        dist.append(distance(shape.part(22), shape.part(24))/factor)
        #眼睛宽度
        dist.append(distance(shape.part(36), shape.part(39))/factor)
        dist.append(distance(shape.part(42), shape.part(45))/factor)
        #眼睛高度
        dist.append(distance(shape.part(37), shape.part(41))/factor)
        dist.append(distance(shape.part(44), shape.part(46))/factor)
        #眼眉距离
        dist.append(distance(shape.part(19), shape.part(37))/factor)
        dist.append(distance(shape.part(24), shape.part(44))/factor)
        #鼻宽
        dist.append(distance(shape.part(31), shape.part(35))/factor)
        #上下唇距
        dist.append(distance(shape.part(51), shape.part(57))/factor)
        #上唇距
        dist.append(distance(shape.part(51), shape.part(62))/factor)
        #下唇距
        dist.append(distance(shape.part(57), shape.part(66))/factor)
        #上唇-下巴距离
        dist.append(distance(shape.part(51), shape.part(8))/factor)
        #下唇-下巴距离
        dist.append(distance(shape.part(57), shape.part(8))/factor)

        #找特征角度：
        angle = []
        #外唇围角度值
        for i in range(48, 61):
            if shape.part(i).x - shape.part(i+1).x == 0:
                gradient = '0'
            else:
                gradient = float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x))
            angle.append(gradient)
        #眉形角度值：
        for i in range(17, 26):
            if shape.part(i).x - shape.part(i+1).x == 0:
                gradient = '0'
            else:
                gradient = float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x))
            angle.append(gradient)
        #轮廓角度值：
        for i in range(0, 16):
            if shape.part(i).x - shape.part(i+1).x == 0:
                gradient = '0'
            else:
                gradient = float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x))
            angle.append(gradient)

        feature = np.zeros(53)
        i = 0
        for d in dist:
            feature[i] = d
            i += 1
        for a in angle:
            feature[i] = a
            i += 1
        return feature


def capture():
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("Unable to connect to camera")
    detector, predictor = face_detector.get_face_detector()
    rf = joblib.load(os.getcwd() + '/model/model.m')

    #reload(sys)
    #sys.setdefaultencoding('utf8')
    #cap = cv2.VideoCapture(0)
    #cap.set(3,640)
    #cap.set(4,480)
    #cap.set(1, 10.0)
    #此处fourcc的在MAC上有效，如果视频保存为空，那么可以改一下这个参数试试, 也可以是-1
    #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    # 第三个参数则是镜头快慢的，10为正常，小于10为慢镜头
    #out = cv2.VideoWriter('/opt/code/video/output2.avi', fourcc,10,(640,480))

    frames = 0
    timestamps = []
    emotion_values = []
    while True:
        ret, frame = cap.read()
        if ret == True:
            #frame = cv2.flip(frame, 1)
            #a = out.write(frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            else:
                frames += 1
                try:
                    feature = get_feature(frame, detector, predictor)
                    if not feature is None:
                        feature = feature.reshape(-1,53)
                    emotion_value = rf.predict(feature)[0]
                    if emotion_value == '0':
                        cv2.putText(frame, 'happy', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '1':
                        cv2.putText(frame, 'sad', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '2':
                        cv2.putText(frame, 'surprise', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '3':
                        cv2.putText(frame, 'angry', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '4':
                        cv2.putText(frame, 'disgust', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '5':
                        cv2.putText(frame, 'fear', (300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    elif emotion_value == '6':
                        cv2.putText(frame, 'normal',(300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                    cv2.imshow("frame", frame)
                    if (frames > 30):
                        frames = 0
                        timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        emotion_values.append(emotion_value)

                except Exception,err:
                    cv2.putText(frame, 'No face detected',(270,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
                    cv2.imshow("frame", frame)

                    if (frames > 30):
                        frames = 0
                        timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        emotion_values.append("null")
                    print 'Error: ', err
        else:
            break
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    return timestamps, emotion_values



if __name__ =="__main__":
    timestamps, emotion_values = capture()
    print(timestamps)
    print(emotion_values)
