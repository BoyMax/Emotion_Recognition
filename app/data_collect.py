# -*- coding: utf-8 -*-

import cv2
import face_detector
import os
import dlib
import numpy as np

def file_clear(FilePath):
    file = open(FilePath, 'w')
    file.seek(0)
    file.truncate()
    file.close()

def write_file(file, content):
    with open(file, 'a') as f:
        f.write(content + '\n')
        f.close()

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

def capture_image(emotion, label):
    file_num =0
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("Unable to connect to camera")
    detector, predictor = face_detector.get_face_detector()
    while True:
        input_str = raw_input("press keyboard to capture " + emotion + ' ' + str(file_num));

        #if cv2.waitKey(-1) & 0xFF == ord('q'):
        #    break
        if file_num == 50:
            return
        file_num += 1
        ret, img = cap.read()

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
            dist.append(str(distance(shape.part(21), shape.part(22))/factor))
            #眉毛中心到眉头距离
            dist.append(str(distance(shape.part(19), shape.part(21))/factor))
            dist.append(str(distance(shape.part(22), shape.part(24))/factor))
            #眼睛宽度
            dist.append(str(distance(shape.part(36), shape.part(39))/factor))
            dist.append(str(distance(shape.part(42), shape.part(45))/factor))
            #眼睛高度
            dist.append(str(distance(shape.part(37), shape.part(41))/factor))
            dist.append(str(distance(shape.part(44), shape.part(46))/factor))
            #眼眉距离
            dist.append(str(distance(shape.part(19), shape.part(37))/factor))
            dist.append(str(distance(shape.part(24), shape.part(44))/factor))
            #鼻宽
            dist.append(str(distance(shape.part(31), shape.part(35))/factor))
            #上下唇距
            dist.append(str(distance(shape.part(51), shape.part(57))/factor))
            #上唇距
            dist.append(str(distance(shape.part(51), shape.part(62))/factor))
            #下唇距
            dist.append(str(distance(shape.part(57), shape.part(66))/factor))
            #上唇-下巴距离
            dist.append(str(distance(shape.part(51), shape.part(8))/factor))
            #下唇-下巴距离
            dist.append(str(distance(shape.part(57), shape.part(8))/factor))

            #找特征角度：
            angle = []
            #外唇围角度值
            for i in range(48, 61):
                if shape.part(i).x - shape.part(i+1).x == 0:
                    gradient = '0'
                else:
                    gradient = str(float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x)))
                angle.append(gradient)
            #眉形角度值：
            for i in range(17, 26):
                if shape.part(i).x - shape.part(i+1).x == 0:
                    gradient = '0'
                else:
                    gradient = str(float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x)))
                angle.append(gradient)
            #轮廓角度值：
            for i in range(0, 16):
                if shape.part(i).x - shape.part(i+1).x == 0:
                    gradient = '0'
                else:
                    gradient = str(float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x)))
                angle.append(gradient)

            #综合距离特征和角度特征：
            feature = ''
            for d in dist:
                feature += d + ","
            for a in angle:
                feature += a + ","
            feature += str(label)
            file = os.getcwd() +"/train/train_data.csv"
            write_file(file, feature)

            for i in range(68):
                shape = shapes[0]
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), -1)
                cv2.putText(img, str(i),(shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            #cv2.imshow('feature points', img)
            #cv2.waitKey(0)
            pic_file = os.getcwd() + "/" +emotion +"_train/"+str(file_num) +".jpg"
            cv2.imwrite(pic_file, img)
    cv2.destroyAllWindows()

if __name__ =="__main__":
    emotions = ["happy","sad","surprise","angry","disgust","fear","normal"]
    emotion = dict()
    file_clear(os.getcwd() +"/train/train_data.csv")
    for i, value in enumerate(emotions):
        emotion[value] = i
        print("Start to capture "+ value + " emotion")
        capture_image(value,i)

