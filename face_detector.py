# -*- coding: utf-8 -*-

import dlib
import os

def get_face_detector():
    #人脸检测器：
    detector = dlib.get_frontal_face_detector()

    #关键点训练器：
    predictor_path = "/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

