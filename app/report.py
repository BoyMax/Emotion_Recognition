# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
import pandas as pd
from datetime import datetime

def get_emotion_dict():
    emotions = dict()
    emotions['0'] = "happy"
    emotions['1'] = "sad"
    emotions['2'] = "surprise"
    emotions['3'] = "angry"
    emotions['4'] = "disgust"
    emotions['5'] = "fear"
    emotions['6'] = "normal"
    emotions['null'] = "null"
    return emotions

def generate_report(timestamps, emotion_values):
    emotions = get_emotion_dict()
    y_labels = []
    for emotion_value in emotion_values:
        y_labels.append(emotions[emotion_value])
    x_time = []
    for time in timestamps:
        x_time.append(datetime.strptime(time,'%H:%M:%S'))

    #画折线图：
    fig=plt.figure(figsize=(10,4))
    plt.plot(x_time, y_labels,linewidth=1,color='b',marker='o', markerfacecolor='red',markersize=5)
    plt.xticks(rotation=70)
    plt.ylabel('Emotion')
    plt.xlabel('Timeline')
    plt.title('emotion changes')
    plt.show()


if __name__ == "__main__":
    #timestamps = ['2018-04-23 14:10:21', '2018-04-23 14:10:24', '2018-04-23 14:10:27', '2018-04-23 14:10:30', '2018-04-23 14:10:34', '2018-04-23 14:10:37', '2018-04-23 14:10:40', '2018-04-23 14:10:44', '2018-04-23 14:10:47', '2018-04-23 14:10:50', '2018-04-23 14:10:54', '2018-04-23 14:10:57', '2018-04-23 14:11:00', '2018-04-23 14:11:04', '2018-04-23 14:11:07']
    timestamps = ['14:10:21', '14:10:24', '14:10:27', '14:10:30', '14:10:34', '14:10:37', '14:10:40', '14:10:44', '14:10:47', '14:10:50', '14:10:54', '14:10:57', '14:11:00', '14:11:04', '14:11:07']

    emotion_values = ['1', '3', '0', '1', '4', '1', '3', '4', '4', '3', '4', '4', '1', '4', '3']
    generate_report(timestamps, emotion_values)
