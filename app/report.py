# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

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
        x_time.append(time)

    #画折线图：
    plt.plot(x_time,y_labels,label='Frist line',linewidth=2,color='b',marker='o', markerfacecolor='red',markersize=10)
    plt.xlabel('Emotion')
    plt.ylabel('Timeline')
    plt.title('emotion changes')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    timestamps = ['2018-04-23 14:10:21', '2018-04-23 14:10:24', '2018-04-23 14:10:27', '2018-04-23 14:10:30', '2018-04-23 14:10:34', '2018-04-23 14:10:37', '2018-04-23 14:10:40', '2018-04-23 14:10:44', '2018-04-23 14:10:47', '2018-04-23 14:10:50', '2018-04-23 14:10:54', '2018-04-23 14:10:57', '2018-04-23 14:11:00', '2018-04-23 14:11:04', '2018-04-23 14:11:07']
    #timestamps = ['2018-04-23 14:10:21', '2018-04-23 14:10:30',  '2018-04-23 14:10:40', '2018-04-23 14:10:50',  '2018-04-23 14:11:00', '2018-04-23 14:11:07']

    emotion_values = ['1', '3', '0', '1', '4', '1', '3', '4', '4', '3', '4', '4', '1', '4', '3']
    generate_report(timestamps, emotion_values)
