# -*- coding: utf-8 -*-

from csv import reader
import random
import csv
import os

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def insertLineIntoFile(filename, index=1, text=''):
    '''
    @brief: 将指定的文件内容插入到文件的指定的一行中
    @param： index
        插入的行数，从1开始。
    @param： filename
        文件名
    '''
    fid = open(filename, 'r')
    lines = []
    for line in fid:
        lines.append(line)
    fid.close()

    text = text + '\n'
    lines.insert(index - 1, text)
    s = ''.join(lines)
    print len(lines)
    fid = open(filename, 'w')
    fid.write(s)
    fid.close()


def file_clear(FilePath):
    file = open(FilePath, 'w')
    file.seek(0)
    file.truncate()
    file.close()


def transformLevel(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        '''
        if value>=1 and value<2:
            row[column] = 1
        elif value>=2 and value<3:
            row[column] = 2
        elif value>=3 and value<4:
            row[column] = 3
        elif value>=4 and value<=5:
            row[column] = 4
        else:
            row[column] = 0
        '''


def combine_all_feature(entropy_file, landmark_file, degree_file, all_feature_file):
    landmark_data = load_csv(landmark_file)
    entropy_data = load_csv(entropy_file)
    degree_data = load_csv(degree_file)
    file_clear(all_feature_file)
    i = 1
    for landmark_row in landmark_data:
        picName = landmark_row[len(landmark_row) - 1]
        for entropy_row in entropy_data:
            if picName == entropy_row[len(entropy_row) - 1]:
                all_feature_row = ''
                for m in range(0, len(entropy_row) - 1):
                    all_feature_row += entropy_row[m] + ','
                for n in range(0, len(landmark_row) - 1):
                    all_feature_row += landmark_row[n] + ','
                emotion_label = picName.split('-')[1][0:2]
                for degree_row in degree_data:
                    if picName == degree_row[len(degree_row) - 1]:
                        for k in range(0, len(degree_row) - 1):
                            all_feature_row += degree_row[k] + ','
                        all_feature_row = all_feature_row + emotion_label + "," + picName
                        insertLineIntoFile(all_feature_file, i, all_feature_row)
                        i += 1
                        break


def spilt_test_data(all_feature_file, train_file, test_file,rate):
    file_clear(test_file)
    file_clear(train_file)
    dataset = load_csv(all_feature_file)
    number = len(dataset)
    limit_number = int(number * rate / 7)
    HA_list = list()
    SA_list = list()
    SU_list = list()
    DI_list = list()
    AN_list = list()
    FE_list = list()
    NE_list = list()
    for row in dataset:
        tag = row[-2]
        if tag == "HA":
            HA_list.append(row)
        elif tag == "SA":
            SA_list.append(row)
        elif tag == "SU":
            SU_list.append(row)
        elif tag == "DI":
            DI_list.append(row)
        elif tag == "AN":
            AN_list.append(row)
        elif tag == "FE":
            FE_list.append(row)
        elif tag == "NE":
            NE_list.append(row)

    test_list = random.sample(HA_list, limit_number) + random.sample(SA_list, limit_number) \
                + random.sample(SU_list,limit_number)+random.sample(DI_list, limit_number) \
                + random.sample(AN_list, limit_number) + random.sample(FE_list,limit_number) \
                + random.sample(NE_list, limit_number)

    train_list = [val for val in dataset if val not in test_list]

    #print ("the number of train set : {}".format(len(train_list)))
    #print ("the number of test set : {}".format(len(test_list)))
    with open(train_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train_list)
        csvfile.close()
    with open(test_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_list)
        csvfile.close()
'''
    for row in dataset:
        tag = row[-2]
        if count_dic[tag] <= limit_number:
            count_dic[tag] += 1
            test_list.append(row)
        else:
            train_list.append(row)
'''



if __name__ == "__main__":

    #仅调用一次 生成所有数据集合。
    entropy_file = os.getcwd() + '/data_set/gray_feature.csv'
    landmark_file = os.getcwd() + '/data_set/landmark_feature.csv'
    degree_file = os.getcwd() + "/data_set/degree.csv"
    all_data_file = os.getcwd() + "/data_set/all_data.csv"
    #combine_all_feature(entropy_file, landmark_file, degree_file, train_file)

    #按比例随机生成
    train_file = os.getcwd() + "/data_set/train.csv"
    test_file = os.getcwd() + "/data_set/test.csv"
    spilt_test_data(all_data_file, train_file, test_file,0.2)

