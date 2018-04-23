#-*- coding: utf-8 -*-



import pandas as pd
import os
import csv_helper
import numpy as np
from datetime import datetime,timedelta
import time

if __name__ == "__main__":

    start = datetime.now()#.strftime('%Y-%m-%d %H:%M:%S')
    print(start)
    time.sleep(2)
    end = datetime.now()
    print end
    print(end-start == timedelta(seconds=2) ) #时间间隔2秒 ,True

    '''分量的相关性系数
    degree_file = os.getcwd() + "/data_set/correlation.csv"
    degree_data = pd.read_csv(degree_file)

    print degree_data
    df = degree_data.corr()
    print df
'''
