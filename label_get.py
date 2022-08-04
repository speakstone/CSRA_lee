import csv
import os
import numpy as np
data_test = "/work/dataset/huawei_2022_2/test_images/"
data_list = os.listdir(data_test)
data_list = sorted(data_list, key=lambda x: float(x.split("_")[1].split(".")[0]))
data_get = [[i, 0] for i in data_list]


import csv
log_path = '/work/dataset/huawei_2022_2/test_label/test.npy'
np.save(log_path, data_get)
# file = open(log_path, 'a+', encoding='utf-8', newline='')
# csv_writer = csv.writer(file)
# for i in data_get:
#     csv_writer.writerow(i)
# file.close()
