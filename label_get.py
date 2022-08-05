import csv
import os
import numpy as np
import csv
# data_test = "/work/dataset/huawei_2022_2/test_images/"
# data_list = os.listdir(data_test)
# data_list = sorted(data_list, key=lambda x: float(x.split("_")[1].split(".")[0]))
# data_get = [[i, 0] for i in data_list]
#
#
# import csv
# log_path = '/work/dataset/huawei_2022_2/test_label/test.npy'
# np.save(log_path, data_get)
# # file = open(log_path, 'a+', encoding='utf-8', newline='')
# # csv_writer = csv.writer(file)
# # for i in data_get:
# #     csv_writer.writerow(i)
# # file.close()

import csv
#读取csv文件
ff_list = []
with open("submission.csv", "r") as f:
    f_ = csv.reader(f)
    heard = next(f_)
    for rr in f_:
        ff_list.append(rr)

ff_to_list = []
for ii in ff_list:
    # if float(ii[1]) > 0.9:
    #     ff_to_list.append([ii[0], "0.999999999"])
    # elif float(ii[1]) < 0.1:
    #     ff_to_list.append([ii[0], "0.0000001"])
    # else:
    #     ff_to_list.append(ii)
    ff_to_list.append([ii[0], float(np.random.random() * 0.0001)])


with open("submission.csv", 'w+', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['imagename', 'defect_prob'])
    for i in ff_to_list:
        csv_writer.writerow(i)
# csv_file = csv.reader(open('submission_0.967.csv'))
# print(csv_file)