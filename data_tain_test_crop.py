import csv
import random
import numpy as np
csv_path = "/work/dataset/huawei_2022_2/train_label_805/train_label.csv"
train_path = "train_label_805/rows_train.npy"
test_path = "train_label_805/rows_test.npy"

# 载入csv
ff_list = []
ff_dict = {}
ff_dict_list = [[] for _ in range(8)]
with open(csv_path, "r") as f:
    f_ = csv.reader(f)
    heard = next(f_)
    for rr in f_:
        ff_list.append([i for i in rr if len(i) > 0])


for ff in ff_list:
    ff_dict_list[int(random.choice(ff[1:]))].append(ff[0])
    ff_dict[ff[0]] = ff[1:]

ff_dict_list_up = [j for i in ff_dict_list for j in i]

ff_len = [len(i) for i in ff_dict_list]


# 随机挑选400个负样本和正样本
f_f_roat = [int(i/sum(ff_len[1:]) * 405) for i in ff_len[1:]]

f_f_data = [np.random.choice(i, j,  replace=False).tolist() for i,j in zip(ff_dict_list[1:], f_f_roat)]
f_f_data = [j for i in f_f_data for j in i]
f_t_data = np.random.choice(ff_dict_list[0], 400,  replace=False).tolist()

test_data = f_f_data + f_t_data
train_data = list(set(ff_dict_list_up) - set(test_data))

test_npy = [[i, *ff_dict[i]] for i in test_data]
train_npy = [[i, *ff_dict[i]] for i in train_data]

np.save(test_path, test_npy)
np.save(train_path, train_npy)