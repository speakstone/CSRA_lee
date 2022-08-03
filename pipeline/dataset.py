import json
import os
import csv
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np

# img = Image.open("/work/dataset/huawei_2022_2/test_images/test_00001.png")
# w, h = img.size
# resize = transforms.Resize([960,432])
# img = resize(img)
# img.save('2.png')
# modify for transformation for vit
# modfify wider crop-person images


class DataSet(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                datadir,
                num_cls,
                train
                ):
        self.dataset = dataset
        self.datadir = datadir
        self.ann_files = ann_files
        self.num_cls = num_cls
        self.train = train
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        # in wider dataset we use vit models
        # so transformation has been changed
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ] 
            )        

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if "rotate" in augs:
            t.append(transforms.RandomHorizontalFlip(p=0.2))
            t.append(transforms.RandomVerticalFlip(p=0.2))
        # if 'RandAugment' in augs:
        #     t.append(RandAugment())

        t.append(transforms.Resize((img_size[0], img_size[1])))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = {}
        for idx, ann_file in enumerate(self.ann_files):
            json_results = {}
            json_npy = np.load(ann_file, allow_pickle=True).tolist()
            # 进行一次数据均衡
            if self.train:
                json_npy = self.data_balance(json_npy)

            for index, js in enumerate(json_npy):
                json_results[index] = {}
                json_results[index]["img_path"] = os.path.join(self.datadir, js[0])
                target_i = np.zeros(self.num_cls)
                for i in js[1:]:
                    target_i[int(i)] = 1
                json_results[index]["target"] = np.array(target_i, dtype=np.int)

            # with open(ann_file, 'r', encoding="utf-8") as f:
            #     reader = csv.reader(f)
            #     for index, item in enumerate(reader):
            #         # 忽略第一行
            #         if reader.line_num == 1:
            #             continue
            #         json_results[index] = {}
            #         json_results[index]["img_path"] = os.path.join(self.datadir, item[0])
            #         target_i = np.zeros(self.num_cls)
            #         for i in item[1].split(","):
            #             target_i[int(i)] = 1
            #         json_results[index]["target"] = np.array(target_i, dtype=np.int)
            anns_len = len(self.anns)
            for key, value in json_results.items():
                self.anns[key+anns_len] = value
            self.anns = self.anns

    def data_balance(self, json_npy):
        """
        每一类数据扩充到相同
        :param json_npy:
        :return:
        """
        json_list = []
        json_dict = {str(i) : [] for i in range(self.num_cls)}
        for index, j_i in enumerate(json_npy):
            for cls in j_i[1:]:
                json_dict[cls].append(index)
        json_len = [len(list(i)) for i in json_dict.values()]
        jlmax = max(json_len)
        for js, jl in zip(list(json_dict.values()), json_len):
            js_ = js * min(jlmax // jl, 50)
            np.random.shuffle(js_)
            for js_i in js_:
                json_list.append(json_npy[js_i])
        return json_list

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        # while not os.path.isfile(ann["img_path"]):
        #     idx *= 2
        #     idx = idx % len(self)
        #     ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")

        if self.dataset == "wider":
            x, y, w, h = ann['bbox']
            img_area = img.crop([x, y, x+w, y+h])
            img_area = self.augment(img_area)
            img_area = self.transform(img_area)
            message = {
                "img_path": ann['img_path'],
                "target": torch.Tensor(ann['target']),
                "img": img_area
            }
        elif self.dataset == "Lane":
            # img.save("0.jpg")
            img = img.crop((0, 305, img.size[0], 2160))
            # img.save("1.jpg")
            img = self.augment(img)
            img = self.transform(img)

            from torchvision import utils as vutils
            vutils.save_image(img, "1.jpg")

            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(np.array(ann["target"], dtype=np.float)),
                "img": img
            }
        else: # voc and coco
            img = self.augment(img)
            img = self.transform(img)
            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(np.array(ann["target"], dtype=np.float)),
                "img": img
            }

        return message
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }
