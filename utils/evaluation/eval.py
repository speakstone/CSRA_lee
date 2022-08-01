import argparse
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3


voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
wider_classes = (
                "Male","longHair","sunglass","Hat","Tshiirt","longSleeve","formal",
                "shorts","jeans","longPants","skirt","faceMask", "logo","stripe")

Lane_classes = (
                "True","center_line","stop_line","guide_surface","shoulder",
                "pavement", "arrow", "lane_line",
                )

class_dict = {
    "voc07": voc_classes,
    "coco": coco_classes,
    "wider": wider_classes,
    "Lane": Lane_classes
}



def evaluation(result, types, ann_path, datadir= "", num_cls=8):
    print("Evaluation")
    classes = class_dict[types]
    aps = np.zeros(len(classes), dtype=np.float64)

    json_results = {}
    json_npy = np.load(ann_path, allow_pickle=True)
    for index, js in enumerate(json_npy):
        json_results[index] = {}
        json_results[index]["img_path"] = os.path.join(datadir, js[0])
        target_i = np.zeros(num_cls)
        # for i in js[1:]:
        #     target_i[int(i)] = 1
        # json_results[index]["target"] = np.array(target_i, dtype=np.int)
        if js[1] != "0":
            json_results[index]["target"] = [0, 1]
        else:
            json_results[index]["target"] = [1, 0]

    # pred_json = result
    pred_json = []
    ann_json = json_results
    for i in result:
        # dot = np.prod([1 - float(k) for k in i["scores"][1:]])
        # i["scores"] = [1 - dot, dot]
        i["scores"] = [1-min(sum(i["scores"][1:]), 0.999999), min(sum(i["scores"][1:]), 0.99999)]

        pred_json.append(i )
    # 修改为@2分类@
    classes = classes[:2]
    for i, _ in enumerate(tqdm(classes)):
        ap = json_map(i, pred_json, ann_json, types)
        aps[i] = ap
    OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes), types)
    print("mAP: {:4f}".format(np.mean(aps)))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))



