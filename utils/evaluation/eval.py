import argparse
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

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



#---自己按照公式实现
def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


def evaluation(result, types, ann_path, datadir= "", num_cls=8):
    print("Evaluation")
    classes = class_dict[types]
    classes = classes[:1]
    aps = np.zeros(len(classes), dtype=np.float64)

    json_results = {}
    json_npy = np.load(ann_path, allow_pickle=True)
    for index, js in enumerate(json_npy):
        json_results[index] = {}
        json_results[index]["img_path"] = os.path.join(datadir, js[0])
        target_i = np.zeros(num_cls)
        if js[1] != "0":
            json_results[index]["target"] = [0]
        else:
            json_results[index]["target"] = [1]

    # pred_json = result
    pred_json = []
    ann_json = json_results
    for i in result:
        i["scores"] = [i["scores"][0]]
        # i["scores"] = [i["scores"][0], 1 - i["scores"][0]]
        # dot = np.prod([1 - float(k) for k in i["scores"][1:]])
        # i["scores"] = [1 - dot, dot]
        # i["scores"] = [1-min(sum(i["scores"][1:]), 0.999999), min(sum(i["scores"][1:]), 0.99999)]
        pred_json.append(i)

    for i, _ in enumerate(tqdm(classes)):
        ap = json_map(i, pred_json, ann_json, types)
        aps[i] = ap
    OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes), types)

    ro_gt = {i['img_path'].split(".")[0] : i['target'][0] for i in ann_json.values()}
    ro_pred = {i['file_name'].split(".")[0] : i['scores'][0] for i in pred_json}
    ro_gt_list = []
    ro_pred_list = []
    for sample in ro_gt.keys():
        ro_gt_list.append(ro_gt[sample])
        ro_pred_list.append(ro_pred[sample])
    fpr, tpr, thresholds = roc_curve(ro_gt_list, ro_pred_list, pos_label=1)
    print("-----ROC:", auc(fpr, tpr))
    print("mAP: {:4f}".format(np.mean(aps)))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))




