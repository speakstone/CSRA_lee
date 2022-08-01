import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA, ResNet_CSRA_50
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
# from utils.evaluation.eval import WarmUpLR
from tqdm import tqdm


def Args():
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # model default resnet101
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--load_from", default="checkpoint/resnet50/epoch_1.pth", type=str)
    # dataset
    parser.add_argument("--datadir", default="/work/dataset/huawei_2022_2/train_image/labeled_data/", type=str)
    parser.add_argument("--dataset", default="Lane", type=str)
    parser.add_argument("--num_cls", default=8, type=int)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=[768,768], type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    args = parser.parse_args()
    return args
    

def val(args, model, test_loader, test_file):
    model.eval()
    print("Test on Pretrained Models")
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        # if index > 10:
        #     break
    # for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)
            logit = torch.mean(logit, -1)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])


def main():
    args = Args()
    # model 
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.model == "resnet50":
        model = ResNet_CSRA_50(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)

    model.cuda()
    print("Loading weights from {}".format(args.load_from))
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.module.load_state_dict(torch.load(args.load_from))
    else:
        model.load_state_dict(torch.load(args.load_from))

    # data
    if args.dataset == "voc07":
        test_file = ['data/voc07/test_voc07.json']
    if args.dataset == "coco":
        test_file = ['data/coco/val_coco2014.json']
    if args.dataset == "wider":
        test_file = ['data/wider/test_wider.json']
    if args.dataset == "Lane":
        train_file = ['/work/dataset/huawei_2022_2/train_label/rows_train.npy']
        test_file = ['/work/dataset/huawei_2022_2/train_label/rows_test.npy']
        step_size = 5
    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset, args.datadir, args.num_cls,  False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val(args, model, test_loader, test_file)


if __name__ == "__main__":
    main()
