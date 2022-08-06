from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from .csra import CSRA, MHA
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




class ResNet_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)

        self.classifier = MHA(num_heads, lam, input_dim, num_classes) 
        self.loss_func = F.binary_cross_entropy_with_logits
        self.loss_ce = nn.CrossEntropyLoss()
        self.softxmax = nn.Softmax(dim=1)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def loss_func_zeros(self, logit, target):
        logit_t = torch.sum(logit[:, 1:, :], -2)
        logit_f = torch.sum(logit[:, :1, :], -2)
        logit_s = torch.stack((logit_f, logit_t), -1)

        target = target.unsqueeze(-1)
        target_t = torch.sum(target[:, 1:, :], -2)
        target_f = torch.sum(target[:, :1, :], -2)
        target_s = torch.stack((target_f, target_t), -1)

        # 设置最大值最小值
        target_s[target_s > 0] = 1
        logit_s[logit_s < 0.0000001] = 0.0000001
        logit_s[logit_s > 0.9999999] = 0.9999999

        loss_l = -1 * torch.mean(target_s * torch.log(logit_s) + (1 - target_s) * torch.log(1 - logit_s))

        # target_m = torch.mean(target_s, 1)
        # target_m = torch.argmax(target_m, -1)
        # logit_m = torch.mean(logit_s, 1)
        # loss_ce = self.loss_ce(logit_m, target_m)

        return loss_l

    def loss_func_lee(self, logit, target):
        target = target.unsqueeze(-1)
        loss_f = -1 * torch.sum((1 - target) * torch.log(1 - logit + 0.0000001)) / (torch.sum((1 - target)) + 0.0000001)
        pred_t = torch.sum(target * (logit - 0.00000001), 1)
        loss_t = -1 * torch.mean(torch.log(pred_t + 0.0000001))
        # if torch.isnan(loss_f) or torch.isnan(loss_t):
        #     print(loss_f, loss_t, target.max())

        return (loss_f + loss_t) / 2

    def forward_train_lee(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        logit = self.softxmax(logit)
        # loss = self.loss_func(logit, target, reduction="mean")
        loss1 = self.loss_func_lee(logit, target)
        loss2 = self.loss_func_zeros(logit, target)
        loss = loss1 + 0.5 * loss2
        # loss = loss1
        return logit, loss, loss1, loss2

    def forward_test_lee(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        logit = self.softxmax(x)
        return logit

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train_lee(x, target)
        else:
            return self.forward_test_lee(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original 1000-class fc
        self.fc = nn.Sequential()


class ResNet_CSRA_50(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=50, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA_50, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)

        self.classifier = MHA(num_heads, lam, input_dim, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits
        self.loss_ce = nn.CrossEntropyLoss()
        self.softxmax = nn.Softmax(dim=1)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def loss_func_zeros(self, logit, target):
        logit_t = torch.sum(logit[:, 1:, :], -2)
        logit_f = torch.sum(logit[:, :1, :], -2)
        logit_s = torch.stack((logit_f, logit_t), -1)
        logit_m = torch.mean(logit_s, 1)

        target = target.unsqueeze(-1)
        target_t = torch.sum(target[:, 1:, :], -2)
        target_f = torch.sum(target[:, :1, :], -2)
        target_s = torch.stack((target_f , target_t), -1)
        target_m = torch.mean(target_s, 1)
        target_m = torch.argmax(target_m, -1)
        return self.loss_ce(logit_m, target_m)

    def loss_func_lee(self, logit, target):
        target = target.unsqueeze(-1)
        loss_f = -1 * torch.sum((1 - target) * torch.log(1 - logit + 0.0000001)) / (torch.sum((1 - target)) + 0.0000001)
        pred_t = torch.sum(target * (logit - 0.00000001), 1)
        loss_t = -1 * torch.mean(torch.log(pred_t + 0.0000001))

        return (loss_f + loss_t) / 2

    def forward_train_lee(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        logit = self.softxmax(logit)
        # loss = self.loss_func(logit, target, reduction="mean")
        loss1 = self.loss_func_lee(logit, target)
        loss2 = self.loss_func_zeros(logit, target)
        # loss = (loss1 + loss2)/2
        loss = loss1 + loss2
        return logit, loss, loss1, loss2

    def forward_test_lee(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        logit = self.softxmax(x)
        return logit

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train_lee(x, target)
        else:
            return self.forward_test_lee(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original 1000-class fc
        self.fc = nn.Sequential()