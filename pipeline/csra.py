import torch
import torch.nn as nn



class CSRA(nn.Module): # one basic block 
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda
        self.num_classes = num_classes
        self.head1 = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.head2 = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.head3 = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.head4 = nn.Conv2d(input_dim, num_classes, 1, bias=False)

        # to 4 * head
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # # x (B d H W)
        # # normalize classifier
        # # score (B C HxW)
        # score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
        # score = score.flatten(2)
        # base_logit = torch.mean(score, dim=2)
        #
        # if self.T == 99: # max-pooling
        #     att_logit = torch.max(score, dim=2)[0]
        # else:
        #     score_soft = self.softmax(score * self.T)
        #     att_logit = torch.sum(score * score_soft, dim=2)
        #
        # return base_logit + self.lam * att_logit
        # (8 * 4)
        # [[0,1], [0,1], [0,1], [0,1], [0,1]...]
        # [[0,1], ...]

        score1 = self.head1(x) / torch.norm(self.head1.weight, dim=1, keepdim=True).transpose(0, 1)
        score2 = self.head2(x) / torch.norm(self.head2.weight, dim=1, keepdim=True).transpose(0, 1)
        score3 = self.head3(x) / torch.norm(self.head3.weight, dim=1, keepdim=True).transpose(0, 1)
        score4 = self.head4(x) / torch.norm(self.head4.weight, dim=1, keepdim=True).transpose(0, 1)
        score = torch.stack((score1, score2, score3, score4), 2)
        score = score.flatten(3)
        # score1 = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        # score = score.flatten(2)
        # score = torch.stack(torch.chunk(score, 4, 1), 2)
        base_logit = torch.mean(score, -1)

        # return base_logit
        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=-1)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=-1)

        # if self.T == 99: # max-pooling
        #     att_logit = torch.max(score, dim=2)[0]
        # else:
        #     score_soft = self.softmax(score * self.T)
        #     att_logit = torch.sum(score * score_soft, dim=2)
        #
        # return base_logit + self.lam * att_logit.unsqueeze(-1)

        return base_logit + self.lam * att_logit



class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
