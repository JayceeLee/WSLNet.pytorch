import torch 
import torch.nn as nn 

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def pairwise_distance(inputs_):
    n = inputs_.size(0)
    # print(4*'\n', x.size())
    inputs_ = normalize(inputs_)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist 


class DivLoss(nn.Module):
    def __init__(self, margin=1):
        super(DivLoss, self).__init__()
        self.margin = margin 

    def forward(self, input):
        b, c, h, w = input.size()
        loss_sum = 0 
        for i in range(b):
            feat = input[i].view(c, h*w)
            dist_mat = pairwise_(feat)
            maxout = torch.clamp(dist_mat, min=self.margin)
            loss_sum = loss_sum + torch.sum(maxout)
        return loss_sum / b




