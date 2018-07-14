import torch 
import torch.nn as nn 
import random 

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
    def __init__(self, num_classes, num_maps, margin=1, num_samples=5):
        super(DivLoss, self).__init__()
        self.margin = margin 
        self.num_classes = num_classes
        self.num_maps = num_maps 
        self.num_samples = num_samples

    def forward(self, input_):
        b, _, h, w = input_.size()
        num_samples = self.num_samples
        if num_samples > b:
            num_samples = b
        samples_idx = random.sample(list(range(b)), num_samples)
        samples = input_[samples_idx, :, :, :]
        samples = samples.view(num_samples*self.num_classes, self.num_maps, h*w)
        b_, _, _ = samples.size()
        loss_sum = 0 
        for i in range(b_):
            feat = samples[i]
            dist_mat = pairwise_distance(feat)
            maxout = torch.clamp(self.margin - dist_mat, min=0)
            loss_sum = loss_sum + torch.sum(maxout)
        return loss_sum / b




