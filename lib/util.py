import math
from urllib.request import urlretrieve

import torch
from PIL import Image
from tqdm import tqdm

import numpy as np 
from sklearn.metrics import precision_recall_fscore_support

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, threshold=0.5):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.threshold = threshold

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            # scores[scores > 0.5] = 1
            # scores[scores <= 0.5] = 0
            targets = self.targets[:, k]

            # compute average precision
            # ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
        
        return ap
    
    def metrics(self):
        """Returns the model's P-C, R-C, F1-C, P-O, R-O, F1-O
        Return value
        """
        PC, RC, FC, PO, RO, FO = AveragePrecisionMeter.precision_recall_f1(self.scores, self.targets, threshold=self.threshold)

        return 100*PC, 100*RC, 100*FC, 100*PO, 100*RO, 100*FO
    # @staticmethod
    # def average_precision(output, target, difficult_examples=True):

    #     # sort examples
    #     sorted, indices = torch.sort(output, dim=0, descending=True)

    #     # Computes prec@i
    #     pos_count = 0.
    #     total_count = 0.
    #     precision_at_i = 0.
    #     for i in indices:
    #         label = target[i]
    #         pred = output[i]
    #         if difficult_examples and label == 0:
    #             continue
    #         if label == 1:
    #             pos_count += 1
    #         total_count += 1
    #         if label == 1:
    #             precision_at_i += pos_count / total_count
    #     precision_at_i /= pos_count
    #     return precision_at_i
    @staticmethod
    def average_precision(output, target):

        sorted_pred, indices = torch.sort(output, dim=0, descending=True)
        sorted_pred = sorted_pred.numpy()
        sorted_label = target[indices].numpy() 

        tp = (sorted_label == 1)
        fp = (sorted_label != 1)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        
        npos = np.sum(sorted_label)
        
        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp*1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.])) 
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap_at_k = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap_at_k

    @staticmethod 
    def precision_recall_f1(output, target, threshold=0.5):
        y_true = target.numpy()
        y_pred = output.numpy()

        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0

        pre_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average='macro')

        pre_o, rec_o, f1_o, _ = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average='micro')

        return pre_c, rec_c, f1_c, pre_o, rec_o, f1_o

