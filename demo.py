import argparse

import torch
import torch.nn as nn

from experiment.engine import MultiLabelMAPEngine
from experiment.create import create_model, create_dataset 

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=1, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--maps', default=1, type=int,
                    metavar='N', help='number of maps per class (default: 1)')
parser.add_argument('-s', '--save', default='./expes/models/', type=str, metavar='DIR',
                    help='path to save checkpoints (e.g. ../expes/')
parser.add_argument('-d', '--dataname', default='coco', type=str,
                    help='dataset name (e.g. coco, nus)')
parser.add_argument('-m', '--model', default='ours_50', type=str,
                    help='model name (e.g. ours_50, ours_101, baseline_50, wildcat_50)')
parser.add_argument('--threshold', default=0.5, type=float, 
                    help='threshold of predicted labels, if larger than threshold then set 1')

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = create_dataset(args.dataname, args.data, 'train')
    val_dataset = create_dataset(args.dataname, args.data, 'test')
    num_classes = train_dataset.get_number_classes()

    # load model
    model = create_model(args.model, num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)

    # define loss function (criterion)
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'threshold': args.threshold}
    # state['difficult_examples'] = True
    state['save_model_path'] = args.save
    
    engine = MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main()
