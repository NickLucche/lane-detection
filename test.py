import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segnet_conv_lstm_model import SegnetConvLSTM
from utils import config
from utils.data_utils import TUSimpleDataset
from utils.train_utils import AverageMeter, ProgressMeter
from utils.cuda_device import device
from utils.config import Configs
import utils.train_utils as tu

import numpy as np
import cv2
import argparse
import time

"""
    This file is used to assess model results on
    test set on measure like accuracy, precision, 
    recall, f1 score and inference time.
"""


def validate(val_loader, model, criterion, log_every=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.4f')
    f1 = AverageMeter('F1', ':6.4f')
    prec = AverageMeter('Prec', ':6.4f')
    rec = AverageMeter('Recall', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, f1, prec, rec],
        prefix='Test: ')

    # model.eval() evaluate mode highly decreases performance
    model.train()

    correct = 0
    error = 0
    precision = 0.
    recall = 0.
    with torch.no_grad():
        end = time.time()
        for batch_no, (samples, targets) in enumerate(val_loader):
            # move data to gpu (or cpu if device is unavailable)
            samples = [t.to(device) for t in samples]
            targets = targets.squeeze(1).long().to(device)

            # compute output
            output = model(samples)
            # compute loss
            loss = criterion(output, targets)
            losses.update(loss.item(), targets.size(0))
            # compute f1 score
            f, (p, r) = f1_score(output, targets.float())
            f1.update(f)
            prec.update(p)
            rec.update(r)
            # compute accuracy
            acc.update(pixel_accuracy(output, targets), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_no % log_every == 0:
                progress.display(batch_no)

        return acc.avg

def pixel_accuracy(prediction:torch.Tensor, target:torch.Tensor):
    """
        Computes simple pixel-wise accuracy measure
        between target lane and prediction map; this
        measure has little meaning (if not backed up
        by other metrics) in tasks like this where
        there's a huge unbalance between 2 classes
        (background and lanes pixels).
    """
    # get prediction positive channel (lanes)
    out = (prediction[:, 1, :, :] > 0.).float()
    return (out == target).float().mean().item()

def f1_score(output, target, epsilon=1e-7):
    # turn output into 0-1 map
    probas = (output[:, 1, :, :] > 0.).float()

    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean().item(), (precision.mean().item(), recall.mean().item())


cc = Configs()
print("Loading stored model")
model = SegnetConvLSTM(cc.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(cc.hidden_dims),
                       vgg_decoder_config=cc.decoder_config)
tu.load_model_checkpoint(model, '../train-results/model-fixed.torch', inference=False, map_location=device)
model.to(device)
print("Model loaded")

tu_test_dataset = TUSimpleDataset(config.ts_root, config.ts_subdirs, config.ts_flabels, shuffle=False)#, shuffle_seed=9)

# build data loader
tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=cc.test_batch, shuffle=True, num_workers=2)

# using crossentropy for weighted loss
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

validate(tu_test_dataloader, model, criterion, log_every=1)