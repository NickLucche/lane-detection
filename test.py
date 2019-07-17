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

    # evaluate mode highly decreases performance
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
            # print(loss.item())
            f, (p, r) = f1_score(output, targets.float())
            f1.update(f)
            prec.update(p)
            rec.update(r)

            # slightly smooth output
            output = torch.softmax(output, dim=1)
            pred = output.max(1, keepdim=True)[1].float()
            img = torch.squeeze(pred).cpu().numpy() * 255
            lab = torch.squeeze(targets).cpu().numpy() * 255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))
            # accuracy
            # pred = output.max(1, keepdim=True)[1]
            targets = targets.float()
            correct += pred.eq(targets.view_as(pred)).float().sum().item()

            # precision,recall,f1
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img * label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b == 0:
                error = error + 1
                continue
            else:
                precision += float(a / b)
            c = len(np.nonzero(pred_recall * lab)[1])
            d = len(np.nonzero(lab)[1])
            if d == 0:
                error = error + 1
                continue
            else:
                recall += float(c / d)
            F1_measure = (2 * precision * recall) / (precision + recall)
            print(F1_measure)
            # store various accuracy measures
            acc.update(pred.eq(targets.view_as(pred)).float().mean().item(), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_no % log_every == 0:
                progress.display(batch_no)

        test_acc = 100. * int(correct) / (len(val_loader.dataset) * 128 * 256)
        print('\nAccuracy: {}/{} ({:.5f}%)'.format(
            int(correct), len(val_loader.dataset), test_acc))

        precision = precision / (len(val_loader.dataset) - error)
        recall = recall / (len(val_loader.dataset) - error)
        F1_measure = F1_measure / (len(val_loader.dataset) - error)
        print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision, recall, F1_measure))

        # currently returning loss instead of accuracy
        return losses.avg


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