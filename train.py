from utils.train_utils import AverageMeter, ProgressMeter
import time
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import TUSimpleDataset
from utils.data_utils import DataLoader
from utils.cuda_device import device
import utils.train_utils as trainu
from segnet_conv_lstm_model import SegnetConvLSTM
from utils import config
import json
import cv2


def train(train_loader:DataLoader, model:SegnetConvLSTM, criterion, optimizer, epoch, log_every=1):
    """
    Do a training step, iterating over all batched samples
    as returned by the DataLoader passed as argument.
    Various measurements are taken and returned, such as
    accuracy, loss, precision, recall, f1 and batch time.
    """

    batch_time = AverageMeter('BatchTime', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.4f')
    f1 = AverageMeter('F1', ':6.4f')
    prec = AverageMeter('Prec', ':6.4f')
    rec = AverageMeter('Recall', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc, f1, prec, rec],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_no, (list_batched_samples, batched_targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to gpu (or cpu if device is unavailable)
        list_batched_samples = [t.to(device) for t in list_batched_samples]
        batched_targets = batched_targets.long().to(device)
        # squeeze target channels to compute loss
        batched_targets = batched_targets.squeeze(1)

        # compute output
        output = model(list_batched_samples)
        # print("Output size:", output.size(), "Target size:", batched_targets.size())

        # loss executes Sigmoid inside (efficiently)
        loss = criterion(output, batched_targets)
        # print("Train loss value:",loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach output to compute metrics without storing computational graph
        output = output.detach()
        # record loss, dividing by sample size
        losses.update(loss.item(), batched_targets.size(0))

        batched_targets = batched_targets.float()
        accuracy = pixel_accuracy(output, batched_targets)
        acc.update(accuracy, batched_targets.size(0))
        f, (p, r) = f1_score(output, batched_targets)

        f1.update(f)
        prec.update(p)
        rec.update(r)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_no % log_every == 0:
            print("Output min", output.min().item(), "Output (softmax-ed) sum:", (output > 0.).float().sum().item(), "Output max:", torch.max(output).item())
            print("Targets sum:", batched_targets.sum())#, "Targets max:", torch.max(batched_targets))
            print("Base acc:{} - base prec: {}- base recall: {}- base f1: {}".
                  format(pixel_accuracy(output, batched_targets), p, r, f))
            progress.display(batch_no)

        # torch.cuda.empty_cache()
    return losses.avg, acc.avg, f1.avg


def IoU_accuracy(prediction, target):
    """
        Computes the intersection over union accuracy; this measure
        if often used in semantic segmentation tasks as well as
        object recognition tasks.
        Predictions whose IoUs are larger than certain threshold
        are viewed as true positives (TP).
    """
    # todo add 'border' to pixels
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)

    return np.sum(intersection) / np.sum(union)


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
    # output has to be passed though sigmoid and then thresholded
    # this way we directly threshold it efficiently
    probas = (output[:, 1, :, :] > 0.).float()

    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean().item(), (precision.mean().item(), recall.mean().item())

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# hyperparameters configured in utils/config.py
cc = config.Configs()
epochs = cc.epochs
init_lr = cc.init_lr
batch_size = cc.batch_size
workers = cc.workers
momentum = cc.momentum
weight_decay = cc.weight_decay
hidden_dims = cc.hidden_dims
decoder_config = cc.decoder_config

# **DATA**

tu_tr_dataset = TUSimpleDataset(config.tr_root, config.tr_subdirs, config.tr_flabels, shuffle=False)#, shuffle_seed=9)
# tu_test_dataset = TUSimpleDataset(config.ts_root, config.ts_subdirs, config.ts_flabels, shuffle=False)#, shuffle_seed=9)

# build data loader
tu_train_dataloader = DataLoader(tu_tr_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
# tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=cc.test_batch, shuffle=False, num_workers=4)


# **MODEL**
# output size must have dimension (B, C..), where C = number of classes
model = SegnetConvLSTM(hidden_dims, decoder_out_channels=2, lstm_nlayers=len(hidden_dims), vgg_decoder_config=decoder_config)
if cc.load_model:
    trainu.load_model_checkpoint(model, '../train-results/model.torch', inference=False, map_location=device)

model.to(device)

# define loss function (criterion) and optimizer
# using crossentropy for weighted loss on background and lane classes
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([17.])).to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), init_lr)#, weight_decay=weight_decay)

losses_ = []
accs = []
f1s = []
test_losses = []
for epoch in range(epochs):
    # adjust_learning_rate(optimizer, epoch, init_lr)

    # do one train step
    loss_val, a, f = train(tu_train_dataloader, model, criterion, optimizer, epoch, log_every=16)
    losses_.append(loss_val)
    accs.append(a)
    f1s.append(f)

    # did not evaluate model performance on eval set during training since
    # it's already expensive enough as it is

    # save model at each epoch (overwrite model because of memory usage)
    trainu.save_model_checkpoint(model, 'model.torch', epoch=epoch)

print("Saving loss values to json..")
with open('tr-losses.json', 'w') as f, open('tr-acc.json', 'w') as ff, open('tr-f1score.json', 'w') as fff:
    json.dump(losses_, f)
    json.dump(accs, ff)
    json.dump(f1s, fff)
