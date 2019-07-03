from utils.train_utils import AverageMeter, ProgressMeter
import time
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import TUSimpleDataset
from utils.data_utils import DataLoader
from utils.cuda_device import device
import utils.train_utils as trainu
from torchvision import transforms
from segnet_conv_lstm_model import SegnetConvLSTM
from utils import config
import json


def train(train_loader:DataLoader, model:SegnetConvLSTM, criterion, optimizer, epoch, log_every=1):
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
        batched_targets = batched_targets.to(device)

        # compute output
        output = model(list_batched_samples)
        # print("Output size:", output.size(), "Target size:", batched_targets.size())

        # loss executes Sigmoid inside (efficiently)
        loss = criterion(output, batched_targets)
        # print("Train loss value:",loss.item())
        # record loss, dividing by sample size
        losses.update(loss.item(), batched_targets.size(0))

        batched_targets = batched_targets.float()
        acc.update(pixel_accuracy(output, batched_targets), batched_targets.size(0))
        f, (p, r) = f1_score(output, batched_targets)

        f1.update(f)
        prec.update(p)
        rec.update(r)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_no % log_every == 0:
            print("Output sum:", output.sum().item(), "Output max:", torch.max(output).item())
            print("Targets sum:", batched_targets.sum(), "Targets max:", torch.max(batched_targets))
            print("Base acc:{} - base prec: {}- base recall: {}- base f1: {}".
                  format(pixel_accuracy(output, batched_targets), p, r, f))
            progress.display(batch_no)

        torch.cuda.empty_cache()
    return losses.avg


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

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_no, (list_batched_samples, batched_targets) in enumerate(val_loader):
            # move data to gpu (or cpu if device is unavailable)
            list_batched_samples = [t.to(device) for t in list_batched_samples]
            batched_targets = batched_targets.to(device)

            # compute output
            output = model(list_batched_samples)

            loss = criterion(output, batched_targets)

            losses.update(loss.item(), batched_targets.size(0))

            # store various accuracy measures
            acc.update(pixel_accuracy(output, batched_targets), batched_targets.size(0))
            f, (p, r) = f1_score(output, batched_targets)

            f1.update(f)
            prec.update(p)
            rec.update(r)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_no % log_every == 0:
                progress.display(batch_no)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

        # currently returning loss instead of accuracy
        return losses.avg


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
    output = (prediction > 0.5).float()
    return (output == target).float().mean()

def f1_score(output, target, epsilon=1e-7):
    # sigmoid is executed inside loss
    probas = torch.sigmoid(output)
    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean(), (precision.mean(), recall.mean())

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# hyperparameters
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
tu_test_dataset = TUSimpleDataset(config.ts_root, config.ts_subdirs, config.ts_flabels, shuffle=False)#, shuffle_seed=9)

# build data loader
tu_train_dataloader = DataLoader(tu_tr_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=cc.test_batch, shuffle=False, num_workers=4)


# **MODEL**
# output size must have dimension (B, C..), where C = number of classes
model = SegnetConvLSTM(hidden_dims, decoder_out_channels=1, lstm_nlayers=3, vgg_decoder_config=decoder_config)
model.to(device)

# define loss function (criterion) and optimizer
# loss function is a binary crossentropy evaluated pixel-wise
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([.4, 1.])).to(device) # using crossentropy for weighted loss
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([8.])).to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), init_lr, weight_decay=weight_decay)

losses_ = []
test_losses = []
optimizer.zero_grad()
for epoch in range(epochs):
    #adjust_learning_rate(optimizer, epoch, init_lr)

    # do one train step
    loss_val = train(tu_train_dataloader, model, criterion, optimizer, epoch, log_every=16)
    losses_.append(loss_val)
    # evaluate model performance
    #loss_eval_val = validate(tu_test_dataloader, model, criterion, log_every=24)
    #test_losses.append(loss_eval_val)
    if epoch % 2 == 0:
        trainu.save_model_checkpoint(model, 'model-epoch-{}.pt'.format(epoch), epoch=epoch, tr_loss=loss_val,
                                     ev_loss=1.)

print("Saving loss values to json..")
with open('tr-losses.json', 'w') as f, open('test-losses.json', 'w') as ff:
    json.dump(losses_, f)
    json.dump(test_losses, ff)
