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

def train(train_loader:DataLoader, model:SegnetConvLSTM, criterion, optimizer, epoch, log_every=1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    f1 = AverageMeter('F1', ':6.2f')
    prec = AverageMeter('Prec', ':6.2f')
    rec = AverageMeter('Recall', ':6.2f')
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
        print("Train loss value:",loss.item())
        # record loss, dividing by sample size
        losses.update(loss.item(), batched_targets.size(0))

        batched_targets = batched_targets.float()
        acc.update(pixel_accuracy(output, batched_targets), batched_targets.size(0))
        f, (p, r) = f1_score(output, batched_targets)
        f1.update(f, batched_targets.size(0))
        prec.update(p, batched_targets.size(0))
        rec.update(r, batched_targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_no % log_every == 0:
            progress.display(batch_no)

    return losses.avg


def validate(val_loader, model, criterion, log_every=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    f1 = AverageMeter('F1', ':6.2f')
    prec = AverageMeter('Prec', ':6.2f')
    rec = AverageMeter('Recall', ':6.2f')
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
            f1.update(f, batched_targets.size(0))
            prec.update(p, batched_targets.size(0))
            rec.update(r, batched_targets.size(0))


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
epochs = 30
init_lr = 0.01
batch_size = 5
workers = 4
momentum = 0.9
weight_decay = 0.0001


data_transform = transforms.Compose([
    transforms.Resize((128, 256)),
    # transforms.RandomHorizontalFlip(), #need to apply flip to all samples and target too
    transforms.ToTensor(),
])
tu_tr_dataset = TUSimpleDataset(config.tr_root, config.tr_subdirs, config.tr_flabels, transforms=data_transform, shuffle_seed=9)
tu_test_dataset = TUSimpleDataset(config.ts_root, config.ts_subdirs, config.ts_flabels, transforms=data_transform, shuffle_seed=9)

# build data loader
tu_train_dataloader = DataLoader(tu_tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=12, shuffle=True, num_workers=4)

# model, output size must have dimension (B, C..), where C = number of classes
model = SegnetConvLSTM(decoder_out_channels=1)
model.to(device)

# define loss function (criterion) and optimizer
# loss function is a binary crossentropy evaluated pixel-wise
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([.4, 1.])).to(device) # using crossentropy for weighted loss
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3.])).to(device)
# todo try BCEWithLogits and pos_weight loss, changing output channels to 1

# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), init_lr)

optimizer.zero_grad()
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, init_lr)

    # do one train step
    loss_val = train(tu_train_dataloader, model, criterion, optimizer, epoch)

    # evaluate model performance
    loss_eval_val = validate(tu_test_dataloader, model, criterion)

    trainu.save_model_checkpoint(model, 'model.pt', epoch=epoch, tr_loss=loss_val, ev_loss=loss_eval_val)