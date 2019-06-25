from utils.train_utils import AverageMeter, ProgressMeter
import time
import torch
import torch.nn as nn
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
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
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

        # compute output
        output = model(list_batched_samples)
        assert output.max().item() < 1. and output.min().item()>0.
        assert batched_targets.max().item() <= 1. and batched_targets.min().item()>=0.
        # todo
        #for each lane marking whoseexistence value is larger than 0.5, we search the correspond-ing
        #probmap every 20 rows for the position with the high-est response.These positions are then connected by cubicsplines,
        # which are the final predictions
        # target must not have channel/class dimension so we squeeze it
        loss = criterion(output, batched_targets.squeeze(1))
        print(loss.item())
        # record loss, dividing by sample size
        losses.update(loss.item(), batched_targets.size(0))
        # todo compute accuracy (check f1score and other measures on paper)

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
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_no, (list_batched_samples, batched_targets) in enumerate(val_loader):
            # move data to gpu (or cpu if device is unavailable)
            list_batched_samples = [t.to(device) for t in list_batched_samples]
            batched_targets = batched_targets.long().to(device)

            # compute output
            output = model(list_batched_samples)

            # target must not have channel/class dimension so we squeeze it
            loss = criterion(output, batched_targets.squeeze(1))

            # todo measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), batched_targets.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
model = SegnetConvLSTM(decoder_out_channels=2)
model.to(device)

# define loss function (criterion) and optimizer
# loss function is a binary crossentropy evaluated pixel-wise
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([.4, 1.])).to(device) # using crossentropy for weighted loss
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

    trainu.save_model_checkpoint(model, 'model.ptr', epoch=epoch, tr_loss=loss_val, ev_loss=loss_eval_val)