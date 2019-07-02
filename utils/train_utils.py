import torch

# classes from https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L354
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):  #toString()
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_model_checkpoint(model:torch.nn.Module, filename:str, optimizer:torch.optim=None,epoch=None,
                          tr_loss=None, ev_loss=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'tr_loss': tr_loss,
        'ev_loss': ev_loss
    }, filename)

def load_model_checkpoint(model:torch.nn.Module, filename:str, inference:bool,
                          optimizer:torch.optim=None):
    """
    Load a model checkpoint
    :param model:
    :param filename:
    :param inference:
    :param optimizer:
    :return:
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    if inference:
        model.eval()
    else:
        model.train()
    return checkpoint


def loss_weight_balance(labels:list):
    """
    Compute the ratio between pixels belonging
    to the positive class (lanes) and all the
    rest, due to the high unbalance in terms of
    frequency in these two classes.
    Result will be used to weight the computation
    of the cross-entropy loss function.
    :param labels: list of annotated data following
        the tusimple format.
    :return:
    """

    pos_pixels = 0
    neg_pixels = 0
    img_size = 1280*720/10
    for l in labels:
        for lane in l['lanes']:
            pos_pixels += len([lval for lval in lane if lval>=0])
            neg_pixels = img_size - pos_pixels