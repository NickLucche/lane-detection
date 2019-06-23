import torch

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