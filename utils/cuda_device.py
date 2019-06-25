import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
def set_gpu_number(n_gpu):
    global device
    device = torch.device("cuda:{}".format(n_gpu)) if torch.cuda.is_available() else torch.device("cpu")