import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device==torch.device("cuda"):
    print("Using device:", torch.cuda.get_device_name(0))
else:
    print("Using device:", device)

def set_gpu_number(n_gpu):
    global device
    device = torch.device("cuda:{}".format(n_gpu)) if torch.cuda.is_available() else torch.device("cpu")