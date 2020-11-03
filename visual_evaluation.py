import torch
import torch.nn as nn
from segnet import vgg_16_encoder, vgg16_decoder
from lstm import convlstm
from utils import train_utils as tu
from segnet_conv_lstm_model import SegnetConvLSTM
import utils.config as config
from utils.config import Configs
from utils.cuda_device import device
from utils.data_utils import TUSimpleDataset
from utils.data_utils import DataLoader
from utils.data_utils import show_plain_images
import cv2
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

""" 
    Asses model output quality visually, by plotting
    inputs-target-prediction.
"""
if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-m", '--model-path', required=True, type=str, help='Pre-trained model filepath')
    args = args.parse_args()
    # load model
    cc = Configs()
    print("Loading stored model")
    model = SegnetConvLSTM(cc.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(cc.hidden_dims), vgg_decoder_config=cc.decoder_config)
    model = model.to(device)
    tu.load_model_checkpoint(model, args.model_path, inference=False, map_location=device)
    print("Model loaded")
    # create dataloader
    tu_test_dataset = TUSimpleDataset(config.ts_root, config.ts_subdirs, config.ts_flabels)
    tu_dataloader = DataLoader(tu_test_dataset, batch_size=2, shuffle=True)
    model.train()
    with torch.no_grad():

        for batchno, (frames, targets) in enumerate(tu_dataloader):
            frames = [f.to(device) for f in frames]

            output = model(frames)
            targets_ = targets.squeeze(1).long().to(device)

            print("Loss:", nn.CrossEntropyLoss(weight=torch.FloatTensor(cc.loss_weights).to(device))(output, targets_))
            output = (torch.sigmoid(output[:, 1, :, :]) > .5).float()
            print("Output max:", output.max().item(), "Output mean", output.mean().item())
            print("Pixel lane points:", targets.sum().item(), output.sum().item())
            # print inputs and target
            for i in range(2):
                samples = []
                for j in range(len(frames)):
                    a, b = torch.chunk(frames[j], 2, dim=0)
                    if i == 0:
                        samples.append(a.squeeze())
                    else:
                        samples.append(b.squeeze())
                # print("Single target shape:", targets[i].size())
                # print("Single output shape:", output[i][1, :, :].unsqueeze(0).size())
                # o = cv2.addWeighted(samples[-1].permute(1, 2, 0).numpy(), 1., targets[i].permute(1, 2, 0).numpy().repeat(3, axis=2), 1, 0)
                o = samples[0].permute(1, 2, 0).cpu().numpy() + targets[i].permute(1, 2, 0).cpu().numpy()
                # plt.imshow(o.astype(np.float32))
                # plt.show()
                show_plain_images(samples + [targets[i]] + [output[i].unsqueeze(0)], len(samples) + 2, save=False,
                                  fname=f'visual_{batchno}-{i}.png')
