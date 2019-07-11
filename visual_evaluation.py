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

if __name__ == '__main__':

    cc = Configs()
    print("Loading stored model")
    model = SegnetConvLSTM(cc.hidden_dims, decoder_out_channels=1, lstm_nlayers=3, vgg_decoder_config=cc.decoder_config)
    tu.load_model_checkpoint(model, 'model-epoch-4.pt', inference=True, map_location=device)
    # print("Model with {} parameters correctly loaded", len(model.parameters()))
    print("Model loaded")
    tu_test_dataset = TUSimpleDataset(config.tr_root, config.tr_subdirs, config.tr_flabels)
    tu_dataloader = DataLoader(tu_test_dataset, batch_size=2, shuffle=False)

    with torch.no_grad():

        for i, (frames, targets) in enumerate(tu_dataloader):
            output = model(frames)
            print("Loss", nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([17.]))(output, targets))
            output = (torch.sigmoid(output) > .5)
            print(targets.sum().item(), output.sum().item())
            # print inputs and target
            for i in range(2):
                samples = []
                for j in range(len(frames)):
                    a, b = torch.chunk(frames[j], 2, dim=0)
                    if i == 0:
                        samples.append(a.squeeze())
                    else:
                        samples.append(b.squeeze())
                    # print("samples info")
                    # print(a.squeeze().size())
                    # print(torch.max(a), torch.mean(a), torch.sum(a))
                # print("Single target shape:", batched_target[i].size())
                show_plain_images(samples + [targets[i]] + [output[i]], len(samples) + 2)
