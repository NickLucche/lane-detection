import torch
import torch.nn as nn
import torch.nn.functional as F
import segnet.vgg16_decoder as decoder
import segnet.vgg_16_encoder as encoder
from lstm import convlstm
from utils.data_utils import TUSimpleDataset
from utils.data_utils import DataLoader
from torchvision import transforms
from utils.cuda_device import device


class SegnetConvLSTM(nn.Module):
    """
        This class implements the whole end-to-end trainable
        model as described in the paper 'Robust Lane detection
        From Continuous Driving Scenes Using Deep Neural Networks'.
        The model comprises of a fully convolutional encoder-decoder
        (namely Segnet) and of a ConvLSTM, which 'fuses'
        information coming from multiple contiguous frames.
        Data structure is preserved throughout the network.
    """

    def __init__(self, lstm_hidden_dim:list, lstm_nlayers:int=2, decoder_out_channels:int=1,
                 vgg_decoder_config:list=None, verbose=False):
        """

        :param lstm_hidden_dim: list of hidden layers dimensions used to define the
            convlstm architecture (e.g [512, 512]).
        :param lstm_nlayers: number of hidden layers in lstm (== len(lstm_hidden_dim))
        :param decoder_out_channels: number of channels the decoder output will
                have; it is generally the same as the number of classes we need
                to recognize for a certain task.
        :param vgg_decoder_config: if provided, custom configuration passed to the decoder.
        """

        super(SegnetConvLSTM, self).__init__()
        assert lstm_nlayers == len(lstm_hidden_dim)

        self.n_classes = decoder_out_channels
        self.v = verbose

        # define encoder-decoder structure
        self.encoder = encoder.VGGencoder()
        self.decoder = decoder.VGGDecoder(decoder_out_channels, config=vgg_decoder_config)

        # define ConvLSTM block
        self.lstm = convlstm.ConvLSTM(input_size=(4, 8), input_dim=512, hidden_dim=lstm_hidden_dim,
                                      kernel_size=(3, 3), num_layers=lstm_nlayers, batch_first=True)

    def forward(self, x:list):
        """
        Forward step of the model. We're expecting the
        input to be in the format of a list of
        batched samples, 1 for each timestep/frame,
        which is then stacked to form a single input to
        feed to the lstm. Batches are assumed to be of equal constant size.

        :param x: list of batched samples; len(x)=n_frames=seq_len.
        :return: output of the model, a probability map telling
                whether pixel (i, j) is a lane or not.
        """

        # T x B x C x W x H
        # batched_code_sequence = torch.tensor((len(x), len(x[0]), 512, 4, 8))
        y = []
        # todo can be executed in parallel on multiple GPUs
        for i, batched_samples in enumerate(x):
            # only keep indices of last batch of samples,
            # the frame for which we have ground truth
            encoded, unpool_indices, unpool_sizes = self.encoder(batched_samples)
            if self.v: print("Encoded size:", encoded.size())
            # batched_code_sequence[i] = encoded
            y.append(encoded)
        batched_code_sequence = torch.stack(y, dim=1)

        # now feed the batched output of the encoder to the lstm
        # (prefer batch_first format)
        # batched_code_sequence.permute(1, 0, 2, 3, 4)
        if self.v: print("Batched sequence of codes size:", batched_code_sequence.size())

        # keep last LSMT output
        # _, last_state = self.lstm(batched_code_sequence)
        output, _ = self.lstm(batched_code_sequence)
        output = output[0][:, -1, :, :, :]  # batch size must be first!
        # last_state = last_state[0][0]  same as above, may be less clear
        if self.v: print("LSTM output size:", output.size())

        # now decode the hidden representation
        decoded = self.decoder(output, unpool_indices, unpool_sizes)

        # return a probability map of the same size of each frame input to the model
        return decoded  # (NOTE: softmax is applied inside loss for efficiency)


# this won't work if not run in parent directory
if __name__ == '__main__':
    root = '/Users/nick/Desktop/train_set/clips/'
    subdirs = ['0601', '0531', '0313-1', '0313-2']
    flabels = ['/Users/nick/Desktop/train_set/label_data_0601.json',
               '/Users/nick/Desktop/train_set/label_data_0531.json',
               '/Users/nick/Desktop/train_set/label_data_0313.json']

    tu_dataset = TUSimpleDataset(root, subdirs, flabels, shuffle_seed=9)

    # build data loader
    tu_dataloader = DataLoader(tu_dataset, batch_size=3, shuffle=True, num_workers=2)
    model = SegnetConvLSTM()
    for batch_no, (list_batched_samples, batched_targets) in enumerate(tu_dataloader):
        with torch.no_grad():
            out = model(list_batched_samples)
            print(out.size())
        if batch_no == 1:
            break