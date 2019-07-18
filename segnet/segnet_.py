from segnet import vgg_16_encoder as encoder, vgg16_decoder as decoder
import torch
import torch.nn as nn


class Segnet(nn.Module):
    """
        Implementation of simple Segnet
        network comes 'for free' since we
        already implemented encoder and
        decoder.
    """
    def __init__(self, decoder_out_channels, verbose=False):
        super(Segnet, self).__init__()
        self.verbose = verbose

        self.encoder = encoder.VGGencoder(segnet_=True, verbose=verbose)
        self.decoder = decoder.VGGDecoder(decoder_out_channels, verbose=verbose)


    def forward(self, x):
        code, pool_indices, output_sizes = self.encoder(x)
        # print([p.numpy().shape for p in pool_indices])
        output = self.decoder(code, pool_indices, output_sizes)
        return output


if __name__ == '__main__':

    segnet = Segnet(verbose=True)
    with torch.no_grad():
        print(segnet(torch.randn(10, 3, 128, 256)).numpy().shape)