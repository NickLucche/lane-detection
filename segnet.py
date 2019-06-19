import vgg_16_encoder as encoder
import vgg16_decoder as decoder
import torch
import torch.nn as nn

class Segnet(nn.Module):

    def __init__(self, verbose=False):
        super(Segnet, self).__init__()
        self.verbose = verbose

        self.encoder = encoder.VGGencoder(segnet_=verbose, verbose=verbose)
        self.decoder = decoder.VGGDecoder(3, verbose=verbose)


    def forward(self, x):
        code, pool_indices, output_sizes = self.encoder(x)
        # print([p.numpy().shape for p in pool_indices])
        output = self.decoder(code, pool_indices, output_sizes)
        return output


if __name__ == '__main__':

    segnet = Segnet(verbose=True)
    with torch.no_grad():
        print(segnet(torch.randn(10, 3, 128, 256)).numpy().shape)
    # flush gradient buffer in training