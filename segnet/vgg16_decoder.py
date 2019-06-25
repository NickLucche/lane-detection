import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGDecoder(nn.Module):
    vgg_16_config = [64, 128, 256, 512, 512]

    def __init__(self, output_channels, verbose=False):
        """
        Instantiate the decoder needed to build the segnet model.
        :param output_channels: number of channels the output needs
                to have; this is referred as K, and it needs to be
                the same as the number of classes we need to recognize
                in a certain task.
        :param verbose: 
        """
        super(VGGDecoder, self).__init__()
        # self.decoder = utils.VGG16utils.make_decoder_layers(batch_norm=batch_norm)
        self.decoder = nn.ModuleList()
        vgg16_decoder_blocks = (3, 3, 3, 2, 2)
        config = VGGDecoder.vgg_16_config[::-1] + [output_channels]

        for i, n_blocks in enumerate(vgg16_decoder_blocks):
            self.decoder.append(Decoder_block(n_blocks, config[i], config[i+1]))


        self.verbose = verbose
        if verbose:
            print(self.decoder)

    def forward(self, input:torch.Tensor, indices:list, output_size:list):
        x = input
        # revert list so that you get the correct indices
        indices = indices[::-1]
        output_size = output_size[::-1]
        if self.verbose:
            print([p.numpy().shape for p in indices])

        for i, decoder in enumerate(self.decoder):
            x = decoder(x, indices[i], output_size[i])

        return x

class Decoder_block(nn.Module):
    # each decoder block starts with a max unpool operation
    def __init__(self, n_blocks, in_channels, out_channels):
        super(Decoder_block, self).__init__()
        layers = []
        block = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
                  nn.BatchNorm2d(in_channels),
                  nn.ReLU(inplace=True)]

        for i in range(n_blocks-1):
            layers += block

        # last layer has to convolve to the dimension of the next block
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]

        self.decoder_block = nn.Sequential(*layers)


    def forward(self, x:torch.Tensor, unpool_indices:list, unpool_size:list):
        x = F.max_unpool2d(x, unpool_indices, kernel_size=2, stride=2, padding=0, output_size=unpool_size)
        return self.decoder_block(x)


if __name__ == "__main__":
    decoder = VGGDecoder(3, verbose=True)
    #indeces must be tensor same shape as input all 0s but the max elem

