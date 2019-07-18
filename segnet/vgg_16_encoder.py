import torch
import torch.nn as nn
import torchvision.models as models


class VGGencoder(nn.Module):
    """
        This class implements a CNN encoder, more
        specifically the encoder used by the fully
        convolutional Segnet network for
        semantic segmentation.
        The encoder is obtained from the VGG16 architecture,
        by removing the fully-connected part.
        Max pool is performed storing indices, but
        other implementation will be added of this encoder
        (e.g. UNet, minor modification of VGG16,
        storing feature maps before max pool operations).
    """
    def __init__(self, segnet_:bool=True, pretrained=True, verbose=False):
        """
        :param segnet_: Whether to use the vgg16 encoder for the segnet;
                    this is currently the only supported implementation,
                    but UNet version of the encoder will soon be added.
        :param pretrained: use ImageNet pre-trained weights for vgg16.
        """
        super(VGGencoder, self).__init__()
        self.verbose = verbose
        # download pre-trained model with bn (batch normalization)
        # super(VGGencoder, self).__init__()
        self.segnet_ = segnet_
        vgg16 = models.vgg16_bn(pretrained=pretrained)
        # segnet implementation of vgg for efficiency,
        # max pool layers must return the indices needed for unpooling
        if segnet_:
            print("Instantiating Segnet encoder with max unpooling")
            params = vgg16.features.state_dict()
            layers = []
            for layer in vgg16.features.children():
                if isinstance(layer, nn.MaxPool2d):
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2,
                                               padding=0, dilation=1, ceil_mode=False, return_indices=True))
                else:
                    layers.append(layer)

            self.encoder = nn.Sequential(*layers)
            # load old params
            self.encoder.load_state_dict(params)
            if verbose:
                print(self.encoder)
        else:
            print("Instantiating UNet encoder")
            raise NotImplementedError()

    def forward(self, input:torch.Tensor):
        x = input
        indices = []
        output_sizes = []
        if self.segnet_:
            for layer in self.encoder.children():
                if isinstance(layer, nn.MaxPool2d):
                    output_sizes.append(x.size())
                    x, index = layer(x)
                    if self.verbose:
                        print(x.shape, index.shape)
                    indices.append(index)
                else:
                    x = layer(x)
        else:
            x = self.encoder(input)

        return x, indices, output_sizes


if __name__ == '__main__':
    encoder = VGGencoder(segnet_=True, verbose=True)
    with torch.no_grad():
        print(encoder(torch.randn((1, 3, 128, 256)))[0].numpy().shape)
