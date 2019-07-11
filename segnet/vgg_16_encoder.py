import torch
import torch.nn as nn
import torchvision.models as models


class VGGencoder(nn.Module):

    def __init__(self, segnet_:bool=True, pretrained=True, verbose=False):
        super(VGGencoder, self).__init__()
        self.verbose = verbose
        # download pre-trained model with bn (batch normalization)
        # super(VGGencoder, self).__init__()
        self.segnet_ = segnet_
        self.vgg16 = models.vgg16_bn(pretrained=pretrained)
        # segnet implementation of vgg for efficiency,
        # max pool layers must return the indices needed for unpooling
        if segnet_:
            print("Instantiating Segnet encoder with max unpooling")
            params = self.vgg16.features.state_dict()
            # print(next(self.vgg16.parameters()))
            layers = []
            for layer in self.vgg16.features.children():
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
            print("Instantiating simple Segnet encoder (VGG16)")
            self.encoder = nn.Sequential(*self.vgg16.features)
            raise NotImplementedError()
            # print(self.encoder)

    def forward(self, input:torch.Tensor):
        # self.vgg16(input)
        x = input
        indices = []
        output_sizes = []
        if self.segnet_:
            for layer in self.encoder.children():
                # print(x.shape)
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

    # remember model.eval() or train() and move to cuda()