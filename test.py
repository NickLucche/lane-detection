import torch
from segnet import vgg_16_encoder, vgg16_decoder
from lstm import convlstm
from utils import train_utils as tu
from segnet_conv_lstm_model import SegnetConvLSTM
from utils.config import Configs

if __name__ == '__main__':

    model = SegnetConvLSTM(hidden_dims, decoder_out_channels=1, lstm_nlayers=3, vgg_decoder_config=decoder_config)
    model = tu.load_model_checkpoint()

    with torch.no_grad():
        # seq of images
        enc_image_seq = torch.zeros((3, 512, 4, 8))
        encoder = vgg_16_encoder.VGGencoder(True)
        encoder.eval()
        # single image
        for i in range(len(enc_image_seq)):
            # use zeros to save computations
            image = torch.zeros((1, 3, 128, 256))
            print("Input image", image.size())
            encoded_out, pool_indices, output_sizes = encoder(image)
            print("Encoded image", encoded_out.size())
            enc_image_seq[i] = encoded_out

        print("Encoded images sequence:", enc_image_seq.unsqueeze(0).size())
        enc_image_seq = enc_image_seq.unsqueeze(0)
        convlstm = convlstm.ConvLSTM(input_size=(4, 8), input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=4, batch_first=True)
        _, last_state = convlstm(enc_image_seq)
        print("LSTM Output {} ".format(last_state[0][0].size()))    # get LATEST lstm hidden state as a repr summary


        # merge info of timeseries using 1x1 Convolution
        # nn.Conv2d(3, )

        #
        decoder = vgg16_decoder.VGGDecoder(2)
        decoder.eval()
        reconstr = decoder(last_state[0][0], pool_indices, output_sizes)
        print("Final Result:", reconstr.size())