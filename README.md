# Implementation of a SegnetConvLSTM for Lane Detection
#### Please refer to the publication [Robust Lane Detection from Continuous DrivingScenes Using Deep Neural Networks](https://arxiv.org/pdf/1903.02193.pdf) by Qin Zou, Hanwen Jiang, Qiyu Dai, Yuanhao Yue, Long Chen, and Qian Wang for further information.

 This repo contains my PyTorch implementation of the afore mentioned paper, a model having the following architecture:
 
 ![architecture](https://i.imgur.com/oW5ouzb.png)
 
 _*UNetConvLSTM implementation will (hopefully) follow soon._
 
 The model comprises of a fully-convolutional encoder-decoder network, which takes as input multiple contiguous frames, in order to exploit *spatiotemporal* relations of data, hoping to infer lane position from past information (very useful in case of car/shadow occlusion).
 
 Note that data structure is maintained throughout the whole network: no vector flattening is ever applied.
 
 In order to do so, the FConv encoder is followed by a ConvLSTM, which fuses information coming from contiguous frames, replacing matrix multiplications in gates with convolutions,
 therefore exploiting the sparsity of the convolution operations for efficiency and better exploiting of data structure.
 
 Please note that the LSTM implementation provided (from [@ndrplz](https://github.com/ndrplz/ConvLSTM_pytorch), which I thank here) slightly differs from 
 the one formulated in the paper, as it does not include 'peephole connections' in the LSTM cell.
 *Update: it turns out the authors themselves made us of the same implementation.*
 
 
 Here's a short GIF highlighting some of the results you can easily achieve by training the network for a few hours:
 ![](https://media.giphy.com/media/TIEplKmoAVA2opXB7G/giphy.gif)
 
 And here's some pictures:
 
![Result-1](https://i.imgur.com/086ZAVu.png)
![Result-2](https://i.imgur.com/yfT9dZM.png)
![Result-3](https://i.imgur.com/Uyr5Mvo.png)
![Result-4](https://i.imgur.com/vOjNR9u.png)
 
 Even better results can be obtained by simply training for a longer time on 
 a good GPU or also refining/smoothing results with CV techniques or other neural/probabilistic models.
 
# Usage
 
 Make sure to download the 'TuSimple benchmark dataset' from https://github.com/TuSimple/tusimple-benchmark/issues/3,
 since that's the one I used in my experiments and for which I provide a pytorch DataLoader (utils/data_utils.py).
 The authors' of the paper augmented this dataset with additional data they recorded and labeled, and this is probably the biggest difference with the work they presented.
 
 Once you have the data unpacked, head to utils/config.py and set the global variables as well as the hyperparameters for training from the file itself (variable name should be self-explanatory).
 I know using _argparse_ would've offered a better interface in terms of usability, but I was too lazy to set that up during development.. my apologies.
 
 I trained on a TeslaV100 for some hours(~4) and it took around ~0.3s per batch; this time should theoretically be divided by the number of frames in input if the feeding to the fully convolutional encoder was to be parallelized.  
 
# Model & additional resources

I also uploaded the trained model which is available [here](https://drive.google.com/file/d/123xT-45HuPkuPptqz_ce0GPZFhs8CtMU/view?usp=sharing).
You can (visually) evaluate the results by running the script 'visual_evaluation.py' while you can (analytically) evaluate the results by running the 'test.py' script, both in the root folder; make sure you change the directory to point to the downloaded model when the model is loaded (I left my own 'fixed' directories here too.. again, sorry :P). 

In case you're very very interested in the model architecture or the whole process you can take a look at the slides I utilized to introduce lane detection at the ML Milan Meetup at Politecnico di Milano on 09/05/19: here's the [power point version](https://drive.google.com/file/d/1hfW4FK8Kioz8QmK3uljXQY-W6uWuHelG/view?usp=sharing)(which I suggest because of animations and videos I used; I'll make sure to use a more shareable format next time) and the [pdf one](https://drive.google.com/file/d/1SUHWx8TT70efgoN1SQ-AIjEA8DyK-0Gw/view?usp=sharing).
