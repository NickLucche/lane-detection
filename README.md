# Implementation of a SegnetConvLSTM for Lane Detection
#### Please refer to the publication [Robust Lane Detection from Continuous DrivingScenes Using Deep Neural Networks](https://arxiv.org/pdf/1903.02193.pdf) by Qin Zou, Hanwen Jiang, Qiyu Dai, Yuanhao Yue, Long Chen, and Qian Wang for further information.

 This repo contains my PyTorch implementation of the afore mentioned paper, a model having the following architecture:
 
 ![architecture](https://www.groundai.com/media/arxiv_projects/518710/x2.png)
 
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
 
![Imgur](https://i.imgur.com/086ZAVu.png)
![Imgur](https://i.imgur.com/yfT9dZM.png)
![Imgur](https://i.imgur.com/Uyr5Mvo.png)
![Imgur](https://i.imgur.com/vOjNR9u.png)
 
 Even better results can be obtained by simply training for a longer time on 
 a good GPU or also refining results with some CV techniques or trying to apply Markov 
 Random Fields for smoothing.
