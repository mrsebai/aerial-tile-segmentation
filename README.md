# Semantic Segmentation of Satellite Images using Tensorflow 2.0
This repo details the steps carried out in order to perform a [Semantic Segmentation](https://nanonets.com/blog/semantic-image-segmentation-2020/) task on Satellite and/or Aerial images (aka tiles). We use a deep learning model that we will be trained using the [ISPRS dataset](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/). The data set contains 38 patches (of 6000x6000 pixels), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic. Each tile is paired with a reference segmentation mask depicting the 6 classes with differents colors (see below).

![example](/images/isprs-example-tile-mask.JPG)
