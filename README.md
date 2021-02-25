# Semantic Segmentation of Satellite Images using Tensorflow 2.0
This repo details the steps carried out in order to perform a [Semantic Segmentation](https://nanonets.com/blog/semantic-image-segmentation-2020/) task on Satellite and/or Aerial images (aka tiles). A tensorflow 2.0 deep learning model is trained using the [ISPRS dataset](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/). The data set contains 38 patches (of 6000x6000 pixels), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic. 

Each tile is paired with a reference segmentation mask depicting the 6 classes with differents colors (see below).
![example](/images/isprs-example-tile-mask.JPG)

## Tile extraction and data augmentation using tf.data input pipeline
![example](/images/tile-patching.png)
## Tensorflow model architecture
![example](/images/model-arch.JPG)
## Model training
![example](/images/learning-rate-finder.JPG)
![example](/images/one-cycle-training-policy.JPG)
![example](/images/trainin-metrics.png)
## Best Model Performance Metrics
![example](/images/validation-ds-metrics.JPG)
## Tile Prediction using Test-Time Augmentation
- Without test-time augmentation
![example](/images/tile-pred-no-tsa.JPG)

- With test-time augmentation
![example](/images/tile-pred-with-tsa.JPG)

## Jupy
