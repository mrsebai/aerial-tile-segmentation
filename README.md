# Semantic Segmentation of Satellite Images using Tensorflow 2.0
This repo details the steps carried out in order to perform a [Semantic Segmentation](https://nanonets.com/blog/semantic-image-segmentation-2020/) task on Satellite and/or Aerial images (aka tiles). A tensorflow 2.0 deep learning model is trained using the [ISPRS dataset](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/). The data set contains 38 patches (of 6000x6000 pixels), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic. 

Each tile is paired with a reference segmentation mask depicting the 6 classes with differents colors (see below).
![example](/images/isprs-example-tile-mask.JPG)

## Tile extraction and data augmentation using tf.data input pipeline
Due to the size of the tiles (600x6000 pixels) it is not possible to feed them directly to the Tensorflow model which has image input size limited to 256x256 pixels. Thus it is crucial to build an efficient and flexible input pipeline that read the tile file, extract smaller patches, perform data augmentation techniques while being fast enough to avoid data starvation of the model sitting on the GPU during training phase. Fortunately, Tensorflow's [tf.data](https://www.tensorflow.org/guide/data) allows building of such pipeline. The tile and its corresponding reference mask are processed in parrallel and the produced smaller patches are like shown in the following grid:
![example](/images/tile-patching.png)
## Tensorflow model architecture
The model is based on [U-Net](https://en.wikipedia.org/wiki/U-Net) convolutional neural network that was enhanced using Residual blocks borrowed from [Residual Neural Network](https://en.wikipedia.org/wiki/Residual_neural_network) that help enhance the flow of the gradient during backpropagation step. [Keras functional API](https://www.tensorflow.org/guide/keras/functional) was used to implement the model.
![example](/images/model-arch.JPG)
## Model training
We experimented with several loss functions based on recent A.I litterature.
- The classical Sparse [Categorical Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy).
- The [Categorical focal loss](https://arxiv.org/pdf/1708.02002.pdf) helpful when we have imbalanced target classes.
- A custom made J[accard Distance loss function](https://arxiv.org/abs/1908.03851) as we look to maximize the Intersection over Union (IoU).

In addition, we adopted the [learning rate finder](https://arxiv.org/abs/1506.01186) to spot the optimum learning-rate for the a choosen loss function. The finding process produces the following loss curve showing the learning rate sweet spot that should be picked (right before global minimum) for optimum training.
![example](/images/learning-rate-finder.JPG)

Once the optimum learning rate is found, the training is performed using the [one-cycle policy](https://arxiv.org/abs/1708.07120) training strategy. The curves below depicts the evolution of the learning rate and the SGD momemtum during training.
![example](/images/one-cycle-training-policy.JPG)

Naturally, during training, we monitor the performance metrics: Accurancy, IoU and the loss function as shown below. The training is halted thanks to early stopping strategy once the performance metrics stagnates.
![example](/images/trainin-metrics.png)
## Best Model Performance Metrics
![example](/images/validation-ds-metrics.JPG)
## Tile Prediction using Test-Time Augmentation
- Without test-time augmentation
![example](/images/tile-pred-no-tsa.JPG)

- With test-time augmentation
![example](/images/tile-pred-with-tsa.JPG)

## Implementation and Report
- [Link to Jupyter Notebook](Prototype_Segmenter.ipynb)
- [Link to Report Slides](Maher%20SEBAI%20internship%20presentation.pdf)
