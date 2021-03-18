![Tensorflow](https://img.shields.io/badge/TensorFlow-2.0-orange) ![python](https://img.shields.io/badge/Python-3.5-green) ![jupyter](https://img.shields.io/badge/Made%20with-Jupyter-blue)


# Semantic Segmentation of Satellite Images using Tensorflow 2.0
This repo details the steps carried out in order to perform a [Semantic Segmentation](https://nanonets.com/blog/semantic-image-segmentation-2020/) task on Satellite and/or Aerial images (aka tiles). A Tensorflow 2.0 deep learning model is trained using the [ISPRS dataset](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/). The data set contains 38 patches (of 6000x6000 pixels), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic. 

Each tile is paired with a reference segmentation mask depicting the 6 classes with different colors (see below).
![example](/images/isprs-example-tile-mask.JPG)

## Development Environment
Tools and libraries:
- Python 3.5
- Imageio 2.6
- Deep Learning Libraries:
- Low-Level API: Tensorflow 2.0 (with eager_execution enabled)
- High-Level API: Keras 2.2
- Input pipeline API: Tensorflow.data
- Monitoring API: TensorBoard

Infrastructure:
- 16-Core, 64GB RAM
- Nvidia 16GB GPU (Tesla P100)
- VM instance on GCP 



## Patch extraction and data augmentation using tf.data input pipeline
Due to the size of the tiles (600x6000 pixels), it is not possible to feed them directly to the Tensorflow model which has an image input size limited to 256x256 pixels. Thus it is crucial to build an efficient and flexible input pipeline that reads the tile file, extracts smaller patches, performs data augmentation techniques while being fast enough to avoid data starvation of the model sitting on the GPU during the training phase. Fortunately, Tensorflow's [tf.data](https://www.tensorflow.org/guide/data) allows the building of such a pipeline. The tile and its corresponding reference mask are processed in parallel and the produced smaller patches are like shown in the following grid:
![example](/images/tile-patching.png)
## Tensorflow model architecture
The model is based on [U-Net](https://en.wikipedia.org/wiki/U-Net) convolutional neural network that was enhanced using skip connections and residual blocks borrowed from the [Residual Neural Networks](https://en.wikipedia.org/wiki/Residual_neural_network) that help enhance the flow of the gradient during the backpropagation step. [Keras functional API](https://www.tensorflow.org/guide/keras/functional) was used to implement the model.
![example](/images/model-arch.JPG)
## Model training
We experimented with several loss functions based on recent A.I literature.
- The classical Sparse [Categorical Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy).
- The [Categorical focal loss](https://arxiv.org/pdf/1708.02002.pdf) helpful when we have imbalanced target classes.
- A custom-made J[accard Distance loss function](https://arxiv.org/abs/1908.03851) as we look to maximize the Intersection over Union (IoU).

In addition, we adopted the [learning rate finder](https://arxiv.org/abs/1506.01186) to spot the optimum learning-rate for the a choosen loss function. The finding process produces the following loss curve showing the learning rate sweet spot that should be picked (right before global minimum) for optimum training.
![example](/images/learning-rate-finder.JPG)

Once the optimum learning rate is found, the training is performed using the [one-cycle policy](https://arxiv.org/abs/1708.07120) training strategy. The curves below depict the evolution of the learning rate and the SGD momentum during training.
![example](/images/one-cycle-training-policy.JPG)

Naturally, during training, we monitor the performance metrics: Accuracy, IoU, and the loss function as shown below. The training is halted thanks to the [Early Stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) strategy once the performance metrics stagnate.
![example](/images/trainin-metrics.png)
## Best Model Performance Metrics
The model performance measured on the validation dataset is quite amazing especially on the `Building`, `Road` and `Car` classes (IoU > 0.8). Below are the Confusion Matrix and the Per-class IoU metrics along with some reference visuals for the IoU metric.
![example](/images/validation-ds-metrics.JPG)
## Tile Prediction using Test-Time Augmentation
Applying the inference pipeline to a new tile of the same size (6000x6000) could be slow if we loop through the tile to extract the patches, make  a batch prediction, and stitch them  together the patches to reconstruct the tile. Fortunately, can we perform such inference without any loop thanks to a clever tile reconstruction trick using Tensorflow's [tf.scatter_nd](https://www.tensorflow.org/api_docs/python/tf/scatter_nd). Inference time on tile is reduced from minutes to seconds. 

In addition, once we performed an inference on a tile, [Test-time augmentation](https://arxiv.org/abs/2011.11156) technique enhances by several points the prediction quality as shown below:
- Without test-time augmentation
![example](/images/tile-pred-no-tsa.JPG)

- With test-time augmentation
![example](/images/tile-pred-with-tsa.JPG)

## Implementation and Report
- [Link to Jupyter Notebook](Prototype_Segmenter.ipynb)
- [Link to Report Slides](Maher%20SEBAI%20internship%20presentation.pdf)
