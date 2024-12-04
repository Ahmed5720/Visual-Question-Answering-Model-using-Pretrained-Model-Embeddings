Visual Question Answering Using preTrained Model Embeddings

In this project, we make an attempt at transfer learning to use pre-existing learned parameters to help us downstream a different task. We simplify our task to a classification task, however this classification requires learned parameters from two seperate encoders: an image, and a text encoder. ![Our architecture](https://github.com/Ahmed5720/Visual-Question-Answering-Model-using-Pretrained-Model-Embeddings/blob/main/Model12.jpg)

The architecture involves a linear layer with a ReLU activation for dimension reduction. We then follow it by concatenating the processed embeddings from both pretrained models. Finally this concatenated represenation is passed through a linear classifier.

We start by loading our dataset. The dataset we are using is the DAQUAR dataset (the Dataset on Question answering on Real-world images) which can be found [here](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset) 
```python
test
```
