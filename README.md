Visual Question Answering Using preTrained Model Embeddings

In this project, we make an attempt at transfer learning to use pre-existing learned parameters to help us downstream a different task. We simplify our task to a classification task, however this classification requires learned parameters from two seperate encoders: an image, and a text encoder. 


# Visual Question Answering (VQA) using preTrained model Embeddings



This project implements a Visual Question Answering (VQA) model using the DAQUAR dataset. The model combines image and question embeddings, reduces their dimensions, and trains a classifier to predict answers.
The architecture involves a linear layer with a ReLU activation for dimension reduction. We then follow it by concatenating the processed embeddings from both pretrained models. Finally this concatenated represenation is passed through a linear classifier.

![Our architecture](https://github.com/Ahmed5720/Visual-Question-Answering-Model-using-Pretrained-Model-Embeddings/blob/main/Model12.jpg)



## Dataset

This code uses the DAQUAR dataset, structured as CSV files containing:
1. A column for questions.
2. A column for corresponding answers.
3. A column for image filenames.

-  images are stored in the `image_dir` directory.
- CSV file paths:
  - Training: `data_train.csv`
  - Validation: `data_val.csv`
  - Test: `data_test.csv`

## Pipeline Description

### Dataloader
The `DAQUARDataset` class defines a custom PyTorch dataset to handle the DAQUAR dataset. It:
- Reads image paths and labels from a CSV file.
- Applies preprocessing transforms to images.

```python
class DAQUARDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        ...
```
**Transforms:**
- Resizes images to 224x224.
- Normalizes pixel values.

### Image Embeddings
Image embeddings are extracted using a pretrained ResNet-50 model from `torchvision.models`. The classification head is removed, leaving only the convolutional backbone to produce 2048-dimensional embeddings for each image.

```python
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
```

### Sentence Embeddings
Sentence embeddings for questions are generated using the `sentence-transformers` library's `all-MiniLM-L6-v2` model. These embeddings are 384-dimensional.

```python
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Dimensionality Reduction
To ensure embeddings from different modalities (image and text) have the same dimension, a `ReduceDimension` class is implemented using a linear layer followed by ReLU activation.

```python
class ReduceDimension(nn.Module):
    def __init__(self, input_dim, output_dim):
        ...
```

- Image embeddings (2048) are reduced to 512 dimensions.
- Sentence embeddings (384) are reduced to 512 dimensions.

### Embedding Concatenation
Image and sentence embeddings are concatenated into a single vector of size 1024 for classification.

```python
def concat_embeddings(image_embeddings, sentence_embeddings):
    return torch.cat((image_embeddings, sentence_embeddings), dim=1)
```

### Classifier
A simple fully connected neural network serves as the classifier. It maps the concatenated embeddings (1024 dimensions) to the number of possible answers.

```python
class Classifier(nn.Module):
    def __init__(self, input_dim, class_num):
        ...
```

## Training
The `train_model` function trains the classifier using the following steps:
1. Extract image embeddings from ResNet.
2. Generate sentence embeddings using the SentenceTransformer model.
3. Apply dimensionality reduction.
4. Concatenate embeddings and pass them through the classifier.
5. Compute loss using cross-entropy.

Best model weights are saved based on validation accuracy.

```python
train_model(train_loader, val_loader, classifier, criterion, optimizer)
```

## Evaluation
The `evaluate_model` function computes accuracy on the test set and saves predictions to a CSV file (`test_predictions.csv`). 

```python
evaluate_model(test_loader, classifier)
```
