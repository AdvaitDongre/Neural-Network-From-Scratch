# Neural Network from Scratch: Digit Recognition

This project demonstrates how to build a simple neural network from scratch to recognize handwritten digits using the MNIST dataset. The implementation is done using basic Python and NumPy without relying on deep learning frameworks like TensorFlow or PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Introduction

The objective of this project is to create a neural network from scratch to classify digits from the MNIST dataset. The MNIST dataset is a benchmark dataset in the machine learning community, consisting of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9.

## Requirements

To run this project, you will need the following libraries:
- Python 3.x
- NumPy
- Matplotlib
- sklearn

You can install the required libraries using the following command:
```bash
pip install numpy matplotlib scikit-learn
```

## Dataset

The MNIST dataset contains 28x28 grayscale images of handwritten digits. Each image is labeled with the correct digit. The dataset is available from various sources such as `keras.datasets` or `sklearn.datasets`.

Kaggle Link: https://www.kaggle.com/competitions/digit-recognizer

## Model Architecture

The neural network in this project consists of:
- Input layer: 784 neurons (28x28 pixels flattened)
- One hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation (one for each digit)

## Training the Model

The model is trained using the stochastic gradient descent (SGD) algorithm. The loss function used is categorical cross-entropy, which is standard for multi-class classification problems.

## Evaluation

The trained model is evaluated on the test set to measure its accuracy. The notebook includes code to visualize the predictions and the confusion matrix to understand the performance better.

## Usage

To run the project, execute the Jupyter notebook `neural-network-from-scratch-digits-recognition.ipynb`. The notebook includes the following steps:
1. Loading and preprocessing the MNIST dataset
2. Building the neural network model
3. Training the model
4. Evaluating the model on the test set
5. Visualizing the results

## Results

The model achieves an accuracy of approximately 84% on the test set, demonstrating a reasonable level of performance given its simplicity.


This README provides an overview of the project, instructions for setup and usage, and a summary of the results. Feel free to adjust it to fit the specifics of your implementation and project structure.
