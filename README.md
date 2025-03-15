# Object-Recognition-in-Images

## Overview
Image Classifier is a deep learning-based image classification system that recognizes objects in images using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. This project is designed to classify images into 10 distinct categories and can be deployed using Docker for seamless execution.

## Features
- CNN-based image classification.
- Pretrained model support.
- API-based interaction.

## Dataset
The model is trained on the CIFAR-10 dataset, which consists of 60,000 images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow/Keras
- Flask

## Model Training
To train the model from scratch, run:
```sh
python train.py
```
This will train the CNN model and save it as `model.h5`.

## Running the Flask API
To start the API server:
```sh
python app.py
```
The API will be accessible at `http://localhost:5000`.

