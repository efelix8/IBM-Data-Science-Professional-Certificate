# ~10 Mini Projects in Machine Learning

## 1. Introduction to Supervised Learning with Iris Data

In this project, I apply Logistic Regression, the K-Nearest Neighbors algorithm, and the Support Vector Machine algorithm to analyze the Iris dataset.

## 2. Introduction to Unsupervised Learning with K-Means Clustering

Here, I implement the K-Means Clustering algorithm using self-generated data.

## 3. Board Game Review Prediction

Reviews significantly influence product success, especially for board games. In this project, I use **linear regression** and **random forest regressor** models to predict the average review score of a board game based on features like the minimum/maximum number of players, playing time, complexity, etc.

The dataset used is cloned from a GitHub repository using the following command:

`git clone https://github.com/ThaWeatherman/scrapers.git`

## 4. A Deep Reinforcement Learning Algorithm in OpenAI Gym

This project focuses on solving a cart and pole balancing problem using reinforcement learning in an OpenAI Gym environment. I use a deep neural network to implement the solution.

### Installing and Loading Libraries

First, install OpenAI Gym with the following commands:

`git clone https://github.com/openai/gym`
`cd gym`
`pip3 install -e . # minimal install`

I also install libraries like NumPy, Keras, and Theano.

## 5. Credit Card Fraud Detection

In this project, I build machine learning models to detect fraudulent credit card transactions. Using unsupervised anomaly detection algorithms like Local Outlier Factor (LOF) and the Isolation Forest Algorithm, I identify potential fraud. I evaluate the models using precision, recall, F1-scores, and explore data visualization techniques to understand the data.

## 6. Getting Started with Natural Language Processing in Python

Topics covered include:

- Tokenizing sentences and words
- Part-of-Speech tagging
- Chunking

I use the Natural Language Toolkit (NLTK) and train a Support Vector Classifier to classify movie reviews as positive or negative.

Install NLTK using:

`pip install nltk`

## 7. Object Recognition

This project involves using a convolutional neural network (CNN) for object recognition. I implement the All-CNN network from the 2015 ICLR paper, "Striving For Simplicity: The All Convolutional Net," using Keras with Theano as the backend.

The dataset used is CIFAR-10, which contains 60,000 32x32 color images across 10 classes.

## 8. Super Resolution Convolutional Neural Network for Image Restoration

I deploy the super-resolution convolutional neural network (SRCNN) to improve image quality, based on the 2014 paper "Image Super-Resolution Using Deep Convolutional Networks." Metrics like PSNR, MSE, and SSIM are used for evaluation. Additionally, I use OpenCV to process the images by converting between color spaces.

## 9. Natural Language Processing for Text Classification with NLTK and Scikit-learn

This project focuses on spam text message detection using:

- Regular Expressions
- Feature Engineering
- Multiple scikit-learn Classifiers
- Ensemble Methods

## 10. K-Means Clustering for Imagery Analysis

Here, I use K-Means clustering to classify images from the MNIST dataset, using Scikit-learn for implementation and visualization of cluster performance.

## 11. Data Compression and Visualization using Principal Component Analysis (PCA)

In this project, I use PCA to reduce dimensionality and visualize clusters formed by K-Means clustering on the Iris dataset. Techniques like the Elbow Method and meshgrid visualizations for PCA-reduced data are employed.
