# Machine Learning Practice Library

This library contains a collection of fundamental machine learning algorithms built completely from scratch using only mathematics. The goal of this repository is to demonstrate how these models function under the hood without relying on external or black box machine learning frameworks.

## Table of Contents

1. [Supervised Learning](#supervised-learning)
2. [Unsupervised Learning](#unsupervised-learning)
3. [Projects Overview](#projects-overview)

## Supervised Learning

### Linear Regression
Predicts a continuous output based on one or more input features by fitting a straight line to the data.
* **File**: `linear-regression/linear_regression.py`
* **Inner Project**: **House Predictor**
  Predicts house prices based on input data. Built with a full frontend and backend architecture.

### Locally Weighted Linear Regression
A non parametric variation of linear regression. It fits a model to a target point by giving more weight to the training data that is closest to it locally.
* **File**: `locally-weighted-linear-regression/LWLR.py`
* **Inner Project**: **Weather Predictor**
  Predicts weather trends using local data features from 2013 to 2024.

### Logistic Regression
Used for binary classification. It outputs probabilities for a certain class using a mathematical sigmoid function.
* **File**: `logistic-regression/Logistic_Regression.py`
* **Inner Project**: **SMS Spam Detector**
  Detects whether an SMS message is spam or not. Uses a custom TF IDF Vectorizer.

### Softmax Regression
A generalization of logistic regression used for multi class classification problems.
* **File**: `softmax-regression/softmax_regression.py`
* **Inner Project**: **Digit Reader**
  A web application that reads and classifies handwritten digits.

### Gaussian Discriminative Analysis
A generative classification model that assumes the features follow a multivariate normal distribution.
* **File**: `Gaussian-Discriminative-Analysis/GDA.py`
* **Inner Project**: **Breast Cancer Predictor**
  Classifies breast cancer data into varying severity levels based on clinical features.

### Naive Bayes
A simple but effective probabilistic classifier based on applying Bayes theorem with strong independence assumptions.
* **File**: `Naive-Bayes/naive_bayes.py`

### Perceptron
The simplest type of artificial neural network used for linear binary classification.
* **File**: `perceptron/perceptron.py`
* **Inner Project**: **Digit Reader**
  An alternative frontend application to classify handwritten digits using the linear perceptron algorithm.

### Trees
Contains implementations related to decision tree algorithms for classification and regression tasks.
* **Directory**: `Trees/`

## Unsupervised Learning

### K Means Clustering
Partitions a dataset into k distinct groups or clusters based on feature similarity.
* **File**: `K-Means-Clustering/k_means_clustering.py`

### Principal Component Analysis
A dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the mathematical variance.
* **File**: `Principal-Component-Analysis/PCA.py`

### Independent Component Analysis
A computational method for separating a multivariate signal into additive and independent subcomponents.
* **Directory**: `Indenpendent-Component-Analysis/`

### Factor Analysis
Used to describe variability among observed correlated variables in terms of a lower number of unobserved variables called factors.
* **File**: `Factor-Analysis/Factor-Analysis.py`

### Mixtures of Gaussian
A probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
* **File**: `mixtures-of-gaussian/GMM.py`

## Projects Overview

Several of these mathematical models power full stack applications included in the repository. Each project directory typically contains the following structure:

* `requirements.txt`: Python dependencies required to run the web server.
* `app.py` or `main.py`: The entry point for the backend server.
* `plugins/`: Core implementation logic and custom helpers.
* `templates/` and `statics/`: Frontend HTML views, CSS, and JavaScript files to interact with the pure math models via a user interface.

To run any of the inner projects, navigate to the specific project directory, install the requirements, and execute the python server script.
