﻿# Dry Bean Classification

 
This project demonstrates the application of a Support Vector Machine (SVM) with a radial basis function (RBF) kernel to classify dry beans in a dataset. The dataset contains various features extracted from images of dry beans, and the task is to classify them into different categories based on these features. The goal is to tune the model's hyperparameters to optimize performance and assess its generalization ability.

## Overview

Dataset Description: 
The dataset used in this project is the Dry Bean Dataset. It consists of multiple features representing various attributes of dry beans, such as size, shape, and texture, derived from images. The dataset is designed to help classify different varieties of dry beans. The features are stored as continuous values, and the target variable represents the classification label for each dry bean sample.


Features: Multiple continuous numeric features representing characteristics of dry beans

Label: The classification label of each dry bean (the last column in the dataset)

Data Preprocessing:
The dataset contains integer columns that are converted to float64 for consistency.

If the dataset exceeds 10,000 samples, a random sample of 10,000 records is selected for training.

Features are standardized using StandardScaler to improve the model's performance.

## Model Description
This project uses a Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel. The SVC model is tuned using GridSearchCV to optimize hyperparameters like C and gamma using cross-validation. The hyperparameters C and gamma were selected from a grid of values to find the best combination that provides the highest cross-validation score.

Hyperparameter Search:
The following hyperparameters are tuned via grid search:
 - C: Regularization parameter, controlling the trade-off between achieving a low error on the training data and minimizing the model complexity.
 - gamma: Kernel coefficient for the RBF kernel, controlling the influence of a single training example.

## Results

 - Best parameters found: {'C': 10, 'gamma': 0.125}
 - Best cross-validation score: 0.9292857142857143
 - Train Accuracy: 0.944
 - Test Accuracy: 0.931
 - Number of support vectors: 1392  

<br>

![drybeansmatrix](https://github.com/user-attachments/assets/f0593c79-015b-4a72-8029-9f4bbf364c49)


