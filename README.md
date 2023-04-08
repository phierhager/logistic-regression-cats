# Logistic Regression for Cats

## Overview

This repository contains a machine learning model for distinguishing between cat and non-cat images. It uses logistic regression to classify the images and has been trained on a provided dataset consisting of a training set and a test set.

## How to Use

To use the model, follow these steps:

1. Install poetry on your computer.
2. Set up the virtual environment with `poetry install`.
3. Run `poetry run run` to execute a script that compares different learning rates.

## Technical Details

### Mathematical Expression of the Algorithm

The logistic regression algorithm works as follows:

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

### Backpropagation

The backpropagation algorithm works as follows:

- We get X
- We compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
- We calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$

Two formulas implemented:

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

## References

- The original code for this model is based on Andrew Ng's "Neural Networks and Deep Learning" course on Coursera.
