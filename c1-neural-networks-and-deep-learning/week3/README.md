---
layout: page
title: W3 - Shallow Neural Networks
permalink: /c1-neural-networks-and-deep-learning/week3/
parent: "C1 - Neural Networks and Deep Learning"
---

# Shallow Neural Networks

Build a neural network with one hidden layer, using forward propagation and backpropagation.

Learning Objectives
- Describe hidden units and hidden layers
- Use units with a non-linear activation function, such as tanh
- Implement forward and backward propagation
- Apply random initialization to your neural network
- Increase fluency in Deep Learning notations and Neural Network Representations
- Implement a 2-class classification neural network with a single hidden layer
- Compute the cross entropy loss


# Shallow Neural Networks

## Neural Networks Overview

> <img src="./images/w03-01-Neural_Networks_Overview/img_2023-03-12_19-49-48.png">

## Neural Network Representation

- Input layer / Hidden Layer / Output layer
- The term hidden layer refers to the fact that in the training set, the true values for these nodes in the middle are not observed
- Input is vector $x = a^{[0]}$, a is stands for activation

> <img src="./images/w03-02-Neural_Network_Representation/img_2023-03-12_19-52-28.png">

## Computing a Neural Network's Output

Each unit or node of our neural network compute a linear regression :
- $z=w^Tx+b$
- $a=\sigma(z)$

By convention, we denote $a_i^{[l]}$ and $z_i^{[l]}$ where :
- $i$ is the unit number
- and $l$ the layer number

> <img src="./images/w03-03-Computing_a_Neural_Networks_Output/img_2023-03-13_18-53-50.png">

Vectorization using vectr and matrix for hidden layer

> <img src="./images/w03-03-Computing_a_Neural_Networks_Output/img_2023-03-13_21-17-59.png">

Vectorization using vectr and matrix for output layer with $x = a^{[0]}$


> <img src="./images/w03-03-Computing_a_Neural_Networks_Output/img_2023-03-13_21-18-01.png">

## Vectorizing Across Multiple Examples

We define :
- $x^{(i)}$ the example # i 
- and $a^{[2] (i)}$ the prediction # i 

> <img src="./images/w03-04-Vectorizing_Across_Multiple_Examples/img_2023-03-13_21-22-09.png">

Instead of implementing a loop on differents training example, we build a matrix with all traing vectors. Each column is one example

> <img src="./images/w03-04-Vectorizing_Across_Multiple_Examples/img_2023-03-13_21-22-11.png">

## Explanation for Vectorized Implementation

> <img src="./images/w03-05-Explanation_for_Vectorized_Implementation/img_2023-03-13_21-26-55.png">
> <img src="./images/w03-05-Explanation_for_Vectorized_Implementation/img_2023-03-13_21-26-57.png">

## Activation Functions

Also see : https://github.com/mauvaisetroupe/machine-learning-specialization-coursera/blob/main/c2-advanced-learning-algorithms/week2/README.md#choosing-activation-functions

In the previous examples, we used sigmoid function. Sigmoid function is called an **activation function**.

The **hyperbolic tangent** function (tanh) works almost always better than the sigmoid function (because centering the data around zero is efficient when training algorithm). One exception is for output layer on binary classification (prediction 0 or 1 given by sigmoid is more adapted).

> <img src="./images/w03-06-Activation_Functions/img_2023-03-13_22-17-18.png">

Rules of thumb for choosing activation functions : 
- never use sigmoid activation function except for the output layer of binomial classification
- prefere hyperbolic tangent
- ReLU is the default choice (but )
- or try Leaky ReLu $max(0.01*z,z)$

> <img src="./images/w03-06-Activation_Functions/img_2023-03-13_22-17-21.png">


## Why do you need Non-Linear Activation Functions?

https://github.com/mauvaisetroupe/machine-learning-specialization-coursera/blob/main/c2-advanced-learning-algorithms/week2/README.md#why-do-we-need-activation-functions

> <img src="./images/w03-07-Why_do_you_need_Non-Linear_Activation_Functions/img_2023-03-13_22-23-56.png">s

## Derivatives of Activation Functions

> <img src="./images/w03-08-Derivatives_of_Activation_Functions/img_2023-03-13_22-35-18.png">

Technically, derivative not defined in zero, but for algorithm, could consider g'(0)=0 

> <img src="./images/w03-08-Derivatives_of_Activation_Functions/img_2023-03-13_22-35-21.png">

> <img src="./images/w03-08-Derivatives_of_Activation_Functions/img_2023-03-13_22-35-23.png">

## Gradient Descent for Neural Networks

> <img src="./images/w03-09-Gradient_Descent_for_Neural_Networks/img_2023-03-15_07-01-36.png">
> <img src="./images/w03-09-Gradient_Descent_for_Neural_Networks/img_2023-03-15_07-01-38.png">

## Backpropagation Intuition (Optional)

This slide for logistic regression explained here : https://github.com/mauvaisetroupe/deep-learning-specialization-coursera/tree/main/c1-neural-networks-and-deep-learning/week2#logistic-regression-gradient-descent

> <img src="./images/w03-10-Backpropagation_Intuition/img_2023-03-15_07-02-17.png">

We have exactly the same calculus for a neural network with two layers :

> <img src="./images/w03-10-Backpropagation_Intuition/img_2023-03-15_07-02-20.png">

Explanation of matrix usage to compute over all training examples (stacking them into a matrix) is here : https://github.com/mauvaisetroupe/deep-learning-specialization-coursera/tree/main/c1-neural-networks-and-deep-learning/week3#explanation-for-vectorized-implementation

> <img src="./images/w03-10-Backpropagation_Intuition/img_2023-03-15_09-15-43.png">


We've seen here the gradient decsent algorith here : https://github.com/mauvaisetroupe/deep-learning-specialization-coursera/tree/main/c1-neural-networks-and-deep-learning/week2#vectorizing-logistic-regressions-gradient-output

> <img src="./images/w03-10-Backpropagation_Intuition/img_2023-03-15_12-13-43.png">

If we apply derivation over all training examples to run gradient descent algorith, we obtain:

> <img src="./images/w03-10-Backpropagation_Intuition/img_2023-03-15_09-15-51.png">



## Random Initialization

If we initialize the weights to zero, all hidden units are symmetric. And no matter how long we're upgrading the center, all continue to compute exactly the same function. The solution to this is to initialize your parameters randomly. 

> <img src="./images/w03-11-Random_Initialization/img_2023-03-15_12-19-16.png">

We prefer to initialize the weights to very small random values. Because if you are using a tanh or sigmoid activation function, to avoid being in the flat parts of these functions

> <img src="./images/w03-11-Random_Initialization/img_2023-03-15_12-19-18.png">



# Heroes of Deep Learning (Optional)

## Ian Goodfellow Interview
