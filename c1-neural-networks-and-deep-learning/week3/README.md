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
- $a^{[2](i)}$ the prediction # i 

> <img src="./images/w03-04-Vectorizing_Across_Multiple_Examples/img_2023-03-13_21-22-09.png">

Instead of implementing a loop on differents training example, we build a matrix with all traing vectors. Each column is one example

> <img src="./images/w03-04-Vectorizing_Across_Multiple_Examples/img_2023-03-13_21-22-11.png">

## Explanation for Vectorized Implementation

> <img src="./images/w03-05-Explanation_for_Vectorized_Implementation/img_2023-03-13_21-26-55.png">
> <img src="./images/w03-05-Explanation_for_Vectorized_Implementation/img_2023-03-13_21-26-57.png">

## Activation Functions

## Why do you need Non-Linear Activation Functions?

## Derivatives of Activation Functions

## Gradient Descent for Neural Networks

## Backpropagation Intuition (Optional)

## Random Initialization



# Heroes of Deep Learning (Optional)

## Ian Goodfellow Interview
