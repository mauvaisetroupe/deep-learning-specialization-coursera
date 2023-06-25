---
layout: page
title: W4 - Deep L-layer Neural Network
permalink: /c1-neural-networks-and-deep-learning/week4/
parent: "C1 - Neural Networks and Deep Learning"
---

# Deep L-layer Neural Network
{: .no_toc }

Analyze the key computations underlying deep learning, then use them to build and train deep neural networks for computer vision tasks.

Learning Objectives
- Describe the successive block structure of a deep neural network
- Build a deep L-layer neural network
- Analyze matrix and vector dimensions to check neural network implementations
- Use a cache to pass information from forward to back propagation
- Explain the role of hyperparameters in deep learning
- Build a 2-layer neural network

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Deep L-layer Neural Network

## Deep L-layer Neural Network

Welcome to the fourth week of this course. By now, you've seen 
- forward propagation and back propagation in the context of a neural network, with a single hidden layer, 
- logistic regression, 
- vectorization, 
- and when it's important to initialize the ways randomly. 

If you've done the past couple weeks homework, you've also implemented and seen some of these ideas work for yourself

- **shallow** model (one layer) vs **deep** model (5 hidden layer)

> <img src="./images/w02-01-Deep_L-layer_Neural_Network/img_2023-03-18_09-00-45.png">

Notation for deep neural network:
- l the number of layer
- $n^{[l]}$ = number of unit in layer l
- $a^{[l]}$ = activation layer in layer l 
- $a^{[l]} = g^{[l]}(z^{[l]})$  with g the activation function
> <img src="./images/w02-01-Deep_L-layer_Neural_Network/img_2023-03-18_09-00-50.png">

## Forward Propagation in a Deep Network

> <img src="./images/w02-02-Forward_Propagation_in_a_Deep_Network/img_2023-03-18_09-08-22.png">

## Getting your Matrix Dimensions Right

> <img src="./images/w02-03-Getting_your_Matrix_Dimensions_Right/img_2023-03-18_09-10-01.png">

> <img src="./images/w02-03-Getting_your_Matrix_Dimensions_Right/img_2023-03-18_09-10-07.png">

> <img src="./images/w02-03-Getting_your_Matrix_Dimensions_Right/img_2023-03-18_10-14-11.png">

## Why Deep Representations?

> <img src="./images/w02-04-Why_Deep_Representations/img_2023-03-18_10-16-39.png">

> <img src="./images/w02-04-Why_Deep_Representations/img_2023-03-18_10-16-42.png">

 Now, in addition to this reasons for preferring deep neural networks, to be perfectly honest, I think the other reasons the term deep learning has taken off is just branding. 

## Building Blocks of Deep Neural Networks


In the earlier videos from this week, as well as from the videos from the past several weeks, you've already seen the basic building blocks of forward propagation and back propagation, the key components you need to implement a deep neural network. Let's see how you can put these components together to build your deep net.

Explanation on layer `l`.

> <img src="./images/w02-05-Building_Blocks_of_Deep_Neural_Networks/img_2023-03-18_10-42-11.png">

Note that during forward, we cache `Z[l]`, but also `W[l]` and `b[l]`

> <img src="./images/w02-05-Building_Blocks_of_Deep_Neural_Networks/img_2023-03-18_10-42-16.png">

 So you've now seen what are the basic building blocks for implementing a deep neural network. In each layer there's a forward propagation step and there's a corresponding backward propagation step. And has a cache to pass information from one to the other. In the next video, we'll talk about how you can actually implement these building blocks

## Forward and Backward Propagation

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_14-56-35.png">

In addition to input value, we have cached values :
- `Z[l]` but also `W[l]` and `b[l]`
- I didn't explicitly put `a[l-1]` in the cache, but it turns out you need this as well 

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_14-58-45.png">

- For the forward recursion, we will initialize it with the input data X
- For backward loop, when using logistic regression (for binary classification), we initialize `da[] = -y/a + (1-y)/1-a)` (see [logistic regression recap](../week2/#logistic-regression-gradient-descent)) 

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_15-13-33.png">

## Parameters vs Hyperparameters

> <img src="./images/w02-07-Parameters_vs_Hyperparameters/img_2023-03-18_15-14-12.png">

> <img src="./images/w02-07-Parameters_vs_Hyperparameters/img_2023-03-18_15-14-14.png">
## What does this have to do with the brain?

> <img src="./images/w02-08-What_does_this_have_to_do_with_the_brain/img_2023-03-18_15-15-04.png">
