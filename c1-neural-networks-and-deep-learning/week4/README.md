# Deep L-layer Neural Network

Analyze the key computations underlying deep learning, then use them to build and train deep neural networks for computer vision tasks.

Learning Objectives
- Describe the successive block structure of a deep neural network
- Build a deep L-layer neural network
- Analyze matrix and vector dimensions to check neural network implementations
- Use a cache to pass information from forward to back propagation
- Explain the role of hyperparameters in deep learning
- Build a 2-layer neural network


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


Let's put together basic building blocks, forward propagation and back propagation to implement a deep neural network. 

> <img src="./images/w02-05-Building_Blocks_of_Deep_Neural_Networks/img_2023-03-18_10-42-11.png">

> <img src="./images/w02-05-Building_Blocks_of_Deep_Neural_Networks/img_2023-03-18_10-42-16.png">

## Forward and Backward Propagation

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_14-56-35.png">

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_14-58-45.png">

> <img src="./images/w02-06-Forward_and_Backward_Propagation/img_2023-03-18_15-13-33.png">

## Parameters vs Hyperparameters

> <img src="./images/w02-07-Parameters_vs_Hyperparameters/img_2023-03-18_15-14-12.png">

> <img src="./images/w02-07-Parameters_vs_Hyperparameters/img_2023-03-18_15-14-14.png">
## What does this have to do with the brain?

> <img src="./images/w02-08-What_does_this_have_to_do_with_the_brain/img_2023-03-18_15-15-04.png">
