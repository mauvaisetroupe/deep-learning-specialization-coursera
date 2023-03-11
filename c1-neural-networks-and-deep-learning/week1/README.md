# Introduction to deep leatning



# Welcome

## Welcome

> <img src="./images/w01-01-Welcome/img_2023-03-04_12-26-25.png">

Starting about 100 years ago, the electrification of our society transformed every major industry, from transportation, manufacturing, to healthcare, to communications. Today, we see a surprisingly clear path for AI to bring a similar big transformation.

> <img src="./images/w01-01-Welcome/img_2023-03-04_12-26-37.png">

1. First course explains how to build a neural network, including a deep neural network
2. Second course is about improving deep learning perormance
3. Third course explains how to structure a machine learning project (training / development / cross-validation / test). This lesson share a lot of experiences from Andrew Ng
4. Course 4 is about **convolutional neural networks (CNN)**, often applied to images
5. Course 5 explains sequence models and how to apply them to natural language processing and other problems. Sequence models includes **Recurrent Neural Networks (RNN)**, and **Long Short Term Memory models (LSTM)**, often apply for natural **language processing (NLP)** problems, speech recognition, music generations

> <img src="./images/w01-01-Welcome/img_2023-03-04_12-26-39.png">




# Introduction to deep leatning

## What is a Neural Network?

To predict price of a house as a function of its size, we can use linear regression
As price cannot be negative, we compose this linear regression with function f(x)=max(0,x) (RELU, rectified linear unit)
This function function predict the housing prices can be considered as a very simple neural network. 

> <img src="./images/w01-02-What_is_a_Neural_Network/img_2023-03-11_08-03-18.png">

If we add other features (number of bedrooms, zip code, and wealth), we can consider that 
- one of the things that really affects the price of a house is family size. And that the size and the number of bedrooms determines whether or not a house can fit the family's family size.
-  zip code and wealth can estimate the school quality
- ... 

> <img src="./images/w01-02-What_is_a_Neural_Network/img_2023-03-11_08-03-22.png">


> <img src="./images/w01-02-What_is_a_Neural_Network/img_2023-03-11_08-07-01.png">

## Supervised Learning with Neural Networks

But when working with neural networks, you actually implements the following
The job of the neural network is to predict the price, with features in inputs
Notes that all units in the middle (calles hiddent units of neural networks) are connected to all features

And rather than saying this first node represents family size and family size depends only on the features X1 and X2. Instead, we're going to let the neural network decide whatever this node to be. And we'll give the network all four input features to compute whatever it wants.

Because every input feature is connected to every one of these circles in the middle. And the remarkable thing about neural networks is that, given enough data about x and y, given enough training examples with both x and y, neural networks are remarkably good at figuring out functions that accurately map from x to y.

## Why is Deep Learning taking off?

## About this Course





# Heroes of Deep Learning

## Geoffrey Hinton Interview

