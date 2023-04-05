# Deep Convolutional Models: Case Studies

Discover some powerful practical tricks and methods used in deep CNNs, straight from the research papers, then apply transfer learning to your own deep CNN.

Learning Objectives
- Implement the basic building blocks of ResNets in a deep neural network using Keras
- Train a state-of-the-art neural network for image classification
- Implement a skip connection in your network
- Create a dataset from a directory
- Preprocess and augment data using the Keras Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tine a classifier's final layers to improve accuracy


# Case Studies

## Why look at case studies?


[Last week](../week1/README.md) we learned about the basic building blocks, such as convolutional layers, pooling layers, and fully connected layers of convnet. 

In the past few years, a lot of computer vision research has been done to put together these basic building blocks to form effective convolutional neural networks. 

As many may have learned to write code by reading other people's codes, a good way to gain intuition and how the build confidence is to read or to see other examples of effective confidence. It turns out that a neural network architecture that works well on one computer vision tasks often works well on other tasks as well.

We will see the following 
- standard networks :
    - LeNet-5
    - AlexNet
    - VGG
- ResNet, neural network trained a very deep 152 layer neural network
- Inception 

After seeing these neural networks, I think you have much better intuition about how to build effective convolutional neural networks. Even if you don't end up building computer vision applications yourself, I think you'll find some of these ideas very interesting and helpful for your work.

> <img src="./images/w02-01-Why_look_at_case_studies/img_2023-04-04_21-35-02.png">

## Classic Networks

> <img src="./images/w02-02-Classic_Networks/img_2023-04-04_21-36-22.png">
> <img src="./images/w02-02-Classic_Networks/img_2023-04-04_21-36-24.png">
> <img src="./images/w02-02-Classic_Networks/img_2023-04-04_21-36-26.png">

## ResNets

> <img src="./images/w02-03-ResNets/img_2023-04-04_21-36-44.png">
> <img src="./images/w02-03-ResNets/img_2023-04-04_21-36-45.png">

## Why ResNets Work?

> <img src="./images/w02-04-Why_ResNets_Work/img_2023-04-04_21-36-59.png">
> <img src="./images/w02-04-Why_ResNets_Work/img_2023-04-04_21-37-00.png">

## Networks in Networks and 1x1 Convolutions

> <img src="./images/w02-05-Networks_in_Networks_and_1x1_Convolutions/img_2023-04-04_21-37-21.png">
> <img src="./images/w02-05-Networks_in_Networks_and_1x1_Convolutions/img_2023-04-04_21-37-22.png">

## Inception Network Motivation

> <img src="./images/w02-06-Inception_Network_Motivation/img_2023-04-04_21-37-40.png">
> <img src="./images/w02-06-Inception_Network_Motivation/img_2023-04-04_21-37-42.png">
> <img src="./images/w02-06-Inception_Network_Motivation/img_2023-04-04_21-37-43.png">

## Inception Network

> <img src="./images/w02-07-Inception_Network/img_2023-04-04_21-38-51.png">
> <img src="./images/w02-07-Inception_Network/img_2023-04-04_21-38-53.png">
> <img src="./images/w02-07-Inception_Network/img_2023-04-04_21-38-55.png">

## MobileNet

> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-07.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-09.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-10.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-13.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-15.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-17.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-18.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-20.png">
> <img src="./images/w02-08-MobileNet/img_2023-04-04_21-39-22.png">

## MobileNet Architecture

> <img src="./images/w02-09-MobileNet_Architecture/img_2023-04-04_21-39-35.png">
> <img src="./images/w02-09-MobileNet_Architecture/img_2023-04-04_21-39-37.png">
> <img src="./images/w02-09-MobileNet_Architecture/img_2023-04-04_21-39-38.png">
> <img src="./images/w02-09-MobileNet_Architecture/img_2023-04-04_21-39-40.png">

## EfficientNet

> <img src="./images/w02-10-EfficientNet/img_2023-04-04_21-39-52.png">



# Practical Advice for Using ConvNets

## Using Open-Source Implementation

## Transfer Learning

> <img src="./images/w02-12-Transfer_Learning/img_2023-04-04_21-40-15.png">

## Data Augmentation

> <img src="./images/w02-13-Data_Augmentation/img_2023-04-04_21-40-52.png">
> <img src="./images/w02-13-Data_Augmentation/img_2023-04-04_21-40-54.png">
> <img src="./images/w02-13-Data_Augmentation/img_2023-04-04_21-40-56.png">

## State of Computer Vision

> <img src="./images/w02-14-State_of_Computer_Vision/img_2023-04-04_21-41-21.png">
> <img src="./images/w02-14-State_of_Computer_Vision/img_2023-04-04_21-41-23.png">
> <img src="./images/w02-14-State_of_Computer_Vision/img_2023-04-04_21-41-27.png">

