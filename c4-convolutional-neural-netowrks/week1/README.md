Implement the foundational layers of CNNs (pooling, convolutions) and stack them properly in a deep network to solve multi-class image classification problems.

Learning Objectives
- Explain the convolution operation
- Apply two different types of pooling operations
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build a convolutional neural network
- Implement convolutional and pooling layers in numpy, including forward propagation
- Implement helper functions to use when implementing a TensorFlow model
- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API
- Build and train a ConvNet in TensorFlow for a binary classification problem
- Build and train a ConvNet in TensorFlow for a multiclass classification problem
- Explain different use cases for the Sequential and Functional APIs


# Convolutional Neural Networks

## Computer Vision

> <img src="./images/w01-01-Computer_Vision/img_2023-03-31_22-43-08.png">
> <img src="./images/w01-01-Computer_Vision/img_2023-04-01_09-44-37.png">
> <img src="./images/w01-01-Computer_Vision/img_2023-04-01_09-44-38.png">

## Edge Detection Example

> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_09-45-59.png">
> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_09-46-01.png">
> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_09-46-03.png">

## More Edge Detection

> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-23.png">
> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-24.png">
> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-26.png">

## Padding

> <img src="./images/w01-04-Padding/img_2023-04-01_09-46-39.png">
> <img src="./images/w01-04-Padding/img_2023-04-01_09-46-41.png">

## Strided Convolutions

> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-46-55.png">
> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-46-58.png">
> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-47-00.png">

## Convolutions Over Volume

> <img src="./images/w01-06-Convolutions_Over_Volume/img_2023-04-01_09-47-12.png">
> <img src="./images/w01-06-Convolutions_Over_Volume/img_2023-04-01_09-47-13.png">
> <img src="./images/w01-06-Convolutions_Over_Volume/img_2023-04-01_09-47-15.png">

## One Layer of a Convolutional Network

> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-26.png">
> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-27.png">
> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-29.png">

## Simple Convolutional Network Example

> <img src="./images/w01-08-Simple_Convolutional_Network_Example/img_2023-04-01_09-47-41.png">
> <img src="./images/w01-08-Simple_Convolutional_Network_Example/img_2023-04-01_09-47-43.png">

## Pooling Layers

> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-47-56.png">
> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-47-59.png">
> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-48-01.png">
> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-48-03.png">

## CNN Example

> <img src="./images/w01-10-CNN_Example/img_2023-04-01_09-48-29.png">
> <img src="./images/w01-10-CNN_Example/img_2023-04-01_09-48-31.png">

## Why Convolutions?

> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-46.png">
> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-48.png">
> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-49.png">

# Heroes of Deep Learning (Optional)

## Yann LeCun Interview