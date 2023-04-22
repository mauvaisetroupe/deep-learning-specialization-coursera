# Special Applications: Face recognition & Neural Style Transfer

Explore how CNNs can be applied to multiple fields, including art generation and face recognition, then implement your own algorithm to generate art and recognize faces!

Learning Objectives
- Differentiate between face recognition and face verification
- Implement one-shot learning to solve a face recognition problem
- Apply the triplet loss function to learn a network's parameters in the context of face recognition
- Explain how to pose face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Perform face verification and face recognition with these encodings
- Implement the Neural Style Transfer algorithm
- Generate novel artistic images using Neural Style Transfer
- Define the style cost function for Neural Style Transfer
- Define the content cost function for Neural Style Transfer

# Face Recognition

## What is Face Recognition?

- Face verification 
    - Output whether the input image is that of the claimed person
- Face recognition
    - Output ID if the image is any of the K persons in the database (or “not recognized”)

If you are building a facial recognition system based on your facial verification system, the verification system must be over 99.9% accurate before you can run it on a database of 100 people and have a good chance to be correct.

What we do in the next few videos is 
1. focus on building a face verification system as a building block and then if the accuracy is high enough, 
2. then you probably use that in a recognition system as well

> <img src="./images/w04-01-what_is_face_recognition/img_2023-04-14_22-06-18.png">

## One Shot Learning

One of the challenges of face recognition is that you need to solve the one-shot learning problem (you need to be able to recognize a person given just one single image)

Let's say you have a database of 4 pictures of employees in you're organization.

When someone shows up at the office :
- despite ever having seen only one image of Danielle, to recognize that this is actually the same person. 
- and in contrast, if it sees someone that's not in this database, then it should recognize that this is not any of the four persons in the database

So one approach you could try is to input the image of the person, feed it too a ConvNet with a softmax unit with 5 outputs (corresponding to each of these four persons or none of the above). Because you have such a small training set it is really not enough to train a robust neural network for this task

And also what if a new person joins your team? So now you have 5 persons you need to recognize, so there should now be six outputs. Do you have to retrain the ConvNet every time?

> <img src="./images/w04-02-one_shot_learning/img_2023-04-14_22-07-17.png">

So instead, to make this work, what you're going to do instead is learn a similarity function

> <img src="./images/w04-02-one_shot_learning/img_2023-04-14_22-07-19.png">

## Siamese Network

The job of the function d, which you learned about in the last video, is to input two faces and tell you how similar or how different they are. A good way to do this is to use a Siamese network

In a Siamese network :
- we use a convolutional network, 
- but we remove the last layer
- instead of making a classification by a softmax unit, we focus on the vector computed by a fully connected layer as an encoding of the input image.
- we define the function d as the distance between x1 and x2, the norm of the difference between the encodings of these two images.

> <img src="./images/w04-03-siamese_network/img_2023-04-14_22-07-33.png">


We train the neural network so that if two pictures, xi and xj, are of the same person, then you want that distance between their encodings to be small.

> <img src="./images/w04-03-siamese_network/img_2023-04-14_22-07-35.png">

## Triplet Loss

One way to learn the parameters of the neural network, so that it gives you a good encoding for your pictures of faces, is to define and apply gradient descent on the triplet loss function. 

In the terminology of the triplet loss, you have :
- an anchor image A, 
- a positive image P, 
- a negative image N

The learning objective is to have d(A,P) ≤ d(A,N). But if f always equals zero or f always outputs the same (encoding for every image is identical), this trivial solution satisfy the inequation. That why we add α the **margin paramter**

> <img src="./images/w04-04-triplet_loss/img_2023-04-14_22-07-45.png">

Let's formalize the equation and define the triplet loss function. We start with the iniequation from previous slide and me take the max. 
The effect of **taking the max** is that so long as this is less than zero, then the loss is zero. Then so long as the difference is zero or less than equal to zero, the neural network doesn't care how much further negative it is. 

Note that :
- for training, you nee multiple pictures of the same person ( 10,000 pictures of 1,000 different persons, so on average 10 pictures by person)
- after having trained a system, you can then apply it to your one-shot learning problem where for your face recognition system with only a single picture of someone you might be trying to recognize

> <img src="./images/w04-04-triplet_loss/img_2023-04-14_22-07-47.png">

In order to have a performant network, you cannot choose you training set randomly (if pictures ar too differents, you won´t push the algorithm), you need to chose "hard" negatives.


> <img src="./images/w04-04-triplet_loss/img_2023-04-14_22-07-49.png">

Today's Face recognition systems, especially the large-scale commercial face recognition systems are trained on very large dataset (more tha 100 millions images). These dataset assets are not easy to acquire.
Fortunately, some of these companies have trained these large networks and posted parameters online. Rather than trying to train one of these networks from scratch, this is one domain where because of the sheer data volumes sizes, it might be useful for you to download someone else's pre-trained model rather than do everything from scratch yourself.

> <img src="./images/w04-04-triplet_loss/img_2023-04-14_22-07-51.png">

## Face Verification and Binary Classification

The Triplet Loss is one good way to learn the parameters of a continent for face recognition. There's another way to learn these parameters. 

Another way to train a neural network, is 
- to take 2 Siamese Network and have them both compute the 128 dimensional encodings vectors
- add a logistic regression unit to then just make a prediction where the target output is :
    - 1 if both of the images are the same persons, 
    - and 0 if both of these are of different persons

So, this is a way to treat face recognition just as a binary classification problem where you use for example a sigmoid function on the element-wide difference of the 2 vectors

Note that :
- in green, a variant call chi-square similarity
- when you compare 2 images, one comes fron the database, so you can precompute the output of the Siamese Network

> <img src="./images/w04-05-face_verification_and_binary_classification/img_2023-04-14_22-08-01.png">

For training the network, we need to prepare pairs, with output (0 or 1)

> <img src="./images/w04-05-face_verification_and_binary_classification/img_2023-04-14_22-08-02.png">

# Neural Style Transfer

## What is Neural Style Transfer?

Notation :
- C for Content image
- S for style image
- G for generated image

> <img src="./images/w04-06-what_is_neural_style_transfer/img_2023-04-14_22-08-15.png">

In order to implement Neural Style Transfer, you need to look at the features extracted by ConvNet at various layers :
- the shallow 
- and the deeper layers of a ConvNet

## What are deep ConvNets learning?

Lets say you've trained a ConvNet, this is an AlexNet like network, and you want to visualize what the hidden units in different layers are computing. 

You could scan through your training sets and find out what are the images or what are the image patches (pieces) that maximize that unit's activation. So in other words pause your training set through your neural network, and figure out what is the image that maximizes that particular unit's activation. 

Notice that :
- on hidden unit in layer 1 will see only a relatively small portion of the neural network. And so it makes makes sense to plot just a small image patches
- in the deeper layers, one hidden unit will see a larger region of the image. 


> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-28.png">

<!--
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-30.png">
-->

This visualization shows 
- nine different representative neurons 
    - and for each of them the nine image patches that they maximally activate on.

Layer 1, you can see there's an edge maybe at that angle

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-32.png">

Layer 2 detects more complex shapes and patterns like vertical texture with lots of vertical lines, rounder shape, etc.

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-34.png">

<!--
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-37.png">
-->

Layer 3, round shapes, but start detecting people

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-39.png">

Layer 4, dog detectors, legs of birds, water

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-42.png">

Layer 5, even more sophisticated things

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-44.png">

## Cost Function

$J(G) = \alpha * J_{content}(C, G) + \beta * J_{style}(S, G)$ :
- first part (content component) measures how similar is the contents of the generated image G to the content of the content image C
- second part (style component) measures how similar is the contents of the generated image G to the style of the style image J

> <img src="./images/w04-08-cost_function/img_2023-04-14_22-08-56.png">

Algorithm to train the network :

> <img src="./images/w04-08-cost_function/img_2023-04-14_22-08-57.png">

## Content Cost Function

We use a specific hidden layer l to compute the content part of the cost function. Usually, choose some layer in the middle, neither too shallow nor too dee. If l is to be small (like layer 1), we will force the network to get similar output to the original content image.

We define $J_{content}(C,G)$ as the element-wise difference between these hidden unit activations in layer l for content image and generated image (use norm-2 as activation are considered as vectors)


> <img src="./images/w04-09-content_cost_function/img_2023-04-14_22-09-07.png">

## Style Cost Function

We chosen some layer L to define the measure of the style of an image, and we're going to ask how correlated are the activations across different channels.

So in this below example, we consider five channels to make it easier to draw. 

> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-16.png">

Let's look at the first two channels (the red channel and the yellow channels) and say how correlated are activations in these first two channels. 

> <img src="./images/w04-10-style_cost_function/01.png">

- Highly **correlated** means is whatever part of the image has this type of subtle vertical texture (neurone corresponding to rad channel), that part of the image will probably have these orange-ish tint (neuron associated with yellow channel)
- **Uncorrelated** means if a value appeared in a specific channel doesn't mean that another value will appear (not depend on each other)

The correlation tells you how a components might occur or not occur together in the same image.

If we use the degree of correlation between channels as a measure of the style, you can compare how similar is the style of the generated image to the style of the input style image.

> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-18.png">

So let's now formalize this intuition. 

Matrix G, called style matrix or Gram matrix:
- G is matrix of shape $n_c^{[l]}$ x $n_c^{[l]}$ (numbre of channels)
- $G_{k,k'}$ is the cell of the matrix G that tell how correlated is a channel k to another channel k'
    - if both of activations $a_{i,j,k}^{[l]}$ and $a_{i,j,k´}^{[l]}$ tend to be lashed together then $G_{k,k'}$ will be large, 
    - whereas if they are uncorrelated then $G_{k,k'}$ might be small.

The term correlation has been used to convey intuition but this is actually the unnormalized cross-correlation because we're not subtracting out the mean (here just a elements-wide multiplication).

> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-19.png">

And, finally, it turns out that you get more visually pleasing results if you use the style cost function from multiple different layers. So, the overall style cost function, you can define as sum over all the different layers of the style cost function for that layer. It allows you to use different layers in a neural network and cause a neural network to take both low level and high level correlations into account when computing style :
- early ones, which measure relatively simpler low level features like edges 
- as well as some later layers, which measure high level features 

> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-21.png">

Definition of Frobenius norm 

> <img src="./images/w04-10-style_cost_function/02.png">


## 1D and 3D Generalizations

> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-36.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-38.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-41.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-43.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-44.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-46.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-47.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-49.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-52.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-54.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-55.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-57.png">
> <img src="./images/w04-11-1d_and_3d_generalizations/img_2023-04-14_22-09-59.png">