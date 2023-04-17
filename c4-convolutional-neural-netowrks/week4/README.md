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

> <img src="./images/w04-05-face_verification_and_binary_classification/img_2023-04-14_22-08-01.png">
> <img src="./images/w04-05-face_verification_and_binary_classification/img_2023-04-14_22-08-02.png">

# Neural Style Transfer

## What is Neural Style Transfer?

> <img src="./images/w04-06-what_is_neural_style_transfer/img_2023-04-14_22-08-15.png">

## What are deep ConvNets learning?

> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-28.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-30.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-32.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-34.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-37.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-39.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-42.png">
> <img src="./images/w04-07-what_are_deep_convnets_learning/img_2023-04-14_22-08-44.png">

## Cost Function

> <img src="./images/w04-08-cost_function/img_2023-04-14_22-08-56.png">
> <img src="./images/w04-08-cost_function/img_2023-04-14_22-08-57.png">

## Content Cost Function

> <img src="./images/w04-09-content_cost_function/img_2023-04-14_22-09-07.png">

## Style Cost Function

> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-16.png">
> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-18.png">
> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-19.png">
> <img src="./images/w04-10-style_cost_function/img_2023-04-14_22-09-21.png">

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