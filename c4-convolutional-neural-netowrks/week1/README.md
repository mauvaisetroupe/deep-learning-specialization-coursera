# Foundations of Convolutional Neural Networks

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

Computer vision is one of the areas that's been advancing rapidly thanks to deep learning :
- self-driving cars
- face recognition
- unlock a phone or a door
- enabling new types of art t

Deep learning for computer vision is exciting because:
- rapid advances in computer vision are enabling new applications
- even if you don't end up building computer vision systems, because the computer vision research community has been so creative and so inventive in coming up with new neural network architectures and algorithms, it inspires other areas


Examples of a computer vision problems includes:
- Image classification.
- Object detection with localization (draw boxes around them)
- Neural style transfer (changes the style of a __content__ image using another __style__ image)

> <img src="./images/w01-01-Computer_Vision/img_2023-04-01_09-44-37.png">


One of the challenges of computer vision problems is that the inputs can get really big. A 1000x1000 image will be represented by 3 millions feature.

With 1000 units in the first hidden layer, we will work with a (1000 x 3'000,000) matrix


And with that many parameters, it's difficult
- to get enough data to prevent a neural network from **overfitting**
- **memory requirements** to train a neural network with three billion parameters is just a bit infeasible

To solve that, you need to implement the **convolution operation**, which is one of the fundamental building blocks of convolutional neural networks.


> <img src="./images/w01-01-Computer_Vision/img_2023-04-01_09-44-38.png">

## Edge Detection Example

The convolution operation is one of the fundamental building blocks of a convolutional neural network.
For a computer to figure out what are the objects in this picture, the first thing you might do is maybe detect horizontal and vertical edges in the image.

> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_09-45-59.png">

An example of convolution operation to detect vertical edges

All the deep learning frameworks that have a good support for computer vision will have some functions for implementing this convolution operator:
- In tensorflow cf. ```tf.nn.conv2d```
- keras ```conv2d```

> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_17-09-32.png">

Why is this doing vertical edge detection?

If you plot this right most matrix's image it will look like that where there is this lighter region right in the middle and that corresponds to this having detected this vertical edge down the middle of your 6 by 6 image.

Dimensions here seem a little bit wrong, that's only because we are working with very small images in this example. If you are using a 1000 x 1000 image rather than a 6 x 6 image then you find that this does a pretty good job, really detecting the vertical edges in your image.

> <img src="./images/w01-02-Edge_Detection_Example/img_2023-04-01_09-46-03.png">

## More Edge Detection

But this particular filter does make a difference between the light to dark versus the dark to light edges. And if you don't care which of these two cases it is, you could take absolute values of this output matrix.

> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-23.png">

This three by three filter we've seen allows you to detect vertical edges. So maybe it should not surprise you too much that this three by three filter will allow you to detect horizontal edges.

> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-24.png">

Different filters allow you to find vertical and horizontal edges. The three by three vertical edge detection filter we've used is just one possible choice. Historically, in the computer vision literature, there was a fair amount of debate about what is the best set of numbers to use.

```
Sobel filter:           Scharr filter:

1, 0, -1                 3, 0, -3
2, 0, -2                10, 0, -10
1, 0, -1                 3, 0, -3
```

The advantage of these filters is there is more weight to the central row, the central pixel, and this makes it maybe a little bit more robust.

You also have Sobel and Scharr filter for horizontal edge detection by flipping rgem 90 degrees,

And so by just letting all of these numbers be parameters and learning them automatically from data, we find that neural networks can actually learn low level features, can learn features such as edges, even more robustly than computer vision researchers are generally able to code up these things by hand.

```
w1, w2, w3
w4, w5, w6
w7, w8, w9
```
But underlying all these computations is still this convolution operation

> <img src="./images/w01-03-More_Edge_Detection/img_2023-04-01_09-46-26.png">

## Padding

In order to build deep neural networks one modification to the basic convolutional operation that you need to really use is padding.

Convolition filter change the size of your image.

|image size|convolution size|after convolution|
|--|--|--|
|6 x 6|3 x 3|4 x 4|
|n x n|f x f|(n-f+1) x (n-f+1)|


The two downsides of a convolutional operator :
- your image shrinks (after a hundred layers you end up with a very small image)
- you're throwing away a lot of the information near the edge of the image

To solve theses problem, we use padding

> <img src="./images/w01-04-Padding/padding.png">

|image size|convolution size|padding size|after convolution|
|--|--|--|--|
|6 x 6|3 x 3||4 x 4|
|n x n|f x f||(n-f+1) x (n-f+1)|
|n x n|f x f|p|(n+2p-f+1) x (n+2p-f+1)|
|6 x 6|3 x 3|1|6 x 6|

(padding with 2 pixel, p = 2, is also possible)

> <img src="./images/w01-04-Padding/img_2023-04-01_09-46-39.png">

 In terms of how much to pad, two common choices:
 - **Valid** convolutions without padding
 - **Same** convolutions with zero-padding (output size is the same as the input size,)

In computer vision f (size of the filter) is usually odd. Some of the reasons is that its have a center value.

> <img src="./images/w01-04-Padding/img_2023-04-01_09-46-41.png">

## Strided Convolutions

Stride convolutions is another piece of the basic building block of convolutions as using convolution neural networks.
The filter must lie entirely within the image or the image plus the padding region.

> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-46-55.png">

Stride convolution divide the size of the image by 2.

Output size after convolution: floor((n+2p-f)/s+1) x floor((n+2p-f)/s+1)
- after convolution with padding : ```(n+2p-f+1)```
- after convolution with padding and strided : ```floor(1 + (n+2p-f)/s) ```

> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-46-58.png">

In math textbooks the conv operation is **flipping** the filter before using it both horizontaly and verticaly.

The way we've defined the convolution operation in these videos is that we've skipped this mirroring operation.

It turns out that in signal processing or in certain branches of mathematics, doing the flipping in the definition of convolution causes convolution operator to enjoy associativity. This is nice for some signal processing applications. But for deep neural networks, it really doesn't matter, and so omitting this double mirroring operation just simplifies the code.

By convention, in machine learning, we usually do not bother with this flipping operation. Technically this operation is maybe better called **cross-correlation**. But most of the deep learning literature just causes the convolution operator.

> <img src="./images/w01-05-Strided_Convolutions/img_2023-04-01_09-47-00.png">

## Convolutions Over Volume

We see how convolution works with 2D images, now lets see if we want to convolve 3D images (RGB image)

Filter has 3 dimensions : ```height x width x channel``` , channel dimension corresponds to the red, green, and blue channels.

In term of dimensions, we have ```6x6x6 * 3x3x3 = 4x4 ```

> <img src="./images/w01-06-Convolutions_Over_Volume/img_2023-04-01_09-47-13.png">

Last crucial idea  for building convolutional neural networks : we don't just wanted to detect vertical edges, but also horizontal edges, maybe 45 degree edges, maybe 70 degree edges as well

In other words, what if you want to use multiple filters at the same time.

> <img src="./images/w01-06-Convolutions_Over_Volume/img_2023-04-01_09-47-15.png">

## One Layer of a Convolutional Network

With 2 filters, we have :

|Step|Description|Size (2 filters)|size (10 filters)|NN similarity|
|--|--|--|--|--|
|step 0|Input image|6x6x3|6x6x3|$a^{[0]}$|
|step 1|2 filters|2 (3x3x3)|10 (3x3x3)|$W^{[1]}$|
|step 2|2 images after filters|4 x 4 x 2|4 x 4 x 10|$W^{[1]}.a^{[1]}$|
|step 3|2 additional real b1, b2|4 x 4 x 2|4 x 4 x 10|$z^{[1]} = W^{[1]}.a^{[1]} + b^{[1]}$|
|step 4|Apply ReLU on the 2 4 x 4 images|4 x 4 x 2|4 x 4 x 10|$a^{[1]} = g(z^{[1]})$|

> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-26.png">

Detail of number of paremeters for 10 (3 x 3 x 3) filters :
- 3 * 3 * 3 = 27
- bias b = 1
- total = 28 for one filter
- total = 280 for 10 filters

Notice one nice thing about this, is that no matter how big the input image is, the input image could be 1,000 by 1,000 or 5,000 by 5,000, but the number of parameters you have still remains fixed as 280. And you can use these ten filters to detect features, vertical edges, horizontal edges maybe other features anywhere even in a very, very large image is just a very small number of parameters.

> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-27.png">

Summary on notation :
- $f^{[l]}$ : filter size
- $p^{[l]}$ : padding (default is zero)
- $s^{[l]}$ : stride
- Input $n_H^{[l-1]}$ x $n_W^{[l-1]}$ x  $n_c^{[l-1]}$ with:
    - $n_H^{[l-1]}$ : input height,
    - $n_W^{[l-1]}$ : input width,
    - $n_c^{[l-1]}$ : number of channels
- Output $n_H^{[l]}$ x $n_W^{[l]}$ x  $n_c^{[l]}$ with:
    - $n_H^{[l]}$ : input height,
    - $n_W^{[l]}$ : input width,
    - $n_c^{[l]}$ : number of channels



|Description|Equation|
|--|--|
|Number of channnels of the output = number of filters|#filters = $n_c^{[l]}$|
|The depth of each filter need to correspond to the number of channel of input|filer -> $f^{[l]}$  x $f^{[l]}$  x $n_c^{[l-1]}$|
|Activation dimensions for one example are the same as the output dimensions|$a^{[l]}$ -> $n_H^{[l]}$ x $n_W^{[l]}$ x  $n_c^{[l]}$$|
|Activation dimensions for m examples dimensions|$A^{[l]}$ -> $m$ x $n_H^{[l]}$ x $n_W^{[l]}$ x  $n_c^{[l]}$|
|Bias dimensions is equals to number of filters| $b^{[l]}$ -> $n_c^{[l]}$|



> <img src="./images/w01-07-One_Layer_of_a_Convolutional_Network/img_2023-04-01_09-47-29.png">

## Simple Convolutional Network Example

We saw the building blocks of a single layer, of a single convolution layer in the ConvNet.
Now let's go through a concrete example of a deep convolutional neural network.

Typical example of a ConvNet :

- Layer 0
    - For the sake of this example, we use a fairly small image (39 x 39 x 3)
- First Layer, convolution:
    - 10 filters 3x3
    - no padding (Valid convolution)
    - => output will be 37 x 37 x 10
- Second Layer, convolution:
    - 20 filters 5x5
    - stride = 2
    - no padding
    - => output will be 37 x 37 x 20
- Third Layer, convolution
    - 40 filters 5x5
    - stride = 2
    - no padding
    - => output will be 7 x 7 x 40
- Fourth Layer, Softmax
    - 1960 units in that layer (7 x 7 x 40)


A lot of the work in designing convolutional neural net is selecting **hyperparameters** like these, deciding what's the total size? What's the stride? What's the padding and how many filters are used?

> <img src="./images/w01-08-Simple_Convolutional_Network_Example/img_2023-04-01_09-47-41.png">

Typicals layers in convolutional network (convNet)
- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)

Fortunately pooling layers and fully connected layers are a bit simpler than convolutional layers to define.

> <img src="./images/w01-08-Simple_Convolutional_Network_Example/img_2023-04-01_09-47-43.png">

## Pooling Layers


One interesting property of max pooling is that it has a set of hyperparameters but it has no parameters to learn. There's actually nothing for gradient descent to learn.

In the following example, we use a Max pooling layer with :
- size, f = 2
- stride, s = 2
- paddin, p = 0

The intuition behind max pooling is that if you consider that having a large number in a region means a particular feature detected (a vertical edge, etc), then the max operation preserves these features detected anywhere in this filter.


> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-47-56.png">

In the following example, we use a Max pooling layer with :
- size, f = 2
- stride, s = 1

Formulas developed for convolutional layer for figuring out the output size also works for max pooling


> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-47-59.png">

There is another type of pooling that isn't used very often, called average pooling.

It's exactly the same mechanism than max pooling, replacing max operations within each filter by the average opeartion.

> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-48-01.png">

To summarize :
- pooling layer has some hyperparameters : fliter size, stride, opeartion (maximum vs average)
- no parameter to learn (no gradient descent needed)
- when you do max pooling, usually, you do not use any padding
- formulas for dimension of max pooling output is the same as the one of convolutional layer

> <img src="./images/w01-09-Pooling_Layers/img_2023-04-01_09-48-03.png">

## CNN Example

Let's look at an example af network to detect handwritten digit numbers in a 32 x 32 x 3 RGB image

This example is inspired of one of the classic neural networks called LeNet-5 created by Yann LeCun many years ago.

| layer |Descrription| activation shape | activation size | # parameters |
|---|---|---|---|---|
| Input || (32,32,3) | 3072 | 0 |
| CONV1 |6 filters f=5, s=1| (28,28,6) | 4'704 | 456 `=(5*5*3+1)*6` |
| POOL1 |Max pooling f=2, s=2| (14,14,6) | 1'176 | 0 |
| CONV2 |16 filters f=5, s=1 | (10,10,16) | 1'600 | 2'416 `=(5*5*6+1)*16` |
| POOL2 |Max pooling, f=2, s=2| (5,5,16) | 400 | 0 |
| Flatten |Flaten 5 x 5 x 16 |400 x 1||
| FC3 |120 units standard layer| (120,1)|  120 | 48'120 `=400*120+120` |
| FC4 |84 units standard layer| (84,1) | 84 | 10'164 `=120*84+84` |
| softmax |softmax 10 outputs| (10,1) | 10 | 850 `=84*10+10` |


In the literature of a ConvNet there are two conventions which are inside the inconsistent about what you call a layer. 
- CONV and POOL are considered as 2 layers
- CONV + POOL are considered as 1 single layer

Because the pooling layer has no weights, has no parameters, only a few hyper parameters we could consider POOL1 is part of Layer 1

FC3 is called **fully connected** because each of the 400 units is connected to each of the 120 units here (densely connected).

> <img src="./images/w01-10-CNN_Example/img_2023-04-01_09-48-29.png">

There a lot of hyper parameters. Maybe one common guideline is to actually not try to invent your own settings of hyper parameters, but to look in the literature to see what hyper parameters you work for others. And to just choose an architecture that has worked well for someone else, and there's a chance that will work for your application as well. 

Usually, as you go deeper in the network : 
- the height and width will decrease, 
- whereas the number of channels will increase.

<!--
> <img src="./images/w01-10-CNN_Example/img_2023-04-01_09-48-31.png">
-->

## Why Convolutions?

If we compare a standard neural network (fully connected) and convolutional network we have, fo a **small image** (32 x 32 x 3) :
- 14 millions of parameters for a standard neural network (NN)
- 156 parameters for a CNN

> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-46.png">

ConvNet has few parameters for two reasons:
- parameter sharing : a 3x3 filter for detecting vertical edges us useful for all the pixels in the images, the same filter, idenpendently of the place in the image
- sparse connections : the '0'  circled in green in the slide below depends only of the 9 (3x3) features in green in the image (right side), so 9 out of these 36 (6x6) input features. 

> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-48.png">

Let's put it all together and see how you can train one of these networks.
We choose a convolutional neural network structure:
- insert the image 
- have a neural convolutional and pooling layers 
- some fully connected layers 
- followed by a softmax output

> <img src="./images/w01-11-Why_Convolutions/img_2023-04-01_09-48-49.png">

# Heroes of Deep Learning (Optional)

## Yann LeCun Interview