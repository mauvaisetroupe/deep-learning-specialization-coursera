# Practical Aspects of Deep Learning

Discover and experiment with a variety of different initialization methods, apply L2 regularization and dropout to avoid model overfitting, then apply gradient checking to identify errors in a fraud detection model.

Learning Objectives
- Give examples of how different types of initializations can lead to different results
- Examine the importance of initialization in complex neural networks
- Explain the difference between train/dev/test sets
- Diagnose the bias and variance issues in your model
- Assess the right time and place for using regularization methods such as dropout or L2 regularization
- Explain Vanishing and Exploding gradients and how to deal with them
- Use gradient checking to verify the accuracy of your backpropagation implementation
- Apply zeros initialization, random initialization, and He initialization
- Apply regularization to a deep learning model


# Setting up your Machine Learning Application

##  Train / Dev / Test sets


When training a neural network, you have to make a lot of decisions, (how many layers, how many hidden units, ...).
In practice, applied machine learning is a highly iterative process, in which you often start with an idea and then you just have to code it up and try it, by running your code. 

Intuitions from one domain or from one application area often do not transfer to other application areas

> <img src="./images/w01-01-Train_Dev_Test_sets/img_2023-03-19_09-34-08.png">

Development set is  used to see which of many different models performs best. And then after having done this long enough, when you have a final model that you want to evaluate, you can take the best model you have found and evaluate it on your test set in order to get an unbiased estimate of how well your algorithm is doing. S

| Area | Range of data | Split|
|---|---|---|
|Previous area|100 - 100'000|60%-20%-20%|
|Big Data area|1'000'000|98%-1%-1%|

> <img src="./images/w01-01-Train_Dev_Test_sets/img_2023-03-19_09-34-10.png">

The rule of thumb I'd encourage you to follow, in this case, is to make sure that the dev and test sets come from the same distribution.

> <img src="./images/w01-01-Train_Dev_Test_sets/img_2023-03-19_09-34-13.png">


##  Bias / Variance

Bias and Variance is one of those concepts that's easily learned but difficult to master. 

> <img src="./images/w01-02-Bias_Variance/img_2023-03-19_09-49-15.png">

Assumimg that humans achieve 0% errors, 15% is not a good score (**Bayes error rate**)

> <img src="./images/w01-02-Bias_Variance/img_2023-03-19_09-49-17.png">

What does high bias and high variance look like? Example of classifier that is mostly linear, and therefore, underfits the data (we're drawing this is purple), but if somehow your classifier does some weird things, then it is actually overfitting parts of the data as well.

> <img src="./images/w01-02-Bias_Variance/img_2023-03-19_09-49-19.png">


##  Basic Recipe for Machine Learning

If your algorithm has a high bias, you can try 
- increasing the size of the neural network by adding more layers 
- increasing the size of the hidden units
- running it for a longer time 
- using different optimization algorithms. I

f your algorithm has a high variance, you can try 
- collecting more data 
- applying regularization techniques. 

It is recommended to try these methods iteratively until a low bias and low variance are achieved. In the past, there was a "Bias/variance tradeoff," but with the advent of deep learning, there are more options available to address this problem. 

Training a bigger neural network is a viable option to consider.

> <img src="./images/w01-03-Basic_Recipe_for_Machine_Learning/img_2023-03-19_09-55-29.png">

# Regularizing your Neural Network

##  Regularization

For logistic regression

- L2 regularization : $Loss = Error(Y - \widehat{Y}) +  \frac{\lambda}{2m}   \sum_1^n w_i^{2}$
- L1 regularization : $Loss = Error(Y - \widehat{Y}) +  \frac{\lambda}{2m}   \sum_1^n |w_i|$

Lambda is a reserved keyword in python (use lambd instead)

> <img src="./images/w01-04-Regularization/img_2023-03-19_09-56-34.png">

We introduce the frobenius norm  : $\sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{l-1}}(w_{ij}^{[l]})^2$
L2 regularization has an impact onthe calculation od dW
L2 regularization is sometimes also called **weight decay** because it's just like the ordinary gradient descent, where you update w by subtracting alpha, times the original gradient you got from backprop. But now you're also, you know, multiplying w by a factor little bit less than 1

> <img src="./images/w01-04-Regularization/img_2023-03-19_09-56-36.png">

##  Why Regularization Reduces Overfitting?

When lambda increases, the weights of matrices W tend to be set closer to zero.
As a result, this simplified neural network becomes smaller and almost like a logistic regression unit stacked multiple layers deep. The network moves from the overfitting case to the high bias case.

However, the intuition that many hidden units are completely zeroed out is not entirely accurate. Instead, all hidden units are still used, but their impact is significantly reduced.

> <img src="./images/w01-05-Why_Regularization_Reduces_Overfitting/img_2023-03-19_10-24-17.png">

When lambda is large, the weights of the network are penalized for being too large. smaller weights lead to smaller values of z. causing the TANH function to behave more linearly. This means that each layer of the network will behave more like linear regression, and the entire network will be essentially a linear function.

> <img src="./images/w01-05-Why_Regularization_Reduces_Overfitting/img_2023-03-19_10-24-19.png">

When using regularization in gradient descent, it is important to plot the cost function with the new regularization term, rather than just the old cost function without the regularization term (could have impact of the decreasing of the function)

##  Dropout Regularization

> <img src="./images/w01-06-Dropout_Regularization/img_2023-03-19_13-53-53.png">

> <img src="./images/w01-06-Dropout_Regularization/img_2023-03-19_13-53-55.png">

> <img src="./images/w01-06-Dropout_Regularization/img_2023-03-19_13-53-58.png">


##  Understanding Dropout

##  Other Regularization Methods



# Setting Up your Optimization Problem

##  Normalizing Inputs

##  Vanishing / Exploding Gradients

##  Weight Initialization for Deep Networks

##  Numerical Approximation of Gradients

##  Gradient Checking

##  Gradient Checking Implementation Notes


# Heroes of deep learning

##  Yoshua Bengio Interview