---
layout: page
title: W2 - Neural networks Basics
permalink: /c1-neural-networks-and-deep-learning/week2/
parent: "C1 - Neural Networks and Deep Learning"
---

# Neural networks Basics
{: .no_toc}

Set up a machine learning problem with a neural network mindset and use vectorization to speed up your models.

Learning Objectives
- Build a logistic regression model structured as a shallow neural network
- Build the general architecture of a learning algorithm, including parameter initialization, cost function and gradient calculation, and optimization implemetation (gradient descent)
Implement computationally efficient and highly vectorized versions of models
- Compute derivatives for logistic regression, using a backpropagation mindset
- Use Numpy functions and Numpy matrix/vector operations
- Work with iPython Notebooks
- Implement vectorization across multiple training examples
- Explain the concept of broadcasting


<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Logistic Regression as a Neural Network

## Binary Classification

Logistic regression is an algorithm for binary classification.
Here's an example of a binary classification problem : image contains a cat (output = 1) or not (output = 0), with an image of 64 pixels x 64 pixels 

> <img src="./images/w02-01-w02-01-Binary_Classification/img_2023-03-11_12-51-22.png">

- A single training example is represented by a pair, (x,y) where 
    - x is an $n_x$-dimensional feature vector 
    - y, the label, is either 0 or 1
- Training example is $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ... (x^{(m)}, y^{(m)})$
- $m$ or $m_{train}$ the number of train examples
- $m_{test}$ the number of test examples

> <img src="./images/w02-01-w02-01-Binary_Classification/img_2023-03-11_12-51-25.png">

Finally, to output all of the training examples into a more compact notation, we're going to define a matrix, X with :
- $m$ columns (number of train examples)
- $n_x$ rows, where $n_x$ is the dimemsion of the input feature x

In python $Y.shape=(n_x,m)$

> <img src="./images/w02-01-w02-01-Binary_Classification/img_2023-03-11_14-26-29.png">

Notice that in other causes, you might see the matrix capital X defined by stacking up the train examples in rows, X1 transpose down to Xm transpose. Implementing neural networks using this first convention makes the implementation much easier.

Concenring label we also use matrix notation. The dimension of the matrix is (1 x m), in python $Y.shape = (1,m)$

> <img src="./images/w02-01-w02-01-Binary_Classification/img_2023-03-11_14-38-07.png">


## Logistic Regression

 > <img src="./images/w02-02-Logistic_Regression/img_2023-03-11_14-47-23.png">


$\hat{y}$ (y hat) is the prediction of y, is the probability of of y=1, given the input x
> <img src="./images/w02-02-Logistic_Regression/img_2023-03-11_14-52-53.png">

With w ($n_x$ dimension vector) and b (real number) as parameter, $\hat{y} = w^T.x + b$, with $w^T$ the w transpose (column instead of line for matrix multiplication compataibility)

> <img src="./images/w02-02-Logistic_Regression/img_2023-03-11_14-59-28.png">

This is linear regression, that is not correct for binary classification (0 < y < 1). That's why we use the sigmoid function

> <img src="./images/w02-02-Logistic_Regression/img_2023-03-11_15-04-02.png">

When we programmed neural networks, we'll usually keep the parameter W and parameter B separate, but there is another convention in which you merge w and b, introducing an extra feature $x_0=1$

> <img src="./images/w02-02-Logistic_Regression/img_2023-03-11_15-03-05.png">

## Logistic Regression Cost Function

> <img src="./images/w02-03-Logistic_Regression_Cost_Function/img_2023-03-11_15-08-12.png">

Logitic regression model

> <img src="./images/w02-03-Logistic_Regression_Cost_Function/img_2023-03-11_15-11-46.png">

With **loss function** or **error function** we can use to measure how well algorithm is performant.
The following **square error** function doesn't fit for logistic regression, because it's not convex 

> <img src="./images/w02-03-Logistic_Regression_Cost_Function/img_2023-03-11_15-15-06.png">

That's why we introduce the following loss function, called **cross entropy** 

> <img src="./images/w02-03-Logistic_Regression_Cost_Function/img_2023-03-11_15-19-11.png">

Loss function is defined to a single train example. **Cost function** is for the whole set of training example

> <img src="./images/w02-03-Logistic_Regression_Cost_Function/img_2023-03-11_15-27-32.png">

## Gradient Descent

https://github.com/mauvaisetroupe/machine-learning-specialization-coursera/blob/main/c1-supervised-ML-regression-and-classification/week1/README.md#gradient-descent

> <img src="./images/w02-04-Gradient_Descent/img_2023-03-11_15-31-29.png">

> <img src="./images/w02-04-Gradient_Descent/img_2023-03-11_15-31-32.png">

Inthe code, by convention we use **dw** and **db** 
> <img src="./images/w02-04-Gradient_Descent/img_2023-03-11_15-40-28.png">


## Derivatives

> <img src="./images/w02-05-Derivatives/img_2023-03-11_15-42-32.png">

## More Derivative Examples

> <img src="./images/w02-06-More_Derivative_Examples/img_2023-03-11_15-42-40.png">

> <img src="./images/w02-06-More_Derivative_Examples/img_2023-03-11_15-42-41.png">

## Computation Graph

https://github.com/mauvaisetroupe/machine-learning-specialization-coursera/blob/c1e3ee9a248c4dfa2c129fc1d5bd7d5b64b71f78/c2-advanced-learning-algorithms/week2/README.md#computation-graph-optional

> <img src="./images/w02-07-Computation_Graph/img_2023-03-11_15-47-28.png">


## Derivatives with a Computation Graph

https://github.com/mauvaisetroupe/machine-learning-specialization-coursera/blob/c1e3ee9a248c4dfa2c129fc1d5bd7d5b64b71f78/c2-advanced-learning-algorithms/week2/README.md#computation-graph-optional

<!--
> <img src="./images/w02-08-Derivatives_with_a_Computation_Graph/img_2023-03-11_15-49-21.png">
> <img src="./images/w02-08-Derivatives_with_a_Computation_Graph/img_2023-03-11_15-49-25.png">
-->

> <img src="./images/w02-08-Derivatives_with_a_Computation_Graph/img_2023-03-12_08-53-44.png">


Python convention, dJ/da is denoted da, dJ/dv is denoted dv

> <img src="./images/w02-08-Derivatives_with_a_Computation_Graph/img_2023-03-11_16-10-43.png">

In summary :

> <img src="./images/w02-08-Derivatives_with_a_Computation_Graph/img_2023-06-25_07-03-26.png">

## Logistic Regression Gradient Descent

**This demo don't use backward propagation** (adding a epsilon and check how much computed value increase), but use calculus derivatives, especially chain derivative rules.

> <img src="./images/w02-09-Logistic_Regression_Gradient_Descent/img_2023-03-11_16-18-28.png">


<!-- > <img src="./images/w02-09-Logistic_Regression_Gradient_Descent/img_2023-03-11_16-18-31.png"> -->

We calculate the partial derivatives and apply chain derivatives rules

> <img src="./images/w02-09-Logistic_Regression_Gradient_Descent/img_2023-03-11_17-34-10.png">


$dz = a - y$

$dw_1 = x_1 . dz,  \quad   dw_2 = x_2 . dz$

If we put derivatives on the computation graph (summary)


> <img src="./images/w02-09-Logistic_Regression_Gradient_Descent/img_2023-06-25_07-32-47.png">


## Gradient Descent on m Examples

> <img src="./images/w02-10-Gradient_Descent_on_m_Examples/img_2023-03-11_17-36-35.png">

One single step of gradient descent, with 2 loops (one for the training example, and another one for the features). This will be avoided by vectorization

Remember :
$dz = a - y, \quad  dw_1 = x_1 . dz, \quad  dw_2 = x_2 . dz$

> <img src="./images/w02-10-Gradient_Descent_on_m_Examples/img_2023-03-11_17-36-37.png">

# Python and Vectorization

## Vectorization

Vectorization is basically the art of getting rid of explicit for loops in your code. 

> <img src="./images/w02-11-Vectorization/img_2023-03-11_17-48-38.png">

Python code

> <img src="./images/w02-11-Vectorization/img_2023-03-12_09-02-43.png">

Performance result

> <img src="./images/w02-11-Vectorization/img_2023-03-12_09-03-04.png">

## More Vectorization Examples

Example 2 : Matrix x vector

> <img src="./images/w02-12-More_Vectorization_Examples/img_2023-03-11_17-50-36.png">

Example 2 : apply operation (exponential, log, ...) on every element of this vector v

> <img src="./images/w02-12-More_Vectorization_Examples/img_2023-03-11_17-50-39.png">

> <img src="./images/w02-12-More_Vectorization_Examples/img_2023-03-11_17-50-48.png">

## Vectorizing Logistic Regression

> <img src="./images/w02-13-Vectorizing_Logistic_Regression/img_2023-03-12_10-05-05.png">

## Vectorizing Logistic Regression's Gradient Output

> <img src="./images/w02-14-Vectorizing_Logistic_Regression_Gradient_Output/img_2023-03-12_10-07-53.png">

To implement multiple iterations as a gradient descent, we still need a for loop over the number of iterations. 

> <img src="./images/w02-14-Vectorizing_Logistic_Regression_Gradient_Output/img_2023-03-12_10-07-56.png">


## Broadcasting in Pyhton

Broadcasting is an operation of matching the dimensions of differently shaped arrays in order to be able to perform further operations on those arrays (eg per-element arithmetic).

> <img src="./images/w02-15-Broadcasting_in_Python/img_2023-03-12_16-05-06.png">


## A Note on Python/Numpy Vectors

- it's a strength because they create expressivity of the language. A great flexibility of the language lets you get a lot done even with just a single line of code. 
The ability of python to allow you to use broadcasting operations and more generally, the great flexibility of the python numpy program language is, both a strength as well as a weakness of the programming language: 
- cut also a weakness because broadcasting and flexibility, sometimes can introduce very subtle bugs. 

> <img src="./images/w02-16-A_Note_on_Python_Numpy_Vectors/img_2023-03-12_16-50-17.png">

## Explanation of Logistic Regression Cost Function (Optional)

> <img src="./images/w02-18-Explanation_of_Logistic_Regression_Cost_Function/img_2023-03-12_18-51-33.png">

- `log(p(y|x))` = - Loss Function
- Minimizing the loss corresponds with maximizing `log(p(y|x))`

> <img src="./images/w02-18-Explanation_of_Logistic_Regression_Cost_Function/img_2023-03-12_18-12-06.png">

> <img src="./images/w02-18-Explanation_of_Logistic_Regression_Cost_Function/img_2023-03-12_18-53-06.png">

# Heroes of Deep Learning

## Pieter Abbeel Interview

