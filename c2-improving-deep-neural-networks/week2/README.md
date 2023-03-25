# Optimization Algorithms

Develop your deep learning toolbox by adding more advanced optimizations, random minibatching, and learning rate decay scheduling to speed up your models.

Learning Objectives
- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization
- Describe the benefits of learning rate decay and apply it to your optimization

# Optimization Algorithms

## Mini-batch Gradient Descent

Vectorization allows processing a large number of examples relatively quickly. However, even with vectorization, processing large datasets (5,000,000 samples), can still be slow, requiring processing the entire training set before taking one step of gradient descent.

Mini-batch gradient descent is a technique to process large datasets by splitting them into smaller batches, allowing the algorithm to start making progress before processing the entire training set. 

In the given example, with 5 million training samples, we can split the data into 5000 mini-batches with 1000 examples each. 

Notations:
- (i): the i-th training sample
- [l]: the l-th layer of the neural network
- {t}: the t-th mini batch

> <img src="./images/w02-01-mini-batch_gradient_descent/img_2023-03-25_16-17-57.png">

With mini-batch gradient descent, a single pass through the training set is one epoch, which in the above 5 million example, means 5000 gradient descent steps. 
- In Batch gradient descent we run the gradient descent on the whole dataset.
- While in Mini-Batch gradient descent we run the gradient descent on the mini datasets.

```
for t = 1:nb_batches
	cost, caches    = forward_propagation(X{t}, Y{t})
	gradients       = backward_propagation(X{t}, Y{t}, caches)
	update_parameters(gradients)
```
> <img src="./images/w02-01-mini-batch_gradient_descent/img_2023-03-25_16-17-59.png">

## Understanding Mini-batch Gradient Descent

> <img src="./images/w02-02-understanding_mini-batch_gradient_descent/img_2023-03-25_16-18-22.png">

> <img src="./images/w02-02-understanding_mini-batch_gradient_descent/img_2023-03-25_16-18-24.png">

> <img src="./images/w02-02-understanding_mini-batch_gradient_descent/img_2023-03-25_16-18-26.png">

## Exponentially Weighted Averages

> <img src="./images/w02-03-exponentially_weighted_averages/img_2023-03-25_16-18-42.png">

> <img src="./images/w02-03-exponentially_weighted_averages/img_2023-03-25_16-18-44.png">


## Understanding Exponentially Weighted Averages

> <img src="./images/w02-04-understanding_exponentially_weighted_averages/img_2023-03-25_16-19-11.png">

> <img src="./images/w02-04-understanding_exponentially_weighted_averages/img_2023-03-25_16-19-13.png">

> <img src="./images/w02-04-understanding_exponentially_weighted_averages/img_2023-03-25_16-19-15.png">

## Bias Correction in Exponentially Weighted Averages

> <img src="./images/w02-05-bias_correction_in_exponentially_weighted_averages/img_2023-03-25_16-19-29.png">


## Gradient Descent with Momentum

> <img src="./images/w02-06-gradient_descent_with_momentum/img_2023-03-25_16-19-43.png">

> <img src="./images/w02-06-gradient_descent_with_momentum/img_2023-03-25_16-19-45.png">


## RMSprop

> <img src="./images/w02-07-rmsprop/img_2023-03-25_16-20-03.png">

## Adam Optimization Algorithm

> <img src="./images/w02-08-adam_optimization_algorithm/img_2023-03-25_16-20-13.png">

> <img src="./images/w02-08-adam_optimization_algorithm/img_2023-03-25_16-20-15.png">


## Learning Rate Decay

> <img src="./images/w02-09-learning_rate_decay/img_2023-03-25_16-20-27.png">

> <img src="./images/w02-09-learning_rate_decay/img_2023-03-25_16-20-29.png">

> <img src="./images/w02-09-learning_rate_decay/img_2023-03-25_16-20-31.png">

# The problem of local optima

> <img src="./images/w02-10-the-problem-of-local-optima/img_2023-03-25_16-20-44.png">

> <img src="./images/w02-10-the-problem-of-local-optima/img_2023-03-25_16-20-46.png">


## Yuanqing Lin Interview


