# ML Strategy

Streamline and optimize your ML production workflow by implementing strategic guidelines for goal-setting and applying human-level performance to help define key priorities.

Learning Objectives
- Explain why Machine Learning strategy is important
- Apply satisficing and optimizing metrics to set up your goal for ML projects
- Choose a correct train/dev/test split of your dataset
- Define human-level performance
- Use human-level performance to define key priorities in ML projects
- Take the correct ML Strategic decision based on observations of performances and dataset

# Introduction to ML Strategy

## Why ML Strategy

There is a lot of ideas for how to improve your deep learning system.


> <img src="./images/w01-01-why_ml_strategy/img_2023-03-28_21-13-36.png">

## Orthogonalization

- Each knob has a relatively interpretable function. Imagine a signle knob that would change everything at same time, should be almost impossible to tune the TV. In that context orthogonalization  refers to the fact that TV designer have designed the knobs to do only one thing
- Other example is about the 3 controls (steering wheel, acceleration, and braking) in a car with well indentified actions. But now imagine if someone build a car with a joystick, where one axis of the joystick controls 0.3 x your steering angle,- 0.8 x your speed. And you had a different control that controls 2 x the steering angle, + 0.9 x the speed of your car. In theory, by tuning these two knobs, you could get your car to steer at the angle and at the speed you want. But it's much harder than if you had just one single control for controlling the steering angle, and a separate, distinct set of controls for controlling the speed.
- Orthogonalization is having one dimension by knobs

> <img src="./images/w01-02-orthogonalization/img_2023-03-28_21-13-53.png">

<table>
        <thead>
                <tr>
                        <th>Chain of assumptions in ML</th>
                        <th>Possible tuning</th>
                </tr>
        </thead>
        <tbody>
                <tr>
                        <td>Fit training set well on cost function</td>
                        <td>- Bigger network<br />
                        - Better optimization algorithm like Adam</td>
                </tr>
                <tr>
                        <td>Fit dev set well on cost function</td>
                        <td>- Regularization<br />
                        - Bigger training set</td>
                </tr>
                <tr>
                        <td>Fit test set well on cost function</td>
                        <td>- bigger dev set</td>
                </tr>
                <tr>
                        <td>Performs well in real world</td>
                        <td>- change dev set or cost function<br /></td>
                </tr>
        </tbody>
</table>

> <img src="./images/w01-02-orthogonalization/img_2023-03-28_21-13-56.png">

# Setting Up your Goal

## Single Number Evaluation Metric

You'll find that your progress will be much faster if you have a single real number evaluation metric that lets you quickly tell if the new thing you just tried is working better or worse than your last idea.

In the following exemple, it's easier to work with F1-score mmore than working on Precision and Record (that could evlolve in a opposite directions)

| metric        | definition    |
| :-:           | :--           |
| Precision     | percentage of true positive in predicted positive |
| Recall        | percentage of true positive predicted in all real positive |
| F1 score      | harmonic mean of precision and recall |

> <img src="./images/w01-03-single_number_evaluation_metric/img_2023-03-28_21-14-10.png">

Other example. It's very difficult to look at these numbers and quickly decide if algorithm A or algorithm B is superior.  So what I recommend in this example is, in addition to tracking your performance in the four different geographies, to also compute the average. And assuming that average performance is a reasonable single real number evaluation metric, by computing the average, you can quickly tell that it looks like algorithm C has a lowest average error.


> <img src="./images/w01-03-single_number_evaluation_metric/img_2023-03-28_21-14-14.png">

## Satisficing and Optimizing Metric

First example (left column on the slide) : cat classifier :
1. accurancy is to optimize (get best value)
2. response time is to satisfy (< 1000 ms)

Second example is aout voice recognition and wakewords ("ok google"):
1. optimize accutacy
2. accepr one false positive by 24 hours


More generally :
1. Maximizing 1 value       # optimizing metric (one optimizing metric)
2. Satisfying N-1 values    # satisficing metric (N-1 satisficing metrics)

> <img src="./images/w01-04-satisficing_and_optimizing_metric/img_2023-03-28_21-14-32.png">

## Train/Dev/Test Distributions

Taking data for DEV in some region and data for test in other regions is a very bad idea because dev and test sets come from different distributions.

Define what target you want to aim at :
 - setting up the dev set, 
 - defining the single role number evaluation metric
The team can then innovate very quickly, try different ideas, run experiments and very quickly use the dev set and the metric to evaluate crossfires and try to pick the best one. 

> <img src="./images/w01-05-train_dev_test_distributions/img_2023-03-28_21-14-52.png">

3 months lost because of a bad data distribution  

> <img src="./images/w01-05-train_dev_test_distributions/img_2023-03-28_21-14-56.png">

Choose dev set and test set to reflect data you expect to get in the future and consider important to do well on.

> <img src="./images/w01-05-train_dev_test_distributions/img_2023-03-28_21-14-59.png">

## Size of the Dev and Test Sets

An old way of splitting the data was 
- 70% training, 30% test 
- or 60% training, 20% dev, 20% test

But in the modern machine learning era, we are now used to working with much larger data set sizes (1 million training examples), it might be quite reasonable to have 98% in the training set, and 1% dev, and 1% test.

> <img src="./images/w01-06-size_of_the_dev_and_test_sets/img_2023-03-28_21-15-13.png">

The guideline is, to set your test set to big enough to give high confidence in the overall performance of your system.

> <img src="./images/w01-06-size_of_the_dev_and_test_sets/img_2023-03-28_21-15-19.png">

## When to Change Dev/Test Sets and Metrics?

> <img src="./images/w01-07-when_to_change_dev_test_sets_and_metrics/img_2023-03-28_21-15-37.png">
> <img src="./images/w01-07-when_to_change_dev_test_sets_and_metrics/img_2023-03-28_21-15-39.png">
> <img src="./images/w01-07-when_to_change_dev_test_sets_and_metrics/img_2023-03-28_21-15-41.png">


# Comparing to Human-level Performance

## Why Human-level Performance?

> <img src="./images/w01-08-why_human-level_performance/img_2023-03-28_21-15-54.png">
> <img src="./images/w01-08-why_human-level_performance/img_2023-03-28_21-15-56.png">

## Avoidable Bias

> <img src="./images/w01-09-avoidable_bias/img_2023-03-28_21-16-08.png">
> <img src="./images/w01-09-avoidable_bias/img_2023-03-28_21-16-11.png">
> <img src="./images/w01-09-avoidable_bias/img_2023-03-28_21-16-13.png">

## Understanding Human-level Performance

> <img src="./images/w01-10-understanding_human-level_performance/img_2023-03-28_21-16-28.png">
> <img src="./images/w01-10-understanding_human-level_performance/img_2023-03-28_21-16-30.png">
> <img src="./images/w01-10-understanding_human-level_performance/img_2023-03-28_21-16-32.png">

## Surpassing Human-level Performance

> <img src="./images/w01-11-surpassing_human-level_performance/img_2023-03-28_21-16-46.png">
> <img src="./images/w01-11-surpassing_human-level_performance/img_2023-03-28_21-16-48.png">

## Improving your Model Performance

> <img src="./images/w01-12-improving_your_model_performance/img_2023-03-28_21-17-00.png">
> <img src="./images/w01-12-improving_your_model_performance/img_2023-03-28_21-17-02.png">


# Heroes of Deep Learning (Optional)

## Andrej Karpathy Interview