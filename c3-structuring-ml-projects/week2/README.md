# ML Strategy

Develop time-saving error analysis procedures to evaluate the most worthwhile options to pursue and gain intuition for how to split your data and when to use multi-task, transfer, and end-to-end deep learning.

Learning Objectives
- Describe multi-task learning and transfer learning
- Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

# Error Analysis

##  Carrying Out Error Analysis

If you're trying to get a learning algorithm to do a task that humans can do. And if your learning algorithm is not yet at the performance of a human. Then manually examining mistakes that your algorithm is making, can give you insights into what to do next. This process is called error analysis.

Error analysis approach:
- get about, say 100 mislabeled dev set examples, 
- Count up how many are dogs
    - if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 9.5% (called ceiling), which can be too little.
    - if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on that

> <img src="./images/w02-01-Carrying_Out_Error_Analysis/img_2023-03-31_11-24-33.png">

Sometimes, you can evaluate multiple error analysis ideas in parallel and choose the best idea. 
Create a spreadsheet to do that and decide

For example, clearly in this example, a lot of the mistakes were made on blurry images, and quite a lot on were made on great cat images

> <img src="./images/w02-01-Carrying_Out_Error_Analysis/img_2023-03-31_11-24-35.png">

##  Cleaning Up Incorrectly Labeled Data

Deep learning algorithms are quite robust to ***random*** errors in the training set

> <img src="./images/w02-02-Cleaning_Up_Incorrectly_Labeled_Data/img_2023-03-31_11-25-03.png">

My advice is, if it makes a significant difference to your ability to evaluate algorithms on your dev set, then go ahead and spend the time to fix incorrect labels. But if it doesn't make a significant difference to your ability to use the dev set to evaluate classifiers, then it might not be the best use of your time

> <img src="./images/w02-02-Cleaning_Up_Incorrectly_Labeled_Data/img_2023-03-31_11-25-06.png">

Guidelines :

> <img src="./images/w02-02-Cleaning_Up_Incorrectly_Labeled_Data/img_2023-03-31_11-25-08.png">

##  Build your First System Quickly, then Iterate


See pdf notes provided by Courseara : [1-Build_System_Quickly.pdf](./1-Build_System_Quickly.pdf)

Depending on the area of application, the guideline below will help you prioritize when you build your
system.

Guideline
1. Set up development/ test set and metrics
    - Set up a target
2. Build an initial system quickly
    - Train training set quickly: Fit the parameters
    - Development set: Tune the parameters
    - Test set: Assess the performance
3. Use Bias/Variance analysis & Error analysis to prioritize next steps

> <img src="./images/w02-03-Build_your_First_System_Quickly_then_Iterate/img_2023-03-31_11-25-27.png">



# Mismatched Training and Dev/Test Set

##  Training and Testing on Different Distributions

See pdf notes provided by Courseara : [2-Training_and_testing_on_different_distributions.pdf](./2-Training_and_testing_on_different_distributions.pdf)

So in this video, we see examples of when allowing your training set data to come from a different distribution than your dev and test set allows you to have much more training data (sometime you cannot avoid working mixing real data and data fron purchase or other...)

First option is to shuffle the data tand extract randomly training and dev/test sets.
- (+) All the sets now come from the same distribution.
- (-) real world distribution that was in the dev/test sets will occur less in the new dev/test sets and that might be not what you want to achieve.

Second option: take some of the dev/test  examples and add them to the training set.
- (+) the distribution important for your application is your targey
- (-) different distributions

First example with cat classifier

> <img src="./images/w02-04-Training_and_Testing_on_Different_Distributions/img_2023-03-31_11-25-47.png">

Second exemple with rearview mirror

> <img src="./images/w02-04-Training_and_Testing_on_Different_Distributions/img_2023-03-31_11-25-50.png">

##  Bias and Variance with Mismatched Data Distributions

See pdf notes provided by Courseara : [3-Bias_and_variance_with_mismatched_data_distributions.pdf](./3-Bias_and_variance_with_mismatched_data_distributions.pdf)

> <img src="./images/w02-05-Bias_and_Variance_with_Mismatched_Data_Distributions/img_2023-03-31_11-26-02.png">
> <img src="./images/w02-05-Bias_and_Variance_with_Mismatched_Data_Distributions/img_2023-03-31_11-26-04.png">
> <img src="./images/w02-05-Bias_and_Variance_with_Mismatched_Data_Distributions/img_2023-03-31_11-26-06.png">

##  Addressing Data Mismatch

See pdf notes provided by Courseara : [4-Adressing_data_mismatch.pdf](./4-Adressing_data_mismatch.pdf)

> <img src="./images/w02-06-Addressing_Data_Mismatch/img_2023-03-31_11-26-17.png">
> <img src="./images/w02-06-Addressing_Data_Mismatch/img_2023-03-31_11-26-19.png">
> <img src="./images/w02-06-Addressing_Data_Mismatch/img_2023-03-31_11-26-21.png">

# Learning from Multiple Tasks


##  Transfer Learning

See pdf notes provided by Courseara : [5-Transfer_Learning.pdf](./5-Transfer_Learning.pdf)

> <img src="./images/w02-07-Transfer_Learning/img_2023-03-31_11-26-32.png">
> <img src="./images/w02-07-Transfer_Learning/img_2023-03-31_11-26-33.png">

##  Multi-task Learning

See pdf notes provided by Courseara : [6-Multi_Task_Learning.pdf](./6-Multi_Task_Learning.pdf)

> <img src="./images/w02-08-Multi-task_Learning/img_2023-03-31_11-26-45.png">
> <img src="./images/w02-08-Multi-task_Learning/img_2023-03-31_11-26-47.png">
> <img src="./images/w02-08-Multi-task_Learning/img_2023-03-31_11-26-48.png">


# End-to-end Deep Learning

##  What is End-to-end Deep Learning?

See pdf notes provided by Courseara : [7-What_is_end_to_end_deep_learning.pdf](./7-What_is_end_to_end_deep_learning.pdf)

> <img src="./images/w02-09-What_is_End-to-end_Deep_Learning?/img_2023-03-31_11-26-59.png">
> <img src="./images/w02-09-What_is_End-to-end_Deep_Learning?/img_2023-03-31_11-27-01.png">
> <img src="./images/w02-09-What_is_End-to-end_Deep_Learning?/img_2023-03-31_11-27-03.png">

##  Whether to use End-to-end Deep Learning

See pdf notes provided by Courseara : [8-Whether_to_use_end_to_end_deep_learning.pdf](./8-Whether_to_use_end_to_end_deep_learning.pdf)

> <img src="./images/w02-10-Whether_to_use_End-to-end_Deep_Learning/img_2023-03-31_11-27-14.png">
> <img src="./images/w02-10-Whether_to_use_End-to-end_Deep_Learning/img_2023-03-31_11-27-16.png">

# Heroes of Deep Learning (Optional)

##  Ruslan Salakhutdinov Interview
