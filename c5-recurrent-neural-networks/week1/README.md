# Recurrent Neural Networks

Discover recurrent neural networks, a type of model that performs extremely well on temporal data, and several of its variants, including LSTMs, GRUs and Bidirectional RNNs,


Learning Objectives
- Define notation for building sequence models
- Describe the architecture of a basic RNN
- Identify the main components of an LSTM
- Implement backpropagation through time for a basic RNN and an LSTM
- Give examples of several types of RNN
- Build a character-level text generation model using an RNN
- Store text data for processing using an RNN
- Sample novel sequences in an RNN
- Explain the vanishing/exploding gradient problem in RNNs
- Apply gradient clipping as a solution for exploding gradients
- Describe the architecture of a GRU
- Use a bidirectional RNN to take information from two points of a sequence
- Stack multiple RNNs on top of each other to create a deep RNN
- Use the flexible Functional API to create complex models
- Generate your own jazz music with deep learning
- Apply an LSTM to a music generation task

# Recurrent Neural Networks

## Why Sequence Models?

Examples of sequence data

|Example|Type|Input|Output|
|-|-|-|-|
|Speech recognition|wave sequence|text sequence|
|Music generation|nothing or an integer with the type of music|wave sequence|
|Sentiment classification|text sequence|integer rating from one to five|
|DNA sequence analysis|DNA sequence corresponds to a protein|DNA Labels|
|Machine translation|text sequence (in one language)|text sequence (in other language)|
|Video activity recognition|video frames|label (activity)|
|Name entity recognition|text sequence|label sequence|


> <img src="./images/w01-01-why_sequence_models/img_2023-04-25_20-50-11.png">

## Notation

We want to find people name in the sentence. This problem is called entity recognition and is used by search engine to index people mentioned in the news articles in the last 24 hours.
We want the output where (this output representation is not the best one, just for ullustration) :
- 1 means its a name,
- 0 means its not a name

|||||||||||
|-|-|-|-|-|-|-|-|-|-|
|input|x<sup><1></sup>|x<sup><2></sup>|x<sup><3></sup>|x<sup><4></sup>|x<sup><5></sup>|x<sup><6></sup>|x<sup><7></sup>|x<sup><8></sup>|x<sup><9></sup>|
||Harry|Potter|and|Hermione|Granger|invented|a|new|spell|
|output|y<sup><1></sup>|y<sup><2></sup>|y<sup><3></sup>|y<sup><4></sup>|y<sup><5></sup>|y<sup><6></sup>|y<sup><7></sup>|y<sup><8></sup>|y<sup><9></sup>|
||1|1|0|1|1|0|0|0|0|

Notation:
- x<sup><t></sup> is t-th word
- y<sup><t></sup> is the output of the t-th word
- T<sub>x</sub> is the size of the input sequence
- T<sub>y</sub> is the size of the output sequence.

We introduce the concept of i-th example :
- x<sup>(i)\<t></sup> is t-th word of the i-th input example,
- T<sup>x(i)</sup> is the length of the i-th example.


> <img src="./images/w01-02-notation/img_2023-04-25_20-51-03.png">

- Representing words: we build a *dictionary* that is a vocabulary list that contains all the words in our training sets
- Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
- We use *one-Hot representation* for a specific word : vector with 1 in position of the word in the dictionary and 0 everywhere else
- We add a token in the vocabulary with name <UNK> (unknown text)

> <img src="./images/w01-02-notation/img_2023-04-25_20-51-05.png">

<!--
> <img src="./images/w01-02-notation/img_2023-04-25_20-51-07.png">
-->

## Recurrent Neural Network Model

Why not to use a standard network ?
- Inputs, outputs can be different lengths in different examples
- Neural network architecture doesn't share features learned across different positions of text ("Harry Potter" is a name even if found in another part, at a different position in the text)

> <img src="./images/w01-03-recurrent_neural_network_model/img_2023-04-25_20-51-15.png">
> <img src="./images/w01-03-recurrent_neural_network_model/img_2023-04-25_20-51-17.png">
> <img src="./images/w01-03-recurrent_neural_network_model/img_2023-04-25_20-51-18.png">
> <img src="./images/w01-03-recurrent_neural_network_model/img_2023-04-25_20-51-20.png">

## Backpropagation Through Time

> <img src="./images/w01-04-backpropagation_through_time/img_2023-04-25_20-51-27.png">
> <img src="./images/w01-04-backpropagation_through_time/img_2023-04-25_20-51-29.png">

## Different Types of RNNs

> <img src="./images/w01-05-different_types_of_RNNS/img_2023-04-25_20-51-42.png">
> <img src="./images/w01-05-different_types_of_RNNS/img_2023-04-25_20-51-44.png">
> <img src="./images/w01-05-different_types_of_RNNS/img_2023-04-25_20-51-46.png">
> <img src="./images/w01-05-different_types_of_RNNS/img_2023-04-25_20-51-47.png">

## Language Model and Sequence Generation

> <img src="./images/w01-06-language_model_and_sequence_generation/img_2023-04-25_20-51-58.png">
> <img src="./images/w01-06-language_model_and_sequence_generation/img_2023-04-25_20-51-59.png">
> <img src="./images/w01-06-language_model_and_sequence_generation/img_2023-04-25_20-52-01.png">

## Sampling Novel Sequences

> <img src="./images/w01-07-sampling_novel_sequences/img_2023-04-25_20-52-15.png">
> <img src="./images/w01-07-sampling_novel_sequences/img_2023-04-25_20-52-16.png">
> <img src="./images/w01-07-sampling_novel_sequences/img_2023-04-25_20-52-18.png">

## Vanishing Gradients with RNNs

> <img src="./images/w01-08-vanishing_gradients_with_RNNS/img_2023-04-25_20-52-28.png">

## Gated Recurrent Unit (GRU)

> <img src="./images/w01-09-gated_recurrent_unit_GRU/img_2023-04-25_20-52-40.png">
> <img src="./images/w01-09-gated_recurrent_unit_GRU/img_2023-04-25_20-52-42.png">
> <img src="./images/w01-09-gated_recurrent_unit_GRU/img_2023-04-25_20-52-43.png">

## Long Short Term Memory (LSTM)

> <img src="./images/w01-10-long_short_term_memory_LSTM/img_2023-04-25_20-52-59.png">
> <img src="./images/w01-10-long_short_term_memory_LSTM/img_2023-04-25_20-53-00.png">
> <img src="./images/w01-10-long_short_term_memory_LSTM/img_2023-04-25_20-53-02.png">

## Bidirectional RNN

> <img src="./images/w01-11-bidirectional_RNN/img_2023-04-25_20-53-10.png">
> <img src="./images/w01-11-bidirectional_RNN/img_2023-04-25_20-53-12.png">

## Deep RNNs

> <img src="./images/w01-12-deep_RNNS/img_2023-04-25_20-53-20.png">
