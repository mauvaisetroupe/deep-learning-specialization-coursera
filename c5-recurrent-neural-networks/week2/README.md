---
layout: page
title: "W2 - Natural Language Processing & Word Embeddings"
permalink: /c5-recurrent-neural-networks/week2/
parent: "C5 - Sequence Models"
---
# Natural Language Processing & Word Embeddings

Natural language processing with deep learning is a powerful combination. Using word vector representations and embedding layers, train recurrent neural networks with outstanding performance across a wide variety of applications, including sentiment analysis, named entity recognition and neural machine translation.

Learning Objectives
- Explain how word embeddings capture relationships between words
- Load pre-trained word vectors
- Measure similarity between word vectors using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
- Reduce bias in word embeddings
- Create an embedding layer in Keras with pre-trained word vectors
- Describe how negative sampling learns word vectors more efficiently than other methods
- Explain the advantages and disadvantages of the GloVe algorithm
- Build a sentiment classifier using word embeddings
- Build and train a more sophisticated classifier using an LSTM

# Introduction to Word Embeddings

## Word Representation

NLP (Natural Language Processing) has been revolutionized by deep learning. One of the key ideas is word embeddings, which is a way of representing words that let your algorithms automatically understand analogies like that, man is to woman, as king is to queen.

- [Previously](../week1/README.md#notation), we use one-hot representation and vocabulary to encode words.
- we use $$O_{5391}$$ to denote the one-hot vector with `1` in position `5391` (and `0` elseweher)
- One of the weaknesses of this representation is that it treats each word as a thing in itself, and that it does not allow an algorithm to find common senses
    - even if algorithm has learned `I want a glass of orange juice`, algorithm cannot complete `I want a glass of apple ...` because there is no specific proximity between apple and orange (disatnce is the same between any pair of vectors)

> <img src="./images/w02-02-word_representation/img_2023-05-02_07-56-25.png">

Won't it be nice if instead of a one-hot presentation we can instead learn a featurized representation with each of these words, we could learn a set of features and values for each of them
- We decide to have for example 300 features, so each word column is a 300-dimensional vector
- vetcor e5391 describe `man` word features vector (man is the 5391th word in vocabulary dictionary)
- with this representation, `Orange` and `apple` share lot of similar features

> <img src="./images/w02-02-word_representation/img_2023-05-02_07-56-28.png">

- With this simplified 2D vizualization (real visualization should have 300 axis), we can easily find categories
    - people
    - animals
    - living beings
    - ...
- One common algorithm for visualizing word representation is the t-SNE algorithm (t-distributed stochastic neighbor embedding)
- This representation is called Word embeddings
- The name comes because a word is "embedded" in a 300-D volume (represented with a 3-D volume in the slide)
- Word Embeddings is one of the important ideas in NLP

> <img src="./images/w02-02-word_representation/img_2023-05-02_07-56-30.png">

## Using Word Embeddings

Continuing with named entity recognition,
- with the sentence `Sally Johnson is an orange farmer`, you know that `Sally Johnson` is a person's name rather a company name because you know that `orange farmer` ìs person
- if you get the sentence `Robert Lin is an apple farmer` you model should find that `Robert Lin` as a name, as `apple` and `orange` have near representations.
- with this sentence `xxx is a durian cultivator`, the network should know that `durian cultivator` is equivalent to `orange farmer` and so that `xxx` is also a person.
- The proble is that the word `durian` hasn't probably seen during training on a small training set.

> <img src="./images/w02-03-using_word_embeddings/img_2023-05-02_07-58-47.png">

That's why we use **transfer learning**.

As seen in other transfer learning settings, if you're trying to transfer from some task A to some task B, the process of transfer learning is just most useful when you happen to have a ton of data for A and a relatively smaller data set for B.

That means that word embeddings is useful you have small training set, like for Named entity recognition, Text summarization, co-reference resolution, parsing
It's less useful for Language modeling, Machine translation

> <img src="./images/w02-03-using_word_embeddings/img_2023-05-02_07-58-49.png">

Finally, word embeddings has a interesting relationship to the face encoding ideas that you learned during convolutional neural networks course. In the [Siamese network architecture](../../c4-convolutional-neural-netowrks/week4/README.md#siamese-network) we a 128 dimensional representation for different faces. And then you can compare these encodings in order to figure out if these two pictures are of the same face.

The words encoding and embedding mean fairly similar things. So in the face recognition literature, people also use the term encoding to refer to these vectors, f(x(i)) and f(x(j)).

One difference between the face recognition literature and what we do in word embeddings :
- for face recognition, you wanted to train a neural network that can take as input any face picture, even a picture you've never seen before
- for word embeddings, we work with fixed vocabulary (ex. 10,000 words)

> <img src="./images/w02-03-using_word_embeddings/img_2023-05-02_07-58-52.png">

## Properties of Word Embeddings

One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning.

Let's say I pose a question, `man` is to `woman` as `king` is to what?

- If we substract : e<sub>man</sub> - e<sub>woman</sub> ≈ [-2  0  0  0]
- We also have e<sub>king</sub> - e<sub>queen</sub> ≈ [-2  0  0  0]

> <img src="./images/w02-04-properties_of_word_embeddings/img_2023-05-02_08-00-17.png">

So let's formalize how you can turn this into an algorithm.

The idea is to find a word `w` that maximize the similarity with e<sub>king</sub> - e<sub>man</sub> + e<sub>woman</sub>

Note that previously, we talked about using algorithms like t-SNE to visualize words. But t-SNE is a non-linear algorithm that takes 300-D data, and it maps it in a very non-linear way to a 2D space. So after the t-SNE mapping, you should not expect these types of parallelogram relationships.

It's really in this original 300 dimensional space that you can more reliably count on these types of parallelogram relationships in analogy pairs

> <img src="./images/w02-04-properties_of_word_embeddings/img_2023-05-02_08-00-19.png">

To compute the similaririty, we can use :
- cosine similarity, it's the most commonly used similarity function, basically the inner product between u and v
- square distance or Euclidian distance

> wikipedia
>
> In mathematics, the dot product or scalar product is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors), and returns a single number.
>
> In Euclidean geometry, the dot product of the Cartesian coordinates of two vectors is widely used. It is often called the inner product (or rarely projection product) of Euclidean space, even though it is not the only inner product that can be defined on Euclidean space (see Inner product space for more).

> <img src="./images/w02-04-properties_of_word_embeddings/img_2023-05-02_08-00-21.png">

## Embedding Matrix

When you implement an algorithm to learn a word embedding, what you end up learning is an embedding matrix.
- E is the embedding matrix - shape (300, 10000)
- O<sub>6257</sub> is the one-hot vecor -  (10000, 1)
- e<sub>6257</sub> is the multiplication of matrix by one-hot vetor - shape (300, 1)
- e<sub>6257</sub> is the column 6257-th column

> <img src="./images/w02-05-embedding_matrix/img_2023-05-02_08-00-33.png">

# Learning Word Embeddings: Word2vec &amp; GloVe

## Learning Word Embeddings

- In the history of deep learning, word embeddings algorithms werw relatively complex
- And then over time, researchers discovered they can use simpler and simpler and simpler algorithms
- Some of the most popular agorithm today are so simple that could be seem a little bit magical and still get very good results especially for a large dataset
- So, what I'm going to do is start off with some of the slightly more complex algorithms because I think it's actually easier to develop intuition about why they should work

You're building a language model and you do it with a neural network. So, during training, you might want your neural network to do something like input, `I want a glass of orange ___ `, and then predict the next word in the sequence.

Here is an explanation of one of the earlier and pretty successful algorithms for learning word embeddings, for learning this matrix E (Yoshua Bengio, Rejean Ducharme, Pascals Vincent, and Christian Jauvin)

1. We construct a one-hot vector for each word (10,000 dimensional vector)
2. We multiple each one-hot vector by matrix `E` to obtain a embedding vector (300 dimensional vector)
3. Feed all embedding vectors to a neural network, that feeds a softmax function that classifies the 10.000 possible outputs in the vocabulary

Note that :
- The input of the layer is a 1800-dimensional vector (6 words, each word encoded with a 300-dimensional embedded vector)
- What is ore commonly done is that we decide to have a fix historical windows (exemple take only the previous 4 words). Input is then a 1200-dimensional vector (instead if 1800)

The parameters of this model is :
- the matrix `E` (the same for each word)
- `w[1]` and `b[1]` for the first layer
- `w[2]` and `b[2]` for the softmax layer

You then perform gradient descent to maximize the likelihood of your training set to just repeatedly predict given four words in a sequence, what is the next word in your text corpus?

> <img src="./images/w02-07-learning_word_embeddings/img_2023-05-02_08-00-50.png">

Let's generalize this algorithm and see how we can derive even simpler algorithms.

I want to illustrate the other algorithms using a more complex sentence : `I want a glass of orange juice to go along with my cereal`.

If it goes to build a language model then is natural for the context to be a few words right before the target word. But if your goal isn't to learn the language model per se, then you can choose other contexts.

Contexts:
- Last 4 words `a glass of orange`
- 4 words on left & right: `a glass of orange ___ to go along with`
- Last 1 word: `orange`
- Nearby 1 word: `glass`, Which works surprisingly well. This is the idea of a Skip-Gram model

> <img src="./images/w02-07-learning_word_embeddings/img_2023-05-02_08-00-51.png">

## Word2Vec

Word2Vec is a simpler and more efficient algorithm to learn embeddings word matrix.

Skip-gram model is one of possible implementation.

The skip-gram train a simple neural network to perform a certain task, but we’re not actually use that neural network for the task (it's useless in itself), it's only a support to learn word embedding matrix.

What we predict : given the context word, predict the target word that will be randomly choosen in a +/- 10 words window

Obviously, this is not a easy learning problem, becaus in the +/- 10 words windows there is a lot of words
- **but the goals, is not to use this learning algorithm per se**
- the goal is to learn word embbedings

In the skip-gram model:
- we create a supervised learning problem based on a mappimg between context and target pairs
- rather than defining context with the last 4 words, we randomly pick a word in some window (example +/- 10 words)
- for word the context `orange`, we then could randomly choose `juice`, `glass` or `my` as target

> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-03.png">

Here are the details of the model.
- Vocabulary size = 10,000 words
- We want to learn context `c` to target `t`
- `c` is represented by its one-hot vector `O_c`
- We multiply `c` by an embedding matrix `E` to obtain `e_c`
- the we use softmax function to get output. `θ_t` is the parameter associated with the chance of a particular word `t` beeing choosen as target word

Parameters of the network:
- embedding matrix `E`
- softmax parameters : `θ_t`

If you optimize this loss function with respect to the all of these parameters, you actually get a pretty good set of embedding vectors

So this is called the skip-gram model because is taking as input one word like orange and then trying to predict some words skipping a few words from the left or the right side

> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-05.png">

The primary problem of skip-gram model is computational speed, in particular, for the softmax model. Every time you want to evaluate the probability of one word, you need to compute a sum over all 10,000 words in your vocabulary. And, in fact, 10,000 is actually already quite slow, but it makes even harder to scale to larger vocabularies (100,000 or a 1,000,000)

One of the solutions for this coputational problem is to use "Hierarchical softmax classifier". Instead of trying to categorize into all 10,000 categories, you have
- one classifier that tells you if the target word in the first 5000 words in the vocabulary, or is in the second 5000 words in the vocabulary
- if in the first 5000 words, a second classifier that tell you if in the 2500 first or the 2500 last
- ...
- until the exact classification

Hierarchical softmax classifier doesn't use a perfectly balanced symmetric tree, common words tend to be on top of the tree.

We won't go too much in detail, because [negative sampling](#negative-sampling) is a different method even simpler and works really well for speeding up the softmax classifier

How to sample the context:
- One thing you could do is just sample uniformly, at random, from your training corpus. When we do that, you find that there are some words like `the`, `of`, `a`, `and`, ... that appear extremely frequently and so training set be dominated by these words (instead of `orange`, `durian`, ...)
- So in practice the distribution of words `Pc` isn't taken just entirely uniformly at random for the training set purpose, but instead there are different heuristics that you could use in order to balance out something from the common words together with the less common words.

> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-07.png">

Skip gram model is one technic used to implement Word2Vec. Another is called CBOW (Continous Bag-Of-Words) which takes the surrounding contexts from middle word, and uses the surrounding words to try to predict the middle word
(inverse of skip-gram)

## Negative Sampling

The skip-gram train a simple neural network to perform a certain task, but we’re not actually use that neural network for the task (it's useless in itself), it's only a support to learn word embedding matrix. In the **negative samplig algorithm**, we modify the training objective in order to make it run much more efficiently, working in a much bigger training set, with a much better word embeddings learning.

What we predict :
- is (`orange`, `juice`) a context-target pair?
- is (`orange`, `king`) a context-target pair?


To build the training set:
- we generate a positive example as previously picking a context word (`orange`) and randomly choose a target word in a +/- 10 words window (`juice`)
- we generate k negative example by keeping the same context word (`orange`) by choosing randomly words from the dictionary (`king`, ...)
    - if target is in the learning sentence (`of` in the below slide), it doesn't matter
    - k is in [5,20] for small data set
    - k is in [2,5] for large data set

> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-19.png">

- Previously, we had the softmax model
- We now define a logistic regression model with sigmoid function and the same parameters
    - one parameter vector θ<sub>t</sub> for each possible target word
    - a separate parameter vector e<sub>c</sub>, the embedding vector, for each possible context word
- we have now a new network with 10.000 logistic (binary) regression classifier (one for `juice`, one for `king`, ...)
- but instead of training all 10,000 of them on every iteration, we're only going to train 5 of them (1 positive and 4 negative sampling in our example)


> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-21.png">

How do we choose the negatives examples:
- We can sample according to empirical frequencies in words corpus `f(wi)` (how often different words appears). But the problem with that is that we will have more frequent words like `the`, `of`, `and`...
- Other extreme would be to use `p(wi)=1/|V|` to sample uniformly at random. But this is also very non-representative of the distribution of English words
- Paper author reporetd that empirically that is in-between the extreme values above (not sure this is very theoretically justified)

> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-23.png">

So to summarize :
- you've seen how you can learn word vectors in a Softmax classier, but it's very computationally expensive
- you saw how by changing that to a bunch of binary classification problems, you can very efficiently learn words vectors. 
- And if you run this algorithm, you will be able to learn pretty good word vectors. 

Now of course, as is the case in other areas of deep learning as well, there are open source implementations. And there are also pre-trained word vectors that others have trained and released online under permissive licenses. And so if you want to get going quickly on a NLP problem, it'd be reasonable to download someone else's word vectors and use that as a starting point.

## GloVe Word Vectors

Another algorithm that has some momentum in the NLP community is the GloVe (global vectors for word representation) algorithm. This is not used as much as the Word2Vec or the skip-gram models, but it has some enthusiasts. Because in part of its simplicity

- We define $$X_{ij}$$ as the number of times that a word i appears in the context of word j (close to each other)
    - i is playing the role of t (target)
    - j is playing the role of c (context)
- Depending on the definition of context and target words, you might have that $$X_{ij}$$ equals $$X_{ji}$$. And in fact, if you're defining context and target in terms of whether or not they appear within +/- 10 words of each other, then it would be a symmetric relationship (not symetric if your context was defined as the word immediately before the target word)


> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-33.png">

GloVe Word Vectors :
- θ<sub>i</sub><sup>T</sup>e<sub>j</sub>&nbsp;plays the role of θ<sub>t</sub><sup>T</sup>e<sub>c</sub>&nbsp;in the [negative sampling alorithm](/#negative-sampling)
- f(x<sub>ij</sub>) is a weighting term, used for 2 mains reasons:
    - log(0) is not define, so by convention we define that f(0)=0 and 0*log(0)=0
    - manage with difference offrequencies between :
        - decrease importamce of **stop words**, that appears too frequently in the English language (`this`, `is`, `of`, `a`, ...)
        - gives a meaningful amount of computation, even to the less frequent words like `durian`
- Read paper if yuo need more detail on heuristics for choosing this weighting function f that neither gives these words too much weight nor gives the infrequent words too little weight
- θ<sub>i</sub> and e<sub>j</sub> are symetric, so one way to train th algorithm is :
    - to initialize θ<sub>i</sub> and e<sub>j</sub> both uniformly random
    - run gradient descent to minimize its objective,
    - and then take the average

One confusing part of this algorithm is, if you look at this equation, it seems almost too simple. How could it be that just minimizing a square cost function like this allows you to learn meaningful word embeddings? But it turns out that this works. And the way that the inventors end up with this algorithm was, they were building on the history of much more complicated algorithms and then this came later. And we really hope to simplify all of the earlier algorithms.

> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-35.png">

- In the GloVe algorithm you cannot guarantee that the individual components of the embeddings are interpretable
- You even cannot guarantee that axis are orthogonal
- Mathematically, you could easily find invertible matrix A that proves that you can't guarantee that the axis used to represent the features will be well-aligned with what might be easily humanly interpretable axis
-  But despite this type of linear transformation, the parallelogram map that we worked out when we were describing analogies, that still works. And so, despite this potentially arbitrary linear transformation of the features, you end up learning the parallelogram map for figure analogies still works.

> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-37.png">

# Applications Using Word Embeddings

## Sentiment Classification

- Sentiment classification map a text sequence into an integer that represents rating (from 1 to 5)
- One of the challenges of sentiment classification is you might not have a huge label training set for it.
- But with word embeddings, you're able to build good sentiment classifiers even with only modest-size label training sets

> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-41.png">

- embedding matrix may have been trained on say 100 billion words.
- 300 number of features in word embedding
- we can use sum or average given all the words then pass it to a softmax classifier
    - works for short or long sentences
    - but ignores word order and so have a bad interpreation (`lacking in good taste` is bad evev ig `good` word is present)

> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-43.png">

- A more sophisticated model consoists in using a RNN for sentiment classification (many-to-one RNN architecture)
- it will be much better at taking word sequence into account and realize that `things are lacking in good taste` is a negative review
- it will be much better with `Completely absent of good taste, good service, and good ambiance` or something, then even if the word `absent` is not in your label training set (similarity with `lacking` word in the training set)

> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-45.png">

## Debiasing Word Embeddings

Machine learning and AI algorithms are increasingly trusted to help with, or to make, extremely important decisions. They're influencing everything ranging from college admissions, to the way people find jobs, to loan applications.

Word embeddings algorithm can pick up the biases of the text used to train the model and so the biases they pick up or tend to reflect the biases in text as is written by people.

We want to reduce or ideally eliminate undesirable biases (gender, ethnicity, sexual orientation bias)

> <img src="./images/w02-13-debiasing_word_embeddings/img_2023-05-02_08-10-01.png">

- The first thing we're going to do is to identify the direction corresponding to a particular bias we want to reduce or eliminate
    - calculating e<sub>she</sub> - e<sub>he</sub>...
    - this help to find the bias and not bias axis
- For each word (`babysitter`) that should not be definitional linked to for example the gender, project words on the non-bias axis
- Equalize the distance of gender-specific words (`grand mother`, `grand father`)

> <img src="./images/w02-13-debiasing_word_embeddings/img_2023-05-02_08-10-03.png">