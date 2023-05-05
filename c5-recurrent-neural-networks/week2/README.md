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
- we use $O_{5391}$ to denote the one-hot vector with `1` in position `5391` (and `0` elseweher)
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

Most of the ideas are due to Tomas Mikolov, Kai Chen, Greg Corrado, and Jeff Dean

> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-03.png">
> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-05.png">
> <img src="./images/w02-08-word2vec/img_2023-05-02_08-01-07.png">

## Negative Sampling

> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-19.png">
> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-21.png">
> <img src="./images/w02-09-negative_sampling/img_2023-05-02_08-01-23.png">

## GloVe Word Vectors

> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-33.png">
> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-35.png">
> <img src="./images/w02-10-glove_word_vectors/img_2023-05-02_08-01-37.png">

# Applications Using Word Embeddings

## Sentiment Classification

> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-41.png">
> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-43.png">
> <img src="./images/w02-12-sentiment_classification/img_2023-05-02_08-09-45.png">

## Debiasing Word Embeddings

> <img src="./images/w02-13-debiasing_word_embeddings/img_2023-05-02_08-10-01.png">
> <img src="./images/w02-13-debiasing_word_embeddings/img_2023-05-02_08-10-03.png">