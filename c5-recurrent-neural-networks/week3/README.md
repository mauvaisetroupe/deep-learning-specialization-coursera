# Sequence Models & Attention Mechanism

Augment your sequence models using an attention mechanism, an algorithm that helps your model decide where to focus its attention given a sequence of inputs. Then, explore speech recognition and how to deal with audio data.

Learning Objectives
- Describe a basic sequence-to-sequence model
- Compare and contrast several different algorithms for language translation
- Optimize beam search and analyze it for errors
- Use beam search to identify likely translations
- Apply BLEU score to machine-translated text
- Implement an attention model
- Train a trigger word detection model and make predictions
- Synthesize and process audio recordings to create train/dev datasets
- Structure a speech recognition project

# Various Sequence To Sequence Architectures

##  Basic Models

In this week, we work on sequence-to-sequence models, which are useful for everything from machine translation to speech recognition.

Machine Translation:
- First, we have encoder network
    - built as a RNN, GRU or LSTM, that feed in the input French words one word at a time
    - After ingesting the input sequence the RNN then outputs a vector that represents the input sentence.
- Then a decoder
    - which takes as input the encoding
    - then can be trained to output the translation one word at a time
    - until end of sequence.
- One of the most remarkable recent results in deep learning is that this model works.
    - Given enough pairs of French and English sentences, if you train a model to input a French sentence and output the corresponding English translation, this will actually work decently well.
    - This model simply uses an encoding network whose job it is to find an encoding of the input French sentence, and then use a decoding network to then generate the corresponding English translatio

> <img src="./images/w03-01-basic_models/img_2023-05-10_17-34-56.png">

Image captioning:
- An architecture very similar to this also works for image captioning.
- First a pretrained CNN (AlexNet) as an encoder for the image,
    - we get rid of this final Softmax unit,
    - so the pre-trained AlexNet can give you a 4096-dimensional feature vector than encodes and represents the image
- Then the decoder (RNN)
    - similary at the translation machine
- Model works pretty well for captioning

> <img src="./images/w03-01-basic_models/img_2023-05-10_17-35-00.png">

There are some differences between how you'll run a model like this (machine translation or picture captioning) that generates a sequence compared to the one used for synthesizing a novel text using a language model.

One of the key differences is you don't want to randomly choose in translation, you may be want the most likely translation. Or you don't want to randomly choose in caption but you want the best caption and most likely caption.

Let's see in the next video how you go about generating that.

##  Picking the Most Likely Sentence

The machine translation is very similar to a **conditional language model**.

- In language modeling (network we had built in the [first week](../week1/README.md)), the model allows you to estimate the probability of a sentence. That's what a language model does (schema upper in the slide)
- The decoder part of the machine translation model is identical to the language model,
    - except that instead of always starting along with the vector of all zeros,
    - it has an encoder network that figures out some representation for the input sentence
- Instead of modeling the probability of any sentence, machine translation is now modeling the probability of the output English translation **conditioned on** some input French sentence. In other words, we estimate the probability of an English translation.

> <img src="./images/w03-02-picking_the_most_likely_sentence/img_2023-05-10_17-37-34.png">

This now tells you what is the probability of different English translations of that French input. And, what you do not want is to sample outputs at random, sometimes you may sample a bad output:

|Translation|Comment|
|-|-|
|Jane is visiting Africa in September.|Good translation|
|Jane is going to be visiting Africa in September.| sounds a little awkward, not the best one|
|In September, Jane will visit Africa.||
|Her African friend welcomed Jane in September.|Bad translation|

So, when you're using this model for machine translation, you're not trying to sample at random from this distribution. Instead, what you would like is to find the English sentence, y, that **maximizes** that conditional probability. The most common algorithm is the beam search, which we will explain in the next video.

> <img src="./images/w03-02-picking_the_most_likely_sentence/img_2023-05-10_17-37-36.png">

But, before moving on to describe beam search, you might wonder, why not just use greedy search? So, what is greedy search? 

Greedy search is an algorithm from computer science which says to  
- generate the first word just pick whatever is the most likely first word according to your conditional language model
- you then pick whatever is the second word that seems most likely
- and so on

And it turns out that the greedy approach doesn't really work.

The first sentence "Jane is visiting Africa in September." i sa better solution (no unnecessary words)

But, if the algorithm has picked "Jane is" as the first two words, because "going" is a more common English word, probably the chance of "Jane is going," given the French input, this might actually be higher than the chance of "Jane is visiting," given the French sentence.

I know this was may be a slightly hand-wavey argument, but, 
- this is an example of a broader phenomenon, where if you want to find the sequence of words, y1, y2, ... that together maximize the probability, it's not always optimal to just pick one word at a time.
- the total number of combinations of words in the English sentence is exponentially larger (10'000 vocabulary, 10 words in sentence, so you have 10'000<sup>10</sup> possible sentences) and it's impossible to rate them all, which is why the most common thing to do is use an approximate search algorithm


> <img src="./images/w03-02-picking_the_most_likely_sentence/img_2023-05-10_17-37-37.png">


So, to summarize: 
- machine translation can be posed as a conditional language modeling problem
- one major difference between this and the earlier language modeling problems is rather than wanting to generate a sentence at random, you may want to try to find the most likely English sentence, most likely English translation
- but the set of all English sentences of a certain length is too large to exhaustively enumerate
- So, we have to resort to a search algorithm

##  Beam Search

> <img src="./images/w03-03-beam_search/img_2023-05-10_17-38-03.png">
> <img src="./images/w03-03-beam_search/img_2023-05-10_17-38-04.png">
> <img src="./images/w03-03-beam_search/img_2023-05-10_17-38-05.png">

##  Refinements to Beam Search

> <img src="./images/w03-04-refinements_to_beam_search/img_2023-05-10_17-38-17.png">
> <img src="./images/w03-04-refinements_to_beam_search/img_2023-05-10_17-38-19.png">

##  Error Analysis in Beam Search

> <img src="./images/w03-05-error_analysis_in_beam_search/img_2023-05-10_17-38-32.png">
> <img src="./images/w03-05-error_analysis_in_beam_search/img_2023-05-10_17-38-33.png">
> <img src="./images/w03-05-error_analysis_in_beam_search/img_2023-05-10_17-38-34.png">

##  Bleu Score (Optional)

> <img src="./images/w03-06-bleu_score/img_2023-05-10_17-38-44.png">
> <img src="./images/w03-06-bleu_score/img_2023-05-10_17-38-45.png">
> <img src="./images/w03-06-bleu_score/img_2023-05-10_17-38-46.png">
> <img src="./images/w03-06-bleu_score/img_2023-05-10_17-38-47.png">

##  Attention Model Intuition

> <img src="./images/w03-07-attention_model_intuition/img_2023-05-10_17-38-58.png">
> <img src="./images/w03-07-attention_model_intuition/img_2023-05-10_17-39-00.png">

##  Attention Model

> <img src="./images/w03-08-attention_model/img_2023-05-10_17-39-09.png">
> <img src="./images/w03-08-attention_model/img_2023-05-10_17-39-10.png">
> <img src="./images/w03-08-attention_model/img_2023-05-10_17-39-11.png">

# Speech Recognition - Audio Data

##  Speech Recognition

> <img src="./images/w03-09-speech_recognition/img_2023-05-10_17-39-20.png">
> <img src="./images/w03-09-speech_recognition/img_2023-05-10_17-39-21.png">
> <img src="./images/w03-09-speech_recognition/img_2023-05-10_17-39-22.png">

##  Trigger Word Detection

> <img src="./images/w03-10-trigger_word_detection/img_2023-05-10_17-39-31.png">
> <img src="./images/w03-10-trigger_word_detection/img_2023-05-10_17-39-32.png">