## Data
The ideal dataset would be one with predefined summaries, in addition to the main body of the text. The most applicable data would be news articles, and you could consider news article headlines to be summaries (as long as the source is not The Sun)

This dataset will then be split into training and test data so that the model produced can be effectively validated. It is imperative to minimise the chance of under/overfitting, and so the optimum ratio of training to test data will need to be found. As per current NLP conventions, initially a 70:30 or 80:20 split will be used.

The aim would then be to find another dataset, one that is perhaps a different type of body of text, and explore how well the model can summarise the given corpora.

Any data used will need to be cleaned and shaped. This will include techniques such as removal of stop words, and lemmatisation.

* Dataset: https://cs.nyu.edu/~kcho/DMQA/
* There are two datasets available CNN and Daily Mail. As CNN has more data I will start with this
* Each 'story' file has the following format
  * Body
  * @headline
  * Summary text
* The first step of cleaning this data will be to seperate the bodies and summaries into seperate files, so that it becomes easier to train and test data.


## Project Goals
1. Summarise large amounts of texts, into single lines or a singular paragraph
2. Produce easy to interpret outputs
3. Ensure summarised text is comprehensible and grammatically correct.
3. Try and achieve a validation accuracy of 70% (not sure how feasible this is at the current stage)
4. Use a variety of NLP techniques, where applicable
5. 

## Project Approach
There are two types of text summarisation approaches: 
- Extractive
  - Selecting phrases from the text and ranking them to deduce which is most relevant to the meaning of the corpus.
- Abstractive
  - Generating a new phrase based on the corpus content, that effectively captures its' meaning.

Historically, it seems extractive has been a more successful approach due to its ease of implementation in comparison the abstractive summarisation. The results derived from abstractive methods do however prove to be more general and "human-like".

Text summarisation can be achieved through both unsupervised and supervised methods, with supervised approaches (like Neural Networks), potentially deriving better results. 

The simplest approach of achieving summarisation is to extract keywords from a text, based on statistical methods such as TF-IDF, word embeddings (Word2Vec), and TextRank.

To extract phrases and sentence of high importance, the use of N-Grams along with these statistical methods will be imperative.

To achieve a Neural Networks/Deep Learning-based approach, ML libraries such as TensorFlow, Keras or Pytorch would be used, otherwise, the project would steer more towards creating ML algorithms. 

From initial reading around the subject, it seems using Recurrent Neural Networks to produce a Sequence-to-Sequence based architecture are quite powerful for this task, and so this will be a potential starting point.

## Why Is This Important
In modern-day society, so much data available to us, and most of it is too long and laborious for the everyday person to read. Summaries have always proven to be the easiest and most practical method of allowing readers to understand aspects of documents. The appeal for a system that enables anyone to input text of any: size, format, or structure, and receive concise, well-structured summaries is valuable. Text summarisation is also a task that speeds up other NLP problems such as sentiment analysis.

## Evaluation
The first stage of evaluating the model, will be to validate it using the testing data. There are some other more complex cross-validation models such as k-folds, which may be more comprehensive, and so if time permits, these may be worth exploring.

The best measure for evaluating would probably be accuracy. It's also important to have some form of human-based checking (once data has been summarised), to ensure that the text is "human-friendly".

In addition there is the ROGUE metric - https://en.wikipedia.org/wiki/ROUGE_%28metric%29

## Project Plan

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Projet Timeline Plan


section Planning
Literature Reading            :         des1, 2019-10-13, 2019-10-19
Find Dataset                  :         des2, 2019-10-13, 1d
Clean/Shape Data              :         des3, 2019-10-14, 2019-10-15
Train/Test Split              :         des3, 2019-10-15, 2019-10-16
Project Proposal              :         des4, 2019-10-23, 2019-10-25

section Project
Basic Feature Phase           :active,  des6, 2019-10-20, 2019-12-13
Text Scoring Model            :         des1, 2019-10-20, 2019-10-27
Basic Summarisation           :         des1, 2019-10-23, 2019-10-27
Inspection Week               :         des3, 2019-12-13, 2019-12-06
Working Prototype             :         des5, 2019-12-13, 1d

Complex Feaures Phase         :active,  des6, 2019-12-15, 2020-02-28
Potential Front End Imp.      :         des6, 2020-02-20, 2020-02-28
New Feature Coding Completion :         des6, 2020-02-29, 1d

Section Finalising
Project Refractoring          :         des6, 2020-03-01, 2020-03-08
Documentation                 :         des6, 2020-03-10, 2020-03-21
Demo Week                     :         des7, 2020-03-17, 2020-03-21
Tweaks Based On Feedback      :         des7, 2020-03-21, 2020-03-26
Submission Deadline           :         des8, 2020-03-27, 1d
```

## Cue Phrases
Sentences that begin with concluding phrases like "in summary", "in conclusion", "the most important" etc. can be considered as indicators of a potential summary of the document. 
It is apt to assign a higher score to sentences that contain cue words and phrases using the formula
    Cue-Phrase Score - (Number of Cue Phrases in the sentence / Total number of cue phrases in the document)

Ref : https://reader.elsevier.com/reader/sd/pii/S0957417413002601?token=010CE8840408FF777CA5D27B71A76694E192D34312B9C11EEAD1CD8B48560F327CE4E2821B7E466924E4E6FFB65BE8FA 2.2.1
