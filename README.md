# Minimising the Universe.
## Finding the Optimal Summary Using Abstractive Summarization Methods.

### Setup
- Install required dependenices (required for first run)
    ```
    make setup 
    ```
- Install required python packages 
    </br> *only run if requirements have changed since make setup has been run  + no new nltk packages are required to be installed*
    ```
    pip install -r requirements.txt
    ```

### Run
- To run with default settigs
    ```
    make run
    ```
- To alter runtime variables
    ```
    make run TRAIN_DATA_PATH=<> OUTPUT_CSV=<>
    ```

### Proposal
In modern-day society, so much data available to us, and most of it is too long and laborious for the everyday person to read. Summaries have always proven to be the easiest and most practical method of allowing readers to understand aspects of documents. The appeal for a system that enables anyone to input text of any: size, format, or structure, and receive concise, well-structured summaries is valuable. 
Over the past few years, a lot of text summarization algorithms have been implemented, with average performing extractive methods becoming trivial to implement. The main challenges with these methods are creating summaries that effectively convey the whole message in a human-like way. With the rise of deep learning, recurrent neural networks have been instrumental in driving abstractive based method, however there are still issues surrounding reproducing factual details and ensuring lack of repetition.  
As a consequence of discoviring this, I have chosen to experiment with the effect of combining abstractive and extractve methodologies.
I am looking to build a Sequence2Sequence based Recurrent Neural Network (RNN), in order to create a model capable of abstractive text summarization. To enhance the model, I will experiment with adding in a group of traditionally extractive summarization based methods, such as TextRank and Ontology-based classification, to see how they impact the models’ overall performance, particularly when dealing with different types of data (e.g. academic papers rather than news articles).
The most challenging part of this will be to create an initial machine learning model that can be easily adapted to accommodate for the introduction of these extractive 'modules'. Additionally, checking the minimised form of the text is grammatically and human-friendly will be quite challenging.

### Project Plan

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Projet Timeline Plan


section Planning
Literature Reading                      :active,  des1, 2019-10-13, 2019-10-26
Project Proposal                        :done,    des1, 2019-10-20, 2019-10-22
Preliminary Project Plan                :done,    des1, 2019-10-13, 2019-10-21

section Data                  
Find Training Dataset                   :done,    des1, 2019-10-14, 2019-10-15
Manual Look Through Of Dataset          :done,    des1, 2019-10-15, 2019-10-16
Read In Data                            :done,    des1, 2019-10-19, 2019-10-21
Clean/Shape Data                        :done,   des1, 2019-10-20, 2019-10-25

section Project Semester 1
Removal of Stop Words                   :done,    des1, 2019-10-23, 2019-10-25
Lemmatization Of Text                   :done,    des1, 2019-10-24, 2019-10-27
Part Of Speech                          :done,    des1, 2019-10-25, 2019-10-27
RNN Seq2Seq Model                       :done,    des1, 2019-10-28, 2019-11-02
Setup Full Working Flow                 :done,    des1, 2019-11-11, 2019-11-24
K-Folds Cross Validation Algorithm      :         des1, 2019-11-18, 2019-11-25
Evaluation Method                       :done,    des1, 2019-11-15, 2019-11-25
Improving/Fixing the Model              :done,    des1, 2019-11-20, 2019-12-13
Inspection Week                         :done,    des1, 2019-12-13, 2019-12-06
Bidirectional Model                     :done,    des1, 2019-12-08, 2019-12-03
Working Prototype                       :done,    des1, 2019-12-13, 2019-12-06
End Of Semester 1                       :         des1, 2019-12-13, 1d

section Project Semester 2
Start of Semester 2                     :         des1, 2020-01-13, 1d
Experimenting with WordEmbeddings       :done,    des1, 2020-01-15, 2020-01-22
Experimenting with TextRank             :active,    des1, 2020-01-22, 2020-01-26
Experimenting with Attention Mechanisms :         des1, 2020-01-28, 2020-02-04
Ontology Based Classification           :         des1, 2020-02-02, 2020-02-13
Adding in some more Extractive methods  :         des1, 2020-02-05, 2020-02-20
Experimenting with N-Words              :         des1, 2020-02-13, 2020-02-19
Experimenting with Catch Phrases        :         des1, 2020-02-05, 2020-02-08
Experimenting with Grammar based methods:         des1, 2020-02-09, 2020-02-15
Testing the model with different data   :         des1, 2020-02-15, 2020-02-20 
Front End Implementation                :         des1, 2020-02-20, 2020-02-28
Experimental Features Coding Completion :         des1, 2020-02-29, 1d

Section Finalising
Project Refractoring                    :         des1, 2020-03-01, 2020-03-08
Documentation                           :         des1, 2020-03-10, 2020-03-21
Demo Week                               :         des1, 2020-03-17, 2020-03-21
Tweaks Based On Feedback                :         des1, 2020-03-21, 2020-03-26
Submission Deadline                     :         des1, 2020-03-27, 1d
```

### Actual Outcome

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Projet Outcome Timeline

section Planning
Literature Reading                      :active,  des1, 2019-10-13, 2019-10-26
Project Proposal                        :done,    des1, 2019-10-20, 2019-10-22
Preliminary Project Plan                :done,    des1, 2019-10-13, 2019-10-21

section Data                  
Find Training Dataset                   :done,    des1, 2019-10-14, 2019-10-15
Manual Look Through Of Dataset          :done,    des1, 2019-10-15, 2019-10-16
Read In Data                            :done,    des1, 2019-10-19, 2019-10-21
Clean/Shape Data                        :done,    des1, 2019-10-20, 2019-10-25

section Project Semester 1
Removal of Stop Words                   :done,    des1, 2019-10-23, 2019-10-25
Lemmatization Of Text                   :done,    des1, 2019-10-24, 2019-10-27
Part Of Speech                          :done,    des1, 2019-10-25, 2019-10-27
RNN Seq2Seq Uni Directional Model       :done,    des1, 2019-10-28, 2019-11-08
Debbugging the Model                    :done,    des1, 2019-11-08, 2020-01-03
Setup Full Working Flow                 :done,    des1, 2019-11-14, 2019-11-24
Improving/Fixing the Model              :done,    des1, 2019-11-25, 2019-12-03
Evaluation Method                       :done,    des1, 2019-11-23, 2019-12-29
Project Inspection                      :done,    des1, 2019-12-11, 1d
Working Prototype                       :done,    des1, 2019-12-13, 2019-12-06
Bidirectional Model                     :done,    des1, 2020-01-03, 2020-01-04
End Of Semester 1                       :         des1, 2019-12-13, 1d

section Project Semester 2
Start of Semester 2                     :         des1, 2020-01-13, 1d
Experimenting with WordEmbeddings       :done,    des1, 2020-01-13, 2020-01-26
K-Folds Cross Validation Algorithm      :         des1, 2020-01-13, 2020-01-16
Ontology Based Classification           :         des1, 2020-01-16, 2020-01-20
Experimenting with TextRank             :active,  des1, 2020-02-02, 2020-02-03
Experimenting with Attention Mechanisms :         des1, 2020-01-28, 2020-02-04
Adding in some more Extractive methods  :         des1, 2020-02-05, 2020-02-20
Experimenting with N-Words              :         des1, 2020-02-13, 2020-02-19
Experimenting with Catch Phrases        :         des1, 2020-02-05, 2020-02-08
Experimenting with Grammar based methods:         des1, 2020-02-09, 2020-02-15
Testing the model with different data   :         des1, 2020-02-15, 2020-02-20 
Front End Implementation                :         des1, 2020-02-20, 2020-02-28
Experimental Features Coding Completion :         des1, 2020-02-29, 1d

Section Finalising
Project Refractoring                    :         des1, 2020-03-01, 2020-03-08
Documentation                           :         des1, 2020-03-10, 2020-03-21
Demo Week                               :         des1, 2020-03-17, 2020-03-21
Tweaks Based On Feedback                :         des1, 2020-03-21, 2020-03-26
Submission Deadline                     :         des1, 2020-03-27, 1d
```

### Flow

- Data processing
    - Read in data
    - Clean data
        - POS
        - Lemmatization
        - Stop Word Removal
    - Reshape data    
- Write dataframe to CSV
- Read CSV into model
- Model 
    - word embeddings
    - encoder
    - decoder
    - inference model
    - evaluation

