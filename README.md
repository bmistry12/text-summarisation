# Finding the Optimal Summary By Combining Extractive and Abstractive Summarization Methods.

In modern-day society, we are surrounded by data, most of which is too long and laborious for the everyday person to read comprehensively. With the art of generating extractive summaries becoming increasingly trivial, the challenge of producing abstractive summaries that can understand and convey the meaning of a document remains prominent within the field of natural language processing. Although summarisation is a task that humans can easily complete, developing automated approaches that can generalise well with different data formats remains a challenge. Despite a recent surge in research to improve abstractive summarisation methods, there are still outstanding issues regarding the validity of generated summaries. The method proposed experiments with combining extractive and abstractive summarisation methods, to evaluate whether these extractive methods can be effective in driving more human-like abstractive summaries. The overall aims and evaluators are the reduction of repetition, and the improvement of precision, recall and human-judged grammatical correctness.

## Setup
- This project requires Python 3.0+ to run.
- Install required dependencies (required for first run to ensure NLTK packages are installed)
    ```
    make setup 
    ```
- Install required python packages 
    </br> *only run if requirements have changed since make setup has been run  + no new NLTK packages are required to be installed*
    ```
    make requirements
    ```
- To run the GloVe model, download the pretrained word embeddings found [here](https://nlp.stanford.edu/projects/glove/) and ensure they are place on the specified path in a file named "glove".

## Run
- To run data processing with default settings (as shown below)
    ```
    make run-data
    ```
- To run model with default settings - bidirectional model
    ```
    make run-model
    ```
- To alter runtime variables follow the following example
    ```
    make run-data TRAIN_DATA_PATH=<> OUTPUT_CSV=<> ...
    ```

*Note: For any machine running python 3.0 via the python3 command, append -labs to the end of any make commands (e.g. make-setup-labs)*

## MakeFile
The default settings for all runtime variables that can be altered are shown below.
```
    ## Variables
    ### Data Processing
    TRAIN_DATA_PATH="./cnn/originals"
    OUTPUT_CSV="./data/cnn-tr.csv"
    TRAIN_DATA_PATH_LABS="/tmp/bhm699/dailymail/originals"
    OUTPUT_CSV_LABS="/tmp/bhm699/dailymail-wf.csv"

    STOP_WORDS=True
    LEMMATIZE=True
    LEMMATIZE_WITH_POS=True
    SENT_POS=False
    #### Only one of these can be true at any given time
    TEXT_RANK=False
    WORD_FREQ=False

    ### Model Running
    MODEL_ID=1  # 0 = unidirectional, 1=bidirectional, 2=GloVe model
    WORD_REMOVAL=False # remove words using uncommon_word_thr
    CSV_NAME="cnn-all.csv" # csv data to run model against
```

## Flow
- dataProcesing.py
    - Read in Data
    - Clean Data
        - Stop Word Removal
        - POS
        - Lemmatization
    - Extractive Methods
      - TextRank?
      - Word Frequency?
      - Sentence Position?
  - Write Dataframme to CSV
- modelCommon.py and models.py
  - Read CSV into Dataframe
  - Data Cleaning
  - Uncommon Word Removal?
  - Max Text Lengths
  - Training Validation Split
  - Word Embeddings  
    - Reshape Data  
    - Learning Model
      - Encoder
      - Decoder
      - Combined LSTM Model
      - Training
    - Inference Model
      - Encoder
      - Decoder
      - Reverse Word Embeddings
    - Evaluation
      - Validation Data
      - Training Data
      - Test Data

## Project Plan

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Projet Timeline Plan

section Planning
Literature Reading                      :done,    des1, 2019-10-13, 2019-10-26
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
Inspection Week                         :done,    des1, 2019-12-06, 2019-12-13
Bidirectional Model                     :done,    des1, 2019-12-03, 2019-12-08
Working Prototype                       :done,    des1, 2019-12-06, 2019-12-13
End Of Semester 1                       :         des1, 2019-12-13, 1d

section Project Semester 2
Start of Semester 2                     :         des1, 2020-01-13, 1d
Experimenting with WordEmbeddings       :done,    des1, 2020-01-15, 2020-01-22
Experimenting with TextRank             :done,    des1, 2020-01-22, 2020-01-26
Experimenting with Attention Mechanisms :         des1, 2020-01-28, 2020-02-04
Ontology Based Classification           :         des1, 2020-02-02, 2020-02-13
Adding in some more Extractive methods  :done,    des1, 2020-02-05, 2020-02-20
Experimenting with N-Words              :         des1, 2020-02-05, 2020-02-10
Experimenting with Word Frequency       :done,    des1, 2020-02-010, 2020-02-12
Experimenting with Grammar based methods:         des1, 2020-02-12, 2020-02-15
Testing the model with different data   :done,    des1, 2020-02-15, 2020-02-20 
Experimenting with Stop Word            :done,    des1, 2020-02-20, 2020-02-28
Experimental Features Coding Completion :done,    des1, 2020-02-29, 1d

Section Finalising
Project Refractoring                    :done,    des1, 2020-03-01, 2020-03-08
Documentation                           :active,  des1, 2020-03-10, 2020-03-26
First Draft Completed                   :done,    des1, 2020-02-16, 1d
Demo Week                               :done,    des1, 2020-03-02, 2020-03-06
Tweaks Based On Feedback                :done,    des1, 2020-03-07, 2020-03-26
Submission Deadline                     :         des1, 2020-03-27, 1d
```

### Actual Outcome

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Projet Outcome Timeline

section Planning
Literature Reading                      :done,    des1, 2019-10-13, 2019-10-26
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
Working Prototype                       :done,    des1, 2019-12-06, 2019-12-13
Bug Fixing                              :done,    des1, 2019-12-16, 2020-02-20
Bidirectional Model                     :done,    des1, 2020-01-03, 2020-01-04
End Of Semester 1                       :         des1, 2019-12-13, 1d

section Project Semester 2
Start of Semester 2                     :         des1, 2020-01-13, 1d
Experimenting with WordEmbeddings       :done,    des1, 2020-01-13, 2020-01-26
Hyperparameter Testing                  :done,    des1, 2020-02-20, 2020-02-22
Experimenting with TextRank             :done,    des1, 2020-02-02, 2020-03-25
Adding in some more Extractive methods  :done,    des1, 2020-02-05, 2020-02-20
Experimenting with Word Frequency       :done,    des1, 2020-02-20, 2020-02-22
Experimenting with Sentence Position    :done,    des1, 2020-02-20, 2020-02-25
Testing the model with different data   :done,    des1, 2020-02-15, 2020-03-16
Experimental Features Coding Completion :done,    des1, 2020-02-29, 1d

Section Finalising
Project Refractoring                    :done,    des1, 2020-02-20, 2020-02-23
Documentation                           :active,  des1, 2020-02-15, 2020-03-26
First Draft Completed                   :done,    des1, 2020-02-16, 1d
Demo                                    :done,    des1, 2020-03-02, 2020-03-04
Tweaks Based On Feedback                :done,    des1, 2020-03-05, 2020-03-06
Submission Deadline                     :         des1, 2020-03-27, 1d
New Submission Deadline                 :         des1, 2020-04-10, 1d
```