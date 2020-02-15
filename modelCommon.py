import re
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.layers import Attention
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from rouge import Rouge

class Common():
    def __init__(self):
        self.PATH = ""
        self.BATCH_SIZE = 30
        self.EPOCHS = 50
        self.LATENT_DIM = 128
        self.EMBEDDING_DIM = 256
        self.TEST_TRAIN_SPLIT = 0.15
        self.LEARNING_RATE = 0.0001
        self.FILE_NAME = "originals-l.csv"
        self.MAX_TEXT_LEN = 20
        self.MAX_SUMMARY_LEN = 10
        self.UNCOMMON_WORD_THRESHOLD = 100
        self.COLAB = False  # true if running on colab
        self.build_number = "1"

    def read_and_clean_data(self):
        if self.COLAB:
            from google.colab import drive
            drive.mount('/content/drive')
            self.PATH = "./drive/My Drive/"
        self.df = df.pd_read_csv(PATH + FILE_NAME)
        print(self.df.head())
        print(self.df.count)

        """Remove .'s that appear in stuff like U.S.A and U.N - Eventually need to move this to dataprocessing.py"""
        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(
            lambda x: re.sub(r'\..*$', ' ', str(x)))
        print(self.df['summary'][0])

        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(
            lambda x: re.sub(r'\.', '', str(x)))
        print(self.df['summary'][0])

        """Check for rows with null values in them, and copy these into a new dataframe (df1). Drop any rows in df1 from df to ensure no NaN valued rows are present/
        *Note. using simply dropna(how='any') does not seem to drop any of the rows*"""
        print(self.df.isnull().values.any())
        print(self.df.shape)

        df1 = self.df[self.df.isna().any(axis=1)]
        print(df1.shape)

        self.df.drop(df1.index, axis=0, inplace=True)
        print(self.df.shape)
        print(self.df.isnull().values.any())
        self.word_count_distribution(self.df['text'], self.df['summary'], "precutdown")


    def word_processing(self, word_removal):
        """Cut down text to MAX_TEXT_LEN words, and summaries to MAX_SUMMARY_LEN"""
        print(df['text'][0])
        df['text'] = df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join(x[:MAX_TEXT_LEN]))
        print(df['text'][0])

        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join(x[:self.MAX_SUMMARY_LEN]))
        print(self.df['summary'][0])
        self.word_count_distribution(self.df['text'], self.df['summary'], "cutdown")
        if word_removal:
            self.df['text'] = self.infrequent_word_removal(self.df['text'])
            self.df['summary'] = self.infrequent_word_removal(self.df['summary'])
            self.word_count_distribution(self.df['text'], self.df['summary'], "word_removal")

        """Update Max Text Lengths"""
        self.MAX_TEXT_LEN = max([len(txt.split(' ')) for txt in self.df['text']])
        self.MAX_SUMMARY_LEN = max([len(txt.split(' ')) for txt in self.df['summary']])
        print(self.MAX_TEXT_LEN)
        print(self.MAX_SUMMARY_LEN)

    def infrequent_word_removal(self, dataframe):
        word_dict = {}
        text = dataframe.apply(lambda x: nltk.word_tokenize(x))

        for index, row in text.iteritems():
            for word in row:
                if word not in word_dict.keys():
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1

        print(len(word_dict))
        sorted_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        print(sorted_dict)
        x, y = zip(*sorted_dict)

        # only accept words that occur more than UNCOMMON_WORD_THRESHOLD times
        accept_words = []
        for word, occ in sorted_dict:
            if int(occ) > self.UNCOMMON_WORD_THRESHOLD:
                accept_words.append(word)
            else:
                break

        accept_words = [x.lower() for x in accept_words]
        print(accept_words)
        print(text[2])
        text = text.apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if word.lower() in accept_words]))
        print(text[2])

        word_dict_after = {}
        text2 = text.apply(lambda x: nltk.word_tokenize(x))

        for index, row in text2.iteritems():
            for word in row:
                if word not in x_word_dict_after.keys():
                    x_word_dict_after[word] = 1
                else:
                    x_word_dict_after[word] += 1

        print(len(x_word_dict_after))

        return text

    def word_count_distribution(self, text, summary, desc):
        """Word Count Distribution"""
        text_word_count = []
        summary_word_count = []

        # populate the lists with sentence lengths
        for i in text:
            text_word_count.append(len(i.split(' ')))

        for i in summary:
            summary_word_count.append(len(i.split(' ')))

        length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

        length_df.hist(bins = 30)
        plt.ylabel('Documents')
        plt.xlabel('Word Count')
        plt.savefig('word_count_distro_model' + str(self.build_number) + str(desc) + '.png')


    def training_validation_split(self):
        """Training-Validation Split
            X - Articles text 
            Y - Summaries
        """
        # convert to numpy array
        self.X = np.array(self.df['text'])
        self.Y = np.array(self.df['summary'])

        x_tr,x_val,y_tr,y_val=train_test_split(self.X,self.Y,test_size=self.TEST_TRAIN_SPLIT,random_state=0,shuffle=True)
        print(x_tr.shape)
        print(x_val.shape)
        print(y_tr.shape)
        print(y_val.shape)
        return x_tr,x_val,y_tr,y_val

    def get_df(self):
        return self.df
    
    def get_x(self):
        return self.X
    
    def get_y(self):
        return self.Y