import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

class ReadWriteData():
    """
        Read in the data - read each text file that we are processing into a pd dataframe 
        Write out data to a specified CSV file
    """

    def __init__(self, loc):
        self.datapath = loc
        self.file_names = [file for file in os.listdir(
            loc) if file.endswith('.story')]
        self.df = pd.DataFrame(columns=['file', 'text', 'summary'])

    def read_in_files(self, filesToRead):
        """
            Read in files from datapath defined by self.datapath
            @filesToRead = The number of files to read in from the directory
        """
        files_to_read = self.file_names[:filesToRead]
        print(len(files_to_read))
        for file in files_to_read:
            with open(self.datapath + "/" + file, encoding="utf8") as f:
                data = f.read()
                # 0 - body, 1 - summary
                data_split = re.split("@highlight", data, maxsplit=1)
                # appending to df
                self.df = self.df.append({'file': file, 'text': str(
                    data_split[0]), 'summary': str(data_split[1])}, ignore_index=True)
        print("read in files")
        print(self.df.head())

    def get_df(self):
        """
            Return panda dataframe in it's current state
        """
        return self.df

    def df_to_csv(self, csv_name):
        """
            Convert a panda dataframe to a csv with the name @csv_name
        """
        self.df.to_csv(csv_name, index=True)


class CleanData():
    """
        Use the data that is read in, as a pd df, and clean it - remove any whitespace and empty columns
        Shape the data as necessary
        Removal of stop words - Could this cause an issue when dealing with creating gramatically correct summaries?
        Lemmatization - reduing words down to the stem forms
    """

    def __init__(self, dataframe):
        self.df = dataframe

    def sent_pos_clean(self, doc):
        """
            Cleaning Data when sentPos=True. We pass in an array of tokenized sentences that need to be cleaned and have <eos> tokens appended to the end.
        """
        print("sent_pos_clean")
        doc = [re.sub(r'\(CNN\)|(Daily\sMail)|--|[^\w\s\.]', '', x) for x in doc]
        doc = [re.sub(r'(\.(?=[\s\r\n]|$))', '', x) for x in doc]
        doc = [re.sub(r'\n', ' ', x) for x in doc]
        doc = [re.sub(r'\.', '', x) for x in doc]
        # add eos token so that we can split the document into sentences easier in sent_position method
        doc = [x + ' <eos>' for x in doc]
        doc = "".join(doc)
        return doc

    def clean_data(self, textRank, wordFreq, sentPos):
        """
            Clean data by removing punctuation and words relating to the source of the article
            @textRank = True if text rank is being run. In this case the summary seperator @highlight is not removed
            @wordFreq = True if word frequency is being run. In this case the summary seperator @highlight is not removed
            @sentPos = True if sentence position is being run. In this case sent_pos_clean is called and <eos> tokens are added to the end of each sentence.
        """
        # dropping duplicates
        self.df.drop_duplicates(subset=['file'], inplace=True)
        self.df.dropna(axis=0, inplace=True)  # dropping na
        # clean texts
        if (sentPos == "True"):
            print("sent position clean")
            # add in eos tokens
            self.df['text'] = self.df['text'].apply(lambda x: nltk.sent_tokenize(x, language='english')).apply(lambda x: self.sent_pos_clean(x))            
        else: 
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\(CNN\)|(Daily\sMail)|--|[^\w\s\.]', '', x)).apply(lambda x: re.sub(r'(\.(?=[\s\r\n]|$))', '', x)).apply(lambda x: re.sub(r'\n', ' ', x)).apply(lambda x: re.sub(r'\.', '', x))
        # separate the summaries using a '.'
        if (textRank == "True") or (wordFreq == "True"):
            self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n|[^\w\s\.\@]', '', x))
        else:
            self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n|[^\w\s\.\@]', '', x)).apply(lambda x: re.sub(r'@highlight', ' ', x))
        print("cleaned data")
        print(self.df.head())

    def remove_stop_words(self):
        """
            Remove stop words from the text	and summaries using nltk stopwords
        """
        stop_words = set(stopwords.words('english'))
        self.df['text'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(
            lambda x: " ".join([word for word in x if not word.lower() in stop_words]))
        self.df['summary'] = self.df['summary'].apply(lambda x: nltk.word_tokenize(x)).apply(
            lambda x: " ".join([word for word in x if not word.lower() in stop_words]))
        print(self.df.head())
        print("removed stop words")

    def get_pos(self, word):
        """
            Get nltk part of speech tag for a token, and translate it to a wordnet part of speech symbol
            Note: WordNet POS does not recognise "I" - Preposition, "M" - modal, "C" - conjunction, "P" - pronoun
        """
        pos = nltk.pos_tag([word])[0][1][0]
        wordnet_conv = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
        if pos in wordnet_conv.keys():
            return wordnet_conv.get(pos)
        return ""

    def lemmatization(self, pos):
        """
            Lemmatization of articles using the WordNet Lemmatizer. This reduces tokens down to their stem form.
            @pos = Value can be True or False. Used to indicate whether or not to use POS whilst lemmatizing
        """
        # initialise wordnet lemmatizer
        lemmatizer = WordNetLemmatizer()
        text_tokenized = self.df['text'].apply(lambda x: nltk.word_tokenize(x))
        if pos == "True":
            print("lemmatize with pos")
            for i in range(0, len(text_tokenized)):
                text_lemmatized = []
                # loop through each word, get its pos, and then lemmatize it
                for word in text_tokenized[i]:
                    self.get_pos(word)
                    pos = self.get_pos(word)
                    if pos != "":
                        lemma = lemmatizer.lemmatize(word, pos)
                        text_lemmatized.append(lemma)
                    else:
                        # if it has no pos token, simply lemmatize it without
                        text_lemmatized.append(word)
                text_lemmatized = ' '.join(map(str, text_lemmatized))
                # replace original text with the lemmatized form
                self.df['text'][i] = text_lemmatized
        else:
            print("lemmatize w/o POS")
            # This currently doesn't output text in the right format (I think)
            self.df['text'] = text_tokenized.apply(
                lambda x: [lemmatizer.lemmatize(w) for w in x])
            self.df['text'] = self.df['text'].apply(lambda x: ' '.join(x))

    def drop_null_rows(self):
        """
            Check for rows with null values in them, and copy these into a new dataframe (df1). 
            Drop any rows in df1 from df to ensure no NaN valued rows are present/
            *Note. using simply dropna(how='any') does not seem to drop any of the rows*
        """
        print(self.df.isnull().values.any())
        print(self.df.shape)

        df1 = self.df[self.df.isna().any(axis=1)]
        print(df1.shape)

        self.df.drop(df1.index, axis=0, inplace=True)
        print(self.df.shape)
        print(self.df.isnull().values.any())