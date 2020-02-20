import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
# from nltk.stem import WordNetLemmatizer
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
        # Set Hyperparameters
        self.PATH = ""
        self.BATCH_SIZE = 30
        self.EPOCHS = 50
        self.LATENT_DIM = 128
        self.EMBEDDING_DIM = 256
        self.TEST_TRAIN_SPLIT = 0.15
        self.LEARNING_RATE = 0.0001
        self.FILE_NAME = "originals-l.csv" # csv data to run model against
        self.MAX_TEXT_LEN = 20
        self.MAX_SUMMARY_LEN = 10
        self.UNCOMMON_WORD_THRESHOLD = 100
        self.COLAB = False  # true if running on colab
        self.build_number = "1"
        self.rouge = Rouge()

    def read_and_clean_data(self):
        # Read in CSV file
        if self.COLAB:
            from google.colab import drive
            drive.mount('/content/drive')
            self.PATH = "./drive/My Drive/"
        # CSV -> PD DataFrame
        self.df = pd.read_csv(self.PATH + self.FILE_NAME)
        # Head of df
        print(self.df.head())
        print(self.df.count)

        # Remove .'s that appear in stuff like U.S.A and U.N  from summaries - Eventually need to move this to dataprocessing.py
        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(
            lambda x: re.sub(r'\..*$', ' ', str(x)))
        print(self.df['summary'][0])

        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(
            lambda x: re.sub(r'\.', '', str(x)))
        print(self.df['summary'][0])

        # drop null rows
        self.drop_null_rows()
        # word count distro graph before any word processing
        self.word_count_distribution(self.df['text'], self.df['summary'], "precutdown")

    def drop_null_rows(self):
        """Check for rows with null values in them, and copy these into a new dataframe (df1). 
        Drop any rows in df1 from df to ensure no NaN valued rows are present/
        *Note. using simply dropna(how='any') does not seem to drop any of the rows*"""
        print(self.df.isnull().values.any())
        print(self.df.shape)

        df1 = self.df[self.df.isna().any(axis=1)]
        print(df1.shape)

        self.df.drop(df1.index, axis=0, inplace=True)
        print(self.df.shape)
        print(self.df.isnull().values.any())

    def word_processing(self, word_removal):
        """
            Process the data by cutting down each artcile and summary to MAX_TEXT_LEN and MAX_SUMMARY_LEN respectively.
            If @word_removal = True, remove infrequent words
        """
        # Cut down text to MAX_TEXT_LEN words, and summaries to MAX_SUMMARY_LEN
        print(self.df['text'][0])
        self.df['text'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join(x[:self.MAX_TEXT_LEN]))
        print(self.df['text'][0])

        print(self.df['summary'][0])
        self.df['summary'] = self.df['summary'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join(x[:self.MAX_SUMMARY_LEN]))
        print(self.df['summary'][0])
        self.word_count_distribution(self.df['text'], self.df['summary'], "cutdown")
        # if we're removing uncommon words call the infrequent_word_removal method
        if word_removal:
            self.df['text'] = self.infrequent_word_removal(self.df['text'])
            self.df['summary'] = self.infrequent_word_removal(self.df['summary'])
            self.word_count_distribution(self.df['text'], self.df['summary'], "word_removal")

        """Update Max Text Lengths"""
        self.MAX_TEXT_LEN = max([len(txt.split(' ')) for txt in self.df['text']])
        self.MAX_SUMMARY_LEN = max([len(txt.split(' ')) for txt in self.df['summary']])
        print(self.MAX_TEXT_LEN)
        print(self.MAX_SUMMARY_LEN)
        # drop any null rows from word removal that may have occured
        self.drop_null_rows()
        # add in start and end tokens to summaries
        self.df['summary'] = self.df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')
        print(self.df['summary'].head())

    def infrequent_word_removal(self, dataframe):
        """
            Finding Uncommon Words and Removing Them.
            Uncommon words are classified as those that occur in the whole corpus less times than UNCOMMON_WORD_THRESHOLD
            The corpus is passed in as a Panda Dataframe
        """
        word_dict = {}
        text = dataframe.apply(lambda x: nltk.word_tokenize(x))

        for _, row in text.iteritems():
            for word in row:
                if word not in word_dict.keys():
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        # Check vocab size before word removal
        print("Word Count Before Uncommon Word Removal: ")
        print(len(word_dict))
        sorted_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        print(sorted_dict)
        # x, y = zip(*sorted_dict)

        # only accept words that occur more than UNCOMMON_WORD_THRESHOLD times
        accept_words = []
        for word, occ in sorted_dict:
            if int(occ) > self.UNCOMMON_WORD_THRESHOLD:
                accept_words.append(word)
            else:
                break
        # remove uncommon words  
        accept_words = [x.lower() for x in accept_words]
        print(accept_words)
        print(text[2])
        text = text.apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if word.lower() in accept_words]))
        print(text[2])

        # Check vocab size after word removal
        word_dict_after = {}
        text2 = text.apply(lambda x: nltk.word_tokenize(x))

        for _, row in text2.iteritems():
            for word in row:
                if word not in word_dict_after.keys():
                    word_dict_after[word] = 1
                else:
                    word_dict_after[word] += 1

        print("Word Count After Uncommon Word Removal: ")
        print(len(word_dict_after))
        # return the new dataframe that has had uncommon words removed
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
        """
            Training-Validation Split
            Split the data into X and Y, where Y accounts for TEST_TRAIN_SPLIT of the total data
            
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
        # return training and validation splits
        return x_tr,x_val,y_tr,y_val

    def seq_to_text(self, input_seq, reverse_word_index, target_word_index, summary):
        """
            Convert vectorized article/summaries back into text 
            @reverse_word_index = reverse_target_word_index for summaries, reverse_source_word_index for articles
            @summary =  True if summary 
        """
        textString=''
        if not summary:
            for i in input_seq:
                if(i!=0):
                    textString = textString + ' ' + reverse_word_index[i]
        else:
            for i in input_seq:
                if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
                    newString=newString+reverse_word_index[i]+' '
        return textString

    # def decode_sequence(self, input_seq, encoder_model, decoder_model, reverse_target_word_index, y_voc): 
    #     """
    #         Standard Decode Sequence Method - selects most probable word using argmax
    #     """
    #     # Encode the input as state vectors.
    #     e_out, e_h, e_c = encoder_model.predict(input_seq)
    #     target_seq = np.zeros((1, y_voc))
    #     stop_condition = False
    #     decoded_sentence = ''

    #     while not stop_condition:
    #         # Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs2] + decoder_states)
    #         output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c]) 
    #         sampled_token_index = np.argmax(output_tokens[0, -1, :][1:])
    #         # print(output_tokens[0, -1, :].shape) (y_voc, None)
    #         if (sampled_token_index != 0 ):
    #             sampled_token = reverse_target_word_index[sampled_token_index]
    #             decoded_sentence += ' '+sampled_token
    #         else :
    #             print("sadface")
    #             stop_condition = True
    #         if (len(decoded_sentence.split()) >= (self.MAX_SUMMARY_LEN-1)):
    #                 stop_condition = True
    #         # Update the target sequence for next input
    #         target_seq = np.zeros((1, y_voc))
    #         target_seq[0, sampled_token_index] = 1
    #         # Update internal states
    #         e_h, e_c = h, c
    #     return decoded_sentence
    
    def decode_sequence(self, input_seq, encoder_model, decoder_model, reverse_target_word_index, target_word_index):
        """
            Standard Decode Sequence Method - selects most probable word using argmax
        """
        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1
        target_seq = np.zeros((1,1))
        # First word of target sequence is the start token = sostok
        target_seq[0, 0] = target_word_index['sostok']
        stop_condition = False
        generated_sentence = ''

        while not stop_condition:
            # Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs2] + decoder_states)
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            if(sampled_token != 'eostok'):
                generated_sentence += ' '+sampled_token
            # If max length or end of sentence token found then stop
            if (sampled_token == 'eostok'  or len(generated_sentence.split()) >= (self.MAX_SUMMARY_LEN-1)):
                stop_condition = True
            # Update the target sequence for next input
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            # Update internal states
            e_h, e_c = h, c
        return generated_sentence

    def evaluation(self, x, y, reverse_target_word_index, target_word_index, encoder_model, decoder_model): 
        """
            Evaluate the model against the whole training or validation dataset (defined by x, y)
            Returns combined Rouge-1 F, P and R score
        """
        target_summary = []
        generated_summary = []
        x_len = len(x)

        f_ov = 0
        p_ov = 0
        r_ov = 0
        # x_val_len = 1
        for i in range(0,x_len):
            original = self.seq_to_text(y[i], reverse_target_word_index, target_word_index, False)
            if original != "" :
                target_summary.append(original)
                x_i = x[i].reshape(1,self.MAX_TEXT_LEN)
                summary = self.decode_sequence(x_i, encoder_model, decoder_model, reverse_target_word_index, target_word_index)
                print(i)
                print(original)
                print(summary)
                print("-----")
                generated_summary.append(summary)
                score = self.getRouge(str(summary), str(original))
                f_ov += float(score[0].get('rouge-1').get('f'))
                p_ov += float(score[0].get('rouge-1').get('p'))
                r_ov += float(score[0].get('rouge-1').get('r'))
        return f_ov, p_ov, r_ov

    def getRouge(self, gt, pred):
        """ 
            Get rouge score for a given ground truth and prediction string
        """
        return self.rouge.get_scores(pred, gt)

    def get_df(self):
        """
            Get dataframe
        """
        return self.df
    
    def get_x(self):
        """
            Get articles in numpy form - X
        """
        return self.X
    
    def get_y(self):
        """
            Get summaries in numpy form - Y
        """
        return self.Y