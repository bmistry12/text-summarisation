import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

build_num = 1

class WordEmbeddings():
    def __init__(self, df, max_text_len, max_summary_len):
        self.df = df
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len


    def main(self):
        text_word_count = []
        summary_word_count = []

        # populate the lists with sentence lengths
        for i in self.df['text']:
            text_word_count.append(len(i.split(' ')))

        for i in self.df['summary']:
            summary_word_count.append(len(i.split(' ')))

        length_df = pd.DataFrame(
            {'text': text_word_count, 'summary': summary_word_count})

        length_df.hist(bins=30)
        plt.savefig('word_count_distro' + str(build_num) + '.png')


    def getMaxSize(self):
        max_text_seq_length = max([len(txt) for txt in self.df['text']])
        max_summary_seq_length = max([len(txt) for txt in self.df['summary']])
        return(max_text_seq_length, max_summary_seq_length)


    def set_x_tokenizer(self, x_data, x_tr, x_val):
        self.x_tokenizer = Tokenizer()
        self.x_tokenizer.fit_on_texts(list(x_data))

        # convert text sequences into integer sequences
        x_tr_seq = self.x_tokenizer.texts_to_sequences(x_tr)
        x_val_seq = self.x_tokenizer.texts_to_sequences(x_val)
        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr_seq,  maxlen=self.max_text_len, padding='post')
        x_val = pad_sequences(x_val_seq, maxlen=self.max_text_len, padding='post')
        # size of vocabulary ( +1 for padding token)
        x_voc = self.x_tokenizer.num_words + 1
        print(x_voc)


    def get_x_tokenizer(self):
        return self.x_tokenizer


    def set_y_tokenizer(self, y_data, y_tr, y_val):
        self.y_tokenizer = Tokenizer()
        self.y_tokenizer.fit_on_texts(list(y_data))

        # convert text sequences into integer sequences
        y_tr_seq = self.x_tokenizer.texts_to_sequences(y_tr)
        y_val_seq = self.x_tokenizer.texts_to_sequences(y_val)
        # padding zero upto maximum length
        y_tr = pad_sequences(
            y_tr_seq,  maxlen=self.max_summary_len, padding='post')
        y_val = pad_sequences(
            y_val_seq, maxlen=self.max_summary_len, padding='post')
        # size of vocabulary ( +1 for padding token)
        y_voc = self.x_tokenizer.num_words + 1
        print(y_voc)


    def get_y_tokenizer(self):
        return self.y_tokenizer


    def set_word_indexs(self):
        self.reverse_target_word_index = self.y_tokenizer.index_word
        self.reverse_source_word_index = self.x_tokenizer.index_word


    def seq2summary(self, input_seq):
        newString = ''
        for i in input_seq:
            if(i != 0):
                newString = newString+self.reverse_target_word_index[i]+' '
        return newString


    def seq2text(self, input_seq):
        newString = ''
        for i in input_seq:
            if(i != 0):
                newString = newString+self.reverse_source_word_index[i]+' '
        return newString


    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)
        # print(e_out)
        target_seq = np.zeros((1, 1))
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            if (sampled_token_index != 0):
                sampled_token = self.reverse_target_word_index[sampled_token_index]
                decoded_sentence += ' '+sampled_token
            else:
                stop_condition = True
            if (len(decoded_sentence.split()) >= (self.max_summary_len-1)):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c
        return decoded_sentence
