import keras
import pandas
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Add
from sklearn.model_selection import train_test_split

class Seq2SeqRNN():
    def __init__(self, dataframe, vocab_size):
        self.df = dataframe
        self.df_shape = self.df.shape
        self.vocab_size = vocab_size
        self.units = 200
        self.epochs = 1
        self.batch_size = 5
        # LSTM - units = dimentionality of the output space  

    def encoder(self):
        # use bidirectional RNN -> LSTM
        self.encoder_LSTM = LSTM(self.units, return_state=True, name="encoder_lstm")
        inputs = Input(shape=self.df_shape)
        outputs, state_h, state_c = self.encoder_LSTM(inputs)
        encoder_states = [state_h, state_c]
        return inputs, outputs, encoder_states
       

    def decoder(self, encoder_states):
        self.decoder_LSTM = LSTM(self.units, return_sequences=True, return_state=True, name="decoder_lstm")
        inputs =  Input(shape=self.df_shape)
        outputs, _, _ = self.decoder_LSTM(inputs, initial_state=encoder_states)
        dense = Dense(self.units, activation='softmax')
        outputs = dense(outputs)
        return inputs, outputs

    def run_model(self):
        en_inputs, en_ouputs, encoder_states = self.encoder()
        dec_inputs, dec_outputs = self.decoder(encoder_states)
        self.model = Model([en_inputs, dec_inputs], dec_outputs)
        self.model.compile(optimizer = 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        X = self.df['text']
        Y = self.df['summary']

        #Use train_test_split to split arrays or matrices into random train and test subsets
        # X need to be made into a np array or such
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test))
