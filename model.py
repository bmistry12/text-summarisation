import keras
import pandas
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Concatenate, Embedding
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.seqeuence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


class Seq2SeqRNN():
    def __init__(self, dataframe, vocab_size, batch_size, epochs, latent_dim, embedding_dim, traintest_split):
        self.df = dataframe
        self.df_shape = self.df.shape
        self.max_text_seq_length, self.max_summary_seq_length = vocab_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.traintest_split = traintest_split
        # LSTM - units = dimentionality of the output space

    def model(self, dropout):
        # encoder
        encoder_inputs = Input(shape=(self.max_text_seq_length, ))
        encoder_embedding = Embedding(x_vocab, self.embedding_dim, trainable=True)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        # decoder, use encoder_states as initial state.
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(y_vocab, self.embedding_dim, trainable=True)(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, dropout=dropout, recurrent_dropout=dropout/2)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

        # dense layer
        decoder_dense = Dense(y_vocab, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # model
        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.training_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        print(model.summary())
        
    def inference_model(self)
        reverse_target_word_index = y_tokenizer.index_word
        reverse_source_word_index = x_tokenizer.index_word
        target_word_index = y_tokenizer.word_index

        # Encode the input sequence to get the feature vector
        self.encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_text_len, self.latent_dim))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_inputs)

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)

        # Final decoder model
        self.decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])
        self.decoder_model.summary()

    def seq2summary(self, input_seq):
        newString = ''
        for i in input_seq:
            if(i != 0):
                newString = newString+reverse_target_word_index[i]+' '
        return newString


    def seq2text(self, input_seq):
        newString = ''
        for i in input_seq:
            if(i != 0):
                newString = newString+reverse_source_word_index[i]+' '
        return newString

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)
        # print(e_out)
        target_seq = np.zeros((1, 1))
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if (sampled_token_index != 0):
                sampled_token = reverse_target_word_index[sampled_token_index]
                decoded_sentence += ' '+sampled_token
            else:
                stop_condition = True
            if (len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c
        return decoded_sentence 
    
    def run_model(self):
        self.model(0.4)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=4)
        history=self.training_model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

        self.inference_model()

        for i in range(0,5):
            print("Article:",seq2text(x_tr[i]))
            print("Original summary:",seq2summary(y_tr[i]))
            print("Generated summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
            print("\n")
       