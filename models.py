import re
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Attention
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import modelCommon as modelCommon


class UniModel():
    def __init__(self, word_removal):
        self.common = modelCommon.Common()
        self.main(word_removal)

    def main(self, word_removal):
        self.common.read_and_clean_data()
        self.common.word_processing(word_removal)
        self.df = self.common.get_df()
        self.x_tr, self.x_val, self.y_tr, self.y_val = self.common.training_validation_split()
        self.X = self.common.get_x()
        self.Y = self.common.get_y()
        self.word_embeddings()
        # learning model
        model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense = self.learning_model()

        """
            - Early Stopping Callback to ensure we stop when Validation Loss is lowest - minimises risk of overfitting
            - Model Checkpoint saves the model after each epoch so that we can load the model with the best weights later on. Alternatively, it allows us to continue training the model at a later data
        """
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=2, restore_best_weights=False)
        filepath = "./model/saved-model-{epoch:02d}.hdf5"
        if self.common.COLAB:
            filepath = self.common.PATH + "project-model/saved-model-{epoch:02d}.hdf5"
        mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

        # Train a new model.
        ## reshape y from two dimensions to three dimensions
        y_tr_3d = self.y_tr.reshape(self.y_tr.shape[0], self.y_tr.shape[1], 1)[:, 1:]
        y_val_3d = self.y_val.reshape(self.y_val.shape[0], self.y_val.shape[1], 1)[:, 1:]

        history = model.fit([self.x_tr, self.y_tr[:, :-1]], y_tr_3d, batch_size=self.common.BATCH_SIZE, epochs=self.common.EPOCHS,
                            callbacks=[es, mc], validation_data=([self.x_val, self.y_val[:, :-1]], y_val_3d))

        # Plot training and validation loss overtime
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('loss_uni' + str(self.common.build_number) + '.png')

        # Get word index's from tokenizers
        reverse_source_word_index = self.x_tokenizer.index_word
        reverse_target_word_index = self.y_tokenizer.index_word
        print(reverse_source_word_index)
        print(reverse_target_word_index)
        # get max target word index
        target_word_index = self.y_tokenizer.word_index

        # inference model
        encoder_model, decoder_model = self.inference_model(encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense)
        # test model
        self.evaluate_model(reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model)

    def learning_model(self):
        """
            Learning Model
        """
        """Encoder Model"""
        encoder_inputs = Input(shape=(self.common.MAX_TEXT_LEN,))
        # embedding layer
        enc_emb =  Embedding(self.x_voc,self.common.EMBEDDING_DIM,trainable=True)(encoder_inputs)
        # unidirectional encoder lstm 
        encoder_lstm = LSTM(self.common.LATENT_DIM,return_sequences=True,return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        """#### Decoder Model"""
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        # embedding layer
        dec_emb_layer = Embedding(self.y_voc, self.common.EMBEDDING_DIM, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(self.common.LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=encoder_states)

        # dense layer
        decoder_dense = Dense(self.y_voc, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        """#### Combined LSTM Model"""

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        optimizer = RMSprop(lr=self.common.LEARNING_RATE, rho=0.9)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        return model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense

    def inference_model(self, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense):     
        """
            Inference Model
        """
        # Encoder Setup
        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        print(encoder_model.summary())

        # Decoder Setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.common.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.common.LATENT_DIM,))
        decoder_hidden_state_input = Input(shape=(self.common.MAX_TEXT_LEN, self.common.LATENT_DIM))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_inputs)
        decoder_states = [state_h2, state_c2]
        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        # Final Decoder model
        decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + decoder_states)
        print(decoder_model.summary())

        return encoder_model, decoder_model

    def word_embeddings(self):
        """
            Word Embeddings - Tokenization
        """
        """X Tokenizer"""
        word_dict = {}
        text = self.df['text']

        for row in text:
            for word in row.split(" "):
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        print(len(word_dict))

        # #prepare a tokenizer for reviews on training data
        self.x_tokenizer = Tokenizer(num_words=len(word_dict), split=" ")
        self.x_tokenizer.fit_on_texts(list(self.X))
        # convert text sequences into integer sequences
        x_tr_seq = self.x_tokenizer.texts_to_sequences(self.x_tr)
        x_val_seq = self.x_tokenizer.texts_to_sequences(self.x_val)
        # padding zero upto maximum length
        self.x_tr = pad_sequences(x_tr_seq,  maxlen=self.common.MAX_TEXT_LEN, padding='post')
        self.x_val = pad_sequences(x_val_seq, maxlen=self.common.MAX_TEXT_LEN, padding='post')

        # size of vocabulary ( +1 for padding token)
        self.x_voc = self.x_tokenizer.num_words + 1
        print(self.x_voc)

        """Y Tokenizer"""
        y_word_dict = {}
        summ = self.df['summary']

        for row in summ:
            for word in row.split(" "):
                if word not in y_word_dict:
                    y_word_dict[word] = 1
                else:
                    y_word_dict[word] += 1
        print(len(y_word_dict))

        # prepare a tokenizer for reviews on training data
        self.y_tokenizer = Tokenizer(num_words=len(y_word_dict), split=" ")
        self.y_tokenizer.fit_on_texts(list(self.Y))
        # convert text sequences into integer sequences
        y_tr_seq = self.y_tokenizer.texts_to_sequences(self.y_tr)
        y_val_seq = self.y_tokenizer.texts_to_sequences(self.y_val)
        # padding zero upto maximum length
        self.y_tr = pad_sequences(y_tr_seq, maxlen=self.common.MAX_SUMMARY_LEN, padding='post')
        self.y_val = pad_sequences(y_val_seq, maxlen=self.common.MAX_SUMMARY_LEN, padding='post')
        # size of vocabulary + 1 for padding
        self.y_voc = self.y_tokenizer.num_words + 1
        print(self.y_voc)

        # save tokenizers
        with open('xtokenizer_uni.pickle', 'wb') as handle:
            pickle.dump(self.x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('ytokenizer_uni.pickle', 'wb') as handle:
            pickle.dump(self.y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def evaluate_model(self, reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model):
        for i in range(0,1):
            article = self.common.seq_to_text(self.x_tr[i], reverse_source_word_index, target_word_index, False)
            original = self.common.seq_to_text(self.y_tr[i], reverse_target_word_index, target_word_index, True)

            if (original !=""):
                # reshape data into correct format for encoder
                x_tr_i_reshaped = self.x_tr[i].reshape(1,self.common.MAX_TEXT_LEN)
                summary = self.common.decode_sequence(x_tr_i_reshaped, encoder_model, decoder_model, reverse_target_word_index, self.y_voc)
                print("Article: " + article)
                print("Generated summary:",summary)
                print("\n")

                if summary != "":    
                    print("ROUGE score: ")
                    score = self.common.getRouge(str(summary), str(original))
                    print(score)
                    print(score[0].get('rouge-1').get('f'))
                    print(score[0].get('rouge-1').get('p'))
                    print(score[0].get('rouge-1').get('r'))


class BiModel():
    def __init__(self, word_removal):
        self.common = modelCommon.Common()
        self.main(word_removal)

    def main(self, word_removal):
        # read in data and clean it
        self.common.read_and_clean_data()
        # word procesing
        self.common.word_processing(word_removal)
        self.df = self.common.get_df()
        # test training split
        self.x_tr, self.x_val, self.y_tr, self.y_val = self.common.training_validation_split()
        # note X and Y are only defined after common.training_validation_split is carried out
        self.X = self.common.get_x()
        self.Y = self.common.get_y()
        # word tokenization
        self.word_embeddings()
        # learning model
        model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense = self.learning_model()

        """
            - Early Stopping Callback to ensure we stop when Validation Loss is lowest - minimises risk of overfitting
            - Model Checkpoint saves the model after each epoch so that we can load the model with the best weights later on. Alternatively, it allows us to continue training the model at a later data
        """
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=2, restore_best_weights=False)
        filepath = "./model/saved-model-{epoch:02d}.hdf5"
        if self.common.COLAB:
            filepath = self.common.PATH + "project-model/saved-model-{epoch:02d}.hdf5"
        mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

        # Train a new model.
        ## reshape y from two dimensions to three dimensions
        y_tr_3d = self.y_tr.reshape(self.y_tr.shape[0], self.y_tr.shape[1], 1)[:, 1:]
        y_val_3d = self.y_val.reshape(self.y_val.shape[0], self.y_val.shape[1], 1)[:, 1:]

        history = model.fit([self.x_tr, self.y_tr[:, :-1]], y_tr_3d, batch_size=self.common.BATCH_SIZE, epochs=self.common.EPOCHS,
                            callbacks=[es, mc], validation_data=([self.x_val, self.y_val[:, :-1]], y_val_3d))

        # Plot training and validation loss overtime
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('loss_bi' + str(self.common.build_number) + '.png')

        # Get word index's from tokenizers
        reverse_source_word_index = self.x_tokenizer.index_word
        reverse_target_word_index = self.y_tokenizer.index_word
        print(reverse_source_word_index)
        print(reverse_target_word_index)
        # get max target word index
        target_word_index = self.y_tokenizer.word_index

        # inference model
        encoder_model, decoder_model = self.inference_model(encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense)
        # test model
        self.evaluate_model(reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model, ev_val=True, ev_tr=True)

    def learning_model(self):
        """
            Learning Model
        """
        """Encoder Model"""
        encoder_inputs = Input(shape=(self.common.MAX_TEXT_LEN,))
        # embedding layer
        enc_emb = Embedding(self.x_voc, self.common.EMBEDDING_DIM, trainable=True)(encoder_inputs)
        # bidirectional encoder lstm
        encoder_lstm = Bidirectional(LSTM(self.common.LATENT_DIM, return_sequences=True, return_state=True))
        encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = encoder_lstm(enc_emb)

        state_h = Concatenate()([fw_state_h, bw_state_h])
        state_c = Concatenate()([fw_state_c, bw_state_c])
        encoder_states = [state_h, state_c]

        """#### Decoder Model"""
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        # embedding layer
        dec_emb_layer = Embedding(self.y_voc, self.common.EMBEDDING_DIM, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(self.common.LATENT_DIM*2, return_sequences=True, return_state=True)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=encoder_states)

        # dense layer
        decoder_dense = Dense(self.y_voc, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        """#### Combined LSTM Model"""

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        optimizer = RMSprop(lr=self.common.LEARNING_RATE, rho=0.9)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        return model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense

    def inference_model(self, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense):     
        """
            Inference Model
        """
        # Encoder Setup
        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        print(encoder_model.summary())

        # Decoder Setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.common.LATENT_DIM*2,))
        decoder_state_input_c = Input(shape=(self.common.LATENT_DIM*2,))
        decoder_hidden_state_input = Input(shape=(self.common.MAX_TEXT_LEN, self.common.LATENT_DIM*2))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_inputs)
        decoder_states = [state_h2, state_c2]
        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        # Final Decoder model
        decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + decoder_states)
        print(decoder_model.summary())
        # save encoder and decoder
        encoder_model.save(self.common.PATH + "encoder_model.h5")
        decoder_model.save(self.common.PATH + "decoder_model.h5")
        return encoder_model, decoder_model

    def word_embeddings(self):
        """
            Word Embeddings - Tokenization
        """
        """X Tokenizer"""
        word_dict = {}
        text = self.df['text']

        for row in text:
            for word in row.split(" "):
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        print(len(word_dict))

        # #prepare a tokenizer for reviews on training data
        self.x_tokenizer = Tokenizer(num_words=len(word_dict), split=" ")
        self.x_tokenizer.fit_on_texts(list(self.X))
        # convert text sequences into integer sequences
        x_tr_seq = self.x_tokenizer.texts_to_sequences(self.x_tr)
        x_val_seq = self.x_tokenizer.texts_to_sequences(self.x_val)
        # padding zero upto maximum length
        self.x_tr = pad_sequences(x_tr_seq,  maxlen=self.common.MAX_TEXT_LEN, padding='post')
        self.x_val = pad_sequences(x_val_seq, maxlen=self.common.MAX_TEXT_LEN, padding='post')

        # size of vocabulary ( +1 for padding token)
        self.x_voc = self.x_tokenizer.num_words + 1
        print(self.x_voc)

        """Y Tokenizer"""
        y_word_dict = {}
        summ = self.df['summary']

        for row in summ:
            for word in row.split(" "):
                if word not in y_word_dict:
                    y_word_dict[word] = 1
                else:
                    y_word_dict[word] += 1
        print(len(y_word_dict))

        # prepare a tokenizer for reviews on training data
        self.y_tokenizer = Tokenizer(num_words=len(y_word_dict), split=" ")
        self.y_tokenizer.fit_on_texts(list(self.Y))
        # convert text sequences into integer sequences
        y_tr_seq = self.y_tokenizer.texts_to_sequences(self.y_tr)
        y_val_seq = self.y_tokenizer.texts_to_sequences(self.y_val)
        # padding zero upto maximum length
        self.y_tr = pad_sequences(y_tr_seq, maxlen=self.common.MAX_SUMMARY_LEN, padding='post')
        self.y_val = pad_sequences(y_val_seq, maxlen=self.common.MAX_SUMMARY_LEN, padding='post')
        # size of vocabulary + 1 for padding
        self.y_voc = self.y_tokenizer.num_words + 1
        print(self.y_voc)
        print("Number of start and end tokens: ")
        print(self.y_tokenizer.word_counts['sostok'], self.y_tokenizer.word_counts['eostok'])

        # save tokenizers
        with open('xtokenizer_bi.pickle', 'wb') as handle:
            pickle.dump(self.x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('ytokenizer_bi.pickle', 'wb') as handle:
            pickle.dump(self.y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate_model(self, reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model, ev_val, ev_tr):
        """
            Evaluate the model - 10 test examples from the training data fully printed
            @ev_val: True - Run evaluation to get ROUGE metrics for validation data
            @ev_tr: True - Run evaluation to get ROUGE metrics for training data
        """
        for i in range(0,10):
            article = self.common.seq_to_text(self.x_tr[i], reverse_source_word_index, target_word_index, False)
            original = self.common.seq_to_text(self.y_tr[i], reverse_target_word_index, target_word_index, True)

            if (original !=""):
                # # feed the first word of the original summary into the encoder (used for decode_seq3)
                # a = nltk.word_tokenize(original)[0]
                # for val,word in reverse_target_word_index.items():
                #     if word == a:
                #         i = val
                print("Article: " + article)
                print("Original summary:", original)
                # reshape data into correct format for encoder (1, max_text_len)
                x_tr_i_reshaped = self.x_tr[i].reshape(1,self.common.MAX_TEXT_LEN)
                summary = self.common.decode_sequence(x_tr_i_reshaped, encoder_model, decoder_model, reverse_target_word_index, target_word_index)
                print("Generated summary:",summary)
                print("\n")

                if summary != "":    
                    print("ROUGE score: ")
                    score = self.common.getRouge(str(summary), str(original))
                    print(score)
                    print(score[0].get('rouge-1').get('f'))
                    print(score[0].get('rouge-1').get('p'))
                    print(score[0].get('rouge-1').get('r'))
        if ev_val:
            # evaluatate on validation data
            x_val_len = len(self.x_val)
            print("Evaluate against validation data: " + str(x_val_len))
            f_ov_val, p_ov_val, r_ov_val = self.common.evaluation(self.x_val, self.y_val, reverse_target_word_index, target_word_index, encoder_model, decoder_model)
            print("Avg Val F Score: " + str(f_ov_val/x_val_len))
            print("Avg Val Precision: " + str(p_ov_val/x_val_len))
            print("Avg Val Recall: " + str(r_ov_val/x_val_len))
        if ev_tr:
            # evaluate on training data
            x_tr_len = len(self.x_tr)
            print("Evaluate against training data: " + str(x_tr_len))
            f_ov_tr, p_ov_tr, r_ov_tr = self.common.evaluation(self.x_tr, self.y_tr, reverse_target_word_index, target_word_index, encoder_model, decoder_model)
            print("Avg Tr F Score: " + str(f_ov_tr/x_tr_len))
            print("Avg Tr Precision: " + str(p_ov_tr/x_tr_len))
            print("Avg Tr Recall: " + str(r_ov_tr/x_tr_len))
            

class GloveModel():
    def __init__(self, word_removal):
        self.common = modelCommon.Common()
        self.main(word_removal)

    def main(self, word_removal):
        self.common.read_and_clean_data()
        self.common.word_processing(word_removal)
        self.df = self.common.get_df()
        self.x_tr, self.x_val, self.y_tr, self.y_val = self.common.training_validation_split()
        self.X = self.common.get_x()
        self.Y = self.common.get_y()
        # get emebedding index
        embedding_index = self.glove()
        # create word embeddings using GloVe
        x_embedding_matrix, y_embedding_matrix, x_word_dict, y_word_dict = self.word_embeddings(embedding_index)
    
        # GloVe Word Coverage
        text_total = len(x_word_dict)
        text_covered = self.coverage(x_word_dict,text_total, embedding_index)

        summ_total = len(y_word_dict)
        summ_covered = self.coverage(y_word_dict,summ_total, embedding_index)

        print("Original Text Coverage: " + str(text_covered) + "%")
        print("Summary Coverage: " + str(summ_covered) + "%")

        # learning model
        model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense = self.learning_model(
            x_embedding_matrix, y_embedding_matrix)

        """
            - Early Stopping Callback to ensure we stop when Validation Loss is lowest - minimises risk of overfitting
            - Model Checkpoint saves the model after each epoch so that we can load the model with the best weights later on. Alternatively, it allows us to continue training the model at a later data
        """
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=2, restore_best_weights=False)
        filepath = "./model/saved-model-{epoch:02d}.hdf5"
        if self.common.COLAB:
            filepath = self.common.PATH + "project-model/saved-model-{epoch:02d}.hdf5"
        mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

        # Train a new model.
        ## reshape y from two dimensions to three dimensions
        y_tr_3d = self.y_tr.reshape(self.y_tr.shape[0], self.y_tr.shape[1], 1)[:, 1:]
        y_val_3d = self.y_val.reshape(self.y_val.shape[0], self.y_val.shape[1], 1)[:, 1:]

        history = model.fit([self.x_tr, self.y_tr[:, :-1]], y_tr_3d, batch_size=self.common.BATCH_SIZE, epochs=self.common.EPOCHS,
                            callbacks=[es, mc], validation_data=([self.x_val, self.y_val[:, :-1]], y_val_3d))

        # Plot training and validation loss overtime
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('loss_glove' + str(self.common.build_number) + '.png')

        # Get word index's from tokenizers
        reverse_source_word_index = self.x_tokenizer.index_word
        reverse_target_word_index = self.y_tokenizer.index_word
        print(reverse_source_word_index)
        print(reverse_target_word_index)
        # get max target word index
        target_word_index = self.y_tokenizer.word_index

        # inference model
        encoder_model, decoder_model = self.inference_model(encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense)
        # test model
        self.evaluate_model(reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model, ev_val=False, ev_tr=False)

    def learning_model(self, x_embedding_matrix, y_embedding_matrix):
        """
            Learning Model
        """
        """Encoder Model"""
        encoder_inputs = Input(shape=(self.common.MAX_TEXT_LEN,))
        # embedding layer with embedding matrix
        enc_emb_layer = Embedding(self.x_voc, self.common.EMBEDDING_DIM, weights=[x_embedding_matrix], 
                          input_length=self.x_voc, trainable=False)
        enc_emb = enc_emb_layer(encoder_inputs)
        # bidirectional encoder lstm
        encoder_lstm = Bidirectional(LSTM(self.common.LATENT_DIM, return_sequences=True, return_state=True))
        encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = encoder_lstm(enc_emb)

        state_h = Concatenate()([fw_state_h, bw_state_h])
        state_c = Concatenate()([fw_state_c, bw_state_c])
        encoder_states = [state_h, state_c]

        """#### Decoder Model"""
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        # embedding layer
        dec_emb_layer = Embedding(self.y_voc, self.common.EMBEDDING_DIM,
                          weights=[y_embedding_matrix], input_length=self.x_voc, 
                          trainable=False)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(self.common.LATENT_DIM*2, return_sequences=True, return_state=True)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=encoder_states)

        # dense layer
        decoder_dense = Dense(self.y_voc, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        """#### Combined LSTM Model"""

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        optimizer = RMSprop(lr=self.common.LEARNING_RATE, rho=0.9)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        return model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense

    def inference_model(self, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense):     
        """
            Inference Model
        """
        # Encoder Setup
        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        print(encoder_model.summary())

        # Decoder Setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.common.LATENT_DIM*2,))
        decoder_state_input_c = Input(shape=(self.common.LATENT_DIM*2,))
        decoder_hidden_state_input = Input(shape=(self.x_voc, self.common.LATENT_DIM*2))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_inputs)
        decoder_states = [state_h2, state_c2]
        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        # Final Decoder model
        decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + decoder_states)
        print(decoder_model.summary())

        return encoder_model, decoder_model

    def glove(self):
        """
            Load Glove Embeddings
        """
        embedding_index = {}
        glove_path = self.common.PATH + "glove/glove.6B."
        with open(glove_path + str(self.common.EMBEDDING_DIM) + 'd.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        print("Length of embeddings: " + str(len(embedding_index)))
        print(embedding_index.get("us"))

        return embedding_index

    def coverage(self, dictionary, total, embedding_index):
        covered = 0
        for word, _ in dictionary.items():
            if embedding_index.get(word) is not None:
                covered += 1
        return (covered/total * 100)

    def word_embeddings(self, embedding_index):
        """
            Word Embeddings - Tokenization
        """
        """X Tokenizer"""
        word_dict = {}
        text = self.df['text']

        for row in text:
            for word in row.split(" "):
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        print(len(word_dict))

        # #prepare a tokenizer for reviews on training data
        self.x_tokenizer = Tokenizer(num_words=len(word_dict), split=" ")
        self.x_tokenizer.fit_on_texts(list(self.X))

        # embedding matrix based on glove
        x_embedding_matrix = np.zeros(((self.x_tokenizer.num_words)+1, self.common.EMBEDDING_DIM),dtype='float32')
        print(x_embedding_matrix.shape)
        for word,i in self.x_tokenizer.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # Words not found in glove will be zeros
                x_embedding_matrix[i] = embedding_vector

        # convert text sequences into integer sequences
        x_tr_seq = self.x_tokenizer.texts_to_sequences(self.x_tr)
        x_val_seq = self.x_tokenizer.texts_to_sequences(self.x_val)

        # size of vocabulary ( +1 for padding token)
        self.x_voc = self.x_tokenizer.num_words + 1
        print(self.x_voc)

        # padding zero upto maximum length
        self.x_tr = pad_sequences(x_tr_seq,  maxlen=self.x_voc, padding='post')
        self.x_val = pad_sequences(x_val_seq, maxlen=self.x_voc, padding='post')

        """Y Tokenizer"""
        y_word_dict = {}
        summ = self.df['summary']

        for row in summ:
            for word in row.split(" "):
                if word not in y_word_dict:
                    y_word_dict[word] = 1
                else:
                    y_word_dict[word] += 1
        print(len(y_word_dict))

        # prepare a tokenizer for reviews on training data
        self.y_tokenizer = Tokenizer(num_words=len(y_word_dict), split=" ")
        self.y_tokenizer.fit_on_texts(list(self.Y))

        # y embedding matrix
        y_embedding_matrix = np.zeros((self.y_tokenizer.num_words +1, self.common.EMBEDDING_DIM),dtype='float32')
        print(y_embedding_matrix.shape)
        print(len(self.y_tokenizer.word_index))
        for word,i in self.y_tokenizer.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
            # Words not found in glove will be zeros
                y_embedding_matrix[i] = embedding_vector

        # convert text sequences into integer sequences
        y_tr_seq = self.y_tokenizer.texts_to_sequences(self.y_tr)
        y_val_seq = self.y_tokenizer.texts_to_sequences(self.y_val)

        # size of vocabulary + 1 for padding
        self.y_voc = self.y_tokenizer.num_words + 1
        print(self.y_voc)

        # padding zero upto maximum length
        self.y_tr = pad_sequences(y_tr_seq, maxlen=self.y_voc, padding='post')
        self.y_val = pad_sequences(y_val_seq, maxlen=self.y_voc, padding='post')
        

        return x_embedding_matrix, y_embedding_matrix, word_dict, y_word_dict

    def evaluate_model(self, reverse_source_word_index, reverse_target_word_index, target_word_index, encoder_model, decoder_model, ev_val, ev_tr):
        """
            Evaluate the model - 10 test examples from the training data fully printed
            @ev_val: True - Run evaluation to get ROUGE metrics for validation data
            @ev_tr: True - Run evaluation to get ROUGE metrics for training data
        """
        for i in range(0,10):
            article = self.common.seq_to_text(self.x_tr[i], reverse_source_word_index, target_word_index, False)
            original = self.common.seq_to_text(self.y_tr[i], reverse_target_word_index, target_word_index, True)

            if (original !=""):
                # # feed the first word of the original summary into the encoder (used for decode_seq3)
                # a = nltk.word_tokenize(original)[0]
                # for val,word in reverse_target_word_index.items():
                #     if word == a:
                #         i = val
                print("Article: " + article)
                print("Original summary:", original)
                # reshape data into correct format for encoder (1, max_text_len)
                x_tr_i_reshaped = self.x_tr[i].reshape(1,self.common.MAX_TEXT_LEN)
                summary = self.common.decode_sequence(x_tr_i_reshaped, encoder_model, decoder_model, reverse_target_word_index, target_word_index)
                print("Generated summary:",summary)
                print("\n")

                if summary != "":    
                    print("ROUGE score: ")
                    score = self.common.getRouge(str(summary), str(original))
                    print(score)
                    print(score[0].get('rouge-1').get('f'))
                    print(score[0].get('rouge-1').get('p'))
                    print(score[0].get('rouge-1').get('r'))
        if ev_val:
            # evaluatate on validation data
            x_val_len = len(self.x_val)
            print("Evaluate against validation data: " + str(x_val_len))
            f_ov_val, p_ov_val, r_ov_val = self.common.evaluation(self.x_val, self.y_val, reverse_target_word_index, target_word_index, encoder_model, decoder_model)
            print("Avg Val F Score: " + str(f_ov_val/x_val_len))
            print("Avg Val Precision: " + str(p_ov_val/x_val_len))
            print("Avg Val Recall: " + str(r_ov_val/x_val_len))
        if ev_tr:
            # evaluate on training data
            x_tr_len = len(self.x_tr)
            print("Evaluate against training data: " + str(x_tr_len))
            f_ov_tr, p_ov_tr, r_ov_tr = self.common.evaluation(self.x_tr, self.y_tr, reverse_target_word_index, target_word_index, encoder_model, decoder_model)
            print("Avg Tr F Score: " + str(f_ov_tr/x_tr_len))
            print("Avg Tr Precision: " + str(p_ov_tr/x_tr_len))
            print("Avg Tr Recall: " + str(r_ov_tr/x_tr_len))
