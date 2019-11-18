import re
import pickle
import numpy as np
import pandas as pd
import dataProcessing as process
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

if __name__ == "__main__":
    build_num = 2
    test_data_path = "./cnn_small"
    model_data_path = "./model/"

    print(test_data_path)
    readwrite = process.Read_Write_Data(test_data_path)
    readwrite.read_in_files()
    data = readwrite.get_df()
    print(data.head())
    cleaner = process.Clean_Data(data)
    print("Cleaning data")
    cleaner.clean_data()
    cleaner.remove_stop_words()
    cleaner.lemmatization(True)
    data = readwrite.get_df()

    # Load encoder model
    encoder_model = load_model(str(model_data_path) + 'encoder_model' + str(build_num) + '.h5')
    print("loading encoder model")
    decoder_model = load_model(str(model_data_path) + 'decoder_model' + str(build_num) + '.h5')
    print("loading decoder model")

    rows = len(data)
    print(rows)
    try:
        with open(str(model_data_path) + 'xtokenizer.pickle', 'rb') as handle:
            x_tokenizer = pickle.load(handle)
            print("loading x_tokenizer")
    except Exception as e:
        print(str(e))

    try:
        with open(str(model_data_path) + 'ytokenizer.pickle', 'rb') as handle:
            y_tokenizer = pickle.load(handle)
            print("loading y_tokenizer")
    except Exception as e:
        print(str(e))

    try:
        with open('data.txt') as f:
            data = f.read()
            print(data)
            data = data.split(';')
            print(data)
            max_text_len = data[0]
            max_summary_len = data[1]
            print(max_text_len)
            print(max_summary_len)
    except Exception as e:
        print(str(e))

    reverse_target_word_index = y_tokenizer.index_words
    reverse_source_word_index = x_tokenizer.index_words
    X = np.array(data['text'])

    x = x_tokenizer.texts_to_sequences(X)
    x = pad_sequences(x,  maxlen=max_text_len, padding='post')

    # for i in range (0, rows):
    #   original_summary = data.loc[[i]]['summary']
    #   text = data.loc[[i]]['text']
    #   file_name = data.loc[[i]]['file']
    for i in range(0, rows):
        print("Article:", seq2text(x[i]))
        print("Original summary:", data.loc[[i]]['summary'])
        print("Generated summary:", decode_sequence(x))
        print("\n")


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, max_summary_len))
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # print(sampled_token_index)
        if (sampled_token_index != 0):
            sampled_token = reverse_target_word_index[sampled_token_index]
            decoded_sentence += ' '+sampled_token
        else:
            stop_condition = True
        if (len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True
        # Update the target sequence (of length 1).
        # target_seq = np.zeros((1,1))
        target_seq = np.zeros((1, 1, max_summary_len))
        target_seq[0, 0, sampled_token_index] = 1

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString+reverse_source_word_index[i]+' '
    return newString
