#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# !python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger
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
from sklearn.model_selection import train_test_split


# In[2]:


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# ### Hyperparameters

# In[3]:


BATCH_SIZE=20
EPOCHS=100
latent_dim=128
embedding_dim=128
test_train_split=0.30
build_number="1"
# LEARNING_RATE=0.0001


# ## Data Processing

# Read In Data

# In[3]:


# Only needed if running on Google Colab
# from google.colab import drive
# drive.mount('/content/drive')


# In[4]:


# df = pd.read_csv('./drive/My Drive/originals-l.csv')
df = pd.read_csv('./originals-l.csv')
df.head(1)


# In[5]:


df.count


# Split for now so we are only aiming for one summary per text (This is now in dataprocessing.py

# In[6]:


# print(df['summary'][0])
# df['summary'] = df['summary'].apply(lambda x: re.sub(r'\..*$',' ',str(x)))
# print(df['summary'][0])


# Remove .'s that appear in stuff like U.S.A and U.N - Eventually need to move this to dataprocessing.py

# In[6]:


print(df['summary'][0])
df['summary'] = df['summary'].apply(lambda x: re.sub(r'\.','',str(x)))
print(df['summary'][0])


# Check for rows with null values in them, and copy these into a new dataframe (df1). Drop any rows in df1 from df to ensure no NaN valued rows are present/
# 
# *Note. using simply dropna(how='any') does not seem to drop any of the rows*

# In[7]:


print(df.isnull().values.any())
print(df.shape)

df1 = df[df.isna().any(axis=1)]
print(df1.shape)

df.drop(df1.index, axis=0,inplace=True)
print(df.shape)
print(df.isnull().values.any())


# Word Count Distribution

# In[8]:


text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
for i in df['text']:
      text_word_count.append(len(i.split(' ')))

for i in df['summary']:
      summary_word_count.append(len(i.split(' ')))

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

length_df.hist(bins = 30)
plt.ylabel('Documents')
plt.xlabel('Word Count')
plt.savefig('word_count_distro' + str(build_number) + '.png')
#plt.show()


# Max Text Lengths

# In[9]:


max_text_len = max([len(txt) for txt in df['text']])
max_summary_len = max([len(txt) for txt in df['summary']])
print(max_text_len)
print(max_summary_len)


# ### Training-Validation Split

# X - Articles text </br>
# Y - Summaries

# In[10]:


# convert to numpy array
X = np.array(df['text'])
Y = np.array(df['summary'])


# In[11]:


x_tr,x_val,y_tr,y_val=train_test_split(X,Y,test_size=test_train_split,random_state=0,shuffle=True)
print(x_tr.shape)
print(x_val.shape)
print(y_tr.shape)
print(y_val.shape)


# ### Word Embeddings - Tokenization

# X Tokenizer

# In[12]:


word_dict = {}
text = df['text']

for row in text: 
  for word in row.split(" "):
    if word not in word_dict:
      word_dict[word] = 1
    else:
      word_dict[word] += 1


# In[13]:


# #prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=len(word_dict)) 
x_tokenizer.fit_on_texts(list(X))

#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1
print(x_voc)


# In[15]:


with open('xtokenizer.pickle', 'wb') as handle:
  pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Y Tokenizer

# In[14]:


word_dict = {}
text = df['summary']

for row in text: 
  for word in row.split(" "):
    if word not in word_dict:
      word_dict[word] = 1
    else:
      word_dict[word] += 1


# In[15]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=len(word_dict)) 
y_tokenizer.fit_on_texts(list(Y))

#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1
print(y_voc)


# In[18]:


with open('ytokenizer.pickle', 'wb') as handle:
  pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Learning Model

# #### Encoder Model

# In[16]:


encoder_inputs = Input(shape=(max_text_len,))
#embedding layer
enc_emb =  Embedding(x_voc,embedding_dim,trainable=True)(encoder_inputs)
#encoder lstm 
encoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)


# #### Decoder Model

# In[17]:


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
                                                          
#dense layer
decoder_dense = Dense(y_voc, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# #### Combined LSTM Model

# In[18]:


# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()


# In[19]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


# Early Stopping Callback to ensure we stop when Validation Loss is lowest - minimises risk of overfitting

# In[20]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2, restore_best_weights=False)
# filepath = "./drive/My Drive/project-model/saved-model-{epoch:02d}.hdf5"
filepath = "saved-model.hdf5"
mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


# #### Use this method to train a new model. To continue training a previously trained model see below

# In[23]:


history = model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:], batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[mc], validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))


# #### This method is to only be used when loading a previously partially trained model

# In[21]:


# epoch_num = '05'
# model = load_model("./drive/My Drive/project-model/saved-model-" + epoch_num + ".hdf5")
#model = load_model('saved-model.hdf5')
#history = model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:], callbacks=[mc], batch_size=BATCH_SIZE, epochs=1, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))


# In[ ]:


model.save('model-finished' + str(build_number) + '.h5')


# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss' + str(build_number) + '.png')
#plt.show()


# ## Inference Model

# In[ ]:


reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word


# In[ ]:


# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_inputs)

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2) 

# Final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])


# In[ ]:


encoder_model.summary()


# In[ ]:


decoder_model.summary()


# ### Methhods for Reversing Word Embeddings

# In[ ]:


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


# ### Summarisation Method 

# In[ ]:


def decode_sequence(input_seq): 
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, max_summary_len))
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
      output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c]) 
      # print(output_tokens[0, -1, :][1:])
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      # print(sampled_token_index)
      # print(sampled_token_index)
      if (sampled_token_index != 0 ):
        sampled_token = reverse_target_word_index[sampled_token_index]
        decoded_sentence += ' '+sampled_token
      else :
        stop_condition = True
      if (len(decoded_sentence.split()) >= (max_summary_len-1)):
              stop_condition = True
       # Update the target sequence (of length 1).
      # target_seq = np.zeros((1,1))
      target_seq = np.zeros((1, max_summary_len))
      target_seq[0, sampled_token_index] = 1

      # Update internal states
      e_h, e_c = h, c
      
    return decoded_sentence


# ## Test Model Output

# Note: *I think there isn't enough data being passed in and so the argmax value always is 0 - it can't learn what should be next*

# In[ ]:


for i in range(0,1):
    print("Article:",seq2text(x_tr[i]))
    print("Original summary:",seq2summary(y_tr[i]))
    x_i = x_tr[i].reshape(1,max_text_len)
    print("Generated summary:",decode_sequence(x_i))
    print("\n")


# ## Evaluation

# Using ROUGE (Recall-Orientated Understanding Gisting Evaluation) to evaluate the generated summaries

# In[ ]:


def get_overlapping_words(x, y):
  num=0
  x = nltk.word_tokenize(x)
  y = nltk.word_tokenize(y)
  for word in y:
    if word in x:
      num = num+1
      x.remove(word)
    else:
      return num

def precision(target, generated):
  length = len(target)
  for i in range (0, length):
    num_overlapping_words = get_overlapping_words(target[i], generated[i])
    generated_summary_len = len(generated[i])
    if generated_summary_len == 0 :
        return 0.0
    else : 
      return num_overlapping_words / generated_summary_len


# ### For Training Data

# In[ ]:


print(len(x_tr))


# In[ ]:


val_target_summary = []
val_generated_summary = []
# x_val_len = len(x_val)
x_val_len = 1
for i in range(0,x_val_len):
  print(i)
  val_target_summary.append((y_val[i]))
  x_i = x_val[i].reshape(1,max_text_len)
  val_generated_summary.append(decode_sequence(x_i))


# In[ ]:


print("precision : " + str(precision(tr_target_summary, tr_generated_summary)))


# ### For Validation Data

# In[ ]:


val_target_summary = []
val_generated_summary = []
# x_val_len = len(x_val)
x_val_len = 1
for i in range(0,x_val_len):
  print(i)
  val_target_summary.append((y_val[i]))
  x_i = x_val[i].reshape(1,max_text_len)
  val_generated_summary.append(decode_sequence(x_i))


# In[ ]:


# pre = precision(val_target_summary, val_generated_summary)
# print(pre)
# print("precision : " + str(pre))


# # Inputting New Data

# In[ ]:


def getpos(word):
  pos = nltk.pos_tag([word])[0][1][0]
  wordnet_conv = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
  if pos in wordnet_conv.keys():
    return wordnet_conv.get(pos)
  return ""


# In[ ]:


def lemmatization(text):
  lemmatizer = WordNetLemmatizer()
  text_tokenized = inp_df['text'].apply(lambda x: nltk.word_tokenize(x))
  print("lemmatize with pos")
  for i in range(0,len(text_tokenized)):
    text_lemmatized = []
    for word in text_tokenized[i]:
      pos = getpos(word)
      if pos != "":
        lemma = lemmatizer.lemmatize(word, pos)
        text_lemmatized.append(lemma)
      else :
        text_lemmatized.append(word)
    text_lemmatized = ' '.join(map(str, text_lemmatized))
    inp_df['text'][i] = text_lemmatized


# In[ ]:


input1 = "(CNN) — Earlier this year, Delta Air Lines announced a rethink on reclining seats. In an effort to disrupt fewer passengers' travel experiences, Delta said it'd begin revamping some of its jets to reduce the recline of coach seats from four inches to two inches and the recline of first class seats from 5.5 inches to 3.5 inches. For those who abhor the recline option, it's a small step. And for those who value it, well, it's a compromise. This seemingly innocuous topic is one where there are very much two minds on what's acceptable and what's not. Two CNN Travel staffers engage in a friendly debate about seat recline. Your seat. Your decision. Stacey Lastoe, senior editor at CNN Travel, is of above-average height and makes no apology about reclining; it's her right as a plane, train and bus passenger. She encourages the person sitting in front of her to recline as well. On the first leg of my flight to Japan for my honeymoon, my husband and I got upgraded to first class. Although it would just be a few hours in the sky en route to Dallas, I was excited about sipping Champagne, sitting back and relaxing. Flute in hand, I pushed back to recline my seat for maximum relaxation. But it would not budge; I appeared to be stuck in a dysfunctional seat. Or was I? Turns out the gentleman behind me had a dog in a crate down between his legs, positioned so the seat in front of his -- my seat -- had nowhere to go. Because we were newlyweds and loving every moment of it, I did not mind when my husband turned to the man and told him his wife wanted to recline her seat and asked if he could please rearrange his dog crate to allow for everyones comfort."
inp_df = pd.DataFrame(columns=['text', 'summary'])
inp_df = inp_df.append({'text': str(input1), 'summary': ""}, ignore_index=True)
inp_df.head()


# In[ ]:


inp_df['text'] = inp_df['text'].apply(lambda x: re.sub(r'\(CNN\)|--|[^\w\s\.]','',x)).apply(lambda x: re.sub(r'(\.(?=[\s\r\n]|$))','',x)).apply(lambda x: re.sub(r'\n',' ',x)).apply(lambda x: re.sub(r'\.','',x))


# In[ ]:


# remove stop words
stop_words = set(stopwords.words('english'))
inp_df['text'] = inp_df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if not word in stop_words]))


# In[ ]:


#lemmatize
lemmatization(inp_df['text'])
print(inp_df['text'])


# In[ ]:


seq = np.array(inp_df['text'])
print(seq)


# In[ ]:


seq_tokenizer = x_tokenizer.texts_to_sequences(seq)
#padding zero upto maximum length
seq_tokenizer_padded = pad_sequences(seq_tokenizer,  maxlen=max_text_len, padding='post')

summary = decode_sequence(seq_tokenizer_padded)

# print(seq2text(seq))
print("---")
print("Summary: " + summary)

