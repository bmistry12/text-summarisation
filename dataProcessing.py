import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 

class Read_Write_Data():
	"""
	Read in the data - read each text file that we are processing into a pd dataframe 
	Write out data - to specified CSV file
	"""
	def __init__(self, loc):
		self.datapath = loc
		self.file_names = [file for file in os.listdir(loc) if file.endswith('.story')]
		self.df = pd.DataFrame(columns=['file', 'text', 'summary'])
	

	def read_in_files(self):
		for file in self.file_names:
			with open(self.datapath + "/" + file, encoding="utf8") as f:
				data = f.read()
				# 0 - body, 1 - summary
				data_split = re.split("@highlight", data, maxsplit=1)
				# appending to df 
				self.df = self.df.append({'file': file, 'text': str(data_split[0]), 'summary': str(data_split[1])}, ignore_index=True)

	def get_df(self):
		return self.df

	def df_to_csv(self, csv_name):
		self.df.to_csv(csv_name, index=True)
	

class Clean_Data():
	"""
	Use the data that is read in, as a pd df, and clean it - remove any whitespace and empty columns
	Shape the data as necessary
	Removal of stop words - Could this cause an issue when dealing with creating gramatically correct summaries?
	Lemmatization
	"""
	def __init__(self, dataframe):
		self.df = dataframe

	def clean_data(self):
		self.df.drop_duplicates(subset=['file'],inplace=True)  #dropping duplicates
		self.df.dropna(axis=0,inplace=True)   #dropping na
		self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\(CNN\)|--|[^\w\s\.]','',x)).apply(lambda x: re.sub(r'(\.(?=[\s\r\n]|$))','',x)).apply(lambda x: re.sub(r'\n',' ',x)).apply(lambda x: re.sub(r'\.','',x))
		# separate the summaries using a '.' 
		self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n|[^\w\s\.\@]','',x)).apply(lambda x: re.sub(r'@highlight',' ',x))
		print("cleaned data")
		print(self.df.head())

	def remove_stop_words(self):
		"""
		remove stop words from the text	and summaries
		"""
		stop_words = set(stopwords.words('english'))
		self.df['text'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if not word in stop_words]))
		self.df['summary'] = self.df['summary'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if not word in stop_words]))		
		print(self.df['text'])
		print(self.df.head())
		print("removed stop words")

	def getpos(self, word):
		pos = nltk.pos_tag([word])[0][1][0]
		wordnet_conv = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
		if pos in wordnet_conv.keys():
			return wordnet_conv.get(pos)
		return ""
		# wordnet pos can't deal with "I" - Preposition, "M" - modal, "C" - conjunction, "P" - pronoun

	def lemmatization(self, pos):
		"""
		lemmatization of text - uses wordnet lemmatizer
		@pos - value can be True or False. Used to indicate whether or not to use POS whilst lemmatizing
		"""
		lemmatizer = WordNetLemmatizer()
		text_tokenized = self.df['text'].apply(lambda x: nltk.word_tokenize(x))
		if pos:
			print("lemmatize with pos")
			for i in range(0,len(text_tokenized)):
				text_lemmatized = []
				for word in text_tokenized[i]:
					self.getpos(word)
					pos = self.getpos(word)
					if pos != "":
						lemma = lemmatizer.lemmatize(word, pos)
						text_lemmatized.append(lemma)
					else :
						text_lemmatized.append(word)
				text_lemmatized = ' '.join(map(str, text_lemmatized))
				self.df['text'][i] = text_lemmatized
		else :	
			print("lemmatize w/o POS")
			self.df['text'] = text_tokenized.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

class Manage_Data():
	def __init__(self, df):
		self.df = df

	def getVocabSize(self):
		word_dict = {}
		text = self.df['text']
		summary = self.df['summary']

		for row in text: 
			for word in row.split(" "):
				if word not in word_dict:
					word_dict[word] = 1
				else:
					word_dict[word] += 1
		
		for row in summary: 
			for word in row.split(" "):
				if word not in word_dict:
					word_dict[word] = 1
				else:
					word_dict[word] += 1

		return len(word_dict)
		
		
