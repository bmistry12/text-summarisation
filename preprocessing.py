import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

class Read_Data():
	# Read in the data - read each text file that we are processing into a pd dataframe 
	def __init__(self, loc):
		self.text = []
		self.file_names = [file for file in os.listdir(loc) if file.endswith('.txt')]
		self.df = pd.DataFrame(columns=['fileID', 'text', 'summary'])
		# self.datapath = loc
		# self.train_path = "./cnn/train"
		# self.test_path = "./cnn/test"
	def read_in_files(self):
		# appending to df 
		# df = df.append({'fileID': 1, 'text': 'aaaaa', 'summary': 'a'}, ignore_index=True)
		for file in self.file_names:
			pass

class Clean_Data():
	# Use the data that is read in, as a pd df, and clean it - remove any whitespace and empty columns
	# Shape the data as necessary
	# Removal of stop words - Could this cause an issue when dealing with creating gramatically correct summaries?
	def __init__(self, data):
		# self.datapath = loc
		# self.train_path = "./cnn/train"
		# self.test_path = "./cnn/test"
		self.data = data

	def remove_stop_words(self):
		corpus = self.data.to_string()
		# remove stop words from the text	
		stop_words = set(stopwords.words('english'))
		corpus_filtered = [word for word in corpus if not word in stop_words]
		df = pd.DataFrame([x.split(';') for x in corpus_filtered.split('\n')])
		return df

	def clean_data(self):
		self.data.drop_duplicates(subset=['Text'],inplace=True)  #dropping duplicates
		self.data.dropna(axis=0,inplace=True)   #dropping na
		

		