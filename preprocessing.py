import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

class Read_Data():
	"""
	Read in the data - read each text file that we are processing into a pd dataframe 
	"""
	def __init__(self, loc):
		self.datapath = loc
		self.file_names = [file for file in os.listdir(loc) if file.endswith('.story')]
		self.df = pd.DataFrame(columns=['file', 'text', 'summary'])
	

	def read_in_files(self):
		for file in self.file_names:
			with open(self.datapath + "/" + file) as f:
				data = f.read()
				# 0 - body, 1 - summary
				data_split = re.split("@highlight", data, maxsplit=1)
				# appending to df 
				self.df = self.df.append({'file': file, 'text': str(data_split[0]), 'summary': str(data_split[1])}, ignore_index=True)

	def get_df(self):
		return self.df
	

class Clean_Data():
	"""
	Use the data that is read in, as a pd df, and clean it - remove any whitespace and empty columns
	Shape the data as necessary
	Removal of stop words - Could this cause an issue when dealing with creating gramatically correct summaries?
	"""
	def __init__(self, dataframe):
		self.df = dataframe

	def remove_stop_words(self):
		# corpus = self.data.to_string()
		# remove stop words from the text	
		stop_words = set(stopwords.words('english'))
		# corpus_filtered = [word for word in corpus if not word in stop_words]
		# df = pd.DataFrame([x.split(';') for x in corpus_filtered.split('\n')])
		# return df

	def clean_data(self):
		self.df.drop_duplicates(subset=['text'],inplace=True)  #dropping duplicates
		self.df.dropna(axis=0,inplace=True)   #dropping na
		# clean up summaries - remove the 'highlight' word
		# clean up the whole thing by removing /n
		# remove (CNN) and writd punctuation like --
		

		