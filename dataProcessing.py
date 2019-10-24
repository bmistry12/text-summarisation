import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
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
			with open(self.datapath + "/" + file) as f:
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
	Use the data that is read in, as a pd df, and clfean it - remove any whitespace and empty columns
	Shape the data as necessary
	Removal of stop words - Could this cause an issue when dealing with creating gramatically correct summaries?
	Lemmatization
	"""
	def __init__(self, dataframe):
		self.df = dataframe

	def clean_data(self):
		self.df.drop_duplicates(subset=['file'],inplace=True)  #dropping duplicates
		self.df.dropna(axis=0,inplace=True)   #dropping na
		self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\(CNN\)|--|\n','',x))
		self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n','',x)).apply(lambda x: re.sub(r'@highlight','. ',x))
		print(self.df.head())

	def remove_stop_words(self):
		"""
		remove stop words from the text	
		"""
		stop_words = set(stopwords.words('english'))
		# this doesn't work yet
		self.df['text'] = self.df['text'].apply(lambda x: [word for word in x if not word in stop_words])
		print(self.df['text'])

	def lemmatization(self):
		"""
		lemmatization of text 
		"""
		lemmatizer = WordNetLemmatizer()
