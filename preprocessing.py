import numpy as np
import pandas as pd
from nltk.corpus import stopwords

class Clean_Data():
	def __init__(self, loc):
		self.datapath = loc
		self.train_path = "./cnn/train"
		self.test_path = "./cnn/test"
		self.data = pd.read_csv(self.datapath)

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
		

		