import os
import re
import nltk
import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity

class Read_Write_Data():
	"""
	Read in the data - read each text file that we are processing into a pd dataframe 
	Write out data - to specified CSV file
	"""
	def __init__(self, loc):
		self.datapath = loc
		self.file_names = [file for file in os.listdir(loc) if file.endswith('.story')]
		self.df = pd.DataFrame(columns=['file', 'text', 'summary'])
	
	def read_in_files(self, filesToRead):
		files_to_read = self.file_names[:filesToRead]
		print(len(files_to_read))
		for file in files_to_read:
			with open(self.datapath + "/" + file, encoding="utf8") as f:
				data = f.read()
				# 0 - body, 1 - summary
				data_split = re.split("@highlight", data, maxsplit=1)
				# appending to df 
				self.df = self.df.append({'file': file, 'text': str(data_split[0]), 'summary': str(data_split[1])}, ignore_index=True)
		print("read in files")
		print(self.df.head())

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

	def clean_data(self, textRank):
		self.df.drop_duplicates(subset=['file'],inplace=True)  #dropping duplicates
		self.df.dropna(axis=0,inplace=True)   #dropping na
		# clean texts
		self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\(CNN\)|--|[^\w\s\.]','',x)).apply(lambda x: re.sub(r'(\.(?=[\s\r\n]|$))','',x)).apply(lambda x: re.sub(r'\n',' ',x)).apply(lambda x: re.sub(r'\.','',x))
		# separate the summaries using a '.' 
		if textRank :	
			self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n|[^\w\s\.\@]','',x))
		else :
			self.df['summary'] = self.df['summary'].apply(lambda x: re.sub(r'\n|[^\w\s\.\@]','',x)).apply(lambda x: re.sub(r'@highlight',' ',x))
		print("cleaned data")
		print(self.df.head())

	def remove_stop_words(self):
		"""
		remove stop words from the text	and summaries
		"""
		stop_words = set(stopwords.words('english'))
		self.df['text'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if not word.lower() in stop_words]))
		self.df['summary'] = self.df['summary'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: " ".join([word for word in x if not word.lower() in stop_words]))	
		print(self.df['summary'][0])	
		print(self.df['text'][0])
		# print(self.df['text'])
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
		# Should we not also lemmatize summaries?
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
			# This currently doesn't output text in the right format (I think)
			self.df['text'] = text_tokenized.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
			self.df['text'] = self.df['text'].apply(lambda x: ' '.join(x))	
		

class TextRank():
	"""
	Run TextRank on the data to ensure model is run against the most important texts and summaries
	"""
	def __init__(self, df):
		self.df = df
	
	def main(self):
		# text = self.df['text']
		summaries = self.df['summary']
		# update summaries
		new_summaries = [self.rankSummmaries(summary) for summary in summaries]
		self.df['summary'] = new_summaries

	def rankTexts(self, text):
		pass

	def rankSummmaries(self, summary):
		"""
		Rank summaries and return the one with the highest score
		"""
		summary_split = summary.split("@ highlight")
		print(summary_split)

		embedding_index = self.getWordEmbeddings()
		sentence_vectors  = []
		# get word count vector for each sentence
		for sentence in summary_split:
			words = nltk.word_tokenize(sentence)
			mean_vector_score = sum([embedding_index.get(word, np.zeros((100,))) for word in words])/len(words)
			sentence_vectors.append(mean_vector_score)
		
		# similarity matrix
		sim_matrix = self.getSimilarityMatrix(sentence_vectors)
		#graph of matrix - retrieve a set of scores based on page rank algorithm
		pageRank_scores = self.getGraph(sim_matrix) 
		# rank sentences based off scores and extract top one as the chosen sentence for training
		sent_scores = [(pageRank_scores[i], sent) for i, sent in enumerate(summary_split)]
		sent_scores = sorted(sent_scores, reverse=True)
		chosen_summary = sent_scores[0][1]
		return(chosen_summary)

	def getSimilarityMatrix(self, sentence_vectors):
		sim_matrix = np.zeros([len(sentence_vectors), len(sentence_vectors)])
		# CSim(d1,d2) = cos(x) - use cosine similarity
		for i, d1 in enumerate(sentence_vectors):
			for j, d2 in enumerate(sentence_vectors):
				if i != j :
					print(cosine_similarity(d1.reshape(1,100), d2.reshape(1,100)))
					sim_matrix[i][j] = cosine_similarity(d1.reshape(1,100), d2.reshape(1,100))
		print(sim_matrix)
		return sim_matrix

	def getGraph(self, sim_matrix):
		nx_graph = nx.from_numpy_array(sim_matrix)
		print(nx_graph)
		scores = nx.pagerank(nx_graph)
		print(scores)
		return scores
	
	def getWordEmbeddings(self):
		"""
		Get GloVe Word Embeddings 
		"""
		embedding_index = {}
		with open('./glove/glove.6B.100d.txt', encoding="utf8") as f:
			for line in f:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embedding_index[word] = coefs
		return embedding_index