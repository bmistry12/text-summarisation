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
	def __init__(self, df):
		self.df = df
	
	def rankSummmaries(self):
		# sentences
		summary = self.df['summary'][0]
		summary_split = summary.split("@ highlight")
		print(summary_split)

		embedding_index = self.getWordEmbeddings()
		sentence_vectors  = []
		# get word count vector for each sentence
		for i, sentence in enumerate(summary_split):
			words = nltk.word_tokenize(sentence)
			mean_vector_score = sum([embedding_index.get(word, np.zeros((100,))) for word in words])/len(words)
			sentence_vectors.append(mean_vector_score)
		
		# similarity matrix
		sim_matrix = self.getSimilarityMatrix(sentence_vectors)
		#graph of matrix - retrieve a set of scores based on page rank algorithm
		pageRank_scores = self.getGraph(sim_matrix) 
		# rank sentences based off scores and extract top one as the chosen sentence for training
		sent_scores = [(pageRank_scores[i], sent) for i, sent in enumerate(summary_split)]
		print(sent_scores)
		# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


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

		# article = "Editors note In Behind Scenes series CNN correspondent share experience cover news analyze story behind event Here Soledad OBrien take user inside jail many inmate mentally ill An inmate house forgotten floor many mentally ill inmate house Miami trial MIAMI Florida The ninth floor MiamiDade pretrial detention facility dubbed forgotten floor Here inmate severe mental illness incarcerate theyre ready appear court Most often face drug charge charge assault officer charge Judge Steven Leifman say usually avoidable felony He say arrest often result confrontation police Mentally ill people often wont theyre told police arrive scene confrontation seem exacerbate illness become paranoid delusional less likely follow direction accord Leifman So end ninth floor severely mentally disturbed get real help theyre jail We tour jail Leifman He well know Miami advocate justice mentally ill Even though exactly welcome open arm guard give permission shoot videotape tour floor Go inside forgotten floor At first hard determine people The prisoner wear sleeveless robe Imagine cut hole arm foot heavy wool sleep bag thats kind look like Theyre design keep mentally ill patient injure Thats also shoe lace mattress Leifman say onethird people MiamiDade county jail mentally ill So say sheer volume overwhelm system result see ninth floor Of course jail suppose warm comfort light glare cell tiny loud We see two sometimes three men sometimes robe sometimes naked lie sit cell I son president You need get one man shout He absolutely serious convince help way could reach White House Leifman tell prisonerpatients often circulate system occasionally stabilize mental hospital return jail face charge Its brutally unjust mind become strong advocate change thing Miami Over meal later talk thing get way mental patient Leifman say 200 year ago people consider lunatic lock jail even charge They consider unfit society Over year say public outcry mentally ill move jail hospital But Leifman say many mental hospital horrible shut Where patient go Nowhere The street They become many case homeless say They never get treatment Leifman say 1955 half million people state mental hospital today number reduce 90 percent 40000 50000 people mental hospital The judge say he work change Starting 2008 many inmate would otherwise brought forgotten floor instead sent new mental health facility first step journey toward longterm treatment punishment Leifman say complete answer start Leifman say best part winwin solution The patient win family relieve state save money simply cycling prisoner And Leifman justice serve Email friend"
	
