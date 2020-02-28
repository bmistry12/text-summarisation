import nltk
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
"""
All Extractive Data Processing Methods In One Python File
"""
class TextRank():
    """
    Run TextRank on the data to ensure model is run against the most important summaries
    """

    def __init__(self, df):
        self.df = df
        self.main()

    def main(self):
        # text = self.df['text']
        summaries = self.df['summary']
        # update summaries
        new_summaries = [self.rank_summaries(summary) for summary in summaries]
        self.df['summary'] = new_summaries

    def rank_summaries(self, summary):
        """
        Rank summaries and return the one with the highest score
        """
        summary_split = summary.split("@ highlight")
        print(summary_split)

        embedding_index = self.get_word_embeddings()
        sentence_vectors = []
        # get word count vector for each sentence
        for sentence in summary_split:
            words = nltk.word_tokenize(sentence)
            mean_vector_score = sum([embedding_index.get(
                word, np.zeros((100,))) for word in words])/len(words)
            sentence_vectors.append(mean_vector_score)

        # similarity matrix
        sim_matrix = self.get_similarity_matrix(sentence_vectors)
        # graph of matrix - retrieve a set of scores based on page rank algorithm
        pageRank_scores = self.get_graph(sim_matrix)
        # rank sentences based off scores and extract top one as the chosen sentence for training
        sent_scores = [(pageRank_scores[i], sent)
                       for i, sent in enumerate(summary_split)]
        sent_scores = sorted(sent_scores, reverse=True)
        chosen_summary = sent_scores[0][1]
        return(chosen_summary)

    def get_similarity_matrix(self, sentence_vectors):
        sim_matrix = np.zeros([len(sentence_vectors), len(sentence_vectors)])
        # CSim(d1,d2) = cos(x) - use cosine similarity
        for i, d1 in enumerate(sentence_vectors):
            for j, d2 in enumerate(sentence_vectors):
                if i != j:
                    print(cosine_similarity(
                        d1.reshape(1, 100), d2.reshape(1, 100)))
                    sim_matrix[i][j] = cosine_similarity(
                        d1.reshape(1, 100), d2.reshape(1, 100))
        print(sim_matrix)
        return sim_matrix

    def get_graph(self, sim_matrix):
        nx_graph = nx.from_numpy_array(sim_matrix)
        print(nx_graph)
        scores = nx.pagerank(nx_graph, max_iter=200, alpha=0.9)
        print(scores)
        return scores

    def get_word_embeddings(self):
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

class WordFrequency():
    """
        Run WordFrequency on the data to ensure model is run against the summaries that has the highest word frequency rank with the main article
    """
    def __init__(self, df):
        self.df = df
        self.main()

    def main(self):
        texts = self.df['text']
        summaries = self.df['summary']
        # get sentence scores for each summary
        self.sent_pos = 0 # this is a hack for getting the correct article for each summary
        sentence_scores = [self.score_sentences(summary, texts) for summary in summaries]
        print("Sentence Scores")
        # print(sentence_scores)
        # sentence scores = [("sentence1", value1) ... ("sentecex", valuex)]
        self.df['summary'] = [self.get_best_summary(sentences) for sentences in sentence_scores]

    def score_sentences(self, document, texts):
        """"
        Score each summary based on the number of words that occurs in them that also occur in the highest occuring words in the main document text
        """
        sent_scores = []
        # call word_frequency to get a word frequency table (or rather list of words) from the respective article
        scorable_words = self.word_frequency(texts[self.sent_pos])
        # split the summaries by @highlight token
        summary_split = document.split("@ highlight")
        sentenceValue = 0
        sent_len = 0
        # for each summary calculate the sentence value
        for summary in summary_split:
            words = nltk.word_tokenize(summary)
            sent_len = len(words)
            for word in words:
                if word in scorable_words:
                    sentenceValue =+ 1
            # normalise sentence value based on sentence length so that longer sentences do not get an automatic advantage over shorter ones
            # as null rows havent been dropped yet there may be scores of 0
            if (sentenceValue !=0 and sent_len !=0):
                sentenceValue = sentenceValue / sent_len
            sent_scores.append((summary, sentenceValue))
        return sent_scores

    def word_frequency(self, document):
        """
            Calculate a word frequency table for the words in a given documents
            After this, it removes any words that occur below a given threshold value, returning a list of "acceptable" words from the original corpus
        """
        freq_table = {}
        words = nltk.word_tokenize(document)
        for word in words:
            if word in freq_table:
                freq_table[word] = freq_table.get(word) + 1
            else:
                freq_table[word] = 1
        # cut down the frequency table so that only common words are scored for
        freq_table = sorted(freq_table.items(), key=lambda x: x[1], reverse=True)
        scorable_words = []
        for word, occ in freq_table:
            # set threshold as words appearing 0 times or more
            if int(occ) > 0:
                scorable_words.append(word)
            else:
                break
        self.sent_pos = self.sent_pos + 1 # increment hack variable
        return scorable_words

    def get_best_summary(self, sent_scores):
        """
            Get the best summary based on which has the greatest score
        """
        best_val = 0
        best_sent = ""
        for (sentence, val) in sent_scores:
            if val > best_val:
                best_sent = sentence
                best_val = val
        return best_sent

class SentencePosition():
    """
        Run SentencePosition on the data to ensure model is run against the most important sentences in the articles
        Sentences in the beginning define the theme of
        the document whereas sentences in the end conclude or
        summarize the document.
        The positional value of a sentence is calculated by
        assigning the highest score value to the first sentence and
        the last sentence of the document. Second highest score
        value is assigned to the second sentence from starting and
        second last sentence of the document etc. - https://www.irjet.net/archives/V4/i5/IRJET-V4I5493.pdf
    """

    def __init__(self, df):
        self.df = df
        self.main()
    
    def main(self):
        print("Sentence Scores")
        texts = self.df['text']
        new_texts = [self.sentence_ranker(text) for text in texts]
        self.df['text'] = new_texts
        print(self.df['text'].head())

    def sentence_ranker(self, article):  
        print("Sentence ranker")
        max_rank = 5 # we only care about the first and last five sentence.
        # split by <eos> token added in sent_pos_cleaner
        sentences = article.split("< eos >")
        sent_with_rank = {}
        len_sent = len(sentences)
        for i in range(0, len_sent):
            sentence = sentences[i]
            if max_rank - i > 0 :
                # give rank to first 5 sentences - considered to be intro
                sent_with_rank[sentence] = max_rank - i
            if len_sent - max_rank <= i :
                # give rank to last 5 sentences - considered to be summaries
                if sentence in sent_with_rank: 
                    # there may be situations where a corpus is < 11 lines long as so there will be an overlap 
                    sent_with_rank[sentence] = sent_with_rank.get(sentence) + (max_rank - (len_sent - i) + 1)
                else:
                    sent_with_rank[sentence] = max_rank - (len_sent - i) + 1
        # return the new article joined together
        return "".join(sent_with_rank.keys())

#TO DO
class TFIDF():
    """
        Run WordFrequency on the data to ensure model is run against the summaries that has the highest word frequency rank with the main article
    """
    def __init__(self, df):
        self.df = df
        self.main()

    def main(self):
        texts = self.df['text']
        summaries = self.df['summary']
       
    def compute_tf(self, word_dict, doc):
        tf_dict = {}
        doc_len = len(doc)
        for word, count in word_dict.items():
            tf_dict[word]  = count / float(doc_len)
        return tf_dict

    def compute_idf(self, doc_list):
        idf_dict = {}
        N = len(doc_lit)
        
#TO DO
class PCA():
    pass
#TO DO
class OntologyClassification():
    pass
#TO DO
class Coverage():
    pass
#TO DO
class NWords():
    pass
#TO DO
class StopWords():
    # corpus generated stop words
    pass