import re
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tag import TrigramTagger

class NGrams():
    def __init__(self, corpus, n):
        self.corpus = corpus
        self.corpus_filtered = []
        self.ngrams = []
        self.n = n

    def remove_stop_words(self):
        stop_words = set(stopwords.words('english'))
        self.corpus_filtered = [word for word in self.corpus if not word in stop_words]
        # return self.corpus_filtered


    def generate_ngrams(self):
        self.corpus_filtered = self.corpus_filtered.lower()
        # Replace all none alphanumeric characters with spaces
        self.corpus_filtered = re.sub(r'[^a-zA-Z0-9\']', " ", self.corpus_filtered)
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in self.corpus_filtered.split(" ") if token != ""]
        # ngrams
        self.ngrams = zip(*[tokens[i:] for i in range(self.n)])
        print(self.ngrams)
        return [" ".join(ngram) for ngram in self.ngrams]


    def listToText(self):
        self.corpus_filtered = " ".join(self.corpus_filtered)
   
corpus = brown.words()[:50]
test = NGrams(corpus, 5)
print(len(corpus))

test.remove_stop_words()
test.listToText()
n_grams = test.generate_ngrams()
print(n_grams)
