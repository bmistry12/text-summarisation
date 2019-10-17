import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

def freqTable(text_string):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

def score_sentences(corpus, freqTable):
    sentenceValue = dict()
    for sentence in corpus:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

    return sentenceValue

def find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    average = (sumValues / len(sentenceValue))

    return average

def generate_summary(corpus, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in corpus:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary


def summarise(corpus):
    text_string = str(corpus).strip('[]')
    freq_table = freqTable(text_string)
    sentence_scores = score_sentences(corpus, freq_table)
    threshold = find_average_score(sentence_scores)
    summary = generate_summary(corpus, sentence_scores, 1.5*threshold)
    return summary

if __name__ == '__main__':
    corpus = []
    filename = "summary.txt"

    for line in open(filename, 'r'):
        corpus.append(line)

    result = summarise(corpus)
    print(result)