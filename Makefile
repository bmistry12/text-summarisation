# Change by passing in DATA_OR_MODEL=<>/ TRAIN_DATA_PATH=<> / OUTPUT_CSV=<> via make command

## Variables
### Data Processing
# TRAIN_DATA_PATH="./dailymail/originals"
# OUTPUT_CSV="./data/dailymail-wf.csv"
TRAIN_DATA_PATH="/tmp/bhm699/dailymail/originals"
OUTPUT_CSV="/tmp/bhm699/dailymail-wf.csv"

STOP_WORDS=True
LEMMATIZE=True
LEMMATIZE_WITH_POS=True

#### Only one of these can be true at any given time
TEXT_RANK=False
WORD_FREQ=False
SENT_POS=True
### Model Running
MODEL_ID=1  # 0 = unidirectional, 1=bidirectional, 2=GloVe model
WORD_REMOVAL=False # remove words using uncommon_word_thr

## Commands
setup: 
	python setup.py install --user
	python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

requirements:
	pip install -r requirements.txt --user

run-data:
	python main.py 0 ${TRAIN_DATA_PATH} ${OUTPUT_CSV} ${STOP_WORDS} ${LEMMATIZE} ${LEMMATIZE_WITH_POS} ${TEXT_RANK} ${WORD_FREQ} ${SENT_POS}

run-model:
	python main.py 1 ${MODEL_ID} ${WORD_REMOVAL}

