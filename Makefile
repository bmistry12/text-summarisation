# Change by passing in DATA_OR_MODEL=<>/ TRAIN_DATA_PATH=<> / OUTPUT_CSV=<> via make command

## Variables
TRAIN_DATA_PATH="./dailymail/originals"
OUTPUT_CSV="./data/dailymail-nodp.csv"
TEST=1
MODEL_ID=1  # 0 = unidirectional, 1=bidirectional, 2=GloVe model
WORD_REMOVAL=False
STOP_WORDS=False
LEMMATIZE=False
LEMMATIZE_WITH_POS=False
TEXT_RANK=False

## Commands
setup: 
	python setup.py install --user
	python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

requirements:
	pip install -r requirements.txt --user

run-data:
	python main.py 0 ${TRAIN_DATA_PATH} ${OUTPUT_CSV} ${STOP_WORDS} ${LEMMATIZE} ${LEMMATIZE_WITH_POS} ${TEXT_RANK}

run-model:
	python main.py 1 ${MODEL_ID} ${WORD_REMOVAL}

