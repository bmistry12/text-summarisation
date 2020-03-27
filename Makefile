# Change by passing in DATA_OR_MODEL=<>/ TRAIN_DATA_PATH=<> / OUTPUT_CSV=<> via make command

## Variables
### Data Processing
TRAIN_DATA_PATH="./cnn/originals"
OUTPUT_CSV="./data/cnn-all.csv"
TRAIN_DATA_PATH_LABS="/tmp/bhm699/cnn/originals"
OUTPUT_CSV_LABS="/tmp/bhm699/cnn-all.csv"

STOP_WORDS=True
LEMMATIZE=True
LEMMATIZE_WITH_POS=True
SENT_POS=False
#### Only one of these can be true at any given time
TEXT_RANK=False
WORD_FREQ=False

### Model Running
MODEL_ID=1  # 0 = unidirectional, 1=bidirectional, 2=GloVe model
WORD_REMOVAL=False # remove words using uncommon_word_thr
CSV_NAME="cnn-all.csv" # csv data to run model against

## Commands
setup: 
	python setup.py install --user
	python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

setup-labs: 
	python3 setup.py install --user
	python3 -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

requirements:
	pip install -r requirements.txt --user

run-data:
	python main.py 0 ${TRAIN_DATA_PATH} ${OUTPUT_CSV} ${STOP_WORDS} ${LEMMATIZE} ${LEMMATIZE_WITH_POS} ${TEXT_RANK} ${WORD_FREQ} ${SENT_POS}

run-data-labs:
	python3 main.py 0 ${TRAIN_DATA_PATH_LABS} ${OUTPUT_CSV_LABS} ${STOP_WORDS} ${LEMMATIZE} ${LEMMATIZE_WITH_POS} ${TEXT_RANK} ${WORD_FREQ} ${SENT_POS}

run-model:
	python main.py 1 ${MODEL_ID} ${WORD_REMOVAL} ${CSV_NAME}

run-model-labs:
	python3 main.py 1 ${MODEL_ID} ${WORD_REMOVAL} ${CSV_NAME}