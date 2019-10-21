# Change by passing in TRAIN_DATA_PATH=<> via make command
TRAIN_DATA_PATH="./cnn_small"

requirements:
	pip install -r requirements.txt

setup: 
	python setup.py install
	python -m nltk.downloader brown stopwords

run:
	py main.py ${TRAIN_DATA_PATH}
