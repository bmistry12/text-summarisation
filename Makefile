# Change by passing in TRAIN_DATA_PATH=<> via make command
# TRAIN_DATA_PATH="./cnn_small"
TRAIN_DATA_PATH="./cnn/0"
OUTPUT_CSV="data0.csv"

requirements:
	pip install -r requirements.txt --user

setup: 
	python setup.py install --user
	python -m nltk.downloader stopwords wordnet punkt

run:
	py main.py ${TRAIN_DATA_PATH} ${OUTPUT_CSV}
