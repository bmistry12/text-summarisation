# Change by passing in TRAIN_DATA_PATH=<> via make command
# TRAIN_DATA_PATH="./cnn_small"
TRAIN_DATA_PATH="./cnn/originals"
OUTPUT_CSV="./data/originals-nol.csv"

requirements:
	pip install -r requirements.txt --user

setup: 
	python setup.py install --user
	python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger

run:
	python main.py ${TRAIN_DATA_PATH} ${OUTPUT_CSV}
