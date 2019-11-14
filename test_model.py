import re
import dataProcessing as process
from keras.models import load_model

build_num = 2
test_data_path = "./cnn_small"

readwrite = process.Read_Write_Data(test_data_path)
readwrite.read_in_files()
data = readwrite.get_df()
cleaner = process.Clean_Data(data)
cleaner.clean_data()
cleaner.remove_stop_words()
cleaner.lemmatization(True)
data = readwrite.get_df()

# Load encoder model
encoder_model = load_model('encoder_model' + str(build_num) + '.hs5')
decoder_model = load_model('decoder_model' + str(build_num) + '.hs5')

rows = len(data)
for i in range(0, rows):
    original = seq2text()


'''
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
'''
            