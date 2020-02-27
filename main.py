import os
import sys
# import pandas
# import models as models
import extractive as exdp
import dataProcessing as process

def data_processing():
    """
        Main script for all data processing - arguments are taken from command line via main() method
        Note: you should only have one of arg7 onwards as True.
        @arg2 = training data path
        @arg3 = output csv for procesed data
        @arg4 = remove stop words?
        @arg5 = lemmatize?
        @arg6 = lemmatize with part of speech?
        @arg7 = run text rank?
        @arg8 = run word freq?
        @arg9 = run sent pos?
    """
    output_csv = "default.csv" # default value
    try:
        path = sys.argv[2] 
        print(path)
        try :
            output_csv = sys.argv[3]
            print(output_csv)
        except Exception as e:
            print("Not output csv specified - using default")
        if (os.path.isdir(path)):
            try: 
                # system arguments
                stopWords = sys.argv[4]
                lemmatize = sys.argv[5]
                lemmatize_with_pos = sys.argv[6]
                textRank = sys.argv[7]
                wordFreq = sys.argv[8]
                sentPost = sys.argv[9]
                # run data processing
                readwrite = process.Read_Write_Data(path)
                print("read in files")
                # 219505 = dm 92600= cnn
                readwrite.read_in_files(92580)
                print("done reading files")
                data = readwrite.get_df()
                cleaner = process.Clean_Data(data)
                print("clean data")
                cleaner.clean_data(textRank, wordFreq, sentPost)
                print("done cleaning data")
                if stopWords == "True":
                    print("remove stop words")
                    cleaner.remove_stop_words()
                    print("done removing stop words")
                if lemmatize == "True":
                    print("lemmatize")
                    cleaner.lemmatization(lemmatize_with_pos)
                    print("done lemmatizing")
                #drop null rows
                cleaner.drop_null_rows()
                if textRank == "True":
                    print("run text rank")
                    exdp.TextRank(readwrite.get_df())
                    print("text rank applied")
                if wordFreq == "True":
                    print("run word frequency")
                    exdp.WordFrequency(readwrite.get_df())
                    print("word frequency applied")
                if sentPost == "True":
                    print("run sentence position")
                    exdp.SentencePosition(readwrite.get_df())
                    print("sentence position applied")
                data = readwrite.get_df()
                print(readwrite.get_df())
                print("output to csv")
                readwrite.df_to_csv(output_csv)
                print("csv saved")
            except Exception as e:
                print("A required boolean was not set? Or " + str(e))
        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running data processing : " + str(e))

def model():
    try:
        model_id = sys.argv[2] 
        word_removal = sys.argv[3] # uncommon word removal boolean
        print(model_id)
        if model_id == "0":
            # unidirectional model
            unimodel = models.UniModel(word_removal)
        elif model_id == "1":
            # bidirectional model
            bimodel = models.BiModel(word_removal)
        elif model_id == "2":
            # glove model
            glovemodel = models.GloveModel(word_removal)
        else:
            raise Exception("This is not a valid model id - use 0 for unidirectional, 1 for bidirectional and 2 for glove")
    except Exception as e:
        print("Error running model: " + str(e))

if __name__ == "__main__":
    """
        @arg1 = data_or_model, 0=dataprocessing, 1=model
        @arg2 = training data path OR model to run, 0=uni,1=bi,2=glove
        @arg3 = output csv OR uncommon word removal boolean
        @arg4 = remove stop words
        @arg5 = lemmatize
        @arg6 = lemmatize with part of speech
        @arg7 = run text rank
        @arg8 = run word frequency
        @arg9 = run sent pos
    """
    try:
        data_or_model = sys.argv[1]
        if data_or_model == "0":
            print("data processing")
            data_processing()
        elif data_or_model == "1":
            model()
        else:
            raise Exception("This is not a valid mode - use 0 for data procesing and 1 for running a model")
    except Exception as e:
        print("Error running applicaton: " + str(e))
