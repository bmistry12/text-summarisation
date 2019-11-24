import os
import sys
import pandas
# import model as model
# import test_model as testModel
import dataProcessing as process

def main():
    print("hello!")
    output_csv = "test.csv"
    try:
        path = sys.argv[1]
        print(path)
        try :
            output_csv = sys.argv[2]
            print(output_csv)
        except Exception as e:
            print("Not output csv specified - using default")
        if (os.path.isdir(path)):
            readwrite = process.Read_Write_Data(path)
            print("read in files")
            readwrite.read_in_files(5000)
            print("done reading files")
            data = readwrite.get_df()
            print (data.head())
            cleaner = process.Clean_Data(data)
            print("clean data")
            cleaner.clean_data()
            print("done cleaning data")
            print("remove stop words")
            cleaner.remove_stop_words()
            print("done removing stop words")
            print("lemmatize")
            # cleaner.lemmatization(True)
            print("done lemmatizing")
            data = readwrite.get_df()
            print(readwrite.get_df())
            print("output to csv")
            readwrite.df_to_csv(output_csv)
            print("done all")
            # manage = process.Manage_Data(data)
            # vocab_size = manage.getMaxSize()
            # print(vocab_size)
            # model = model.Seq2SeqRNN(data, vocab_size, 60, 25, 100, 200, 0.35)
            # model.run_model()
            # model.save_model()
            # test = testModel.TestModel()
            # testModel.getInput()
            # cleaner.clean_data()
            # cleaner.remove_stop_words()
            # cleaner.lemmatization(True)
            # new_data = readwrite.get_df()
            # testModel.summarize(new_data)
            # evaluate = evaluate.Evaluate()
            # evaluate.rogue()
        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running applicaton: " + str(e))

if __name__ == "__main__":
    main()
