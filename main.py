import os
import sys
import pandas
# import model as model
import dataProcessing as process

def main():
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
            readwrite.read_in_files()
            data = readwrite.get_df()
            cleaner = process.Clean_Data(data)
            cleaner.clean_data()
            cleaner.remove_stop_words()
            cleaner.lemmatization(True)
            data = readwrite.get_df()
            print(readwrite.get_df())
            readwrite.df_to_csv(output_csv)
            # manage = process.Manage_Data(data)
            # vocab_size = manage.getMaxSize()
            # print(vocab_size)
            # model = model.Seq2SeqRNN(data, vocab_size, 60, 25, 100, 200, 0.35)
            # model.run_model()

        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running applicaton: " + str(e))

if __name__ == "__main__":
    main()
